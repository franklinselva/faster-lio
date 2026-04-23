#include "faster_lio/loop_closer.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include <spdlog/spdlog.h>

namespace faster_lio {

LoopCloser::LoopCloser(const Options &opts) : opts_(opts) {}

LoopCloser::~LoopCloser() = default;

std::size_t LoopCloser::NumSubmaps() const { return submaps_.size(); }

void LoopCloser::AnchorKeyframe(int kf_id, double /*timestamp_s*/,
                                 const Eigen::Isometry3d &pose) {
    Submap s;
    s.keyframe_id = kf_id;
    s.anchor_pose = pose;
    s.local_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    submaps_.push_back(std::move(s));
    active_ = &submaps_.back();
    EvictIfOverCapacity();
}

void LoopCloser::Accumulate(const PointCloudType::Ptr &body_scan,
                             const Eigen::Isometry3d &T_world_base) {
    if (!active_ || !body_scan || body_scan->empty()) return;

    // T_local_base = anchor_pose^-1 * T_world_base
    const Eigen::Matrix4f T_local_base =
        (active_->anchor_pose.inverse() * T_world_base).matrix().cast<float>();

    // Convert PointXYZINormal → PointXYZI (drop normals + curvature) and
    // transform to submap-local in one pass. Faster than a separate convert
    // + transformPointCloud since PCL's transform doesn't know about normals.
    pcl::PointCloud<pcl::PointXYZI> local;
    local.reserve(body_scan->size());
    for (const auto &p : body_scan->points) {
        const Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
        const Eigen::Vector4f w = T_local_base * v;
        pcl::PointXYZI o;
        o.x = w.x();
        o.y = w.y();
        o.z = w.z();
        o.intensity = p.intensity;
        local.push_back(o);
    }
    *active_->local_cloud += local;

    if (active_->local_cloud->size() > opts_.max_points_per_submap) {
        VoxelDownsample(*active_);
    }
}

std::vector<LoopMatch>
LoopCloser::DetectAtKeyframe(int kf_id, const Eigen::Isometry3d &latest_pose) {
    std::vector<LoopMatch> out;
    if (submaps_.empty()) return out;

    // The caller (laser_mapping.cc) runs AnchorKeyframe BEFORE DetectAtKeyframe,
    // so the submap with `keyframe_id == kf_id` is freshly-anchored and EMPTY
    // (scans haven't been accumulated into it yet — that happens on subsequent
    // ticks). The submap that actually holds content representing the local
    // scene around kf_id is the one that was "active" right before the anchor:
    // the second-to-last submap in the deque. Use THAT as the descriptor /
    // ICP source.
    //
    // Introduces a small (≤ keyframe_dist_thresh ≈ 0.5 m) translation error
    // between the descriptor's anchor and kf_id's pose; the pose graph's
    // odometry edges absorb it.
    if (submaps_.size() < 2) return out;
    const std::size_t latest_idx = submaps_.size() - 2;
    const Submap *latest_submap = &submaps_[latest_idx];
    if (!latest_submap->local_cloud || latest_submap->local_cloud->empty()) {
        return out;
    }

    // Per-keyframe candidate entry. The two ranking signals are:
    //   - sc_score: Scan Context column-shift distance (pose-independent)
    //   - d2:       anchor-pose squared distance (IEKF-drift-dependent)
    // We collect both; Scan Context is preferred when IEKF drift may have
    // placed the anchor beyond `revisit_radius`, but we keep a secondary
    // pose-proximity rank for the small-drift case.
    struct Ranked {
        std::size_t idx;
        float  d2;
        double ring_key_dist;
        double sc_score;
        int    sc_shift;
        bool   sc_passed;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(submaps_.size());

    const float r2 = opts_.revisit_radius * opts_.revisit_radius;
    const Eigen::Vector3f latest_xyz = latest_pose.translation().cast<float>();

    // Make sure the latest submap has a descriptor before querying any
    // older ones. Aggregation window is bounded by what's currently in
    // `submaps_`, so early in the run we simply aggregate less.
    if (opts_.sc_enabled) EnsureDescriptor(latest_idx);

    const Eigen::VectorXd &latest_key =
        opts_.sc_enabled ? latest_submap->sc_ring_key : Eigen::VectorXd();
    const Eigen::MatrixXd &latest_desc =
        opts_.sc_enabled ? latest_submap->sc_desc : Eigen::MatrixXd();

    for (std::size_t i = 0; i < submaps_.size(); ++i) {
        if (i == latest_idx) continue;
        const Submap &s = submaps_[i];
        if (std::abs(kf_id - s.keyframe_id) < opts_.min_age_frames) continue;
        if (!s.local_cloud || s.local_cloud->empty()) continue;

        Ranked r;
        r.idx = i;
        r.d2 = (s.anchor_pose.translation().cast<float>() - latest_xyz).squaredNorm();
        r.ring_key_dist = std::numeric_limits<double>::infinity();
        r.sc_score = 1.0;
        r.sc_shift = 0;
        r.sc_passed = false;

        if (opts_.sc_enabled) {
            EnsureDescriptor(i);
            if (s.sc_ready && latest_submap->sc_ready) {
                r.ring_key_dist =
                    ScanContext::RingKeyDistance(latest_key, s.sc_ring_key);
            }
        }
        ranked.push_back(r);
    }

    // Scan Context branch: full column-shift evaluation on the top-K
    // ring-key-closest candidates. These are the cheap-to-compare ones;
    // the full SC pass is O(num_sectors^2) per pair so we bound it.
    if (opts_.sc_enabled && !ranked.empty()) {
        std::sort(ranked.begin(), ranked.end(),
                  [](const Ranked &a, const Ranked &b) {
                      return a.ring_key_dist < b.ring_key_dist;
                  });
        const std::size_t sc_probe =
            std::min<std::size_t>(ranked.size(),
                                   std::max<std::size_t>(opts_.sc_top_k * 4, 8));
        for (std::size_t k = 0; k < sc_probe; ++k) {
            auto &r = ranked[k];
            if (r.ring_key_dist > opts_.sc_ring_key_threshold) break;
            const Submap &s = submaps_[r.idx];
            ScanContext::MatchOptions mo;
            mo.min_overlap_ratio = opts_.sc_min_overlap_ratio;
            const auto m = ScanContext::Distance(latest_desc, s.sc_desc, mo);
            r.sc_score = m.score;
            r.sc_shift = m.shift;
            r.sc_passed = r.sc_score < opts_.sc_score_threshold;
        }
    }

    // Diagnostic: summarize what SC saw on this pass so the user can tune
    // thresholds against observed data rather than guess.
    if (opts_.sc_enabled && !ranked.empty()) {
        double best_rk = std::numeric_limits<double>::infinity();
        double best_sc = std::numeric_limits<double>::infinity();
        int    best_sc_to = -1;
        for (const auto &r : ranked) {
            if (r.ring_key_dist < best_rk) best_rk = r.ring_key_dist;
            if (r.sc_score < best_sc) {
                best_sc = r.sc_score;
                best_sc_to = submaps_[r.idx].keyframe_id;
            }
        }
        if (kf_id % 100 == 0 || best_sc < opts_.sc_score_threshold) {
            spdlog::info("LCD kf {}: best_ring_key={:.3f} best_sc={:.3f} "
                         "(to kf {}, thresholds rk<{:.2f} sc<{:.2f}), candidates={}",
                         kf_id, best_rk, best_sc, best_sc_to,
                         opts_.sc_ring_key_threshold, opts_.sc_score_threshold,
                         ranked.size());
        }
    }

    // Final candidate ordering: SC-passers first (by ascending SC score),
    // then pose-proximity fallback (by ascending anchor distance, gated by
    // revisit_radius). Caps total ICP calls at `max_candidates_per_call`.
    std::vector<Ranked> sc_good;
    std::vector<Ranked> near_good;
    for (const auto &r : ranked) {
        if (r.sc_passed) sc_good.push_back(r);
        else if (r.d2 <= r2) near_good.push_back(r);
    }
    std::sort(sc_good.begin(), sc_good.end(),
              [](const Ranked &a, const Ranked &b) {
                  return a.sc_score < b.sc_score;
              });
    std::sort(near_good.begin(), near_good.end(),
              [](const Ranked &a, const Ranked &b) { return a.d2 < b.d2; });

    auto run_icp = [&](const Ranked &cand) {
        if (out.size() >= opts_.max_candidates_per_call) return;
        const Submap &s = submaps_[cand.idx];

        // Prefer the aggregated cloud (same one SC looked at) — a single
        // submap's narrow-FoV cloud rarely overlaps enough for ICP to
        // converge, even when SC finds a match. Fall back to local_cloud
        // if aggregation wasn't built (pose-proximity path, sc_enabled=false).
        const auto &src_cloud = (cand.sc_passed && latest_submap->aggregated_cloud)
                                     ? latest_submap->aggregated_cloud
                                     : latest_submap->local_cloud;
        const auto &tgt_cloud = (cand.sc_passed && s.aggregated_cloud)
                                     ? s.aggregated_cloud
                                     : s.local_cloud;
        if (!src_cloud || src_cloud->empty() || !tgt_cloud || tgt_cloud->empty()) return;

        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        icp.setInputSource(src_cloud);
        icp.setInputTarget(tgt_cloud);
        icp.setMaxCorrespondenceDistance(opts_.icp_max_correspondence);
        icp.setMaximumIterations(opts_.icp_max_iterations);

        // Initial guess. For SC matches, the descriptor is local-frame so
        // translation = 0, rotation = R_z(shift*delta_theta) — this lets
        // ICP find the true alignment even when anchor poses are meters
        // apart due to drift. For pose-proximity fallback, reuse the
        // anchor-relative transform (drift was small by premise).
        Eigen::Matrix4f init;
        if (cand.sc_passed) {
            const double yaw =
                cand.sc_shift * (2.0 * M_PI) /
                static_cast<double>(std::max(1, opts_.sc_num_sectors));
            Eigen::Isometry3d guess = Eigen::Isometry3d::Identity();
            guess.linear() =
                Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
            init = guess.matrix().cast<float>();
        } else {
            init = (s.anchor_pose.inverse() * latest_pose).matrix().cast<float>();
        }

        pcl::PointCloud<pcl::PointXYZI> aligned;
        icp.align(aligned, init);

        const bool converged = icp.hasConverged();
        const float fitness = converged ? icp.getFitnessScore() : -1.0f;
        // Only log on success or when the score was VERY close to passing.
        // Per-ICP-call logs flooded stdout at 1 line/kf when running
        // hundreds of rejected attempts.
        if (cand.sc_passed && converged &&
            fitness < opts_.icp_fitness_threshold * 2.0f) {
            spdlog::info("LCD ICP: kf {} -> kf {} fitness={:.4f} "
                         "(threshold={:.2f}, sc_score={:.3f}) {}",
                         kf_id, s.keyframe_id, fitness,
                         opts_.icp_fitness_threshold, cand.sc_score,
                         (fitness <= opts_.icp_fitness_threshold) ? "PASS" : "near-miss");
        }
        if (!converged) return;
        if (fitness > opts_.icp_fitness_threshold) return;

        LoopMatch m;
        m.from_id = kf_id;
        m.to_id = s.keyframe_id;
        // PCL's ICP returns T such that T * p_source = p_target — T^cand_latest.
        // g2o's EdgeSE3 measurement is T^from_to = T^latest_cand = (.).inverse().
        // See tests/test_pose_graph.cc::LoopEdgeDirectionConventionMatters.
        const Eigen::Isometry3d T_cand_latest(
            icp.getFinalTransformation().cast<double>());
        m.relative_pose = T_cand_latest.inverse();
        const double w = 1.0 / std::max(1e-3, static_cast<double>(fitness));
        m.information = Eigen::Matrix<double, 6, 6>::Identity() * w;
        out.push_back(m);
    };

    // Cap ICP ATTEMPTS (not just matches) — SC's top-ranked candidate is
    // almost always the right answer when any loop is detectable; running
    // ICP on #2..#K is typically wasted work at ~30-100 ms a pop. The
    // existing `max_candidates_per_call` guard inside run_icp only stops
    // once we have N *accepted* matches, so failed attempts slip past.
    const std::size_t sc_cap = std::min<std::size_t>(sc_good.size(), opts_.sc_top_k);
    for (std::size_t i = 0; i < sc_cap; ++i) run_icp(sc_good[i]);
    const std::size_t near_cap =
        std::min<std::size_t>(near_good.size(), opts_.max_candidates_per_call);
    for (std::size_t i = 0; i < near_cap; ++i) run_icp(near_good[i]);

    return out;
}

void LoopCloser::ApplyCorrection(
    const std::vector<PoseGraph::Keyframe> &corrected) {
    // Small N — linear scan is fine. (faster-lio's pose graph is bounded
    // by `max_submaps` × keyframe spacing.)
    for (auto &s : submaps_) {
        for (const auto &k : corrected) {
            if (k.id == s.keyframe_id) {
                s.anchor_pose = k.pose;
                // Submap local_cloud is unchanged (still in its own local
                // frame), so the SC descriptor — which only depends on
                // local_cloud — remains valid. No invalidation needed.
                break;
            }
        }
    }
}

void LoopCloser::EnsureDescriptor(std::size_t submap_idx) {
    if (submap_idx >= submaps_.size()) return;
    Submap &s = submaps_[submap_idx];
    if (s.sc_ready) return;

    ScanContext::Options sc_opts;
    sc_opts.num_rings   = opts_.sc_num_rings;
    sc_opts.num_sectors = opts_.sc_num_sectors;
    sc_opts.max_range   = opts_.sc_max_range;

    // Aggregate the last W submaps (including s itself) into s's local
    // frame. Widens FoV so a handheld walking bag with a 70°-wide Avia
    // scan still builds a near-panoramic descriptor. W=1 falls back to
    // the single submap's own cloud.
    const int W = std::max(1, opts_.sc_aggregation_window);
    const std::size_t start =
        (submap_idx + 1 >= static_cast<std::size_t>(W))
            ? submap_idx + 1 - static_cast<std::size_t>(W)
            : 0;

    auto agg = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    for (std::size_t j = start; j <= submap_idx; ++j) {
        const Submap &src = submaps_[j];
        if (!src.local_cloud || src.local_cloud->empty()) continue;
        if (j == submap_idx) {
            *agg += *src.local_cloud;
            continue;
        }
        const Eigen::Matrix4f T =
            (s.anchor_pose.inverse() * src.anchor_pose).matrix().cast<float>();
        pcl::PointCloud<pcl::PointXYZI> transformed;
        transformed.reserve(src.local_cloud->size());
        for (const auto &p : src.local_cloud->points) {
            const Eigen::Vector4f v(p.x, p.y, p.z, 1.0f);
            const Eigen::Vector4f w = T * v;
            pcl::PointXYZI o;
            o.x = w.x(); o.y = w.y(); o.z = w.z(); o.intensity = p.intensity;
            transformed.push_back(o);
        }
        *agg += transformed;
    }

    if (agg->empty()) return;

    // Voxel-downsample the aggregated cloud. An 8-submap aggregation can
    // stack ~80k points with heavy overlap (each submap already voxelized
    // at submap-level, but overlapping windows produce near-duplicate
    // points); collapsing to a coarser grid keeps ICP's kd-tree build +
    // correspondence search cheap without losing geometry the descriptor
    // or ICP cares about. Leaf ≥ per-submap voxel_size so we never
    // UP-sample relative to the source resolution.
    const float agg_leaf = std::max(opts_.voxel_size * 2.5f, 0.5f);
    pcl::VoxelGrid<pcl::PointXYZI> vox;
    vox.setLeafSize(agg_leaf, agg_leaf, agg_leaf);
    vox.setInputCloud(agg);
    auto agg_ds = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    vox.filter(*agg_ds);
    if (agg_ds->empty()) return;

    // Cache both the aggregated cloud (for ICP) and the descriptor (for SC).
    s.aggregated_cloud = agg_ds;
    s.sc_desc = ScanContext::ComputeDescriptor(*agg_ds, sc_opts);
    s.sc_ring_key = ScanContext::ComputeRingKey(s.sc_desc);
    s.sc_ready = true;
}

void LoopCloser::EvictIfOverCapacity() {
    while (submaps_.size() > opts_.max_submaps) {
        // If the active submap is at the front (shouldn't normally happen —
        // active_ is the back), drop the active pointer.
        if (active_ == &submaps_.front()) {
            active_ = nullptr;
        }
        submaps_.pop_front();
    }
    // Re-bind active to the back, since pop_front may have invalidated it.
    if (!submaps_.empty()) {
        active_ = &submaps_.back();
    } else {
        active_ = nullptr;
    }
}

void LoopCloser::VoxelDownsample(Submap &s) const {
    if (!s.local_cloud || s.local_cloud->empty() || opts_.voxel_size <= 0.0f) return;
    pcl::VoxelGrid<pcl::PointXYZI> vox;
    vox.setLeafSize(opts_.voxel_size, opts_.voxel_size, opts_.voxel_size);
    vox.setInputCloud(s.local_cloud);
    auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    vox.filter(*out);
    s.local_cloud = std::move(out);
}

}  // namespace faster_lio
