#include "faster_lio/loop_closer.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>

#include <algorithm>
#include <cmath>

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

    // Find the submap matching `kf_id`. Always the most recently anchored
    // one in normal flow, but search defensively in case of out-of-order
    // anchoring.
    const Submap *latest_submap = nullptr;
    for (const auto &s : submaps_) {
        if (s.keyframe_id == kf_id) {
            latest_submap = &s;
            break;
        }
    }
    if (!latest_submap || !latest_submap->local_cloud ||
        latest_submap->local_cloud->empty()) {
        return out;
    }

    // Rank candidates by anchor distance so the tightest geometric match
    // gets ICP-verified first.
    struct Ranked {
        float d2;
        const Submap *submap;
    };
    std::vector<Ranked> ranked;
    const float r2 = opts_.revisit_radius * opts_.revisit_radius;
    const Eigen::Vector3f latest_xyz = latest_pose.translation().cast<float>();

    for (const auto &s : submaps_) {
        if (s.keyframe_id == kf_id) continue;
        if (std::abs(kf_id - s.keyframe_id) < opts_.min_age_frames) continue;
        if (!s.local_cloud || s.local_cloud->empty()) continue;
        const float d2 =
            (s.anchor_pose.translation().cast<float>() - latest_xyz).squaredNorm();
        if (d2 > r2) continue;
        ranked.push_back({d2, &s});
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const Ranked &a, const Ranked &b) { return a.d2 < b.d2; });

    for (const auto &cand : ranked) {
        if (out.size() >= opts_.max_candidates_per_call) break;

        pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
        icp.setInputSource(latest_submap->local_cloud);
        icp.setInputTarget(cand.submap->local_cloud);
        icp.setMaxCorrespondenceDistance(opts_.icp_max_correspondence);
        icp.setMaximumIterations(opts_.icp_max_iterations);

        const Eigen::Matrix4f init =
            (cand.submap->anchor_pose.inverse() * latest_pose).matrix().cast<float>();
        pcl::PointCloud<pcl::PointXYZI> aligned;
        icp.align(aligned, init);

        if (!icp.hasConverged()) continue;
        const float fitness = icp.getFitnessScore();
        if (fitness > opts_.icp_fitness_threshold) continue;

        LoopMatch m;
        m.from_id = kf_id;
        m.to_id = cand.submap->keyframe_id;
        m.relative_pose =
            Eigen::Isometry3d(icp.getFinalTransformation().cast<double>());
        const double w = 1.0 / std::max(1e-3, static_cast<double>(fitness));
        m.information = Eigen::Matrix<double, 6, 6>::Identity() * w;
        out.push_back(m);
    }

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
                break;
            }
        }
    }
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
