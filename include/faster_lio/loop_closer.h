#ifndef FASTER_LIO_LOOP_CLOSER_H
#define FASTER_LIO_LOOP_CLOSER_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstddef>
#include <deque>
#include <memory>
#include <vector>

#include "faster_lio/common_lib.h"
#include "faster_lio/pose_graph.h"
#include "faster_lio/scan_context.h"

namespace faster_lio {

/// One loop-closure match discovered by `LoopCloser::DetectAtKeyframe`.
/// Contracts directly with `PoseGraph::AddLoopClosure(from_id, to_id,
/// relative_pose, information)`.
struct LoopMatch {
    int from_id{-1};
    int to_id{-1};
    Eigen::Isometry3d relative_pose{Eigen::Isometry3d::Identity()};
    Eigen::Matrix<double, 6, 6> information{Eigen::Matrix<double, 6, 6>::Identity()};
};

/// Pose-proximity-gated ICP loop-closure detector.
///
/// One submap per pose-graph keyframe (anchored at the keyframe's pose,
/// holding accumulated body-frame scans transformed into submap-local
/// coordinates). For each new keyframe, ranks older keyframes by anchor
/// distance, ICP-verifies the closest few, and emits matches whose ICP
/// fitness passes the threshold.
///
/// All API calls run on `LaserMapping::Run()`'s thread; no internal
/// synchronisation. Pose-graph corrections are absorbed via
/// `ApplyCorrection`, which only updates `anchor_pose` (no point-cloud
/// work — submaps remain in the local frame).
class LoopCloser {
   public:
    struct Options {
        // ── Detection gate ───────────────────────────────────────────────
        float revisit_radius = 5.0f;           ///< m — max anchor distance for a candidate
        int min_age_frames = 50;               ///< min keyframe-id separation
        float icp_max_correspondence = 0.5f;   ///< m — ICP point-pair cutoff
        int icp_max_iterations = 30;
        float icp_fitness_threshold = 0.3f;    ///< m² — accept if mean-squared correspondence is below this
        std::size_t max_candidates_per_call = 4;

        // ── Submap accumulation ──────────────────────────────────────────
        float voxel_size = 0.10f;              ///< m — periodic downsample leaf
        std::size_t max_points_per_submap = 200'000;
        std::size_t max_submaps = 256;         ///< oldest submaps evicted beyond this

        // ── Scan Context (global, pose-independent) ──────────────────────
        // Used when IEKF drift is large enough that pose-proximity gating
        // misses true revisits (typical for >10 m drift on solid-state
        // LiDAR bags). Built on a per-keyframe aggregated cloud so a
        // narrow-FoV scan still contributes to a near-panoramic descriptor.
        bool  sc_enabled              = true;
        int   sc_num_rings            = 20;
        int   sc_num_sectors          = 60;
        double sc_max_range           = 80.0;  ///< m — beyond → ignored
        int   sc_aggregation_window   = 4;     ///< # of submaps to aggregate per descriptor
        double sc_ring_key_threshold  = 0.20;  ///< L2 ring-key distance to enter full SC match
        double sc_score_threshold     = 0.35;  ///< column-shift distance to be a candidate
        double sc_min_overlap_ratio   = 0.30;  ///< min fraction of sectors in common (both non-empty)
        std::size_t sc_top_k          = 3;     ///< full-SC matches per call that pass to ICP
    };

    explicit LoopCloser(const Options &opts);
    ~LoopCloser();

    LoopCloser(const LoopCloser &) = delete;
    LoopCloser &operator=(const LoopCloser &) = delete;

    /// Anchor a new submap when `PoseGraph::TryAddKeyframe` accepts a keyframe.
    void AnchorKeyframe(int kf_id, double timestamp_s,
                        const Eigen::Isometry3d &pose);

    /// Fold the latest body-frame downsampled scan into the active submap.
    /// Cheap (~57k points × matrix-multiply); call every LIO tick.
    void Accumulate(const PointCloudType::Ptr &body_scan,
                    const Eigen::Isometry3d &T_world_base);

    /// Detect loop closures for the most recently anchored keyframe.
    /// Returns 0..`max_candidates_per_call` matches; each is ready for
    /// `PoseGraph::AddLoopClosure`.
    std::vector<LoopMatch> DetectAtKeyframe(int kf_id,
                                             const Eigen::Isometry3d &latest_pose);

    /// Re-anchor submaps from a corrected keyframe list (matched by id).
    /// Only updates anchor poses — no per-point work.
    void ApplyCorrection(const std::vector<PoseGraph::Keyframe> &corrected);

    std::size_t NumSubmaps() const;

   private:
    struct Submap {
        int keyframe_id = -1;
        Eigen::Isometry3d anchor_pose = Eigen::Isometry3d::Identity();
        pcl::PointCloud<pcl::PointXYZI>::Ptr local_cloud;
        // Wide-FoV cloud built from `sc_aggregation_window` submaps and
        // stored in THIS submap's local frame. Used both to build the SC
        // descriptor and as the source/target for ICP verification —
        // single-submap clouds from solid-state LiDAR are too narrow for
        // ICP to converge even when SC declares a match.
        pcl::PointCloud<pcl::PointXYZI>::Ptr aggregated_cloud;
        // Scan Context descriptor, built lazily on first access via
        // EnsureDescriptor(). Invalidated when anchor_pose changes
        // (ApplyCorrection) or the cloud is rebuilt.
        Eigen::MatrixXd sc_desc;
        Eigen::VectorXd sc_ring_key;
        bool sc_ready = false;
    };

    void EvictIfOverCapacity();
    void VoxelDownsample(Submap &s) const;
    /// Populate s.sc_desc / s.sc_ring_key from an aggregated window of the
    /// last Options::sc_aggregation_window submaps ending at `s`, all
    /// transformed into s's local frame. No-op if already ready.
    void EnsureDescriptor(std::size_t submap_idx);

    Options opts_;
    std::deque<Submap> submaps_;
    Submap *active_ = nullptr;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LOOP_CLOSER_H
