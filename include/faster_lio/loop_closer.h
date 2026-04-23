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
    };

    void EvictIfOverCapacity();
    void VoxelDownsample(Submap &s) const;

    Options opts_;
    std::deque<Submap> submaps_;
    Submap *active_ = nullptr;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LOOP_CLOSER_H
