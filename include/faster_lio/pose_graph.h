#ifndef FASTER_LIO_POSE_GRAPH_H
#define FASTER_LIO_POSE_GRAPH_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <memory>
#include <vector>

// Forward-declare to keep g2o headers out of the public interface.
namespace g2o {
class SparseOptimizer;
}

namespace faster_lio {

/// g2o-backed SE(3) pose graph for trajectory optimization and loop closure.
///
/// Typical call sequence:
///   Init() → [TryAddKeyframe() → if ShouldOptimize() then Optimize()] loop.
///
/// The graph accumulates keyframe vertices connected by odometry edges.
/// Loop closure edges can be injected via AddLoopClosure(). Calling
/// Optimize() runs Levenberg-Marquardt and updates the correction transform
/// returned by GetCorrection().
class PoseGraph {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Options {
        double keyframe_dist_thresh = 0.5;     // m  — min translation to create a keyframe
        double keyframe_angle_thresh = 0.175;  // rad (~10 deg) — min rotation
        int optimize_every_n = 10;             // run optimization every N keyframes
        int max_iterations = 20;               // Levenberg-Marquardt iterations
    };

    struct Keyframe {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int id = -1;
        double timestamp = 0.0;
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    };

    PoseGraph();
    ~PoseGraph();

    /// Initialize the g2o optimizer. Must be called once before anything else.
    void Init(const Options &opts);

    /// Try to add a new odometry keyframe.
    /// @param pose       World-frame SE(3) pose (with any prior correction already applied)
    /// @param timestamp  Frame timestamp (seconds)
    /// @param covariance 6x6 [pos(0..2), rot(3..5)] covariance from the IEKF posterior
    /// @return keyframe id (>=0) if accepted, -1 if rejected (too close to previous keyframe)
    int TryAddKeyframe(const Eigen::Isometry3d &pose, double timestamp,
                       const Eigen::Matrix<double, 6, 6> &covariance);

    /// Add a loop closure constraint between two existing keyframes.
    /// @param from_id       Source keyframe id
    /// @param to_id         Target keyframe id
    /// @param relative_pose Measured relative transform from → to
    /// @param information   6x6 information matrix for the constraint
    void AddLoopClosure(int from_id, int to_id,
                        const Eigen::Isometry3d &relative_pose,
                        const Eigen::Matrix<double, 6, 6> &information);

    /// Run Levenberg-Marquardt on the full graph and update the correction.
    bool Optimize();

    /// True if enough keyframes have accumulated since the last optimization.
    bool ShouldOptimize() const;

    /// Cumulative rigid correction that maps raw odometry poses → optimized
    /// poses. Apply as: T_corrected = GetCorrection() * T_iekf
    ///
    /// This is the COMPOSED product of all per-optimize deltas since Init().
    /// Callers should replace (not compose) their stored copy each time —
    /// composition is handled internally.
    Eigen::Isometry3d GetCorrection() const;

    /// All keyframes with their (possibly optimized) poses.
    std::vector<Keyframe> GetKeyframes() const;

    int NumKeyframes() const { return static_cast<int>(keyframes_.size()); }
    int NumEdges() const { return edges_count_; }
    int NumLoopEdges() const { return loop_edges_count_; }
    bool HasOptimized() const { return has_optimized_; }

   private:
    Options opts_;
    std::unique_ptr<g2o::SparseOptimizer> optimizer_;

    std::vector<Keyframe> keyframes_;
    Eigen::Isometry3d correction_ = Eigen::Isometry3d::Identity();
    bool has_optimized_ = false;
    int edges_count_ = 0;
    int loop_edges_count_ = 0;
    int keyframes_since_optimize_ = 0;
};

}  // namespace faster_lio

#endif  // FASTER_LIO_POSE_GRAPH_H
