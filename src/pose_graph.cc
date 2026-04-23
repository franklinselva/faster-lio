#include "faster_lio/pose_graph.h"

#include <spdlog/spdlog.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/vertex_se3.h>

namespace faster_lio {

PoseGraph::PoseGraph() = default;
PoseGraph::~PoseGraph() = default;

void PoseGraph::Init(const Options &opts) {
    opts_ = opts;
    optimizer_ = std::make_unique<g2o::SparseOptimizer>();

    using BlockSolver = g2o::BlockSolverX;
    using LinearSolver = g2o::LinearSolverEigen<BlockSolver::PoseMatrixType>;

    auto linear = std::make_unique<LinearSolver>();
    auto block = std::make_unique<BlockSolver>(std::move(linear));
    auto *algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block));

    optimizer_->setAlgorithm(algorithm);
    optimizer_->setVerbose(false);

    spdlog::info("PoseGraph: initialized (keyframe_dist={:.2f}m angle={:.1f}deg optimize_every={})",
                 opts_.keyframe_dist_thresh,
                 opts_.keyframe_angle_thresh * 180.0 / M_PI,
                 opts_.optimize_every_n);
}

int PoseGraph::TryAddKeyframe(const Eigen::Isometry3d &pose, double timestamp,
                              const Eigen::Matrix<double, 6, 6> &covariance) {
    if (!optimizer_) return -1;

    // First keyframe is always accepted and anchored (fixed).
    if (keyframes_.empty()) {
        Keyframe kf;
        kf.id = 0;
        kf.timestamp = timestamp;
        kf.pose = pose;
        keyframes_.push_back(kf);

        auto *v = new g2o::VertexSE3();
        v->setId(0);
        v->setEstimate(pose);
        v->setFixed(true);
        optimizer_->addVertex(v);

        keyframes_since_optimize_++;
        spdlog::debug("PoseGraph: first keyframe id=0");
        return 0;
    }

    // Gate by distance / angle from the last keyframe.
    const auto &last_kf = keyframes_.back();
    Eigen::Isometry3d delta = last_kf.pose.inverse() * pose;
    double dist = delta.translation().norm();
    double angle = Eigen::AngleAxisd(delta.rotation()).angle();

    if (dist < opts_.keyframe_dist_thresh && angle < opts_.keyframe_angle_thresh) {
        return -1;
    }

    int id = static_cast<int>(keyframes_.size());
    Keyframe kf;
    kf.id = id;
    kf.timestamp = timestamp;
    kf.pose = pose;
    keyframes_.push_back(kf);

    // Vertex
    auto *v = new g2o::VertexSE3();
    v->setId(id);
    v->setEstimate(pose);
    optimizer_->addVertex(v);

    // Odometry edge from previous keyframe
    auto *edge = new g2o::EdgeSE3();
    edge->setVertex(0, optimizer_->vertex(id - 1));
    edge->setVertex(1, optimizer_->vertex(id));
    edge->setMeasurement(delta);

    // Information = inverse of covariance, with clamped diagonal for stability.
    Eigen::Matrix<double, 6, 6> cov = covariance;
    for (int i = 0; i < 6; i++) {
        cov(i, i) = std::max(cov(i, i), 1e-8);
    }
    edge->setInformation(cov.inverse());
    optimizer_->addEdge(edge);
    edges_count_++;

    keyframes_since_optimize_++;
    spdlog::debug("PoseGraph: keyframe id={} dist={:.3f}m angle={:.2f}deg edges={}",
                  id, dist, angle * 180.0 / M_PI, edges_count_);
    return id;
}

void PoseGraph::AddLoopClosure(int from_id, int to_id,
                               const Eigen::Isometry3d &relative_pose,
                               const Eigen::Matrix<double, 6, 6> &information) {
    if (!optimizer_) return;
    if (from_id < 0 || from_id >= NumKeyframes() ||
        to_id < 0 || to_id >= NumKeyframes()) {
        spdlog::warn("PoseGraph: invalid loop closure ids ({} -> {})", from_id, to_id);
        return;
    }

    auto *edge = new g2o::EdgeSE3();
    edge->setVertex(0, optimizer_->vertex(from_id));
    edge->setVertex(1, optimizer_->vertex(to_id));
    edge->setMeasurement(relative_pose);
    edge->setInformation(information);
    optimizer_->addEdge(edge);
    edges_count_++;
    loop_edges_count_++;

    spdlog::info("PoseGraph: loop closure {} -> {} added (total loop edges: {})",
                 from_id, to_id, loop_edges_count_);
}

bool PoseGraph::Optimize() {
    if (!optimizer_ || keyframes_.size() < 2) return false;

    spdlog::info("PoseGraph: optimizing {} vertices, {} edges",
                 optimizer_->vertices().size(), optimizer_->edges().size());

    // Capture the latest keyframe's pre-optimization pose for correction.
    Eigen::Isometry3d pre_opt_latest = keyframes_.back().pose;

    optimizer_->initializeOptimization();
    int iters = optimizer_->optimize(opts_.max_iterations);

    if (iters <= 0) {
        spdlog::warn("PoseGraph: optimization failed (0 iterations)");
        return false;
    }

    // Read back optimized vertex estimates.
    for (auto &kf : keyframes_) {
        auto *v = static_cast<g2o::VertexSE3 *>(optimizer_->vertex(kf.id));
        if (v) {
            kf.pose = v->estimate();
        }
    }

    // Per-optimize delta: how much the latest vertex moved during THIS
    // call. Compose into the cumulative correction so callers don't have
    // to remember it themselves.
    //
    // Why composition matters: each Optimize() only knows the last
    // vertex's pre-/post-optimize poses, not the raw IEKF pose that
    // produced them. The raw-to-optimized mapping is the product of all
    // per-optimize deltas. If the caller overwrites its stored correction
    // with a single delta, new keyframes get inserted in a frame that
    // disagrees with the existing chain → LM produces a new delta to
    // reconcile → geometric blow-up (observed on hkust_campus_00: 54
    // optimizes, correction saturating at ~150 m with zero loop edges).
    // See tests/test_pose_graph.cc::CorrectionOverwriteCompoundsOnRotatingTrajectory.
    const Eigen::Isometry3d delta =
        keyframes_.back().pose * pre_opt_latest.inverse();
    correction_ = delta * correction_;

    has_optimized_ = true;
    keyframes_since_optimize_ = 0;

    spdlog::info("PoseGraph: converged in {} iterations, chi2={:.4f}, "
                 "delta_t={:.4f}m cumulative_t={:.4f}m",
                 iters, optimizer_->chi2(), delta.translation().norm(),
                 correction_.translation().norm());
    return true;
}

bool PoseGraph::ShouldOptimize() const {
    // Only optimize when there are loop-closure edges to satisfy. An
    // odometry-only chain is already at the LM optimum (every edge built
    // from the same vertex estimates has zero residual), so calling LM is
    // a mathematical no-op. Numerically it's harmful: Cholesky of the
    // ~6N×6N Hessian produces floating-point noise on the order of
    // 1e-15 per call, and each round-trip of kf insert → compose →
    // optimize on a rotating trajectory amplifies it geometrically.
    // On hkust_campus_00 that's how the stored correction reached 150 m
    // with zero real loop edges.
    return keyframes_since_optimize_ >= opts_.optimize_every_n &&
           loop_edges_count_ > 0;
}

Eigen::Isometry3d PoseGraph::GetCorrection() const {
    return correction_;
}

std::vector<PoseGraph::Keyframe> PoseGraph::GetKeyframes() const {
    return keyframes_;
}

}  // namespace faster_lio
