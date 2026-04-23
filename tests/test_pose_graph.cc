// ─────────────────────────────────────────────────────────────────────────
// PoseGraph regression battery.
//
// The stack-state memory flags one critical bug: PoseGraph::Optimize()
// shrinks the keyframe chain even with zero loop-closure edges. On the
// hkust_campus_00 bag, path went 1337 m → 345 m once pose_graph=true.
//
// This file isolates PoseGraph from LaserMapping/IEKF so we can pin the
// root cause without the noise of a full replay. We test four theories
// from most to least likely:
//
//   A. Identity-covariance chain: with perfectly informative edges, chi2
//      is zero at initialization; optimize must be a no-op.
//   B. Growing marginal covariance (mimicking IEKF posterior): the code
//      feeds g2o the *absolute* pose covariance, but EdgeSE3 wants the
//      *relative* measurement covariance. Bet: this is the shrink.
//   C. Non-PD information after diagonal-only clamp (pos-rot cross block).
//      Direct .inverse() on a non-PD matrix produces garbage information.
//   D. Correction composition when Optimize is a no-op: correction should
//      be identity, not drift between calls.
//
// Reporting: every test prints chain length before/after and the norm of
// the correction so `ctest --output-on-failure` shows the evidence.
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "faster_lio/pose_graph.h"

using namespace faster_lio;

namespace {

// Build a straight-line trajectory: N keyframes spaced `step` meters apart
// along +X, zero rotation. This is a degenerate but honest stand-in for
// a stretch of hallway — the case that should be preserved exactly.
std::vector<Eigen::Isometry3d> MakeStraightLine(int n, double step) {
    std::vector<Eigen::Isometry3d> poses;
    poses.reserve(n);
    for (int i = 0; i < n; ++i) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(i * step, 0.0, 0.0);
        poses.push_back(T);
    }
    return poses;
}

// Build a circular trajectory: N keyframes on a circle of radius R, with
// the body yaw always tangent to the circle (forward-facing). This
// exercises the non-trivial rotation part of every odometry edge.
std::vector<Eigen::Isometry3d> MakeCircle(int n, double radius) {
    std::vector<Eigen::Isometry3d> poses;
    poses.reserve(n);
    const double dtheta = 2.0 * M_PI / n;
    for (int i = 0; i < n; ++i) {
        const double theta = i * dtheta;
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(radius * std::cos(theta), radius * std::sin(theta), 0.0);
        T.linear() = Eigen::AngleAxisd(theta + M_PI / 2.0, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        poses.push_back(T);
    }
    return poses;
}

double ChainLength(const std::vector<PoseGraph::Keyframe>& kfs) {
    double length = 0.0;
    for (size_t i = 1; i < kfs.size(); ++i) {
        length += (kfs[i].pose.translation() - kfs[i - 1].pose.translation()).norm();
    }
    return length;
}

PoseGraph::Options DefaultOpts() {
    PoseGraph::Options o;
    o.keyframe_dist_thresh = 0.0;   // accept every pose we feed
    o.keyframe_angle_thresh = 0.0;
    o.optimize_every_n = 1'000'000; // disable auto-optimize; we call explicitly
    o.max_iterations = 30;
    return o;
}

}  // namespace

// ─── Theory A: identity covariance on every edge, zero loop closures.
// Expectation: chi2 starts at 0 (edge measurements built from the same
// input poses), optimize is a no-op, chain length unchanged.
TEST(PoseGraph, IdentityCovChainIsPreservedWithNoLoops) {
    auto poses = MakeStraightLine(50, 0.5);  // 25 m chain
    const double length_before_m = 24.5;     // 49 segments × 0.5 m

    PoseGraph pg;
    pg.Init(DefaultOpts());

    Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
    for (size_t i = 0; i < poses.size(); ++i) {
        int id = pg.TryAddKeyframe(poses[i], static_cast<double>(i), I6);
        ASSERT_EQ(id, static_cast<int>(i));
    }

    ASSERT_TRUE(pg.Optimize());

    const auto kfs = pg.GetKeyframes();
    const double length_after = ChainLength(kfs);
    std::cout << "[A] length before=" << length_before_m << " after=" << length_after
              << " correction=" << pg.GetCorrection().translation().norm() << "m\n";

    EXPECT_NEAR(length_after, length_before_m, 1e-6);
    EXPECT_LT(pg.GetCorrection().translation().norm(), 1e-6);
}

// ─── Theory B (MAIN HYPOTHESIS): growing marginal covariance.
// LaserMapping feeds the IEKF's marginal pos/rot covariance block. That
// grows as dead-reckoning accumulates uncertainty. If we pass it as the
// relative-measurement information, later edges become weakly informed.
// Because vertex 0 is fixed, LM can (and will) shift later vertices
// toward the fixed anchor to reduce the strong early-edge residuals.
TEST(PoseGraph, GrowingMarginalCovShrinksChain) {
    auto poses = MakeStraightLine(100, 0.5);  // 49.5 m chain
    const double length_before_m = 49.5;

    PoseGraph pg;
    pg.Init(DefaultOpts());

    for (size_t i = 0; i < poses.size(); ++i) {
        // Marginal pose covariance grows roughly ~ i (dead reckoning). We
        // scale linearly for a pessimistic-but-realistic picture of what
        // the IEKF posterior looks like far from the last snap.
        Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
        double s = 1e-4 * std::max<double>(1, static_cast<int>(i));  // pos 1cm²→1m²
        cov.block<3, 3>(0, 0) *= s;
        cov.block<3, 3>(3, 3) *= s * 0.1;  // rotation a bit tighter
        int id = pg.TryAddKeyframe(poses[i], static_cast<double>(i), cov);
        ASSERT_EQ(id, static_cast<int>(i));
    }

    ASSERT_TRUE(pg.Optimize());

    const auto kfs = pg.GetKeyframes();
    const double length_after = ChainLength(kfs);
    std::cout << "[B] length before=" << length_before_m << " after=" << length_after
              << " ratio=" << (length_after / length_before_m)
              << " correction=" << pg.GetCorrection().translation().norm() << "m\n";

    // This is the bug: we EXPECT shrinkage here if the hypothesis holds.
    // Either way, the test prints the ratio so we can read the damage.
    // For the "fixed" version we'd flip the expectation to NEAR.
    if (length_after / length_before_m < 0.95) {
        std::cout << "[B] REPRODUCED: chain shrunk to "
                  << 100.0 * length_after / length_before_m << "% of input\n";
    }
    // Assert the healthy outcome — this should fail until the covariance
    // hand-off is fixed (marginal → relative).
    EXPECT_GT(length_after / length_before_m, 0.95);
}

// ─── Theory C: information from .inverse() on a non-PD matrix.
// Construct a covariance that is diagonally positive but has large pos-rot
// cross terms that push overall eigenvalues negative. The diagonal clamp
// in pose_graph.cc:91-93 doesn't rescue this; .inverse() produces noise.
TEST(PoseGraph, NonPDCovarianceProducesBadInformation) {
    auto poses = MakeStraightLine(20, 1.0);  // short chain

    PoseGraph pg;
    pg.Init(DefaultOpts());

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-3;
    // Inject a large symmetric pos-rot coupling that breaks PD-ness.
    cov(0, 3) = cov(3, 0) = 0.5;
    cov(1, 4) = cov(4, 1) = 0.5;
    cov(2, 5) = cov(5, 2) = 0.5;

    // Sanity: confirm the matrix is actually not PD so the test premise holds.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> es(cov);
    const double min_eig = es.eigenvalues().minCoeff();
    std::cout << "[C] min eigenvalue of cov=" << min_eig << "\n";
    ASSERT_LT(min_eig, 0.0) << "test premise — cov should be non-PD";

    for (size_t i = 0; i < poses.size(); ++i) {
        pg.TryAddKeyframe(poses[i], static_cast<double>(i), cov);
    }
    ASSERT_TRUE(pg.Optimize());

    const auto kfs = pg.GetKeyframes();
    const double length_after = ChainLength(kfs);
    std::cout << "[C] length after optimize=" << length_after << " (expected 19.0)\n";

    // Healthy behavior: code should defend against non-PD covariance
    // (e.g., eigenvalue clamp) and produce the identity optimum.
    EXPECT_NEAR(length_after, 19.0, 1e-3);
}

// ─── Theory E: a false-positive loop claiming vertex N is at the origin
// would collapse the chain. Verify the optimizer respects edge information
// vs. the rigid spine of odometry edges (sanity — not actually a bug, but
// pins what a bad LCD match would do).
TEST(PoseGraph, FalsePositiveLoopCollapsesChainIfInfoOutweighsOdom) {
    auto poses = MakeStraightLine(50, 1.0);  // 49 m straight chain

    PoseGraph pg;
    pg.Init(DefaultOpts());

    // Weak odometry covariance so loop edges can dominate.
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
    for (size_t i = 0; i < poses.size(); ++i) {
        pg.TryAddKeyframe(poses[i], static_cast<double>(i), cov);
    }

    // False loop: claim last kf coincides with kf 0, with STRONG information.
    Eigen::Matrix<double, 6, 6> strong =
        Eigen::Matrix<double, 6, 6>::Identity() * 1e6;
    pg.AddLoopClosure(static_cast<int>(poses.size()) - 1, 0,
                      Eigen::Isometry3d::Identity(), strong);

    ASSERT_TRUE(pg.Optimize());

    const auto kfs = pg.GetKeyframes();
    const double length_after = ChainLength(kfs);
    std::cout << "[E] length before=49.0 after=" << length_after
              << " (false-positive-loop scenario)\n";

    // Expect shrink. Not asserting bounds — just documenting behavior.
    EXPECT_LT(length_after, 49.0);
}

// ─── Theory F: mimic the actual LaserMapping integration loop. This is
// the path in MaybeUpdatePoseGraph:
//   corrected = pg_correction_ * iekf_pose
//   pg.TryAddKeyframe(corrected, ...)
//   if optimize fires: pg_correction_ = pg.GetCorrection()
// Feed a straight-line IEKF stream for 300 keyframes, optimize every 10,
// and verify the correction-accumulated trajectory length stays bounded.
// The user's memory says path went 1337 m → 345 m on the real bag; this
// test reproduces that feedback loop with synthetic inputs to see if the
// integration itself (not PG) is what shrinks the output.
TEST(PoseGraph, IntegrationFeedbackLoopPreservesLength) {
    const int N = 300;
    const double step = 0.5;
    auto iekf_poses = MakeStraightLine(N, step);  // 149.5 m raw IEKF

    PoseGraph::Options opts = DefaultOpts();
    opts.optimize_every_n = 10;
    PoseGraph pg;
    pg.Init(opts);

    Eigen::Isometry3d pg_correction = Eigen::Isometry3d::Identity();
    std::vector<Eigen::Isometry3d> output_path;
    output_path.reserve(N);

    for (int i = 0; i < N; ++i) {
        // Real code's covariance: growing marginal from IEKF.
        Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
        double s = 1e-4 * std::max(1, i);
        cov.block<3, 3>(0, 0) *= s;
        cov.block<3, 3>(3, 3) *= s * 0.1;

        // Mimic MaybeUpdatePoseGraph line 1170.
        const Eigen::Isometry3d corrected = pg_correction * iekf_poses[i];
        pg.TryAddKeyframe(corrected, static_cast<double>(i), cov);

        if (pg.ShouldOptimize()) {
            if (pg.Optimize()) {
                pg_correction = pg.GetCorrection();
            }
        }

        // Mimic laser_mapping.cc:442-449 output-path composition.
        output_path.push_back(pg_correction * iekf_poses[i]);
    }

    // Length of the output path (what the viewer draws).
    double output_len = 0.0;
    for (size_t i = 1; i < output_path.size(); ++i) {
        output_len += (output_path[i].translation() -
                       output_path[i - 1].translation()).norm();
    }
    const double expected_len = step * (N - 1);  // 149.5 m
    std::cout << "[F] raw iekf len=" << expected_len
              << " output len=" << output_len
              << " ratio=" << (output_len / expected_len)
              << " final correction t="
              << pg_correction.translation().norm() << "m\n";

    // With no loops, feedback loop should preserve length to machine precision.
    EXPECT_NEAR(output_len, expected_len, 1e-3);
}

// ─── Theory H (reproduced from real-bag evidence then fixed):
//
// On hkust_campus_00 the correction stayed at 0 m for the first 26
// optimizes, then — once the trajectory first turned significantly —
// doubled every call: 0.0001 → 0.0003 → 0.0006 → ... → 153 m.
//
// Two compounding bugs were at play:
//   1. PoseGraph::GetCorrection() used to return the per-optimize delta,
//      and the caller OVERWROTE its stored correction each call, losing
//      the composition. Fixed by making GetCorrection return the
//      cumulative and composing internally on Optimize().
//   2. Even with (1) fixed, LM's Cholesky of the H+λI matrix on a
//      perfectly zero-residual odometry chain produces ~1e-15 m of
//      vertex perturbation per call from floating-point round-off.
//      Composed over 50 calls on a rotating trajectory that amplifies
//      ~2× per step → saturates at roughly the chain size. Fixed by
//      skipping Optimize() entirely when loop_edges_count_ == 0 (an
//      odometry-only chain is already at the optimum).
//
// This test: feed a rotating trajectory through the real integration
// loop with zero loop edges; assert zero optimizes fire and the final
// cumulative correction is exactly identity.
TEST(PoseGraph, NoOptimizeRunsWithoutLoopClosures) {
    const int N = 300;
    auto iekf_poses = MakeCircle(N, 20.0);  // 20 m radius, full loop

    PoseGraph::Options opts = DefaultOpts();
    opts.optimize_every_n = 10;
    PoseGraph pg;
    pg.Init(opts);

    Eigen::Isometry3d pg_correction = Eigen::Isometry3d::Identity();
    int optimizes_fired = 0;

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.block<3, 3>(0, 0) *= 0.05 * 0.05;
    cov.block<3, 3>(3, 3) *= 0.02 * 0.02;

    for (int i = 0; i < N; ++i) {
        const Eigen::Isometry3d corrected = pg_correction * iekf_poses[i];
        pg.TryAddKeyframe(corrected, static_cast<double>(i), cov);

        if (pg.ShouldOptimize()) {
            if (pg.Optimize()) {
                pg_correction = pg.GetCorrection();
                ++optimizes_fired;
            }
        }
    }

    std::cout << "[H] optimizes fired=" << optimizes_fired
              << " final correction t="
              << pg_correction.translation().norm() << "m\n";

    EXPECT_EQ(optimizes_fired, 0);
    EXPECT_LT(pg_correction.translation().norm(), 1e-12);
    EXPECT_LT(Eigen::AngleAxisd(pg_correction.rotation()).angle(), 1e-12);
}

// Sanity: once a single loop edge is added, Optimize DOES run, and the
// cumulative correction is stable (doesn't compound spuriously).
TEST(PoseGraph, OptimizeRunsWhenLoopEdgePresent) {
    const int N = 50;
    auto iekf_poses = MakeStraightLine(N, 0.5);

    PoseGraph::Options opts = DefaultOpts();
    opts.optimize_every_n = 10;
    PoseGraph pg;
    pg.Init(opts);

    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.block<3, 3>(0, 0) *= 0.05 * 0.05;
    cov.block<3, 3>(3, 3) *= 0.02 * 0.02;

    Eigen::Isometry3d pg_correction = Eigen::Isometry3d::Identity();
    for (int i = 0; i < N; ++i) {
        const Eigen::Isometry3d corrected = pg_correction * iekf_poses[i];
        pg.TryAddKeyframe(corrected, static_cast<double>(i), cov);

        // After the 20th keyframe, add a TRUE-POSITIVE loop closure:
        // the 19th keyframe revisits the 0th (unlikely geometrically,
        // but we're injecting the edge directly to test the gating).
        if (i == 20) {
            Eigen::Matrix<double, 6, 6> info =
                Eigen::Matrix<double, 6, 6>::Identity() * 1e4;
            Eigen::Isometry3d meas = Eigen::Isometry3d::Identity();
            // kf 19 at x=9.5, kf 0 at x=0 → T^19_0 = translation(-9.5,0,0)
            meas.translation() = Eigen::Vector3d(-9.5, 0, 0);
            pg.AddLoopClosure(19, 0, meas, info);
        }

        if (pg.ShouldOptimize()) {
            if (pg.Optimize()) {
                pg_correction = pg.GetCorrection();
            }
        }
    }

    std::cout << "[H2] final correction t="
              << pg_correction.translation().norm() << "m\n";
    // With a real loop edge, the correction should have a sensible value
    // (exactly zero is impossible: the loop pulls the chain) but stable.
    EXPECT_LT(pg_correction.translation().norm(), 15.0);
}

// ─── Theory G: g2o EdgeSE3 measurement convention sanity.
// g2o's EdgeSE3 computes error = log(M^-1 * T_from^-1 * T_to), so the
// zero-error measurement M on edge (from→to) is T^from_to = T_from^-1 * T_to.
//
// loop_closer.cc:120-124 builds `m.relative_pose` from
// `icp.getFinalTransformation()`, which maps SOURCE (latest) points into
// TARGET (older) frame — i.e. T^target_source. Then AddLoopClosure is
// called with from=latest, to=older, measurement=T^target_source. But g2o
// wants T^from_to = T^latest_older = (T^target_source)^-1.
//
// This test builds a TRUE-POSITIVE revisit (latest really IS at kf 0) and
// adds the loop edge with BOTH directions. Correct direction → chain
// stays at its correct length. Inverted direction → chain collapses.
TEST(PoseGraph, LoopEdgeDirectionConventionMatters) {
    // 10 kf in a line, then return to origin. Total path 18 m, but
    // endpoint = origin, so a true-positive loop closure has identity
    // relative measurement.
    std::vector<Eigen::Isometry3d> poses;
    for (int i = 0; i <= 9; ++i) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(i * 1.0, 0, 0);
        poses.push_back(T);
    }
    for (int i = 8; i >= 0; --i) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(i * 1.0, 0, 0);
        poses.push_back(T);
    }
    // poses.size() == 19, endpoint = origin

    auto run = [&](const Eigen::Isometry3d& loop_measurement, const char* tag) {
        PoseGraph pg;
        pg.Init(DefaultOpts());
        Eigen::Matrix<double, 6, 6> cov =
            Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
        for (size_t i = 0; i < poses.size(); ++i) {
            pg.TryAddKeyframe(poses[i], static_cast<double>(i), cov);
        }
        // True-positive loop: latest kf (id=18) coincides with kf 0.
        Eigen::Matrix<double, 6, 6> info =
            Eigen::Matrix<double, 6, 6>::Identity() * 1e4;
        pg.AddLoopClosure(static_cast<int>(poses.size()) - 1, 0, loop_measurement, info);
        pg.Optimize();
        const double len = ChainLength(pg.GetKeyframes());
        std::cout << "[G:" << tag << "] chain length=" << len << "\n";
        return len;
    };

    // CORRECT: for a true-positive loop at same pose, M = T_from^-1 * T_to =
    // Identity^-1 * Identity = Identity. Chain length should stay ~18 m.
    const double len_correct = run(Eigen::Isometry3d::Identity(), "correct");

    // INVERTED direction: in this case since both poses are identity, the
    // inverse is also identity — so sign inversion is invisible here. Let's
    // instead test a case where from and to are NOT at the same pose:
    // simulate a loop where latest_pose=(9,0,0) revisits kf_5=(5,0,0). The
    // correct measurement is T^latest_to = T_latest^-1 * T_to = translation
    // of (-4,0,0). The INVERTED measurement (what ICP hands us) would be
    // (4,0,0).
    PoseGraph pg_c;
    pg_c.Init(DefaultOpts());
    Eigen::Matrix<double, 6, 6> cov =
        Eigen::Matrix<double, 6, 6>::Identity() * 1e-2;
    for (int i = 0; i <= 9; ++i) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(i * 1.0, 0, 0);
        pg_c.TryAddKeyframe(T, static_cast<double>(i), cov);
    }
    const double len_before = ChainLength(pg_c.GetKeyframes());

    Eigen::Matrix<double, 6, 6> info =
        Eigen::Matrix<double, 6, 6>::Identity() * 1e4;
    // Correct: measurement T^from_to where from=9 at (9,0,0), to=5 at (5,0,0).
    // T^from_to = translation(-4,0,0).
    Eigen::Isometry3d correct_meas = Eigen::Isometry3d::Identity();
    correct_meas.translation() = Eigen::Vector3d(-4.0, 0, 0);
    pg_c.AddLoopClosure(9, 5, correct_meas, info);
    pg_c.Optimize();
    const double len_after_correct = ChainLength(pg_c.GetKeyframes());
    std::cout << "[G:dir-correct] before=" << len_before
              << " after=" << len_after_correct
              << " (measurement=+(-4,0,0), zero residual)\n";

    // Inverted direction: same setup but measurement = +(4,0,0). Now LM
    // must choose: keep chain rigid (high odom info fights), or squash to
    // satisfy loop (high loop info wins).
    PoseGraph pg_i;
    pg_i.Init(DefaultOpts());
    for (int i = 0; i <= 9; ++i) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() = Eigen::Vector3d(i * 1.0, 0, 0);
        pg_i.TryAddKeyframe(T, static_cast<double>(i), cov);
    }
    Eigen::Isometry3d inverted_meas = Eigen::Isometry3d::Identity();
    inverted_meas.translation() = Eigen::Vector3d(+4.0, 0, 0);
    pg_i.AddLoopClosure(9, 5, inverted_meas, info);
    pg_i.Optimize();
    const double len_after_inv = ChainLength(pg_i.GetKeyframes());
    const double correction_inv = pg_i.GetCorrection().translation().norm();
    std::cout << "[G:dir-inverted] before=" << len_before
              << " after=" << len_after_inv
              << " correction=" << correction_inv
              << "m (measurement=(+4,0,0), SHOULD show damage)\n";

    // Expected: correct direction leaves chain alone, inverted direction
    // pushes the latest vertex by nearly the chain length — the "shrink"
    // the user sees is the same phenomenon cascaded over 50+ bad loops.
    EXPECT_NEAR(len_after_correct, 9.0, 1e-2);
    EXPECT_GT(correction_inv, 5.0);
    (void)len_correct;
}

// ─── Theory D: correction composition when optimize is a no-op.
// With identity covariance + no loops, optimize should not change any
// vertex. The returned correction should be exactly identity. If the
// correction drifts here, subsequent calls to MaybeUpdatePoseGraph will
// accumulate spurious jumps even when nothing in the world has changed.
TEST(PoseGraph, NoOpOptimizeReturnsIdentityCorrection) {
    auto poses = MakeCircle(40, 10.0);

    PoseGraph pg;
    pg.Init(DefaultOpts());

    Eigen::Matrix<double, 6, 6> I6 = Eigen::Matrix<double, 6, 6>::Identity();
    for (size_t i = 0; i < poses.size(); ++i) {
        pg.TryAddKeyframe(poses[i], static_cast<double>(i), I6);
    }
    ASSERT_TRUE(pg.Optimize());

    const Eigen::Isometry3d corr = pg.GetCorrection();
    const double t_norm = corr.translation().norm();
    const double r_angle = Eigen::AngleAxisd(corr.rotation()).angle();
    std::cout << "[D] correction t=" << t_norm << "m r=" << r_angle << "rad\n";

    EXPECT_LT(t_norm, 1e-6);
    EXPECT_LT(std::abs(r_angle), 1e-6);
}
