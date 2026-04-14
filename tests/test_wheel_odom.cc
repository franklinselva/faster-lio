// Unit tests for wheel-odometry fusion.
//
// Two layers:
//   [A] Pure-math tests against `wheel_fusion::ApplyScalarBodyVelUpdate`
//       (header-only, no LaserMapping required). These validate the
//       Jacobian, residual, gain, and cov update in isolation.
//   [B] Integration tests via `LaserMapping::AddWheelOdom` + yaml config
//       that exercise the full plumbing (queue → drain → obs select →
//       scalar update). Kept lightweight — no LiDAR scans processed.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <random>

#include "faster_lio/laser_mapping.h"
#include "faster_lio/types.h"
#include "faster_lio/wheel_fusion.h"

using namespace faster_lio;
using wheel_fusion::kStateDof;
using wheel_fusion::StateCov;
using wheel_fusion::ApplyScalarBodyVelUpdate;

namespace {

// Build an identity state at rest with modest initial covariance.
state_ikfom MakeIdentityState(double vx = 0.0, double vy = 0.0, double vz = 0.0) {
    state_ikfom s;
    // `vect3` is MTK::vect<3, double>; assignment from Eigen::Vector3d is
    // supported but the 3-arg constructor is not, so we assign component-wise.
    s.pos = Eigen::Vector3d(0, 0, 0);
    s.rot = SO3();  // identity quaternion
    s.offset_R_L_I = SO3();
    s.offset_T_L_I = Eigen::Vector3d(0, 0, 0);
    s.vel = Eigen::Vector3d(vx, vy, vz);
    s.bg = Eigen::Vector3d(0, 0, 0);
    s.ba = Eigen::Vector3d(0, 0, 0);
    // grav: default-constructed S2
    return s;
}

StateCov IsotropicCov(double sigma2) {
    return StateCov::Identity() * sigma2;
}

// Covariance that is isotropic in `vel` only and zero (certain) elsewhere.
// Use this when the test asserts world-frame velocity convergence: with
// non-zero rot uncertainty the KF would legitimately satisfy a body-frame
// observation by rotating the body, which is mathematically correct but
// muddles a pure-scalar-convergence assertion.
StateCov VelOnlyCov(double sigma2) {
    StateCov P = StateCov::Zero();
    P.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity() * sigma2;
    return P;
}

// Rotation about Z by angle_rad
SO3 RotZ(double angle_rad) {
    Eigen::AngleAxisd aa(angle_rad, Eigen::Vector3d::UnitZ());
    return SO3(Eigen::Quaterniond(aa));
}

// Numerical Jacobian of h(x) = (R_wb^T · v)[axis] w.r.t. tangent-space δθ
// (rot) and v_world. Used to validate the analytical Jacobian.
double NumericalDhDRot(const state_ikfom &x, int axis, int rot_component, double eps = 1e-6) {
    // Perturb rot by exp(eps * e_i) via right-multiplication (MTK convention).
    Eigen::Vector3d dth = Eigen::Vector3d::Zero();
    dth[rot_component] = eps;
    state_ikfom xp = x;
    Eigen::Matrix<double, kStateDof, 1> d_plus = Eigen::Matrix<double, kStateDof, 1>::Zero();
    d_plus.segment<3>(3) = dth;
    xp.boxplus(d_plus);
    state_ikfom xm = x;
    Eigen::Matrix<double, kStateDof, 1> d_minus = Eigen::Matrix<double, kStateDof, 1>::Zero();
    d_minus.segment<3>(3) = -dth;
    xm.boxplus(d_minus);
    const double hp = (xp.rot.toRotationMatrix().transpose() *
                       Eigen::Vector3d(xp.vel[0], xp.vel[1], xp.vel[2]))[axis];
    const double hm = (xm.rot.toRotationMatrix().transpose() *
                       Eigen::Vector3d(xm.vel[0], xm.vel[1], xm.vel[2]))[axis];
    return (hp - hm) / (2 * eps);
}

double NumericalDhDVel(const state_ikfom &x, int axis, int vel_component, double eps = 1e-6) {
    state_ikfom xp = x;
    xp.vel[vel_component] += eps;
    state_ikfom xm = x;
    xm.vel[vel_component] -= eps;
    const double hp = (xp.rot.toRotationMatrix().transpose() *
                       Eigen::Vector3d(xp.vel[0], xp.vel[1], xp.vel[2]))[axis];
    const double hm = (xm.rot.toRotationMatrix().transpose() *
                       Eigen::Vector3d(xm.vel[0], xm.vel[1], xm.vel[2]))[axis];
    return (hp - hm) / (2 * eps);
}

// Write a minimal yaml config to a tmp path, optionally with a `wheel:`
// block. Returns the path.
std::string WriteTmpYaml(const std::string &suffix, bool wheel_enabled,
                        bool nhc_v_y = true, bool nhc_v_z = true) {
    std::string path = std::string(ROOT_DIR) + "build/test_wheel_" + suffix + ".yaml";
    std::ofstream f(path);
    f << "common:\n  time_sync_en: false\n";
    f << "preprocess:\n  blind: 0.5\n";
    f << "mapping:\n"
      << "  acc_cov: 0.1\n  gyr_cov: 0.1\n  b_acc_cov: 0.0001\n  b_gyr_cov: 0.0001\n"
      << "  det_range: 100.0\n  extrinsic_est_en: false\n"
      << "  extrinsic_T: [0, 0, 0]\n"
      << "  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
    f << "point_filter_num: 1\n";
    f << "max_iteration: 3\n";
    f << "filter_size_surf: 0.5\n  \nfilter_size_map: 0.5\n";
    f << "cube_side_length: 1000\n";
    f << "ivox_grid_resolution: 0.5\n";
    f << "ivox_nearby_type: 18\n";
    f << "esti_plane_threshold: 0.1\n";
    f << "map_quality_threshold: 0.0\n";
    f << "output:\n  path_en: false\n  dense_en: false\n  path_save_en: false\n";
    f << "pcd_save:\n  pcd_save_en: false\n  interval: -1\n";
    if (wheel_enabled) {
        f << "wheel:\n"
          << "  enabled: true\n"
          << "  cov_v_x: 0.01\n  cov_v_y: 0.01\n  cov_v_z: 0.001\n"
          << "  cov_omega_z: 0.01\n"
          << "  emit_nhc_v_y: " << (nhc_v_y ? "true" : "false") << "\n"
          << "  emit_nhc_v_z: " << (nhc_v_z ? "true" : "false") << "\n"
          << "  nhc_cov: 0.001\n"
          << "  max_time_gap: 0.05\n";
    }
    return path;
}

}  // namespace

// ──────────────────────────────────────────────────────────────────────
// [A] Pure-math tests on wheel_fusion::ApplyScalarBodyVelUpdate
// ──────────────────────────────────────────────────────────────────────

TEST(WheelOdomDataTest, DefaultConstructionIsEmpty) {
    WheelOdomData w;
    EXPECT_DOUBLE_EQ(w.timestamp, 0.0);
    EXPECT_DOUBLE_EQ(w.v_body_x, 0.0);
    EXPECT_FALSE(w.v_body_y.has_value());
    EXPECT_FALSE(w.v_body_z.has_value());
    EXPECT_FALSE(w.omega_z.has_value());
}

TEST(WheelOdomDataTest, OptionalFieldsAssignable) {
    WheelOdomData w;
    w.v_body_y = 0.5;
    w.v_body_z = -0.2;
    w.omega_z = 0.1;
    EXPECT_TRUE(w.v_body_y.has_value());
    EXPECT_DOUBLE_EQ(*w.v_body_y, 0.5);
    EXPECT_DOUBLE_EQ(*w.v_body_z, -0.2);
    EXPECT_DOUBLE_EQ(*w.omega_z, 0.1);
}

TEST(ScalarUpdateMath, RejectsInvalidAxis) {
    state_ikfom x = MakeIdentityState(1.0, 0.0, 0.0);
    StateCov P = IsotropicCov(0.1);
    auto rep = ApplyScalarBodyVelUpdate(x, P, /*axis=*/-1, /*z=*/1.0, /*R=*/0.01);
    EXPECT_FALSE(rep.updated);
    EXPECT_EQ(rep.status, wheel_fusion::ScalarUpdateReport::Status::InvalidAxis);
    rep = ApplyScalarBodyVelUpdate(x, P, /*axis=*/3, /*z=*/1.0, /*R=*/0.01);
    EXPECT_FALSE(rep.updated);
    EXPECT_EQ(rep.status, wheel_fusion::ScalarUpdateReport::Status::InvalidAxis);
}

TEST(ScalarUpdateMath, RejectsNonPositiveR) {
    state_ikfom x = MakeIdentityState();
    StateCov P = IsotropicCov(0.1);
    auto rep = ApplyScalarBodyVelUpdate(x, P, 0, 1.0, /*R=*/0.0);
    EXPECT_FALSE(rep.updated);
    EXPECT_EQ(rep.status, wheel_fusion::ScalarUpdateReport::Status::NonPositiveR);
    rep = ApplyScalarBodyVelUpdate(x, P, 0, 1.0, /*R=*/-1.0);
    EXPECT_FALSE(rep.updated);
    EXPECT_EQ(rep.status, wheel_fusion::ScalarUpdateReport::Status::NonPositiveR);
}

TEST(ScalarUpdateMath, PerfectObsZeroResidualNoStateChange) {
    state_ikfom x = MakeIdentityState(1.23, 0.45, -0.10);
    StateCov P = IsotropicCov(0.1);
    for (int axis = 0; axis < 3; ++axis) {
        state_ikfom x_copy = x;
        StateCov P_copy = P;
        const Eigen::Vector3d v_body =
            x.rot.toRotationMatrix().transpose() * Eigen::Vector3d(x.vel[0], x.vel[1], x.vel[2]);
        const double z = v_body[axis];
        const auto rep = ApplyScalarBodyVelUpdate(x_copy, P_copy, axis, z, 0.01);
        EXPECT_TRUE(rep.updated);
        EXPECT_NEAR(rep.residual, 0.0, 1e-12);
        EXPECT_NEAR(x_copy.vel[0], x.vel[0], 1e-12);
        EXPECT_NEAR(x_copy.vel[1], x.vel[1], 1e-12);
        EXPECT_NEAR(x_copy.vel[2], x.vel[2], 1e-12);
    }
}

TEST(ScalarUpdateMath, IdentityRotVxPullsVelTowardMeasurement) {
    state_ikfom x = MakeIdentityState(0.0, 0.0, 0.0);
    StateCov P = IsotropicCov(0.1);
    const double z = 1.0;
    const double R = 0.01;
    const auto rep = ApplyScalarBodyVelUpdate(x, P, /*axis=*/0, z, R);
    ASSERT_TRUE(rep.updated);
    // With P=0.1 I, R=0.01, gain ≈ 0.1/(0.1+0.01) ≈ 0.909 → vx ≈ 0.909
    EXPECT_GT(x.vel[0], 0.85);
    EXPECT_LT(x.vel[0], 0.95);
    // Orthogonal components should not move (identity rot, identity P).
    EXPECT_NEAR(x.vel[1], 0.0, 1e-9);
    EXPECT_NEAR(x.vel[2], 0.0, 1e-9);
}

TEST(ScalarUpdateMath, YawedRotRoutesBodyObsToWorldAxis) {
    // Heading 90° about z: body-x points along +Y in world.
    state_ikfom x = MakeIdentityState(0.0, 0.0, 0.0);
    x.rot = RotZ(M_PI / 2);
    StateCov P = IsotropicCov(0.1);
    // Measure body v_x = 1.0 → expect world vel ≈ (0, +1, 0).
    ApplyScalarBodyVelUpdate(x, P, /*axis=*/0, 1.0, 0.01);
    EXPECT_NEAR(x.vel[0], 0.0, 0.05);
    EXPECT_GT(x.vel[1], 0.85);
    EXPECT_LT(x.vel[1], 0.95);
    EXPECT_NEAR(x.vel[2], 0.0, 0.05);
}

TEST(ScalarUpdateMath, RepeatedObsConvergesToMeasurement) {
    // Pin rot uncertainty to 0 so the update goes entirely to vel. With an
    // isotropic P the KF would *also* rotate the body (which is the right
    // manifold-aware behavior for body-frame obs but makes a world-frame
    // convergence assertion ambiguous).
    state_ikfom x = MakeIdentityState(0.0, 0.0, 0.0);
    StateCov P = VelOnlyCov(0.1);
    const double z = 2.5;
    for (int i = 0; i < 500; ++i) {
        ApplyScalarBodyVelUpdate(x, P, 0, z, 0.01,
                                 /*mahalanobis_gate_sq=*/1e9);
    }
    EXPECT_NEAR(x.vel[0], z, 1e-3);
    EXPECT_LT(P(12, 12), 1e-3);
}

TEST(ScalarUpdateMath, CovarianceRemainsSymmetricAndPSD) {
    state_ikfom x = MakeIdentityState(0.5, -0.3, 0.1);
    x.rot = RotZ(0.3);
    StateCov P = IsotropicCov(0.5);
    for (int i = 0; i < 20; ++i) {
        ApplyScalarBodyVelUpdate(x, P, i % 3, 0.2 * i, 0.02);
    }
    // Symmetry check.
    const double asym = (P - P.transpose()).cwiseAbs().maxCoeff();
    EXPECT_LT(asym, 1e-10);
    // PSD check — smallest eigenvalue ≥ 0 (allow tiny tolerance).
    Eigen::SelfAdjointEigenSolver<StateCov> es(P);
    EXPECT_GT(es.eigenvalues().minCoeff(), -1e-9);
}

TEST(ScalarUpdateMath, AnalyticalJacobianMatchesNumerical) {
    // Seed a state with arbitrary rotation and nonzero velocity.
    state_ikfom x = MakeIdentityState(0.7, -0.4, 0.15);
    x.rot = RotZ(0.42);
    // Apply a tiny pitch perturbation for a generic rotation.
    Eigen::Matrix<double, kStateDof, 1> d = Eigen::Matrix<double, kStateDof, 1>::Zero();
    d(4) = 0.15;  // small rotation about y (tangent index 4)
    x.boxplus(d);

    const Eigen::Matrix3d R_wb = x.rot.toRotationMatrix();
    const Eigen::Vector3d v_world(x.vel[0], x.vel[1], x.vel[2]);
    const Eigen::Vector3d v_body = R_wb.transpose() * v_world;

    for (int axis = 0; axis < 3; ++axis) {
        Eigen::Matrix3d v_body_skew;
        v_body_skew <<         0.0, -v_body.z(),  v_body.y(),
                       v_body.z(),         0.0, -v_body.x(),
                      -v_body.y(),  v_body.x(),         0.0;
        // Rot block (tangent indices 3..5)
        for (int c = 0; c < 3; ++c) {
            const double H_analytic = v_body_skew(axis, c);
            const double H_numeric  = NumericalDhDRot(x, axis, c);
            EXPECT_NEAR(H_analytic, H_numeric, 1e-6)
                << "axis=" << axis << " rot_component=" << c;
        }
        // Vel block (tangent indices 12..14)
        for (int c = 0; c < 3; ++c) {
            const double H_analytic = R_wb.transpose()(axis, c);
            const double H_numeric  = NumericalDhDVel(x, axis, c);
            EXPECT_NEAR(H_analytic, H_numeric, 1e-6)
                << "axis=" << axis << " vel_component=" << c;
        }
    }
}

TEST(ScalarUpdateMath, MahalanobisGateRejectsFarOutliers) {
    state_ikfom x = MakeIdentityState(0.0, 0.0, 0.0);
    StateCov P = IsotropicCov(1e-6);  // tiny uncertainty
    // z=10 vs pred=0, R=1e-6 → mahal² ≈ 10²/(1e-6 + 1e-6) = enormous
    auto rep = ApplyScalarBodyVelUpdate(x, P, 0, 10.0, 1e-6);
    EXPECT_FALSE(rep.updated);
    EXPECT_EQ(rep.status, wheel_fusion::ScalarUpdateReport::Status::GatedByMahalanobis);
    EXPECT_NEAR(x.vel[0], 0.0, 1e-12);
}

TEST(ScalarUpdateMath, NhcVirtualObsPinsLateralVelocityToZero) {
    // Simulate a filter that thinks v_y = 0.3 but NHC says v_y = 0. Use
    // VelOnlyCov so the correction lands on vel alone (rot is "certain").
    state_ikfom x = MakeIdentityState(1.0, 0.3, 0.0);
    StateCov P = VelOnlyCov(0.1);
    for (int i = 0; i < 100; ++i) {
        ApplyScalarBodyVelUpdate(x, P, 1, 0.0, 0.001);
    }
    EXPECT_NEAR(x.vel[1], 0.0, 1e-3);
    EXPECT_NEAR(x.vel[0], 1.0, 1e-3);
    EXPECT_NEAR(x.vel[2], 0.0, 1e-3);
}

TEST(ScalarUpdateMath, NhcWithRotUncertaintyRotatesBodyInsteadOfJustVel) {
    // When rot IS uncertain, the KF legitimately satisfies body-frame NHC
    // by rotating the body. The final *body-frame* lateral velocity must
    // be ~0, though the world-frame lateral velocity may not be.
    state_ikfom x = MakeIdentityState(1.0, 0.3, 0.0);
    StateCov P = IsotropicCov(0.1);
    for (int i = 0; i < 200; ++i) {
        ApplyScalarBodyVelUpdate(x, P, 1, 0.0, 0.001);
    }
    const Eigen::Vector3d v_body =
        x.rot.toRotationMatrix().transpose() * Eigen::Vector3d(x.vel[0], x.vel[1], x.vel[2]);
    EXPECT_NEAR(v_body.y(), 0.0, 1e-3);
}

TEST(ScalarUpdateMath, HolonomicFullVelObsConvergesAllAxes) {
    state_ikfom x = MakeIdentityState(0.0, 0.0, 0.0);
    StateCov P = VelOnlyCov(0.1);
    const Eigen::Vector3d z_body(1.2, -0.5, 0.0);
    for (int i = 0; i < 200; ++i) {
        ApplyScalarBodyVelUpdate(x, P, 0, z_body[0], 0.01,
                                 /*mahalanobis_gate_sq=*/1e9);
        ApplyScalarBodyVelUpdate(x, P, 1, z_body[1], 0.01,
                                 /*mahalanobis_gate_sq=*/1e9);
        ApplyScalarBodyVelUpdate(x, P, 2, z_body[2], 0.001);
    }
    EXPECT_NEAR(x.vel[0], z_body[0], 1e-3);
    EXPECT_NEAR(x.vel[1], z_body[1], 1e-3);
    EXPECT_NEAR(x.vel[2], z_body[2], 1e-3);
}

TEST(ScalarUpdateMath, NanInputIsRejected) {
    state_ikfom x = MakeIdentityState();
    StateCov P = IsotropicCov(0.1);
    const auto rep = ApplyScalarBodyVelUpdate(
        x, P, 0, std::numeric_limits<double>::quiet_NaN(), 0.01);
    // NaN propagates into residual → mahal² NaN → gate rejects (gate_sq>NaN is false).
    EXPECT_FALSE(rep.updated);
}

// ──────────────────────────────────────────────────────────────────────
// [B] Integration tests via LaserMapping::AddWheelOdom + yaml config
// ──────────────────────────────────────────────────────────────────────

class WheelOdomIntegrationTest : public ::testing::Test {
   protected:
    void SetUp() override {
        mapping_ = std::make_shared<LaserMapping>();
    }
    std::shared_ptr<LaserMapping> mapping_;
};

TEST_F(WheelOdomIntegrationTest, InitWithoutWheelBlockSucceeds) {
    const std::string cfg = WriteTmpYaml("noblock", /*enabled=*/false);
    EXPECT_TRUE(mapping_->Init(cfg));
}

TEST_F(WheelOdomIntegrationTest, InitWithWheelEnabledSucceeds) {
    const std::string cfg = WriteTmpYaml("enabled", /*enabled=*/true);
    EXPECT_TRUE(mapping_->Init(cfg));
}

TEST_F(WheelOdomIntegrationTest, AddWheelOdomAcceptsValidTimestamp) {
    const std::string cfg = WriteTmpYaml("accept", /*enabled=*/true);
    ASSERT_TRUE(mapping_->Init(cfg));
    WheelOdomData w;
    w.timestamp = 1.0;
    w.v_body_x = 0.5;
    mapping_->AddWheelOdom(w);
    SUCCEED();  // no crash on valid input
}

TEST_F(WheelOdomIntegrationTest, AddWheelOdomRejectsInvalidTimestamp) {
    const std::string cfg = WriteTmpYaml("invalid_ts", /*enabled=*/true);
    ASSERT_TRUE(mapping_->Init(cfg));
    WheelOdomData w;
    w.timestamp = 0.0;  // invalid
    w.v_body_x = 0.5;
    mapping_->AddWheelOdom(w);
    // Doesn't crash, just logs a warning and drops.
    SUCCEED();
}

TEST_F(WheelOdomIntegrationTest, AddWheelOdomRejectsBackwardsTimestamp) {
    const std::string cfg = WriteTmpYaml("backwards", /*enabled=*/true);
    ASSERT_TRUE(mapping_->Init(cfg));
    WheelOdomData w1; w1.timestamp = 2.0; w1.v_body_x = 1.0;
    mapping_->AddWheelOdom(w1);
    WheelOdomData w2; w2.timestamp = 1.0; w2.v_body_x = 0.5;  // goes backwards
    mapping_->AddWheelOdom(w2);
    // Doesn't crash.
    SUCCEED();
}

TEST_F(WheelOdomIntegrationTest, RunWithoutWheelDataDoesNotCrash) {
    const std::string cfg = WriteTmpYaml("no_data", /*enabled=*/true);
    ASSERT_TRUE(mapping_->Init(cfg));
    // Just Run() a few times without any data — should be no-op.
    mapping_->Run();
    mapping_->Run();
    SUCCEED();
}

TEST_F(WheelOdomIntegrationTest, MassiveWheelQueuePushOverflowsAndDrops) {
    const std::string cfg = WriteTmpYaml("overflow", /*enabled=*/true);
    ASSERT_TRUE(mapping_->Init(cfg));
    // Push way more than the queue capacity (512) — should not crash, just warn.
    for (int i = 0; i < 10000; ++i) {
        WheelOdomData w;
        w.timestamp = 1.0 + i * 0.001;
        w.v_body_x = 0.0;
        mapping_->AddWheelOdom(w);
    }
    SUCCEED();
}
