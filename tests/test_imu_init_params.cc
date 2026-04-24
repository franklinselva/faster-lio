// ─────────────────────────────────────────────────────────────────────────
// test_imu_init_params.cc
//
// Tests for the two faster-lio changes that expose IMU process-noise
// plumbing and the initial Kalman-covariance diagonal:
//
//   A. `imu_init.init_P_diag` YAML block (struct `InitPDiag`) is honoured
//      by ImuProcess::IMUInit and byte-identical to the upstream FAST-LIO
//      hard-coded init_P when no overrides are supplied.
//
//   B. Process-noise Q_ is owned by a single writer (RebuildQ) driven by
//      cov_gyr_ / cov_acc_ / cov_bias_gyr_ / cov_bias_acc_. The previous
//      per-frame rebuild inside UndistortPcl is gone, and the dead
//      `cov_acc_ *= pow(G/|mean_acc|, 2)` rescale is gone. Tests pin the
//      new invariant: after ctor, after bias-cov setters, and at end-of-
//      init, Q_ reflects the right values and doesn't mutate during
//      UndistortPcl.
//
// Every negative case the caller can get from YAML is exercised:
// negative / zero / NaN / Inf values must be clamped rather than abort
// the pipeline.
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "faster_lio/imu_processing.h"
#include "faster_lio/laser_mapping.h"

using namespace faster_lio;

namespace {

// Reference values that must match the upstream FAST-LIO hard-coded init_P.
// If any of these change, the InitPDiag struct defaults are out of sync.
constexpr double kLegacyPos       = 1.0;
constexpr double kLegacyRot       = 1.0;
constexpr double kLegacyOffRLI    = 1.0e-5;
constexpr double kLegacyOffTLI    = 1.0e-5;
constexpr double kLegacyVel       = 1.0;
constexpr double kLegacyBg        = 1.0e-4;
constexpr double kLegacyBa        = 1.0e-3;
constexpr double kLegacyGrav      = 1.0e-5;

// Legacy cov_* defaults set in ImuProcess ctor — these drive the Q_
// diagonal before any *_scale_ or YAML wiring takes effect.
constexpr double kCtorCovGyr      = 0.1;
constexpr double kCtorCovAcc      = 0.1;
constexpr double kCtorCovBiasGyr  = 1.0e-4;
constexpr double kCtorCovBiasAcc  = 1.0e-4;

// A ten-point synthetic lidar batch is the minimum SyncPackages will accept.
PointCloudType::Ptr MakeTinyCloud(double scan_len_s = 0.1) {
    auto cloud = PointCloudType::Ptr(new PointCloudType());
    for (int i = 0; i < 10; ++i) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * static_cast<float>(scan_len_s * 1000.0 / 10.0);
        cloud->push_back(p);
    }
    return cloud;
}

// Minimum batch that walks ImuProcess::Process once — enough samples to
// trigger full init on the legacy unconditional MAX_INI_COUNT path.
common::MeasureGroup MakeStaticBatch(int n_imu = MAX_INI_COUNT + 5,
                                     double dt = 0.005,
                                     double t0 = 0.0,
                                     const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, common::G_m_s2),
                                     const Eigen::Vector3d &gyro  = Eigen::Vector3d::Zero()) {
    common::MeasureGroup meas;
    meas.lidar_ = MakeTinyCloud();
    meas.lidar_bag_time_ = t0;
    meas.lidar_end_time_ = t0 + 0.1;
    for (int i = 0; i < n_imu; ++i) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = t0 + i * dt;
        imu->linear_acceleration = accel;
        imu->angular_velocity = gyro;
        meas.imu_.push_back(imu);
    }
    return meas;
}

// Lightweight fixture that mirrors test_imu_processing_edge_cases: fresh
// ImuProcess + esekf for each test. The bias-cov setters are called with
// the same canonical values as existing tests so the Q_-invariants below
// compare against a well-known baseline.
struct ImuFixture {
    std::shared_ptr<ImuProcess> imu = std::make_shared<ImuProcess>();
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    PointCloudType::Ptr pcl_out{new PointCloudType()};

    ImuFixture() {
        imu->SetExtrinsic(common::Zero3d, common::Eye3d);
        imu->SetGyrCov(common::V3D(kCtorCovGyr, kCtorCovGyr, kCtorCovGyr));
        imu->SetAccCov(common::V3D(kCtorCovAcc, kCtorCovAcc, kCtorCovAcc));
        imu->SetGyrBiasCov(common::V3D(kCtorCovBiasGyr, kCtorCovBiasGyr, kCtorCovBiasGyr));
        imu->SetAccBiasCov(common::V3D(kCtorCovBiasAcc, kCtorCovBiasAcc, kCtorCovBiasAcc));

        std::vector<double> epsi(23, 0.001);
        kf.init_dyn_share(
            get_f, df_dx, df_dw,
            [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {},
            4, epsi.data());
    }

    // Drives Process() with enough samples to complete init, then returns
    // the post-init P (UndistortPcl is NOT called on the init-complete
    // return path, so P is exactly what IMUInit wrote).
    Eigen::Matrix<double, 23, 23> FinishInitAndGetP() {
        auto meas = MakeStaticBatch();
        imu->Process(meas, kf, pcl_out);
        return kf.get_P();
    }
};

// Check the 23 diagonal elements of P against the 8 state blocks.
void ExpectInitPMatches(const Eigen::Matrix<double, 23, 23> &P, const InitPDiag &expected,
                       double tol = 1e-20) {
    for (int i = 0;  i < 3;  ++i) EXPECT_NEAR(P(i, i), expected.pos,       tol) << "pos[" << i << "]";
    for (int i = 3;  i < 6;  ++i) EXPECT_NEAR(P(i, i), expected.rot,       tol) << "rot[" << i << "]";
    for (int i = 6;  i < 9;  ++i) EXPECT_NEAR(P(i, i), expected.off_R_L_I, tol) << "off_R_L_I[" << i << "]";
    for (int i = 9;  i < 12; ++i) EXPECT_NEAR(P(i, i), expected.off_T_L_I, tol) << "off_T_L_I[" << i << "]";
    for (int i = 12; i < 15; ++i) EXPECT_NEAR(P(i, i), expected.vel,       tol) << "vel[" << i << "]";
    for (int i = 15; i < 18; ++i) EXPECT_NEAR(P(i, i), expected.bg,        tol) << "bg[" << i << "]";
    for (int i = 18; i < 21; ++i) EXPECT_NEAR(P(i, i), expected.ba,        tol) << "ba[" << i << "]";
    for (int i = 21; i < 23; ++i) EXPECT_NEAR(P(i, i), expected.grav,      tol) << "grav[" << i << "]";
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// A. InitPDiag — struct defaults + per-field override
// ═════════════════════════════════════════════════════════════════════════

// Baseline: with no setter called, init_P must match the upstream FAST-LIO
// hard-coded values exactly. This is the byte-identical regression guard.
TEST(InitPDiag, DefaultsAreLegacyByteIdentical) {
    ImuFixture fx;
    const auto P = fx.FinishInitAndGetP();

    InitPDiag legacy;
    EXPECT_DOUBLE_EQ(legacy.pos,       kLegacyPos);
    EXPECT_DOUBLE_EQ(legacy.rot,       kLegacyRot);
    EXPECT_DOUBLE_EQ(legacy.off_R_L_I, kLegacyOffRLI);
    EXPECT_DOUBLE_EQ(legacy.off_T_L_I, kLegacyOffTLI);
    EXPECT_DOUBLE_EQ(legacy.vel,       kLegacyVel);
    EXPECT_DOUBLE_EQ(legacy.bg,        kLegacyBg);
    EXPECT_DOUBLE_EQ(legacy.ba,        kLegacyBa);
    EXPECT_DOUBLE_EQ(legacy.grav,      kLegacyGrav);

    ExpectInitPMatches(P, legacy);
}

// Per-field override battery. Each sub-test flips ONE field to an
// unambiguous marker value (42.0) and verifies the other seven fields stay
// at their legacy defaults. Catches off-by-one bugs in the index ranges of
// the IMUInit loop (e.g. grav overwriting ba).
class InitPDiagPerField
    : public ::testing::TestWithParam<std::pair<std::string, std::function<void(InitPDiag &)>>> {};

TEST_P(InitPDiagPerField, OverrideTouchesOnlyOneBlock) {
    ImuFixture fx;

    InitPDiag custom;
    GetParam().second(custom);  // set exactly one field
    fx.imu->SetInitPDiag(custom);

    const auto P = fx.FinishInitAndGetP();

    // The legacy-field tolerance is tight (1e-20) because P is diagonal
    // and no numerical mixing is happening at IMUInit time.
    ExpectInitPMatches(P, custom);
}

INSTANTIATE_TEST_SUITE_P(
    AllEightFields, InitPDiagPerField,
    ::testing::Values(
        std::make_pair("pos",       [](InitPDiag &i){ i.pos       = 42.0; }),
        std::make_pair("rot",       [](InitPDiag &i){ i.rot       = 42.0; }),
        std::make_pair("off_R_L_I", [](InitPDiag &i){ i.off_R_L_I = 42.0; }),
        std::make_pair("off_T_L_I", [](InitPDiag &i){ i.off_T_L_I = 42.0; }),
        std::make_pair("vel",       [](InitPDiag &i){ i.vel       = 42.0; }),
        std::make_pair("bg",        [](InitPDiag &i){ i.bg        = 42.0; }),
        std::make_pair("ba",        [](InitPDiag &i){ i.ba        = 42.0; }),
        std::make_pair("grav",      [](InitPDiag &i){ i.grav      = 42.0; })),
    [](const ::testing::TestParamInfo<InitPDiagPerField::ParamType> &info) { return info.param.first; });

// Sanitize: non-finite or non-positive values must be clamped to the floor
// so the filter never runs with a zero / negative / NaN diagonal entry.
TEST(InitPDiag, SanitizeClampsNonPositive) {
    ImuFixture fx;

    InitPDiag bad;
    bad.pos  = -1.0;
    bad.rot  = 0.0;
    bad.ba   = std::numeric_limits<double>::quiet_NaN();
    bad.grav = std::numeric_limits<double>::infinity();
    fx.imu->SetInitPDiag(bad);

    const auto stored = fx.imu->GetInitPDiag();
    // Floor value matches imu_processing.cc's kFloor.
    constexpr double kFloor = 1.0e-12;
    EXPECT_EQ(stored.pos,  kFloor);
    EXPECT_EQ(stored.rot,  kFloor);
    EXPECT_EQ(stored.ba,   kFloor);
    EXPECT_EQ(stored.grav, kFloor);
    // The other fields (off_R_L_I, off_T_L_I, vel, bg) kept the struct
    // defaults — nothing bad touched them.
    EXPECT_DOUBLE_EQ(stored.off_R_L_I, kLegacyOffRLI);
    EXPECT_DOUBLE_EQ(stored.off_T_L_I, kLegacyOffTLI);
    EXPECT_DOUBLE_EQ(stored.vel,       kLegacyVel);
    EXPECT_DOUBLE_EQ(stored.bg,        kLegacyBg);

    // And the clamped values reach P.
    const auto P = fx.FinishInitAndGetP();
    for (int i = 0; i < 3; ++i)   EXPECT_EQ(P(i, i),      kFloor);
    for (int i = 3; i < 6; ++i)   EXPECT_EQ(P(i, i),      kFloor);
    for (int i = 18; i < 21; ++i) EXPECT_EQ(P(i, i),      kFloor);
    for (int i = 21; i < 23; ++i) EXPECT_EQ(P(i, i),      kFloor);
}

// Very large finite values must pass through unclamped — users may want
// a deliberately loose init prior on, e.g., velocity.
TEST(InitPDiag, HugeFiniteValuesPassThrough) {
    ImuFixture fx;

    InitPDiag loose;
    loose.vel = 1.0e6;
    loose.ba  = 1.0e3;
    fx.imu->SetInitPDiag(loose);

    const auto stored = fx.imu->GetInitPDiag();
    EXPECT_DOUBLE_EQ(stored.vel, 1.0e6);
    EXPECT_DOUBLE_EQ(stored.ba,  1.0e3);

    const auto P = fx.FinishInitAndGetP();
    for (int i = 12; i < 15; ++i) EXPECT_DOUBLE_EQ(P(i, i), 1.0e6);
    for (int i = 18; i < 21; ++i) EXPECT_DOUBLE_EQ(P(i, i), 1.0e3);
}

// Repeated set: last call wins, not "merged" with the previous setter.
TEST(InitPDiag, RepeatedSetLastCallWins) {
    ImuFixture fx;

    InitPDiag first;
    first.pos = 5.0;
    first.bg  = 2.0;
    fx.imu->SetInitPDiag(first);

    InitPDiag second;  // struct defaults
    second.bg = 7.0;
    fx.imu->SetInitPDiag(second);

    const auto stored = fx.imu->GetInitPDiag();
    EXPECT_DOUBLE_EQ(stored.pos, kLegacyPos);  // first's pos was discarded
    EXPECT_DOUBLE_EQ(stored.bg,  7.0);          // second's bg is live
}

// Reset + re-init must keep the custom InitPDiag: the setter installs a
// persistent configuration, not a one-shot scratch value.
TEST(InitPDiag, ResetAndReInitKeepsCustomDiag) {
    ImuFixture fx;

    InitPDiag custom;
    custom.bg = 2.5e-3;
    custom.ba = 7.5e-3;
    fx.imu->SetInitPDiag(custom);

    // First init.
    const auto P1 = fx.FinishInitAndGetP();
    for (int i = 15; i < 18; ++i) EXPECT_DOUBLE_EQ(P1(i, i), 2.5e-3);
    for (int i = 18; i < 21; ++i) EXPECT_DOUBLE_EQ(P1(i, i), 7.5e-3);

    // Reset the IMU processor and drive a fresh init.
    fx.imu->Reset();

    // Reset does NOT touch the esekf's P — the user-level contract is to
    // construct a fresh esekf if they want the filter zeroed too. We
    // verify the setter's persistence by running a new Process() cycle:
    // Reset() re-arms imu_need_init_, so the next Process() calls IMUInit
    // again and writes init_P afresh into the existing kf.
    auto meas = MakeStaticBatch(MAX_INI_COUNT + 5, 0.005, /*t0=*/1.0);
    fx.imu->Process(meas, fx.kf, fx.pcl_out);
    const auto P2 = fx.kf.get_P();
    for (int i = 15; i < 18; ++i) EXPECT_DOUBLE_EQ(P2(i, i), 2.5e-3) << "bg lost after Reset+reinit";
    for (int i = 18; i < 21; ++i) EXPECT_DOUBLE_EQ(P2(i, i), 7.5e-3) << "ba lost after Reset+reinit";
}

// ═════════════════════════════════════════════════════════════════════════
// B. Q_ ownership — single writer via RebuildQ
// ═════════════════════════════════════════════════════════════════════════

// Q_ layout invariant: [0..2] = gyr, [3..5] = acc, [6..8] = bias_gyr,
// [9..11] = bias_acc, all other entries zero. Catches any future code that
// introduces an off-diagonal term by accident.
void ExpectQShape(const Eigen::Matrix<double, 12, 12> &Q,
                  const common::V3D &gyr, const common::V3D &acc,
                  const common::V3D &bgyr, const common::V3D &bacc) {
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(Q(i, i),         gyr(i));
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(Q(3 + i, 3 + i), acc(i));
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(Q(6 + i, 6 + i), bgyr(i));
    for (int i = 0; i < 3; ++i) EXPECT_DOUBLE_EQ(Q(9 + i, 9 + i), bacc(i));
    // Off-diagonal must be zero.
    for (int r = 0; r < 12; ++r) {
        for (int c = 0; c < 12; ++c) {
            if (r == c) continue;
            EXPECT_DOUBLE_EQ(Q(r, c), 0.0) << "off-diag (" << r << "," << c << ")";
        }
    }
}

// After ctor, Q_ reflects the ctor-default cov_* members — not the stale
// process_noise_cov() literals that used to live there.
TEST(QOwnership, CtorUsesRuntimeCovs) {
    ImuProcess imu;
    ExpectQShape(imu.Q_,
                 common::V3D(kCtorCovGyr, kCtorCovGyr, kCtorCovGyr),
                 common::V3D(kCtorCovAcc, kCtorCovAcc, kCtorCovAcc),
                 common::V3D(kCtorCovBiasGyr, kCtorCovBiasGyr, kCtorCovBiasGyr),
                 common::V3D(kCtorCovBiasAcc, kCtorCovBiasAcc, kCtorCovBiasAcc));
}

// Bias-cov setters update Q_ immediately — the user should not need to
// complete init before a custom cov_bias_* value propagates.
TEST(QOwnership, BiasCovSettersUpdateQImmediately) {
    ImuProcess imu;
    const common::V3D new_bgyr(1.0e-3, 2.0e-3, 3.0e-3);
    const common::V3D new_bacc(4.0e-3, 5.0e-3, 6.0e-3);
    imu.SetGyrBiasCov(new_bgyr);
    imu.SetAccBiasCov(new_bacc);

    // Measurement covs still at ctor defaults.
    ExpectQShape(imu.Q_,
                 common::V3D(kCtorCovGyr, kCtorCovGyr, kCtorCovGyr),
                 common::V3D(kCtorCovAcc, kCtorCovAcc, kCtorCovAcc),
                 new_bgyr,
                 new_bacc);
}

// After init completes, the measurement covariance blocks switch from
// the init-time running variance to the YAML-scale values carried by the
// Set*Cov setters. (RebuildQ is called at the end of the init-complete
// branch.)
TEST(QOwnership, PostInitUsesScaleValues) {
    ImuFixture fx;

    // Override the scale values AFTER the fixture's default Set*Cov calls.
    const common::V3D gyr_scale(0.55, 0.55, 0.55);
    const common::V3D acc_scale(0.77, 0.77, 0.77);
    fx.imu->SetGyrCov(gyr_scale);
    fx.imu->SetAccCov(acc_scale);

    (void)fx.FinishInitAndGetP();  // drive init to completion

    ExpectQShape(fx.imu->Q_,
                 gyr_scale, acc_scale,
                 common::V3D(kCtorCovBiasGyr, kCtorCovBiasGyr, kCtorCovBiasGyr),
                 common::V3D(kCtorCovBiasAcc, kCtorCovBiasAcc, kCtorCovBiasAcc));
}

// Q_ must NOT be rewritten inside UndistortPcl. Drive init to completion,
// snapshot Q_, then run a second Process() cycle that goes through
// UndistortPcl and verify Q_ is byte-identical. This guards the removal
// of the per-frame Q_.block().diagonal() = cov_*  writes.
TEST(QOwnership, UndistortPclDoesNotTouchQ) {
    ImuFixture fx;
    (void)fx.FinishInitAndGetP();  // init done

    const Eigen::Matrix<double, 12, 12> Q_before = fx.imu->Q_;

    // Second batch with monotonically-increasing timestamps so AddIMU /
    // SyncPackages don't reject it.
    auto meas2 = MakeStaticBatch(/*n_imu=*/25, /*dt=*/0.005, /*t0=*/0.2);
    fx.imu->Process(meas2, fx.kf, fx.pcl_out);

    EXPECT_EQ(fx.imu->Q_, Q_before)
        << "Q_ mutated inside UndistortPcl — the per-frame rebuild has "
        << "re-appeared somewhere and RebuildQ is no longer the single writer.";
}

// Cross-check: mutating the measurement covs AFTER init (before the next
// frame) does NOT automatically rebuild Q_. The upstream contract was
// that Set*Cov updates the scale buffer and runtime cov_* only at the
// next init. We preserve that — the test pins the behaviour so nobody
// accidentally changes it without noticing.
TEST(QOwnership, PostInitGyrAccSettersDoNotRebuildQ) {
    ImuFixture fx;
    (void)fx.FinishInitAndGetP();
    const Eigen::Matrix<double, 12, 12> Q_before = fx.imu->Q_;

    // These only update cov_gyr_scale_ / cov_acc_scale_; runtime cov_gyr_
    // / cov_acc_ are already set from the init-complete assignment and
    // do not change here.
    fx.imu->SetGyrCov(common::V3D(999.0, 999.0, 999.0));
    fx.imu->SetAccCov(common::V3D(999.0, 999.0, 999.0));
    EXPECT_EQ(fx.imu->Q_, Q_before)
        << "SetGyrCov/SetAccCov triggered an early Q_ rebuild — this "
        << "violates the existing scale-indirection semantics.";
}

// ═════════════════════════════════════════════════════════════════════════
// C. YAML integration through LaserMapping::LoadParamsFromYAML
// ═════════════════════════════════════════════════════════════════════════

namespace {

struct YamlSpec {
    // Sparse overrides — leave a field nullopt to omit from the YAML.
    std::optional<double> pos;
    std::optional<double> rot;
    std::optional<double> off_R_L_I;
    std::optional<double> off_T_L_I;
    std::optional<double> vel;
    std::optional<double> bg;
    std::optional<double> ba;
    std::optional<double> grav;
    bool emit_block = true;  // if false, omit `init_P_diag:` entirely
};

// Process-noise block config for YAML synthesis.
//   kDefaultLegacy = emit `mapping.{gyr,acc,b_gyr,b_acc}_cov: 0.1/0.1/1e-4/1e-4`
//   kLegacyCustom  = emit legacy block with caller-supplied values
//   kNewCustom     = emit `process_noise: {ng, na, nbg, nba}` with caller values
//   kBoth          = emit BOTH blocks (values diverge so the test can see which wins)
//   kNone          = emit neither (expected to make Init() fail)
enum class PnMode { kDefaultLegacy, kLegacyCustom, kNewCustom, kBoth, kNone };

struct PnValues {
    float gyr  = 0.1f;
    float acc  = 0.1f;
    float bgyr = 1.0e-4f;
    float bacc = 1.0e-4f;
};

// Write a self-contained YAML that LaserMapping::Init accepts, with the
// `imu_init.init_P_diag` subtree taken from `spec`. Returns the path.
std::string WriteYAMLWithInitPDiag(const std::string &tag, const YamlSpec &spec,
                                   PnMode pn = PnMode::kDefaultLegacy,
                                   const PnValues &legacy = {},
                                   const PnValues &preferred = {}) {
    const std::string path = std::string(ROOT_DIR) + "config/test_init_p_diag_" + tag + ".yaml";
    std::ofstream f(path);
    f << "common:\n  time_sync_en: false\n"
      << "preprocess:\n  blind: 0.1\n"
      << "mapping:\n";
    const bool emit_legacy = (pn == PnMode::kDefaultLegacy || pn == PnMode::kLegacyCustom ||
                              pn == PnMode::kBoth);
    const bool emit_new    = (pn == PnMode::kNewCustom || pn == PnMode::kBoth);
    if (emit_legacy) {
        f << "  acc_cov: "   << legacy.acc  << "\n"
          << "  gyr_cov: "   << legacy.gyr  << "\n"
          << "  b_acc_cov: " << legacy.bacc << "\n"
          << "  b_gyr_cov: " << legacy.bgyr << "\n";
    }
    f << "  det_range: 100.0\n"
      << "  extrinsic_est_en: false\n"
      << "  extrinsic_T: [0, 0, 0]\n"
      << "  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n";
    if (emit_new) {
        f << "process_noise:\n"
          << "  ng:  " << preferred.gyr  << "\n"
          << "  na:  " << preferred.acc  << "\n"
          << "  nbg: " << preferred.bgyr << "\n"
          << "  nba: " << preferred.bacc << "\n";
    }
    f << "imu_init:\n"
      << "  motion_gate_enabled: false\n";
    if (spec.emit_block) {
        f << "  init_P_diag:\n";
        auto emit = [&](const char *k, const std::optional<double> &v) {
            if (v) f << "    " << k << ": " << *v << "\n";
        };
        emit("pos",       spec.pos);
        emit("rot",       spec.rot);
        emit("off_R_L_I", spec.off_R_L_I);
        emit("off_T_L_I", spec.off_T_L_I);
        emit("vel",       spec.vel);
        emit("bg",        spec.bg);
        emit("ba",        spec.ba);
        emit("grav",      spec.grav);
    }
    f << "max_iteration: 4\n"
      << "point_filter_num: 1\n"
      << "filter_size_surf: 0.15\n"
      << "filter_size_map: 0.15\n"
      << "cube_side_length: 1000\n"
      << "ivox_grid_resolution: 0.15\n"
      << "ivox_nearby_type: 18\n"
      << "esti_plane_threshold: 0.1\n"
      << "map_quality_threshold: 0.0\n"
      << "output:\n  path_en: false\n  dense_en: false\n  path_save_en: false\n"
      << "pcd_save:\n  pcd_save_en: false\n  interval: -1\n";
    return path;
}

}  // namespace

// YAML without any init_P_diag block → ImuProcess stores struct defaults.
TEST(YamlInitPDiag, AbsentBlockKeepsLegacyDefaults) {
    YamlSpec spec;
    spec.emit_block = false;
    const auto path = WriteYAMLWithInitPDiag("absent", spec);

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path)) << "Init failed with " << path;
    const auto *imu = mapping.GetImuProcess();
    ASSERT_NE(imu, nullptr);

    const auto &ip = imu->GetInitPDiag();
    EXPECT_DOUBLE_EQ(ip.pos,       kLegacyPos);
    EXPECT_DOUBLE_EQ(ip.rot,       kLegacyRot);
    EXPECT_DOUBLE_EQ(ip.off_R_L_I, kLegacyOffRLI);
    EXPECT_DOUBLE_EQ(ip.off_T_L_I, kLegacyOffTLI);
    EXPECT_DOUBLE_EQ(ip.vel,       kLegacyVel);
    EXPECT_DOUBLE_EQ(ip.bg,        kLegacyBg);
    EXPECT_DOUBLE_EQ(ip.ba,        kLegacyBa);
    EXPECT_DOUBLE_EQ(ip.grav,      kLegacyGrav);

    std::remove(path.c_str());
}

// YAML with an empty init_P_diag block (header but no children) → same as
// absent. This is a common copy-paste state; it must not crash.
TEST(YamlInitPDiag, EmptyBlockKeepsLegacyDefaults) {
    YamlSpec spec;  // emit_block=true but all fields nullopt → only header
    const auto path = WriteYAMLWithInitPDiag("empty", spec);

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto &ip = mapping.GetImuProcess()->GetInitPDiag();
    EXPECT_DOUBLE_EQ(ip.pos, kLegacyPos);
    EXPECT_DOUBLE_EQ(ip.bg,  kLegacyBg);
    std::remove(path.c_str());
}

// Partial YAML — only two fields. The others stay at struct defaults.
TEST(YamlInitPDiag, PartialOverrideTouchesOnlySpecifiedFields) {
    YamlSpec spec;
    spec.pos = 0.25;
    spec.bg  = 5.0e-5;
    const auto path = WriteYAMLWithInitPDiag("partial", spec);

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto &ip = mapping.GetImuProcess()->GetInitPDiag();
    EXPECT_DOUBLE_EQ(ip.pos, 0.25);
    EXPECT_DOUBLE_EQ(ip.bg,  5.0e-5);
    // Untouched → legacy defaults.
    EXPECT_DOUBLE_EQ(ip.rot,       kLegacyRot);
    EXPECT_DOUBLE_EQ(ip.off_R_L_I, kLegacyOffRLI);
    EXPECT_DOUBLE_EQ(ip.off_T_L_I, kLegacyOffTLI);
    EXPECT_DOUBLE_EQ(ip.vel,       kLegacyVel);
    EXPECT_DOUBLE_EQ(ip.ba,        kLegacyBa);
    EXPECT_DOUBLE_EQ(ip.grav,      kLegacyGrav);
    std::remove(path.c_str());
}

// All eight fields overridden — every value round-trips through YAML
// parsing without truncation or loss. Use distinct values per field to
// catch any key-typo that'd land one value into the wrong slot.
TEST(YamlInitPDiag, FullOverrideRoundTripsAllEightFields) {
    YamlSpec spec;
    spec.pos       = 0.11;
    spec.rot       = 0.22;
    spec.off_R_L_I = 3.3e-5;
    spec.off_T_L_I = 4.4e-5;
    spec.vel       = 0.55;
    spec.bg        = 6.6e-4;
    spec.ba        = 7.7e-3;
    spec.grav      = 8.8e-5;
    const auto path = WriteYAMLWithInitPDiag("full", spec);

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto &ip = mapping.GetImuProcess()->GetInitPDiag();
    EXPECT_DOUBLE_EQ(ip.pos,       0.11);
    EXPECT_DOUBLE_EQ(ip.rot,       0.22);
    EXPECT_DOUBLE_EQ(ip.off_R_L_I, 3.3e-5);
    EXPECT_DOUBLE_EQ(ip.off_T_L_I, 4.4e-5);
    EXPECT_DOUBLE_EQ(ip.vel,       0.55);
    EXPECT_DOUBLE_EQ(ip.bg,        6.6e-4);
    EXPECT_DOUBLE_EQ(ip.ba,        7.7e-3);
    EXPECT_DOUBLE_EQ(ip.grav,      8.8e-5);
    std::remove(path.c_str());
}

// Negative / zero YAML values are clamped by SetInitPDiag — Init() must
// still succeed, not abort the pipeline, and the stored values must be
// positive.
TEST(YamlInitPDiag, NonPositiveValuesAreClampedNotRejected) {
    YamlSpec spec;
    spec.pos = -1.0;
    spec.ba  = 0.0;
    const auto path = WriteYAMLWithInitPDiag("clamp", spec);

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path))
        << "Init rejected YAML with invalid init_P_diag values — must clamp, not abort.";

    const auto &ip = mapping.GetImuProcess()->GetInitPDiag();
    EXPECT_GT(ip.pos, 0.0);
    EXPECT_GT(ip.ba,  0.0);
    std::remove(path.c_str());
}

// ═════════════════════════════════════════════════════════════════════════
// D. Process-noise YAML: new `process_noise` block with back-compat for
//    the legacy `mapping.{gyr,acc,b_gyr,b_acc}_cov` keys.
//
// Both key groups write the same four cov_* members on ImuProcess; the
// tests below read those members back to confirm which block won.
// ═════════════════════════════════════════════════════════════════════════

namespace {

// Shortcut: open a LaserMapping on `path`, return the stored cov_*
// members as a 4-tuple so comparisons are concise.
struct StoredPn { float gyr, acc, bgyr, bacc; };

StoredPn InitAndReadPn(const std::string &path) {
    LaserMapping mapping;
    EXPECT_TRUE(mapping.Init(path)) << "Init failed on " << path;
    const auto *imu = mapping.GetImuProcess();
    EXPECT_NE(imu, nullptr);
    return StoredPn{
        static_cast<float>(imu->cov_gyr_scale_.x()),
        static_cast<float>(imu->cov_acc_scale_.x()),
        static_cast<float>(imu->cov_bias_gyr_.x()),
        static_cast<float>(imu->cov_bias_acc_.x()),
    };
}

}  // namespace

// Legacy-only config (what every in-repo YAML looks like today) still
// populates cov_*_scale_ / cov_bias_* correctly. Regression guard: the
// deprecation path must remain functional as long as downstream configs
// use the old key names.
TEST(YamlProcessNoise, LegacyKeysOnlyStillWorks) {
    YamlSpec spec;
    spec.emit_block = false;
    PnValues legacy{.gyr = 0.22f, .acc = 0.33f, .bgyr = 1.1e-4f, .bacc = 2.2e-4f};
    const auto path = WriteYAMLWithInitPDiag("pn_legacy", spec, PnMode::kLegacyCustom, legacy);

    const auto got = InitAndReadPn(path);
    EXPECT_FLOAT_EQ(got.gyr,  0.22f);
    EXPECT_FLOAT_EQ(got.acc,  0.33f);
    EXPECT_FLOAT_EQ(got.bgyr, 1.1e-4f);
    EXPECT_FLOAT_EQ(got.bacc, 2.2e-4f);
    std::remove(path.c_str());
}

// New `process_noise` block alone — no legacy keys — works end-to-end.
// This is the shape any newly-authored config should use.
TEST(YamlProcessNoise, NewKeysOnlyWorks) {
    YamlSpec spec;
    spec.emit_block = false;
    PnValues new_v{.gyr = 0.44f, .acc = 0.55f, .bgyr = 3.3e-4f, .bacc = 4.4e-4f};
    const auto path = WriteYAMLWithInitPDiag("pn_new", spec, PnMode::kNewCustom, /*legacy*/{}, new_v);

    const auto got = InitAndReadPn(path);
    EXPECT_FLOAT_EQ(got.gyr,  0.44f);
    EXPECT_FLOAT_EQ(got.acc,  0.55f);
    EXPECT_FLOAT_EQ(got.bgyr, 3.3e-4f);
    EXPECT_FLOAT_EQ(got.bacc, 4.4e-4f);
    std::remove(path.c_str());
}

// Conflict case: both blocks present with different values. The new
// `process_noise` block must win. Catches a regression where precedence
// quietly flipped and every run silently uses the deprecated values.
TEST(YamlProcessNoise, NewWinsWhenBothPresent) {
    YamlSpec spec;
    spec.emit_block = false;
    PnValues legacy{.gyr = 9.9f, .acc = 9.9f, .bgyr = 9.9f, .bacc = 9.9f};          // sentinel
    PnValues preferred{.gyr = 0.10f, .acc = 0.20f, .bgyr = 0.30f, .bacc = 0.40f};
    const auto path = WriteYAMLWithInitPDiag("pn_both", spec, PnMode::kBoth, legacy, preferred);

    const auto got = InitAndReadPn(path);
    EXPECT_FLOAT_EQ(got.gyr,  0.10f) << "legacy sentinel 9.9 leaked past `process_noise`";
    EXPECT_FLOAT_EQ(got.acc,  0.20f);
    EXPECT_FLOAT_EQ(got.bgyr, 0.30f);
    EXPECT_FLOAT_EQ(got.bacc, 0.40f);
    std::remove(path.c_str());
}

// Neither block → Init() returns false with an error in the log. Prevents
// silent defaulting to arbitrary values.
TEST(YamlProcessNoise, NeitherBlockIsError) {
    YamlSpec spec;
    spec.emit_block = false;
    const auto path = WriteYAMLWithInitPDiag("pn_none", spec, PnMode::kNone);

    LaserMapping mapping;
    EXPECT_FALSE(mapping.Init(path))
        << "Init succeeded with neither `process_noise` nor `mapping.*_cov` — "
        << "should have errored out.";
    std::remove(path.c_str());
}

// Same numeric values under both key naming schemes must produce
// byte-identical stored cov_*_scale_ / cov_bias_* members. Pins the
// promise that the rename is purely cosmetic for any value.
TEST(YamlProcessNoise, ByteIdenticalForSameValues) {
    YamlSpec spec;
    spec.emit_block = false;
    PnValues vals{.gyr = 0.12f, .acc = 0.34f, .bgyr = 5.6e-5f, .bacc = 7.8e-5f};

    const auto legacy_path = WriteYAMLWithInitPDiag("pn_same_legacy", spec, PnMode::kLegacyCustom, vals);
    const auto new_path    = WriteYAMLWithInitPDiag("pn_same_new",    spec, PnMode::kNewCustom,    /*legacy*/{}, vals);

    const auto got_legacy = InitAndReadPn(legacy_path);
    const auto got_new    = InitAndReadPn(new_path);

    EXPECT_FLOAT_EQ(got_legacy.gyr,  got_new.gyr);
    EXPECT_FLOAT_EQ(got_legacy.acc,  got_new.acc);
    EXPECT_FLOAT_EQ(got_legacy.bgyr, got_new.bgyr);
    EXPECT_FLOAT_EQ(got_legacy.bacc, got_new.bacc);

    std::remove(legacy_path.c_str());
    std::remove(new_path.c_str());
}

// ═════════════════════════════════════════════════════════════════════════
// E. assume_level=true across all body-axis mount orientations.
//
// The imu_processing.h comment historically warned "Leave OFF for
// non-standard mounts (e.g. X-up Xsens on Hilti)" — but the post-init code
// in ImuProcess::Process reads:
//
//     body_up   = mean_acc_ / ‖mean_acc_‖        (measured, mount-agnostic)
//     world_up  = (0, 0, 1)                       (canonical world frame)
//     rot       = Quaternion::FromTwoVectors(body_up, world_up)
//
// So the resulting `post_init.rot` always rotates the body-frame "up" to
// world Z+ regardless of which body axis gravity reads on. The tests
// below prove this for all six axis-aligned mounts. If they all pass
// the docstring caveat can be dropped.
// ═════════════════════════════════════════════════════════════════════════

namespace {

struct MountCase {
    std::string name;
    Eigen::Vector3d body_up;  // direction of +gravity in body frame
};

class AssumeLevelMount : public ::testing::TestWithParam<MountCase> {};

}  // namespace

TEST_P(AssumeLevelMount, PostInitMapsBodyUpToWorldZ) {
    const auto &mc = GetParam();
    ImuFixture fx;
    fx.imu->SetInitAssumeLevel(true);

    // Stationary IMU with gravity along the chosen body axis. mean_acc_
    // is specific force = -g_body; for a resting rig the IMU reads
    // +|g| along the body_up direction.
    const Eigen::Vector3d accel = mc.body_up * common::G_m_s2;
    auto meas = MakeStaticBatch(MAX_INI_COUNT + 5, 0.005, 0.0, accel);
    fx.imu->Process(meas, fx.kf, fx.pcl_out);

    const auto s = fx.kf.get_x();
    const Eigen::Quaterniond rot_q(s.rot);

    // Invariant 1: world gravity points along −Z. Magnitude lives on the
    // S2 manifold whose length = 9.8090 (use-ikfom.hpp line 26). The
    // 0.001 m/s² inconsistency with common::G_m_s2 = 9.81 is intentional
    // (common_lib.h:26 explains why). So check the DIRECTION, not the
    // absolute z value.
    const Eigen::Vector3d grav(s.grav[0], s.grav[1], s.grav[2]);
    const double g_mag = grav.norm();
    ASSERT_GT(g_mag, 1e-3) << mc.name << ": grav collapsed to zero";
    const Eigen::Vector3d grav_dir = grav / g_mag;
    EXPECT_NEAR(grav_dir.x(),  0.0, 1e-9) << mc.name;
    EXPECT_NEAR(grav_dir.y(),  0.0, 1e-9) << mc.name;
    EXPECT_NEAR(grav_dir.z(), -1.0, 1e-9) << mc.name;

    // Invariant 2: post_init.rot maps the body's gravity-up direction to
    // world +Z. This is what makes the post-init state a valid ENU-style
    // world frame regardless of mount.
    const Eigen::Vector3d body_up_in_world = rot_q * mc.body_up;
    const Eigen::Vector3d world_up(0.0, 0.0, 1.0);
    EXPECT_LT((body_up_in_world - world_up).norm(), 1e-5)
        << mc.name << ": rot * body_up should land on (0,0,+1), "
        << "got (" << body_up_in_world.x() << ", " << body_up_in_world.y()
        << ", " << body_up_in_world.z() << ")";
}

// Cover all six axis-aligned mounts. ±Z is the typical Livox/Ouster case;
// +X is the Hilti Xsens case. The antiparallel cases (-Z, -X, -Y) exercise
// Eigen::Quaternion::FromTwoVectors' 180° path which picks an arbitrary
// perpendicular rotation axis — the invariant test above still holds.
INSTANTIATE_TEST_SUITE_P(
    AllSixMountAxes, AssumeLevelMount,
    ::testing::Values(
        MountCase{"Zpos", Eigen::Vector3d( 0,  0,  1)},
        MountCase{"Zneg", Eigen::Vector3d( 0,  0, -1)},
        MountCase{"Xpos", Eigen::Vector3d( 1,  0,  0)},
        MountCase{"Xneg", Eigen::Vector3d(-1,  0,  0)},
        MountCase{"Ypos", Eigen::Vector3d( 0,  1,  0)},
        MountCase{"Yneg", Eigen::Vector3d( 0, -1,  0)}),
    [](const testing::TestParamInfo<AssumeLevelMount::ParamType> &info) { return info.param.name; });

// Also check that grav does NOT end up oriented toward body_up when
// assume_level=false. This is the historical behaviour (world frame carries
// the sensor tilt). We want to ensure the two branches remain distinct —
// otherwise the flag has no effect.
TEST(AssumeLevelMount, AssumeLevelFalseWorldGravCarriesBodyTilt) {
    ImuFixture fx;
    fx.imu->SetInitAssumeLevel(false);

    const Eigen::Vector3d body_up(1.0, 0.0, 0.0);  // X-up mount
    const Eigen::Vector3d accel = body_up * common::G_m_s2;
    auto meas = MakeStaticBatch(MAX_INI_COUNT + 5, 0.005, 0.0, accel);
    fx.imu->Process(meas, fx.kf, fx.pcl_out);

    const auto s = fx.kf.get_x();
    const Eigen::Vector3d grav(s.grav[0], s.grav[1], s.grav[2]);
    // Non-assume-level path sets grav = -mean_acc/|mean_acc| * G. For the
    // X-up rig the world frame is tilted so grav points along body-X−.
    // Again: grav lives on S2 (length 9.8090) so compare by direction.
    const double g_mag = grav.norm();
    ASSERT_GT(g_mag, 1e-3);
    const Eigen::Vector3d grav_dir = grav / g_mag;
    EXPECT_NEAR(grav_dir.x(), -1.0, 1e-9);
    EXPECT_NEAR(grav_dir.y(),  0.0, 1e-9);
    EXPECT_NEAR(grav_dir.z(),  0.0, 1e-9);
}

// ═════════════════════════════════════════════════════════════════════════
// F. Rate-adaptive init gate — time-based configuration resolves to sample
//    counts at the first IMUInit batch using the measured IMU rate. The
//    old hard-coded 100 / 1000 defaults assumed 200 Hz; now the counts
//    scale with whatever the IMU is actually emitting.
// ═════════════════════════════════════════════════════════════════════════

namespace {

// Drive Process() once with a batch at the requested sample rate so the
// rate-resolution path runs. 25 samples is enough to both estimate the
// rate accurately (dur = 24*dt) and consume the init-iter-num bookkeeping
// without completing the 100-sample default gate.
void DriveOneBatchAtRate(ImuFixture &fx, double rate_hz, double t0 = 0.0,
                         const Eigen::Vector3d &accel = Eigen::Vector3d(0.0, 0.0, common::G_m_s2)) {
    const double dt = 1.0 / rate_hz;
    auto meas = MakeStaticBatch(MAX_INI_COUNT + 5, dt, t0, accel);
    fx.imu->Process(meas, fx.kf, fx.pcl_out);
}

}  // namespace

// At 200 Hz the defaults (0.5s / 5.0s) resolve to 100 / 1000 — the exact
// numbers the upstream hard-coded constants assumed. Regression guard.
TEST(InitGateTime, Defaults_200HzIMU_ResolvesTo100And1000) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05,
                                  DEFAULT_INIT_GATE_MIN_TIME_S,
                                  DEFAULT_INIT_GATE_MAX_TIME_S);
    DriveOneBatchAtRate(fx, 200.0);

    EXPECT_NEAR(fx.imu->GetEstimatedImuRateHz(), 200.0, 0.5);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 100);
    EXPECT_EQ(fx.imu->GetResolvedInitMaxTries(),    1000);
}

// Half the rate → half the samples for the same time window. A 100 Hz
// IMU resolves 0.5 s → 50 samples (old code would have used 100 samples
// = 1.0 s, over-delaying init).
TEST(InitGateTime, Defaults_100HzIMU_ResolvesTo50And500) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05,
                                  DEFAULT_INIT_GATE_MIN_TIME_S,
                                  DEFAULT_INIT_GATE_MAX_TIME_S);
    DriveOneBatchAtRate(fx, 100.0);

    EXPECT_NEAR(fx.imu->GetEstimatedImuRateHz(), 100.0, 0.5);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 50);
    EXPECT_EQ(fx.imu->GetResolvedInitMaxTries(),    500);
}

// Higher-rate IMUs need more samples for the same noise-averaging window.
// 400 Hz → 200 samples for 0.5 s (old code would have used 100 = 0.25 s,
// too short for good gravity averaging).
TEST(InitGateTime, Defaults_400HzIMU_ResolvesTo200And2000) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05,
                                  DEFAULT_INIT_GATE_MIN_TIME_S,
                                  DEFAULT_INIT_GATE_MAX_TIME_S);
    DriveOneBatchAtRate(fx, 400.0);

    EXPECT_NEAR(fx.imu->GetEstimatedImuRateHz(), 400.0, 0.5);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 200);
    EXPECT_EQ(fx.imu->GetResolvedInitMaxTries(),    2000);
}

// User-supplied times scale the same way. A 2-second min at 204 Hz (Avia)
// gives 408 samples — matching the explicit `min_accepted: 400` value in
// the current r3live_avia config within rounding.
TEST(InitGateTime, CustomTimes_204HzResolvesCorrectly) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05,
                                  /*min_time_s=*/2.0,
                                  /*max_time_s=*/10.0);
    DriveOneBatchAtRate(fx, 204.0);

    EXPECT_NEAR(fx.imu->GetEstimatedImuRateHz(), 204.0, 1.0);
    // Allow ±1 sample for rounding at non-integer rate.
    EXPECT_NEAR(fx.imu->GetResolvedInitMinAccepted(), 408, 1);
    EXPECT_NEAR(fx.imu->GetResolvedInitMaxTries(),    2040, 1);
}

// Rate estimation falls back to 200 Hz when the first batch has a single
// IMU sample (can't compute duration from one timestamp).
TEST(InitGateTime, SingleSampleBatchFallsBackTo200Hz) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05, 0.5, 5.0);

    // Batch with just one IMU sample — duration = 0, can't estimate rate.
    auto meas = MakeStaticBatch(/*n_imu=*/1, /*dt=*/0.005);
    fx.imu->Process(meas, fx.kf, fx.pcl_out);

    EXPECT_DOUBLE_EQ(fx.imu->GetEstimatedImuRateHz(), DEFAULT_IMU_RATE_HZ_FALLBACK);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 100);
    EXPECT_EQ(fx.imu->GetResolvedInitMaxTries(),    1000);
}

// Zero-duration batch (all samples at identical timestamps) also falls
// back. Otherwise rate = N/0 = inf → integer overflow → garbage counts.
TEST(InitGateTime, ZeroDurationBatchFallsBackTo200Hz) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05, 0.5, 5.0);

    // 25 samples all at t=0.
    common::MeasureGroup meas;
    meas.lidar_ = MakeTinyCloud();
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;
    for (int i = 0; i < MAX_INI_COUNT + 5; ++i) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = 0.0;
        imu->linear_acceleration = Eigen::Vector3d(0, 0, common::G_m_s2);
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }
    fx.imu->Process(meas, fx.kf, fx.pcl_out);

    EXPECT_DOUBLE_EQ(fx.imu->GetEstimatedImuRateHz(), DEFAULT_IMU_RATE_HZ_FALLBACK);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 100);
}

// Explicit-count setter bypasses the time-based path. SetInitMotionGate
// after SetInitMotionGateTime must win — no rate estimation on next init.
TEST(InitGateTime, ExplicitCountAfterTimeSetterWins) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05, 0.5, 5.0);
    fx.imu->SetInitMotionGate(true, 0.05, 0.05, /*min_accepted=*/333, /*max_tries=*/4444);

    // Drive at 400 Hz. If time-based resolution still ran, min_accepted
    // would become 200 = 0.5 * 400. It should stay at 333 instead.
    DriveOneBatchAtRate(fx, 400.0);

    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 333)
        << "Explicit count setter was clobbered by late time-based resolution.";
    EXPECT_EQ(fx.imu->GetResolvedInitMaxTries(), 4444);
}

// Reset() on a time-configured gate must re-invalidate the resolved
// counts so a subsequent init with a different IMU rate re-estimates.
// (Unlikely in a real bag where rate is fixed, but the invariant is what
// lets tests cleanly rebuild fixtures.)
TEST(InitGateTime, ResetRetriggersResolutionAtNewRate) {
    ImuFixture fx;
    fx.imu->SetInitMotionGateTime(true, 0.05, 0.05, 0.5, 5.0);
    DriveOneBatchAtRate(fx, 200.0);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 100);

    fx.imu->Reset();
    // Second init at a different rate — new counts must follow the new rate.
    DriveOneBatchAtRate(fx, 400.0, /*t0=*/0.5);
    EXPECT_NEAR(fx.imu->GetEstimatedImuRateHz(), 400.0, 0.5);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 200);
}

// Reset() on a COUNT-configured gate must NOT invalidate the counts —
// the user's explicit numbers persist across resets. Otherwise callers
// would have to re-call the setter after every Reset().
TEST(InitGateTime, ResetPreservesExplicitCounts) {
    ImuFixture fx;
    fx.imu->SetInitMotionGate(true, 0.05, 0.05, /*min_accepted=*/77, /*max_tries=*/999);
    DriveOneBatchAtRate(fx, 200.0);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 77);

    fx.imu->Reset();
    DriveOneBatchAtRate(fx, 400.0, /*t0=*/0.5);
    EXPECT_EQ(fx.imu->GetResolvedInitMinAccepted(), 77)
        << "Reset cleared an explicit sample-count setter value.";
}

// ── YAML integration for the new time keys ──────────────────────────────

namespace {

// Extend WriteYAMLWithInitPDiag with a lightweight wrapper: emit a YAML
// whose imu_init block contains exactly the keys supplied. Keeps the
// tests below tight rather than growing WriteYAMLWithInitPDiag's signature.
std::string WriteYAMLWithInitGateBlock(const std::string &tag, const std::string &block_body) {
    const std::string path = std::string(ROOT_DIR) + "config/test_init_gate_" + tag + ".yaml";
    std::ofstream f(path);
    f << "common:\n  time_sync_en: false\n"
      << "preprocess:\n  blind: 0.1\n"
      << "mapping:\n"
      << "  acc_cov: 0.1\n  gyr_cov: 0.1\n"
      << "  b_acc_cov: 0.0001\n  b_gyr_cov: 0.0001\n"
      << "  det_range: 100.0\n"
      << "  extrinsic_est_en: false\n"
      << "  extrinsic_T: [0, 0, 0]\n"
      << "  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n"
      << "imu_init:\n"
      << block_body
      << "max_iteration: 4\n"
      << "point_filter_num: 1\n"
      << "filter_size_surf: 0.15\n"
      << "filter_size_map: 0.15\n"
      << "cube_side_length: 1000\n"
      << "ivox_grid_resolution: 0.15\n"
      << "ivox_nearby_type: 18\n"
      << "esti_plane_threshold: 0.1\n"
      << "map_quality_threshold: 0.0\n"
      << "output:\n  path_en: false\n  dense_en: false\n  path_save_en: false\n"
      << "pcd_save:\n  pcd_save_en: false\n  interval: -1\n";
    return path;
}

}  // namespace

// YAML with only time keys → time-based path. Verify the time values
// landed on the ImuProcess (positive GetInitMinTimeS/MaxTimeS), which
// distinguishes the time-setter path from the count-setter path. Count
// resolution itself happens on the first IMUInit batch and is covered by
// the direct-ImuProcess tests above.
TEST(YamlInitGate, TimeKeysOnly_ConfiguresTimePath) {
    const auto path = WriteYAMLWithInitGateBlock("time_only",
        "  motion_gate_enabled: true\n"
        "  min_time_s: 1.5\n"
        "  max_time_s: 15.0\n");

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto *imu = mapping.GetImuProcess();
    ASSERT_NE(imu, nullptr);
    EXPECT_DOUBLE_EQ(imu->GetInitMinTimeS(), 1.5);
    EXPECT_DOUBLE_EQ(imu->GetInitMaxTimeS(), 15.0);
    // Counts are placeholders until rate is measured; rate itself is 0.
    EXPECT_DOUBLE_EQ(imu->GetEstimatedImuRateHz(), 0.0);
    std::remove(path.c_str());
}

// YAML with only count keys → legacy count path. Accessor reads back the
// stored counts directly (no resolution needed — counts are pre-set). The
// time-setter fields are left at their "inactive" sentinel (-1) so a
// subsequent call to Reset() won't retrigger resolution.
TEST(YamlInitGate, CountKeysOnly_UsesLegacyPath) {
    const auto path = WriteYAMLWithInitGateBlock("count_only",
        "  motion_gate_enabled: true\n"
        "  min_accepted: 123\n"
        "  max_tries: 456\n");

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto *imu = mapping.GetImuProcess();
    ASSERT_NE(imu, nullptr);
    EXPECT_EQ(imu->GetResolvedInitMinAccepted(), 123);
    EXPECT_EQ(imu->GetResolvedInitMaxTries(),    456);
    EXPECT_LT(imu->GetInitMinTimeS(), 0.0);   // time path inactive
    EXPECT_LT(imu->GetInitMaxTimeS(), 0.0);
    // Rate estimation never ran because counts were pre-resolved.
    EXPECT_DOUBLE_EQ(imu->GetEstimatedImuRateHz(), 0.0);
    std::remove(path.c_str());
}

// Conflict: both count and time keys present. Counts win (documented
// behaviour so existing per-bag configs with explicit min_accepted keep
// working even if a team-wide default got a time block added).
TEST(YamlInitGate, BothKeys_CountsWin) {
    const auto path = WriteYAMLWithInitGateBlock("both",
        "  motion_gate_enabled: true\n"
        "  min_accepted: 77\n"
        "  min_time_s: 1.0\n"
        "  max_tries: 888\n"
        "  max_time_s: 8.0\n");

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto *imu = mapping.GetImuProcess();
    EXPECT_EQ(imu->GetResolvedInitMinAccepted(), 77);
    EXPECT_EQ(imu->GetResolvedInitMaxTries(),    888);
    std::remove(path.c_str());
}

// Only motion_gate_enabled with no count/time overrides → uses the
// time-based defaults (0.5 s / 5.0 s). Verify via the time accessors so
// we distinguish this from the count path.
TEST(YamlInitGate, DefaultsAreTimeBased_500ms_5s) {
    const auto path = WriteYAMLWithInitGateBlock("default_time",
        "  motion_gate_enabled: true\n");

    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(path));
    const auto *imu = mapping.GetImuProcess();
    EXPECT_DOUBLE_EQ(imu->GetInitMinTimeS(), DEFAULT_INIT_GATE_MIN_TIME_S);
    EXPECT_DOUBLE_EQ(imu->GetInitMaxTimeS(), DEFAULT_INIT_GATE_MAX_TIME_S);
    EXPECT_DOUBLE_EQ(imu->GetEstimatedImuRateHz(), 0.0);
    std::remove(path.c_str());
}
