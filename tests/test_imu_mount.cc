// ─────────────────────────────────────────────────────────────────────────
// Diagnostic test battery for faster-lio.
//
// Goal: isolate *where* drift comes from when the input is synthetic.
// The previous hypothesis — non-standard IMU mount — was falsified by a
// frame-invariance test (all four gravity-axis orientations drifted by
// 1.003632 m after 45 s, to 7 s.f. identical). Now we instrument the
// filter against a well-behaved 3D environment and walk through every
// mechanic that might be responsible for that residual drift.
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>
#include "faster_lio/imu_processing.h"

using namespace faster_lio;

namespace {

constexpr double kScanDt = 0.1;             // 10 Hz LiDAR
constexpr int    kImuPerScan = 20;          // 200 Hz IMU
constexpr double kImuDt = kScanDt / kImuPerScan;
constexpr double kG = common::G_m_s2;

// Synthesize a dense, non-degenerate 3D scan: 5 m × 5 m × 3 m "room",
// sampled uniformly on all six inner walls (including floor + ceiling).
// This gives the IEKF strong observability in all three axes — far from
// the pathological 1D point strip the earlier test used.
PointCloudType::Ptr MakeRoomScan(int points_per_wall = 40,
                                  uint32_t seed = 42) {
    auto cloud = PointCloudType::Ptr(new PointCloudType());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    // Six walls of a box centred at the sensor, half-extent 2.5 m in XY
    // and 1.5 m in Z.
    struct Wall { int axis; float sign; };
    const Wall walls[] = {{0, -1}, {0, +1}, {1, -1}, {1, +1}, {2, -1}, {2, +1}};
    int per = points_per_wall / 6;
    for (const auto& w : walls) {
        for (int i = 0; i < per; ++i) {
            PointType p;
            float a = u(rng), b = u(rng);
            if (w.axis == 0) { p.x = w.sign * 2.5f; p.y = 2.5f * a; p.z = 1.5f * b; }
            else if (w.axis == 1) { p.x = 2.5f * a; p.y = w.sign * 2.5f; p.z = 1.5f * b; }
            else { p.x = 2.5f * a; p.y = 2.5f * b; p.z = w.sign * 1.5f; }
            p.intensity = 100.0f;
            p.curvature = static_cast<float>(i) * kScanDt * 1000.0f / per;
            p.normal_x = p.normal_y = p.normal_z = 0.0f;
            cloud->push_back(p);
        }
    }
    return cloud;
}

common::MeasureGroup MakeBatch(const Eigen::Vector3d& accel_imu,
                                const Eigen::Vector3d& gyro_imu,
                                const PointCloudType::Ptr& scan,
                                double t_start) {
    common::MeasureGroup meas;
    meas.lidar_ = scan;
    meas.lidar_bag_time_ = t_start;
    meas.lidar_end_time_ = t_start + kScanDt;
    for (int i = 0; i < kImuPerScan; ++i) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = t_start + i * kImuDt;
        imu->linear_acceleration = accel_imu;
        imu->angular_velocity = gyro_imu;
        meas.imu_.push_back(imu);
    }
    return meas;
}

struct Fixture {
    std::shared_ptr<ImuProcess> imu;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    PointCloudType::Ptr pcl_out{new PointCloudType()};

    Fixture() {
        imu = std::make_shared<ImuProcess>();
        imu->SetExtrinsic(common::Zero3d, common::Eye3d);
        imu->SetGyrCov(common::V3D(0.1, 0.1, 0.1));
        imu->SetAccCov(common::V3D(0.1, 0.1, 0.1));
        imu->SetGyrBiasCov(common::V3D(1e-4, 1e-4, 1e-4));
        imu->SetAccBiasCov(common::V3D(1e-4, 1e-4, 1e-4));
        std::vector<double> epsi(23, 0.001);
        kf.init_dyn_share(
            get_f, df_dx, df_dw,
            [](state_ikfom&, esekfom::dyn_share_datastruct<double>&) {},
            4, epsi.data());
    }
};

// Drift trace captured at a fixed stride, useful for seeing the shape of
// the error over time (linear accumulation, bias convergence, step, …).
struct DriftTrace {
    std::vector<double> t;
    std::vector<double> px, py, pz;
    std::vector<double> vx, vy, vz;
    std::vector<double> qnorm;
};

DriftTrace RunScans(Fixture& fx,
                     const Eigen::Vector3d& accel,
                     const Eigen::Vector3d& gyro,
                     int n_scans,
                     int log_every = 100,
                     const PointCloudType::Ptr& scan = nullptr) {
    DriftTrace trace;
    auto cloud = scan ? scan : MakeRoomScan();
    double t = 0.0;
    for (int s = 0; s < n_scans; ++s) {
        auto meas = MakeBatch(accel, gyro, cloud, t);
        fx.imu->Process(meas, fx.kf, fx.pcl_out);
        t += kScanDt;
        if (s % log_every == 0 || s == n_scans - 1) {
            auto x = fx.kf.get_x();
            trace.t.push_back(t);
            trace.px.push_back(x.pos.x());
            trace.py.push_back(x.pos.y());
            trace.pz.push_back(x.pos.z());
            trace.vx.push_back(x.vel.x());
            trace.vy.push_back(x.vel.y());
            trace.vz.push_back(x.vel.z());
            trace.qnorm.push_back(x.rot.coeffs().norm());
        }
    }
    return trace;
}

void PrintTrace(const std::string& label, const DriftTrace& tr) {
    std::cerr << "\n" << label << "\n";
    std::cerr << "  t[s]    pos_x      pos_y      pos_z     |v|       |q|\n";
    for (size_t i = 0; i < tr.t.size(); ++i) {
        const double v = std::sqrt(tr.vx[i]*tr.vx[i] + tr.vy[i]*tr.vy[i] + tr.vz[i]*tr.vz[i]);
        std::cerr << "  " << tr.t[i] << "  "
                  << std::showpos << std::fixed
                  << std::setprecision(5) << tr.px[i] << "  "
                  << tr.py[i] << "  " << tr.pz[i] << "  "
                  << std::noshowpos << v << "  "
                  << std::setprecision(7) << tr.qnorm[i] << "\n";
    }
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// Test 1: FRAME INVARIANCE — confirm gravity axis doesn't affect drift.
// Expected: the drift magnitude (|pos|) at any given time is identical
// across all four orientations. If not, faster-lio has a frame-dependent
// bug. The earlier test showed they *are* identical to 7 s.f.
// ═════════════════════════════════════════════════════════════════════════
struct ImuMountCase { const char* name; Eigen::Vector3d accel; };

class FrameInvariance : public ::testing::TestWithParam<ImuMountCase> {};

TEST_P(FrameInvariance, StationaryInputOverLongRun) {
    const auto& cfg = GetParam();
    Fixture fx;
    auto tr = RunScans(fx, cfg.accel, Eigen::Vector3d::Zero(), 3000, 300);
    PrintTrace(cfg.name, tr);

    // Soft assertions just to print values; hard correctness check is in
    // the parameterized frame-equivariance test below.
    EXPECT_LT(tr.pz.back() * 0 + std::abs(tr.px.back()), 50.0);
}

INSTANTIATE_TEST_SUITE_P(
    All, FrameInvariance,
    ::testing::Values(
        ImuMountCase{"Z_up", Eigen::Vector3d(0, 0, kG)},
        ImuMountCase{"X_up", Eigen::Vector3d(kG, 0, 0)},
        ImuMountCase{"Y_up", Eigen::Vector3d(0, kG, 0)},
        ImuMountCase{"Z_down", Eigen::Vector3d(0, 0, -kG)}),
    [](const auto& info) { return info.param.name; });

// ═════════════════════════════════════════════════════════════════════════
// Test 2: DEGENERATE vs NON-DEGENERATE SCANS
// The original mount test used 20 collinear points — degenerate in Y and
// Z. This test pairs the same IMU input with (a) degenerate strip and
// (b) full 3D room. If drift drops with the room scan, scan observability
// is the dominant source of the ~2 cm/s floor we observed.
// ═════════════════════════════════════════════════════════════════════════
TEST(ScanObservability, DegenerateStrip_VS_Room) {
    auto strip = PointCloudType::Ptr(new PointCloudType());
    for (int i = 0; i < 20; ++i) {
        PointType p; p.x = 1.0f + 0.1f * i; p.y = 1.0f; p.z = 0.0f;
        p.intensity = 100.0f; p.curvature = i * 5.0f;
        strip->push_back(p);
    }
    auto room = MakeRoomScan(240);

    Fixture a, b;
    const Eigen::Vector3d acc(0, 0, kG);
    auto ta = RunScans(a, acc, Eigen::Vector3d::Zero(), 500, 100, strip);
    auto tb = RunScans(b, acc, Eigen::Vector3d::Zero(), 500, 100, room);
    PrintTrace("Strip scan (degenerate)", ta);
    PrintTrace("Room scan  (3D dense)  ", tb);

    const double drift_strip = std::hypot(ta.px.back(), ta.py.back());
    const double drift_room  = std::hypot(tb.px.back(), tb.py.back());
    std::cerr << "horizontal drift @ 50 s: strip=" << drift_strip
              << " m,  room=" << drift_room << " m\n";
}

// ═════════════════════════════════════════════════════════════════════════
// Test 3: Z-OBSERVABILITY in a room that LACKS floor/ceiling
// For a BPearl hemisphere indoors, floor returns are good but ceiling is
// sparse. Simulate that by removing the +Z wall from the room and see if
// Z drift explodes relative to a full room.
// ═════════════════════════════════════════════════════════════════════════
TEST(ScanObservability, MissingCeiling_CausesZDrift) {
    // Full room.
    auto full = MakeRoomScan(240);
    // Sans ceiling: rebuild manually.
    auto partial = PointCloudType::Ptr(new PointCloudType());
    for (const auto& p : full->points)
        if (!(p.z > 1.4f))  // drop the top wall
            partial->push_back(p);

    Fixture a, b;
    const Eigen::Vector3d acc(0, 0, kG);
    auto ta = RunScans(a, acc, Eigen::Vector3d::Zero(), 500, 100, full);
    auto tb = RunScans(b, acc, Eigen::Vector3d::Zero(), 500, 100, partial);
    PrintTrace("Full room       ", ta);
    PrintTrace("No-ceiling room ", tb);

    std::cerr << "|pz| @ 50 s: full=" << std::abs(ta.pz.back())
              << " m,  no-ceiling=" << std::abs(tb.pz.back()) << " m\n";
}

// ═════════════════════════════════════════════════════════════════════════
// Test 4: PURE YAW ROTATION
// User's observation: drift grows during turns. Simulate: stationary
// translation + yaw at 30°/s for 10 s. Check how much position drifts.
// Gravity is (0, 0, +G) (Z-up) so gravity compensation is "easy mode".
// ═════════════════════════════════════════════════════════════════════════
TEST(MotionCases, PureYawRotation_ShouldKeepPositionBounded) {
    Fixture fx;
    const Eigen::Vector3d acc(0, 0, kG);
    const Eigen::Vector3d gyr(0, 0, 30.0 * M_PI / 180.0);  // 30 deg/s yaw
    auto tr = RunScans(fx, acc, gyr, 100, 20);             // 10 s @ 10 Hz
    PrintTrace("Pure yaw @ 30 deg/s", tr);

    EXPECT_LT(std::abs(tr.pz.back()), 0.5)
        << "Pure yaw produced |pz|=" << std::abs(tr.pz.back()) << " m";
    EXPECT_LT(std::abs(tr.px.back()), 0.5);
    EXPECT_LT(std::abs(tr.py.back()), 0.5);
}

// ═════════════════════════════════════════════════════════════════════════
// Test 6: DOCUMENTATION of the G_m_s2 vs S2-length trade-off.
//
// faster-lio has a structural inconsistency between two constants:
//   common::G_m_s2          (currently 9.81)   — the IMU rescale target
//   MTK::S2<...,98090,...>  (length 9.8090)   — the filter's gravity magnitude
//
// The IMU preprocessor rescales every accel sample to magnitude G_m_s2
// (imu_processing.cc:175) but the filter holds gravity at S2's length,
// leaving a constant residual of (G_m_s2 - S2_length) on v_dot every
// step — ~0.001 m/s² with the current values, integrating to ~45 m of
// phantom drift per 300 s of stationary input.
//
// The "obvious" fix is to set G_m_s2 = 9.8090 (consistency). This test
// originally asserted that — and synthetic stationary drift does drop to
// near zero. But on the real Hilti site2 bag, G_m_s2 = 9.8090 causes
// catastrophic divergence (Z → 42 km, path → 364 km) at motion onset,
// while G_m_s2 = 9.81 keeps the bag in a bounded "37 m drift" regime.
// See common_lib.h for the full root-cause discussion.
//
// So this test is now DOCUMENTATION, not a regression guard: it records
// the synthetic-vs-bag mismatch with EXPECT_GT (drift > 0 with G=9.81 is
// EXPECTED, not a bug). The real fix is in laser_mapping.cc's outlier
// rejection, tracked separately.
// ═════════════════════════════════════════════════════════════════════════
TEST(GravityConstantTradeoff, StationaryDriftReflectsRescaleMismatch) {
    // With G_m_s2 = 9.81 and S2 length = 9.8090, a stationary input at
    // ANY magnitude is rescaled to 9.81 and compared against gravity 9.8090,
    // so a constant 0.001 m/s² residual integrates over time. The drift
    // is real and INTENTIONAL — it's the price we pay for keeping the bag
    // functional. Upper bound is the analytical worst case 0.5·0.001·t².
    for (double mag : {9.81, 9.80665, 9.8074, 9.8, 10.0}) {
        Fixture fx;
        auto tr = RunScans(fx, Eigen::Vector3d(0, 0, mag),
                            Eigen::Vector3d::Zero(), 500, 500);
        const double drift = std::abs(tr.pz.back());
        // Lower bound: drift MUST be > 0 with the current constants. If a
        // future change makes this zero, common_lib.h's G_m_s2 was probably
        // touched without re-running the bag — read the comment there first.
        EXPECT_GT(drift, 1e-6)
            << "accel=(0,0," << mag << "): drift went to zero — was "
            << "common::G_m_s2 changed to 9.8090? See common_lib.h.";
        // Upper bound: predicted worst case is 0.5·(G_m_s2 - S2_length)·t².
        // With t=50s and Δ=0.001 m/s², that's ~1.25 m. Cap generously at
        // 5 m so a regression that 10× the residual still trips this.
        EXPECT_LT(drift, 5.0)
            << "accel=(0,0," << mag << "): drift > 5 m — the residual is "
            << "much larger than (G_m_s2 - S2_length)·t²/2 predicts.";
    }
}

// ═════════════════════════════════════════════════════════════════════════
// Test 5: FRAME EQUIVARIANCE — the *shape* of the error should rotate
// with the mount. If accel=(0,0,g) gives drift ε_z, then accel=(g,0,0)
// should give drift of the same *magnitude* but on a different axis.
// Stronger than just matching norms — confirms per-axis correspondence.
// ═════════════════════════════════════════════════════════════════════════
TEST(FrameInvarianceStrict, TraceMagnitudesMatch) {
    Fixture za, xa, ya, zd;
    const int n = 500;
    auto az = RunScans(za, Eigen::Vector3d(0, 0, kG),   Eigen::Vector3d::Zero(), n, 100);
    auto ax = RunScans(xa, Eigen::Vector3d(kG, 0, 0),   Eigen::Vector3d::Zero(), n, 100);
    auto ay = RunScans(ya, Eigen::Vector3d(0, kG, 0),   Eigen::Vector3d::Zero(), n, 100);
    auto ad = RunScans(zd, Eigen::Vector3d(0, 0, -kG),  Eigen::Vector3d::Zero(), n, 100);

    auto norm = [](double x, double y, double z) { return std::sqrt(x*x + y*y + z*z); };
    const double nz = norm(az.px.back(), az.py.back(), az.pz.back());
    const double nx = norm(ax.px.back(), ax.py.back(), ax.pz.back());
    const double ny = norm(ay.px.back(), ay.py.back(), ay.pz.back());
    const double nd = norm(ad.px.back(), ad.py.back(), ad.pz.back());

    std::cerr << "|drift| after 50 s: Z_up=" << nz << "  X_up=" << nx
              << "  Y_up=" << ny << "  Z_down=" << nd << "\n";

    // All four should match to within numerical tolerance.
    EXPECT_NEAR(nx, nz, 1e-6);
    EXPECT_NEAR(ny, nz, 1e-6);
    EXPECT_NEAR(nd, nz, 1e-6);
}
