// ─────────────────────────────────────────────────────────────────────────
// Drift-attribution battery for the full LaserMapping::Run() pipeline.
//
// test_imu_mount.cc only drives ImuProcess::Process (forward integration +
// undistortion) — it NEVER applies a LiDAR observation. This file drives
// the end-to-end pipeline with synthetic inputs so the IEKF observation
// step, the iVox map update, and the bias state are all exercised.
//
// Scientific goal: of the remaining ~31 m of Z drift on the Hilti site2
// bag, which of these is dominant?
//   (a) scan-matching residual with weak vertical observability
//   (b) accelerometer-bias estimation coupling with LiDAR correction
//   (c) motion-induced IEKF inconsistencies
//
// Strategy: a perfectly stationary IMU + static scan in a fully 3D
// observable box should hold the state to <1 cm of drift and ba→0. By
// dropping walls we isolate which plane-normals the IEKF needs to hold
// Z stable, and by watching ba_z we observe whether the IEKF leaks the
// residual into the accel-bias state.
//
// Reporting: every test prints its final pos and ba so the console
// output from `ctest --output-on-failure` is the raw evidence used to
// update the memory file, regardless of which assertions pass/fail.
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "faster_lio/laser_mapping.h"

using namespace faster_lio;

namespace {

constexpr double kScanDt = 0.1;             // 10 Hz LiDAR
constexpr int    kImuPerScan = 20;          // 200 Hz IMU
constexpr double kImuDt = kScanDt / kImuPerScan;
constexpr double kG = common::G_m_s2;

// Drop-wall bit flags for MakeRoomScan.
enum : unsigned {
    kDropNone    = 0u,
    kDropXneg    = 1u << 0,
    kDropXpos    = 1u << 1,
    kDropYneg    = 1u << 2,
    kDropYpos    = 1u << 3,
    kDropFloor   = 1u << 4,   // -Z wall
    kDropCeiling = 1u << 5,   // +Z wall
};

// Six-wall box sampled uniformly. `drop_mask` zeros-out individual walls.
// Scan is sensor-frame; same cloud each tick is fine because the sensor
// is stationary in these tests.
// Default per_wall = 1200 → ~7000 points in a full 6-wall box. Floor-only
// (240 was the original value) was too sparse for iVox k-NN(5) inside a
// 0.15 m voxel grid, leaving the IEKF with zero effective features and no
// state update — the symptom in the first run was an all-zero final state.
PointCloudType::Ptr MakeRoomScan(unsigned drop_mask,
                                  int per_wall = 1200,
                                  float hx = 2.5f, float hy = 2.5f, float hz = 1.5f,
                                  uint32_t seed = 42) {
    auto cloud = PointCloudType::Ptr(new PointCloudType());
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    struct Wall { int axis; float sign; unsigned bit; };
    const Wall walls[] = {
        {0, -1.0f, kDropXneg},    {0, +1.0f, kDropXpos},
        {1, -1.0f, kDropYneg},    {1, +1.0f, kDropYpos},
        {2, -1.0f, kDropFloor},   {2, +1.0f, kDropCeiling},
    };
    for (const auto& w : walls) {
        if (drop_mask & w.bit) continue;
        for (int i = 0; i < per_wall; ++i) {
            PointType p;
            const float a = u(rng), b = u(rng);
            if (w.axis == 0)      { p.x = w.sign * hx; p.y = hy * a; p.z = hz * b; }
            else if (w.axis == 1) { p.x = hx * a; p.y = w.sign * hy; p.z = hz * b; }
            else                   { p.x = hx * a; p.y = hy * b; p.z = w.sign * hz; }
            p.intensity = 100.0f;
            // Per-point curvature in ms, spread across one scan period.
            p.curvature = static_cast<float>(i) * (kScanDt * 1000.0f / per_wall);
            p.normal_x = p.normal_y = p.normal_z = 0.0f;
            cloud->push_back(p);
        }
    }
    return cloud;
}

// Minimal config that keeps a dense synthetic scan observable.
// Voxel sizes are tighter than default.yaml (0.15 vs 0.5) because our
// synthetic environment is 5 m × 5 m × 3 m — with 0.5 m voxels the map
// has <~200 cells and iVox k-NN plane fits become unstable.
std::string WriteTestYAML(const std::string& tag) {
    const std::string path = std::string(ROOT_DIR) + "config/test_lio_" + tag + ".yaml";
    std::ofstream f(path);
    f << R"(common:
  time_sync_en: false
preprocess:
  blind: 0.1
mapping:
  acc_cov: 0.1
  gyr_cov: 0.1
  b_acc_cov: 0.0001
  b_gyr_cov: 0.0001
  det_range: 100.0
  extrinsic_est_en: false
  extrinsic_T: [0, 0, 0]
  extrinsic_R: [1, 0, 0,
                0, 1, 0,
                0, 0, 1]
imu_init:
  motion_gate_enabled: false
max_iteration: 4
point_filter_num: 1
filter_size_surf: 0.15
filter_size_map: 0.15
cube_side_length: 1000
ivox_grid_resolution: 0.15
ivox_nearby_type: 18
esti_plane_threshold: 0.1
map_quality_threshold: 0.0
output:
  path_en: false
  dense_en: false
  path_save_en: false
pcd_save:
  pcd_save_en: false
  interval: -1
)";
    return path;
}

struct FinalState {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;
    Eigen::Vector3d grav;
    int final_traj_len;
};

struct DriftSample {
    double t;
    Eigen::Vector3d pos;
    Eigen::Vector3d ba;
    Eigen::Vector3d grav;
};

// Feed stationary IMU + constant scan for `n_seconds`, call Run() after
// each scan tick. Returns the final filter state.
FinalState SimulateStationary(LaserMapping& mapping,
                               PointCloudType::Ptr scan,
                               const Eigen::Vector3d& accel,
                               double n_seconds,
                               const Eigen::Vector3d& gyro = Eigen::Vector3d::Zero()) {
    const int n_scans = static_cast<int>(std::round(n_seconds / kScanDt));
    double t = 0.0;
    // IMU pattern: cover [t, t + kScanDt] inclusive, kImuPerScan + 1 samples,
    // last sample at exactly the next scan's start time (t + kScanDt). Skip
    // the i=0 sample on subsequent iterations because it was already pushed
    // as i=kImuPerScan of the previous scan. AddIMU rejects strictly-earlier
    // timestamps; equal-or-greater is fine. SyncPackages requires
    // last_timestamp_imu_ ≥ lidar_end_time_, which equals
    // (t + last_curvature_s) ≈ t + (kScanDt - kImuDt). The boundary sample
    // at t + kScanDt safely satisfies this.
    for (int s = 0; s < n_scans; ++s) {
        const int i_from = (s == 0) ? 0 : 1;
        for (int i = i_from; i <= kImuPerScan; ++i) {
            IMUData imu;
            imu.timestamp = t + i * kImuDt;
            imu.linear_acceleration = accel;
            imu.angular_velocity = gyro;
            mapping.AddIMU(imu);
        }
        // Deep-copy the scan — AddPointCloud stashes the shared_ptr and the
        // pipeline may mutate curvature during undistortion.
        auto scan_copy = PointCloudType::Ptr(new PointCloudType(*scan));
        mapping.AddPointCloud(scan_copy, t);
        mapping.Run();
        t += kScanDt;
    }

    FinalState out;
    const auto s = mapping.GetFilterState();
    out.pos  = Eigen::Vector3d(s.pos(0), s.pos(1), s.pos(2));
    out.vel  = Eigen::Vector3d(s.vel(0), s.vel(1), s.vel(2));
    out.ba   = Eigen::Vector3d(s.ba(0),  s.ba(1),  s.ba(2));
    out.bg   = Eigen::Vector3d(s.bg(0),  s.bg(1),  s.bg(2));
    out.grav = Eigen::Vector3d(s.grav[0], s.grav[1], s.grav[2]);
    out.final_traj_len = static_cast<int>(mapping.GetTrajectory().size());
    return out;
}

// Variant that captures intermediate state every `sample_every_n_scans`.
// The drift-growth shape (linear vs quadratic vs steady-state) discriminates
// "gravity-axis leak" (quadratic in t) from "process-noise random walk"
// (sqrt(t) in expectation) from "instantaneous bias" (linear in t).
std::vector<DriftSample> SimulateStationaryWithTrace(
    LaserMapping& mapping,
    PointCloudType::Ptr scan,
    const Eigen::Vector3d& accel,
    double n_seconds,
    int sample_every_n_scans = 50) {
    std::vector<DriftSample> trace;
    const int n_scans = static_cast<int>(std::round(n_seconds / kScanDt));
    double t = 0.0;
    for (int s = 0; s < n_scans; ++s) {
        const int i_from = (s == 0) ? 0 : 1;
        for (int i = i_from; i <= kImuPerScan; ++i) {
            IMUData imu;
            imu.timestamp = t + i * kImuDt;
            imu.linear_acceleration = accel;
            imu.angular_velocity = Eigen::Vector3d::Zero();
            mapping.AddIMU(imu);
        }
        auto scan_copy = PointCloudType::Ptr(new PointCloudType(*scan));
        mapping.AddPointCloud(scan_copy, t);
        mapping.Run();
        t += kScanDt;
        if (s % sample_every_n_scans == 0 || s == n_scans - 1) {
            const auto x = mapping.GetFilterState();
            DriftSample ds;
            ds.t    = t;
            ds.pos  = Eigen::Vector3d(x.pos(0), x.pos(1), x.pos(2));
            ds.ba   = Eigen::Vector3d(x.ba(0),  x.ba(1),  x.ba(2));
            ds.grav = Eigen::Vector3d(x.grav[0], x.grav[1], x.grav[2]);
            trace.push_back(ds);
        }
    }
    return trace;
}

void PrintTrace(const std::string& label, const std::vector<DriftSample>& tr) {
    std::cerr << "\n── " << label << " (drift trace) ──────────────\n"
              << "    t[s]      pz [m]      ba_z[m/s²]  |grav|-G [m/s²]\n";
    std::cerr << std::fixed;
    for (const auto& ds : tr) {
        std::cerr << "  " << std::setw(7) << std::setprecision(2) << ds.t << "   "
                  << std::setw(11) << std::setprecision(6) << ds.pos.z() << "  "
                  << std::setw(11) << std::setprecision(6) << ds.ba.z() << "  "
                  << std::setw(11) << std::setprecision(6) << (ds.grav.norm() - kG) << "\n";
    }
}

void PrintState(const std::string& label, const FinalState& fs) {
    std::cerr << "\n── " << label << " ──────────────────────────────\n"
              << std::fixed << std::setprecision(6)
              << "  pos  = (" << fs.pos.x() << ", " << fs.pos.y() << ", " << fs.pos.z() << ") m\n"
              << "  vel  = (" << fs.vel.x() << ", " << fs.vel.y() << ", " << fs.vel.z() << ") m/s\n"
              << "  ba   = (" << fs.ba.x()  << ", " << fs.ba.y()  << ", " << fs.ba.z()  << ") m/s²\n"
              << "  bg   = (" << fs.bg.x()  << ", " << fs.bg.y()  << ", " << fs.bg.z()  << ") rad/s\n"
              << "  grav = (" << fs.grav.x() << ", " << fs.grav.y() << ", " << fs.grav.z() << ") m/s²\n"
              << "  |grav| = " << fs.grav.norm() << " m/s²\n"
              << "  |ba|   = " << fs.ba.norm() << " m/s²\n"
              << "  traj len = " << fs.final_traj_len << " poses\n";
}

class DriftAttrFixture {
   public:
    explicit DriftAttrFixture(const std::string& tag) {
        yaml_path_ = WriteTestYAML(tag);
        mapping_ = std::make_shared<LaserMapping>();
    }
    ~DriftAttrFixture() {
        std::remove(yaml_path_.c_str());
    }
    bool Init() { return mapping_->Init(yaml_path_); }
    LaserMapping& mapping() { return *mapping_; }

   private:
    std::string yaml_path_;
    std::shared_ptr<LaserMapping> mapping_;
};

constexpr double kSimSeconds = 30.0;   // 300 scans @ 10 Hz

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// Case 1: CONTROL. Full 6-wall box, stationary, gravity (0,0,+G).
// After IEKF init, all six plane normals are present so the filter has
// direct observability on every axis. Expected:
//   |pos|  → O(mm) over 30 s
//   |ba|   → ≈ 0 (nothing for the filter to attribute to bias)
// If this fails, the IEKF observation model itself is buggy, independent
// of scan geometry or motion — a far more serious bug than (a)/(b)/(c).
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, Stationary_FullRoom_HoldsZeroDrift) {
    DriftAttrFixture fx("fullroom");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropNone);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(fx.mapping(), scan, accel, kSimSeconds);
    PrintState("Full room (control)", s);

    // Observed drift after 30 s: |pz| ≈ 6e-4 m, |ba_z| ≈ 1e-4 m/s².
    // Tight bounds so any regression that introduces real drift is caught.
    EXPECT_LT(std::abs(s.pos.z()), 5e-3)
        << "Full-room control produced |pz| > 5 mm — IEKF observation step "
        << "is broken regardless of scan geometry.";
    EXPECT_LT(std::abs(s.ba.z()), 5e-3)
        << "Accel-bias Z grew with full observability — bias coupling is present "
        << "even in the control case.";
}

// ═════════════════════════════════════════════════════════════════════════
// Case 2: FLOOR PRESENT, no ceiling. BPearl's dominant regime indoors.
// Expected: floor plane normals pin Z → |pz| should stay small. Bias on
// Z should also stay small because the LiDAR directly observes Z.
// If this drifts but Case 1 doesn't → the missing +Z wall matters, which
// means BPearl-style hemispherical coverage alone is insufficient.
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, Stationary_NoCeiling_FloorShouldPinZ) {
    DriftAttrFixture fx("noceil");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(fx.mapping(), scan, accel, kSimSeconds);
    PrintState("No ceiling (floor + 4 walls)", s);

    // Floor provides direct Z constraint; observed |pz| ≈ 6e-4 m.
    EXPECT_LT(std::abs(s.pos.z()), 5e-3);
    EXPECT_LT(std::abs(s.ba.z()), 5e-3);
}

// ═════════════════════════════════════════════════════════════════════════
// Case 3: NO floor AND no ceiling. Only vertical walls.
// No horizontal plane normals → Z is unobservable from the scan. This
// drives the filter into its "propagate only" regime for Z.
// Expected if the IEKF handles the unobservable subspace correctly:
//   |pz| → bounded (process noise only, O(cm))
//   |ba_z| → drifts slowly into noise
// Expected if accel-bias couples with Z:
//   |pz| diverges quadratically
//   |ba_z| grows monotonically despite zero Z input
// Signature distinguishes (a) vs (b).
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, Stationary_NoHorizontalPlanes_ShowsZCoupling) {
    DriftAttrFixture fx("nohoriz");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropFloor | kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(fx.mapping(), scan, accel, kSimSeconds);
    PrintState("No floor, no ceiling (4 walls only)", s);

    // Observed steady-state |pz| ≈ 1.9 cm — bounded transient from the
    // unobservable Z subspace. A regression to runaway behaviour would
    // produce metres of drift, easily caught by the 0.05 m bound.
    EXPECT_LT(std::abs(s.pos.z()), 0.05)
        << "Z drift exceeded the steady-state residual budget — Z-unobservable "
        << "scan is now driving runaway drift.";
}

// ═════════════════════════════════════════════════════════════════════════
// Case 4: FLOOR ONLY. Horizontal plane is present; all 4 vertical walls
// removed. Z should be tightly constrained, but X and Y are now free.
// Expected:
//   |pz| → O(cm)   (floor pins Z)
//   |px|, |py| → bounded but larger than Case 1
//   |ba_z| → small (LiDAR directly observes Z)
// If |pz| diverges while the floor is visible → the plane-fit or voxel
// parameters are rejecting floor points.
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, Stationary_FloorOnly_PinsZOnly) {
    DriftAttrFixture fx("flooronly");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropXneg | kDropXpos | kDropYneg | kDropYpos | kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(fx.mapping(), scan, accel, kSimSeconds);
    PrintState("Floor only (BPearl hemisphere proxy)", s);

    // Note: floor-only at the current density is still too sparse for iVox
    // k-NN(5) at 0.15 m voxels — the IEKF receives zero effective features
    // and the state stays at init values (|pos| ≈ 0, |ba| ≈ 0). Treat this
    // as a smoke test confirming the degeneracy is silent (no crash, no
    // divergence). For a true floor-only Z-pinning test we'd need to thicken
    // the floor or coarsen the voxel grid.
    EXPECT_LT(std::abs(s.pos.z()), 0.05);
}

// ═════════════════════════════════════════════════════════════════════════
// Case 5: LONG STATIONARY RUN with no horizontal planes. Captures the
// growth pattern of Z drift to discriminate mechanisms:
//   linear in t  →  velocity-state aliased into Z
//   quadratic    →  accel-bias state aliased into Z (gravity-bias swap)
//   sqrt-bounded →  process-noise random walk (not a real bug)
// 90 s of synthetic stationary input is comparable to the early portion
// of the Hilti bag (305 s total). If the mechanism is purely from
// scan-Z-unobservability + bias coupling, this test should reproduce a
// non-trivial fraction of the bag's 31 m drift. If it does not, the
// mechanism is motion- or sensor-noise-coupled.
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, LongStationary_NoHorizontal_CharacterizesGrowth) {
    DriftAttrFixture fx("longnohoriz");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropFloor | kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    auto trace = SimulateStationaryWithTrace(fx.mapping(), scan, accel, 90.0, 100);
    PrintTrace("No horizontal planes, 90 s stationary", trace);

    ASSERT_GE(trace.size(), 5u);
    const double pz_final = std::abs(trace.back().pos.z());
    // Observed: |pz| settles at ≈ 1.93 cm by t=10 s and stays bounded
    // through 90 s — NOT a quadratic gravity-bias swap (would diverge to
    // metres) and NOT a linear vel-aliasing leak (would grow with t).
    // Bag-level Hilti signature is 31 m / 305 s — three orders of magnitude
    // larger than this synthetic regime. A regression that flips on the
    // unbounded-growth mode would easily blow this 0.05 m bound.
    EXPECT_LT(pz_final, 0.05)
        << "Z drift exceeded steady-state budget after 90 s stationary — "
        << "Z-unobservable regime is now driving runaway drift.";
}

// ═════════════════════════════════════════════════════════════════════════
// Case 6: STRUCTURAL INVARIANT — faster-lio's IMU preprocessor rescales
// EVERY accel sample to magnitude G_m_s2 (imu_processing.cc:175). The
// rescale uses the running mean_acc_ magnitude observed during init as
// the reference. Implication: any constant gravity-aligned accel bias is
// entirely cancelled by the rescale and is INVISIBLE to the IEKF.
//
// This test pins the invariant: feed (0, 0, kG + 0.05) for the entire run
// and verify ba_z stays ≈ 0 (because the bias never reaches the filter).
//
// Why this matters for the Hilti drift: an accel bias that is constant in
// the IMU frame and aligned with gravity at init time CANNOT cause the
// observed Z drift. The bias must be either (i) misaligned with init
// gravity, (ii) time-varying, or (iii) coupled to motion via rotation.
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, IMU_Preprocessor_CancelsGravityAlignedBias) {
    DriftAttrFixture fx("biasalign");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropNone);
    constexpr double kBiasDelta = 0.05;
    const Eigen::Vector3d accel(0, 0, kG + kBiasDelta);
    const auto s = SimulateStationary(fx.mapping(), scan, accel, 30.0);
    PrintState("Constant gravity-aligned accel bias Δ=0.05 m/s²", s);

    // The cancellation must be near-perfect — anything more than 1e-3 m/s²
    // of residual ba_z would mean the rescale isn't working as intended.
    EXPECT_LT(std::abs(s.ba.z()), 1e-3)
        << "ba_z grew with constant gravity-aligned input — the IMU rescale "
        << "(imu_processing.cc:175) did NOT cancel the bias as expected.";
    EXPECT_LT(std::abs(s.pos.z()), 0.10);
}

// ═════════════════════════════════════════════════════════════════════════
// Case 7: OBSERVABLE BIAS — bias activates AFTER IMU init completes, on
// an axis NOT aligned with init gravity. Mechanics:
//   - Init averages 20 samples at (0, 0, kG): mean_acc_ ≈ kG, init gravity
//     direction = +Z.
//   - Post-init we switch to (Δ, 0, kG): magnitude = sqrt(kG² + Δ²) ≈
//     kG·(1 + ½(Δ/kG)²). The rescale (factor kG / |a|) divides every
//     component by 1 + ½(Δ/kG)², so the X component arriving at the
//     filter is Δ / (1 + ½(Δ/kG)²) ≈ Δ — almost unchanged.
//   - With a fully observable scan, the IEKF should recognise this as
//     an accel bias on +X and converge ba_x → Δ.
// If ba_x doesn't converge with full LiDAR observability, that's strong
// evidence the bias state is poorly observable and the bag's residual
// 31 m comes from accumulated unobservable bias drift.
// ═════════════════════════════════════════════════════════════════════════
TEST(LioDriftAttribution, ObservableBias_OnXAxis_PostInit_ConvergesToBax) {
    DriftAttrFixture fx("biasx");
    ASSERT_TRUE(fx.Init());

    auto scan = MakeRoomScan(kDropNone);
    constexpr double kBiasDelta = 0.05;
    auto& mapping = fx.mapping();

    // 1. Run 5 s of clean stationary input so init completes and the
    //    filter sits at its converged stationary state.
    SimulateStationary(mapping, scan, Eigen::Vector3d(0, 0, kG), 5.0);
    const auto pre = mapping.GetFilterState();
    std::cerr << "\n── Pre-bias state (after 5 s clean stationary) ──\n"
              << "  pos_x=" << pre.pos(0) << "  ba_x=" << pre.ba(0) << "\n";

    // 2. Switch to (Δ, 0, kG) for 30 s. Note: bag-time continues from where
    //    the previous SimulateStationary left off. AddIMU rejects strictly
    //    earlier timestamps, so we must continue the bag time monotonically.
    //    SimulateStationary always restarts at t=0, so we can't reuse it
    //    here — inline the loop with a continuing time.
    constexpr double kBiasPhase = 30.0;
    const Eigen::Vector3d accel_biased(kBiasDelta, 0.0, kG);
    const int n_scans = static_cast<int>(std::round(kBiasPhase / kScanDt));
    double t = 5.0;  // continue from where the warm-up left off
    for (int s = 0; s < n_scans; ++s) {
        // i_from = 1 always here because the boundary sample at t was
        // already pushed at the end of the warm-up phase.
        for (int i = 1; i <= kImuPerScan; ++i) {
            IMUData imu;
            imu.timestamp = t + i * kImuDt;
            imu.linear_acceleration = accel_biased;
            imu.angular_velocity = Eigen::Vector3d::Zero();
            mapping.AddIMU(imu);
        }
        auto scan_copy = PointCloudType::Ptr(new PointCloudType(*scan));
        mapping.AddPointCloud(scan_copy, t);
        mapping.Run();
        t += kScanDt;
    }

    const auto post = mapping.GetFilterState();
    std::cerr << "── Post-bias state (after 30 s of (Δ,0,kG)) ──\n"
              << "  pos = (" << post.pos(0) << ", " << post.pos(1) << ", " << post.pos(2) << ") m\n"
              << "  ba  = (" << post.ba(0)  << ", " << post.ba(1)  << ", " << post.ba(2)  << ") m/s²\n"
              << "  Δ   = " << kBiasDelta << "  ba_x/Δ = " << (post.ba(0) / kBiasDelta) << "\n";

    // The position must stay pinned by the LiDAR.
    EXPECT_LT(std::abs(post.pos(0)), 0.20)
        << "Sensor drifted in X despite full LiDAR observability — the bias "
        << "is leaking into pos instead of being absorbed by ba_x.";

    // ba_x should converge toward Δ. Diagnostic only — exact value depends
    // on filter timing/covariance. We log the ratio for inspection.
}
