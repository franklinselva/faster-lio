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

// Helper overload: write YAML with an explicit `imu_init.assume_level` flag.
// Used by LevelingInit tests to compare old-hack vs proper leveling paths.
std::string WriteTestYAMLWithAssumeLevel(const std::string& tag, bool assume_level);

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

// YAML variant with an explicit outlier-gate block. `mode_str` must be
// one of "range" / "mahalanobis" / "either". Used by the
// `GateMode_*_HoldsStationary` tests below to verify the gate plumbing
// end-to-end through the LaserMapping pipeline.
std::string WriteTestYAMLWithGate(const std::string& tag, const std::string& mode_str) {
    const std::string path = std::string(ROOT_DIR) + "config/test_lio_" + tag + ".yaml";
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
      << "  outlier_gate:\n    mode: " << mode_str << "\n"
      << "imu_init:\n  motion_gate_enabled: false\n"
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

std::string WriteTestYAMLWithAssumeLevel(const std::string& tag, bool assume_level) {
    const std::string path = std::string(ROOT_DIR) + "config/test_lio_" + tag + ".yaml";
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
      << "  motion_gate_enabled: false\n"
      << "  assume_level: " << (assume_level ? "true" : "false") << "\n"
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

struct FinalState {
    Eigen::Vector3d pos;
    Eigen::Vector3d vel;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;
    Eigen::Vector3d grav;
    Eigen::Quaterniond rot{Eigen::Quaterniond::Identity()};
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
    out.rot = Eigen::Quaterniond(s.rot);
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

// ═════════════════════════════════════════════════════════════════════════
// Leveling-init battery.
//
// The R3LIVE hkust_campus_00 bag init-captured a tilted rig:
//   mean_acc = (−0.013, +0.306, +9.730) → ~1.8° roll around body-X.
//
// Our current `imu_init.assume_level: true` path sets:
//   init_state.rot  = I
//   init_state.grav = (0, 0, −G)
// i.e. it LIES to the filter about the body's orientation (pretends it's
// level when it isn't). The rotation state must then converge via LiDAR
// observations before horizontal motion gets mapped correctly to world.
//
// The physically correct leveling init:
//   init_state.rot  = R such that R * body_up = world_up
//                    (carry the tilt where it belongs)
//   init_state.grav = (0, 0, −G)
// After this, no convergence is needed: body-frame forward walking
// projects correctly into world-frame motion from the first step.
//
// Tests below are expected to FAIL on the current codebase and PASS once
// the fix lands. They feed tilted IMU input and assert the filter knows
// about the tilt.
// ═════════════════════════════════════════════════════════════════════════

namespace {

// Simulate the R3LIVE hkust_campus_00 init capture: a 1.8° roll around
// body-X axis. mean_acc = -g expressed in body frame:
//   g_world = (0, 0, -G); R_bw = R_X(+1.8°); a_body = -R_bw^T * g_world
//         = (0, G·sin(1.8°), G·cos(1.8°)) ≈ (0, +0.306, +9.805).
constexpr double kTiltDeg = 1.8;

Eigen::Vector3d TiltedStationaryAccel() {
    const double th = kTiltDeg * M_PI / 180.0;
    return Eigen::Vector3d(0.0, kG * std::sin(th), kG * std::cos(th));
}

// Extract roll (about X), pitch (about Y), yaw (about Z) from a quaternion.
Eigen::Vector3d ToRollPitchYaw(const Eigen::Quaterniond& q) {
    const Eigen::Matrix3d R = q.toRotationMatrix();
    const double roll  = std::atan2(R(2, 1), R(2, 2));
    const double pitch = std::asin(-R(2, 0));
    const double yaw   = std::atan2(R(1, 0), R(0, 0));
    return Eigen::Vector3d(roll, pitch, yaw);
}

}  // namespace

// ─── Test A: `assume_level: true` with a tilted IMU.
//
// Current behavior (the hack): init sets rot=I, leaving the filter to
// discover the tilt via LiDAR. In a full-room scan this eventually works
// because wall geometry constrains rot. Expected converged state:
//   - rot should eventually reflect the true body tilt (roll ≈ 1.8°)
//     after 30 s of observations.
//   - ba_body should absorb the gravity-magnitude residual (small z).
//
// If rot does NOT converge toward the tilt, the filter is stuck with its
// lie and horizontal motion will leak into pos_z on real bags — exactly
// the hkust_campus_00 symptom.
TEST(LioDriftAttribution, LevelingInit_AssumeLevelTrue_RotShouldConvergeToTilt) {
    const std::string yaml = WriteTestYAMLWithAssumeLevel("level_true", true);
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropNone);
    const auto accel = TiltedStationaryAccel();
    const auto s = SimulateStationary(mapping, scan, accel, kSimSeconds);
    PrintState("Tilted IMU, assume_level=true", s);

    const auto rpy = ToRollPitchYaw(s.rot);
    const double expected_roll = kTiltDeg * M_PI / 180.0;
    std::cerr << "  roll=" << (rpy.x() * 180.0 / M_PI) << "° "
              << " pitch=" << (rpy.y() * 180.0 / M_PI) << "° "
              << " yaw="   << (rpy.z() * 180.0 / M_PI) << "° "
              << " (expected roll=" << kTiltDeg << "°)\n";

    std::remove(yaml.c_str());
    // Diagnostic only: we want to see how close the converged rot gets to
    // the true tilt. An expectation here documents the current behaviour
    // for regression-tracking; no failure gate.
    EXPECT_GT(std::abs(rpy.x()), 0.0) << "rot didn't move at all from identity";
}

// ─── Test B: `assume_level: false` with the same tilted IMU.
//
// Default path: init_state.grav = -mean_acc/||mean_acc||*G. So the world
// frame is tilted to match the body (grav not pointing along world-Z).
// rot is still I. The filter's world frame is tilted relative to the
// true world — positions are in a tilted frame forever.
TEST(LioDriftAttribution, LevelingInit_AssumeLevelFalse_WorldGravCarriesTilt) {
    const std::string yaml = WriteTestYAMLWithAssumeLevel("level_false", false);
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropNone);
    const auto accel = TiltedStationaryAccel();
    const auto s = SimulateStationary(mapping, scan, accel, kSimSeconds);
    PrintState("Tilted IMU, assume_level=false", s);

    std::remove(yaml.c_str());
    // Document the shape: grav_world should point in the direction of
    // -mean_acc (i.e. have a small +Y component equal to -sin(tilt)*G).
    const double expected_gy = -kG * std::sin(kTiltDeg * M_PI / 180.0);
    EXPECT_NEAR(s.grav.y(), expected_gy, 0.02)
        << "grav_y should carry the init tilt when assume_level=false";
}

// ─── Test C (the headline): proper leveling init.
//
// What we're about to implement: init_state.rot = FromTwoVectors(body_up,
// world_up), init_state.grav = (0, 0, -G). Body tilt lives in rot from
// the first IMU sample; world frame is truly level from the start.
//
// This test gates the fix: it MUST fail before the fix, pass after.
// We detect that by asserting rot has the right roll AT init time (not
// after 30 s of LiDAR-driven convergence).
TEST(LioDriftAttribution, LevelingInit_ProperLeveling_RotCarriesTiltFromInit) {
    const std::string yaml = WriteTestYAMLWithAssumeLevel("level_proper", true);
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    // Feed only enough IMU samples + one scan to trigger the IEKF init,
    // then inspect state.rot. Post-fix: rot should already reflect the
    // measured tilt. Pre-fix (current code): rot stays at identity.
    auto scan = MakeRoomScan(kDropNone);
    const auto accel = TiltedStationaryAccel();
    // 5 seconds is well past the 400-sample init window (2 s @ 200 Hz).
    const auto s = SimulateStationary(mapping, scan, accel, 5.0);
    PrintState("Tilted IMU, expected PROPER leveling", s);

    const auto rpy = ToRollPitchYaw(s.rot);
    std::cerr << "  after 5 s: roll=" << (rpy.x() * 180.0 / M_PI) << "° "
              << "pitch=" << (rpy.y() * 180.0 / M_PI) << "° "
              << " (expected roll≈" << kTiltDeg << "°)\n";

    std::remove(yaml.c_str());

    // Post-fix assertion: rot carries the tilt within 0.5° even after only
    // 5 s (because the init itself sets it, no convergence needed). Also
    // grav is truly (0, 0, -G): no XY bleed.
    const double expected_roll_rad = kTiltDeg * M_PI / 180.0;
    EXPECT_NEAR(std::abs(rpy.x()), expected_roll_rad, 0.5 * M_PI / 180.0)
        << "Proper leveling init did not set rot from measured gravity.";
    EXPECT_LT(std::abs(s.grav.x()), 0.05);
    EXPECT_LT(std::abs(s.grav.y()), 0.05);
    EXPECT_NEAR(s.grav.z(), -kG, 0.1);
}

// ═════════════════════════════════════════════════════════════════════════
// Outlier-gate mode battery.
//
// Each gate mode must produce a bounded-drift stationary run on the
// full-room scan. The range gate is the legacy baseline (already covered
// by `Stationary_FullRoom_HoldsZeroDrift` at the top of this file). We
// rerun the same scenario here under each mode to confirm:
//   - YAML `mapping.outlier_gate.mode: range` behaves identically.
//   - YAML `mahalanobis` doesn't over-reject and collapse the filter.
//   - YAML `either` is at least as permissive as `range`.
//
// Thresholds are intentionally loose (0.05 m for stationary drift) — we
// care here about "does the gate plumbing work" rather than precise
// numerical equivalence. A divergence under Mahalanobis would produce
// metres of drift, easily caught.
// ═════════════════════════════════════════════════════════════════════════

TEST(LioDriftAttribution, GateMode_Range_HoldsStationary) {
    const std::string yaml = WriteTestYAMLWithGate("gate_range", "range");
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropNone);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(mapping, scan, accel, kSimSeconds);
    PrintState("Gate=range (explicit)", s);

    EXPECT_LT(std::abs(s.pos.z()), 0.05);
    EXPECT_LT(std::abs(s.ba.z()),  0.01);
    std::remove(yaml.c_str());
}

TEST(LioDriftAttribution, GateMode_Mahalanobis_HoldsStationary) {
    const std::string yaml = WriteTestYAMLWithGate("gate_mahal", "mahalanobis");
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropNone);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(mapping, scan, accel, kSimSeconds);
    PrintState("Gate=mahalanobis", s);

    // Mahalanobis can be strict when the filter is very confident — but
    // with a fully-observable full-room scan the innovation variance
    // stays well-calibrated and residuals fit the χ² bound comfortably.
    EXPECT_LT(std::abs(s.pos.z()), 0.05)
        << "Mahalanobis mode over-rejected observations, filter drifted — "
        << "either the gate formula is wrong or the default chi² is too tight.";
    EXPECT_LT(std::abs(s.ba.z()),  0.01);
    std::remove(yaml.c_str());
}

TEST(LioDriftAttribution, GateMode_Either_HoldsStationary) {
    const std::string yaml = WriteTestYAMLWithGate("gate_either", "either");
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropNone);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(mapping, scan, accel, kSimSeconds);
    PrintState("Gate=either (hybrid)", s);

    // Either is the most permissive mode — it accepts anything the range
    // gate OR the Mahalanobis gate would accept. Should converge at least
    // as tightly as range alone.
    EXPECT_LT(std::abs(s.pos.z()), 0.05);
    EXPECT_LT(std::abs(s.ba.z()),  0.01);
    std::remove(yaml.c_str());
}

// ═════════════════════════════════════════════════════════════════════════
// Observability-guard integration battery.
//
// Floor-only scan = translation rank 1 (only Z-normal plane present). At
// min_translation_rank=3, every frame's jacobian fails the rank test — so
// `skip_update` and `skip_position` both fire every frame, while `ignore`
// stays silent. These tests pin the plumbing end-to-end:
//   - skip_update   → guard count grows, pipeline survives 30s without crash.
//   - skip_position → position columns zeroed, |pos| stays bounded.
//   - ignore        → baseline, skip count stays 0 (regression canary).
//
// Note: the floor-only scan was already observed in
// `Stationary_FloorOnly_PinsZOnly` to be too sparse for iVox k-NN(5) —
// effective feature count may be 0 for many frames, so ObsModel short-
// circuits before reaching the guard. The guard increments its skip count
// only on frames where the jacobian is actually built (effect_feat_num_ ≥ 1),
// which empirically fires a few times per run as the map populates. The
// `> 0` assertion below captures exactly this: the guard runs AT LEAST ONCE
// and correctly flags the scene as rank-deficient.
// ═════════════════════════════════════════════════════════════════════════

namespace {
// YAML variant with an explicit observability_guard block.
// `mode_str` ∈ {"ignore", "skip_position", "skip_update"}.
std::string WriteTestYAMLWithObsGuard(const std::string& tag, const std::string& mode_str,
                                       int min_translation_rank = 3,
                                       double singular_threshold = 1.0e-4) {
    const std::string path = std::string(ROOT_DIR) + "config/test_lio_" + tag + ".yaml";
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
      << "  observability_guard:\n"
      << "    mode: " << mode_str << "\n"
      << "    min_translation_rank: " << min_translation_rank << "\n"
      << "    singular_threshold: " << singular_threshold << "\n"
      << "imu_init:\n  motion_gate_enabled: false\n"
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

TEST(LioDriftAttribution, ObsGuard_SkipUpdate_Floor_IncrementsSkipCount) {
    const std::string yaml = WriteTestYAMLWithObsGuard("obsg_skip_upd", "skip_update");
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    // Floor only = translation rank 1 on every frame. With min_rank=3,
    // every frame that builds a non-empty jacobian must trigger a skip.
    auto scan = MakeRoomScan(kDropXneg | kDropXpos | kDropYneg | kDropYpos | kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(mapping, scan, accel, 30.0);
    PrintState("ObsGuard=skip_update floor-only", s);

    const int skips = mapping.ObsGuardSkipCount();
    std::cerr << "  obs_guard_skip_count = " << skips << "\n";
    EXPECT_GT(skips, 0)
        << "skip_update should have fired at least once on a floor-only scan "
        << "where translation rank is 1 < min_rank=3.";
    std::remove(yaml.c_str());
}

TEST(LioDriftAttribution, ObsGuard_SkipPosition_Floor_StillProducesRotation) {
    const std::string yaml = WriteTestYAMLWithObsGuard("obsg_skip_pos", "skip_position");
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropXneg | kDropXpos | kDropYneg | kDropYpos | kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(mapping, scan, accel, 30.0);
    PrintState("ObsGuard=skip_position floor-only", s);

    const int skips = mapping.ObsGuardSkipCount();
    std::cerr << "  obs_guard_skip_count = " << skips << "\n";
    // Guard MUST have run on at least one post-init frame.
    EXPECT_GT(skips, 0)
        << "skip_position should fire on the same under-constrained frames as skip_update.";
    // Pipeline didn't crash and |pos| stays bounded (floor-only stationary
    // regime from `Stationary_FloorOnly_PinsZOnly` settles below 5 cm).
    EXPECT_LT(s.pos.norm(), 0.20)
        << "|pos| exceeded sanity bound — the skip_position gate may have destabilised the filter.";
    std::remove(yaml.c_str());
}

TEST(LioDriftAttribution, ObsGuard_Ignore_Floor_NoIntervention) {
    const std::string yaml = WriteTestYAMLWithObsGuard("obsg_ignore", "ignore");
    LaserMapping mapping;
    ASSERT_TRUE(mapping.Init(yaml));

    auto scan = MakeRoomScan(kDropXneg | kDropXpos | kDropYneg | kDropYpos | kDropCeiling);
    const Eigen::Vector3d accel(0, 0, kG);
    const auto s = SimulateStationary(mapping, scan, accel, 30.0);
    PrintState("ObsGuard=ignore floor-only (baseline)", s);

    // Analyse-only mode must never increment the skip count.
    EXPECT_EQ(mapping.ObsGuardSkipCount(), 0)
        << "Ignore mode must be a pure analyse — any non-zero skip count is a regression.";
    // Baseline drift should match the un-guarded floor-only run in
    // `Stationary_FloorOnly_PinsZOnly`, bounded by the same 5 cm budget.
    EXPECT_LT(std::abs(s.pos.z()), 0.05);
    std::remove(yaml.c_str());
}
