#ifndef FASTER_LIO_LASER_MAPPING_CONFIG_H
#define FASTER_LIO_LASER_MAPPING_CONFIG_H

// YAML → config struct for LaserMapping. Lives in its own translation unit
// so parsing can be unit-tested without constructing a full LaserMapping
// instance, and so laser_mapping.cc stays focused on the algorithm.
//
// Design:
//   - Per-subtree settings structs mirror the YAML layout.
//   - Struct defaults reproduce upstream FAST-LIO behaviour byte-for-byte
//     where applicable.
//   - The parser resolves back-compat rules (process_noise ↔ mapping.*_cov,
//     min_time_s ↔ min_accepted) and populates the FINAL values — the
//     apply side is dumb.
//   - Absent optional blocks (`pose_graph`, `loop_closure`, `wheel`,
//     `imu_init.init_P_diag`) are represented by `enabled=false` / empty
//     `std::optional<>` so callers can skip work.

#include <yaml-cpp/yaml.h>

#include <optional>
#include <string>
#include <vector>

#include "faster_lio/imu_processing.h"   // InitPDiag, DEFAULT_INIT_GATE_*
#include "faster_lio/loop_closer.h"      // LoopCloser::Options
#include "faster_lio/observability_guard.h"  // ObservabilityGuardMode
#include "faster_lio/pose_graph.h"       // PoseGraph::Options

namespace faster_lio {

// ── Per-subtree settings ───────────────────────────────────────────────

struct CommonSettings {
    bool time_sync_en = false;
};

// Raw preprocessing knobs (blind distance, per-point stride).
struct PreprocessSettings {
    double blind = 0.5;
    int    point_filter_num = 1;
};

// Outlier-rejection gate for LiDAR point-to-plane observations in the
// IEKF update. See `outlier_gate.h` for the gate functions themselves.
//   kRange       — legacy FAST-LIO heuristic (range > ratio × residual²).
//   kMahalanobis — χ² gate on residual² / (HᵀPH + R); covariance-aware
//                  but tight when the filter is over-confident.
//   kEither      — accept if EITHER gate accepts; more permissive.
enum class OutlierGateMode { kRange, kMahalanobis, kEither };

struct OutlierGateSettings {
    OutlierGateMode mode = OutlierGateMode::kRange;  // legacy default preserves behaviour
    double range_ratio       = 81.0;  // 9² — upstream FAST-LIO constant
    double mahalanobis_chi2  = 6.63;  // χ² 1-DoF 99%
};

// Observability guard on the stacked per-frame LiDAR jacobian h_x. See
// `observability_guard.h` for the detection logic; this struct pairs the
// mode selector with its numeric knobs. The default (kIgnore) is a no-op —
// the guard runs the analysis for diagnostics but doesn't gate the update.
struct ObservabilityGuardSettings {
    ObservabilityGuardMode mode = ObservabilityGuardMode::kIgnore;   // no-op default
    int min_translation_rank = 3;     // full rank required to consider "observable"
    double singular_threshold = 1.0e-4;  // absolute σ threshold on info matrix
};

// Mapping / IEKF hyperparameters that don't belong to a nested struct.
struct MappingSettings {
    float  det_range             = 300.0f;
    double filter_size_surf      = 0.5;
    double filter_size_map       = 0.5;
    double cube_side_length      = 1000.0;
    float  esti_plane_threshold  = 0.1f;
    float  map_quality_threshold = 0.0f;  // 0 = disable map-quality gating
    int    max_iteration         = 4;
    // Optional LiDAR point-to-plane measurement covariance override. Absent
    // → options::DEFAULT_LASER_POINT_COV is used (preserves legacy tuning).
    std::optional<double> laser_point_cov;
    OutlierGateSettings outlier_gate{};
    ObservabilityGuardSettings observability_guard{};
};

// LiDAR pose in IMU frame + online refinement flag.
struct ExtrinsicsSettings {
    std::vector<double> T {3, 0.0};
    std::vector<double> R {9, 0.0};
    bool estimate_online = true;
};

// IEKF process-noise densities (diagonal of Q_). Rename of upstream
// `mapping.{gyr,acc,b_gyr,b_acc}_cov` — those legacy names remain
// accepted for back-compat, with a deprecation warning.
struct ProcessNoiseSettings {
    float ng  = 0.1f;     // gyro white noise (rad/s²/Hz)
    float na  = 0.1f;     // accel white noise (m/s²²/Hz)
    float nbg = 1.0e-4f;  // gyro bias random walk
    float nba = 1.0e-4f;  // accel bias random walk
};

// IMU init gate + leveling + initial covariance knobs. Two configuration
// modes for the gate window (count vs time); precedence is explicit in
// `gate_mode`.
struct ImuInitSettings {
    bool motion_gate_enabled = false;
    double acc_rel_thresh    = 0.05;
    double gyr_thresh        = 0.05;

    enum class GateMode { kCountBased, kTimeBased };
    GateMode gate_mode   = GateMode::kTimeBased;
    int      min_accepted = DEFAULT_INIT_GATE_MIN_ACCEPTED;
    int      max_tries    = DEFAULT_INIT_GATE_MAX_TRIES;
    double   min_time_s   = DEFAULT_INIT_GATE_MIN_TIME_S;
    double   max_time_s   = DEFAULT_INIT_GATE_MAX_TIME_S;

    bool assume_level = false;
    // Populated only when YAML has an `imu_init.init_P_diag` block; unset
    // leaves ImuProcess at its struct-default diagonals.
    std::optional<InitPDiag> init_P_diag;
};

// Pose-graph optimisation backend.
struct PoseGraphSettings {
    bool enabled = false;
    PoseGraph::Options options;
    // Relative-measurement std for odometry edges (m / rad per edge).
    double odom_trans_std = 0.05;
    double odom_rot_std   = 0.02;
};

// Loop-closure detection — only active when pose_graph.enabled is also true.
struct LoopClosureSettings {
    bool enabled = false;
    LoopCloser::Options options;
};

// Wheel-odometry fusion (scalar body-velocity updates + NHC virtual obs).
struct WheelSettings {
    bool   enabled         = false;
    double cov_v_x         = 0.01;
    double cov_v_y         = 0.01;
    double cov_v_z         = 0.001;
    double cov_omega_z     = 0.01;
    bool   emit_nhc_v_x    = false;  // default OFF: most robots have body-X = forward
    bool   emit_nhc_v_y    = true;
    bool   emit_nhc_v_z    = true;
    double nhc_cov         = 0.001;
    double max_time_gap    = 0.05;
};

// iVox local-map index.
struct IvoxSettings {
    float resolution = 0.5f;
    int   nearby_type = 18;  // accepted: 0 (CENTER), 6, 18, 26
};

// Output / persistence.
struct OutputSettings {
    bool path_en       = false;
    bool dense_en      = false;
    bool path_save_en  = false;
    bool pcd_save_en   = false;
    int  pcd_save_interval = -1;
};

// Aggregate config — everything the LaserMapping pipeline reads from YAML.
struct LaserMappingConfig {
    CommonSettings        common;
    PreprocessSettings    preprocess;
    MappingSettings       mapping;
    ExtrinsicsSettings    extrinsics;
    ProcessNoiseSettings  process_noise;
    ImuInitSettings       imu_init;
    PoseGraphSettings     pose_graph;
    LoopClosureSettings   loop_closure;
    WheelSettings         wheel;
    IvoxSettings          ivox;
    OutputSettings        output;
};

// ── Parser entry points ────────────────────────────────────────────────

/// Load a YAML file from disk and populate `out` with the parsed config.
/// Returns false on file-open failure, YAML parse error, or missing
/// required keys. Errors are logged via spdlog with the offending key
/// name where possible. On failure `out` is left partially-populated and
/// should not be used.
bool LoadLaserMappingConfig(const std::string &yaml_path, LaserMappingConfig &out);

/// Same as LoadLaserMappingConfig but takes an already-loaded YAML::Node.
/// Useful for tests that construct YAML in-memory.
bool ParseLaserMappingConfig(const YAML::Node &yaml, LaserMappingConfig &out);

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_CONFIG_H
