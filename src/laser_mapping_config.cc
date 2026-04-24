#include "faster_lio/laser_mapping_config.h"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "faster_lio/observability_guard.h"

namespace faster_lio {

namespace {

// Read an optional scalar from `node[key]`. Returns fallback if absent.
// Any yaml-cpp conversion exception is re-thrown with the key name as
// context so the user sees WHICH field failed (the legacy parser wrapped
// the entire LoadParamsFromYAML in a blanket catch with a generic message).
template <typename T>
T readOptional(const YAML::Node &node, const char *key, const T &fallback) {
    if (!node || !node[key]) return fallback;
    try {
        return node[key].as<T>();
    } catch (const YAML::Exception &e) {
        throw std::runtime_error(
            std::string("YAML key '") + key + "' could not be parsed as the expected type: " + e.what());
    }
}

// Read a required scalar. Throws with a contextual message if the key is
// missing or has the wrong type.
template <typename T>
T readRequired(const YAML::Node &node, const char *key) {
    if (!node || !node[key]) {
        throw std::runtime_error(std::string("Missing required YAML key '") + key + "'");
    }
    try {
        return node[key].as<T>();
    } catch (const YAML::Exception &e) {
        throw std::runtime_error(
            std::string("YAML key '") + key + "' could not be parsed as the expected type: " + e.what());
    }
}

void ParseCommon(const YAML::Node &yaml, CommonSettings &out) {
    out.time_sync_en = readRequired<bool>(yaml["common"], "time_sync_en");
}

void ParsePreprocess(const YAML::Node &yaml, PreprocessSettings &out) {
    if (yaml["preprocess"]) {
        out.blind = readOptional<double>(yaml["preprocess"], "blind", out.blind);
    }
    out.point_filter_num = readRequired<int>(yaml, "point_filter_num");
}

void ParseMapping(const YAML::Node &yaml, MappingSettings &out) {
    out.max_iteration         = readRequired<int>(yaml, "max_iteration");
    out.esti_plane_threshold  = readRequired<float>(yaml, "esti_plane_threshold");
    out.filter_size_surf      = readRequired<double>(yaml, "filter_size_surf");
    out.filter_size_map       = readRequired<double>(yaml, "filter_size_map");
    out.cube_side_length      = readRequired<double>(yaml, "cube_side_length");
    out.map_quality_threshold = readOptional<float>(yaml, "map_quality_threshold", out.map_quality_threshold);

    const auto &mapping = yaml["mapping"];
    if (!mapping) throw std::runtime_error("Missing required YAML block 'mapping'");
    out.det_range = readRequired<float>(mapping, "det_range");
    if (mapping["laser_point_cov"]) {
        out.laser_point_cov = mapping["laser_point_cov"].as<double>();
    }

    // Optional outlier-gate sub-block. Absent → struct defaults
    // (mode=range, ratio=81, chi²=6.63 — byte-identical legacy behaviour).
    if (mapping["outlier_gate"]) {
        const auto &og = mapping["outlier_gate"];
        auto &gate = out.outlier_gate;
        if (og["mode"]) {
            const auto mode = og["mode"].as<std::string>();
            if      (mode == "range")       gate.mode = OutlierGateMode::kRange;
            else if (mode == "mahalanobis") gate.mode = OutlierGateMode::kMahalanobis;
            else if (mode == "either")      gate.mode = OutlierGateMode::kEither;
            else {
                throw std::runtime_error(
                    std::string("mapping.outlier_gate.mode must be one of "
                                "'range' / 'mahalanobis' / 'either' — got '") + mode + "'");
            }
        }
        gate.range_ratio      = readOptional<double>(og, "range_ratio",      gate.range_ratio);
        gate.mahalanobis_chi2 = readOptional<double>(og, "mahalanobis_chi2", gate.mahalanobis_chi2);
    }

    // Optional observability-guard sub-block. Absent → struct defaults
    // (mode=ignore, min_translation_rank=3, threshold=1e-4 — analyse-only
    // baseline that preserves legacy behaviour).
    if (mapping["observability_guard"]) {
        const auto &og = mapping["observability_guard"];
        auto &guard = out.observability_guard;
        if (og["mode"]) {
            const auto mode_str = og["mode"].as<std::string>();
            if (!ParseObservabilityGuardMode(mode_str, guard.mode)) {
                throw std::runtime_error(
                    std::string("mapping.observability_guard.mode must be one of "
                                "'ignore' / 'skip_position' / 'skip_update' — got '") +
                    mode_str + "'");
            }
        }
        guard.min_translation_rank = readOptional<int>   (og, "min_translation_rank", guard.min_translation_rank);
        guard.singular_threshold   = readOptional<double>(og, "singular_threshold",   guard.singular_threshold);
    }
}

void ParseExtrinsics(const YAML::Node &yaml, ExtrinsicsSettings &out) {
    const auto &mapping = yaml["mapping"];
    if (!mapping) throw std::runtime_error("Missing required YAML block 'mapping'");
    out.estimate_online = readRequired<bool>(mapping, "extrinsic_est_en");
    out.T = readRequired<std::vector<double>>(mapping, "extrinsic_T");
    out.R = readRequired<std::vector<double>>(mapping, "extrinsic_R");
    if (out.T.size() != 3) {
        throw std::runtime_error("mapping.extrinsic_T must have exactly 3 elements");
    }
    if (out.R.size() != 9) {
        throw std::runtime_error("mapping.extrinsic_R must have exactly 9 elements");
    }
}

// Resolve process-noise densities with back-compat. Preferred `process_noise`
// block wins; legacy `mapping.{gyr,acc,b_gyr,b_acc}_cov` keys are accepted
// with a deprecation warning. Per-field fallback: if the new block is
// partial, missing fields inherit from the legacy keys (if present) before
// falling through to struct defaults. Neither present → return false so
// the caller can report the missing-config error with the right context.
bool ParseProcessNoise(const YAML::Node &yaml, ProcessNoiseSettings &out) {
    const bool has_new    = static_cast<bool>(yaml["process_noise"]);
    const bool has_legacy = yaml["mapping"] && yaml["mapping"]["gyr_cov"];

    if (!has_new && !has_legacy) {
        spdlog::error(
            "No process-noise config found: YAML must define either "
            "`process_noise: {{ng, na, nbg, nba}}` (preferred) or legacy "
            "`mapping.{{gyr_cov, acc_cov, b_gyr_cov, b_acc_cov}}`");
        return false;
    }
    if (has_new && has_legacy) {
        spdlog::warn(
            "Both `process_noise` and legacy `mapping.*_cov` keys present; "
            "using `process_noise` (preferred). Remove legacy keys to silence.");
    }
    if (has_legacy && !has_new) {
        spdlog::warn(
            "Using deprecated `mapping.{{gyr,acc,b_gyr,b_acc}}_cov` YAML keys. "
            "Migrate to `process_noise: {{ng, na, nbg, nba}}` — these are "
            "process-noise densities, not measurement covariances.");
    }

    auto pick = [&](const char *new_key, const char *legacy_key, float fallback) -> float {
        if (has_new    && yaml["process_noise"][new_key])  return yaml["process_noise"][new_key].as<float>();
        if (has_legacy && yaml["mapping"][legacy_key])     return yaml["mapping"][legacy_key].as<float>();
        return fallback;
    };
    out.ng  = pick("ng",  "gyr_cov",   out.ng);
    out.na  = pick("na",  "acc_cov",   out.na);
    out.nbg = pick("nbg", "b_gyr_cov", out.nbg);
    out.nba = pick("nba", "b_acc_cov", out.nba);
    return true;
}

// Parse the optional init_P_diag block, returning a fully-populated
// InitPDiag (missing fields inherit struct defaults = legacy values).
InitPDiag ParseInitPDiag(const YAML::Node &node) {
    InitPDiag ip;  // struct defaults = upstream hard-coded init_P
    ip.pos       = readOptional<double>(node, "pos",       ip.pos);
    ip.rot       = readOptional<double>(node, "rot",       ip.rot);
    ip.off_R_L_I = readOptional<double>(node, "off_R_L_I", ip.off_R_L_I);
    ip.off_T_L_I = readOptional<double>(node, "off_T_L_I", ip.off_T_L_I);
    ip.vel       = readOptional<double>(node, "vel",       ip.vel);
    ip.bg        = readOptional<double>(node, "bg",        ip.bg);
    ip.ba        = readOptional<double>(node, "ba",        ip.ba);
    ip.grav      = readOptional<double>(node, "grav",      ip.grav);
    return ip;
}

void ParseImuInit(const YAML::Node &yaml, ImuInitSettings &out) {
    if (!yaml["imu_init"]) return;  // whole block optional → keep struct defaults
    const auto &ii = yaml["imu_init"];

    out.motion_gate_enabled = readOptional<bool>  (ii, "motion_gate_enabled", out.motion_gate_enabled);
    out.acc_rel_thresh      = readOptional<double>(ii, "acc_rel_thresh",      out.acc_rel_thresh);
    out.gyr_thresh          = readOptional<double>(ii, "gyr_thresh",          out.gyr_thresh);
    out.assume_level        = readOptional<bool>  (ii, "assume_level",        out.assume_level);

    // Gate-window precedence:
    //   any count key present → count mode; absent count fields fall back
    //     to sample-count defaults.
    //   no count keys → time mode; absent time fields fall back to time
    //     defaults (0.5 s / 5.0 s).
    //   both groups present → count wins, warning emitted.
    const bool has_count_min = static_cast<bool>(ii["min_accepted"]);
    const bool has_count_max = static_cast<bool>(ii["max_tries"]);
    const bool has_time_min  = static_cast<bool>(ii["min_time_s"]);
    const bool has_time_max  = static_cast<bool>(ii["max_time_s"]);
    const bool has_any_count = has_count_min || has_count_max;
    const bool has_any_time  = has_time_min  || has_time_max;

    if (has_any_count && has_any_time) {
        spdlog::warn(
            "Both sample-count (min_accepted/max_tries) and time-based "
            "(min_time_s/max_time_s) init-gate keys present in YAML; "
            "using sample counts (remove the time keys to silence).");
    }

    if (has_any_count) {
        out.gate_mode    = ImuInitSettings::GateMode::kCountBased;
        out.min_accepted = readOptional<int>(ii, "min_accepted", DEFAULT_INIT_GATE_MIN_ACCEPTED);
        out.max_tries    = readOptional<int>(ii, "max_tries",    DEFAULT_INIT_GATE_MAX_TRIES);
    } else {
        out.gate_mode  = ImuInitSettings::GateMode::kTimeBased;
        out.min_time_s = readOptional<double>(ii, "min_time_s", DEFAULT_INIT_GATE_MIN_TIME_S);
        out.max_time_s = readOptional<double>(ii, "max_time_s", DEFAULT_INIT_GATE_MAX_TIME_S);
    }

    if (ii["init_P_diag"]) {
        out.init_P_diag = ParseInitPDiag(ii["init_P_diag"]);
    }
}

void ParsePoseGraph(const YAML::Node &yaml, PoseGraphSettings &out) {
    if (!yaml["pose_graph"]) return;
    const auto &pg = yaml["pose_graph"];
    out.enabled                       = readOptional<bool>  (pg, "enabled",          out.enabled);
    out.options.keyframe_dist_thresh  = readOptional<double>(pg, "keyframe_dist",    out.options.keyframe_dist_thresh);
    out.options.keyframe_angle_thresh = readOptional<double>(pg, "keyframe_angle",   out.options.keyframe_angle_thresh);
    out.options.optimize_every_n      = readOptional<int>   (pg, "optimize_every_n", out.options.optimize_every_n);
    out.options.max_iterations        = readOptional<int>   (pg, "max_iterations",   out.options.max_iterations);
    out.odom_trans_std                = readOptional<double>(pg, "odom_trans_std",   out.odom_trans_std);
    out.odom_rot_std                  = readOptional<double>(pg, "odom_rot_std",     out.odom_rot_std);
}

void ParseLoopClosure(const YAML::Node &yaml, LoopClosureSettings &out) {
    if (!yaml["loop_closure"]) return;
    const auto &lc = yaml["loop_closure"];
    auto &o = out.options;
    out.enabled                  = readOptional<bool>        (lc, "enabled",                  out.enabled);
    o.revisit_radius             = readOptional<float>       (lc, "revisit_radius",           o.revisit_radius);
    o.min_age_frames             = readOptional<int>         (lc, "min_age_frames",           o.min_age_frames);
    o.icp_max_correspondence     = readOptional<float>       (lc, "icp_max_correspondence",   o.icp_max_correspondence);
    o.icp_max_iterations         = readOptional<int>         (lc, "icp_max_iterations",       o.icp_max_iterations);
    o.icp_fitness_threshold      = readOptional<float>       (lc, "icp_fitness_threshold",    o.icp_fitness_threshold);
    o.max_candidates_per_call    = readOptional<std::size_t> (lc, "max_candidates_per_call",  o.max_candidates_per_call);
    o.voxel_size                 = readOptional<float>       (lc, "voxel_size",               o.voxel_size);
    o.max_points_per_submap      = readOptional<std::size_t> (lc, "max_points_per_submap",    o.max_points_per_submap);
    o.max_submaps                = readOptional<std::size_t> (lc, "max_submaps",              o.max_submaps);
    // Scan Context (global, pose-independent) knobs — optional.
    o.sc_enabled                 = readOptional<bool>        (lc, "sc_enabled",               o.sc_enabled);
    o.sc_num_rings               = readOptional<int>         (lc, "sc_num_rings",             o.sc_num_rings);
    o.sc_num_sectors             = readOptional<int>         (lc, "sc_num_sectors",           o.sc_num_sectors);
    o.sc_max_range               = readOptional<double>      (lc, "sc_max_range",             o.sc_max_range);
    o.sc_aggregation_window      = readOptional<int>         (lc, "sc_aggregation_window",    o.sc_aggregation_window);
    o.sc_ring_key_threshold      = readOptional<double>      (lc, "sc_ring_key_threshold",    o.sc_ring_key_threshold);
    o.sc_score_threshold         = readOptional<double>      (lc, "sc_score_threshold",       o.sc_score_threshold);
    o.sc_min_overlap_ratio       = readOptional<double>      (lc, "sc_min_overlap_ratio",     o.sc_min_overlap_ratio);
    o.sc_top_k                   = readOptional<std::size_t> (lc, "sc_top_k",                 o.sc_top_k);
}

void ParseWheel(const YAML::Node &yaml, WheelSettings &out) {
    if (!yaml["wheel"]) return;
    const auto &w = yaml["wheel"];
    out.enabled      = readOptional<bool>  (w, "enabled",      out.enabled);
    out.cov_v_x      = readOptional<double>(w, "cov_v_x",      out.cov_v_x);
    out.cov_v_y      = readOptional<double>(w, "cov_v_y",      out.cov_v_y);
    out.cov_v_z      = readOptional<double>(w, "cov_v_z",      out.cov_v_z);
    out.cov_omega_z  = readOptional<double>(w, "cov_omega_z",  out.cov_omega_z);
    out.emit_nhc_v_x = readOptional<bool>  (w, "emit_nhc_v_x", out.emit_nhc_v_x);
    out.emit_nhc_v_y = readOptional<bool>  (w, "emit_nhc_v_y", out.emit_nhc_v_y);
    out.emit_nhc_v_z = readOptional<bool>  (w, "emit_nhc_v_z", out.emit_nhc_v_z);
    out.nhc_cov      = readOptional<double>(w, "nhc_cov",      out.nhc_cov);
    out.max_time_gap = readOptional<double>(w, "max_time_gap", out.max_time_gap);
}

void ParseIvox(const YAML::Node &yaml, IvoxSettings &out) {
    out.resolution   = readRequired<float>(yaml, "ivox_grid_resolution");
    out.nearby_type  = readRequired<int>  (yaml, "ivox_nearby_type");
}

void ParseOutput(const YAML::Node &yaml, OutputSettings &out) {
    const auto &output = yaml["output"];
    if (!output) throw std::runtime_error("Missing required YAML block 'output'");
    out.path_en      = readRequired<bool>(output, "path_en");
    out.dense_en     = readRequired<bool>(output, "dense_en");
    out.path_save_en = readRequired<bool>(output, "path_save_en");

    const auto &pcd = yaml["pcd_save"];
    if (!pcd) throw std::runtime_error("Missing required YAML block 'pcd_save'");
    out.pcd_save_en       = readRequired<bool>(pcd, "pcd_save_en");
    out.pcd_save_interval = readRequired<int> (pcd, "interval");
}

}  // namespace

bool ParseLaserMappingConfig(const YAML::Node &yaml, LaserMappingConfig &out) {
    try {
        ParseCommon     (yaml, out.common);
        ParsePreprocess (yaml, out.preprocess);
        ParseMapping    (yaml, out.mapping);
        ParseExtrinsics (yaml, out.extrinsics);
        if (!ParseProcessNoise(yaml, out.process_noise)) {
            return false;  // missing-config error already logged
        }
        ParseImuInit    (yaml, out.imu_init);
        ParsePoseGraph  (yaml, out.pose_graph);
        ParseLoopClosure(yaml, out.loop_closure);
        ParseWheel      (yaml, out.wheel);
        ParseIvox       (yaml, out.ivox);
        ParseOutput     (yaml, out.output);
    } catch (const std::exception &e) {
        spdlog::error("Failed to parse LaserMappingConfig: {}", e.what());
        return false;
    }

    // Cross-block validation: loop closure requires pose graph. The legacy
    // parser warned about this and silently disabled loop_closure; keep
    // that behaviour here so existing configs don't regress.
    if (out.loop_closure.enabled && !out.pose_graph.enabled) {
        spdlog::warn(
            "Loop closure requested but pose_graph is OFF — disabling LCD. "
            "Set pose_graph.enabled: true to use this feature.");
        out.loop_closure.enabled = false;
    }
    return true;
}

bool LoadLaserMappingConfig(const std::string &yaml_path, LaserMappingConfig &out) {
    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(yaml_path);
    } catch (const YAML::BadFile &e) {
        spdlog::error("Failed to open YAML config file '{}': {}", yaml_path, e.what());
        return false;
    } catch (const YAML::Exception &e) {
        spdlog::error("Failed to parse YAML config file '{}': {}", yaml_path, e.what());
        return false;
    }
    return ParseLaserMappingConfig(yaml, out);
}

}  // namespace faster_lio
