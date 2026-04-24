// ─────────────────────────────────────────────────────────────────────────
// test_laser_mapping_config.cc
//
// Direct tests for the YAML → LaserMappingConfig parser. Exercises the
// parser without constructing a full LaserMapping, so back-compat
// precedence rules and error messages can be pinned in isolation. The
// LaserMapping-level integration tests live in test_imu_init_params.cc
// under the `YamlInitPDiag`, `YamlProcessNoise`, `YamlInitGate` suites.
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <string>

#include "faster_lio/laser_mapping_config.h"

using namespace faster_lio;

namespace {

// A minimal valid YAML payload that parses cleanly. Tests start from this
// and perturb fields as needed.
std::string MinimalYAML() {
    return R"(
common:
  time_sync_en: false
preprocess:
  blind: 0.1
mapping:
  det_range: 100.0
  extrinsic_est_en: false
  extrinsic_T: [0, 0, 0]
  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]
process_noise:
  ng:  0.1
  na:  0.1
  nbg: 0.0001
  nba: 0.0001
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
}

YAML::Node LoadYAML(const std::string &s) { return YAML::Load(s); }

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// A. Happy path — the minimal YAML parses without error.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, MinimalYamlParsesSuccessfully) {
    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(MinimalYAML()), cfg));

    // Spot-check every subtree got something reasonable.
    EXPECT_FALSE(cfg.common.time_sync_en);
    EXPECT_DOUBLE_EQ(cfg.preprocess.blind, 0.1);
    EXPECT_FLOAT_EQ (cfg.mapping.det_range, 100.0f);
    EXPECT_EQ       (cfg.mapping.max_iteration, 4);
    EXPECT_FALSE    (cfg.extrinsics.estimate_online);
    EXPECT_FLOAT_EQ (cfg.process_noise.ng,  0.1f);
    EXPECT_FLOAT_EQ (cfg.process_noise.nbg, 1.0e-4f);
    EXPECT_EQ       (cfg.ivox.nearby_type, 18);
    EXPECT_FALSE    (cfg.output.path_en);
    EXPECT_FALSE    (cfg.pose_graph.enabled);
    EXPECT_FALSE    (cfg.loop_closure.enabled);
    EXPECT_FALSE    (cfg.wheel.enabled);
    EXPECT_FALSE    (cfg.imu_init.motion_gate_enabled);
}

// ═════════════════════════════════════════════════════════════════════════
// B. Missing / malformed — errors must surface with context.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, MissingCommonBlockFails) {
    auto yaml = MinimalYAML();
    // Remove `common:` subtree entirely.
    const auto pos = yaml.find("common:");
    yaml.replace(pos, std::string("common:\n  time_sync_en: false\n").size(), "");
    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
}

TEST(LaserMappingConfigParser, MissingMappingBlockFails) {
    auto yaml = MinimalYAML();
    const auto start = yaml.find("mapping:");
    const auto end   = yaml.find("process_noise:");
    yaml.erase(start, end - start);
    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
}

TEST(LaserMappingConfigParser, MissingProcessNoiseAndLegacyFails) {
    auto yaml = MinimalYAML();
    const auto start = yaml.find("process_noise:");
    const auto end   = yaml.find("max_iteration:");
    yaml.erase(start, end - start);
    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
}

TEST(LaserMappingConfigParser, ExtrinsicT_WrongSize_Fails) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("extrinsic_T:");
    yaml.replace(pos, std::string("extrinsic_T: [0, 0, 0]").size(),
                 "extrinsic_T: [0, 0]");
    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
}

TEST(LaserMappingConfigParser, ExtrinsicR_WrongSize_Fails) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("extrinsic_R:");
    yaml.replace(pos, std::string("extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]").size(),
                 "extrinsic_R: [1, 0, 0, 0]");
    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
}

// Wrong type for a required scalar (string where an int is expected) must
// fail with an error — and the error message names the key. We can't
// easily capture spdlog output, so we settle for "returns false without
// crashing". The key-name-in-message invariant is documented in the
// parser and covered by manual inspection.
TEST(LaserMappingConfigParser, WrongTypeRequiredKeyFails) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("max_iteration: 4");
    yaml.replace(pos, std::string("max_iteration: 4").size(),
                 "max_iteration: not-an-int");
    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
}

// ═════════════════════════════════════════════════════════════════════════
// C. Process-noise back-compat — new `process_noise` ↔ legacy
//    `mapping.*_cov`. Covered for LaserMapping integration in
//    test_imu_init_params.cc; pinned here at the parser level so the
//    helper can't silently break during refactors.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, LegacyProcessNoiseKeysAccepted) {
    auto yaml = MinimalYAML();
    // Replace the process_noise block with legacy mapping.*_cov keys.
    const auto pn_start = yaml.find("process_noise:");
    const auto pn_end   = yaml.find("max_iteration:");
    yaml.erase(pn_start, pn_end - pn_start);
    // Inject legacy keys into the existing mapping: block.
    const auto map_det = yaml.find("det_range: 100.0");
    yaml.insert(map_det + std::string("det_range: 100.0\n").size(),
                "  acc_cov: 0.22\n  gyr_cov: 0.33\n  b_acc_cov: 4.4e-4\n  b_gyr_cov: 5.5e-4\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_FLOAT_EQ(cfg.process_noise.na,  0.22f);
    EXPECT_FLOAT_EQ(cfg.process_noise.ng,  0.33f);
    EXPECT_FLOAT_EQ(cfg.process_noise.nba, 4.4e-4f);
    EXPECT_FLOAT_EQ(cfg.process_noise.nbg, 5.5e-4f);
}

TEST(LaserMappingConfigParser, BothProcessNoiseBlocksNewWins) {
    auto yaml = MinimalYAML();
    // Minimal already has `process_noise: {ng: 0.1, ...}`. Add legacy
    // sentinel values under mapping: which must NOT leak through.
    const auto map_det = yaml.find("det_range: 100.0");
    yaml.insert(map_det + std::string("det_range: 100.0\n").size(),
                "  acc_cov: 9.9\n  gyr_cov: 9.9\n  b_acc_cov: 9.9\n  b_gyr_cov: 9.9\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_FLOAT_EQ(cfg.process_noise.ng,  0.1f);
    EXPECT_FLOAT_EQ(cfg.process_noise.na,  0.1f);
    EXPECT_FLOAT_EQ(cfg.process_noise.nbg, 1.0e-4f);
    EXPECT_FLOAT_EQ(cfg.process_noise.nba, 1.0e-4f);
}

// ═════════════════════════════════════════════════════════════════════════
// D. Init-gate mode precedence — count vs time.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, ImuInit_TimeKeysOnly_SelectsTimeMode) {
    auto yaml = MinimalYAML();
    yaml += "\nimu_init:\n  motion_gate_enabled: true\n  min_time_s: 1.5\n  max_time_s: 15.0\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_TRUE(cfg.imu_init.motion_gate_enabled);
    EXPECT_EQ(cfg.imu_init.gate_mode, ImuInitSettings::GateMode::kTimeBased);
    EXPECT_DOUBLE_EQ(cfg.imu_init.min_time_s, 1.5);
    EXPECT_DOUBLE_EQ(cfg.imu_init.max_time_s, 15.0);
}

TEST(LaserMappingConfigParser, ImuInit_CountKeysOnly_SelectsCountMode) {
    auto yaml = MinimalYAML();
    yaml += "\nimu_init:\n  motion_gate_enabled: true\n  min_accepted: 321\n  max_tries: 6543\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ(cfg.imu_init.gate_mode, ImuInitSettings::GateMode::kCountBased);
    EXPECT_EQ(cfg.imu_init.min_accepted, 321);
    EXPECT_EQ(cfg.imu_init.max_tries,    6543);
}

TEST(LaserMappingConfigParser, ImuInit_BothKeys_CountWins) {
    auto yaml = MinimalYAML();
    yaml +=
        "\nimu_init:\n  motion_gate_enabled: true\n"
        "  min_accepted: 777\n  max_tries: 8888\n"
        "  min_time_s: 9.9\n  max_time_s: 99.9\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ(cfg.imu_init.gate_mode, ImuInitSettings::GateMode::kCountBased);
    EXPECT_EQ(cfg.imu_init.min_accepted, 777);
    EXPECT_EQ(cfg.imu_init.max_tries,    8888);
}

TEST(LaserMappingConfigParser, ImuInit_EmptyBlock_UsesTimeDefaults) {
    auto yaml = MinimalYAML();
    yaml += "\nimu_init:\n  motion_gate_enabled: true\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ(cfg.imu_init.gate_mode, ImuInitSettings::GateMode::kTimeBased);
    EXPECT_DOUBLE_EQ(cfg.imu_init.min_time_s, DEFAULT_INIT_GATE_MIN_TIME_S);
    EXPECT_DOUBLE_EQ(cfg.imu_init.max_time_s, DEFAULT_INIT_GATE_MAX_TIME_S);
}

TEST(LaserMappingConfigParser, ImuInit_InitPDiagOptional) {
    auto yaml = MinimalYAML();
    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_FALSE(cfg.imu_init.init_P_diag.has_value());  // absent → defaults apply via ImuProcess
}

TEST(LaserMappingConfigParser, ImuInit_InitPDiagPartialOverrideKeepsStructDefaults) {
    auto yaml = MinimalYAML();
    yaml +=
        "\nimu_init:\n"
        "  init_P_diag:\n"
        "    pos: 0.5\n"
        "    ba:  2.5e-3\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    ASSERT_TRUE(cfg.imu_init.init_P_diag.has_value());
    const InitPDiag &ip = *cfg.imu_init.init_P_diag;
    EXPECT_DOUBLE_EQ(ip.pos, 0.5);
    EXPECT_DOUBLE_EQ(ip.ba,  2.5e-3);
    // Untouched fields retain InitPDiag struct defaults.
    EXPECT_DOUBLE_EQ(ip.rot,       1.0);
    EXPECT_DOUBLE_EQ(ip.off_R_L_I, 1.0e-5);
    EXPECT_DOUBLE_EQ(ip.vel,       1.0);
    EXPECT_DOUBLE_EQ(ip.bg,        1.0e-4);
    EXPECT_DOUBLE_EQ(ip.grav,      1.0e-5);
}

// ═════════════════════════════════════════════════════════════════════════
// E. Cross-block validation — loop closure requires pose graph.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, LoopClosureWithoutPoseGraphDisablesLCD) {
    auto yaml = MinimalYAML();
    yaml += "\nloop_closure:\n  enabled: true\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_FALSE(cfg.loop_closure.enabled)
        << "Loop closure requested with pose_graph disabled should be silently disabled "
        << "(with a warning) to match legacy behaviour.";
}

TEST(LaserMappingConfigParser, LoopClosureWithPoseGraphStaysEnabled) {
    auto yaml = MinimalYAML();
    yaml +=
        "\npose_graph:\n  enabled: true\n"
        "\nloop_closure:\n  enabled: true\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_TRUE(cfg.pose_graph.enabled);
    EXPECT_TRUE(cfg.loop_closure.enabled);
}

// ═════════════════════════════════════════════════════════════════════════
// F. Optional blocks — pose_graph / loop_closure / wheel default OFF.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, OptionalBlocksDefaultOff) {
    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(MinimalYAML()), cfg));
    EXPECT_FALSE(cfg.pose_graph.enabled);
    EXPECT_FALSE(cfg.loop_closure.enabled);
    EXPECT_FALSE(cfg.wheel.enabled);
    EXPECT_FALSE(cfg.imu_init.motion_gate_enabled);
}

TEST(LaserMappingConfigParser, WheelBlockParsesAllFields) {
    auto yaml = MinimalYAML();
    yaml +=
        "\nwheel:\n"
        "  enabled: true\n"
        "  cov_v_x: 0.021\n  cov_v_y: 0.022\n  cov_v_z: 0.023\n  cov_omega_z: 0.024\n"
        "  emit_nhc_v_x: true\n  emit_nhc_v_y: false\n  emit_nhc_v_z: false\n"
        "  nhc_cov: 0.00123\n  max_time_gap: 0.077\n";

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_TRUE    (cfg.wheel.enabled);
    EXPECT_DOUBLE_EQ(cfg.wheel.cov_v_x,      0.021);
    EXPECT_DOUBLE_EQ(cfg.wheel.cov_v_y,      0.022);
    EXPECT_DOUBLE_EQ(cfg.wheel.cov_v_z,      0.023);
    EXPECT_DOUBLE_EQ(cfg.wheel.cov_omega_z,  0.024);
    EXPECT_TRUE    (cfg.wheel.emit_nhc_v_x);
    EXPECT_FALSE   (cfg.wheel.emit_nhc_v_y);
    EXPECT_FALSE   (cfg.wheel.emit_nhc_v_z);
    EXPECT_DOUBLE_EQ(cfg.wheel.nhc_cov,      0.00123);
    EXPECT_DOUBLE_EQ(cfg.wheel.max_time_gap, 0.077);
}

// ═════════════════════════════════════════════════════════════════════════
// G. Optional `mapping.laser_point_cov` — absent = unset optional.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, LaserPointCov_AbsentIsOptional) {
    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(MinimalYAML()), cfg));
    EXPECT_FALSE(cfg.mapping.laser_point_cov.has_value());
}

TEST(LaserMappingConfigParser, LaserPointCov_PresentIsRoundTripped) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  laser_point_cov: 0.00042\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    ASSERT_TRUE(cfg.mapping.laser_point_cov.has_value());
    EXPECT_DOUBLE_EQ(*cfg.mapping.laser_point_cov, 0.00042);
}

// ═════════════════════════════════════════════════════════════════════════
// H. Outlier-gate block — optional; struct defaults = legacy FAST-LIO.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, OutlierGate_DefaultsAreLegacyRange) {
    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(MinimalYAML()), cfg));
    EXPECT_EQ       (cfg.mapping.outlier_gate.mode, OutlierGateMode::kRange);
    EXPECT_DOUBLE_EQ(cfg.mapping.outlier_gate.range_ratio,      81.0);
    EXPECT_DOUBLE_EQ(cfg.mapping.outlier_gate.mahalanobis_chi2, 6.63);
}

TEST(LaserMappingConfigParser, OutlierGate_ModeRangeParsed) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  outlier_gate:\n    mode: range\n    range_ratio: 49.0\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ       (cfg.mapping.outlier_gate.mode, OutlierGateMode::kRange);
    EXPECT_DOUBLE_EQ(cfg.mapping.outlier_gate.range_ratio, 49.0);
}

TEST(LaserMappingConfigParser, OutlierGate_ModeMahalanobisParsed) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  outlier_gate:\n    mode: mahalanobis\n    mahalanobis_chi2: 10.83\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ       (cfg.mapping.outlier_gate.mode, OutlierGateMode::kMahalanobis);
    EXPECT_DOUBLE_EQ(cfg.mapping.outlier_gate.mahalanobis_chi2, 10.83);
    // Untouched defaults preserved.
    EXPECT_DOUBLE_EQ(cfg.mapping.outlier_gate.range_ratio, 81.0);
}

TEST(LaserMappingConfigParser, OutlierGate_ModeEitherParsed) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  outlier_gate:\n    mode: either\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ(cfg.mapping.outlier_gate.mode, OutlierGateMode::kEither);
}

TEST(LaserMappingConfigParser, OutlierGate_UnknownModeFails) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  outlier_gate:\n    mode: rocket_science\n");

    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg))
        << "Unknown gate mode should fail cleanly with a message naming the valid options.";
}

// ═════════════════════════════════════════════════════════════════════════
// I. Observability-guard block — optional; struct defaults = ignore/no-op.
// ═════════════════════════════════════════════════════════════════════════

TEST(LaserMappingConfigParser, ObservabilityGuard_DefaultsAreIgnore) {
    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(MinimalYAML()), cfg));
    EXPECT_EQ       (cfg.mapping.observability_guard.mode, ObservabilityGuardMode::kIgnore);
    EXPECT_EQ       (cfg.mapping.observability_guard.min_translation_rank, 3);
    EXPECT_DOUBLE_EQ(cfg.mapping.observability_guard.singular_threshold,   1.0e-4);
}

// Parameterised over the three modes. We can't use TEST_P cleanly without
// a fixture, so just iterate the modes inside a single test body — GTest
// failure messages still include the mode string via SCOPED_TRACE.
TEST(LaserMappingConfigParser, ObservabilityGuard_EachModeParsed) {
    struct Case {
        const char *yaml_name;
        ObservabilityGuardMode expected;
    };
    const Case cases[] = {
        {"ignore",        ObservabilityGuardMode::kIgnore},
        {"skip_position", ObservabilityGuardMode::kSkipPosition},
        {"skip_update",   ObservabilityGuardMode::kSkipUpdate},
    };
    for (const auto &c : cases) {
        SCOPED_TRACE(c.yaml_name);
        auto yaml = MinimalYAML();
        const auto pos = yaml.find("det_range: 100.0");
        yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                    std::string("  observability_guard:\n    mode: ") + c.yaml_name + "\n");

        LaserMappingConfig cfg;
        ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
        EXPECT_EQ(cfg.mapping.observability_guard.mode, c.expected);
    }
}

TEST(LaserMappingConfigParser, ObservabilityGuard_UnknownModeFails) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  observability_guard:\n    mode: hallelujah\n");

    LaserMappingConfig cfg;
    EXPECT_FALSE(ParseLaserMappingConfig(LoadYAML(yaml), cfg))
        << "Unknown observability_guard mode must fail with the three valid options named.";
}

TEST(LaserMappingConfigParser, ObservabilityGuard_ThresholdOverride) {
    auto yaml = MinimalYAML();
    const auto pos = yaml.find("det_range: 100.0");
    yaml.insert(pos + std::string("det_range: 100.0\n").size(),
                "  observability_guard:\n"
                "    mode: skip_position\n"
                "    min_translation_rank: 2\n"
                "    singular_threshold: 5.0e-3\n");

    LaserMappingConfig cfg;
    ASSERT_TRUE(ParseLaserMappingConfig(LoadYAML(yaml), cfg));
    EXPECT_EQ       (cfg.mapping.observability_guard.mode, ObservabilityGuardMode::kSkipPosition);
    EXPECT_EQ       (cfg.mapping.observability_guard.min_translation_rank, 2);
    EXPECT_DOUBLE_EQ(cfg.mapping.observability_guard.singular_threshold,   5.0e-3);
}
