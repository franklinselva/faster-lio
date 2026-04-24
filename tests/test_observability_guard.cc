// ─────────────────────────────────────────────────────────────────────────
// test_observability_guard.cc
//
// Pure-function tests for the LiDAR observability guard. Pins the rank
// computation, threshold semantics, and in-place mutations BEFORE the
// guard is wired into ObsModel — so any numeric regression (or refactor)
// in the header surfaces here instead of in the full pipeline.
//
// These tests operate directly on Eigen N×12 matrices synthesized to
// match the specific degeneracy scenarios the guard is designed to catch
// (floor-only, parallel corridor, corridor-with-floor, etc).
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "faster_lio/observability_guard.h"

using faster_lio::AnalyzeJacobian;
using faster_lio::ComputeJacobianRank;
using faster_lio::ObservabilityGuardMode;
using faster_lio::ObservabilityGuardModeName;
using faster_lio::ObservabilitySummary;
using faster_lio::ParseObservabilityGuardMode;
using faster_lio::RankOf3x3;
using faster_lio::SingularValueMinOf;
using faster_lio::ZeroTranslationColumns;

namespace {

// Build an N×12 jacobian where each row carries `trans_normal` in cols 0..2
// (the per-point plane normal) and `rot_normal` in cols 3..5 (point-cross-
// normal). Cols 6..11 are zero — the guard doesn't look at them.
Eigen::MatrixXd MakeJacobian(const std::vector<Eigen::Vector3d> &trans_normals,
                              const std::vector<Eigen::Vector3d> &rot_normals) {
    const int N = static_cast<int>(trans_normals.size());
    Eigen::MatrixXd h_x = Eigen::MatrixXd::Zero(N, 12);
    for (int i = 0; i < N; ++i) {
        h_x(i, 0) = trans_normals[i].x();
        h_x(i, 1) = trans_normals[i].y();
        h_x(i, 2) = trans_normals[i].z();
        const Eigen::Vector3d r = (i < static_cast<int>(rot_normals.size()))
                                      ? rot_normals[i]
                                      : Eigen::Vector3d::Zero();
        h_x(i, 3) = r.x();
        h_x(i, 4) = r.y();
        h_x(i, 5) = r.z();
    }
    return h_x;
}

Eigen::MatrixXd MakeJacobianTrans(const std::vector<Eigen::Vector3d> &trans_normals) {
    return MakeJacobian(trans_normals, {});
}

constexpr double kThresh = 1.0e-4;

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// Group A — Rank computation
// ═════════════════════════════════════════════════════════════════════════

// 1. Empty h_x → both ranks 0, both σ_min = 0. Matches the empty-scan
//    safety contract documented in the header.
TEST(ObservabilityGuard, Empty_HasRankZero) {
    Eigen::MatrixXd h_x(0, 12);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 0);
    EXPECT_EQ(s.rotation_rank,    0);
    EXPECT_DOUBLE_EQ(s.min_singular_translation, 0.0);
    EXPECT_DOUBLE_EQ(s.min_singular_rotation,    0.0);
}

// 2. Six distinct plane normals spanning R³ → translation rank = 3. The
//    rotation block uses point-cross-normal cross-products that also span
//    R³, so rotation rank = 3 too.
TEST(ObservabilityGuard, FullObservability_SixDistinctNormals) {
    std::vector<Eigen::Vector3d> t = {
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0},
        { 0,  0,  1}, { 0,  0, -1},
    };
    // Rotation normals: for each row, cross the normal with a distinct
    // reference direction so the rotation block spans R³.
    std::vector<Eigen::Vector3d> r = {
        { 0,  1,  0}, { 0, -1,  0},
        { 0,  0,  1}, { 0,  0, -1},
        { 1,  0,  0}, {-1,  0,  0},
    };
    auto h_x = MakeJacobian(t, r);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 3);
    EXPECT_EQ(s.rotation_rank,    3);
    EXPECT_GT(s.min_singular_translation, kThresh);
    EXPECT_GT(s.min_singular_rotation,    kThresh);
}

// 3. All rows have normal = (0, 0, 1) → rank 1, σ_min ≈ 0.
TEST(ObservabilityGuard, FloorOnly_RankOne_Translation) {
    std::vector<Eigen::Vector3d> t(10, Eigen::Vector3d(0, 0, 1));
    auto h_x = MakeJacobianTrans(t);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 1);
    EXPECT_LT(s.min_singular_translation, kThresh);
}

// 4. Parallel corridor: normals are (±1, 0, 0). Only X is constrained →
//    rank 1 on translation.
TEST(ObservabilityGuard, ParallelCorridor_RankOne) {
    std::vector<Eigen::Vector3d> t;
    for (int i = 0; i < 10; ++i) t.emplace_back((i % 2 == 0) ? 1.0 : -1.0, 0.0, 0.0);
    auto h_x = MakeJacobianTrans(t);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 1);
}

// 5. Normals split between (±1, 0, 0) and (0, 0, 1) → X and Z constrained
//    but Y is not → rank 2.
TEST(ObservabilityGuard, CorridorWithFloor_RankTwo) {
    std::vector<Eigen::Vector3d> t = {
        { 1, 0, 0}, {-1, 0, 0}, { 1, 0, 0}, {-1, 0, 0},
        { 0, 0, 1}, { 0, 0, 1}, { 0, 0, 1}, { 0, 0, 1},
    };
    auto h_x = MakeJacobianTrans(t);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 2);
    EXPECT_LT(s.min_singular_translation, kThresh);
}

// 6. Single row → rank 1 (can only constrain one direction).
TEST(ObservabilityGuard, SinglePoint_RankOne) {
    std::vector<Eigen::Vector3d> t = {{0, 0, 1}};
    auto h_x = MakeJacobianTrans(t);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 1);
}

// 7. 100 identical rows still only constrain one direction. Redundancy
//    boosts the dominant singular value but doesn't invent new ones.
TEST(ObservabilityGuard, RedundantRows_StillOne) {
    std::vector<Eigen::Vector3d> t(100, Eigen::Vector3d(0, 0, 1));
    auto h_x = MakeJacobianTrans(t);
    const auto s = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s.translation_rank, 1);
}

// 8. Near-parallel normals. With the info-matrix threshold 1e-4 and
//    many rows, even small angles produce non-trivial σ_min so the
//    threshold is crossed easily. We bracket the transition by varying
//    the angle until rank flips 1 → 2.
//    Contract of the test: at the two angle extremes, the classification
//    differs. Exact crossover depends on N and the threshold but is
//    reliably well below 1° for N=50 at threshold 1e-4.
TEST(ObservabilityGuard, NearParallel_ClassifiesCorrectly) {
    const int N = 50;
    auto build = [&](double angle_rad) {
        std::vector<Eigen::Vector3d> t;
        const double c = std::cos(angle_rad);
        const double sn = std::sin(angle_rad);
        for (int i = 0; i < N / 2; ++i) t.emplace_back(1.0, 0.0, 0.0);
        for (int i = 0; i < N / 2; ++i) t.emplace_back(c,   sn,  0.0);
        return MakeJacobianTrans(t);
    };

    // Very small angle (10 µrad) → σ_min ≈ N·sin²(angle)/2 ≈ 2.5e-9, well
    // below threshold → rank 1.
    const auto tiny = AnalyzeJacobian(build(1.0e-5), kThresh);
    EXPECT_EQ(tiny.translation_rank, 1);

    // 5° angle — way above the crossover → rank 2.
    const auto wide = AnalyzeJacobian(build(5.0 * M_PI / 180.0), kThresh);
    EXPECT_EQ(wide.translation_rank, 2);
}

// ═════════════════════════════════════════════════════════════════════════
// Group B — Numerical robustness
// ═════════════════════════════════════════════════════════════════════════

// 9. NaN / Inf in h_x → both ranks return 0 (defensive).
TEST(ObservabilityGuard, NonFiniteInput_ReturnsZero) {
    std::vector<Eigen::Vector3d> t = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    auto h_x = MakeJacobianTrans(t);
    h_x(1, 2) = std::numeric_limits<double>::quiet_NaN();
    const auto s_nan = AnalyzeJacobian(h_x, kThresh);
    EXPECT_EQ(s_nan.translation_rank, 0);
    EXPECT_EQ(s_nan.rotation_rank,    0);

    auto h_x2 = MakeJacobianTrans(t);
    h_x2(0, 0) = std::numeric_limits<double>::infinity();
    const auto s_inf = AnalyzeJacobian(h_x2, kThresh);
    EXPECT_EQ(s_inf.translation_rank, 0);
    EXPECT_EQ(s_inf.rotation_rank,    0);
}

// 10. Wrong width (cols != 12) → both ranks 0.
TEST(ObservabilityGuard, WrongWidth_ReturnsZero) {
    Eigen::MatrixXd h_x_narrow = Eigen::MatrixXd::Identity(6, 6);
    const auto s1 = AnalyzeJacobian(h_x_narrow, kThresh);
    EXPECT_EQ(s1.translation_rank, 0);
    EXPECT_EQ(s1.rotation_rank,    0);

    Eigen::MatrixXd h_x_wide = Eigen::MatrixXd::Identity(6, 20);
    const auto s2 = AnalyzeJacobian(h_x_wide, kThresh);
    EXPECT_EQ(s2.translation_rank, 0);
    EXPECT_EQ(s2.rotation_rank,    0);
}

// 11. Negative threshold → defensive zero. Also catches threshold = NaN.
TEST(ObservabilityGuard, NegativeThreshold_ReturnsZero) {
    std::vector<Eigen::Vector3d> t = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    auto h_x = MakeJacobianTrans(t);

    const auto s_neg = AnalyzeJacobian(h_x, -1.0);
    EXPECT_EQ(s_neg.translation_rank, 0);
    EXPECT_EQ(s_neg.rotation_rank,    0);

    const auto s_nan = AnalyzeJacobian(h_x, std::nan(""));
    EXPECT_EQ(s_nan.translation_rank, 0);
    EXPECT_EQ(s_nan.rotation_rank,    0);
}

// 12. For a fixed h_x, rank is monotone non-increasing as threshold grows.
//     A tighter threshold can never invent observability.
TEST(ObservabilityGuard, ThresholdMonotonicity) {
    std::vector<Eigen::Vector3d> t = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1},
    };
    auto h_x = MakeJacobianTrans(t);

    const std::vector<double> thresholds = {0.0, 1e-6, 1e-3, 1.0, 5.0, 100.0};
    int prev_rank = std::numeric_limits<int>::max();
    for (const double th : thresholds) {
        const auto s = AnalyzeJacobian(h_x, th);
        EXPECT_LE(s.translation_rank, prev_rank)
            << "rank increased from " << prev_rank << " to "
            << s.translation_rank << " when threshold went up to " << th;
        prev_rank = s.translation_rank;
    }
    EXPECT_EQ(prev_rank, 0)  // at threshold 100, σ²-sum = 2 per axis, all below
        << "Largest threshold should drive rank to 0 for unit-normal jacobian.";
}

// ═════════════════════════════════════════════════════════════════════════
// Group C — In-place mutation
// ═════════════════════════════════════════════════════════════════════════

// 13. ZeroTranslationColumns zeros cols 0..2 and leaves 3..11 alone.
TEST(ObservabilityGuard, ZeroTranslationColumns_DoesWhatItSays) {
    Eigen::MatrixXd h_x(4, 12);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 12; ++j) {
            h_x(i, j) = static_cast<double>(1 + i * 12 + j);  // all non-zero
        }
    }
    const Eigen::MatrixXd h_x_before = h_x;

    ZeroTranslationColumns(h_x);

    // Cols 0..2: zero.
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(h_x(i, j), 0.0);
        }
    }
    // Cols 3..11: unchanged.
    for (int i = 0; i < 4; ++i) {
        for (int j = 3; j < 12; ++j) {
            EXPECT_DOUBLE_EQ(h_x(i, j), h_x_before(i, j));
        }
    }
}

// 14. No-op safety on empty / small matrices.
TEST(ObservabilityGuard, ZeroTranslationColumns_EmptyOk) {
    Eigen::MatrixXd empty(0, 12);
    ZeroTranslationColumns(empty);  // must not crash
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 12);

    // Matrix with fewer than 3 columns — function early-returns.
    Eigen::MatrixXd narrow(5, 2);
    narrow.setOnes();
    const Eigen::MatrixXd narrow_before = narrow;
    ZeroTranslationColumns(narrow);
    EXPECT_TRUE(narrow.isApprox(narrow_before))
        << "ZeroTranslationColumns must be a no-op when cols < 3.";
}

// ═════════════════════════════════════════════════════════════════════════
// Group D — Mode parsing
// ═════════════════════════════════════════════════════════════════════════

// 15. Every valid mode name round-trips.
TEST(ObservabilityGuard, ParseMode_AcceptsAllThreeNames) {
    ObservabilityGuardMode m = ObservabilityGuardMode::kSkipUpdate;
    ASSERT_TRUE(ParseObservabilityGuardMode("ignore", m));
    EXPECT_EQ(m, ObservabilityGuardMode::kIgnore);
    ASSERT_TRUE(ParseObservabilityGuardMode("skip_position", m));
    EXPECT_EQ(m, ObservabilityGuardMode::kSkipPosition);
    ASSERT_TRUE(ParseObservabilityGuardMode("skip_update", m));
    EXPECT_EQ(m, ObservabilityGuardMode::kSkipUpdate);
}

// 16. Unknown / mistyped mode strings are rejected without mutating the out.
TEST(ObservabilityGuard, ParseMode_RejectsUnknown) {
    ObservabilityGuardMode m = ObservabilityGuardMode::kSkipUpdate;
    EXPECT_FALSE(ParseObservabilityGuardMode("", m));
    EXPECT_FALSE(ParseObservabilityGuardMode("Ignore", m));           // case sensitive
    EXPECT_FALSE(ParseObservabilityGuardMode("skip", m));
    EXPECT_FALSE(ParseObservabilityGuardMode("nonsense", m));
    // Original value preserved on rejection (important so a bad key can't
    // silently flip the gate).
    EXPECT_EQ(m, ObservabilityGuardMode::kSkipUpdate);
}

// 17. ModeName is the inverse of ParseObservabilityGuardMode for every
//     valid mode.
TEST(ObservabilityGuard, ModeName_RoundTrip) {
    for (auto m : {ObservabilityGuardMode::kIgnore,
                   ObservabilityGuardMode::kSkipPosition,
                   ObservabilityGuardMode::kSkipUpdate}) {
        const std::string name = ObservabilityGuardModeName(m);
        ObservabilityGuardMode round = ObservabilityGuardMode::kSkipUpdate;
        ASSERT_TRUE(ParseObservabilityGuardMode(name, round))
            << "ObservabilityGuardModeName returned a string ("
            << name << ") that ParseObservabilityGuardMode rejects.";
        EXPECT_EQ(round, m);
    }
}
