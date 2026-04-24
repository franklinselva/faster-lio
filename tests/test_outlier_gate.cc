// ─────────────────────────────────────────────────────────────────────────
// test_outlier_gate.cc
//
// Pure-function tests for the LiDAR outlier gates. Pins the math before
// it ships inside ObsModel so any later regression (or refactor) shows
// up here before burning debug cycles in the full pipeline.
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "faster_lio/outlier_gate.h"

using faster_lio::outlier_gate::AcceptEither;
using faster_lio::outlier_gate::AcceptMahalanobis;
using faster_lio::outlier_gate::AcceptRange;

// ═════════════════════════════════════════════════════════════════════════
// Range gate
// ═════════════════════════════════════════════════════════════════════════

// Accept when range > 81 × residual². At 5 m range with 0.05 m residual:
//   81 × 0.05² = 0.2025 m < 5 m → accepted.
TEST(OutlierGateRange, AcceptsTypicalValidPoint) {
    EXPECT_TRUE(AcceptRange(/*range=*/5.0, /*residual=*/0.05));
}

// Reject when residual is large enough that 81 × residual² > range.
// At 1 m range with 0.2 m residual: 81 × 0.04 = 3.24 > 1 → rejected.
TEST(OutlierGateRange, RejectsLargeResidualAtCloseRange) {
    EXPECT_FALSE(AcceptRange(/*range=*/1.0, /*residual=*/0.2));
}

// Sign of residual must not matter — the gate works on magnitude.
TEST(OutlierGateRange, NegativeResidualHandledBySquaring) {
    EXPECT_EQ(AcceptRange(5.0, +0.05), AcceptRange(5.0, -0.05));
    EXPECT_EQ(AcceptRange(1.0, +0.2),  AcceptRange(1.0, -0.2));
}

// Boundary: range == 81 × residual² should REJECT (the gate uses strict >).
TEST(OutlierGateRange, BoundaryIsRejected) {
    const double residual = 0.1;
    const double range_at_boundary = 81.0 * residual * residual;  // 0.81 m
    EXPECT_FALSE(AcceptRange(range_at_boundary, residual));
    // Epsilon over → accepted.
    EXPECT_TRUE(AcceptRange(range_at_boundary + 1e-9, residual));
}

// A tighter ratio (e.g. 25 = 5²) makes the gate more permissive — same
// residual accepted at shorter range.
TEST(OutlierGateRange, LooserRatioAcceptsShorterRange) {
    const double residual = 0.1;
    const double range = 0.3;
    EXPECT_FALSE(AcceptRange(range, residual, /*ratio=*/81.0));  // 0.81 > 0.3
    EXPECT_TRUE (AcceptRange(range, residual, /*ratio=*/25.0));  // 0.25 < 0.3
}

// Degenerate inputs — zero, negative, NaN, Inf must reject (never accept).
TEST(OutlierGateRange, DegenerateInputsReject) {
    EXPECT_FALSE(AcceptRange(0.0,       0.1));     // zero range
    EXPECT_FALSE(AcceptRange(-1.0,      0.1));     // negative range
    EXPECT_FALSE(AcceptRange(std::nan(""), 0.1));  // NaN range
    EXPECT_FALSE(AcceptRange(5.0, std::nan("")));  // NaN residual
    EXPECT_FALSE(AcceptRange(5.0, 0.1, /*ratio=*/0.0));     // zero ratio
    EXPECT_FALSE(AcceptRange(5.0, 0.1, /*ratio=*/-1.0));    // negative ratio
    EXPECT_FALSE(AcceptRange(std::numeric_limits<double>::infinity(), 0.1));
}

// ═════════════════════════════════════════════════════════════════════════
// Mahalanobis gate
// ═════════════════════════════════════════════════════════════════════════

// A residual at the 1-σ level → mahal² = 1, which is ≤ 6.63 → accept.
TEST(OutlierGateMahalanobis, AcceptsOneSigmaResidual) {
    // innovation_std = 0.05 m → innovation_var = 2.5e-3
    // residual = 0.05 m → mahal² = 1.0
    EXPECT_TRUE(AcceptMahalanobis(/*residual=*/0.05, /*innovation_var=*/2.5e-3));
}

// A residual at the 3-σ level → mahal² = 9, above 6.63 default → reject.
TEST(OutlierGateMahalanobis, RejectsThreeSigmaResidualAtDefaultThreshold) {
    EXPECT_FALSE(AcceptMahalanobis(/*residual=*/0.15, /*innovation_var=*/2.5e-3));
}

// Threshold 10.83 (99.9%) is more permissive — the same 3-σ residual passes.
TEST(OutlierGateMahalanobis, LooserThresholdAcceptsThreeSigma) {
    EXPECT_TRUE(AcceptMahalanobis(0.15, 2.5e-3, /*chi2_threshold=*/10.83));
}

// Covariance dependence: large innovation variance accepts bigger
// residuals. At innovation_std=0.5 m, a 0.3 m residual is within 1σ.
TEST(OutlierGateMahalanobis, LargerCovarianceIsMorePermissive) {
    const double residual = 0.3;
    EXPECT_FALSE(AcceptMahalanobis(residual, /*innovation_var=*/1.0e-3));  // tight
    EXPECT_TRUE (AcceptMahalanobis(residual, /*innovation_var=*/0.25));    // loose
}

// This is THE property distinguishing Mahalanobis from Range: filter
// over-confidence (small innov_var) rejects observations that the Range
// gate would accept at long ranges.
TEST(OutlierGateMahalanobis, RejectsWhenFilterOverConfident) {
    // Scenario: long-range point (range = 20 m), small residual (0.15 m),
    // but the filter's covariance is tight (innov_std ≈ 0.02 m).
    const double range      = 20.0;
    const double residual   = 0.15;
    const double innov_var  = 4.0e-4;  // innov_std = 0.02 m

    // Range gate: 81 × 0.15² = 1.82 < 20 → ACCEPT.
    EXPECT_TRUE(AcceptRange(range, residual));
    // Mahalanobis: mahal² = 0.15² / 4e-4 = 56.25 ≫ 6.63 → REJECT.
    EXPECT_FALSE(AcceptMahalanobis(residual, innov_var));
}

// Symmetry: sign of residual must not affect Mahalanobis either.
TEST(OutlierGateMahalanobis, NegativeResidualHandledBySquaring) {
    EXPECT_EQ(AcceptMahalanobis(+0.05, 2.5e-3), AcceptMahalanobis(-0.05, 2.5e-3));
    EXPECT_EQ(AcceptMahalanobis(+0.15, 2.5e-3), AcceptMahalanobis(-0.15, 2.5e-3));
}

// Boundary: mahal² == threshold should ACCEPT (uses <=).
TEST(OutlierGateMahalanobis, BoundaryIsAccepted) {
    const double chi2 = 6.63;
    const double var  = 1.0e-3;
    const double residual_boundary = std::sqrt(chi2 * var);  // exactly at threshold
    EXPECT_TRUE(AcceptMahalanobis(residual_boundary, var, chi2));
    // Epsilon beyond → rejected.
    EXPECT_FALSE(AcceptMahalanobis(residual_boundary + 1e-6, var, chi2));
}

// Zero residual is the ideal observation; always accepted for any
// positive variance and threshold.
TEST(OutlierGateMahalanobis, ZeroResidualAccepted) {
    EXPECT_TRUE(AcceptMahalanobis(0.0, 1e-6));
    EXPECT_TRUE(AcceptMahalanobis(0.0, 1.0));
    EXPECT_TRUE(AcceptMahalanobis(0.0, 1e-12));
}

// Degenerate inputs — zero / negative / NaN innovation variance must
// reject, not pass-through nor crash.
TEST(OutlierGateMahalanobis, DegenerateInputsReject) {
    EXPECT_FALSE(AcceptMahalanobis(0.1,  0.0));                     // zero var
    EXPECT_FALSE(AcceptMahalanobis(0.1, -1.0));                     // negative var
    EXPECT_FALSE(AcceptMahalanobis(0.1,  std::nan("")));            // NaN var
    EXPECT_FALSE(AcceptMahalanobis(std::nan(""), 1.0));             // NaN residual
    EXPECT_FALSE(AcceptMahalanobis(0.1, 1.0, /*threshold=*/-1.0));  // negative threshold
    EXPECT_FALSE(AcceptMahalanobis(std::numeric_limits<double>::infinity(), 1.0));
    EXPECT_FALSE(AcceptMahalanobis(0.1, std::numeric_limits<double>::infinity(), std::nan("")));
}

// ═════════════════════════════════════════════════════════════════════════
// Hybrid Either gate — accept if EITHER gate accepts.
// ═════════════════════════════════════════════════════════════════════════

// Range accepts, Mahalanobis rejects → Either accepts.
TEST(OutlierGateEither, AcceptsWhenRangeAlonePasses) {
    // Same scenario as RejectsWhenFilterOverConfident above.
    const double range     = 20.0;
    const double residual  = 0.15;
    const double innov_var = 4.0e-4;
    ASSERT_TRUE (AcceptRange      (range, residual));
    ASSERT_FALSE(AcceptMahalanobis(residual, innov_var));
    EXPECT_TRUE (AcceptEither     (range, residual, innov_var));
}

// Range rejects, Mahalanobis accepts → Either accepts.
// Construct: close range + large residual (range gate rejects) but
// filter is uncertain enough that Mahalanobis passes.
TEST(OutlierGateEither, AcceptsWhenMahalanobisAlonePasses) {
    const double range     = 0.5;    // close
    const double residual  = 0.2;    // large → 81 × 0.04 = 3.24 > 0.5 → range REJECTS
    const double innov_var = 0.1;    // loose → 0.2² / 0.1 = 0.4 ≤ 6.63 → mahal ACCEPTS
    ASSERT_FALSE(AcceptRange      (range, residual));
    ASSERT_TRUE (AcceptMahalanobis(residual, innov_var));
    EXPECT_TRUE (AcceptEither     (range, residual, innov_var));
}

// Both reject → Either rejects.
TEST(OutlierGateEither, RejectsWhenBothReject) {
    const double range     = 0.1;
    const double residual  = 0.3;
    const double innov_var = 1.0e-4;
    ASSERT_FALSE(AcceptRange      (range, residual));
    ASSERT_FALSE(AcceptMahalanobis(residual, innov_var));
    EXPECT_FALSE(AcceptEither     (range, residual, innov_var));
}

// Both accept → Either accepts (no surprises).
TEST(OutlierGateEither, AcceptsWhenBothAccept) {
    EXPECT_TRUE(AcceptEither(/*range=*/5.0, /*residual=*/0.05, /*innov_var=*/2.5e-3));
}
