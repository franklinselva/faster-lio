// Pure-function tests for Normalized Innovation Squared (NIS) consistency
// diagnostic. Pins the math + aggregator invariants so integration into
// ObsModel can assume the helpers are correct.

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include <cmath>
#include <limits>
#include <random>

#include "faster_lio/nis.h"

using faster_lio::nis::ComputeScalarNIS;
using faster_lio::nis::NISAggregator;

namespace {

using Mat12 = Eigen::Matrix<double, 12, 12>;
using Row12 = Eigen::Matrix<double, 1, 12>;

// Unit plane normal along +Z packed into an H row. Translation block
// (cols 0..2) gets (0, 0, 1); other blocks zero. This is what a
// horizontal floor observation looks like in the FAST-LIO jacobian
// convention.
Row12 FloorNormalH() {
    Row12 h = Row12::Zero();
    h(0, 2) = 1.0;
    return h;
}

// Identity P scaled by sigma² — all components equally uncertain at
// variance sigma². Useful for controlled innovation-covariance tests.
Mat12 IsoP(double sigma2) {
    Mat12 P = Mat12::Identity() * sigma2;
    return P;
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════
// ComputeScalarNIS — mathematical correctness
// ═══════════════════════════════════════════════════════════════════════

// A 1-sigma residual given the innovation covariance should produce NIS=1.
// Sanity-check the formula: NIS = r² / (H·P·Hᵀ + R).
TEST(ComputeScalarNIS, OneSigmaResidual_GivesOne) {
    const auto h = FloorNormalH();
    const auto P = IsoP(0.01);                 // σ² = 0.01 on each state component
    const double R = 0.01;                     // observation noise variance 0.01
    // H·P·Hᵀ = 0.01 (only the Z translation element), + R = 0.02.
    // Residual = sqrt(0.02) → NIS = 0.02 / 0.02 = 1.
    const double S = (h * P * h.transpose())(0, 0) + R;
    const double residual = std::sqrt(S);
    EXPECT_NEAR(ComputeScalarNIS(residual, h, P, R), 1.0, 1e-12);
}

// Residual twice the 1-σ size → NIS = 4 (quadratic).
TEST(ComputeScalarNIS, TwoSigmaResidual_GivesFour) {
    const auto h = FloorNormalH();
    const auto P = IsoP(0.01);
    const double R = 0.01;
    const double S = (h * P * h.transpose())(0, 0) + R;
    const double residual = 2.0 * std::sqrt(S);
    EXPECT_NEAR(ComputeScalarNIS(residual, h, P, R), 4.0, 1e-12);
}

// Zero residual → NIS = 0 regardless of covariance.
TEST(ComputeScalarNIS, ZeroResidual_IsZero) {
    EXPECT_DOUBLE_EQ(ComputeScalarNIS(0.0, FloorNormalH(), IsoP(1.0), 1.0), 0.0);
    EXPECT_DOUBLE_EQ(ComputeScalarNIS(0.0, FloorNormalH(), IsoP(1e-6), 1e-6), 0.0);
}

// Sign of residual doesn't matter (squared).
TEST(ComputeScalarNIS, SignOfResidualIrrelevant) {
    const auto h = FloorNormalH();
    const auto P = IsoP(0.01);
    EXPECT_DOUBLE_EQ(
        ComputeScalarNIS(+0.1, h, P, 0.01),
        ComputeScalarNIS(-0.1, h, P, 0.01));
}

// When P is tight (near-zero) and R dominates, NIS collapses toward r²/R —
// the filter ignores state uncertainty and trusts the observation model.
TEST(ComputeScalarNIS, TightCovariance_ReducesToResidualOverR) {
    const auto h = FloorNormalH();
    const Mat12 P_tiny = IsoP(1e-12);  // effectively zero state uncertainty
    const double R = 0.01;
    const double residual = 0.05;      // σ_obs = 0.1, 0.5σ → NIS = 0.25
    const double nis = ComputeScalarNIS(residual, h, P_tiny, R);
    EXPECT_NEAR(nis, (residual * residual) / R, 1e-9);
}

// When P is huge and R is tiny, NIS → r² / (H·P·Hᵀ) — the filter's own
// uncertainty dominates. Over-inflated Q matches this regime.
TEST(ComputeScalarNIS, LooseCovariance_IsDominatedByHPH) {
    const auto h = FloorNormalH();
    const auto P_huge = IsoP(1.0);     // σ² = 1 on each component
    const double R = 1e-6;
    const double residual = 1.0;       // 1 σ given H·P·Hᵀ ≈ 1
    const double nis = ComputeScalarNIS(residual, h, P_huge, R);
    EXPECT_NEAR(nis, 1.0, 1e-4);
}

// Over-inflated Q (our old config) produces tiny NIS — proof the filter
// is underconfident / padding too much. This is the regime our baseline
// `process_noise.na = 0.1` was likely in.
TEST(ComputeScalarNIS, OverInflatedCovariance_ProducesTinyNIS) {
    const auto h = FloorNormalH();
    const auto P = IsoP(10.0);         // absurd — proxy for huge Q after
                                        // predict has blown P up
    const double R = 0.01;
    const double residual = 0.1;       // modest residual
    const double nis = ComputeScalarNIS(residual, h, P, R);
    EXPECT_LT(nis, 0.01)
        << "Huge P should produce NIS ≪ 1 (filter thinks everything is "
        << "consistent with noise); got " << nis;
}

// ═══════════════════════════════════════════════════════════════════════
// Defensive behaviour — degenerate inputs must never leak NaN / Inf.
// ═══════════════════════════════════════════════════════════════════════

TEST(ComputeScalarNIS, NegativeR_Rejects) {
    EXPECT_EQ(ComputeScalarNIS(0.1, FloorNormalH(), IsoP(1.0), -1.0), 0.0);
}

TEST(ComputeScalarNIS, ZeroR_AndZeroHPH_Rejects) {
    // Both S components zero → S = 0 → would divide by zero. Must reject.
    Row12 h_zero = Row12::Zero();
    Mat12 P_zero = Mat12::Zero();
    EXPECT_EQ(ComputeScalarNIS(0.1, h_zero, P_zero, 0.0), 0.0);
}

TEST(ComputeScalarNIS, NaNResidual_Rejects) {
    EXPECT_EQ(ComputeScalarNIS(std::nan(""), FloorNormalH(), IsoP(1.0), 1.0), 0.0);
}

TEST(ComputeScalarNIS, NaNInH_Rejects) {
    Row12 h = FloorNormalH();
    h(0, 0) = std::nan("");
    EXPECT_EQ(ComputeScalarNIS(0.1, h, IsoP(1.0), 1.0), 0.0);
}

TEST(ComputeScalarNIS, NaNInP_Rejects) {
    Mat12 P = IsoP(1.0);
    P(5, 5) = std::nan("");
    EXPECT_EQ(ComputeScalarNIS(0.1, FloorNormalH(), P, 1.0), 0.0);
}

TEST(ComputeScalarNIS, InfResidual_Rejects) {
    EXPECT_EQ(
        ComputeScalarNIS(std::numeric_limits<double>::infinity(),
                         FloorNormalH(), IsoP(1.0), 1.0),
        0.0);
}

// ═══════════════════════════════════════════════════════════════════════
// NISAggregator — running-statistics invariants.
// ═══════════════════════════════════════════════════════════════════════

TEST(NISAggregator, EmptyStartsAtZero) {
    NISAggregator agg;
    EXPECT_EQ(agg.count(), 0);
    EXPECT_EQ(agg.mean(), 0.0);
    EXPECT_EQ(agg.max(), 0.0);
}

TEST(NISAggregator, SingleValueRoundTrips) {
    NISAggregator agg;
    agg.Push(1.0);
    EXPECT_EQ(agg.count(), 1);
    EXPECT_DOUBLE_EQ(agg.mean(), 1.0);
    EXPECT_DOUBLE_EQ(agg.max(), 1.0);
}

TEST(NISAggregator, MeanIsArithmetic) {
    NISAggregator agg;
    agg.Push(0.5);
    agg.Push(1.0);
    agg.Push(1.5);
    EXPECT_EQ(agg.count(), 3);
    EXPECT_DOUBLE_EQ(agg.mean(), 1.0);  // (0.5+1.0+1.5)/3 = 1.0
}

TEST(NISAggregator, MaxTracksHighest) {
    NISAggregator agg;
    agg.Push(0.3);
    agg.Push(5.0);
    agg.Push(1.1);
    EXPECT_DOUBLE_EQ(agg.max(), 5.0);
}

TEST(NISAggregator, ResetClearsState) {
    NISAggregator agg;
    for (double v : {1.0, 2.0, 3.0}) agg.Push(v);
    agg.Reset();
    EXPECT_EQ(agg.count(), 0);
    EXPECT_EQ(agg.mean(), 0.0);
    EXPECT_EQ(agg.max(), 0.0);
}

TEST(NISAggregator, IgnoresInvalidValues) {
    NISAggregator agg;
    agg.Push(1.0);
    agg.Push(std::nan(""));                                   // ignored
    agg.Push(std::numeric_limits<double>::infinity());        // ignored
    agg.Push(-0.5);                                           // ignored (NIS ≥ 0)
    agg.Push(2.0);
    EXPECT_EQ(agg.count(), 2);
    EXPECT_DOUBLE_EQ(agg.mean(), 1.5);
    EXPECT_DOUBLE_EQ(agg.max(), 2.0);
}

// A realistic scenario: many points per frame with a known χ²(1)-ish
// distribution. Mean NIS from a well-calibrated filter should converge
// toward 1. This pins the interpretation that's the whole point of
// wiring NIS into the diagnostics.
TEST(NISAggregator, ChiSquareOneMeansConvergesToOne) {
    NISAggregator agg;
    // 10000 IID standard-normal squared values → classic χ²(1). Variance
    // of the SAMPLE MEAN is 2/N = 2e-4, std ≈ 0.014. Tolerance 0.1 is
    // ~7σ — comfortably above stochastic flakiness. Seed fixed for
    // reproducibility.
    std::mt19937 rng(42);
    std::normal_distribution<double> N(0.0, 1.0);
    for (int i = 0; i < 10000; ++i) {
        const double z = N(rng);
        agg.Push(z * z);                                       // χ²(1)
    }
    EXPECT_EQ(agg.count(), 10000);
    EXPECT_NEAR(agg.mean(), 1.0, 0.1)
        << "χ²(1) mean should converge to 1 — aggregator is mis-adding.";
}
