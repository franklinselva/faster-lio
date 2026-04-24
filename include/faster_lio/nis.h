#ifndef FASTER_LIO_NIS_H
#define FASTER_LIO_NIS_H

// Normalized Innovation Squared (NIS) — Kalman filter consistency diagnostic.
//
// For a scalar observation z with innovation r = z − H·x̂ and innovation
// covariance S = H·P·Hᵀ + R, define
//
//     NIS = r² / S
//
// For a well-calibrated linear-Gaussian Kalman filter, NIS follows χ²(n)
// where n is the observation dimension. For 1-D LiDAR point-to-plane
// observations n = 1, so E[NIS] = 1. Aggregating NIS across many points
// per frame gives a direct empirical check on filter consistency:
//
//   mean_NIS ≈ 1  →  Q and R are calibrated.
//   mean_NIS ≪ 1  →  filter is UNDERconfident (Q or R too large → S too
//                    large → predicted residuals should be larger than
//                    observed → we're over-padding the uncertainty).
//   mean_NIS ≫ 1  →  filter is OVERconfident (Q or R too small →
//                    observations look surprising → real state error
//                    exceeds filter's P → filter diverges unless pulled
//                    back hard).
//
// Pure-function helpers — no state, no Eigen class members. Lets
// diagnostics be unit-tested directly and lets the per-frame aggregator
// stay a trivial POD.
//
// Reference: Bar-Shalom, Li, Kirubarajan, "Estimation with Applications
// to Tracking and Navigation" §5.4 (NIS) + §10.2 (NEES).

#include <Eigen/Dense>

#include <cmath>
#include <vector>

namespace faster_lio {
namespace nis {

/// NIS for a scalar observation with 12-dim jacobian (translation, rotation,
/// extrinsic T/R blocks — the FAST-LIO plane-point H).
///
/// @param residual   Scalar innovation r = z − H·x̂ (sign ignored).
/// @param h          1×12 row jacobian.
/// @param P          12×12 prior-state covariance block.
/// @param R          Observation noise variance (> 0).
/// @return           r² / (H·P·Hᵀ + R). Zero for degenerate inputs
///                   (non-positive variance, non-finite residual, NaN H/P).
inline double ComputeScalarNIS(double residual,
                               const Eigen::Matrix<double, 1, 12> &h,
                               const Eigen::Matrix<double, 12, 12> &P,
                               double R) {
    if (!std::isfinite(residual) || !std::isfinite(R) || R <= 0.0) return 0.0;
    if (!h.allFinite() || !P.allFinite()) return 0.0;

    const double hph = (h * P * h.transpose())(0, 0);
    if (!std::isfinite(hph)) return 0.0;
    const double innovation_var = hph + R;
    if (innovation_var <= 0.0) return 0.0;
    return (residual * residual) / innovation_var;
}

/// Per-frame NIS aggregator. Caller calls Push() for each accepted
/// LiDAR observation in a frame, then reads mean/max/count for logging.
/// Reset() between frames.
///
/// Kept deliberately simple — no variance estimate, no histogram — so
/// it's cheap enough to run in every frame even on a resource-constrained
/// target. Higher-order statistics can be reconstructed offline from the
/// raw per-frame mean / max / count if needed.
struct NISAggregator {
    int    n        = 0;
    double sum      = 0.0;
    double max_nis  = 0.0;

    void Push(double nis_value) {
        if (!std::isfinite(nis_value) || nis_value < 0.0) return;
        ++n;
        sum += nis_value;
        if (nis_value > max_nis) max_nis = nis_value;
    }

    void Reset() {
        n       = 0;
        sum     = 0.0;
        max_nis = 0.0;
    }

    int count() const { return n; }
    double mean() const { return n > 0 ? sum / n : 0.0; }
    double max() const { return max_nis; }
};

}  // namespace nis
}  // namespace faster_lio

#endif  // FASTER_LIO_NIS_H
