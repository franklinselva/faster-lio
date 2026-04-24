#ifndef FASTER_LIO_OUTLIER_GATE_H
#define FASTER_LIO_OUTLIER_GATE_H

// Outlier rejection for per-point LiDAR observations in the IEKF update.
//
// Two gate flavors:
//
//   Range gate (legacy, upstream FAST-LIO): accept iff
//       range > ratio * residual²
//     where `range = ||p_body||` and `residual = pd2`. Cheap and rank-
//     preserving — large-residual-at-close-range points fail. But it's a
//     heuristic with no statistical interpretation and ignores the
//     filter's own uncertainty.
//
//   Mahalanobis gate (principled): accept iff
//       residual² / innovation_var <= chi2_threshold
//     where `innovation_var = H · P · Hᵀ + R` is the posterior variance
//     of the scalar observation. This treats each observation as a
//     χ²-distributed standard normal and drops the tail.
//
// Both gates are pure functions — no iVox / state coupling — so the math
// can be pinned by direct unit tests before being plumbed into ObsModel.
//
// An `Either` helper combines both conservatively (accept if EITHER
// gate accepts). Useful in practice because Mahalanobis alone can be
// overly strict when the filter's covariance is artificially tight
// (over-confidence), while Range alone ignores covariance entirely.

#include <cmath>

namespace faster_lio {
namespace outlier_gate {

/// Upstream FAST-LIO range-based gate.
/// @param range Sensor-to-point distance (||p_body||), m. Must be > 0.
/// @param residual Point-to-plane residual pd2, m. Sign is ignored.
/// @param ratio Gate tightness. Default 81 = 9² (upstream constant).
/// @return true iff the observation should be kept.
inline bool AcceptRange(double range, double residual, double ratio = 81.0) {
    if (!std::isfinite(range) || !std::isfinite(residual) || !std::isfinite(ratio)) return false;
    if (range <= 0.0 || ratio <= 0.0) return false;
    return range > ratio * residual * residual;
}

/// Mahalanobis χ² gate for a scalar observation.
/// @param residual Scalar innovation (observed − predicted). Sign ignored.
/// @param innovation_var H · P · Hᵀ + R (scalar, strictly positive).
///   - H is the observation jacobian.
///   - P is the (prior, this-iteration) state covariance.
///   - R is the observation noise variance.
/// @param chi2_threshold Upper bound on residual² / innovation_var.
///   Default 6.63 = χ² 1-DoF 99%. Common alternatives: 3.84 (95%),
///   10.83 (99.9%). Caller chooses based on how forgiving they want to
///   be to outliers vs. unmodeled covariance error.
/// @return true iff the normalised squared innovation is within threshold.
inline bool AcceptMahalanobis(double residual, double innovation_var,
                              double chi2_threshold = 6.63) {
    if (!std::isfinite(residual) || !std::isfinite(innovation_var) ||
        !std::isfinite(chi2_threshold)) {
        return false;
    }
    if (innovation_var <= 0.0 || chi2_threshold < 0.0) return false;
    return (residual * residual) <= chi2_threshold * innovation_var;
}

/// Convenience: accept if EITHER gate accepts. Use when you want the
/// range gate's range-proportional tolerance AND the Mahalanobis gate's
/// covariance sensitivity, and a point passing either heuristic is
/// acceptable. More permissive than either alone.
inline bool AcceptEither(double range, double residual, double innovation_var,
                         double range_ratio = 81.0,
                         double chi2_threshold = 6.63) {
    return AcceptRange(range, residual, range_ratio) ||
           AcceptMahalanobis(residual, innovation_var, chi2_threshold);
}

}  // namespace outlier_gate
}  // namespace faster_lio

#endif  // FASTER_LIO_OUTLIER_GATE_H
