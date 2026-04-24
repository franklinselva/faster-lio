#ifndef FASTER_LIO_OBSERVABILITY_GUARD_H
#define FASTER_LIO_OBSERVABILITY_GUARD_H

// Observability-aware degeneracy guard for the per-frame LiDAR observation
// jacobian h_x (N × 12) built inside `LaserMapping::ObsModel`.
//
// Motivation
// ──────────
// On long outdoor walks or textureless scenes, the LiDAR observation matrix
// H stacked from per-feature plane-normal rows becomes rank-deficient:
//   * ground-only scene  → rank 1 on the translation block (only Z-normal)
//   * parallel corridor  → rank 1 or 2 (along-corridor direction unobservable)
//   * empty scene        → rank 0 (already caught by effect_feat_num_ < 1)
// With a rank-deficient H the IEKF legitimately updates the observable
// directions but leaves unobservable ones to integrate on IMU alone. On a
// long corridor that integration leaks into ba and position blows up. The
// guard is a first-line detector + (optional) gate to prevent catastrophic
// drift before the Mahalanobis gate even sees a residual.
//
// Design
// ──────
// Pure functions on Eigen matrices — no state, no iVox coupling — so the
// math can be pinned by direct unit tests. The guard lives OUTSIDE the
// per-point parallel loop and runs once per frame on the stacked N×12
// jacobian, so the cost is one SVD of a small (3×3 or 6×6) matrix.
//
//     AnalyzeJacobian(h_x, threshold)
//       │
//       ├─ Extracts the translation block (columns 0..2) — the plane
//       │   normals stacked as a matrix.
//       ├─ Computes A_t = h_x_trans^T * h_x_trans (3×3 info matrix on
//       │   translation).
//       ├─ SVD on A_t → singular values σ_t = (σ_0, σ_1, σ_2) sorted
//       │   descending. translation_rank = count of σ ≥ threshold.
//       ├─ Same for rotation block (columns 3..5), yielding rotation_rank.
//       └─ Returns ObservabilitySummary{translation_rank, rotation_rank,
//           min_singular_t, min_singular_r}.
//
// Thresholding
// ────────────
// The singular values of the info matrix scale as N × (mean normal magnitude)²,
// which is much larger than 1 for typical scans. A small absolute threshold
// (1e-4) on the info-matrix min-singular correctly classifies empty /
// floor-only / corridor scenarios. The threshold is configurable via YAML
// so unusual sensor geometries (Livox Avia's narrow FOV) can tune it.
//
// Modes
// ─────
//   kIgnore        — analyse + log but don't gate. Baseline; default.
//   kSkipPosition  — when translation rank is below min_translation_rank,
//                    zero out the translation columns (0..2) of h_x so the
//                    IEKF Kalman gain for position is 0 this frame while
//                    rotation still updates. Prevents the position state
//                    from being pulled by an under-constrained jacobian.
//   kSkipUpdate    — hard skip: set ekfom_data.valid = false, the IEKF
//                    falls back to IMU-only prediction this frame.
//
// All three modes return the same ObservabilitySummary so diagnostics are
// uniform.

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <cmath>
#include <string>

namespace faster_lio {

/// Mode selector for the observability guard. Identical naming convention
/// to OutlierGateMode — YAML parser converts string ↔ enum.
enum class ObservabilityGuardMode {
    kIgnore,         // analyse only; don't gate the update
    kSkipPosition,   // zero out translation columns when translation rank < min
    kSkipUpdate,     // skip the IEKF update entirely (pure IMU predict)
};

/// Result of a single-frame observability analysis. All ranks are in [0, 3].
/// min_singular_translation / min_singular_rotation are the smallest
/// singular values of the 3×3 sub-information-matrix on the translation
/// and rotation blocks respectively. Always ≥ 0; 0 when the block is
/// entirely zero (e.g. empty h_x).
struct ObservabilitySummary {
    int    translation_rank       = 0;
    int    rotation_rank          = 0;
    double min_singular_translation = 0.0;
    double min_singular_rotation    = 0.0;
};

/// Parse a YAML string into the mode enum. Returns true on success.
/// Accepts "ignore" / "skip_position" / "skip_update" (case-sensitive).
inline bool ParseObservabilityGuardMode(const std::string &s, ObservabilityGuardMode &out) {
    if      (s == "ignore")        { out = ObservabilityGuardMode::kIgnore;       return true; }
    else if (s == "skip_position") { out = ObservabilityGuardMode::kSkipPosition; return true; }
    else if (s == "skip_update")   { out = ObservabilityGuardMode::kSkipUpdate;   return true; }
    return false;
}

/// Inverse — useful for log messages.
inline const char *ObservabilityGuardModeName(ObservabilityGuardMode m) {
    switch (m) {
        case ObservabilityGuardMode::kIgnore:       return "ignore";
        case ObservabilityGuardMode::kSkipPosition: return "skip_position";
        case ObservabilityGuardMode::kSkipUpdate:   return "skip_update";
    }
    return "unknown";
}

// ───── Internal helpers (inline for header-only testing) ─────────────────

/// Rank of a 3×3 symmetric PSD block against `threshold` (absolute).
/// A singular value σ is counted iff σ ≥ threshold. Matrices with NaN /
/// Inf entries return rank = 0 (defensive — a broken jacobian must not
/// silently claim observability).
inline int RankOf3x3(const Eigen::Matrix3d &A, double threshold, double *min_singular_out) {
    if (!A.allFinite() || threshold < 0.0 || !std::isfinite(threshold)) {
        if (min_singular_out) *min_singular_out = 0.0;
        return 0;
    }
    // JacobiSVD is overkill for 3×3 but keeps the dependency tree minimal
    // and is numerically stable at this scale. ComputeU/V not needed.
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
    const auto sv = svd.singularValues();  // sorted descending, non-negative
    int rank = 0;
    for (int i = 0; i < 3; ++i) {
        if (sv(i) >= threshold) ++rank;
    }
    // Smallest singular value — threshold-independent diagnostic.
    if (min_singular_out) *min_singular_out = sv(2);
    return rank;
}

// ───── Public pure-function API ──────────────────────────────────────────

/// Analyse the observation jacobian `h_x` (N × 12) and report the rank of
/// the translation (columns 0..2) and rotation (columns 3..5) sub-blocks.
///
/// Semantics:
///   * Empty h_x (0 rows) → both ranks 0, both σ_min = 0.
///   * h_x must have exactly 12 columns (the FAST-LIO plane-point jacobian
///     convention). Other widths return all zeros.
///   * Non-finite entries return all zeros (defensive — don't claim
///     observability on a corrupted jacobian).
inline ObservabilitySummary AnalyzeJacobian(const Eigen::MatrixXd &h_x, double threshold) {
    ObservabilitySummary out;
    if (h_x.rows() == 0 || h_x.cols() != 12) return out;

    // Info matrix = Hᵀ·H; per-block we only need the diagonal 3×3 sub-matrices
    // for translation (cols 0..2) and rotation (cols 3..5). Extracting the
    // blocks first keeps the SVD tiny (3×3) regardless of N.
    const auto Ht = h_x.leftCols(3);                 // N×3 translation jacobian
    const auto Hr = h_x.block(0, 3, h_x.rows(), 3);  // N×3 rotation jacobian
    const Eigen::Matrix3d At = Ht.transpose() * Ht;
    const Eigen::Matrix3d Ar = Hr.transpose() * Hr;

    out.translation_rank = RankOf3x3(At, threshold, &out.min_singular_translation);
    out.rotation_rank    = RankOf3x3(Ar, threshold, &out.min_singular_rotation);
    return out;
}

/// Convenience: translation-block rank only.
inline int ComputeJacobianRank(const Eigen::MatrixXd &h_x, double threshold = 1.0e-4) {
    return AnalyzeJacobian(h_x, threshold).translation_rank;
}

/// Convenience: min singular value of the translation block.
inline double SingularValueMinOf(const Eigen::MatrixXd &h_x) {
    if (h_x.rows() == 0 || h_x.cols() != 12 || !h_x.allFinite()) return 0.0;
    const auto Ht = h_x.leftCols(3);
    const Eigen::Matrix3d At = Ht.transpose() * Ht;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(At);
    return svd.singularValues()(2);
}

/// In-place mutation: zero out the translation columns (0..2) of h_x.
/// Semantics: after this call, no Kalman gain is produced for translation
/// state from this frame's observation. Rotation / extrinsic blocks are
/// preserved. Caller keeps the residual vector `h` unchanged — the filter
/// projects the residual onto the nonzero columns of h_x, so residual
/// energy on the (now zeroed) translation columns is discarded.
inline void ZeroTranslationColumns(Eigen::MatrixXd &h_x) {
    if (h_x.cols() < 3) return;
    h_x.leftCols(3).setZero();
}

}  // namespace faster_lio

#endif  // FASTER_LIO_OBSERVABILITY_GUARD_H
