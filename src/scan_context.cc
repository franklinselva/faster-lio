#include "faster_lio/scan_context.h"

#include <cmath>
#include <limits>

namespace faster_lio {

namespace {
constexpr double kTwoPi = 2.0 * M_PI;
// Empty-cell sentinel. Z values below this are treated as "no point in bin".
// We pick a value below any physical scene z so a real point always wins.
constexpr double kEmptyCell = -1e6;
}  // namespace

ScanContext::Descriptor ScanContext::ComputeDescriptor(
    const pcl::PointCloud<pcl::PointXYZI> &cloud, const Options &opts) {
    Descriptor desc =
        Descriptor::Constant(opts.num_rings, opts.num_sectors, kEmptyCell);
    if (cloud.empty() || opts.num_rings <= 0 || opts.num_sectors <= 0 ||
        opts.max_range <= 0.0) {
        desc.setZero();
        return desc;
    }

    const double ring_step   = opts.max_range / opts.num_rings;
    const double sector_step = kTwoPi / opts.num_sectors;

    for (const auto &p : cloud.points) {
        const double r = std::sqrt(static_cast<double>(p.x) * p.x +
                                    static_cast<double>(p.y) * p.y);
        if (r >= opts.max_range) continue;
        double theta = std::atan2(static_cast<double>(p.y), static_cast<double>(p.x));
        if (theta < 0.0) theta += kTwoPi;  // [0, 2π)

        int ring_idx   = static_cast<int>(r / ring_step);
        int sector_idx = static_cast<int>(theta / sector_step);
        // Defensive clamping against floating-point overruns at the boundary.
        if (ring_idx   >= opts.num_rings)   ring_idx   = opts.num_rings   - 1;
        if (sector_idx >= opts.num_sectors) sector_idx = opts.num_sectors - 1;
        if (sector_idx < 0) sector_idx = 0;

        const double z = static_cast<double>(p.z);
        if (z > desc(ring_idx, sector_idx)) {
            desc(ring_idx, sector_idx) = z;
        }
    }

    // Replace empty cells with 0. This lets the cosine-distance comparison
    // treat empty cells as a null contribution via the squaredNorm == 0 gate
    // in Distance(), while keeping the matrix bounded.
    for (int i = 0; i < desc.rows(); ++i) {
        for (int j = 0; j < desc.cols(); ++j) {
            if (desc(i, j) <= kEmptyCell + 1.0) desc(i, j) = 0.0;
        }
    }

    return desc;
}

ScanContext::RingKey ScanContext::ComputeRingKey(const Descriptor &desc) {
    RingKey key = RingKey::Zero(desc.rows());
    if (desc.cols() == 0) return key;
    const double inv_cols = 1.0 / static_cast<double>(desc.cols());
    for (int r = 0; r < desc.rows(); ++r) {
        int non_empty = 0;
        for (int c = 0; c < desc.cols(); ++c) {
            if (desc(r, c) != 0.0) ++non_empty;
        }
        key(r) = static_cast<double>(non_empty) * inv_cols;
    }
    return key;
}

ScanContext::MatchResult ScanContext::Distance(const Descriptor &a,
                                                const Descriptor &b,
                                                const MatchOptions &opts) {
    MatchResult result;
    result.score = 1.0;
    result.shift = 0;
    result.valid_columns = 0;

    if (a.rows() == 0 || a.rows() != b.rows() || a.cols() != b.cols()) {
        return result;
    }

    const int S = static_cast<int>(a.cols());
    const int min_valid = static_cast<int>(
        std::ceil(opts.min_overlap_ratio * static_cast<double>(S)));

    // Precompute column norms so the inner loop doesn't recompute them.
    Eigen::VectorXd a_norms(S);
    Eigen::VectorXd b_norms(S);
    for (int c = 0; c < S; ++c) {
        a_norms(c) = a.col(c).norm();
        b_norms(c) = b.col(c).norm();
    }

    double best_score = std::numeric_limits<double>::infinity();
    int    best_shift = 0;
    int    best_valid = 0;

    for (int s = 0; s < S; ++s) {
        double sum = 0.0;
        int    valid = 0;
        for (int c = 0; c < S; ++c) {
            const int cb = (c + s) % S;
            const double na = a_norms(c);
            const double nb = b_norms(cb);
            // Only count columns where BOTH descriptors have content.
            // Averaging over non-empty-overlapping columns makes the score
            // robust to sparse descriptors — but only meaningful when the
            // overlap is a non-trivial fraction of the total.
            if (na < 1e-9 || nb < 1e-9) continue;
            const double cos_sim = a.col(c).dot(b.col(cb)) / (na * nb);
            sum += 1.0 - cos_sim;
            ++valid;
        }
        if (valid < min_valid) continue;  // discard: too little overlap to trust
        const double score = sum / static_cast<double>(valid);
        if (score < best_score) {
            best_score = score;
            best_shift = s;
            best_valid = valid;
        }
    }

    if (best_score == std::numeric_limits<double>::infinity()) {
        return result;  // score = 1.0 (default) — no sufficient overlap
    }
    result.score = best_score;
    result.shift = best_shift;
    result.valid_columns = best_valid;
    return result;
}

ScanContext::MatchResult ScanContext::Distance(const Descriptor &a,
                                                const Descriptor &b) {
    return Distance(a, b, MatchOptions{});
}

double ScanContext::RingKeyDistance(const RingKey &a, const RingKey &b) {
    if (a.size() != b.size() || a.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return (a - b).norm();
}

}  // namespace faster_lio
