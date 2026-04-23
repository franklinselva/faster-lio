#ifndef FASTER_LIO_SCAN_CONTEXT_H
#define FASTER_LIO_SCAN_CONTEXT_H

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>

namespace faster_lio {

/// Scan Context global descriptor (Kim & Kim 2018) for pose-independent
/// loop-closure candidate matching.
///
/// A `Descriptor` is an N_rings × N_sectors matrix. For each point in the
/// input cloud (expressed in a local frame whose origin is the descriptor
/// center), we bin by (radial distance → ring, azimuth → sector) and store
/// the maximum Z value per bin. Comparing two descriptors searches every
/// cyclic column shift: the minimum column-averaged cosine distance is the
/// match score, and the shift index recovers the relative yaw angle.
///
/// Adaptation for solid-state LiDAR: a single 70°-wide Avia scan can't
/// populate all sectors. The integration layer must pass an AGGREGATED
/// cloud (several keyframes' worth of scans, body-rotating during walking)
/// so the descriptor is not a narrow wedge. That aggregation is not the
/// concern of this file — here we assume the caller has already shaped
/// the input appropriately.
///
/// Typical parameter ranges (paper + LIO-SAM):
///   num_rings   = 20    (radial bins)
///   num_sectors = 60    (azimuth bins, 6° each)
///   max_range   = 80 m  (beyond → ignored)
class ScanContext {
   public:
    using Descriptor = Eigen::MatrixXd;   // [num_rings × num_sectors]
    using RingKey    = Eigen::VectorXd;   // [num_rings]

    struct Options {
        int    num_rings   = 20;
        int    num_sectors = 60;
        double max_range   = 80.0;     // m
    };

    struct MatchOptions {
        /// Minimum fraction of sectors that must be non-empty in BOTH
        /// descriptors for the match to count. Sparse descriptors (narrow
        /// FoV) would otherwise produce spurious perfect matches from
        /// ≤2 coincident bins.
        double min_overlap_ratio = 0.30;
    };

    /// Build a descriptor from a cloud expressed in the frame the descriptor
    /// is anchored in. Only (x, y) radial/azimuth are used; z is binned as
    /// the cell value via max-height aggregation. Points beyond `max_range`
    /// in the xy plane are dropped.
    static Descriptor ComputeDescriptor(
        const pcl::PointCloud<pcl::PointXYZI> &cloud, const Options &opts);

    /// Rotation-invariant 1-D hash: ring_key[r] = fraction of non-empty
    /// columns in ring r. Cheap to compare across a large candidate pool.
    static RingKey ComputeRingKey(const Descriptor &desc);

    /// Cyclic column-shift minimum cosine distance. Returns {score, shift}
    /// where `shift ∈ [0, num_sectors)` is the column shift that best
    /// aligns `b` onto `a`. Score is in [0, 1]: 0 = identical, 1 = orthogonal.
    ///
    /// The recovered yaw of b relative to a is `shift * 2π / num_sectors`.
    /// (Rotating b by -that angle would make it match a.)
    struct MatchResult {
        double score = 1.0;
        int    shift = 0;
        int    valid_columns = 0;  ///< # sectors that contributed to the score
    };
    static MatchResult Distance(const Descriptor &a, const Descriptor &b,
                                const MatchOptions &opts);
    /// Overload with default MatchOptions{} — convenient for tests and
    /// callers that don't need to tune the overlap threshold.
    static MatchResult Distance(const Descriptor &a, const Descriptor &b);

    /// L2 distance between two ring keys. Invariant under column shifts;
    /// use as a cheap pre-filter before the full Distance() column search.
    static double RingKeyDistance(const RingKey &a, const RingKey &b);
};

}  // namespace faster_lio

#endif  // FASTER_LIO_SCAN_CONTEXT_H
