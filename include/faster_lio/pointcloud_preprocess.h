#ifndef FASTER_LIO_POINTCLOUD_PROCESSING_H
#define FASTER_LIO_POINTCLOUD_PROCESSING_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "faster_lio/common_lib.h"

namespace faster_lio {

/// Generic point cloud preprocessor.
///
/// Operates purely on PointXYZINormal clouds (faster-lio's internal type).
/// Sensor-specific format conversion is the caller's responsibility — adapters
/// live outside the library. The preprocessor applies three steps:
///   1. Filter zero-origin points (pre-allocated padding)
///   2. Filter points within the blind distance
///   3. Downsample by point_filter_num (keep every Nth point)
///
/// Per-point timing, if present, should be stored in the `curvature` field
/// as milliseconds from scan start (the same convention the IMU-undistortion
/// stage expects). Clouds without per-point timing should set curvature to 0;
/// SyncPackages will estimate the scan period from inter-scan timestamp gaps.
class PointCloudPreprocess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloudPreprocess() = default;
    ~PointCloudPreprocess() = default;

    /// Run the generic preprocessing pipeline on `cloud` and write the
    /// filtered, downsampled result into `pcl_out`.
    void Process(const PointCloudType &cloud, PointCloudType::Ptr &pcl_out);

    // accessors
    double Blind() const { return blind_; }
    void SetBlind(double v) { blind_ = v; }
    int PointFilterNum() const { return point_filter_num_; }
    void SetPointFilterNum(int v) { point_filter_num_ = v; }

   private:
    void GenericHandler(const PointCloudType &cloud);

    PointCloudType cloud_out_;

    int point_filter_num_ = 1;
    double blind_ = 0.01;
};

}  // namespace faster_lio

#endif
