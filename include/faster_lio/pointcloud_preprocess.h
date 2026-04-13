#ifndef FASTER_LIO_POINTCLOUD_PROCESSING_H
#define FASTER_LIO_POINTCLOUD_PROCESSING_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cstdint>

#include "faster_lio/common_lib.h"

namespace velodyne_pcl {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace velodyne_pcl

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_pcl::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (float, time, time)
                                (std::uint16_t, ring, ring)
)
// clang-format on

namespace ouster_pcl {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace ouster_pcl

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_pcl::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                // use std::uint32_t to avoid conflicting with pcl::uint32_t
                                (std::uint32_t, t, t)
                                (std::uint16_t, reflectivity, reflectivity)
                                (std::uint8_t, ring, ring)
                                (std::uint16_t, ambient, ambient)
                                (std::uint32_t, range, range)
)
// clang-format on

namespace hesai_pcl {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    double timestamp;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace hesai_pcl

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(hesai_pcl::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (double, timestamp, timestamp)
                                (std::uint16_t, ring, ring)
)
// clang-format on

namespace robosense_pcl {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint16_t ring;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace robosense_pcl

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(robosense_pcl::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (std::uint16_t, ring, ring)
                                (double, timestamp, timestamp)
)
// clang-format on

namespace livox_pcl {
struct EIGEN_ALIGN16 Point {
    PCL_ADD_POINT4D;
    float intensity;
    uint8_t tag;
    uint8_t line;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace livox_pcl

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(livox_pcl::Point,
                                (float, x, x)
                                (float, y, y)
                                (float, z, z)
                                (float, intensity, intensity)
                                (std::uint8_t, tag, tag)
                                (std::uint8_t, line, line)
                                (double, timestamp, timestamp)
)
// clang-format on

namespace faster_lio {

enum class LidarType { AVIA = 1, VELO32, OUST64, HESAIxt32, ROBOSENSE, LIVOX };

/**
 * point cloud preprocess
 * just unify the point format from livox/velodyne to PCL
 */
class PointCloudPreprocess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PointCloudPreprocess() = default;
    ~PointCloudPreprocess() = default;

    /// processors - typed PCL overloads (no ROS msg dependency)
    void Process(const pcl::PointCloud<velodyne_pcl::Point> &msg, PointCloudType::Ptr &pcl_out);
    void Process(const pcl::PointCloud<ouster_pcl::Point> &msg, PointCloudType::Ptr &pcl_out);
    void Process(const pcl::PointCloud<hesai_pcl::Point> &msg, PointCloudType::Ptr &pcl_out);
    void Process(const pcl::PointCloud<robosense_pcl::Point> &msg, PointCloudType::Ptr &pcl_out);
    void Process(const pcl::PointCloud<livox_pcl::Point> &msg, PointCloudType::Ptr &pcl_out);
    void Process(const LivoxCloud &msg, PointCloudType::Ptr &pcl_out);

    /// Generic handler for pre-formed PointXYZINormal clouds.
    /// Filters zeros and blind-distance points, applies point_filter_num downsampling.
    /// Preserves existing curvature values; leaves them at 0 when absent.
    /// Scan-period estimation for IMU sync is handled by SyncPackages, not here.
    void Process(const PointCloudType &cloud, PointCloudType::Ptr &pcl_out);

    void Set(LidarType lid_type, double bld, int pfilt_num);

    // accessors
    double Blind() const { return blind_; }
    void SetBlind(double v) { blind_ = v; }
    int NumScans() const { return num_scans_; }
    void SetNumScans(int v) { num_scans_ = v; }
    int PointFilterNum() const { return point_filter_num_; }
    void SetPointFilterNum(int v) { point_filter_num_ = v; }
    bool FeatureEnabled() const { return feature_enabled_; }
    void SetFeatureEnabled(bool v) { feature_enabled_ = v; }
    float TimeScale() const { return time_scale_; }
    void SetTimeScale(float v) { time_scale_ = v; }
    LidarType GetLidarType() const { return lidar_type_; }
    void SetLidarType(LidarType lt) { lidar_type_ = lt; }

   private:
    void AviaHandler(const LivoxCloud &msg);
    void Oust64Handler(const pcl::PointCloud<ouster_pcl::Point> &msg);
    void VelodyneHandler(const pcl::PointCloud<velodyne_pcl::Point> &msg);
    template <typename PointT>
    void TimestampRingHandler(const pcl::PointCloud<PointT> &pl_orig);
    void HesaiHandler(const pcl::PointCloud<hesai_pcl::Point> &msg);
    void RobosenseHandler(const pcl::PointCloud<robosense_pcl::Point> &msg);
    void LivoxHandler(const pcl::PointCloud<livox_pcl::Point> &msg);
    void GenericHandler(const PointCloudType &cloud);

    PointCloudType cloud_full_, cloud_out_;

    LidarType lidar_type_ = LidarType::AVIA;
    bool feature_enabled_ = false;
    int point_filter_num_ = 1;
    int num_scans_ = 6;
    double blind_ = 0.01;
    float time_scale_ = 1e-3;
    bool given_offset_time_ = false;
};
}  // namespace faster_lio

#endif
