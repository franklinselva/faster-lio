#ifndef FASTER_LIO_TYPES_H
#define FASTER_LIO_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace faster_lio {

struct IMUData {
    double timestamp = 0.0;
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();

    using Ptr = std::shared_ptr<IMUData>;
    using ConstPtr = std::shared_ptr<const IMUData>;
};

struct Pose6D {
    double offset_time = 0.0;
    std::array<double, 3> acc = {0, 0, 0};
    std::array<double, 3> gyr = {0, 0, 0};
    std::array<double, 3> vel = {0, 0, 0};
    std::array<double, 3> pos = {0, 0, 0};
    std::array<double, 9> rot = {1, 0, 0, 0, 1, 0, 0, 0, 1};
};

struct Pose {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct PoseStamped {
    double timestamp = 0.0;
    Pose pose;
};

struct Odometry {
    double timestamp = 0.0;
    std::string frame_id;
    std::string child_frame_id;
    Pose pose;
    std::array<double, 36> covariance = {};
};

}  // namespace faster_lio

#endif  // FASTER_LIO_TYPES_H
