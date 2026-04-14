#ifndef FASTER_LIO_TYPES_H
#define FASTER_LIO_TYPES_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace faster_lio {

struct IMUData {
    double timestamp = 0.0;
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();

    using Ptr = std::shared_ptr<IMUData>;
    using ConstPtr = std::shared_ptr<const IMUData>;
};

/// Body-frame wheel-odometry observation. Chassis-agnostic: the caller fills
/// only the components its chassis can measure. LIO applies a scalar Kalman
/// update per present component.
///   - v_body_x    : forward linear velocity (required on every message)
///   - v_body_y    : lateral velocity (holonomic / mecanum / 4WS chassis)
///   - v_body_z    : vertical velocity (unusual; drone / aquatic)
///   - omega_z     : yaw rate (rarely better than the IMU gyro)
///
/// Non-holonomic kinematic constraints (v_y = 0, v_z = 0) are a chassis
/// property, emitted separately by LaserMapping when configured — they are
/// NOT part of this message.
struct WheelOdomData {
    double timestamp = 0.0;
    double v_body_x = 0.0;
    std::optional<double> v_body_y;
    std::optional<double> v_body_z;
    std::optional<double> omega_z;

    using Ptr = std::shared_ptr<WheelOdomData>;
    using ConstPtr = std::shared_ptr<const WheelOdomData>;
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
