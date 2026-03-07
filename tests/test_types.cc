#include <gtest/gtest.h>
#include "faster_lio/types.h"

using namespace faster_lio;

TEST(TypesTest, IMUDataDefaultConstruction) {
    IMUData imu;
    EXPECT_DOUBLE_EQ(imu.timestamp, 0.0);
    EXPECT_EQ(imu.angular_velocity, Eigen::Vector3d::Zero());
    EXPECT_EQ(imu.linear_acceleration, Eigen::Vector3d::Zero());
}

TEST(TypesTest, IMUDataFieldAccess) {
    IMUData imu;
    imu.timestamp = 1.5;
    imu.angular_velocity = Eigen::Vector3d(0.1, 0.2, 0.3);
    imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);

    EXPECT_DOUBLE_EQ(imu.timestamp, 1.5);
    EXPECT_DOUBLE_EQ(imu.angular_velocity.x(), 0.1);
    EXPECT_DOUBLE_EQ(imu.linear_acceleration.z(), 9.81);
}

TEST(TypesTest, Pose6DDefaultConstruction) {
    Pose6D pose;
    EXPECT_DOUBLE_EQ(pose.offset_time, 0.0);
    EXPECT_DOUBLE_EQ(pose.acc[0], 0.0);
    EXPECT_DOUBLE_EQ(pose.rot[0], 1.0);  // identity
    EXPECT_DOUBLE_EQ(pose.rot[4], 1.0);
    EXPECT_DOUBLE_EQ(pose.rot[8], 1.0);
}

TEST(TypesTest, LivoxPointConstruction) {
    LivoxPoint p;
    p.x = 1.0f;
    p.y = 2.0f;
    p.z = 3.0f;
    p.reflectivity = 128;
    p.tag = 0x10;
    p.line = 2;
    p.offset_time = 1000;

    EXPECT_FLOAT_EQ(p.x, 1.0f);
    EXPECT_EQ(p.reflectivity, 128);
    EXPECT_EQ(p.tag, 0x10);
    EXPECT_EQ(p.offset_time, 1000u);
}

TEST(TypesTest, LivoxCloudConstruction) {
    LivoxCloud cloud;
    cloud.timebase = 100.0;
    cloud.point_num = 2;
    cloud.points.resize(2);
    cloud.points[0].x = 1.0f;
    cloud.points[1].x = 2.0f;

    EXPECT_EQ(cloud.points.size(), 2u);
    EXPECT_FLOAT_EQ(cloud.points[0].x, 1.0f);
}

TEST(TypesTest, PoseStampedConstruction) {
    PoseStamped ps;
    EXPECT_DOUBLE_EQ(ps.timestamp, 0.0);
    EXPECT_EQ(ps.pose.position, Eigen::Vector3d::Zero());

    // Check identity quaternion
    EXPECT_DOUBLE_EQ(ps.pose.orientation.w(), 1.0);
    EXPECT_DOUBLE_EQ(ps.pose.orientation.x(), 0.0);
}

TEST(TypesTest, OdometryConstruction) {
    Odometry odom;
    odom.timestamp = 42.0;
    odom.frame_id = "world";
    odom.child_frame_id = "body";
    odom.covariance[0] = 0.01;

    EXPECT_DOUBLE_EQ(odom.timestamp, 42.0);
    EXPECT_EQ(odom.frame_id, "world");
    EXPECT_DOUBLE_EQ(odom.covariance[0], 0.01);
}
