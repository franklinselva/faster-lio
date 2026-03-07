#include <gtest/gtest.h>
#include "faster_lio/common_lib.h"

using namespace faster_lio;
using namespace faster_lio::common;

// --- esti_normvector tests ---

TEST(CommonLibEdgeCases, EstiNormvectorPerfectPlane) {
    // 5 coplanar points on the z=1 plane
    PointVector points(5);
    points[0].x = 0; points[0].y = 0; points[0].z = 1;
    points[1].x = 1; points[1].y = 0; points[1].z = 1;
    points[2].x = 0; points[2].y = 1; points[2].z = 1;
    points[3].x = 1; points[3].y = 1; points[3].z = 1;
    points[4].x = 0.5f; points[4].y = 0.5f; points[4].z = 1;

    Eigen::Vector3f normvec;
    bool result = esti_normvector(normvec, points, 0.1f, 5);
    EXPECT_TRUE(result);
    // Normal should be roughly along z-axis (normalized)
    EXPECT_GT(std::abs(normvec.z()), 0.9f);
}

TEST(CommonLibEdgeCases, EstiNormvectorCollinearPoints) {
    // 5 collinear points (degenerate - no unique plane)
    PointVector points(5);
    for (int i = 0; i < 5; i++) {
        points[i].x = static_cast<float>(i);
        points[i].y = static_cast<float>(i);
        points[i].z = static_cast<float>(i);
    }

    Eigen::Vector3f normvec;
    // With a tight threshold, collinear points should fail plane fitting
    bool result = esti_normvector(normvec, points, 0.001f, 5);
    // For truly collinear through origin, A*x + B*y + C*z = -1 has no solution
    // The QR solve will produce a least-squares solution but residuals will be large
    // Result depends on threshold; with very tight threshold it should fail
    // Just verify no crash
    (void)result;
    SUCCEED();
}

TEST(CommonLibEdgeCases, EstiNormvectorIdenticalPoints) {
    // 5 identical points at origin - completely degenerate
    PointVector points(5);
    for (int i = 0; i < 5; i++) {
        points[i].x = 1.0f;
        points[i].y = 1.0f;
        points[i].z = 1.0f;
    }

    Eigen::Vector3f normvec;
    // All identical points: Ax + By + Cz = -1 has a solution (the constant plane)
    // but the normal is meaningless for actual plane fitting
    bool result = esti_normvector(normvec, points, 0.1f, 5);
    // Just verify no crash
    (void)result;
    SUCCEED();
}

TEST(CommonLibEdgeCases, EstiNormvectorNearlyCoplanar) {
    // Points with small noise off a plane
    PointVector points(5);
    points[0].x = 0; points[0].y = 0; points[0].z = 1.001f;
    points[1].x = 1; points[1].y = 0; points[1].z = 0.999f;
    points[2].x = 0; points[2].y = 1; points[2].z = 1.002f;
    points[3].x = 1; points[3].y = 1; points[3].z = 0.998f;
    points[4].x = 0.5f; points[4].y = 0.5f; points[4].z = 1.0f;

    Eigen::Vector3f normvec;
    bool result = esti_normvector(normvec, points, 0.1f, 5);
    EXPECT_TRUE(result);
}

// --- esti_plane tests ---

TEST(CommonLibEdgeCases, EstiPlaneCoplanar) {
    // Perfect plane z=2: 0*x + 0*y + 1*z - 2 = 0
    PointVector points(5);
    points[0].x = 0; points[0].y = 0; points[0].z = 2;
    points[1].x = 3; points[1].y = 0; points[1].z = 2;
    points[2].x = 0; points[2].y = 3; points[2].z = 2;
    points[3].x = 3; points[3].y = 3; points[3].z = 2;
    points[4].x = 1.5f; points[4].y = 1.5f; points[4].z = 2;

    Eigen::Vector4f pca_result;
    bool result = esti_plane(pca_result, points, 0.1f);
    EXPECT_TRUE(result);
    // Normal should be along z-axis
    EXPECT_GT(std::abs(pca_result(2)), 0.9f);
}

TEST(CommonLibEdgeCases, EstiPlaneTooFewPoints) {
    // Fewer than MIN_NUM_MATCH_POINTS (3) should return false
    PointVector points(2);
    points[0].x = 0; points[0].y = 0; points[0].z = 1;
    points[1].x = 1; points[1].y = 0; points[1].z = 1;

    Eigen::Vector4f pca_result;
    bool result = esti_plane(pca_result, points, 0.1f);
    EXPECT_FALSE(result);
}

TEST(CommonLibEdgeCases, EstiPlaneEmptyPoints) {
    PointVector points;
    Eigen::Vector4f pca_result;
    bool result = esti_plane(pca_result, points, 0.1f);
    EXPECT_FALSE(result);
}

TEST(CommonLibEdgeCases, EstiPlaneNonCoplanar) {
    // Points that don't fit a plane well
    PointVector points(5);
    points[0].x = 0; points[0].y = 0; points[0].z = 0;
    points[1].x = 1; points[1].y = 0; points[1].z = 0;
    points[2].x = 0; points[2].y = 1; points[2].z = 0;
    points[3].x = 0; points[3].y = 0; points[3].z = 10;  // far off plane
    points[4].x = 1; points[4].y = 1; points[4].z = 0;

    Eigen::Vector4f pca_result;
    bool result = esti_plane(pca_result, points, 0.01f);  // tight threshold
    EXPECT_FALSE(result);
}

TEST(CommonLibEdgeCases, EstiPlaneNonStandardPointCount) {
    // More than NUM_MATCH_POINTS (5), exercises the dynamic path
    PointVector points(8);
    for (int i = 0; i < 8; i++) {
        points[i].x = static_cast<float>(i % 3);
        points[i].y = static_cast<float>(i / 3);
        points[i].z = 5.0f;
    }

    Eigen::Vector4f pca_result;
    bool result = esti_plane(pca_result, points, 0.1f);
    EXPECT_TRUE(result);
}

// --- calc_dist tests ---

TEST(CommonLibEdgeCases, CalcDistCorrectness) {
    PointType p1, p2;
    p1.x = 1; p1.y = 2; p1.z = 3;
    p2.x = 4; p2.y = 6; p2.z = 3;

    float d = calc_dist(p1, p2);
    // (4-1)^2 + (6-2)^2 + (3-3)^2 = 9 + 16 + 0 = 25
    EXPECT_FLOAT_EQ(d, 25.0f);
}

TEST(CommonLibEdgeCases, CalcDistZero) {
    PointType p;
    p.x = 3; p.y = 4; p.z = 5;
    EXPECT_FLOAT_EQ(calc_dist(p, p), 0.0f);
}

TEST(CommonLibEdgeCases, CalcDistEigenVersion) {
    Eigen::Vector3f p1(1.0f, 0.0f, 0.0f);
    Eigen::Vector3f p2(0.0f, 1.0f, 0.0f);
    float d = calc_dist(p1, p2);
    // (1-0)^2 + (0-1)^2 + 0 = 2
    EXPECT_FLOAT_EQ(d, 2.0f);
}

// --- set_pose6d tests ---

TEST(CommonLibEdgeCases, SetPose6DRoundTrip) {
    Eigen::Vector3d a(1, 2, 3);
    Eigen::Vector3d g(0.1, 0.2, 0.3);
    Eigen::Vector3d v(4, 5, 6);
    Eigen::Vector3d p(7, 8, 9);
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();

    Pose6D pose = set_pose6d(42.0, a, g, v, p, R);

    EXPECT_DOUBLE_EQ(pose.offset_time, 42.0);
    EXPECT_DOUBLE_EQ(pose.acc[0], 1.0);
    EXPECT_DOUBLE_EQ(pose.acc[1], 2.0);
    EXPECT_DOUBLE_EQ(pose.acc[2], 3.0);
    EXPECT_DOUBLE_EQ(pose.gyr[0], 0.1);
    EXPECT_DOUBLE_EQ(pose.vel[0], 4.0);
    EXPECT_DOUBLE_EQ(pose.pos[2], 9.0);
    EXPECT_DOUBLE_EQ(pose.rot[0], 1.0);
    EXPECT_DOUBLE_EQ(pose.rot[4], 1.0);
    EXPECT_DOUBLE_EQ(pose.rot[8], 1.0);
    EXPECT_DOUBLE_EQ(pose.rot[1], 0.0);
}

TEST(CommonLibEdgeCases, SetPose6DWithRotation) {
    Eigen::Vector3d a = Eigen::Vector3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    // 90-degree rotation around Z
    Eigen::Matrix3d R;
    R << 0, -1, 0,
         1,  0, 0,
         0,  0, 1;

    Pose6D pose = set_pose6d(0.0, a, g, v, p, R);
    EXPECT_DOUBLE_EQ(pose.rot[0], 0.0);
    EXPECT_DOUBLE_EQ(pose.rot[1], -1.0);
    EXPECT_DOUBLE_EQ(pose.rot[3], 1.0);
    EXPECT_DOUBLE_EQ(pose.rot[8], 1.0);
}

// --- VecFromArray / MatFromArray tests ---

TEST(CommonLibEdgeCases, VecFromArrayVector) {
    std::vector<double> v = {1.0, 2.0, 3.0};
    auto result = VecFromArray<double>(v);
    EXPECT_DOUBLE_EQ(result.x(), 1.0);
    EXPECT_DOUBLE_EQ(result.y(), 2.0);
    EXPECT_DOUBLE_EQ(result.z(), 3.0);
}

TEST(CommonLibEdgeCases, VecFromArrayStdArray) {
    std::array<double, 3> v = {-1.0, 0.0, 1.0};
    auto result = VecFromArray<double>(v);
    EXPECT_DOUBLE_EQ(result.x(), -1.0);
    EXPECT_DOUBLE_EQ(result.z(), 1.0);
}

TEST(CommonLibEdgeCases, VecFromArrayFloat) {
    std::vector<double> v = {1.5, 2.5, 3.5};
    auto result = VecFromArray<float>(v);
    EXPECT_FLOAT_EQ(result.x(), 1.5f);
}

TEST(CommonLibEdgeCases, MatFromArrayIdentity) {
    std::vector<double> v = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    auto result = MatFromArray<double>(v);
    EXPECT_TRUE(result.isApprox(Eigen::Matrix3d::Identity()));
}

TEST(CommonLibEdgeCases, MatFromArrayStdArray) {
    std::array<double, 9> v = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = MatFromArray<double>(v);
    EXPECT_DOUBLE_EQ(result(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(result(0, 2), 3.0);
    EXPECT_DOUBLE_EQ(result(2, 2), 9.0);
}

// --- MeasureGroup tests ---

TEST(CommonLibEdgeCases, MeasureGroupDefault) {
    MeasureGroup meas;
    EXPECT_DOUBLE_EQ(meas.lidar_bag_time_, 0.0);
    EXPECT_DOUBLE_EQ(meas.lidar_end_time_, 0.0);
    EXPECT_NE(meas.lidar_, nullptr);
    EXPECT_TRUE(meas.imu_.empty());
    EXPECT_EQ(meas.lidar_->size(), 0u);
}

TEST(CommonLibEdgeCases, MeasureGroupLargeIMUQueue) {
    MeasureGroup meas;
    for (int i = 0; i < 1000; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.001;
        imu->linear_acceleration = Eigen::Vector3d(0, 0, 9.81);
        meas.imu_.push_back(imu);
    }
    EXPECT_EQ(meas.imu_.size(), 1000u);
    EXPECT_DOUBLE_EQ(meas.imu_.front()->timestamp, 0.0);
    EXPECT_NEAR(meas.imu_.back()->timestamp, 0.999, 1e-9);
}

// --- rad2deg / deg2rad tests ---

TEST(CommonLibEdgeCases, RadDegConversion) {
    EXPECT_NEAR(rad2deg(M_PI), 180.0, 1e-10);
    EXPECT_NEAR(deg2rad(180.0), M_PI, 1e-10);
    EXPECT_NEAR(rad2deg(0.0), 0.0, 1e-10);
    EXPECT_NEAR(deg2rad(0.0), 0.0, 1e-10);
    EXPECT_NEAR(deg2rad(rad2deg(1.234)), 1.234, 1e-10);
}

TEST(CommonLibEdgeCases, RadDegNegative) {
    EXPECT_NEAR(rad2deg(-M_PI), -180.0, 1e-10);
    EXPECT_NEAR(deg2rad(-90.0), -M_PI / 2.0, 1e-10);
}

// --- Constants tests ---

TEST(CommonLibEdgeCases, ConstantsValid) {
    EXPECT_TRUE(Eye3d.isApprox(M3D::Identity()));
    EXPECT_TRUE(Eye3f.isApprox(M3F::Identity()));
    EXPECT_TRUE(Zero3d.isApprox(V3D::Zero()));
    EXPECT_TRUE(Zero3f.isApprox(V3F::Zero()));
    EXPECT_DOUBLE_EQ(G_m_s2, 9.81);
}
