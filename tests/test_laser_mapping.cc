#include <gtest/gtest.h>
#include "faster_lio/laser_mapping.h"

using namespace faster_lio;

class LaserMappingTest : public ::testing::Test {
   protected:
    void SetUp() override {
        mapping_ = std::make_shared<LaserMapping>();
    }

    std::shared_ptr<LaserMapping> mapping_;
};

TEST_F(LaserMappingTest, InitFromYAML) {
    // Test initialization with the sample config
    std::string config = std::string(ROOT_DIR) + "config/default.yaml";
    bool result = mapping_->Init(config);
    EXPECT_TRUE(result);
}

TEST_F(LaserMappingTest, AddIMUDoesNotCrash) {
    std::string config = std::string(ROOT_DIR) + "config/default.yaml";
    ASSERT_TRUE(mapping_->Init(config));

    IMUData imu;
    imu.timestamp = 1.0;
    imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    imu.angular_velocity = Eigen::Vector3d::Zero();

    mapping_->AddIMU(imu);
    SUCCEED();
}

TEST_F(LaserMappingTest, GetCurrentPoseDefault) {
    std::string config = std::string(ROOT_DIR) + "config/default.yaml";
    ASSERT_TRUE(mapping_->Init(config));

    PoseStamped pose = mapping_->GetCurrentPose();
    // Initially should be at origin
    EXPECT_DOUBLE_EQ(pose.timestamp, 0.0);
}

TEST_F(LaserMappingTest, GetTrajectoryEmpty) {
    std::string config = std::string(ROOT_DIR) + "config/default.yaml";
    ASSERT_TRUE(mapping_->Init(config));

    const auto &traj = mapping_->GetTrajectory();
    EXPECT_TRUE(traj.empty());
}

TEST_F(LaserMappingTest, SyncPackagesReturnsFalseWhenEmpty) {
    std::string config = std::string(ROOT_DIR) + "config/default.yaml";
    ASSERT_TRUE(mapping_->Init(config));

    // No data added, sync should return false
    EXPECT_FALSE(mapping_->SyncPackages());
}

TEST_F(LaserMappingTest, RunReturnsEarlyWhenNoData) {
    std::string config = std::string(ROOT_DIR) + "config/default.yaml";
    ASSERT_TRUE(mapping_->Init(config));

    // Run with no data should not crash
    mapping_->Run();
    SUCCEED();
}
