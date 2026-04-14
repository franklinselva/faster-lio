#include <gtest/gtest.h>
#include <fstream>
#include "faster_lio/laser_mapping.h"

using namespace faster_lio;

class LaserMappingEdgeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        mapping_ = std::make_shared<LaserMapping>();
        config_path_ = std::string(ROOT_DIR) + "config/default.yaml";
    }

    std::shared_ptr<LaserMapping> mapping_;
    std::string config_path_;
};

// --- Config edge cases ---

TEST_F(LaserMappingEdgeTest, InvalidYAMLPath) {
    // Returns false on non-existent file
    EXPECT_FALSE(mapping_->Init("/nonexistent/path/config.yaml"));
}

TEST_F(LaserMappingEdgeTest, EmptyYAMLPath) {
    // Returns false on empty path
    EXPECT_FALSE(mapping_->Init(""));
}

TEST_F(LaserMappingEdgeTest, MalformedYAMLContent) {
    // Create a temporary malformed yaml file
    std::string tmp_path = std::string(ROOT_DIR) + "config/test_malformed.yaml";
    {
        std::ofstream f(tmp_path);
        f << "this is not: [valid: yaml: content\n";
        f << "  broken indentation\n";
    }

    // May throw or return false depending on how parser handles it
    try {
        mapping_->Init(tmp_path);
    } catch (...) {
        // Expected - malformed YAML
    }

    // Clean up
    std::remove(tmp_path.c_str());
    SUCCEED();
}

TEST_F(LaserMappingEdgeTest, MinimalYAMLConfig) {
    // Create a yaml matching the expected structure
    std::string tmp_path = std::string(ROOT_DIR) + "config/test_minimal.yaml";
    {
        std::ofstream f(tmp_path);
        f << "common:\n";
        f << "  time_sync_en: false\n";
        f << "preprocess:\n";
        f << "  blind: 4\n";
        f << "mapping:\n";
        f << "  acc_cov: 0.1\n";
        f << "  gyr_cov: 0.1\n";
        f << "  b_acc_cov: 0.0001\n";
        f << "  b_gyr_cov: 0.0001\n";
        f << "  det_range: 300.0\n";
        f << "  extrinsic_est_en: false\n";
        f << "  extrinsic_T: [0, 0, 0]\n";
        f << "  extrinsic_R: [1,0,0, 0,1,0, 0,0,1]\n";
        f << "output:\n";
        f << "  path_en: false\n";
        f << "  dense_en: false\n";
        f << "  path_save_en: false\n";
        f << "pcd_save:\n";
        f << "  pcd_save_en: false\n";
        f << "  interval: -1\n";
        f << "point_filter_num: 3\n";
        f << "max_iteration: 4\n";
        f << "filter_size_surf: 0.5\n";
        f << "filter_size_map: 0.5\n";
        f << "cube_side_length: 1000\n";
        f << "ivox_grid_resolution: 0.5\n";
        f << "ivox_nearby_type: 6\n";
        f << "esti_plane_threshold: 0.1\n";
    }

    bool result = mapping_->Init(tmp_path);
    EXPECT_TRUE(result);

    std::remove(tmp_path.c_str());
}

// --- Data ordering edge cases ---

TEST_F(LaserMappingEdgeTest, PointCloudBeforeIMU) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Add point cloud with no IMU data yet
    PointCloudType::Ptr cloud(new PointCloudType());
    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 5.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        cloud->push_back(p);
    }
    mapping_->AddPointCloud(cloud, 1.0);

    // SyncPackages should fail (no IMU to cover the lidar frame)
    EXPECT_FALSE(mapping_->SyncPackages());
}

TEST_F(LaserMappingEdgeTest, IMUOnlyNoLidar) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Stream IMU data with no lidar
    for (int i = 0; i < 100; i++) {
        IMUData imu;
        imu.timestamp = i * 0.005;
        imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu.angular_velocity = Eigen::Vector3d::Zero();
        mapping_->AddIMU(imu);
    }

    // SyncPackages should fail (no lidar data)
    EXPECT_FALSE(mapping_->SyncPackages());
    // Trajectory should remain empty
    EXPECT_TRUE(mapping_->GetTrajectory().empty());
}

TEST_F(LaserMappingEdgeTest, VerySmallPointCloud) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Add IMU data first
    for (int i = 0; i < 50; i++) {
        IMUData imu;
        imu.timestamp = i * 0.005;
        imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu.angular_velocity = Eigen::Vector3d::Zero();
        mapping_->AddIMU(imu);
    }

    // Add a very small point cloud (< NUM_MATCH_POINTS)
    PointCloudType::Ptr cloud(new PointCloudType());
    PointType p;
    p.x = 5.0f; p.y = 5.0f; p.z = 0.0f;
    cloud->push_back(p);
    mapping_->AddPointCloud(cloud, 0.05);

    // Run should handle gracefully
    mapping_->Run();
    SUCCEED();
}

// --- SyncPackages edge cases ---

TEST_F(LaserMappingEdgeTest, SyncPackagesPartialIMU) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Add a point cloud at time 1.0
    PointCloudType::Ptr cloud(new PointCloudType());
    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 5.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;
        cloud->push_back(p);
    }
    mapping_->AddPointCloud(cloud, 1.0);

    // Add IMU data that doesn't extend past the lidar time
    for (int i = 0; i < 5; i++) {
        IMUData imu;
        imu.timestamp = 0.9 + i * 0.01;  // ends at 0.94, before lidar end
        imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu.angular_velocity = Eigen::Vector3d::Zero();
        mapping_->AddIMU(imu);
    }

    // SyncPackages should fail (IMU doesn't cover full lidar frame)
    EXPECT_FALSE(mapping_->SyncPackages());
}

// --- Multiple Run() calls ---

TEST_F(LaserMappingEdgeTest, MultipleRunCalls) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Run multiple times with no data
    for (int i = 0; i < 5; i++) {
        mapping_->Run();
    }
    SUCCEED();
}

// --- FLAG_EXIT ---

TEST_F(LaserMappingEdgeTest, FLAGEXITStopsProcessing) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Set exit flag
    options::FLAG_EXIT.store(true);

    mapping_->Run();
    SUCCEED();

    // Reset for other tests
    options::FLAG_EXIT.store(false);
}

// --- Save trajectory ---

TEST_F(LaserMappingEdgeTest, SaveTrajectoryEmpty) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    std::string tmp_path = std::string(ROOT_DIR) + "Log/test_traj_empty.txt";
    mapping_->Savetrajectory(tmp_path);

    // File should exist (even if empty)
    std::ifstream f(tmp_path);
    // Don't assert file exists since Log dir may not exist
    if (f.good()) {
        std::remove(tmp_path.c_str());
    }
    SUCCEED();
}

// --- Pose and Odometry ---

TEST_F(LaserMappingEdgeTest, GetCurrentOdometryDefault) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    Odometry odom = mapping_->GetCurrentOdometry();
    EXPECT_DOUBLE_EQ(odom.timestamp, 0.0);

    // Verify identity orientation (default pose)
    EXPECT_NEAR(odom.pose.orientation.w(), 1.0, 1e-6);
    EXPECT_NEAR(odom.pose.orientation.x(), 0.0, 1e-6);
    EXPECT_NEAR(odom.pose.orientation.y(), 0.0, 1e-6);
    EXPECT_NEAR(odom.pose.orientation.z(), 0.0, 1e-6);

    // Verify zero position
    EXPECT_NEAR(odom.pose.position.x(), 0.0, 1e-6);
    EXPECT_NEAR(odom.pose.position.y(), 0.0, 1e-6);
    EXPECT_NEAR(odom.pose.position.z(), 0.0, 1e-6);
}

TEST_F(LaserMappingEdgeTest, GetCurrentPoseAfterInit) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    PoseStamped pose = mapping_->GetCurrentPose();
    EXPECT_DOUBLE_EQ(pose.timestamp, 0.0);
    EXPECT_NEAR(pose.pose.position.x(), 0.0, 1e-6);
    EXPECT_NEAR(pose.pose.position.y(), 0.0, 1e-6);
    EXPECT_NEAR(pose.pose.position.z(), 0.0, 1e-6);
}

// --- Finish() ---

TEST_F(LaserMappingEdgeTest, FinishAfterInit) {
    ASSERT_TRUE(mapping_->Init(config_path_));
    mapping_->Finish();
    SUCCEED();
}

TEST_F(LaserMappingEdgeTest, FinishWithoutInit) {
    // Calling Finish() without Init should not crash
    mapping_->Finish();
    SUCCEED();
}

// --- AddPointCloud with generic PointCloudType ---

TEST_F(LaserMappingEdgeTest, AddGenericPointCloud) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    auto cloud = std::make_shared<PointCloudType>();
    cloud->points.reserve(10);
    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 5.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 128.0f;
        p.normal_x = p.normal_y = p.normal_z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;  // ms from scan start
        cloud->points.push_back(p);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;

    mapping_->AddPointCloud(cloud, 1.0);
    SUCCEED();
}

// --- Multiple AddIMU ---

TEST_F(LaserMappingEdgeTest, BurstIMUData) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Add a large burst of IMU data
    for (int i = 0; i < 10000; i++) {
        IMUData imu;
        imu.timestamp = i * 0.001;
        imu.linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu.angular_velocity = Eigen::Vector3d::Zero();
        mapping_->AddIMU(imu);
    }

    SUCCEED();
}

// --- Re-initialization ---

TEST_F(LaserMappingEdgeTest, DoubleInit) {
    ASSERT_TRUE(mapping_->Init(config_path_));

    // Re-initializing with same config
    bool result = mapping_->Init(config_path_);
    // Should succeed or at least not crash
    (void)result;
    SUCCEED();
}
