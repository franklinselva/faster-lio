#include <gtest/gtest.h>
#include "faster_lio/pointcloud_preprocess.h"

using namespace faster_lio;

class PointCloudPreprocessTest : public ::testing::Test {
   protected:
    void SetUp() override {
        preprocess_ = std::make_shared<PointCloudPreprocess>();
    }

    std::shared_ptr<PointCloudPreprocess> preprocess_;
};

TEST_F(PointCloudPreprocessTest, VelodyneProcess) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);

    // Create synthetic velodyne cloud
    pcl::PointCloud<velodyne_pcl::Point> cloud;
    for (int i = 0; i < 100; i++) {
        velodyne_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 100.0f;
        p.time = static_cast<float>(i) * 0.001f;
        p.ring = i % 16;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_GT(pcl_out->size(), 0u);
    // All points should be beyond blind distance
    for (const auto &p : pcl_out->points) {
        float dist2 = p.x * p.x + p.y * p.y + p.z * p.z;
        EXPECT_GT(dist2, 0.25f);  // blind^2 = 0.5^2 = 0.25
    }
}

TEST_F(PointCloudPreprocessTest, OusterProcess) {
    preprocess_->SetLidarType(LidarType::OUST64);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);

    pcl::PointCloud<ouster_pcl::Point> cloud;
    for (int i = 0; i < 50; i++) {
        ouster_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 0.5f;
        p.z = 0.5f;
        p.intensity = 50.0f;
        p.t = i * 1000;  // nanoseconds
        p.ring = i % 64;
        p.reflectivity = 100;
        p.ambient = 0;
        p.range = 1000;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_GT(pcl_out->size(), 0u);
}

TEST_F(PointCloudPreprocessTest, LivoxCustomMsgProcess) {
    preprocess_->SetLidarType(LidarType::AVIA);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(6);

    LivoxCloud cloud;
    cloud.timebase = 100.0;
    cloud.point_num = 50;
    cloud.points.resize(50);

    for (int i = 0; i < 50; i++) {
        cloud.points[i].x = static_cast<float>(i) + 1.0f;
        cloud.points[i].y = 1.0f;
        cloud.points[i].z = 1.0f;
        cloud.points[i].reflectivity = 128;
        cloud.points[i].tag = 0x10;
        cloud.points[i].line = i % 6;
        cloud.points[i].offset_time = i * 1000000;  // nanoseconds
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // Some points should pass through
    EXPECT_GT(pcl_out->size(), 0u);
}

TEST_F(PointCloudPreprocessTest, BlindZoneFiltering) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(2.0);  // large blind zone
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    // Points within blind zone
    for (int i = 0; i < 10; i++) {
        velodyne_pcl::Point p;
        p.x = 0.1f;
        p.y = 0.1f;
        p.z = 0.1f;
        p.intensity = 1.0f;
        p.time = static_cast<float>(i) * 0.001f;
        p.ring = 0;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // All points should be filtered out (within blind zone)
    EXPECT_EQ(pcl_out->size(), 0u);
}
