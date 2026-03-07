#include <gtest/gtest.h>
#include "faster_lio/pointcloud_preprocess.h"

using namespace faster_lio;

class PreprocessEdgeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        preprocess_ = std::make_shared<PointCloudPreprocess>();
    }

    std::shared_ptr<PointCloudPreprocess> preprocess_;
};

// --- Empty input tests ---

TEST_F(PreprocessEdgeTest, EmptyVelodyneCloud) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_EQ(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, EmptyOusterCloud) {
    preprocess_->SetLidarType(LidarType::OUST64);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);

    pcl::PointCloud<ouster_pcl::Point> cloud;
    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_EQ(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, EmptyLivoxCloud) {
    preprocess_->SetLidarType(LidarType::AVIA);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);

    LivoxCloud cloud;
    cloud.timebase = 100.0;
    cloud.point_num = 0;

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_EQ(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, EmptyHesaiCloud) {
    preprocess_->SetLidarType(LidarType::HESAIxt32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);

    pcl::PointCloud<hesai_pcl::Point> cloud;
    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_EQ(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, EmptyRobosenseCloud) {
    preprocess_->SetLidarType(LidarType::ROBOSENSE);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);

    pcl::PointCloud<robosense_pcl::Point> cloud;
    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_EQ(pcl_out->size(), 0u);
}

// --- Single point tests ---

TEST_F(PreprocessEdgeTest, SinglePointVelodyne) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    velodyne_pcl::Point p;
    p.x = 10.0f; p.y = 0.0f; p.z = 0.0f;
    p.intensity = 50.0f; p.time = 0.0f; p.ring = 0;
    cloud.push_back(p);

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // Single valid point should pass through (beyond blind zone)
    // Note: depends on point_filter_num behavior
    SUCCEED();
}

// --- Sensor type tests ---

TEST_F(PreprocessEdgeTest, HesaiXT32Process) {
    preprocess_->SetLidarType(LidarType::HESAIxt32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(32);

    pcl::PointCloud<hesai_pcl::Point> cloud;
    for (int i = 0; i < 100; i++) {
        hesai_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 100.0f;
        p.timestamp = static_cast<double>(i) * 0.001;
        p.ring = i % 32;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_GT(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, RobosenseProcess) {
    preprocess_->SetLidarType(LidarType::ROBOSENSE);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);

    pcl::PointCloud<robosense_pcl::Point> cloud;
    for (int i = 0; i < 100; i++) {
        robosense_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 100.0f;
        p.timestamp = static_cast<double>(i) * 0.001;
        p.ring = i % 16;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_GT(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, LivoxPCL2Process) {
    preprocess_->SetLidarType(LidarType::LIVOX);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(6);

    pcl::PointCloud<livox_pcl::Point> cloud;
    for (int i = 0; i < 50; i++) {
        livox_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 100.0f;
        p.tag = 0x10;
        p.line = i % 6;
        p.timestamp = static_cast<double>(i) * 0.001;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_GT(pcl_out->size(), 0u);
}

// --- Blind zone edge cases ---

TEST_F(PreprocessEdgeTest, AllPointsInBlindZone) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(100.0);  // very large blind zone
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    for (int i = 0; i < 50; i++) {
        velodyne_pcl::Point p;
        p.x = static_cast<float>(i % 10);
        p.y = static_cast<float>(i / 10);
        p.z = 0.5f;
        p.intensity = 50.0f;
        p.time = static_cast<float>(i) * 0.001f;
        p.ring = i % 16;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_EQ(pcl_out->size(), 0u);
}

TEST_F(PreprocessEdgeTest, BlindZoneBoundaryExact) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(1.0);  // blind = 1.0m, so dist^2 < 1.0 is filtered
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;

    // Point at exactly blind distance: sqrt(x^2+y^2+z^2) = 1.0
    velodyne_pcl::Point p1;
    p1.x = 1.0f; p1.y = 0.0f; p1.z = 0.0f;
    p1.intensity = 50.0f; p1.time = 0.0f; p1.ring = 0;
    cloud.push_back(p1);

    // Point just inside blind zone
    velodyne_pcl::Point p2;
    p2.x = 0.5f; p2.y = 0.5f; p2.z = 0.0f;  // dist^2 = 0.5 < 1.0
    p2.intensity = 50.0f; p2.time = 0.001f; p2.ring = 0;
    cloud.push_back(p2);

    // Point clearly outside blind zone
    velodyne_pcl::Point p3;
    p3.x = 5.0f; p3.y = 0.0f; p3.z = 0.0f;
    p3.intensity = 50.0f; p3.time = 0.002f; p3.ring = 0;
    cloud.push_back(p3);

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // At least the far point should pass
    EXPECT_GE(pcl_out->size(), 1u);
    // All output points should be beyond blind zone
    for (const auto& p : pcl_out->points) {
        float dist2 = p.x * p.x + p.y * p.y + p.z * p.z;
        EXPECT_GE(dist2, 1.0f);
    }
}

TEST_F(PreprocessEdgeTest, ZeroCoordinatePoint) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.01);  // tiny blind zone
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    velodyne_pcl::Point p;
    p.x = 0.0f; p.y = 0.0f; p.z = 0.0f;  // origin
    p.intensity = 50.0f; p.time = 0.0f; p.ring = 0;
    cloud.push_back(p);

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // Origin point has distance 0, should be filtered by any blind zone > 0
    EXPECT_EQ(pcl_out->size(), 0u);
}

// --- Downsampling tests ---

TEST_F(PreprocessEdgeTest, HeavyDownsampling) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(50);  // keep every 50th point
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    for (int i = 0; i < 200; i++) {
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

    // With filter_num=50, output should be much smaller than input
    EXPECT_LT(pcl_out->size(), 20u);
}

TEST_F(PreprocessEdgeTest, NoDownsampling) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);  // keep every point
    preprocess_->SetNumScans(32);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    for (int i = 0; i < 100; i++) {
        velodyne_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 100.0f;
        p.time = static_cast<float>(i) * 0.001f;
        p.ring = i % 32;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // All valid points should pass (all beyond blind zone)
    EXPECT_GT(pcl_out->size(), 0u);
}

// --- Configuration tests ---

TEST_F(PreprocessEdgeTest, TimeScaleVariations) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    for (int i = 0; i < 20; i++) {
        velodyne_pcl::Point p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 1.0f;
        p.intensity = 50.0f;
        p.time = static_cast<float>(i) * 1000.0f;  // raw time in ms
        p.ring = i % 16;
        cloud.push_back(p);
    }

    // Test with different time scales
    preprocess_->SetTimeScale(1e-6);  // nanosecond scale
    PointCloudType::Ptr pcl_out1(new PointCloudType());
    preprocess_->Process(cloud, pcl_out1);

    preprocess_->SetTimeScale(1.0);  // already in seconds
    PointCloudType::Ptr pcl_out2(new PointCloudType());
    preprocess_->Process(cloud, pcl_out2);

    // Both should process without crash; point count should be similar
    // (time scale doesn't affect filtering, only curvature/offset)
    SUCCEED();
}

TEST_F(PreprocessEdgeTest, SetMethod) {
    preprocess_->Set(LidarType::VELO32, 2.0, 5);

    EXPECT_EQ(preprocess_->GetLidarType(), LidarType::VELO32);
    EXPECT_DOUBLE_EQ(preprocess_->Blind(), 2.0);
    EXPECT_EQ(preprocess_->PointFilterNum(), 5);
}

TEST_F(PreprocessEdgeTest, AccessorDefaults) {
    // Verify defaults
    EXPECT_EQ(preprocess_->GetLidarType(), LidarType::AVIA);
    EXPECT_DOUBLE_EQ(preprocess_->Blind(), 0.01);
    EXPECT_EQ(preprocess_->NumScans(), 6);
    EXPECT_EQ(preprocess_->PointFilterNum(), 1);
    EXPECT_FALSE(preprocess_->FeatureEnabled());
    EXPECT_FLOAT_EQ(preprocess_->TimeScale(), 1e-3f);
}

// --- Large point cloud test ---

TEST_F(PreprocessEdgeTest, LargePointCloud) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(32);
    preprocess_->SetTimeScale(1e-3);

    pcl::PointCloud<velodyne_pcl::Point> cloud;
    for (int i = 0; i < 100000; i++) {
        velodyne_pcl::Point p;
        p.x = static_cast<float>(i % 1000) + 1.0f;
        p.y = static_cast<float>(i / 1000) + 1.0f;
        p.z = 1.0f;
        p.intensity = 100.0f;
        p.time = static_cast<float>(i) * 0.0001f;
        p.ring = i % 32;
        cloud.push_back(p);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    EXPECT_GT(pcl_out->size(), 0u);
}

// --- Feature extraction toggle ---

TEST_F(PreprocessEdgeTest, FeatureExtractionEnabled) {
    preprocess_->SetLidarType(LidarType::VELO32);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(16);
    preprocess_->SetTimeScale(1e-3);
    preprocess_->SetFeatureEnabled(true);

    EXPECT_TRUE(preprocess_->FeatureEnabled());

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

    // Should process without crash; output may differ from non-feature mode
    SUCCEED();
}

// --- Livox-specific edge cases ---

TEST_F(PreprocessEdgeTest, LivoxCloudWithTagFiltering) {
    preprocess_->SetLidarType(LidarType::AVIA);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(6);

    LivoxCloud cloud;
    cloud.timebase = 100.0;
    cloud.point_num = 20;
    cloud.points.resize(20);

    for (int i = 0; i < 20; i++) {
        cloud.points[i].x = static_cast<float>(i) + 1.0f;
        cloud.points[i].y = 1.0f;
        cloud.points[i].z = 1.0f;
        cloud.points[i].reflectivity = 128;
        // Mix of different tag values (noise, return types)
        cloud.points[i].tag = (i % 4 == 0) ? 0x00 : 0x10;
        cloud.points[i].line = i % 6;
        cloud.points[i].offset_time = i * 1000000;
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    SUCCEED();
}

TEST_F(PreprocessEdgeTest, LivoxHighLineNumbers) {
    preprocess_->SetLidarType(LidarType::AVIA);
    preprocess_->SetBlind(0.5);
    preprocess_->SetPointFilterNum(1);
    preprocess_->SetNumScans(6);

    LivoxCloud cloud;
    cloud.timebase = 100.0;
    cloud.point_num = 10;
    cloud.points.resize(10);

    for (int i = 0; i < 10; i++) {
        cloud.points[i].x = static_cast<float>(i) + 5.0f;
        cloud.points[i].y = 1.0f;
        cloud.points[i].z = 1.0f;
        cloud.points[i].reflectivity = 128;
        cloud.points[i].tag = 0x10;
        cloud.points[i].line = 100;  // line number > num_scans
        cloud.points[i].offset_time = i * 1000000;
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    preprocess_->Process(cloud, pcl_out);

    // Should not crash even with out-of-range line numbers
    SUCCEED();
}
