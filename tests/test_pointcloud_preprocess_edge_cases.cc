#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "faster_lio/pointcloud_preprocess.h"

using namespace faster_lio;

class PreprocessEdgeCaseTest : public ::testing::Test {
   protected:
    void SetUp() override {
        preprocess_ = std::make_shared<PointCloudPreprocess>();
        preprocess_->SetBlind(0.5);
        preprocess_->SetPointFilterNum(1);
    }

    std::shared_ptr<PointCloudPreprocess> preprocess_;

    static PointType MakePoint(float x, float y, float z, float intensity = 0.0f, float curvature = 0.0f) {
        PointType p;
        p.x = x;
        p.y = y;
        p.z = z;
        p.intensity = intensity;
        p.normal_x = p.normal_y = p.normal_z = 0.0f;
        p.curvature = curvature;
        return p;
    }
};

TEST_F(PreprocessEdgeCaseTest, EmptyCloud) {
    PointCloudType cloud;
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_TRUE(out->points.empty());
}

TEST_F(PreprocessEdgeCaseTest, AllPointsInBlindZone) {
    PointCloudType cloud;
    for (int i = 0; i < 20; ++i) {
        cloud.points.push_back(MakePoint(0.1f, 0.1f, 0.0f));
    }
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_TRUE(out->points.empty());
}

TEST_F(PreprocessEdgeCaseTest, BlindZoneBoundaryExact) {
    // A point exactly at blind distance is kept (filter uses strict less-than).
    preprocess_->SetBlind(2.0);
    PointCloudType cloud;
    cloud.points.push_back(MakePoint(1.9f, 0.0f, 0.0f));  // range < blind → drop
    cloud.points.push_back(MakePoint(2.0f, 0.0f, 0.0f));  // range == blind → kept
    cloud.points.push_back(MakePoint(2.1f, 0.0f, 0.0f));  // range > blind → kept

    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_EQ(out->points.size(), 2u);
}

TEST_F(PreprocessEdgeCaseTest, ZeroCoordinatePoint) {
    // Points at origin (pre-allocated padding) are always filtered,
    // regardless of blind setting.
    preprocess_->SetBlind(0.001);
    PointCloudType cloud;
    cloud.points.push_back(MakePoint(0.0f, 0.0f, 0.0f));  // drop
    cloud.points.push_back(MakePoint(1.0f, 0.0f, 0.0f));  // keep
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_EQ(out->points.size(), 1u);
}

TEST_F(PreprocessEdgeCaseTest, HeavyDownsampling) {
    preprocess_->SetBlind(0.01);
    preprocess_->SetPointFilterNum(10);
    PointCloudType cloud;
    for (int i = 0; i < 100; ++i) {
        cloud.points.push_back(MakePoint(static_cast<float>(i + 1), 0.0f, 0.0f));
    }
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_EQ(out->points.size(), 10u);
}

TEST_F(PreprocessEdgeCaseTest, NoDownsampling) {
    preprocess_->SetBlind(0.01);
    preprocess_->SetPointFilterNum(1);
    PointCloudType cloud;
    for (int i = 0; i < 50; ++i) {
        cloud.points.push_back(MakePoint(static_cast<float>(i + 1), 0.0f, 0.0f));
    }
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_EQ(out->points.size(), 50u);
}

TEST_F(PreprocessEdgeCaseTest, CurvaturePreservedThroughFiltering) {
    preprocess_->SetBlind(0.01);
    preprocess_->SetPointFilterNum(1);
    PointCloudType cloud;
    cloud.points.push_back(MakePoint(0.0f, 0.0f, 0.0f, 1.0f, 0.0f));   // padding → drop
    cloud.points.push_back(MakePoint(1.0f, 0.0f, 0.0f, 2.0f, 10.0f));  // keep
    cloud.points.push_back(MakePoint(2.0f, 0.0f, 0.0f, 3.0f, 20.0f));  // keep

    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    ASSERT_EQ(out->points.size(), 2u);
    EXPECT_FLOAT_EQ(out->points[0].curvature, 10.0f);
    EXPECT_FLOAT_EQ(out->points[1].curvature, 20.0f);
}

TEST_F(PreprocessEdgeCaseTest, LargePointCloud) {
    preprocess_->SetBlind(0.01);
    preprocess_->SetPointFilterNum(1);
    PointCloudType cloud;
    cloud.points.reserve(100000);
    for (int i = 0; i < 100000; ++i) {
        float r = static_cast<float>((i % 100) + 1);
        cloud.points.push_back(MakePoint(r, 0.0f, 0.0f));
    }
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_EQ(out->points.size(), 100000u);
}
