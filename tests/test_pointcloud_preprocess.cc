#include <gtest/gtest.h>
#include "faster_lio/pointcloud_preprocess.h"

using namespace faster_lio;

class PointCloudPreprocessTest : public ::testing::Test {
   protected:
    void SetUp() override { preprocess_ = std::make_shared<PointCloudPreprocess>(); }

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

TEST_F(PointCloudPreprocessTest, FiltersZeroOriginPadding) {
    preprocess_->SetBlind(0.1);
    preprocess_->SetPointFilterNum(1);

    PointCloudType cloud;
    cloud.points.push_back(MakePoint(1.0f, 0.0f, 0.0f));
    cloud.points.push_back(MakePoint(0.0f, 0.0f, 0.0f));  // padding
    cloud.points.push_back(MakePoint(0.0f, 2.0f, 0.0f));

    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);

    EXPECT_EQ(out->points.size(), 2u);
}

TEST_F(PointCloudPreprocessTest, FiltersBlindZone) {
    preprocess_->SetBlind(1.0);
    preprocess_->SetPointFilterNum(1);

    PointCloudType cloud;
    cloud.points.push_back(MakePoint(0.5f, 0.0f, 0.0f));   // within blind → drop
    cloud.points.push_back(MakePoint(2.0f, 0.0f, 0.0f));   // outside blind → keep
    cloud.points.push_back(MakePoint(0.0f, 0.3f, 0.0f));   // within blind → drop
    cloud.points.push_back(MakePoint(0.0f, 0.0f, 5.0f));   // outside blind → keep

    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);

    EXPECT_EQ(out->points.size(), 2u);
}

TEST_F(PointCloudPreprocessTest, DownsamplesByPointFilterNum) {
    preprocess_->SetBlind(0.01);
    preprocess_->SetPointFilterNum(3);

    PointCloudType cloud;
    for (int i = 0; i < 10; ++i) {
        cloud.points.push_back(MakePoint(static_cast<float>(i + 1), 0.0f, 0.0f));
    }

    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);

    // Keeps indices 0, 3, 6, 9 → 4 points
    EXPECT_EQ(out->points.size(), 4u);
}

TEST_F(PointCloudPreprocessTest, PreservesCurvature) {
    preprocess_->SetBlind(0.01);
    preprocess_->SetPointFilterNum(1);

    PointCloudType cloud;
    cloud.points.push_back(MakePoint(1.0f, 0.0f, 0.0f, 50.0f, 1.5f));
    cloud.points.push_back(MakePoint(2.0f, 0.0f, 0.0f, 60.0f, 3.0f));

    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);

    ASSERT_EQ(out->points.size(), 2u);
    EXPECT_FLOAT_EQ(out->points[0].curvature, 1.5f);
    EXPECT_FLOAT_EQ(out->points[1].curvature, 3.0f);
    EXPECT_FLOAT_EQ(out->points[0].intensity, 50.0f);
    EXPECT_FLOAT_EQ(out->points[1].intensity, 60.0f);
}

TEST_F(PointCloudPreprocessTest, EmptyCloudIsNoOp) {
    PointCloudType cloud;
    PointCloudType::Ptr out(new PointCloudType);
    preprocess_->Process(cloud, out);
    EXPECT_TRUE(out->points.empty());
}

TEST_F(PointCloudPreprocessTest, AccessorDefaults) {
    PointCloudPreprocess fresh;
    EXPECT_NEAR(fresh.Blind(), 0.01, 1e-9);
    EXPECT_EQ(fresh.PointFilterNum(), 1);
}

TEST_F(PointCloudPreprocessTest, AccessorsRoundTrip) {
    preprocess_->SetBlind(0.8);
    preprocess_->SetPointFilterNum(5);
    EXPECT_NEAR(preprocess_->Blind(), 0.8, 1e-9);
    EXPECT_EQ(preprocess_->PointFilterNum(), 5);
}
