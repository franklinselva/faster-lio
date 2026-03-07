#include <gtest/gtest.h>
#include "ivox3d/ivox3d.h"
#include "faster_lio/common_lib.h"

using namespace faster_lio;

class IVoxTest : public ::testing::Test {
   protected:
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

    void SetUp() override {
        IVoxType::Options options;
        options.resolution_ = 0.5;
        options.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        ivox_ = std::make_shared<IVoxType>(options);
    }

    std::shared_ptr<IVoxType> ivox_;
};

TEST_F(IVoxTest, InsertAndQuery) {
    PointVector points;
    PointType p;
    p.x = 1.0f; p.y = 2.0f; p.z = 3.0f; p.intensity = 1.0f;
    points.push_back(p);

    p.x = 1.1f; p.y = 2.1f; p.z = 3.1f; p.intensity = 2.0f;
    points.push_back(p);

    ivox_->AddPoints(points);
    EXPECT_GT(ivox_->NumValidGrids(), 0);
}

TEST_F(IVoxTest, NearestNeighbor) {
    PointVector points;
    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) * 0.1f;
        p.y = 0.0f;
        p.z = 0.0f;
        p.intensity = static_cast<float>(i);
        points.push_back(p);
    }
    ivox_->AddPoints(points);

    PointType query;
    query.x = 0.05f; query.y = 0.0f; query.z = 0.0f;

    PointVector result;
    ivox_->GetClosestPoint(query, result, 3);
    EXPECT_GE(result.size(), 1u);
}

TEST_F(IVoxTest, EmptyQuery) {
    PointType query;
    query.x = 100.0f; query.y = 100.0f; query.z = 100.0f;

    PointVector result;
    ivox_->GetClosestPoint(query, result, 5);
    EXPECT_TRUE(result.empty());
}
