#include <gtest/gtest.h>
#include "ivox3d/ivox3d.h"
#include "faster_lio/common_lib.h"

using namespace faster_lio;

// Helper to create a point
static PointType MakePoint(float x, float y, float z, float intensity = 1.0f) {
    PointType p;
    p.x = x; p.y = y; p.z = z; p.intensity = intensity;
    return p;
}

// --- DEFAULT node type tests ---

class IVoxEdgeTest : public ::testing::Test {
   protected:
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;

    std::shared_ptr<IVoxType> MakeIVox(float resolution = 0.5f,
                                        IVoxType::NearbyType nearby = IVoxType::NearbyType::NEARBY6,
                                        size_t capacity = 1000000) {
        IVoxType::Options options;
        options.resolution_ = resolution;
        options.nearby_type_ = nearby;
        options.capacity_ = capacity;
        return std::make_shared<IVoxType>(options);
    }
};

TEST_F(IVoxEdgeTest, LRUCacheEviction) {
    // Tiny capacity: only 5 voxels
    auto ivox = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6, 5);

    // Insert points into 10 different voxels (each at integer coords)
    for (int i = 0; i < 10; i++) {
        PointVector pts = {MakePoint(static_cast<float>(i) * 2.0f, 0, 0)};
        ivox->AddPoints(pts);
    }

    // Should have at most 5 voxels
    EXPECT_LE(ivox->NumValidGrids(), 5u);

    // The most recent voxels should be searchable
    PointVector result;
    PointType query = MakePoint(18.0f, 0, 0);
    ivox->GetClosestPoint(query, result, 1, 2.0);
    EXPECT_FALSE(result.empty());

    // The oldest voxels (near x=0) should have been evicted
    PointVector result_old;
    PointType query_old = MakePoint(0.0f, 0, 0);
    ivox->GetClosestPoint(query_old, result_old, 1, 0.5);
    EXPECT_TRUE(result_old.empty());
}

TEST_F(IVoxEdgeTest, Nearby6Pattern) {
    auto ivox6 = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6);

    // Place a point at (0,0,0) and query from (1.4, 0, 0) which is in a neighbor voxel
    PointVector pts = {MakePoint(0.0f, 0.0f, 0.0f)};
    ivox6->AddPoints(pts);

    PointVector result;
    ivox6->GetClosestPoint(MakePoint(1.4f, 0, 0), result, 1, 5.0);
    // NEARBY6 should find it since (1,0,0) is a neighbor
    EXPECT_GE(result.size(), 1u);
}

TEST_F(IVoxEdgeTest, Nearby18FindsMoreNeighbors) {
    auto ivox18 = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY18);

    // Place points in a grid
    PointVector pts;
    for (int x = -2; x <= 2; x++) {
        for (int y = -2; y <= 2; y++) {
            for (int z = -2; z <= 2; z++) {
                pts.push_back(MakePoint(x, y, z));
            }
        }
    }
    ivox18->AddPoints(pts);

    PointVector result;
    ivox18->GetClosestPoint(MakePoint(0, 0, 0), result, 20, 5.0);
    EXPECT_GT(result.size(), 0u);
}

TEST_F(IVoxEdgeTest, Nearby26FindsCornerNeighbors) {
    auto ivox26 = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY26);

    // Place a point in a diagonal corner voxel (1,1,1).
    // With resolution=1.0, point (0.9, 0.9, 0.9) maps to voxel round(0.9)=(1,1,1).
    // Query from (0,0,0) maps to voxel (0,0,0). NEARBY26 includes offset (1,1,1).
    PointVector pts = {MakePoint(0.9f, 0.9f, 0.9f)};
    ivox26->AddPoints(pts);

    PointVector result;
    ivox26->GetClosestPoint(MakePoint(0, 0, 0), result, 1, 5.0);
    // NEARBY26 should find the corner neighbor
    EXPECT_GE(result.size(), 1u);

    // NEARBY6 should NOT find it (diagonal not in 6-neighbors)
    auto ivox6 = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6);
    ivox6->AddPoints(pts);

    PointVector result6;
    ivox6->GetClosestPoint(MakePoint(0, 0, 0), result6, 1, 5.0);
    EXPECT_TRUE(result6.empty());
}

TEST_F(IVoxEdgeTest, CenterOnlyPattern) {
    auto ivox = MakeIVox(1.0f, IVoxType::NearbyType::CENTER);

    PointVector pts = {MakePoint(0.1f, 0.1f, 0.1f)};
    ivox->AddPoints(pts);

    // Query from same voxel should find it
    PointVector result;
    ivox->GetClosestPoint(MakePoint(0.2f, 0.2f, 0.2f), result, 1, 5.0);
    EXPECT_GE(result.size(), 1u);

    // Query from neighbor voxel should NOT find it
    PointVector result2;
    ivox->GetClosestPoint(MakePoint(1.5f, 0, 0), result2, 1, 5.0);
    EXPECT_TRUE(result2.empty());
}

TEST_F(IVoxEdgeTest, KNNWithMaxRange) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector pts;
    pts.push_back(MakePoint(0.0f, 0.0f, 0.0f));   // dist from query: ~0
    pts.push_back(MakePoint(0.1f, 0.0f, 0.0f));   // dist: 0.1
    pts.push_back(MakePoint(2.0f, 0.0f, 0.0f));   // dist: 2.0 (should be excluded)
    ivox->AddPoints(pts);

    PointVector result;
    PointType query = MakePoint(0.05f, 0, 0);
    ivox->GetClosestPoint(query, result, 5, 0.5);  // max_range = 0.5m

    for (const auto& p : result) {
        float d = std::sqrt(common::calc_dist(p, query));
        EXPECT_LT(d, 0.5f);
    }
}

TEST_F(IVoxEdgeTest, KNNWithK1) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector pts;
    for (int i = 0; i < 10; i++) {
        pts.push_back(MakePoint(i * 0.05f, 0, 0));
    }
    ivox->AddPoints(pts);

    PointVector result;
    ivox->GetClosestPoint(MakePoint(0.02f, 0, 0), result, 1, 5.0);
    EXPECT_EQ(result.size(), 1u);
}

TEST_F(IVoxEdgeTest, KNNWithK5MatchPoints) {
    auto ivox = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6);

    PointVector pts;
    for (int i = 0; i < 20; i++) {
        pts.push_back(MakePoint(i * 0.1f, 0, 0));
    }
    ivox->AddPoints(pts);

    PointVector result;
    ivox->GetClosestPoint(MakePoint(0, 0, 0), result, 5, 10.0);
    EXPECT_EQ(result.size(), 5u);
}

TEST_F(IVoxEdgeTest, DuplicatePointInsertion) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointType p = MakePoint(1.0f, 1.0f, 1.0f);
    PointVector pts(10, p);  // same point 10 times
    ivox->AddPoints(pts);

    // All go to same voxel, but IVoxNode stores all duplicates
    EXPECT_EQ(ivox->NumValidGrids(), 1u);
    EXPECT_EQ(ivox->NumPoints(), 10u);
}

TEST_F(IVoxEdgeTest, VoxelBoundaryPoints) {
    auto ivox = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6);

    // Points exactly on voxel boundaries (multiples of resolution)
    PointVector pts;
    pts.push_back(MakePoint(0.5f, 0, 0));   // boundary of voxel 0 and 1
    pts.push_back(MakePoint(1.5f, 0, 0));   // boundary of voxel 1 and 2
    pts.push_back(MakePoint(-0.5f, 0, 0));  // boundary of voxel -1 and 0
    ivox->AddPoints(pts);

    EXPECT_GT(ivox->NumValidGrids(), 0u);
    EXPECT_EQ(ivox->NumPoints(), 3u);
}

TEST_F(IVoxEdgeTest, OriginPoint) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector pts = {MakePoint(0.0f, 0.0f, 0.0f)};
    ivox->AddPoints(pts);

    EXPECT_EQ(ivox->NumPoints(), 1u);
    EXPECT_EQ(ivox->NumValidGrids(), 1u);

    PointVector result;
    ivox->GetClosestPoint(MakePoint(0.0f, 0.0f, 0.0f), result, 1, 1.0);
    EXPECT_EQ(result.size(), 1u);
    EXPECT_FLOAT_EQ(result[0].x, 0.0f);
}

TEST_F(IVoxEdgeTest, LargeCoordinates) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector pts;
    pts.push_back(MakePoint(1000.0f, 1000.0f, 1000.0f));
    pts.push_back(MakePoint(-1000.0f, -1000.0f, -1000.0f));
    pts.push_back(MakePoint(999.9f, 1000.0f, 1000.0f));
    ivox->AddPoints(pts);

    EXPECT_EQ(ivox->NumPoints(), 3u);

    // Should find the nearby point
    PointVector result;
    ivox->GetClosestPoint(MakePoint(999.95f, 1000.0f, 1000.0f), result, 1, 1.0);
    EXPECT_GE(result.size(), 1u);
}

TEST_F(IVoxEdgeTest, NegativeCoordinates) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    // Points in all 8 octants
    PointVector pts;
    float signs[] = {-1.0f, 1.0f};
    for (float sx : signs) {
        for (float sy : signs) {
            for (float sz : signs) {
                pts.push_back(MakePoint(sx * 5.0f, sy * 5.0f, sz * 5.0f));
            }
        }
    }
    ivox->AddPoints(pts);

    EXPECT_EQ(ivox->NumPoints(), 8u);

    // Query near each octant should find the correct point
    PointVector result;
    ivox->GetClosestPoint(MakePoint(-4.9f, -4.9f, -4.9f), result, 1, 1.0);
    EXPECT_GE(result.size(), 1u);
}

TEST_F(IVoxEdgeTest, SinglePointIVox) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector pts = {MakePoint(5.0f, 5.0f, 5.0f)};
    ivox->AddPoints(pts);

    PointVector result;
    ivox->GetClosestPoint(MakePoint(5.0f, 5.0f, 5.0f), result, 3, 1.0);
    EXPECT_EQ(result.size(), 1u);
}

// Note: Batch GetClosestPoint(cloud, closest_cloud) uses NNPoint which is only
// available in PHC node type. Tested via PHC-specific tests when built with
// IVOX_NODE_TYPE_PHC. For DEFAULT node type, use the KNN overload instead.

TEST_F(IVoxEdgeTest, MultipleSequentialQueries) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    // Insert a grid of points
    PointVector pts;
    for (int i = 0; i < 100; i++) {
        pts.push_back(MakePoint(i * 0.1f, 0, 0));
    }
    ivox->AddPoints(pts);

    // Multiple sequential KNN queries
    for (int i = 0; i < 50; i++) {
        PointVector result;
        PointType query = MakePoint(i * 0.2f + 0.05f, 0, 0);
        ivox->GetClosestPoint(query, result, 3, 5.0);
        EXPECT_GE(result.size(), 1u);
    }
}

TEST_F(IVoxEdgeTest, StatGridPointsAccuracy) {
    auto ivox = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6);

    // Add 3 points to one voxel and 2 to another
    PointVector pts;
    pts.push_back(MakePoint(0.1f, 0.1f, 0.1f));
    pts.push_back(MakePoint(0.2f, 0.2f, 0.2f));
    pts.push_back(MakePoint(0.3f, 0.3f, 0.3f));
    pts.push_back(MakePoint(5.0f, 5.0f, 5.0f));
    pts.push_back(MakePoint(5.1f, 5.1f, 5.1f));
    ivox->AddPoints(pts);

    auto stats = ivox->StatGridPoints();
    ASSERT_EQ(stats.size(), 5u);
    // [valid_num, average, max, min, stddev]
    EXPECT_EQ(static_cast<int>(stats[0]), 2);  // 2 non-empty grids
    EXPECT_FLOAT_EQ(stats[1], 2.5f);           // average: (3+2)/2
    EXPECT_EQ(static_cast<int>(stats[2]), 3);   // max
    EXPECT_EQ(static_cast<int>(stats[3]), 2);   // min
}

TEST_F(IVoxEdgeTest, EmptyIVoxStats) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    EXPECT_EQ(ivox->NumPoints(), 0u);
    EXPECT_EQ(ivox->NumValidGrids(), 0u);
}

TEST_F(IVoxEdgeTest, HighDensitySingleVoxel) {
    auto ivox = MakeIVox(100.0f, IVoxType::NearbyType::NEARBY6);  // huge voxel

    // Insert 500 points all in same voxel
    PointVector pts;
    for (int i = 0; i < 500; i++) {
        float x = static_cast<float>(i) * 0.01f;
        pts.push_back(MakePoint(x, 0, 0));
    }
    ivox->AddPoints(pts);

    EXPECT_EQ(ivox->NumValidGrids(), 1u);
    EXPECT_EQ(ivox->NumPoints(), 500u);

    // KNN should still work
    PointVector result;
    ivox->GetClosestPoint(MakePoint(2.5f, 0, 0), result, 5, 50.0);
    EXPECT_EQ(result.size(), 5u);
}

TEST_F(IVoxEdgeTest, AddEmptyPointVector) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector empty;
    ivox->AddPoints(empty);

    EXPECT_EQ(ivox->NumPoints(), 0u);
    EXPECT_EQ(ivox->NumValidGrids(), 0u);
}

TEST_F(IVoxEdgeTest, QueryEmptyIVox) {
    auto ivox = MakeIVox(0.5f, IVoxType::NearbyType::NEARBY6);

    PointVector result;
    bool found = ivox->GetClosestPoint(MakePoint(0, 0, 0), result, 5, 5.0);
    EXPECT_FALSE(found);
    EXPECT_TRUE(result.empty());
}

TEST_F(IVoxEdgeTest, LRUCacheRefreshOnAccess) {
    // Verify that accessing a voxel refreshes its position in LRU
    auto ivox = MakeIVox(1.0f, IVoxType::NearbyType::NEARBY6, 3);

    // Insert into 3 voxels (fills capacity)
    ivox->AddPoints({MakePoint(0, 0, 0)});   // voxel A (oldest)
    ivox->AddPoints({MakePoint(5, 0, 0)});   // voxel B
    ivox->AddPoints({MakePoint(10, 0, 0)});  // voxel C (newest)

    // Access voxel A by inserting another point to refresh it
    ivox->AddPoints({MakePoint(0.1f, 0, 0)});  // goes to voxel A, refreshes it

    // Now insert a new voxel D - should evict voxel B (the least recently used)
    ivox->AddPoints({MakePoint(15, 0, 0)});

    // Voxel A should still be present (was refreshed)
    PointVector result_a;
    ivox->GetClosestPoint(MakePoint(0, 0, 0), result_a, 1, 1.0);
    EXPECT_FALSE(result_a.empty());

    // Voxel B should be evicted
    PointVector result_b;
    ivox->GetClosestPoint(MakePoint(5, 0, 0), result_b, 1, 0.5);
    EXPECT_TRUE(result_b.empty());
}

TEST_F(IVoxEdgeTest, ResolutionAffectsVoxelCount) {
    // Same points with different resolutions should give different voxel counts
    PointVector pts;
    for (int i = 0; i < 10; i++) {
        pts.push_back(MakePoint(i * 0.5f, 0, 0));
    }

    auto ivox_fine = MakeIVox(0.3f, IVoxType::NearbyType::NEARBY6);
    auto ivox_coarse = MakeIVox(5.0f, IVoxType::NearbyType::NEARBY6);

    ivox_fine->AddPoints(pts);
    ivox_coarse->AddPoints(pts);

    // Fine resolution should create more voxels
    EXPECT_GT(ivox_fine->NumValidGrids(), ivox_coarse->NumValidGrids());
}
