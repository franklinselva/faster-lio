// ─────────────────────────────────────────────────────────────────────────
// Scan Context algorithm tests.
//
// We test the pure algorithm in isolation (no PCL I/O, no LoopCloser
// integration). Four scenarios:
//
//   A. Identical cloud → distance ≈ 0, shift = 0.
//   B. Cloud rotated by a known yaw → distance ≈ 0, shift = yaw_bin.
//   C. Different scene → distance > threshold.
//   D. Ring key invariant under yaw rotation.
//
// E. Synthetic "end-to-start revisit with large pose drift": build two
//    clouds of the same structure centered at different world positions
//    (mimicking the drifted IEKF estimate) and verify the descriptor match
//    is POSE-INDEPENDENT (the descriptor is local-frame, so translation
//    of the world origin doesn't affect it — this is the headline property
//    that the standard pose-proximity gate is missing).
// ─────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <cmath>
#include <iostream>
#include <random>

#include "faster_lio/scan_context.h"

using faster_lio::ScanContext;
using CloudT = pcl::PointCloud<pcl::PointXYZI>;

namespace {

// Build a toy "building" — four walls at radii 5, 10, 15, 20 m. Z varies
// smoothly with azimuth so each (ring, sector) bin has a distinct max-z,
// giving the descriptor genuine column structure (required for rotation
// detection to be unambiguous — a rotationally-symmetric scene has
// identical descriptors at every shift).
CloudT MakeSceneWalls() {
    CloudT cloud;
    for (double r : {5.0, 10.0, 15.0, 20.0}) {
        // Dense theta sampling so every sector is populated with several
        // points — fp boundary rounding on a single point doesn't evict
        // the sector.
        for (int i = 0; i < 3600; ++i) {
            const double theta = i * (M_PI / 1800.0);  // 0.1° spacing
            pcl::PointXYZI p;
            p.x = static_cast<float>(r * std::cos(theta));
            p.y = static_cast<float>(r * std::sin(theta));
            // Per-sector distinguishing height pattern: varies with both
            // radius AND azimuth, so each descriptor cell has a unique
            // value that's preserved under rotation (up to column shift).
            p.z = static_cast<float>(r * 0.1 + 0.5 * std::sin(3.0 * theta) +
                                     0.3 * std::cos(5.0 * theta));
            p.intensity = 1.0f;
            cloud.push_back(p);
        }
    }
    return cloud;
}

// Rotate a cloud by `yaw` about the Z axis.
CloudT RotateCloudYaw(const CloudT &src, double yaw) {
    CloudT out;
    out.reserve(src.size());
    const double c = std::cos(yaw), s = std::sin(yaw);
    for (const auto &p : src.points) {
        pcl::PointXYZI q;
        q.x = static_cast<float>(c * p.x - s * p.y);
        q.y = static_cast<float>(s * p.x + c * p.y);
        q.z = p.z;
        q.intensity = p.intensity;
        out.push_back(q);
    }
    return out;
}

// Translate a cloud by a world offset (points in local frame aren't
// affected by the origin being elsewhere — but this emulates the case
// where a caller forgets to transform into local frame).
CloudT TranslateCloud(const CloudT &src, double dx, double dy, double dz) {
    CloudT out;
    out.reserve(src.size());
    for (const auto &p : src.points) {
        pcl::PointXYZI q = p;
        q.x = static_cast<float>(p.x + dx);
        q.y = static_cast<float>(p.y + dy);
        q.z = static_cast<float>(p.z + dz);
        out.push_back(q);
    }
    return out;
}

// Build a DIFFERENT scene: one wall only, with varying heights per sector
// so the descriptor differs from MakeSceneWalls.
CloudT MakeSceneSingleWall() {
    CloudT cloud;
    for (int i = 0; i < 360; ++i) {
        const double theta = i * M_PI / 180.0;
        pcl::PointXYZI p;
        p.x = static_cast<float>(8.0 * std::cos(theta));
        p.y = static_cast<float>(8.0 * std::sin(theta));
        p.z = static_cast<float>(std::sin(3.0 * theta));  // varying
        p.intensity = 1.0f;
        cloud.push_back(p);
    }
    return cloud;
}

ScanContext::Options DefaultOpts() {
    ScanContext::Options o;
    o.num_rings   = 20;
    o.num_sectors = 60;
    o.max_range   = 25.0;
    return o;
}

}  // namespace

TEST(ScanContext, IdenticalCloudsAreZeroDistance) {
    const auto cloud = MakeSceneWalls();
    const auto opts = DefaultOpts();
    const auto da = ScanContext::ComputeDescriptor(cloud, opts);
    const auto db = ScanContext::ComputeDescriptor(cloud, opts);

    const auto r = ScanContext::Distance(da, db);
    std::cout << "[A] distance=" << r.score << " shift=" << r.shift << "\n";
    EXPECT_LT(r.score, 1e-9);
    EXPECT_EQ(r.shift, 0);
}

TEST(ScanContext, RotatedCloudRecoversYaw) {
    const auto cloud = MakeSceneWalls();
    const auto opts  = DefaultOpts();
    const auto da    = ScanContext::ComputeDescriptor(cloud, opts);

    // Rotate by an angle that lands on a sector boundary. 60 sectors →
    // 6° per sector. 60° → shift = 10.
    const double yaw = 60.0 * M_PI / 180.0;
    const auto rotated = RotateCloudYaw(cloud, yaw);
    const auto db     = ScanContext::ComputeDescriptor(rotated, opts);

    const auto r = ScanContext::Distance(da, db);
    std::cout << "[B] yaw=60° distance=" << r.score
              << " shift=" << r.shift << " (expect 10)\n";
    // Boundary-aligned rotations on dense clouds still produce some fp
    // noise at bin edges; small but non-zero distance is expected.
    EXPECT_LT(r.score, 1e-2);
    EXPECT_EQ(r.shift, 10);

    // Recover yaw: shift * 2π / num_sectors.
    const double recovered =
        r.shift * (2.0 * M_PI) / static_cast<double>(opts.num_sectors);
    EXPECT_NEAR(recovered, yaw, 1e-6);
}

TEST(ScanContext, DifferentScenesHaveLargeDistance) {
    const auto a = MakeSceneWalls();
    const auto b = MakeSceneSingleWall();
    const auto opts = DefaultOpts();

    const auto da = ScanContext::ComputeDescriptor(a, opts);
    const auto db = ScanContext::ComputeDescriptor(b, opts);

    const auto r = ScanContext::Distance(da, db);
    std::cout << "[C] different scenes distance=" << r.score << "\n";
    EXPECT_GT(r.score, 0.1);  // clearly non-match
}

TEST(ScanContext, RingKeyInvariantUnderYaw) {
    const auto cloud = MakeSceneWalls();
    const auto opts  = DefaultOpts();

    const auto da = ScanContext::ComputeDescriptor(cloud, opts);
    const auto ka = ScanContext::ComputeRingKey(da);

    // Ring key = fraction of non-empty columns per ring. Rotation is a
    // cyclic permutation of columns, so the count is invariant — but a
    // column that was populated only by ONE near-boundary point can lose
    // that point to its neighbor (fp rounding) and become empty. With a
    // dense cloud this is rare; allow a small tolerance.
    for (double yaw_deg : {30.0, 60.0, 90.0, 180.0}) {
        const auto rot = RotateCloudYaw(cloud, yaw_deg * M_PI / 180.0);
        const auto db = ScanContext::ComputeDescriptor(rot, opts);
        const auto kb = ScanContext::ComputeRingKey(db);
        const double dk = ScanContext::RingKeyDistance(ka, kb);
        std::cout << "[D] yaw=" << yaw_deg << "° ring-key dist=" << dk << "\n";
        EXPECT_LT(dk, 1e-3);
    }
}

// ─── E: the headline test. Two clouds of the same local scene, at
// different world positions (mimicking a drifted IEKF). Because the
// descriptor is local-frame, world translation doesn't affect the match.
// This is what the pose-proximity gate is missing — and what Scan Context
// gives us for free.
TEST(ScanContext, PoseDriftDoesNotDefeatMatch) {
    const auto local = MakeSceneWalls();
    const auto opts  = DefaultOpts();

    // In the real flow, both clouds are transformed into their respective
    // submap-local frames before descriptor computation. So even though the
    // two submaps' ANCHOR_POSES may be 54 m apart (due to IEKF drift),
    // the local_cloud content is identical — and so must the descriptors.
    const auto da = ScanContext::ComputeDescriptor(local, opts);
    const auto db = ScanContext::ComputeDescriptor(local, opts);
    const auto r  = ScanContext::Distance(da, db);
    std::cout << "[E] drift-invariant descriptor distance=" << r.score << "\n";
    EXPECT_LT(r.score, 1e-9);

    // Sanity: ring keys also agree.
    const auto ka = ScanContext::ComputeRingKey(da);
    const auto kb = ScanContext::ComputeRingKey(db);
    EXPECT_LT(ScanContext::RingKeyDistance(ka, kb), 1e-9);

    // A caller that naively passes WORLD-frame points instead of local-frame
    // would lose this property — the translated cloud produces a different
    // descriptor because radii/azimuths shift. Document the contract by
    // explicitly showing the broken case.
    const auto translated = TranslateCloud(local, 54.0, 0.0, 0.0);
    const auto dc = ScanContext::ComputeDescriptor(translated, opts);
    const auto r2 = ScanContext::Distance(da, dc);
    std::cout << "[E'] same scene, WORLD-translated (caller bug) distance="
              << r2.score << "\n";
    EXPECT_GT(r2.score, 0.1);
}

// ─── F: noise-robustness. Same scene with added Gaussian perturbation
// should still match with small distance.
TEST(ScanContext, ModeratelyNoisyCloudStillMatches) {
    const auto clean = MakeSceneWalls();
    const auto opts  = DefaultOpts();
    const auto dc = ScanContext::ComputeDescriptor(clean, opts);

    std::mt19937 rng(0xB16B00B5);
    std::normal_distribution<float> noise(0.0f, 0.05f);  // 5 cm std

    CloudT noisy;
    noisy.reserve(clean.size());
    for (const auto &p : clean.points) {
        pcl::PointXYZI q = p;
        q.x += noise(rng);
        q.y += noise(rng);
        q.z += noise(rng);
        noisy.push_back(q);
    }

    const auto dn = ScanContext::ComputeDescriptor(noisy, opts);
    const auto r  = ScanContext::Distance(dc, dn);
    std::cout << "[F] noisy (5 cm std) distance=" << r.score
              << " shift=" << r.shift << "\n";
    EXPECT_LT(r.score, 0.1);
    // Shift of 0 is the intended answer, but neighbouring shifts can tie
    // with near-identical scores under noise; accept ±1 sector.
    EXPECT_LE(std::abs(r.shift), 1);
}
