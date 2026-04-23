#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <deque>
#include <vector>
#include <string>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>
// ArpackSupport removed - not needed and causes build issues on some Eigen versions

#include "faster_lio/types.h"
#include "faster_lio/options.h"
#include "faster_lio/so3_math.h"

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;

namespace faster_lio::common {

// KEEP THIS AT 9.81 — even though the S2 manifold length is 9.8090 and a
// "consistent" choice would be G_m_s2 = 9.8090, on the Hilti site2 bag
// G_m_s2 = 9.8090 produces catastrophic divergence (Z → 42 km, path → 364 km),
// while G_m_s2 = 9.81 produces a bounded "37 m drift" baseline.
//
// Why: with G_m_s2 = 9.8090 the IMU rescale-vs-gravity is internally
// consistent and the IEKF posterior covariance contracts aggressively. At
// motion onset (~t=42 s in site2) the filter's confidence is so high that
// valid LiDAR observations get rejected by `p_body.norm() > 81 * pd2 * pd2`
// in laser_mapping.cc:644 → "No effective feature points" cascade → IMU
// integrates unchecked. With G_m_s2 = 9.81 the constant 0.001 m/s² residual
// keeps covariance "looser" and the filter stays receptive to LiDAR pulls.
//
// This is a partial fix — the right repair is to make the outlier check
// adapt to posterior covariance, not to live with a known mathematical
// inconsistency. Tracked separately. DO NOT change to 9.8090 without an
// end-to-end bag test (synthetic stationary tests pass at 9.8090 but
// don't catch the motion-regime divergence).
constexpr double G_m_s2 = 9.81;

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const std::vector<double> &v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 1> VecFromArray(const std::array<double, 3> &v) {
    return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const std::vector<double> &v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

template <typename S>
inline Eigen::Matrix<S, 3, 3> MatFromArray(const std::array<double, 9> &v) {
    Eigen::Matrix<S, 3, 3> m;
    m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
    return m;
}

inline std::string DEBUG_FILE_DIR(const std::string &name) { return std::string(ROOT_DIR) + "Log/" + name; }

using Pose6D = faster_lio::Pose6D;
using V3D = Eigen::Vector3d;
using V4D = Eigen::Vector4d;
using V5D = Eigen::Matrix<double, 5, 1>;
using M3D = Eigen::Matrix3d;
using M4D = Eigen::Matrix4d;
using V3F = Eigen::Vector3f;
using V4F = Eigen::Vector4f;
using V5F = Eigen::Matrix<float, 5, 1>;
using M3F = Eigen::Matrix3f;
using M4F = Eigen::Matrix4f;

using VV3D = std::vector<V3D, Eigen::aligned_allocator<V3D>>;
using VV3F = std::vector<V3F, Eigen::aligned_allocator<V3F>>;
using VV4F = std::vector<V4F, Eigen::aligned_allocator<V4F>>;
using VV4D = std::vector<V4D, Eigen::aligned_allocator<V4D>>;
using VV5F = std::vector<V5F, Eigen::aligned_allocator<V5F>>;
using VV5D = std::vector<V5D, Eigen::aligned_allocator<V5D>>;

inline const M3D Eye3d = M3D::Identity();
inline const M3F Eye3f = M3F::Identity();
inline const V3D Zero3d(0, 0, 0);
inline const V3F Zero3f(0, 0, 0);

/// sync imu and lidar measurements
struct MeasureGroup {
    MeasureGroup() { this->lidar_ = std::make_shared<PointCloudType>(); };

    double lidar_bag_time_ = 0;
    double lidar_end_time_ = 0;
    PointCloudType::Ptr lidar_ = nullptr;
    std::deque<IMUData::Ptr> imu_;
};

template <typename T>
T rad2deg(const T &radians) {
    return radians * 180.0 / M_PI;
}

template <typename T>
T deg2rad(const T &degrees) {
    return degrees * M_PI / 180.0;
}

/**
 * set a pose 6d from ekf status
 */
template <typename T>
Pose6D set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                  const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R) {
    Pose6D rot_kp;
    rot_kp.offset_time = t;
    for (int i = 0; i < 3; i++) {
        rot_kp.acc[i] = a(i);
        rot_kp.gyr[i] = g(i);
        rot_kp.vel[i] = v(i);
        rot_kp.pos[i] = p(i);
        for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
    }
    return rot_kp;
}

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec_:  normalized x0
*/
template <typename T>
bool esti_normvector(Eigen::Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold,
                     const int &point_num) {
    Eigen::MatrixXf A(point_num, 3);
    Eigen::MatrixXf b(point_num, 1);
    b.setOnes();
    b *= -1.0f;

    for (int j = 0; j < point_num; j++) {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }
    normvec = A.colPivHouseholderQr().solve(b);

    for (int j = 0; j < point_num; j++) {
        if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold) {
            return false;
        }
    }

    normvec.normalize();
    return true;
}

inline float calc_dist(const PointType &p1, const PointType &p2) {
    return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
}

inline float calc_dist(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) { return (p1 - p2).squaredNorm(); }

/// Fit a plane to k-NN points and compute fit quality.
///
/// Solves n·p + 1 = 0 via least-squares, then checks each neighbor's
/// residual against `threshold`. Returns false if any neighbor exceeds it.
///
/// `fit_quality` (output): RMS of per-neighbor residuals. Lower = better plane.
/// This is always written (even on failure) so callers can inspect the quality
/// of rejected fits. A value of 0 means perfect coplanarity.
template <typename T>
inline bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point,
                       const T &threshold = 0.1f, T *fit_quality = nullptr) {
    if (point.size() < options::MIN_NUM_MATCH_POINTS) {
        if (fit_quality) *fit_quality = std::numeric_limits<T>::max();
        return false;
    }

    Eigen::Matrix<T, 3, 1> normvec;

    if (point.size() == options::NUM_MATCH_POINTS) {
        Eigen::Matrix<T, options::NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<T, options::NUM_MATCH_POINTS, 1> b;

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < options::NUM_MATCH_POINTS; j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        normvec = A.householderQr().solve(b);
    } else {
        Eigen::MatrixXd A(point.size(), 3);
        Eigen::VectorXd b(point.size(), 1);

        A.setZero();
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < point.size(); j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }

        Eigen::MatrixXd n = A.colPivHouseholderQr().solve(b);
        normvec(0, 0) = n(0, 0);
        normvec(1, 0) = n(1, 0);
        normvec(2, 0) = n(2, 0);
    }

    T n = normvec.norm();
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    // Compute per-neighbor residuals and RMS fit quality
    T sum_sq = 0;
    bool all_pass = true;
    for (const auto &p : point) {
        Eigen::Matrix<T, 4, 1> temp = p.getVector4fMap();
        temp[3] = 1.0;
        T res = fabs(pca_result.dot(temp));
        sum_sq += res * res;
        if (res > threshold) {
            all_pass = false;
        }
    }

    if (fit_quality) {
        *fit_quality = std::sqrt(sum_sq / static_cast<T>(point.size()));
    }
    return all_pass;
}

}  // namespace faster_lio::common
#endif
