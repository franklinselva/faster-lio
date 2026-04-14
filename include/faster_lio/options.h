//
// Created by xiang on 2021/10/8.
//

#ifndef FAST_LIO_OPTIONS_H
#define FAST_LIO_OPTIONS_H

#include <atomic>

namespace faster_lio::options {

/// fixed params
constexpr double INIT_TIME = 0.1;
constexpr double DEFAULT_LASER_POINT_COV = 0.001;
constexpr int PUBFRAME_PERIOD = 20;
constexpr int NUM_MATCH_POINTS = 5;      // required matched points in current
constexpr int MIN_NUM_MATCH_POINTS = 3;  // minimum matched points in current

/// configurable params (kept as globals for backward compatibility; also stored in LaserMapping)
extern int NUM_MAX_ITERATIONS;      // max iterations of ekf
extern float ESTI_PLANE_THRESHOLD;  // plane threshold
/// LiDAR point-to-plane measurement noise variance passed to the IEKF
/// update. Too small → filter over-trusts LiDAR, posterior becomes
/// overconfident (chi² ≫ 1). Configurable via `mapping.laser_point_cov`
/// in yaml (default: 0.001).
extern double LASER_POINT_COV;
extern std::atomic<bool> FLAG_EXIT; // flag for exiting (atomic for signal-handler safety)

}  // namespace faster_lio::options

#endif  // FAST_LIO_OPTIONS_H
