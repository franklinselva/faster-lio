//
// Created by xiang on 2021/10/8.
//

#include "faster_lio/options.h"

namespace faster_lio::options {

int NUM_MAX_ITERATIONS = 0;
float ESTI_PLANE_THRESHOLD = 0.1;
double LASER_POINT_COV = DEFAULT_LASER_POINT_COV;
std::atomic<bool> FLAG_EXIT{false};

}  // namespace faster_lio