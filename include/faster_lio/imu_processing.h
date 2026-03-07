#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <spdlog/spdlog.h>
#include <cmath>
#include <deque>
#include <fstream>

#include "faster_lio/common_lib.h"
#include "faster_lio/so3_math.h"
#include "faster_lio/use-ikfom.hpp"
#include "faster_lio/utils.h"

namespace faster_lio {

constexpr int MAX_INI_COUNT = 20;

bool time_list(const PointType &x, const PointType &y);

/// IMU Process and undistortion
class ImuProcess {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();
    ~ImuProcess();

    void Reset();
    void SetExtrinsic(const common::V3D &transl, const common::M3D &rot);
    void SetGyrCov(const common::V3D &scaler);
    void SetAccCov(const common::V3D &scaler);
    void SetGyrBiasCov(const common::V3D &b_g);
    void SetAccBiasCov(const common::V3D &b_a);
    void Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                 PointCloudType::Ptr pcl_un_);

    std::ofstream fout_imu_;
    Eigen::Matrix<double, 12, 12> Q_;
    common::V3D cov_acc_;
    common::V3D cov_gyr_;
    common::V3D cov_acc_scale_;
    common::V3D cov_gyr_scale_;
    common::V3D cov_bias_gyr_;
    common::V3D cov_bias_acc_;

   private:
    void IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
    void UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                      PointCloudType &pcl_out);

    PointCloudType::Ptr cur_pcl_un_;
    IMUData::Ptr last_imu_;
    std::deque<IMUData::Ptr> v_imu_;
    std::vector<common::Pose6D> IMUpose_;
    std::vector<common::M3D> v_rot_pcl_;
    common::M3D Lidar_R_wrt_IMU_;
    common::V3D Lidar_T_wrt_IMU_;
    common::V3D mean_acc_;
    common::V3D mean_gyr_;
    common::V3D angvel_last_;
    common::V3D acc_s_last_;
    double last_lidar_end_time_ = 0;
    int init_iter_num_ = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

}  // namespace faster_lio

#endif
