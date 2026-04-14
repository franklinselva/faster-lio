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

// Defaults for the optional motion-gated init (see SetInitMotionGate).
constexpr int DEFAULT_INIT_GATE_MIN_ACCEPTED = 100;  // ~0.5 s @ 200 Hz IMU
constexpr int DEFAULT_INIT_GATE_MAX_TRIES = 1000;    // ~5 s fail-soft cap

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

    /// Optional motion-gated init. When enabled, IMUInit rejects samples that
    /// look non-static and only completes once `min_accepted` samples have
    /// passed the gate; fails soft after `max_tries` total samples.
    ///
    /// The accel gate is *relative* to the running mean ‖acc‖ so the same
    /// threshold works whether the sensor reports accel in m/s² or in g:
    ///   reject iff |‖cur_acc‖ − ‖mean_acc‖| / ‖mean_acc‖ > acc_rel_thresh
    ///           OR ‖cur_gyr‖ > gyr_thresh  (gyro is always rad/s)
    ///
    /// Disabled by default → init behavior matches legacy Faster-LIO exactly
    /// (MAX_INI_COUNT samples, unconditional averaging).
    void SetInitMotionGate(bool enabled, double acc_rel_thresh, double gyr_thresh,
                           int min_accepted = DEFAULT_INIT_GATE_MIN_ACCEPTED,
                           int max_tries = DEFAULT_INIT_GATE_MAX_TRIES);
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

    // Optional motion-gated init state. Default-disabled → legacy behavior.
    bool init_gate_enabled_ = false;
    double init_acc_rel_thresh_ = 0.05;  // fractional deviation of ‖cur_acc‖ from ‖mean_acc_‖
    double init_gyr_thresh_ = 0.05;      // rad/s max ‖cur_gyr‖
    int init_min_accepted_ = DEFAULT_INIT_GATE_MIN_ACCEPTED;
    int init_max_tries_ = DEFAULT_INIT_GATE_MAX_TRIES;
    int init_iter_rejected_ = 0;
    int init_iter_tried_ = 0;
};

}  // namespace faster_lio

#endif
