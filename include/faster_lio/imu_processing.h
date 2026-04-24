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

// Legacy sample-count defaults for the motion-gated init. These assume a
// 200 Hz IMU (the original FAST-LIO reference hardware). Kept because
// downstream code and the old YAML key names (min_accepted / max_tries)
// still use sample counts — but new configs should prefer the time-based
// path via `SetInitMotionGateTime` and YAML `min_time_s / max_time_s`
// below, which automatically adapts to the measured IMU rate.
constexpr int DEFAULT_INIT_GATE_MIN_ACCEPTED = 100;  // ~0.5 s @ 200 Hz IMU
constexpr int DEFAULT_INIT_GATE_MAX_TRIES = 1000;    // ~5 s fail-soft cap

// Time-based defaults. These are what the upstream 100 / 1000 counts
// actually encode — half a second of accepted static samples, five
// seconds of fail-soft cap, independent of IMU rate.
constexpr double DEFAULT_INIT_GATE_MIN_TIME_S = 0.5;
constexpr double DEFAULT_INIT_GATE_MAX_TIME_S = 5.0;

// Fallback IMU rate used when rate estimation from the first batch fails
// (e.g. single-sample batch, non-monotonic timestamps). Matches the rate
// the legacy hard-coded counts implicitly assumed.
constexpr double DEFAULT_IMU_RATE_HZ_FALLBACK = 200.0;

/// Initial diagonal of the IEKF covariance P at IMUInit completion.
///
/// State layout (23-DoF tangent space):
///   pos       indices 0-2
///   rot       indices 3-5    (SO3; tangent is 3-DoF)
///   off_R_L_I indices 6-8    (IMU↔LiDAR extrinsic rotation)
///   off_T_L_I indices 9-11   (IMU↔LiDAR extrinsic translation)
///   vel       indices 12-14
///   bg        indices 15-17  (gyroscope bias)
///   ba        indices 18-20  (accelerometer bias)
///   grav      indices 21-22  (S2; tangent is 2-DoF)
///
/// Each field is the σ² for that state block. Higher = weaker prior.
/// Defaults reproduce the upstream FAST-LIO hard-coded init_P exactly.
/// Override per-bag when you have better-than-default knowledge of the
/// calibration (extrinsic σ²) or sensor stability (bias σ²).
struct InitPDiag {
    double pos       = 1.0;
    double rot       = 1.0;
    double off_R_L_I = 1.0e-5;
    double off_T_L_I = 1.0e-5;
    double vel       = 1.0;
    double bg        = 1.0e-4;
    double ba        = 1.0e-3;
    double grav      = 1.0e-5;
};

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
    ///
    /// This sample-count form is the legacy path. Prefer `SetInitMotionGateTime`
    /// below for rate-independent configs — it converts time → count using
    /// the IMU rate measured on the first batch.
    void SetInitMotionGate(bool enabled, double acc_rel_thresh, double gyr_thresh,
                           int min_accepted = DEFAULT_INIT_GATE_MIN_ACCEPTED,
                           int max_tries = DEFAULT_INIT_GATE_MAX_TRIES);

    /// Time-based motion-gated init. Same semantics as `SetInitMotionGate` but
    /// configured in seconds — sample counts are resolved on the first
    /// IMUInit batch from the measured inter-sample interval. If rate
    /// estimation fails (single sample, bad timestamps) we fall back to
    /// `DEFAULT_IMU_RATE_HZ_FALLBACK = 200 Hz`.
    ///
    /// Two-second handheld warmups at 204 Hz Livox Avia, 5-sec fail-soft caps
    /// at any IMU rate, etc. — the time is the sensor-independent intent.
    void SetInitMotionGateTime(bool enabled, double acc_rel_thresh, double gyr_thresh,
                               double min_time_s = DEFAULT_INIT_GATE_MIN_TIME_S,
                               double max_time_s = DEFAULT_INIT_GATE_MAX_TIME_S);

    /// Resolved sample counts after rate estimation. Undefined until the
    /// first IMUInit batch has been processed (or until a sample-count
    /// setter was called, in which case these echo the user values).
    /// Exposed for tests + diagnostics.
    int GetResolvedInitMinAccepted() const { return init_min_accepted_; }
    int GetResolvedInitMaxTries() const { return init_max_tries_; }
    double GetEstimatedImuRateHz() const { return init_estimated_rate_hz_; }

    /// Configured time windows — positive iff the gate was set up via
    /// `SetInitMotionGateTime` (or the YAML time path). The count setter
    /// clears these to -1. Exposed so tests can distinguish the two
    /// configuration modes without driving a full IMU batch.
    double GetInitMinTimeS() const { return init_min_time_s_; }
    double GetInitMaxTimeS() const { return init_max_time_s_; }

    /// When true, at init completion the world frame is snapped to
    /// canonical ENU (gravity = (0, 0, -G) in world, with G as set by the
    /// S2 manifold length). The sensor's orientation is absorbed into
    /// `init_state.rot` via `FromTwoVectors(mean_acc/|mean_acc|, world_up)`,
    /// which works for ANY body-axis mount (Z-up Livox/Ouster, X-up
    /// Hilti Xsens, Y-up, or any intermediate tilt) — the rotation
    /// aligns whichever body axis reads +g with world Z+. Horizontal
    /// motion in body frame then projects cleanly to horizontal motion
    /// in world frame without leaking into world-Z.
    ///
    /// Verified by the parameterized `AllSixMountAxes/AssumeLevelMount`
    /// test battery.
    ///
    /// When false (default), the world frame inherits the sensor's
    /// startup tilt: `init_state.grav = -mean_acc/|mean_acc| * G`
    /// points along whatever body axis was up at init, and rot stays at
    /// identity. Positions are expressed in this tilted world. Keep
    /// false if you need that (e.g. direct comparison against a stored
    /// baseline captured under the same convention).
    void SetInitAssumeLevel(bool enabled);

    /// Override the initial filter covariance (P) diagonal applied at
    /// IMUInit completion. Absent overrides keep the struct's field
    /// defaults, which match the upstream FAST-LIO hard-coded init_P
    /// byte-for-byte. Any negative or zero value is clamped to a tiny
    /// positive floor (1e-12) so P stays positive-definite.
    ///
    /// Safe to call before or between Reset() cycles — the stored diagonal
    /// is consulted on every IMUInit() completion.
    void SetInitPDiag(const InitPDiag &ip);

    /// Read-only accessor for the stored InitPDiag (tests / diagnostics).
    const InitPDiag &GetInitPDiag() const { return init_P_diag_; }

    /// True once the IMU warm-up (gravity alignment + static-sample averaging)
    /// has completed. Before this, the filter's pose is still at the default
    /// identity/origin and shouldn't be published.
    bool Initialized() const { return !imu_need_init_; }
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

    /// Rebuild the 12×12 process-noise covariance Q_ from the runtime
    /// cov_gyr_ / cov_acc_ / cov_bias_gyr_ / cov_bias_acc_ members. The
    /// result is diagonal. Called on construction, from the bias-cov
    /// setters, and at the end of IMUInit after the measurement covs are
    /// switched from the init-time running variance to the YAML-scale
    /// values. Previously this was rebuilt inside UndistortPcl every scan.
    void RebuildQ();

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

    // Time-based counterparts. When >0 they take precedence over the
    // sample counts above and are resolved (time × rate → count) on the
    // first IMUInit batch. -1 means "use the sample count directly".
    double init_min_time_s_ = -1.0;
    double init_max_time_s_ = -1.0;
    double init_estimated_rate_hz_ = 0.0;  // 0 = not measured yet
    bool init_counts_resolved_ = false;    // first-batch rate conversion done?

    int init_iter_rejected_ = 0;
    int init_iter_tried_ = 0;
    bool init_assume_level_ = false;

    // Sanitized init-covariance diagonal applied at IMUInit completion.
    // Defaults match upstream FAST-LIO byte-for-byte. Updated via
    // SetInitPDiag() (clamps non-positive values to a tiny floor).
    InitPDiag init_P_diag_{};
};

}  // namespace faster_lio

#endif
