#include "faster_lio/imu_processing.h"

#include <tbb/parallel_for.h>
#include <algorithm>

namespace faster_lio {

bool time_list(const PointType &x, const PointType &y) { return (x.curvature < y.curvature); }

ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true) {
    init_iter_num_ = 1;
    Q_ = process_noise_cov();
    cov_acc_ = common::V3D(0.1, 0.1, 0.1);
    cov_gyr_ = common::V3D(0.1, 0.1, 0.1);
    cov_bias_gyr_ = common::V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc_ = common::V3D(0.0001, 0.0001, 0.0001);
    mean_acc_ = common::V3D(0, 0, -1.0);
    mean_gyr_ = common::V3D(0, 0, 0);
    angvel_last_ = common::Zero3d;
    Lidar_T_wrt_IMU_ = common::Zero3d;
    Lidar_R_wrt_IMU_ = common::Eye3d;
    last_imu_ = std::make_shared<IMUData>();
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
    mean_acc_ = common::V3D(0, 0, -1.0);
    mean_gyr_ = common::V3D(0, 0, 0);
    angvel_last_ = common::Zero3d;
    imu_need_init_ = true;
    init_iter_num_ = 1;
    init_iter_rejected_ = 0;
    init_iter_tried_ = 0;
    v_imu_.clear();
    IMUpose_.clear();
    last_imu_ = std::make_shared<IMUData>();
    cur_pcl_un_ = std::make_shared<PointCloudType>();
}

void ImuProcess::SetExtrinsic(const common::V3D &transl, const common::M3D &rot) {
    Lidar_T_wrt_IMU_ = transl;
    Lidar_R_wrt_IMU_ = rot;
}

void ImuProcess::SetGyrCov(const common::V3D &scaler) { cov_gyr_scale_ = scaler; }

void ImuProcess::SetAccCov(const common::V3D &scaler) { cov_acc_scale_ = scaler; }

void ImuProcess::SetGyrBiasCov(const common::V3D &b_g) { cov_bias_gyr_ = b_g; }

void ImuProcess::SetAccBiasCov(const common::V3D &b_a) { cov_bias_acc_ = b_a; }

void ImuProcess::SetInitMotionGate(bool enabled, double acc_rel_thresh, double gyr_thresh,
                                   int min_accepted, int max_tries) {
    init_gate_enabled_ = enabled;
    init_acc_rel_thresh_ = acc_rel_thresh;
    init_gyr_thresh_ = gyr_thresh;
    init_min_accepted_ = std::max(1, min_accepted);
    init_max_tries_ = std::max(init_min_accepted_, max_tries);
    if (enabled) {
        spdlog::info("IMU init motion-gate ENABLED: acc_rel_thresh={:.4f} (|Δ‖a‖|/‖a‖), "
                     "gyr_thresh={:.4f} rad/s, min_accepted={}, max_tries={}",
                     acc_rel_thresh, gyr_thresh, init_min_accepted_, init_max_tries_);
    }
}

void ImuProcess::SetInitAssumeLevel(bool enabled) {
    init_assume_level_ = enabled;
    if (enabled) {
        spdlog::info("IMU init assume-level ENABLED: world-up snapped to (0, 0, +1) "
                     "regardless of mean_acc direction at init");
    }
}

void ImuProcess::IMUInit(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         int &N) {
    common::V3D cur_acc, cur_gyr;

    if (b_first_frame_) {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu_.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu_.front()->angular_velocity;
        mean_acc_ << imu_acc.x(), imu_acc.y(), imu_acc.z();
        mean_gyr_ << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();
    }

    for (const auto &imu : meas.imu_) {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x(), imu_acc.y(), imu_acc.z();
        cur_gyr << gyr_acc.x(), gyr_acc.y(), gyr_acc.z();

        // Optional motion gate: drop samples that look non-static so the
        // running mean → gravity estimate isn't contaminated by motion.
        // Accel gate is *relative* to current ‖mean_acc_‖ so it works
        // regardless of sensor units (m/s² vs g).
        if (init_gate_enabled_) {
            ++init_iter_tried_;
            const double ref = mean_acc_.norm();
            const double acc_rel_dev = ref > 1e-6 ? std::abs(cur_acc.norm() - ref) / ref : 1e9;
            const double gyr_mag = cur_gyr.norm();
            if (acc_rel_dev > init_acc_rel_thresh_ || gyr_mag > init_gyr_thresh_) {
                ++init_iter_rejected_;
                continue;
            }
        }

        mean_acc_ += (cur_acc - mean_acc_) / N;
        mean_gyr_ += (cur_gyr - mean_gyr_) / N;

        cov_acc_ =
            cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
        cov_gyr_ =
            cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

        N++;
    }
    state_ikfom init_state = kf_state.get_x();
    if (init_assume_level_) {
        // Standard Z-up mount: trust the rig is roughly level at startup and
        // use the canonical gravity direction. Avoids inheriting any residual
        // XY component of mean_acc_ (from noise or gentle motion during init)
        // as a permanent world-frame skew.
        init_state.grav = S2(0.0, 0.0, -common::G_m_s2);
    } else {
        init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * common::G_m_s2);
    }

    init_state.bg = mean_gyr_;
    init_state.offset_T_L_I = Lidar_T_wrt_IMU_;
    init_state.offset_R_L_I = Lidar_R_wrt_IMU_;
    kf_state.change_x(init_state);

    esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
    init_P.setIdentity();
    init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
    init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
    init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
    init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
    init_P(21, 21) = init_P(22, 22) = 0.00001;
    kf_state.change_P(init_P);
    last_imu_ = meas.imu_.back();
}

void ImuProcess::UndistortPcl(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                              PointCloudType &pcl_out) {
    /*** add the imu_ of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu_;
    v_imu.push_front(last_imu_);
    const double &imu_beg_time = v_imu.front()->timestamp;
    const double &imu_end_time = v_imu.back()->timestamp;
    const double &pcl_beg_time = meas.lidar_bag_time_;
    const double &pcl_end_time = meas.lidar_end_time_;

    /*** sort point clouds by offset time ***/
    pcl_out = *(meas.lidar_);
    sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

    /*** Initialize IMU pose ***/
    state_ikfom imu_state = kf_state.get_x();
    IMUpose_.clear();
    IMUpose_.reserve(v_imu.size());
    IMUpose_.push_back(common::set_pose6d(0.0, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                          imu_state.rot.toRotationMatrix()));

    /*** forward propagation at each imu_ point ***/
    common::V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
    common::M3D R_imu;

    double dt = 0;

    input_ikfom in;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (tail->timestamp < last_lidar_end_time_) {
            continue;
        }

        angvel_avr << 0.5 * (head->angular_velocity.x() + tail->angular_velocity.x()),
            0.5 * (head->angular_velocity.y() + tail->angular_velocity.y()),
            0.5 * (head->angular_velocity.z() + tail->angular_velocity.z());
        acc_avr << 0.5 * (head->linear_acceleration.x() + tail->linear_acceleration.x()),
            0.5 * (head->linear_acceleration.y() + tail->linear_acceleration.y()),
            0.5 * (head->linear_acceleration.z() + tail->linear_acceleration.z());

        acc_avr = acc_avr * common::G_m_s2 / mean_acc_.norm();

        if (head->timestamp < last_lidar_end_time_) {
            dt = tail->timestamp - last_lidar_end_time_;
        } else {
            dt = tail->timestamp - head->timestamp;
        }

        in.acc = acc_avr;
        in.gyro = angvel_avr;
        Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
        Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
        Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
        Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
        kf_state.predict(dt, Q_, in);

        /* save the poses at each IMU measurements */
        imu_state = kf_state.get_x();
        angvel_last_ = angvel_avr - imu_state.bg;
        acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba);
        for (int i = 0; i < 3; i++) {
            acc_s_last_[i] += imu_state.grav[i];
        }

        double &&offs_t = tail->timestamp - pcl_beg_time;
        IMUpose_.emplace_back(common::set_pose6d(offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                                 imu_state.rot.toRotationMatrix()));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    kf_state.predict(dt, Q_, in);

    imu_state = kf_state.get_x();
    last_imu_ = meas.imu_.back();
    last_lidar_end_time_ = pcl_end_time;

    /*** undistort each lidar point (backward propagation, parallel) ***/
    if (pcl_out.points.empty()) {
        return;
    }

    // Pre-extract IMUpose_ data into flat arrays so the parallel loop reads
    // hot, sequential memory and avoids repeated MatFromArray/VecFromArray
    // calls inside the inner kernel.
    const int n_poses = static_cast<int>(IMUpose_.size());
    std::vector<double>      pose_offset_t(n_poses);
    std::vector<common::M3D> pose_R(n_poses);
    std::vector<common::V3D> pose_v(n_poses);
    std::vector<common::V3D> pose_p(n_poses);
    std::vector<common::V3D> pose_acc(n_poses);
    std::vector<common::V3D> pose_gyr(n_poses);
    for (int i = 0; i < n_poses; ++i) {
        const auto &p = IMUpose_[i];
        pose_offset_t[i] = p.offset_time;
        pose_R[i]        = common::MatFromArray<double>(p.rot);
        pose_v[i]        = common::VecFromArray<double>(p.vel);
        pose_p[i]        = common::VecFromArray<double>(p.pos);
        pose_acc[i]      = common::VecFromArray<double>(p.acc);
        pose_gyr[i]      = common::VecFromArray<double>(p.gyr);
    }

    // Capture extrinsics and end-of-scan state before the parallel region.
    const auto offset_R_L_I_inv = imu_state.offset_R_L_I.conjugate();
    const auto rot_inv          = imu_state.rot.conjugate();
    const common::V3D offset_T_L_I_v = imu_state.offset_T_L_I;
    const common::V3D pos_end        = imu_state.pos;
    const double pose_t_min          = pose_offset_t.front();

    const int n_pts = static_cast<int>(pcl_out.points.size());
    auto *points = pcl_out.points.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, n_pts),
        [&](const tbb::blocked_range<int> &r) {
            for (int i = r.begin(); i != r.end(); ++i) {
                auto &pt = points[i];
                const double t_pt = pt.curvature / 1000.0;

                // Match original strict-> behavior: skip points at/before the
                // earliest IMU pose boundary (they have no preceding interval).
                if (t_pt <= pose_t_min) continue;

                // Binary search for the IMU interval that covers t_pt:
                //   pose_offset_t[idx]  <  t_pt  <=  pose_offset_t[idx+1]
                auto it = std::upper_bound(pose_offset_t.begin(),
                                           pose_offset_t.end(), t_pt);
                int idx = static_cast<int>(std::distance(pose_offset_t.begin(), it)) - 1;
                if (idx < 0 || idx >= n_poses - 1) continue;

                const double dt = t_pt - pose_offset_t[idx];
                const auto &R_imu_     = pose_R[idx];
                const auto &vel_imu_   = pose_v[idx];
                const auto &pos_imu_   = pose_p[idx];
                const auto &acc_imu_   = pose_acc[idx + 1];
                const auto &angvel_avr_ = pose_gyr[idx + 1];

                const common::M3D R_i(R_imu_ * Exp(angvel_avr_, dt));
                const common::V3D P_i(pt.x, pt.y, pt.z);
                const common::V3D T_ei(pos_imu_ + vel_imu_ * dt
                                       + 0.5 * acc_imu_ * dt * dt - pos_end);
                const common::V3D p_compensate =
                    offset_R_L_I_inv *
                    (rot_inv * (R_i * (imu_state.offset_R_L_I * P_i + offset_T_L_I_v) + T_ei) -
                     offset_T_L_I_v);

                pt.x = p_compensate(0);
                pt.y = p_compensate(1);
                pt.z = p_compensate(2);
            }
        });
}

void ImuProcess::Process(const common::MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudType::Ptr cur_pcl_un_) {
    if (meas.imu_.empty()) {
        return;
    }

    if (!meas.lidar_) {
        spdlog::critical("ImuProcess::Process called with null lidar point cloud");
        std::abort();
    }

    if (imu_need_init_) {
        /// The very first lidar frame
        IMUInit(meas, kf_state, init_iter_num_);

        imu_need_init_ = true;

        last_imu_ = meas.imu_.back();

        state_ikfom imu_state = kf_state.get_x();

        bool complete = false;
        bool failed_soft = false;
        if (init_gate_enabled_) {
            if (init_iter_num_ > init_min_accepted_) {
                complete = true;
            } else if (init_iter_tried_ > init_max_tries_) {
                complete = true;
                failed_soft = true;
            }
        } else if (init_iter_num_ > MAX_INI_COUNT) {
            complete = true;
        }

        if (complete) {
            cov_acc_ *= pow(common::G_m_s2 / mean_acc_.norm(), 2);
            imu_need_init_ = false;

            cov_acc_ = cov_acc_scale_;
            cov_gyr_ = cov_gyr_scale_;
            if (init_gate_enabled_) {
                if (failed_soft) {
                    spdlog::warn("IMU init FAIL-SOFT: only {} / {} required static samples after {} tries "
                                 "(rejected={}); gravity/bias may be inaccurate — widen gate or start static",
                                 init_iter_num_, init_min_accepted_, init_iter_tried_, init_iter_rejected_);
                } else {
                    spdlog::info("IMU init complete (gated): accepted={} rejected={} tried={} "
                                 "|mean_acc|={:.4f} |mean_gyr|={:.5f}",
                                 init_iter_num_, init_iter_rejected_, init_iter_tried_,
                                 mean_acc_.norm(), mean_gyr_.norm());
                }
            } else {
                spdlog::info("IMU initialization complete, {} iterations", init_iter_num_);
            }
            fout_imu_.open(common::DEBUG_FILE_DIR("imu_.txt"), std::ios::out);
        }

        return;
    }

    Timer::Evaluate([&, this]() { UndistortPcl(meas, kf_state, *cur_pcl_un_); }, "Undistort Pcl");
}

}  // namespace faster_lio
