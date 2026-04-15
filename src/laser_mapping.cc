#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include "faster_lio/compat.h"

#include "faster_lio/laser_mapping.h"
#include "faster_lio/utils.h"

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
#include <sys/resource.h>
#endif

namespace faster_lio {

bool LaserMapping::Init(const std::string &config_yaml) {
    spdlog::info("Initializing laser mapping from {}", config_yaml);
    if (!LoadParamsFromYAML(config_yaml)) {
        return false;
    }

    // localmap init (after LoadParams)
    ivox_ = std::make_shared<IVoxType>(ivox_options_);

    // esekf init
    std::vector<double> epsi(23, 0.001);
    kf_.init_dyn_share(
        get_f, df_dx, df_dw,
        [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
        num_max_iterations_, epsi.data());

    if (std::is_same<IVoxType, IVox<3, IVoxNodeType::PHC, pcl::PointXYZI>>::value == true) {
        spdlog::info("Using PHC iVox");
    } else if (std::is_same<IVoxType, IVox<3, IVoxNodeType::DEFAULT, pcl::PointXYZI>>::value == true) {
        spdlog::info("Using default iVox");
    }

    initialized_ = true;
    return true;
}

bool LaserMapping::LoadParamsFromYAML(const std::string &yaml_file) {
    int ivox_nearby_type;
    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    double filter_size_surf_min;
    common::V3D lidar_T_wrt_IMU;
    common::M3D lidar_R_wrt_IMU;

    YAML::Node yaml;
    try {
        yaml = YAML::LoadFile(yaml_file);
    } catch (const YAML::BadFile &e) {
        spdlog::error("Failed to open YAML config file '{}': {}", yaml_file, e.what());
        return false;
    }
    try {
        path_pub_en_ = yaml["output"]["path_en"].as<bool>();
        dense_pub_en_ = yaml["output"]["dense_en"].as<bool>();
        path_save_en_ = yaml["output"]["path_save_en"].as<bool>();

        num_max_iterations_ = yaml["max_iteration"].as<int>();
        esti_plane_threshold_ = yaml["esti_plane_threshold"].as<float>();
        if (yaml["map_quality_threshold"]) {
            map_quality_threshold_ = yaml["map_quality_threshold"].as<float>();
        }
        options::NUM_MAX_ITERATIONS = num_max_iterations_;
        options::ESTI_PLANE_THRESHOLD = esti_plane_threshold_;
        time_sync_en_ = yaml["common"]["time_sync_en"].as<bool>();

        filter_size_surf_min = yaml["filter_size_surf"].as<float>();
        filter_size_map_min_ = yaml["filter_size_map"].as<float>();
        cube_len_ = yaml["cube_side_length"].as<int>();
        det_range_ = yaml["mapping"]["det_range"].as<float>();
        gyr_cov = yaml["mapping"]["gyr_cov"].as<float>();
        acc_cov = yaml["mapping"]["acc_cov"].as<float>();
        b_gyr_cov = yaml["mapping"]["b_gyr_cov"].as<float>();
        b_acc_cov = yaml["mapping"]["b_acc_cov"].as<float>();

        // LiDAR point-to-plane measurement covariance (optional; preserves
        // baseline when absent). Raise when filter is overconfident
        // (chi² ≫ 1 in diagnostics).
        if (yaml["mapping"]["laser_point_cov"]) {
            options::LASER_POINT_COV = yaml["mapping"]["laser_point_cov"].as<double>();
            spdlog::info("LASER_POINT_COV overridden from yaml: {:.6f}", options::LASER_POINT_COV);
        } else {
            options::LASER_POINT_COV = options::DEFAULT_LASER_POINT_COV;
        }

        // Preprocessing (sensor-agnostic). Only two knobs: blind distance
        // and per-point downsampling stride.
        if (yaml["preprocess"] && yaml["preprocess"]["blind"]) {
            preprocess_->SetBlind(yaml["preprocess"]["blind"].as<double>());
        }
        preprocess_->SetPointFilterNum(yaml["point_filter_num"].as<int>());

        extrinsic_est_en_ = yaml["mapping"]["extrinsic_est_en"].as<bool>();
        pcd_save_en_ = yaml["pcd_save"]["pcd_save_en"].as<bool>();
        pcd_save_interval_ = yaml["pcd_save"]["interval"].as<int>();
        extrinT_ = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
        extrinR_ = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

        ivox_options_.resolution_ = yaml["ivox_grid_resolution"].as<float>();
        ivox_nearby_type = yaml["ivox_nearby_type"].as<int>();
    } catch (...) {
        spdlog::error("Failed to parse YAML config: invalid parameter type conversion");
        return false;
    }

    if (ivox_nearby_type == 0) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
    } else if (ivox_nearby_type == 6) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
    } else if (ivox_nearby_type == 18) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    } else if (ivox_nearby_type == 26) {
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
    } else {
        spdlog::warn("Unknown ivox_nearby_type: {}, falling back to NEARBY18", ivox_nearby_type);
        ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
    }

    voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

    lidar_T_wrt_IMU = common::VecFromArray<double>(extrinT_);
    lidar_R_wrt_IMU = common::MatFromArray<double>(extrinR_);

    p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
    p_imu_->SetGyrCov(common::V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->SetAccCov(common::V3D(acc_cov, acc_cov, acc_cov));
    p_imu_->SetGyrBiasCov(common::V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->SetAccBiasCov(common::V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    // Optional: motion-gated IMU init. If the yaml block is absent, legacy
    // behavior is preserved (MAX_INI_COUNT=20 samples, no gating).
    if (yaml["imu_init"]) {
        const auto &ii = yaml["imu_init"];
        const bool en = ii["motion_gate_enabled"] ? ii["motion_gate_enabled"].as<bool>() : false;
        const double acc_rel_thresh = ii["acc_rel_thresh"] ? ii["acc_rel_thresh"].as<double>() : 0.05;
        const double gyr_thresh = ii["gyr_thresh"] ? ii["gyr_thresh"].as<double>() : 0.05;
        const int min_accepted = ii["min_accepted"] ? ii["min_accepted"].as<int>() : DEFAULT_INIT_GATE_MIN_ACCEPTED;
        const int max_tries = ii["max_tries"] ? ii["max_tries"].as<int>() : DEFAULT_INIT_GATE_MAX_TRIES;
        p_imu_->SetInitMotionGate(en, acc_rel_thresh, gyr_thresh, min_accepted, max_tries);
    }

    // Optional: wheel-odometry fusion. Off by default → no path change.
    if (yaml["wheel"]) {
        const auto &w = yaml["wheel"];
        wheel_enabled_      = w["enabled"]         ? w["enabled"].as<bool>()           : false;
        wheel_cov_v_x_      = w["cov_v_x"]         ? w["cov_v_x"].as<double>()         : 0.01;
        wheel_cov_v_y_      = w["cov_v_y"]         ? w["cov_v_y"].as<double>()         : 0.01;
        wheel_cov_v_z_      = w["cov_v_z"]         ? w["cov_v_z"].as<double>()         : 0.001;
        wheel_cov_omega_z_  = w["cov_omega_z"]     ? w["cov_omega_z"].as<double>()     : 0.01;
        wheel_emit_nhc_v_y_ = w["emit_nhc_v_y"]    ? w["emit_nhc_v_y"].as<bool>()      : true;
        wheel_emit_nhc_v_z_ = w["emit_nhc_v_z"]    ? w["emit_nhc_v_z"].as<bool>()      : true;
        wheel_nhc_cov_      = w["nhc_cov"]         ? w["nhc_cov"].as<double>()         : 0.001;
        wheel_max_time_gap_ = w["max_time_gap"]    ? w["max_time_gap"].as<double>()    : 0.05;
        if (wheel_enabled_) {
            spdlog::info("Wheel odometry ENABLED: cov_v_x={:.4f} cov_v_y={:.4f} cov_v_z={:.5f} "
                         "nhc_y={} nhc_z={} nhc_cov={:.5f} max_gap={:.3f}s",
                         wheel_cov_v_x_, wheel_cov_v_y_, wheel_cov_v_z_,
                         wheel_emit_nhc_v_y_, wheel_emit_nhc_v_z_, wheel_nhc_cov_, wheel_max_time_gap_);
        }
    }

    return true;
}

LaserMapping::LaserMapping() {
    preprocess_ = std::make_shared<PointCloudPreprocess>();
    p_imu_ = std::make_shared<ImuProcess>();
}

void LaserMapping::AddIMU(const IMUData &imu) {
    publish_count_++;
    auto msg = std::make_shared<IMUData>(imu);

    if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
        msg->timestamp = timediff_lidar_wrt_imu_ + imu.timestamp;
    }

    double timestamp = msg->timestamp;

    if (timestamp < last_timestamp_imu_) {
        spdlog::warn("IMU timestamp went backwards (ts={}), dropping sample", timestamp);
        return;
    }
    last_timestamp_imu_ = timestamp;

    // Lock-free push. If the consumer is stalled and the queue is full,
    // the sample is dropped — visibility into backpressure.
    if (!imu_queue_.try_push(msg)) {
        spdlog::warn("IMU queue full (cap={}), dropping sample", kImuQueueCapacity);
    }
}

void LaserMapping::AddPointCloud(const PointCloudType::Ptr &cloud, double timestamp) {
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (timestamp < last_timestamp_lidar_) {
                spdlog::error("Lidar timestamp went backwards, dropping scan");
                return;
            }

            last_timestamp_lidar_ = timestamp;

            if (!time_sync_en_ && std::abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 &&
                last_timestamp_imu_ > 0.0 && last_timestamp_lidar_ > 0.0) {
                spdlog::info("IMU and LiDAR not Synced, IMU time: {}, lidar header time: {}",
                             last_timestamp_imu_, last_timestamp_lidar_);
            }

            if (time_sync_en_ && !timediff_set_flg_ &&
                std::abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                last_timestamp_imu_ > 0.0) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                spdlog::info("Self sync IMU and LiDAR, time diff is {}", timediff_lidar_wrt_imu_);
            }

            auto ptr = std::make_shared<PointCloudType>();
            preprocess_->Process(*cloud, ptr);
            if (!lidar_queue_.try_push(LidarItem{ptr, timestamp})) {
                spdlog::warn("LiDAR queue full (cap={}), dropping scan", kLidarQueueCapacity);
            }
        },
        "Preprocess (Generic)");
}

void LaserMapping::AddWheelOdom(const WheelOdomData &odom) {
    // Always accept into the queue, regardless of wheel_enabled_. If the
    // fusion is off, ApplyWheelObservations will short-circuit and the
    // queue drains anyway (prevents stale data when fusion is toggled).
    if (odom.timestamp <= 0.0) {
        spdlog::warn("Wheel odom has invalid timestamp ({}), dropping", odom.timestamp);
        return;
    }
    if (last_timestamp_wheel_ > 0.0 && odom.timestamp < last_timestamp_wheel_) {
        spdlog::warn("Wheel odom timestamp went backwards ({} < {}), dropping",
                     odom.timestamp, last_timestamp_wheel_);
        return;
    }
    last_timestamp_wheel_ = odom.timestamp;
    auto msg = std::make_shared<WheelOdomData>(odom);
    if (!wheel_queue_.try_push(msg)) {
        spdlog::warn("Wheel queue full (cap={}), dropping sample", kWheelQueueCapacity);
    }
}

PoseStamped LaserMapping::GetCurrentPose() const {
    std::lock_guard<std::mutex> lock(mtx_state_);
    return msg_body_pose_;
}

Odometry LaserMapping::GetCurrentOdometry() const {
    std::lock_guard<std::mutex> lock(mtx_state_);
    Odometry odom;
    odom.timestamp = lidar_end_time_;
    odom.pose.position = Eigen::Vector3d(state_point_.pos(0), state_point_.pos(1), state_point_.pos(2));
    odom.pose.orientation = Eigen::Quaterniond(state_point_.rot.coeffs()[3], state_point_.rot.coeffs()[0],
                                                state_point_.rot.coeffs()[1], state_point_.rot.coeffs()[2]);

    auto P = kf_.get_P();
    for (int i = 0; i < 6; i++) {
        int k = i < 3 ? i + 3 : i - 3;
        odom.covariance[i * 6 + 0] = P(k, 3);
        odom.covariance[i * 6 + 1] = P(k, 4);
        odom.covariance[i * 6 + 2] = P(k, 5);
        odom.covariance[i * 6 + 3] = P(k, 0);
        odom.covariance[i * 6 + 4] = P(k, 1);
        odom.covariance[i * 6 + 5] = P(k, 2);
    }
    return odom;
}

void LaserMapping::Run() {
    if (!initialized_) {
        spdlog::warn("Run() called before Init(), ignoring");
        return;
    }

    if (!SyncPackages()) {
        return;
    }

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    using clk = std::chrono::steady_clock;
    const auto t_run_start = clk::now();
    auto t0 = clk::now();
#endif

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        spdlog::warn("Empty undistorted scan, skipping frame");
        return;
    }

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    diag_t_undistort_us_ = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
#endif

    /// the first scan
    if (flg_first_scan_) {
        state_point_ = kf_.get_x();
        scan_down_world_->resize(scan_undistort_->size());
        for (int i = 0; i < scan_undistort_->size(); i++) {
            PointBodyToWorld(&scan_undistort_->points[i], &scan_down_world_->points[i]);
        }
        ivox_->AddPoints(scan_down_world_->points);
        first_lidar_time_ = measures_.lidar_bag_time_;
        flg_first_scan_ = false;
        return;
    }
    flg_EKF_inited_ = (measures_.lidar_bag_time_ - first_lidar_time_) >= options::INIT_TIME;

    /// downsample
#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    t0 = clk::now();
#endif
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_undistort_);
            voxel_scan_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");
#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    diag_t_downsample_us_ = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
#endif

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        spdlog::warn("Too few points after downsampling ({} -> {}), skipping frame", scan_undistort_->size(), scan_down_body_->size());
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    // No value-init needed: ObsModel writes every element of residuals_ and point_selected_surf_
    residuals_.resize(cur_pts);
    fit_quality_.resize(cur_pts, 0.0f);
    point_selected_surf_.resize(cur_pts);
    plane_coef_.resize(cur_pts, common::V4F::Zero());

    // ICP and iterated Kalman filter update
#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    t0 = clk::now();
#endif
    Timer::Evaluate(
        [&, this]() {
            double solve_H_time = 0;
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            // Optional: wheel-odom sequential update. No-op when disabled.
            ApplyWheelObservations(lidar_end_time_);
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
        },
        "IEKF Solve and Update");
#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    diag_t_iekf_us_ = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
    t0 = clk::now();
#endif

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");
#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    diag_t_mapinc_us_ = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t0).count();
    diag_t_run_total_us_ = std::chrono::duration_cast<std::chrono::microseconds>(clk::now() - t_run_start).count();
#endif

    spdlog::debug("Mapping frame {}: input={} downsampled={} map_grids={} effective_features={}",
                  frame_num_, scan_undistort_->points.size(), cur_pts, ivox_->NumValidGrids(), effect_feat_num_);

    // update pose and path
    {
        std::lock_guard<std::mutex> lock(mtx_state_);
        SetPosestamp(msg_body_pose_);
        msg_body_pose_.timestamp = lidar_end_time_;
        if (path_pub_en_ || path_save_en_) {
            path_.push_back(msg_body_pose_);
        }
    }

    // save map pcd if enabled
    if (pcd_save_en_) {
        SaveFrameWorld();
    }

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    WriteDiagnosticsRow();
#endif

    // Debug variables
    frame_num_++;
}

bool LaserMapping::SyncPackages() {
    // Drain SPSC queues into consumer-side scratch deques. The SPSC queues
    // are lock-free; only this thread (the Run() caller) reads from them.
    while (auto *imu = imu_queue_.front()) {
        imu_buffer_.push_back(*imu);
        imu_queue_.pop();
    }
    while (auto *item = lidar_queue_.front()) {
        lidar_buffer_.push_back(item->cloud);
        time_buffer_.push_back(item->timestamp);
        lidar_queue_.pop();
    }

    if (lidar_buffer_.empty() || imu_buffer_.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed_) {
        measures_.lidar_ = lidar_buffer_.front();
        measures_.lidar_bag_time_ = time_buffer_.front();

        if (measures_.lidar_->points.size() <= 1) {
            spdlog::warn("Input point cloud has <= 1 point, using mean scan time estimate");
            lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
        } else {
            double last_curvature_s = measures_.lidar_->points.back().curvature / double(1000);

            if (last_curvature_s > 1e-6 &&
                (lidar_mean_scantime_ < 1e-6 || last_curvature_s >= 0.5 * lidar_mean_scantime_)) {
                // Per-point timing available — use curvature directly
                scan_num_++;
                lidar_end_time_ = measures_.lidar_bag_time_ + last_curvature_s;
                lidar_mean_scantime_ += (last_curvature_s - lidar_mean_scantime_) / scan_num_;
            } else if (last_curvature_s > 1e-6) {
                // Curvature present but suspiciously small — use running average
                lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
            } else {
                // No per-point timing (curvature ≈ 0). Estimate scan period from
                // the timestamp gap between consecutive scans.
                if (prev_lidar_bag_time_ > 0.0) {
                    double dt = measures_.lidar_bag_time_ - prev_lidar_bag_time_;
                    if (dt > 1e-6 && dt < 1.0) {
                        scan_num_++;
                        lidar_mean_scantime_ += (dt - lidar_mean_scantime_) / scan_num_;
                    }
                }
                if (lidar_mean_scantime_ > 1e-6) {
                    lidar_end_time_ = measures_.lidar_bag_time_ + lidar_mean_scantime_;
                } else {
                    // Very first scan, no reference — default to 0.1s (10 Hz)
                    lidar_mean_scantime_ = 0.1;
                    scan_num_ = 1;
                    lidar_end_time_ = measures_.lidar_bag_time_ + 0.1;
                }
            }
        }

        prev_lidar_bag_time_ = measures_.lidar_bag_time_;
        measures_.lidar_end_time_ = lidar_end_time_;
        lidar_pushed_ = true;
    }

    if (last_timestamp_imu_ < lidar_end_time_) {
        return false;
    }

    /*** push imu_ data, and pop from imu_ buffer ***/
    double imu_time = imu_buffer_.front()->timestamp;
    measures_.imu_.clear();
    while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
        imu_time = imu_buffer_.front()->timestamp;
        if (imu_time > lidar_end_time_) break;
        measures_.imu_.push_back(imu_buffer_.front());
        imu_buffer_.pop_front();
    }

    lidar_buffer_.pop_front();
    time_buffer_.pop_front();
    lidar_pushed_ = false;
    return true;
}

void LaserMapping::PrintState(const state_ikfom &s) {
    spdlog::debug("EKF state rot=[{},{},{},{}] pos=[{},{},{}] offset_rot=[{},{},{},{}] offset_t=[{},{},{}]",
                 s.rot.coeffs()[0], s.rot.coeffs()[1], s.rot.coeffs()[2], s.rot.coeffs()[3],
                 s.pos[0], s.pos[1], s.pos[2],
                 s.offset_R_L_I.coeffs()[0], s.offset_R_L_I.coeffs()[1], s.offset_R_L_I.coeffs()[2], s.offset_R_L_I.coeffs()[3],
                 s.offset_T_L_I[0], s.offset_T_L_I[1], s.offset_T_L_I[2]);
}

void LaserMapping::MapIncremental() {
    PointVector points_to_add;
    PointVector point_no_need_downsample;

    int cur_pts = scan_down_body_->size();
    points_to_add.reserve(cur_pts);
    point_no_need_downsample.reserve(cur_pts);

    std::vector<size_t> index(cur_pts);
    for (size_t i = 0; i < cur_pts; ++i) {
        index[i] = i;
    }

    // Two-pass approach to avoid data race on shared vectors:
    // Pass 1 (parallel): classify each point into ADD / NO_DOWNSAMPLE / SKIP
    enum PointAction : uint8_t { SKIP = 0, ADD = 1, NO_DOWNSAMPLE = 2 };
    std::vector<uint8_t> actions(cur_pts, SKIP);

    faster_lio::compat::for_each(faster_lio::compat::unseq, index.begin(), index.end(), [&](const size_t &i) {
        /* transform to world frame */
        PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

        /* decide if need add to map */
        PointType &point_world = scan_down_world_->points[i];

        // Dynamic point rejection: skip points whose map-plane fit quality
        // is poor (high RMS residual among k-NN neighbors). These are likely
        // on dynamic objects or at surface boundaries where the map is corrupted.
        if (map_quality_threshold_ > 0.0f && point_selected_surf_[i] &&
            fit_quality_[i] > map_quality_threshold_) {
            return;  // actions[i] stays SKIP
        }

        if (!nearest_points_[i].empty() && flg_EKF_inited_) {
            const PointVector &points_near = nearest_points_[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) * filter_size_map_min_;

            Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center;

            if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                actions[i] = NO_DOWNSAMPLE;
                return;
            }

            bool need_add = true;
            float dist = common::calc_dist(point_world.getVector3fMap(), center);
            if (points_near.size() >= options::NUM_MATCH_POINTS) {
                for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                    if (common::calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                        need_add = false;
                        break;
                    }
                }
            }
            if (need_add) {
                actions[i] = ADD;
            }
        } else {
            actions[i] = ADD;
        }
    });

    // Pass 2 (sequential): collect classified points into the two vectors
    for (int i = 0; i < cur_pts; ++i) {
        if (actions[i] == ADD) {
            points_to_add.emplace_back(scan_down_world_->points[i]);
        } else if (actions[i] == NO_DOWNSAMPLE) {
            point_no_need_downsample.emplace_back(scan_down_world_->points[i]);
        }
    }

    Timer::Evaluate(
        [&, this]() {
            ivox_->AddPoints(points_to_add);
            ivox_->AddPoints(point_no_need_downsample);
        },
        "    IVox Add Points");
}

// Thread safety: ObsModel uses par_unseq for read-only iVox queries and per-index
// writes to nearest_points_[i], point_selected_surf_[i], plane_coef_[i], residuals_[i].
// This is safe because Run() is sequential, and MapIncremental runs only after the
// EKF update completes, so no concurrent mutation of the iVox map occurs.
void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
    int cnt_pts = scan_down_body_->size();

    std::vector<size_t> index(cnt_pts);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
    }

    Timer::Evaluate(
        [&, this]() {
            auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
            auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

            faster_lio::compat::for_each(faster_lio::compat::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                PointType &point_body = scan_down_body_->points[i];
                PointType &point_world = scan_down_world_->points[i];

                common::V3F p_body = point_body.getVector3fMap();
                point_world.getVector3fMap() = R_wl * p_body + t_wl;
                point_world.intensity = point_body.intensity;

                auto &points_near = nearest_points_[i];
                if (ekfom_data.converge) {
                    points_near.clear();
                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                    if (point_selected_surf_[i]) {
                        point_selected_surf_[i] =
                            common::esti_plane(plane_coef_[i], points_near, esti_plane_threshold_,
                                               &fit_quality_[i]);
                    }
                }

                if (point_selected_surf_[i]) {
                    auto temp = point_world.getVector4fMap();
                    temp[3] = 1.0;
                    float pd2 = plane_coef_[i].dot(temp);

                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                    if (valid_corr) {
                        point_selected_surf_[i] = true;
                        residuals_[i] = pd2;
                    } else {
                        point_selected_surf_[i] = false;
                    }
                }
            });
        },
        "    ObsModel (Lidar Match)");

    effect_feat_num_ = 0;

    corr_pts_.resize(cnt_pts);
    corr_norm_.resize(cnt_pts);
    for (int i = 0; i < cnt_pts; i++) {
        if (point_selected_surf_[i]) {
            corr_norm_[effect_feat_num_] = plane_coef_[i];
            corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
            corr_pts_[effect_feat_num_][3] = residuals_[i];

            effect_feat_num_++;
        }
    }
    corr_pts_.resize(effect_feat_num_);
    corr_norm_.resize(effect_feat_num_);

    if (effect_feat_num_ < 1) {
        ekfom_data.valid = false;
        spdlog::warn("No effective feature points for IEKF update, skipping observation");
        return;
    }

    Timer::Evaluate(
        [&, this]() {
            ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);
            ekfom_data.h.resize(effect_feat_num_);

            index.resize(effect_feat_num_);
            const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
            const common::V3F off_t = s.offset_T_L_I.cast<float>();
            const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

            faster_lio::compat::for_each(faster_lio::compat::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                common::V3F point_this_be = corr_pts_[i].head<3>();
                common::M3F point_be_crossmat = SKEW_SYM_MATRIX(point_this_be);
                common::V3F point_this = off_R * point_this_be + off_t;
                common::M3F point_crossmat = SKEW_SYM_MATRIX(point_this);

                common::V3F norm_vec = corr_norm_[i].head<3>();

                common::V3F C(Rt * norm_vec);
                common::V3F A(point_crossmat * C);

                if (extrinsic_est_en_) {
                    common::V3F B(point_be_crossmat * off_R.transpose() * C);
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                        B[1], B[2], C[0], C[1], C[2];
                } else {
                    ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0;
                }

                ekfom_data.h(i) = -corr_pts_[i][3];
            });
        },
        "    ObsModel (IEKF Build Jacobian)");
}

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

void LaserMapping::SetPosestamp(PoseStamped &out) {
    out.pose.position = Eigen::Vector3d(state_point_.pos(0), state_point_.pos(1), state_point_.pos(2));
    out.pose.orientation = Eigen::Quaterniond(state_point_.rot.coeffs()[3], state_point_.rot.coeffs()[0],
                                               state_point_.rot.coeffs()[1], state_point_.rot.coeffs()[2]);
}

void LaserMapping::SaveFrameWorld() {
    if (!pcd_save_en_) {
        return;
    }

    PointCloudType::Ptr laserCloudWorld;
    if (dense_pub_en_) {
        PointCloudType::Ptr laserCloudFullRes(scan_undistort_);
        int size = laserCloudFullRes->points.size();
        laserCloudWorld = std::make_shared<PointCloudType>(size, 1);
        for (int i = 0; i < size; i++) {
            PointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
    } else {
        laserCloudWorld = scan_down_world_;
    }

    *pcl_wait_save_ += *laserCloudWorld;

    static int scan_wait_num = 0;
    scan_wait_num++;
    if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
        pcd_index_++;
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index_) +
                                   std::string(".pcd"));
        pcl::PCDWriter pcd_writer;
        spdlog::info("Saved point cloud scan to {}", all_points_dir);
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
        pcl_wait_save_->clear();
        scan_wait_num = 0;
    }
}

std::vector<PoseStamped> LaserMapping::GetTrajectory() const {
    std::lock_guard<std::mutex> lock(mtx_state_);
    return path_;
}

CloudPtr LaserMapping::GetUndistortedCloud() const {
    std::lock_guard<std::mutex> lock(mtx_state_);
    auto copy = std::make_shared<PointCloudType>();
    if (scan_undistort_) {
        *copy = *scan_undistort_;
    }
    return copy;
}

CloudPtr LaserMapping::GetDownsampledWorldCloud() const {
    std::lock_guard<std::mutex> lock(mtx_state_);
    auto copy = std::make_shared<PointCloudType>();
    if (scan_down_world_) {
        *copy = *scan_down_world_;
    }
    return copy;
}

CloudPtr LaserMapping::GetMapCloud() const {
    std::lock_guard<std::mutex> lock(mtx_state_);
    auto copy = std::make_shared<PointCloudType>();
    if (pcl_wait_save_) {
        *copy = *pcl_wait_save_;
    }
    return copy;
}

void LaserMapping::Savetrajectory(const std::string &traj_file) {
    std::ofstream ofs;
    ofs.open(traj_file, std::ios::out);
    if (!ofs.is_open()) {
        spdlog::error("Failed to open trajectory file for writing: {}", traj_file);
        return;
    }

    std::lock_guard<std::mutex> lock(mtx_state_);
    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : path_) {
        ofs << std::fixed << std::setprecision(6) << p.timestamp << " " << std::setprecision(15)
            << p.pose.position.x() << " " << p.pose.position.y() << " " << p.pose.position.z() << " "
            << p.pose.orientation.x() << " " << p.pose.orientation.y() << " " << p.pose.orientation.z() << " "
            << p.pose.orientation.w() << std::endl;
    }

    ofs.close();
}

///////////////////////////  private method /////////////////////////////////////////////////////////////////////

void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
    common::V3D p_body(pi->x, pi->y, pi->z);
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void LaserMapping::PointBodyToWorld(const common::V3F &pi, PointType *const po) {
    common::V3D p_body(pi.x(), pi.y(), pi.z());
    common::V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                         state_point_.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = std::abs(po->z);
}

void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    common::V3D p_body_lidar(pi->x, pi->y, pi->z);
    common::V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void LaserMapping::Finish() {
    if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
        std::string file_name = std::string("scans.pcd");
        std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        spdlog::info("Saved final point cloud to {}", file_name);
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
    }

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    if (diag_csv_ && diag_csv_->is_open()) {
        diag_csv_->flush();
        diag_csv_->close();
    }
#endif

    spdlog::info("Mapping finished, processed {} frames", frame_num_);
}

void LaserMapping::EnableDiagnostics(const std::string &csv_path) {
#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    if (csv_path.empty()) {
        diag_csv_.reset();
        return;
    }
    diag_csv_ = std::make_unique<std::ofstream>(csv_path);
    if (!diag_csv_->is_open()) {
        spdlog::error("Failed to open diagnostics CSV: {}", csv_path);
        diag_csv_.reset();
        return;
    }
    // Header — keep columns stable so downstream plotters can rely on them.
    (*diag_csv_)
        << "frame,timestamp,"
        << "scan_undistort_pts,scan_down_pts,map_grids,effect_feat,effect_ratio,"
        << "residual_mean,residual_rms,residual_max,"
        << "fit_quality_mean,fit_quality_rms,"
        // Dynamic-load proxy: among IEKF-selected features, fraction whose
        // k-NN plane fit RMS exceeded `map_quality_threshold` and were
        // therefore excluded from map insertion. Spikes when the scan is
        // dominated by moving objects or surface boundaries with corrupted
        // local geometry. 0.0 when threshold is disabled.
        << "dynamic_rejected,dynamic_fraction,"
        << "curv_min_ms,curv_max_ms,"
        << "pos_x,pos_y,pos_z,"
        << "vel_x,vel_y,vel_z,"
        << "quat_w,quat_x,quat_y,quat_z,"
        << "bg_x,bg_y,bg_z,ba_x,ba_y,ba_z,"
        << "grav_x,grav_y,grav_z,"
        << "ext_T_x,ext_T_y,ext_T_z,"
        << "t_undistort_us,t_downsample_us,t_iekf_us,t_mapinc_us,t_run_total_us,"
        << "rss_mb,cpu_delta_us,"
        // Filter's own 1-σ posterior std on pos (m), rot (rad), vel (m/s).
        // These come from the diagonal of P after the IEKF (+ optional wheel)
        // update. Compare to observed APE: if σ_pos ≪ APE the filter is
        // optimistic and noise covariances are mis-tuned.
        << "sigma_pos_x,sigma_pos_y,sigma_pos_z,"
        << "sigma_rot_x,sigma_rot_y,sigma_rot_z,"
        << "sigma_vel_x,sigma_vel_y,sigma_vel_z\n";
    diag_prev_cpu_us_ = 0;
    spdlog::info("Diagnostics CSV enabled: {}", csv_path);
#else
    (void)csv_path;
    spdlog::warn("EnableDiagnostics called but FASTER_LIO_ENABLE_DIAGNOSTICS is OFF");
#endif
}

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
void LaserMapping::WriteDiagnosticsRow() {
    if (!diag_csv_ || !diag_csv_->is_open()) return;

    const int cur_pts = scan_down_body_ ? static_cast<int>(scan_down_body_->size()) : 0;
    const int undist_pts = scan_undistort_ ? static_cast<int>(scan_undistort_->size()) : 0;
    const int map_grids = ivox_ ? static_cast<int>(ivox_->NumValidGrids()) : 0;

    // Residual stats over selected features only.
    // Same loop counts dynamic-rejected points: features that the IEKF used
    // (point_selected_surf_=1) but that exceeded `map_quality_threshold_`
    // and were therefore excluded from MapIncremental's iVox insertion.
    double res_sum = 0, res_sum_sq = 0, res_max = 0;
    double fq_sum = 0, fq_sum_sq = 0;
    int res_n = 0;
    int dyn_rejected = 0;
    for (int i = 0; i < cur_pts; ++i) {
        if (!point_selected_surf_[i]) continue;
        const double r = std::fabs(residuals_[i]);
        res_sum += r;
        res_sum_sq += r * r;
        if (r > res_max) res_max = r;
        fq_sum += fit_quality_[i];
        fq_sum_sq += fit_quality_[i] * fit_quality_[i];
        ++res_n;
        if (map_quality_threshold_ > 0.0f && fit_quality_[i] > map_quality_threshold_) {
            ++dyn_rejected;
        }
    }
    const double res_mean = res_n ? res_sum / res_n : 0.0;
    const double res_rms = res_n ? std::sqrt(res_sum_sq / res_n) : 0.0;
    const double fq_mean = res_n ? fq_sum / res_n : 0.0;
    const double fq_rms = res_n ? std::sqrt(fq_sum_sq / res_n) : 0.0;
    const double dyn_fraction = res_n ? double(dyn_rejected) / res_n : 0.0;

    // Curvature span (per-point timing sanity check)
    float curv_min = 0, curv_max = 0;
    if (scan_undistort_ && !scan_undistort_->empty()) {
        curv_min = scan_undistort_->points.front().curvature;
        curv_max = scan_undistort_->points.back().curvature;
    }

    const auto &s = state_point_;
    const double effect_ratio = cur_pts ? double(effect_feat_num_) / cur_pts : 0.0;

    // Resource usage: RSS in MB, CPU time delta in microseconds since last row.
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    // macOS reports ru_maxrss in BYTES
    const double rss_mb = ru.ru_maxrss / (1024.0 * 1024.0);
#else
    // Linux reports ru_maxrss in KB
    const double rss_mb = ru.ru_maxrss / 1024.0;
#endif
    const int64_t cpu_us =
        static_cast<int64_t>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) * 1000000 +
        static_cast<int64_t>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
    const int64_t cpu_delta_us = diag_prev_cpu_us_ ? (cpu_us - diag_prev_cpu_us_) : 0;
    diag_prev_cpu_us_ = cpu_us;

    // Posterior 1-σ from P diagonal. State tangent layout (23): pos 0..2,
    // rot 3..5, offset_R 6..8, offset_T 9..11, vel 12..14, bg 15..17,
    // ba 18..20, grav 21..22. Clamp tiny negatives to 0 before sqrt.
    auto safe_sqrt = [](double x) { return x > 0.0 ? std::sqrt(x) : 0.0; };
    const auto &P = kf_.get_P();
    const double sigma_pos_x = safe_sqrt(P(0, 0));
    const double sigma_pos_y = safe_sqrt(P(1, 1));
    const double sigma_pos_z = safe_sqrt(P(2, 2));
    const double sigma_rot_x = safe_sqrt(P(3, 3));
    const double sigma_rot_y = safe_sqrt(P(4, 4));
    const double sigma_rot_z = safe_sqrt(P(5, 5));
    const double sigma_vel_x = safe_sqrt(P(12, 12));
    const double sigma_vel_y = safe_sqrt(P(13, 13));
    const double sigma_vel_z = safe_sqrt(P(14, 14));

    (*diag_csv_) << std::fixed << std::setprecision(9)
        << frame_num_ << "," << lidar_end_time_ << ","
        << undist_pts << "," << cur_pts << "," << map_grids << ","
        << effect_feat_num_ << "," << effect_ratio << ","
        << res_mean << "," << res_rms << "," << res_max << ","
        << fq_mean << "," << fq_rms << ","
        << dyn_rejected << "," << dyn_fraction << ","
        << curv_min << "," << curv_max << ","
        << s.pos(0) << "," << s.pos(1) << "," << s.pos(2) << ","
        << s.vel(0) << "," << s.vel(1) << "," << s.vel(2) << ","
        << s.rot.w() << "," << s.rot.x() << "," << s.rot.y() << "," << s.rot.z() << ","
        << s.bg(0) << "," << s.bg(1) << "," << s.bg(2) << ","
        << s.ba(0) << "," << s.ba(1) << "," << s.ba(2) << ","
        << s.grav[0] << "," << s.grav[1] << "," << s.grav[2] << ","
        << s.offset_T_L_I(0) << "," << s.offset_T_L_I(1) << "," << s.offset_T_L_I(2) << ","
        << diag_t_undistort_us_ << "," << diag_t_downsample_us_ << ","
        << diag_t_iekf_us_ << "," << diag_t_mapinc_us_ << "," << diag_t_run_total_us_ << ","
        << rss_mb << "," << cpu_delta_us << ","
        << sigma_pos_x << "," << sigma_pos_y << "," << sigma_pos_z << ","
        << sigma_rot_x << "," << sigma_rot_y << "," << sigma_rot_z << ","
        << sigma_vel_x << "," << sigma_vel_y << "," << sigma_vel_z << "\n";
}
#endif

// =====================================================================
// Wheel-odometry fusion (optional, opt-in via yaml `wheel.enabled`).
// =====================================================================
// Applied as a sequential Kalman update AFTER the LiDAR IEKF converges.
// Each observation component is a scalar body-frame velocity:
//   z = (R_wb^T · v_world)[axis]
// Jacobian rows (nonzero only at rot δθ block 3..5 and vel block 12..14):
//   ∂z/∂δθ    = [v_body]×  [axis, :]
//   ∂z/∂v_world = R_wb^T   [axis, :]
// All other state components (pos, offsets, biases, gravity) carry zero
// in H, so the update touches only the (rot, vel) sub-blocks of the
// covariance and state. The cov update uses Joseph form for numerical
// stability under small-R scenarios (e.g. nhc_cov = 1e-3).

void LaserMapping::DrainWheelQueue() {
    while (auto p = wheel_queue_.front()) {
        wheel_buffer_.push_back(*p);
        wheel_queue_.pop();
    }
}

WheelOdomData::Ptr LaserMapping::FindWheelObs(double target_time) const {
    WheelOdomData::Ptr best;
    double best_gap = std::numeric_limits<double>::infinity();
    for (const auto &w : wheel_buffer_) {
        const double gap = std::fabs(w->timestamp - target_time);
        if (gap < best_gap) {
            best_gap = gap;
            best = w;
        }
    }
    if (!best || best_gap > wheel_max_time_gap_) return nullptr;
    return best;
}

void LaserMapping::ApplyBodyVelScalarUpdate(int axis, double z_body, double R_obs) {
    // Delegate to the pure-math helper in wheel_fusion.h. Keeping the math
    // outside this class keeps it independently testable without friend access.
    state_ikfom x = kf_.get_x();
    wheel_fusion::StateCov P = kf_.get_P();
    const auto rep = wheel_fusion::ApplyScalarBodyVelUpdate(x, P, axis, z_body, R_obs);
    if (rep.updated) {
        kf_.change_x(x);
        kf_.change_P(P);
        return;
    }
    switch (rep.status) {
        case wheel_fusion::ScalarUpdateReport::Status::InvalidAxis:
            spdlog::warn("Wheel update: invalid axis {}", axis);
            break;
        case wheel_fusion::ScalarUpdateReport::Status::NonPositiveR:
            spdlog::warn("Wheel update: non-positive R_obs ({})", R_obs);
            break;
        case wheel_fusion::ScalarUpdateReport::Status::NonFiniteInnov:
            spdlog::warn("Wheel update: non-finite innovation cov {}", rep.innov_cov);
            break;
        case wheel_fusion::ScalarUpdateReport::Status::GatedByMahalanobis:
            spdlog::warn("Wheel obs rejected (axis={}, mahalanobis²={:.2f}, z={} residual={})",
                         axis, rep.mahalanobis2, z_body, rep.residual);
            break;
        default:
            break;
    }
}

void LaserMapping::ApplyWheelObservations(double lidar_end_time) {
    if (!wheel_enabled_) return;
    DrainWheelQueue();

    WheelOdomData::Ptr obs = FindWheelObs(lidar_end_time);

    // GC stale wheel samples (> 1 s past the current frame) to cap memory.
    while (!wheel_buffer_.empty() &&
           wheel_buffer_.front()->timestamp < lidar_end_time - 1.0) {
        wheel_buffer_.pop_front();
    }

    const bool has_y = obs && obs->v_body_y.has_value();
    const bool has_z = obs && obs->v_body_z.has_value();
    // omega_z observation not yet integrated — see design note. The gyro
    // is a better source. Fields are accepted but no update is applied.

    if (obs) {
        ApplyBodyVelScalarUpdate(0, obs->v_body_x, wheel_cov_v_x_);
        if (has_y) ApplyBodyVelScalarUpdate(1, *obs->v_body_y, wheel_cov_v_y_);
        if (has_z) ApplyBodyVelScalarUpdate(2, *obs->v_body_z, wheel_cov_v_z_);
    }

    // Non-holonomic virtual observations for axes NOT supplied by the
    // wheel message. Emitting these when `has_y`/`has_z` would double-count
    // the lateral velocity with contradictory measurements.
    if (wheel_emit_nhc_v_y_ && !has_y) ApplyBodyVelScalarUpdate(1, 0.0, wheel_nhc_cov_);
    if (wheel_emit_nhc_v_z_ && !has_z) ApplyBodyVelScalarUpdate(2, 0.0, wheel_nhc_cov_);
}

}  // namespace faster_lio
