#include <pcl/io/pcd_io.h>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iomanip>
#include "faster_lio/compat.h"

#include "faster_lio/laser_mapping.h"
#include "faster_lio/utils.h"

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
    // get params from yaml
    int lidar_type, ivox_nearby_type;
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
        preprocess_->SetBlind(yaml["preprocess"]["blind"].as<double>());
        if (yaml["preprocess"]["time_scale"]) {
            preprocess_->SetTimeScale(yaml["preprocess"]["time_scale"].as<float>());
        }
        lidar_type = yaml["preprocess"]["lidar_type"].as<int>();
        preprocess_->SetNumScans(yaml["preprocess"]["scan_line"].as<int>());
        preprocess_->SetPointFilterNum(yaml["point_filter_num"].as<int>());
        preprocess_->SetFeatureEnabled(yaml["feature_extract_enable"].as<bool>());
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

    spdlog::info("Lidar type: {}", lidar_type);
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        spdlog::info("Using AVIA Lidar");
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        spdlog::info("Using Velodyne 32 Lidar");
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        spdlog::info("Using OUST 64 Lidar");
    } else if (lidar_type == 4) {
        preprocess_->SetLidarType(LidarType::HESAIxt32);
        spdlog::info("Using Hesai Pandar 32 Lidar");
    } else if (lidar_type == 5) {
        preprocess_->SetLidarType(LidarType::ROBOSENSE);
        spdlog::info("Using Robosense Lidar");
    } else if (lidar_type == 6) {
        preprocess_->SetLidarType(LidarType::LIVOX);
        spdlog::info("Using Livox Lidar");
    } else {
        spdlog::warn("Unknown lidar_type: {}, expected 1-6", lidar_type);
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

    {
        std::lock_guard<std::mutex> lock(mtx_buffer_);
        if (timestamp < last_timestamp_imu_) {
            spdlog::warn("IMU timestamp went backwards (ts={}), clearing IMU buffer", timestamp);
            imu_buffer_.clear();
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.emplace_back(msg);
    }
}

void LaserMapping::AddPointCloud(const PointCloudType::Ptr &cloud, double timestamp) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (timestamp < last_timestamp_lidar_) {
                spdlog::error("Lidar timestamp went backwards, clearing lidar buffer");
                lidar_buffer_.clear();
            }

            last_timestamp_lidar_ = timestamp;

            if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                spdlog::info("IMU and LiDAR not Synced, IMU time: {}, lidar header time: {}",
                             last_timestamp_imu_, last_timestamp_lidar_);
            }

            if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                spdlog::info("Self sync IMU and LiDAR, time diff is {}", timediff_lidar_wrt_imu_);
            }

            auto ptr = std::make_shared<PointCloudType>();
            preprocess_->Process(*cloud, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(timestamp);
        },
        "Preprocess (Generic)");
}

void LaserMapping::AddPointCloud(const LivoxCloud &cloud) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            double timestamp = cloud.timebase;
            if (timestamp < last_timestamp_lidar_) {
                spdlog::warn("Lidar timestamp went backwards, clearing lidar buffer");
                lidar_buffer_.clear();
            }

            last_timestamp_lidar_ = timestamp;

            if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 && !imu_buffer_.empty() &&
                !lidar_buffer_.empty()) {
                spdlog::info("IMU and LiDAR not Synced, IMU time: {}, lidar header time: {}",
                             last_timestamp_imu_, last_timestamp_lidar_);
            }

            if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
                !imu_buffer_.empty()) {
                timediff_set_flg_ = true;
                timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
                spdlog::info("Self sync IMU and LiDAR, time diff is {}", timediff_lidar_wrt_imu_);
            }

            auto ptr = std::make_shared<PointCloudType>();
            preprocess_->Process(cloud, ptr);
            lidar_buffer_.emplace_back(ptr);
            time_buffer_.emplace_back(last_timestamp_lidar_);
        },
        "Preprocess (Livox)");
}

template <typename PointT>
void LaserMapping::AddPointCloud(const pcl::PointCloud<PointT> &cloud, double timestamp) {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    Timer::Evaluate(
        [&, this]() {
            scan_count_++;
            if (timestamp < last_timestamp_lidar_) {
                spdlog::error("Lidar timestamp went backwards, clearing lidar buffer");
                lidar_buffer_.clear();
            }

            auto ptr = std::make_shared<PointCloudType>();
            preprocess_->Process(cloud, ptr);
            lidar_buffer_.push_back(ptr);
            time_buffer_.push_back(timestamp);
            last_timestamp_lidar_ = timestamp;
        },
        "Preprocess (Standard)");
}

// Explicit template instantiations
template void LaserMapping::AddPointCloud(const pcl::PointCloud<velodyne_pcl::Point> &, double);
template void LaserMapping::AddPointCloud(const pcl::PointCloud<ouster_pcl::Point> &, double);
template void LaserMapping::AddPointCloud(const pcl::PointCloud<hesai_pcl::Point> &, double);
template void LaserMapping::AddPointCloud(const pcl::PointCloud<robosense_pcl::Point> &, double);
template void LaserMapping::AddPointCloud(const pcl::PointCloud<livox_pcl::Point> &, double);

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

    /// IMU process, kf prediction, undistortion
    p_imu_->Process(measures_, kf_, scan_undistort_);
    if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
        spdlog::warn("Empty undistorted scan, skipping frame");
        return;
    }

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
    Timer::Evaluate(
        [&, this]() {
            voxel_scan_.setInputCloud(scan_undistort_);
            voxel_scan_.filter(*scan_down_body_);
        },
        "Downsample PointCloud");

    int cur_pts = scan_down_body_->size();
    if (cur_pts < 5) {
        spdlog::warn("Too few points after downsampling ({} -> {}), skipping frame", scan_undistort_->size(), scan_down_body_->size());
        return;
    }
    scan_down_world_->resize(cur_pts);
    nearest_points_.resize(cur_pts);
    // No value-init needed: ObsModel writes every element of residuals_ and point_selected_surf_
    residuals_.resize(cur_pts);
    point_selected_surf_.resize(cur_pts);
    plane_coef_.resize(cur_pts, common::V4F::Zero());

    // ICP and iterated Kalman filter update
    Timer::Evaluate(
        [&, this]() {
            double solve_H_time = 0;
            kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV, solve_H_time);
            state_point_ = kf_.get_x();
            euler_cur_ = SO3ToEuler(state_point_.rot);
            pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
        },
        "IEKF Solve and Update");

    // update local map
    Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");

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

    // Debug variables
    frame_num_++;
}

bool LaserMapping::SyncPackages() {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
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
                            common::esti_plane(plane_coef_[i], points_near, esti_plane_threshold_);
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

    spdlog::info("Mapping finished, processed {} frames", frame_num_);
}
}  // namespace faster_lio
