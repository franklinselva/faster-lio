#include <gflags/gflags.h>
#include <pcl/io/pcd_io.h>
#include <unistd.h>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "faster_lio/laser_mapping.h"
#include "faster_lio/logger.h"
#include "faster_lio/utils.h"

/// Run faster-LIO on extracted sensor data and evaluate against OptiTrack ground truth.
/// Supports both Livox binary (.bin) and PCD file inputs.
/// Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error).

DEFINE_string(config_file, "./config/default.yaml", "path to config file");
DEFINE_string(lidar_dir, "", "directory containing LiDAR files (Livox .bin or .pcd, numbered from 0)");
DEFINE_string(imu_file, "", "path to IMU CSV file (timestamp,ax,ay,az,gx,gy,gz)");
DEFINE_string(ground_truth_file, "", "path to OptiTrack ground truth CSV");
DEFINE_string(traj_log_file, "./Log/traj.txt", "path to estimated trajectory output (TUM format)");
DEFINE_string(gt_tum_file, "./Log/ground_truth_tum.txt", "path to ground truth output (TUM format)");
DEFINE_string(time_log_file, "./Log/time.log", "path to time log file");
DEFINE_int32(num_scans, 0, "number of LiDAR scans to process");
DEFINE_string(lidar_format, "livox", "LiDAR data format: 'livox' (.bin) or 'pcd' (.pcd)");
DEFINE_string(timestamps_file, "", "path to timestamps file (one timestamp per line, matching scan index)");
DEFINE_double(time_offset, 0.0, "time offset to add to estimated timestamps before matching ground truth (seconds)");
DEFINE_double(eval_start, 0.0, "skip estimated poses before this many seconds from start (warmup exclusion)");
DEFINE_double(eval_end, 0.0, "skip estimated poses after this many seconds from start (0=use all)");

void SigHandle(int sig) {
    faster_lio::options::FLAG_EXIT = true;
    spdlog::warn("Caught signal {}, shutting down", sig);
}

struct IMUEntry {
    double timestamp;
    double ax, ay, az;
    double gx, gy, gz;
};

struct GTPose {
    double timestamp;  // seconds
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
};

std::vector<IMUEntry> LoadIMUCSV(const std::string &path) {
    std::vector<IMUEntry> entries;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::error("Cannot open IMU file: {}", path);
        return entries;
    }

    std::string line;
    std::getline(ifs, line);
    if (!line.empty() && (line[0] == '#' || line[0] == 't')) {
        // header line, skip
    } else {
        ifs.seekg(0);
    }

    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        IMUEntry e;
        char comma;
        ss >> e.timestamp >> comma >> e.ax >> comma >> e.ay >> comma >> e.az >> comma >> e.gx >> comma >> e.gy >>
            comma >> e.gz;
        entries.push_back(e);
    }
    return entries;
}

/// Load a Livox binary file produced by extract_bag.py and convert it
/// directly to PointXYZINormal. Per-point offset_time (nanoseconds) becomes
/// curvature (milliseconds) — the faster-lio internal convention.
///
/// Format: double timebase, uint32 point_num, uint8 lidar_id,
///         then point_num * {float x, float y, float z, uint8 reflectivity, uint8 tag, uint8 line, uint32 offset_time}
///
/// Returns {cloud_ptr, timebase_seconds} or {nullptr, 0} on failure.
std::pair<PointCloudType::Ptr, double> LoadLivoxBin(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        spdlog::warn("Cannot open Livox file: {}", path);
        return {nullptr, 0.0};
    }

    double timebase;
    uint32_t point_num;
    uint8_t lidar_id;
    ifs.read(reinterpret_cast<char *>(&timebase), sizeof(double));
    ifs.read(reinterpret_cast<char *>(&point_num), sizeof(uint32_t));
    ifs.read(reinterpret_cast<char *>(&lidar_id), sizeof(uint8_t));

    auto cloud = std::make_shared<PointCloudType>();
    cloud->points.reserve(point_num);

    for (uint32_t i = 0; i < point_num; i++) {
        float x, y, z;
        uint8_t reflectivity, tag, line;
        uint32_t offset_time_ns;
        ifs.read(reinterpret_cast<char *>(&x), sizeof(float));
        ifs.read(reinterpret_cast<char *>(&y), sizeof(float));
        ifs.read(reinterpret_cast<char *>(&z), sizeof(float));
        ifs.read(reinterpret_cast<char *>(&reflectivity), sizeof(uint8_t));
        ifs.read(reinterpret_cast<char *>(&tag), sizeof(uint8_t));
        ifs.read(reinterpret_cast<char *>(&line), sizeof(uint8_t));
        ifs.read(reinterpret_cast<char *>(&offset_time_ns), sizeof(uint32_t));

        pcl::PointXYZINormal p;
        p.x = x;
        p.y = y;
        p.z = z;
        p.intensity = static_cast<float>(reflectivity);
        p.normal_x = 0;
        p.normal_y = 0;
        p.normal_z = 0;
        // offset_time is nanoseconds → curvature is milliseconds
        p.curvature = static_cast<float>(offset_time_ns) * 1e-6f;
        cloud->points.push_back(p);
    }
    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;

    return {cloud, timebase};
}

std::vector<double> LoadTimestamps(const std::string &path) {
    std::vector<double> timestamps;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::error("Cannot open timestamps file: {}", path);
        return timestamps;
    }
    double ts;
    while (ifs >> ts) {
        timestamps.push_back(ts);
    }
    spdlog::info("Loaded {} timestamps from {}", timestamps.size(), path);
    return timestamps;
}

std::vector<GTPose> LoadGroundTruth(const std::string &path) {
    std::vector<GTPose> poses;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::error("Cannot open ground truth file: {}", path);
        return poses;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        char comma;
        double ts_ns, x, y, z, euler_x, euler_y, euler_z, qw, qx, qy, qz;
        ss >> ts_ns >> comma >> x >> comma >> y >> comma >> z >> comma >> euler_x >> comma >> euler_y >> comma >>
            euler_z >> comma >> qw >> comma >> qx >> comma >> qy >> comma >> qz;

        GTPose gt;
        gt.timestamp = ts_ns * 1e-9;  // ns to seconds
        gt.position = Eigen::Vector3d(x, y, z);
        gt.orientation = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        poses.push_back(gt);
    }

    spdlog::info("Loaded {} ground truth poses from {}", poses.size(), path);
    return poses;
}

int FindNearestGT(const std::vector<GTPose> &gt, double query_time, double max_dt = 0.05) {
    if (gt.empty()) return -1;

    auto it = std::lower_bound(gt.begin(), gt.end(), query_time,
                               [](const GTPose &p, double t) { return p.timestamp < t; });

    int idx = -1;
    double best_dt = max_dt;

    if (it != gt.end()) {
        double dt = std::abs(it->timestamp - query_time);
        if (dt < best_dt) {
            best_dt = dt;
            idx = static_cast<int>(std::distance(gt.begin(), it));
        }
    }
    if (it != gt.begin()) {
        --it;
        double dt = std::abs(it->timestamp - query_time);
        if (dt < best_dt) {
            idx = static_cast<int>(std::distance(gt.begin(), it));
        }
    }
    return idx;
}

Eigen::Matrix4d AlignTrajectories(const std::vector<Eigen::Vector3d> &src,
                                  const std::vector<Eigen::Vector3d> &dst) {
    assert(src.size() == dst.size() && src.size() >= 3);

    int n = static_cast<int>(src.size());
    Eigen::MatrixXd src_mat(3, n), dst_mat(3, n);
    for (int i = 0; i < n; i++) {
        src_mat.col(i) = src[i];
        dst_mat.col(i) = dst[i];
    }

    Eigen::Matrix4d T = Eigen::umeyama(src_mat, dst_mat, false);
    return T;
}

struct EvalMetrics {
    double ate_rmse = 0.0;
    double ate_mean = 0.0;
    double ate_max = 0.0;
    double rpe_rmse = 0.0;
    double rpe_mean = 0.0;
    int num_pairs = 0;
};

EvalMetrics ComputeMetrics(const std::vector<faster_lio::PoseStamped> &estimated,
                           const std::vector<GTPose> &ground_truth) {
    EvalMetrics metrics;

    std::vector<Eigen::Vector3d> est_positions, gt_positions;

    // Determine time window for evaluation
    double first_ts = estimated.front().timestamp;
    double eval_start_ts = first_ts + FLAGS_eval_start;
    double eval_end_ts = (FLAGS_eval_end > 0.0) ? first_ts + FLAGS_eval_end : std::numeric_limits<double>::max();

    if (FLAGS_eval_start > 0.0 || FLAGS_eval_end > 0.0) {
        spdlog::info("Evaluation window: [{:.1f}s, {:.1f}s] from start", FLAGS_eval_start,
                     FLAGS_eval_end > 0.0 ? FLAGS_eval_end : (estimated.back().timestamp - first_ts));
    }

    for (size_t i = 0; i < estimated.size(); i++) {
        if (estimated[i].timestamp < eval_start_ts || estimated[i].timestamp > eval_end_ts) continue;
        int gt_idx = FindNearestGT(ground_truth, estimated[i].timestamp + FLAGS_time_offset);
        if (gt_idx >= 0) {
            est_positions.push_back(estimated[i].pose.position);
            gt_positions.push_back(ground_truth[gt_idx].position);
        }
    }

    metrics.num_pairs = static_cast<int>(est_positions.size());
    if (metrics.num_pairs < 3) {
        spdlog::warn("Only {} matched pose pairs, need at least 3 for alignment", metrics.num_pairs);
        return metrics;
    }

    spdlog::info("Matched {} / {} estimated poses to ground truth", metrics.num_pairs, estimated.size());

    Eigen::Matrix4d T_align = AlignTrajectories(est_positions, gt_positions);
    Eigen::Matrix3d R_align = T_align.block<3, 3>(0, 0);
    Eigen::Vector3d t_align = T_align.block<3, 1>(0, 3);

    spdlog::info("Alignment translation: [{:.4f}, {:.4f}, {:.4f}]", t_align.x(), t_align.y(), t_align.z());

    // ATE
    std::vector<double> ate_errors;
    for (int i = 0; i < metrics.num_pairs; i++) {
        Eigen::Vector3d aligned = R_align * est_positions[i] + t_align;
        double err = (aligned - gt_positions[i]).norm();
        ate_errors.push_back(err);
    }

    double sum_sq = 0.0, sum = 0.0, max_err = 0.0;
    for (double e : ate_errors) {
        sum_sq += e * e;
        sum += e;
        max_err = std::max(max_err, e);
    }
    metrics.ate_rmse = std::sqrt(sum_sq / ate_errors.size());
    metrics.ate_mean = sum / ate_errors.size();
    metrics.ate_max = max_err;

    // RPE
    std::vector<double> rpe_errors;
    for (int i = 1; i < metrics.num_pairs; i++) {
        Eigen::Vector3d est_rel = (R_align * est_positions[i] + t_align) - (R_align * est_positions[i - 1] + t_align);
        Eigen::Vector3d gt_rel = gt_positions[i] - gt_positions[i - 1];
        double err = (est_rel - gt_rel).norm();
        rpe_errors.push_back(err);
    }

    if (!rpe_errors.empty()) {
        double rpe_sum_sq = 0.0, rpe_sum = 0.0;
        for (double e : rpe_errors) {
            rpe_sum_sq += e * e;
            rpe_sum += e;
        }
        metrics.rpe_rmse = std::sqrt(rpe_sum_sq / rpe_errors.size());
        metrics.rpe_mean = rpe_sum / rpe_errors.size();
    }

    return metrics;
}

void SaveGTasTUM(const std::vector<GTPose> &gt, const std::string &path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        spdlog::error("Cannot open file for writing: {}", path);
        return;
    }
    ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
    for (const auto &p : gt) {
        ofs << std::fixed << std::setprecision(6) << p.timestamp << " " << std::setprecision(15) << p.position.x()
            << " " << p.position.y() << " " << p.position.z() << " " << p.orientation.x() << " "
            << p.orientation.y() << " " << p.orientation.z() << " " << p.orientation.w() << std::endl;
    }
    spdlog::info("Saved ground truth TUM trajectory to: {}", path);
}

void PrintMetrics(const EvalMetrics &m) {
    spdlog::info("============================================");
    spdlog::info("  Trajectory Evaluation Results");
    spdlog::info("============================================");
    spdlog::info("  Matched poses:    {}", m.num_pairs);
    spdlog::info("  ATE RMSE:         {:.4f} m", m.ate_rmse);
    spdlog::info("  ATE Mean:         {:.4f} m", m.ate_mean);
    spdlog::info("  ATE Max:          {:.4f} m", m.ate_max);
    spdlog::info("  RPE RMSE:         {:.6f} m", m.rpe_rmse);
    spdlog::info("  RPE Mean:         {:.6f} m", m.rpe_mean);
    spdlog::info("============================================");
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    faster_lio::InitLogger(spdlog::level::info);

    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    if (!laser_mapping->Init(FLAGS_config_file)) {
        spdlog::error("Laser mapping initialization failed for config: {}", FLAGS_config_file);
        return -1;
    }

    signal(SIGINT, SigHandle);

    if (FLAGS_imu_file.empty() || FLAGS_lidar_dir.empty() || FLAGS_num_scans <= 0) {
        spdlog::error("Missing required arguments: --lidar_dir, --imu_file, and --num_scans");
        return -1;
    }

    bool use_livox = (FLAGS_lidar_format == "livox");
    spdlog::info("LiDAR format: {}", FLAGS_lidar_format);

    // Load ground truth (optional)
    std::vector<GTPose> ground_truth;
    if (!FLAGS_ground_truth_file.empty()) {
        ground_truth = LoadGroundTruth(FLAGS_ground_truth_file);
    }

    // Load scan timestamps (optional — uses IMU-derived timestamps if absent)
    std::vector<double> scan_timestamps;
    if (!FLAGS_timestamps_file.empty()) {
        scan_timestamps = LoadTimestamps(FLAGS_timestamps_file);
    }

    // Load all IMU data
    auto imu_entries = LoadIMUCSV(FLAGS_imu_file);
    spdlog::info("Loaded {} IMU entries from {}", imu_entries.size(), FLAGS_imu_file);

    size_t imu_idx = 0;

    for (int scan_i = 0; scan_i < FLAGS_num_scans && !faster_lio::options::FLAG_EXIT; scan_i++) {
        // Load the scan (either .bin with per-point timing or .pcd) and its timestamp.
        PointCloudType::Ptr cloud;
        double scan_timestamp;

        if (use_livox) {
            std::string livox_path = FLAGS_lidar_dir + "/" + std::to_string(scan_i) + ".bin";
            auto [loaded, ts] = LoadLivoxBin(livox_path);
            if (!loaded || loaded->empty()) {
                spdlog::warn("Empty or missing Livox scan: {}", livox_path);
                continue;
            }
            cloud = loaded;
            scan_timestamp = ts;
        } else {
            std::string pcd_path = FLAGS_lidar_dir + "/" + std::to_string(scan_i) + ".pcd";
            cloud = std::make_shared<PointCloudType>();
            if (pcl::io::loadPCDFile(pcd_path, *cloud) == -1) {
                spdlog::warn("Cannot load PCD file: {}", pcd_path);
                continue;
            }

            if (scan_i < static_cast<int>(scan_timestamps.size())) {
                scan_timestamp = scan_timestamps[scan_i];
            } else if (!imu_entries.empty() && imu_idx < imu_entries.size()) {
                scan_timestamp = imu_entries[std::min(imu_idx + 10, imu_entries.size() - 1)].timestamp;
            } else {
                scan_timestamp = scan_i * 0.1;
            }
        }

        // Feed IMU samples up to this scan's timestamp.
        while (imu_idx < imu_entries.size() && imu_entries[imu_idx].timestamp <= scan_timestamp) {
            faster_lio::IMUData imu;
            imu.timestamp = imu_entries[imu_idx].timestamp;
            imu.linear_acceleration =
                Eigen::Vector3d(imu_entries[imu_idx].ax, imu_entries[imu_idx].ay, imu_entries[imu_idx].az);
            imu.angular_velocity =
                Eigen::Vector3d(imu_entries[imu_idx].gx, imu_entries[imu_idx].gy, imu_entries[imu_idx].gz);
            laser_mapping->AddIMU(imu);
            imu_idx++;
        }

        laser_mapping->AddPointCloud(cloud, scan_timestamp);

        faster_lio::Timer::Evaluate([&laser_mapping]() { laser_mapping->Run(); }, "Laser Mapping Single Run");
    }

    spdlog::info("Finishing mapping");
    laser_mapping->Finish();

    double fps = 1.0 / (faster_lio::Timer::GetMeanTime("Laser Mapping Single Run") / 1000.);
    spdlog::info("Average FPS: {:.2f}", fps);

    // Save estimated trajectory
    spdlog::info("Saving estimated trajectory to: {}", FLAGS_traj_log_file);
    laser_mapping->Savetrajectory(FLAGS_traj_log_file);

    // Evaluate against ground truth
    if (!ground_truth.empty()) {
        const auto &trajectory = laser_mapping->GetTrajectory();

        if (trajectory.empty()) {
            spdlog::warn("No trajectory data to evaluate");
        } else {
            spdlog::info("Evaluating {} estimated poses against {} ground truth poses", trajectory.size(),
                         ground_truth.size());

            EvalMetrics metrics = ComputeMetrics(trajectory, ground_truth);
            PrintMetrics(metrics);

            SaveGTasTUM(ground_truth, FLAGS_gt_tum_file);
        }
    } else {
        spdlog::info("No ground truth provided, skipping evaluation");
    }

    faster_lio::Timer::PrintAll();
    faster_lio::Timer::DumpIntoFile(FLAGS_time_log_file);

    return 0;
}
