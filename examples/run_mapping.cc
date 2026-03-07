#include <gflags/gflags.h>
#include <pcl/io/pcd_io.h>
#include <unistd.h>
#include <csignal>
#include <fstream>
#include <sstream>

#include "faster_lio/logger.h"
#include "faster_lio/laser_mapping.h"
#include "faster_lio/utils.h"

/// run faster-LIO in offline mode: reads PCD files + IMU CSV, feeds data through the API

DEFINE_string(config_file, "./config/avia.yaml", "path to config file");
DEFINE_string(pcd_dir, "", "directory containing PCD files (numbered: 0.pcd, 1.pcd, ...)");
DEFINE_string(imu_file, "", "path to IMU CSV file (timestamp,ax,ay,az,gx,gy,gz)");
DEFINE_string(time_log_file, "./Log/time.log", "path to time log file");
DEFINE_string(traj_log_file, "./Log/traj.txt", "path to traj log file");
DEFINE_int32(num_pcd, 0, "number of PCD files to process");

void SigHandle(int sig) {
    faster_lio::options::FLAG_EXIT = true;
    spdlog::warn("Caught signal {}, shutting down", sig);
}

struct IMUEntry {
    double timestamp;
    double ax, ay, az;
    double gx, gy, gz;
};

std::vector<IMUEntry> LoadIMUCSV(const std::string &path) {
    std::vector<IMUEntry> entries;
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        spdlog::error("Cannot open IMU file: {}", path);
        return entries;
    }

    std::string line;
    // skip header if present
    std::getline(ifs, line);
    if (!line.empty() && (line[0] == '#' || line[0] == 't')) {
        // header line, skip
    } else {
        // not a header, parse it
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

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    faster_lio::InitLogger(spdlog::level::info);

    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    if (!laser_mapping->Init(FLAGS_config_file)) {
        spdlog::error("Laser mapping initialization failed for config: {}", FLAGS_config_file);
        return -1;
    }

    signal(SIGINT, SigHandle);

    if (FLAGS_imu_file.empty() || FLAGS_pcd_dir.empty() || FLAGS_num_pcd <= 0) {
        spdlog::error("Missing required arguments: --pcd_dir, --imu_file, and --num_pcd");
        return -1;
    }

    // Load all IMU data
    auto imu_entries = LoadIMUCSV(FLAGS_imu_file);
    spdlog::info("Loaded {} IMU entries from {}", imu_entries.size(), FLAGS_imu_file);

    size_t imu_idx = 0;

    for (int pcd_i = 0; pcd_i < FLAGS_num_pcd && !faster_lio::options::FLAG_EXIT; pcd_i++) {
        std::string pcd_path = FLAGS_pcd_dir + "/" + std::to_string(pcd_i) + ".pcd";
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>());
        if (pcl::io::loadPCDFile(pcd_path, *cloud) == -1) {
            spdlog::warn("Cannot load PCD file: {}", pcd_path);
            continue;
        }

        // Assume PCD timestamp is encoded in the filename or we use a simple index-based time
        double pcd_timestamp = pcd_i * 0.1;  // placeholder: 10 Hz
        if (!imu_entries.empty() && imu_idx < imu_entries.size()) {
            pcd_timestamp = imu_entries[std::min(imu_idx + 10, imu_entries.size() - 1)].timestamp;
        }

        // Feed IMU data up to this point cloud timestamp
        while (imu_idx < imu_entries.size() && imu_entries[imu_idx].timestamp <= pcd_timestamp) {
            faster_lio::IMUData imu;
            imu.timestamp = imu_entries[imu_idx].timestamp;
            imu.linear_acceleration = Eigen::Vector3d(imu_entries[imu_idx].ax, imu_entries[imu_idx].ay,
                                                       imu_entries[imu_idx].az);
            imu.angular_velocity = Eigen::Vector3d(imu_entries[imu_idx].gx, imu_entries[imu_idx].gy,
                                                    imu_entries[imu_idx].gz);
            laser_mapping->AddIMU(imu);
            imu_idx++;
        }

        // Feed point cloud
        PointCloudType::Ptr pcl_cloud(new PointCloudType());
        *pcl_cloud = *cloud;
        laser_mapping->AddPointCloud(pcl_cloud, pcd_timestamp);

        faster_lio::Timer::Evaluate([&laser_mapping]() { laser_mapping->Run(); }, "Laser Mapping Single Run");
    }

    spdlog::info("Finishing mapping");
    laser_mapping->Finish();

    double fps = 1.0 / (faster_lio::Timer::GetMeanTime("Laser Mapping Single Run") / 1000.);
    spdlog::info("Average FPS: {:.2f}", fps);

    spdlog::info("Saving trajectory to: {}", FLAGS_traj_log_file);
    laser_mapping->Savetrajectory(FLAGS_traj_log_file);

    faster_lio::Timer::PrintAll();
    faster_lio::Timer::DumpIntoFile(FLAGS_time_log_file);

    return 0;
}
