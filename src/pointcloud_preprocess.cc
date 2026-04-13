#include "faster_lio/pointcloud_preprocess.h"

#include <spdlog/spdlog.h>
#include "faster_lio/compat.h"

namespace faster_lio {

void PointCloudPreprocess::Set(LidarType lid_type, double bld, int pfilt_num) {
    lidar_type_ = lid_type;
    blind_ = bld;
    point_filter_num_ = pfilt_num;
}

void PointCloudPreprocess::Process(const LivoxCloud &msg, PointCloudType::Ptr &pcl_out) {
    AviaHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const pcl::PointCloud<velodyne_pcl::Point> &msg, PointCloudType::Ptr &pcl_out) {
    VelodyneHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const pcl::PointCloud<ouster_pcl::Point> &msg, PointCloudType::Ptr &pcl_out) {
    Oust64Handler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const pcl::PointCloud<hesai_pcl::Point> &msg, PointCloudType::Ptr &pcl_out) {
    HesaiHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const pcl::PointCloud<robosense_pcl::Point> &msg, PointCloudType::Ptr &pcl_out) {
    RobosenseHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::Process(const pcl::PointCloud<livox_pcl::Point> &msg, PointCloudType::Ptr &pcl_out) {
    LivoxHandler(msg);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::AviaHandler(const LivoxCloud &msg) {
    cloud_out_.clear();
    cloud_full_.clear();
    int plsize = msg.point_num;

    if (plsize < 2) {
        spdlog::warn("AviaHandler: received point cloud with {} points, skipping", plsize);
        return;
    }

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<char> is_valid_pt(plsize, false);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;
    }

    faster_lio::compat::for_each(faster_lio::compat::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        if ((msg.points[i].line < num_scans_) &&
            ((msg.points[i].tag & 0x30) == 0x10 || (msg.points[i].tag & 0x30) == 0x00)) {
            if (i % point_filter_num_ == 0) {
                cloud_full_[i].x = msg.points[i].x;
                cloud_full_[i].y = msg.points[i].y;
                cloud_full_[i].z = msg.points[i].z;
                cloud_full_[i].intensity = msg.points[i].reflectivity;
                cloud_full_[i].curvature = static_cast<float>(msg.points[i].offset_time) / static_cast<float>(1000000);

                if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
                    (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
                    (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                        (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                             cloud_full_[i].z * cloud_full_[i].z >
                         (blind_ * blind_))) {
                    is_valid_pt[i] = true;
                }
            }
        }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }
}

void PointCloudPreprocess::Oust64Handler(const pcl::PointCloud<ouster_pcl::Point> &pl_orig) {
    cloud_out_.clear();
    cloud_full_.clear();
    int plsize = pl_orig.size();
    if (plsize == 0) {
        spdlog::warn("Oust64Handler: received empty point cloud, skipping");
        return;
    }

    cloud_out_.reserve(plsize);

    for (int i = 0; i < pl_orig.points.size(); i++) {
        if (i % point_filter_num_ != 0) continue;

        double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                       pl_orig.points[i].z * pl_orig.points[i].z;

        if (range < (blind_ * blind_)) continue;

        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pl_orig.points[i].t / 1e6;  // curvature unit: ms

        cloud_out_.points.push_back(added_pt);
    }
}

void PointCloudPreprocess::VelodyneHandler(const pcl::PointCloud<velodyne_pcl::Point> &pl_orig) {
    cloud_out_.clear();
    cloud_full_.clear();

    int plsize = pl_orig.points.size();
    if (plsize == 0) {
        spdlog::warn("VelodyneHandler: received empty point cloud, skipping");
        return;
    }

    cloud_out_.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);    // yaw of first scan point
    std::vector<float> yaw_last(num_scans_, 0.0);   // yaw of last scan point
    std::vector<float> time_last(num_scans_, 0.0);  // last offset time

    if (pl_orig.points[plsize - 1].time > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        double yaw_end = yaw_first;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
                break;
            }
        }
    }

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_scale_;  // curvature unit: ms

        if (!given_offset_time_) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            // compute offset time
            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer]) added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }
}

// Shared handler for lidar types with timestamp + ring fields (Hesai, Robosense)
template <typename PointT>
void PointCloudPreprocess::TimestampRingHandler(const pcl::PointCloud<PointT> &pl_orig) {
    cloud_out_.clear();
    cloud_full_.clear();

    int plsize = pl_orig.points.size();
    if (plsize == 0) {
        spdlog::warn("TimestampRingHandler: received empty point cloud, skipping");
        return;
    }

    cloud_out_.reserve(plsize);

    double omega_l = 3.61;  // scan angular velocity
    std::vector<bool> is_first(num_scans_, true);
    std::vector<double> yaw_fp(num_scans_, 0.0);
    std::vector<float> yaw_last(num_scans_, 0.0);
    std::vector<float> time_last(num_scans_, 0.0);

    if (pl_orig.points[plsize - 1].timestamp > 0) {
        given_offset_time_ = true;
    } else {
        given_offset_time_ = false;
        double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
        int layer_first = pl_orig.points[0].ring;
        for (uint i = plsize - 1; i > 0; i--) {
            if (pl_orig.points[i].ring == layer_first) {
                break;
            }
        }
    }

    double time_head = pl_orig.points[0].timestamp;

    for (int i = 0; i < plsize; i++) {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = (pl_orig.points[i].timestamp - time_head) * 1000.f;  // curvature unit: ms

        if (!given_offset_time_) {
            int layer = pl_orig.points[i].ring;
            double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

            if (is_first[layer]) {
                yaw_fp[layer] = yaw_angle;
                is_first[layer] = false;
                added_pt.curvature = 0.0;
                yaw_last[layer] = yaw_angle;
                time_last[layer] = added_pt.curvature;
                continue;
            }

            if (yaw_angle <= yaw_fp[layer]) {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle) / omega_l;
            } else {
                added_pt.curvature = (yaw_fp[layer] - yaw_angle + 360.0) / omega_l;
            }

            if (added_pt.curvature < time_last[layer]) added_pt.curvature += 360.0 / omega_l;

            yaw_last[layer] = yaw_angle;
            time_last[layer] = added_pt.curvature;
        }

        if (i % point_filter_num_ == 0) {
            if (added_pt.x * added_pt.x + added_pt.y * added_pt.y + added_pt.z * added_pt.z > (blind_ * blind_)) {
                cloud_out_.points.push_back(added_pt);
            }
        }
    }
}

void PointCloudPreprocess::HesaiHandler(const pcl::PointCloud<hesai_pcl::Point> &pl_orig) {
    TimestampRingHandler(pl_orig);
}

void PointCloudPreprocess::RobosenseHandler(const pcl::PointCloud<robosense_pcl::Point> &pl_orig) {
    TimestampRingHandler(pl_orig);
}

void PointCloudPreprocess::Process(const PointCloudType &cloud, PointCloudType::Ptr &pcl_out) {
    GenericHandler(cloud);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::GenericHandler(const PointCloudType &cloud) {
    cloud_out_.clear();

    int plsize = cloud.points.size();
    if (plsize == 0) {
        spdlog::warn("GenericHandler: received empty point cloud, skipping");
        return;
    }

    cloud_out_.reserve(plsize);

    for (int i = 0; i < plsize; i++) {
        if (i % point_filter_num_ != 0) continue;

        const auto &pt = cloud.points[i];

        float d2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;

        // Filter zero-origin points (pre-allocated padding)
        if (d2 < 1e-6f) continue;

        // Filter blind distance
        if (d2 < blind_ * blind_) continue;

        PointType added_pt;
        added_pt.x = pt.x;
        added_pt.y = pt.y;
        added_pt.z = pt.z;
        added_pt.intensity = pt.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pt.curvature;  // preserve if caller set it, 0 otherwise

        cloud_out_.points.push_back(added_pt);
    }
}

void PointCloudPreprocess::LivoxHandler(const pcl::PointCloud<livox_pcl::Point> &pl_orig) {
    cloud_out_.clear();
    cloud_full_.clear();

    int plsize = pl_orig.points.size();

    if (plsize < 2) {
        spdlog::warn("LivoxHandler: received point cloud with {} points, skipping", plsize);
        return;
    }

    cloud_out_.reserve(plsize);
    cloud_full_.resize(plsize);

    std::vector<char> is_valid_pt(plsize, false);
    std::vector<uint> index(plsize - 1);
    for (uint i = 0; i < plsize - 1; ++i) {
        index[i] = i + 1;
    }

    double timebase = pl_orig.points[0].timestamp;

    faster_lio::compat::for_each(faster_lio::compat::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        if ((pl_orig.points[i].line < num_scans_) &&
            ((pl_orig.points[i].tag & 0x30) == 0x10 || (pl_orig.points[i].tag & 0x30) == 0x00)) {
            if (i % point_filter_num_ == 0) {
                cloud_full_[i].x = pl_orig.points[i].x;
                cloud_full_[i].y = pl_orig.points[i].y;
                cloud_full_[i].z = pl_orig.points[i].z;
                cloud_full_[i].intensity = pl_orig.points[i].intensity;
                cloud_full_[i].curvature =
                    static_cast<float>(pl_orig.points[i].timestamp - timebase) / static_cast<float>(1000000);

                if ((abs(cloud_full_[i].x - cloud_full_[i - 1].x) > 1e-7) ||
                    (abs(cloud_full_[i].y - cloud_full_[i - 1].y) > 1e-7) ||
                    (abs(cloud_full_[i].z - cloud_full_[i - 1].z) > 1e-7) &&
                        (cloud_full_[i].x * cloud_full_[i].x + cloud_full_[i].y * cloud_full_[i].y +
                             cloud_full_[i].z * cloud_full_[i].z >
                         (blind_ * blind_))) {
                    is_valid_pt[i] = true;
                }
            }
        }
    });

    for (uint i = 1; i < plsize; i++) {
        if (is_valid_pt[i]) {
            cloud_out_.points.push_back(cloud_full_[i]);
        }
    }
}

}  // namespace faster_lio
