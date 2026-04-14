#include "faster_lio/pointcloud_preprocess.h"

#include <spdlog/spdlog.h>

namespace faster_lio {

void PointCloudPreprocess::Process(const PointCloudType &cloud, PointCloudType::Ptr &pcl_out) {
    GenericHandler(cloud);
    *pcl_out = cloud_out_;
}

void PointCloudPreprocess::GenericHandler(const PointCloudType &cloud) {
    cloud_out_.clear();

    const int plsize = static_cast<int>(cloud.points.size());
    if (plsize == 0) {
        spdlog::warn("GenericHandler: received empty point cloud, skipping");
        return;
    }

    cloud_out_.reserve(plsize);

    for (int i = 0; i < plsize; ++i) {
        if (i % point_filter_num_ != 0) continue;

        const auto &pt = cloud.points[i];
        const float d2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;

        // Filter zero-origin padding points
        if (d2 < 1e-6f) continue;

        // Filter blind zone
        if (d2 < blind_ * blind_) continue;

        PointType added_pt;
        added_pt.x = pt.x;
        added_pt.y = pt.y;
        added_pt.z = pt.z;
        added_pt.intensity = pt.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pt.curvature;  // per-point timing (ms), preserved if set

        cloud_out_.points.push_back(added_pt);
    }
}

}  // namespace faster_lio
