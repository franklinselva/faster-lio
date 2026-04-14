#include "faster_lio/pointcloud_preprocess.h"

#include <spdlog/spdlog.h>

namespace faster_lio {

void PointCloudPreprocess::Process(const PointCloudType &cloud, PointCloudType::Ptr &pcl_out) {
    // Write directly into pcl_out — no member buffer, no trailing whole-cloud copy.
    pcl_out->clear();

    const int plsize = static_cast<int>(cloud.points.size());
    if (plsize == 0) {
        spdlog::warn("Preprocess: received empty point cloud, skipping");
        return;
    }

    pcl_out->points.reserve(plsize);

    const float blind_sq = static_cast<float>(blind_ * blind_);
    for (int i = 0; i < plsize; ++i) {
        if (i % point_filter_num_ != 0) continue;

        const auto &pt = cloud.points[i];
        const float d2 = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;

        // Combined zero-padding + blind-zone filter (both are short-range rejections)
        if (d2 < blind_sq || d2 < 1e-6f) continue;

        PointType added_pt;
        added_pt.x = pt.x;
        added_pt.y = pt.y;
        added_pt.z = pt.z;
        added_pt.intensity = pt.intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pt.curvature;  // per-point timing (ms), preserved if set

        pcl_out->points.push_back(added_pt);
    }

    pcl_out->width = static_cast<uint32_t>(pcl_out->points.size());
    pcl_out->height = 1;
    pcl_out->is_dense = true;
}

void PointCloudPreprocess::GenericHandler(const PointCloudType & /*cloud*/) {
    // Kept for ABI compatibility — Process now writes directly to its output.
    // If a caller still routes through here, it gets an empty result.
    cloud_out_.clear();
}

}  // namespace faster_lio
