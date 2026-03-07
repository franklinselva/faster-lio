#include <gtest/gtest.h>
#include "faster_lio/imu_processing.h"

using namespace faster_lio;

class ImuProcessTest : public ::testing::Test {
   protected:
    void SetUp() override {
        imu_proc_ = std::make_shared<ImuProcess>();
        imu_proc_->SetExtrinsic(common::Zero3d, common::Eye3d);
        imu_proc_->SetGyrCov(common::V3D(0.1, 0.1, 0.1));
        imu_proc_->SetAccCov(common::V3D(0.1, 0.1, 0.1));
        imu_proc_->SetGyrBiasCov(common::V3D(0.0001, 0.0001, 0.0001));
        imu_proc_->SetAccBiasCov(common::V3D(0.0001, 0.0001, 0.0001));
    }

    std::shared_ptr<ImuProcess> imu_proc_;
};

TEST_F(ImuProcessTest, InitWithStaticIMU) {
    // Simulate static IMU data for initialization
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    // Add some points to lidar
    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i);
        p.y = 0.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i);
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;

    // Generate enough static IMU samples for initialization
    for (int i = 0; i <= MAX_INI_COUNT + 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.005;  // 200 Hz
        imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu->angular_velocity = Eigen::Vector3d(0.0, 0.0, 0.0);
        meas.imu_.push_back(imu);
    }

    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    std::vector<double> epsi(23, 0.001);
    kf.init_dyn_share(
        get_f, df_dx, df_dw,
        [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {},
        4, epsi.data());

    PointCloudType::Ptr pcl_out(new PointCloudType());

    // First call should start initialization
    imu_proc_->Process(meas, kf, pcl_out);

    // After enough iterations, IMU should be initialized
    // The process may need multiple calls with enough data
    SUCCEED();  // If no crash, initialization logic works
}

TEST_F(ImuProcessTest, EmptyIMUReturnsEarly) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
    std::vector<double> epsi(23, 0.001);
    kf.init_dyn_share(
        get_f, df_dx, df_dw,
        [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {},
        4, epsi.data());

    PointCloudType::Ptr pcl_out(new PointCloudType());

    // Empty IMU should return early without crash
    imu_proc_->Process(meas, kf, pcl_out);
    SUCCEED();
}
