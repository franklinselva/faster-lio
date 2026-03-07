#include <gtest/gtest.h>
#include "faster_lio/imu_processing.h"

using namespace faster_lio;

class ImuProcessEdgeTest : public ::testing::Test {
   protected:
    void SetUp() override {
        imu_proc_ = std::make_shared<ImuProcess>();
        imu_proc_->SetExtrinsic(common::Zero3d, common::Eye3d);
        imu_proc_->SetGyrCov(common::V3D(0.1, 0.1, 0.1));
        imu_proc_->SetAccCov(common::V3D(0.1, 0.1, 0.1));
        imu_proc_->SetGyrBiasCov(common::V3D(0.0001, 0.0001, 0.0001));
        imu_proc_->SetAccBiasCov(common::V3D(0.0001, 0.0001, 0.0001));

        // Initialize EKF
        std::vector<double> epsi(23, 0.001);
        kf_.init_dyn_share(
            get_f, df_dx, df_dw,
            [](state_ikfom &, esekfom::dyn_share_datastruct<double> &) {},
            4, epsi.data());
    }

    // Create a standard MeasureGroup with lidar and static IMU data
    common::MeasureGroup MakeStaticMeasure(int num_imu = 25, double dt = 0.005,
                                            int num_lidar_pts = 10,
                                            double lidar_start = 0.0,
                                            double lidar_end = 0.1) {
        common::MeasureGroup meas;
        meas.lidar_.reset(new PointCloudType());

        for (int i = 0; i < num_lidar_pts; i++) {
            PointType p;
            p.x = static_cast<float>(i) + 1.0f;
            p.y = 1.0f;
            p.z = 0.0f;
            p.curvature = static_cast<float>(i) * 10.0f;  // offset time
            meas.lidar_->push_back(p);
        }
        meas.lidar_bag_time_ = lidar_start;
        meas.lidar_end_time_ = lidar_end;

        for (int i = 0; i < num_imu; i++) {
            auto imu = std::make_shared<IMUData>();
            imu->timestamp = lidar_start + i * dt;
            imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
            imu->angular_velocity = Eigen::Vector3d::Zero();
            meas.imu_.push_back(imu);
        }
        return meas;
    }

    std::shared_ptr<ImuProcess> imu_proc_;
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;
};

TEST_F(ImuProcessEdgeTest, HighNoiseIMU) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;

    // IMU with very high noise
    for (int i = 0; i <= MAX_INI_COUNT + 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.005;
        // Large random-like noise on acc and gyr
        double noise = (i % 3 - 1) * 5.0;
        imu->linear_acceleration = Eigen::Vector3d(noise, noise * 0.5, 9.81 + noise);
        imu->angular_velocity = Eigen::Vector3d(noise * 0.1, noise * 0.2, noise * 0.05);
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    // Should not crash even with very noisy IMU
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, GravityMisalignment) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;

    // IMU mounted at 45 degrees - gravity split between y and z
    double g_component = 9.81 / std::sqrt(2.0);
    for (int i = 0; i <= MAX_INI_COUNT + 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.005;
        imu->linear_acceleration = Eigen::Vector3d(0.0, g_component, g_component);
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, HighAngularVelocity) {
    // First initialize with static data
    auto init_meas = MakeStaticMeasure(MAX_INI_COUNT + 5);
    PointCloudType::Ptr pcl_init(new PointCloudType());
    imu_proc_->Process(init_meas, kf_, pcl_init);

    // Now process with high angular velocity
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 0.2;
    meas.lidar_end_time_ = 0.3;

    for (int i = 0; i < 25; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = 0.2 + i * 0.005;
        imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu->angular_velocity = Eigen::Vector3d(5.0, 5.0, 5.0);  // very fast rotation
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);

    // Should produce undistorted points without NaN
    for (const auto& p : pcl_out->points) {
        EXPECT_FALSE(std::isnan(p.x));
        EXPECT_FALSE(std::isnan(p.y));
        EXPECT_FALSE(std::isnan(p.z));
    }
}

TEST_F(ImuProcessEdgeTest, TimestampGap) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 1.0;  // long frame

    // IMU with a large gap (0.1 to 0.9 missing)
    for (int i = 0; i < 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.02;
        imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }
    // Big jump
    for (int i = 0; i < 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = 0.9 + i * 0.02;
        imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, ResetAndReinitialize) {
    // First initialize
    auto meas = MakeStaticMeasure(MAX_INI_COUNT + 5);
    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);

    // Reset
    imu_proc_->Reset();

    // Re-initialize with new data
    auto meas2 = MakeStaticMeasure(MAX_INI_COUNT + 5, 0.005, 10, 1.0, 1.1);
    PointCloudType::Ptr pcl_out2(new PointCloudType());
    imu_proc_->Process(meas2, kf_, pcl_out2);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, CovarianceSetters) {
    common::V3D gyr_cov(0.5, 0.5, 0.5);
    common::V3D acc_cov(0.3, 0.3, 0.3);

    imu_proc_->SetGyrCov(gyr_cov);
    imu_proc_->SetAccCov(acc_cov);

    EXPECT_DOUBLE_EQ(imu_proc_->cov_gyr_scale_.x(), 0.5);
    EXPECT_DOUBLE_EQ(imu_proc_->cov_acc_scale_.x(), 0.3);

    common::V3D bias_gyr(0.001, 0.001, 0.001);
    common::V3D bias_acc(0.002, 0.002, 0.002);
    imu_proc_->SetGyrBiasCov(bias_gyr);
    imu_proc_->SetAccBiasCov(bias_acc);

    EXPECT_DOUBLE_EQ(imu_proc_->cov_bias_gyr_.x(), 0.001);
    EXPECT_DOUBLE_EQ(imu_proc_->cov_bias_acc_.x(), 0.002);
}

TEST_F(ImuProcessEdgeTest, ExtrinsicTransform) {
    // Set a non-trivial extrinsic
    common::V3D transl(0.1, 0.2, 0.3);
    common::M3D rot;
    rot << 0, -1, 0,
           1,  0, 0,
           0,  0, 1;

    imu_proc_->SetExtrinsic(transl, rot);

    // Initialize and process
    auto meas = MakeStaticMeasure(MAX_INI_COUNT + 5);
    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, SingleIMUSample) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    PointType p;
    p.x = 5.0f; p.y = 5.0f; p.z = 0.0f;
    p.curvature = 0.0f;
    meas.lidar_->push_back(p);
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;

    // Only one IMU sample
    auto imu = std::make_shared<IMUData>();
    imu->timestamp = 0.0;
    imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
    imu->angular_velocity = Eigen::Vector3d::Zero();
    meas.imu_.push_back(imu);

    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, ZeroDurationLidarFrame) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    for (int i = 0; i < 5; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = 0.0f;  // all at same time
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 1.0;
    meas.lidar_end_time_ = 1.0;  // zero duration

    for (int i = 0; i <= MAX_INI_COUNT + 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = 0.9 + i * 0.005;
        imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, ZeroGravityIMU) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());

    for (int i = 0; i < 10; i++) {
        PointType p;
        p.x = static_cast<float>(i) + 1.0f;
        p.y = 1.0f;
        p.z = 0.0f;
        p.curvature = static_cast<float>(i) * 10.0f;
        meas.lidar_->push_back(p);
    }
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;

    // Zero gravity (free fall or zero-g environment)
    for (int i = 0; i <= MAX_INI_COUNT + 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.005;
        imu->linear_acceleration = Eigen::Vector3d::Zero();
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    // Zero gravity is degenerate for init but should not crash
    imu_proc_->Process(meas, kf_, pcl_out);
    SUCCEED();
}

TEST_F(ImuProcessEdgeTest, EmptyLidarCloud) {
    common::MeasureGroup meas;
    meas.lidar_.reset(new PointCloudType());
    // No points in lidar
    meas.lidar_bag_time_ = 0.0;
    meas.lidar_end_time_ = 0.1;

    for (int i = 0; i <= MAX_INI_COUNT + 5; i++) {
        auto imu = std::make_shared<IMUData>();
        imu->timestamp = i * 0.005;
        imu->linear_acceleration = Eigen::Vector3d(0.0, 0.0, 9.81);
        imu->angular_velocity = Eigen::Vector3d::Zero();
        meas.imu_.push_back(imu);
    }

    PointCloudType::Ptr pcl_out(new PointCloudType());
    imu_proc_->Process(meas, kf_, pcl_out);
    EXPECT_EQ(pcl_out->size(), 0u);
}
