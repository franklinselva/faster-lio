#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <pcl/filters/voxel_grid.h>
#include <mutex>

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
#include <fstream>
#include <memory>
#endif

#include <rigtorp/SPSCQueue.h>

#include "faster_lio/imu_processing.h"
#include "ivox3d/ivox3d.h"
#include "faster_lio/loop_closer.h"
#include "faster_lio/pointcloud_preprocess.h"
#include "faster_lio/pose_graph.h"
#include "faster_lio/wheel_fusion.h"

namespace faster_lio {

class LaserMapping {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
    using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
    using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
#endif

    LaserMapping();
    ~LaserMapping() {
        scan_down_body_ = nullptr;
        scan_undistort_ = nullptr;
        scan_down_world_ = nullptr;
        spdlog::debug("LaserMapping instance destroyed");
    }

    /// init from yaml config (the only init path)
    bool Init(const std::string &config_yaml);

    /// Returns true if Init() has been successfully called.
    bool IsInitialized() const { return initialized_; }

    /// True once the IMU initialization (gravity alignment, bias warm-up)
    /// has completed AND the IEKF has produced at least one post-init pose.
    /// Until this returns true, `GetCurrentPose()` returns an uninitialized
    /// pose that shouldn't be logged or published — doing so pins the
    /// visualization's base_link at the world origin for the first ~2 s of
    /// the run, which looks like a timestamp jump in the viewer.
    bool IsImuInitialized() const;

    /// Run one iteration of the mapping pipeline.
    /// Must be called from a single thread. NOT re-entrant.
    void Run();

    // input API — thread-safe, can be called from sensor callback threads
    void AddIMU(const IMUData &imu);
    void AddPointCloud(const PointCloudType::Ptr &cloud, double timestamp);
    /// Optional wheel-odometry observation. Caller fills a `WheelOdomData`
    /// with whichever body-frame velocity components its chassis can
    /// measure (see types.h). No-op unless `wheel.enabled: true` in yaml.
    void AddWheelOdom(const WheelOdomData &odom);

    // output API — thread-safe, can be called concurrently with Run()
    PoseStamped GetCurrentPose() const;
    Odometry GetCurrentOdometry() const;
    std::vector<PoseStamped> GetTrajectory() const;

    /// Access to the embedded pose graph for external consumers (loop-closure
    /// detectors, map deformers). Returns nullptr if the pose graph is disabled.
    PoseGraph *GetPoseGraph() { return pose_graph_.get(); }
    const PoseGraph *GetPoseGraph() const { return pose_graph_.get(); }

    /// Snapshot of the full IEKF state (pos, rot, vel, biases, gravity,
    /// extrinsic). Intended for test instrumentation — the live pipeline
    /// should use GetCurrentPose / GetCurrentOdometry instead.
    state_ikfom GetFilterState() const;

    /// Read-only view of the embedded ImuProcess. Intended purely for test
    /// instrumentation (e.g. verifying that YAML keys landed on the IMU
    /// preprocessor's configuration). Never null post-Init().
    const ImuProcess *GetImuProcess() const { return p_imu_.get(); }

    /// Get a copy of the latest undistorted scan (body frame).
    CloudPtr GetUndistortedCloud() const;

    /// Get a copy of the latest downsampled scan (world frame).
    CloudPtr GetDownsampledWorldCloud() const;

    /// Get a copy of the accumulated map cloud.
    CloudPtr GetMapCloud() const;

    // sync lidar with imu
    bool SyncPackages();

    /// interface of mtk, customized observation model
    void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);

    void Savetrajectory(const std::string &traj_file);

    /// Enable per-frame diagnostic CSV. Call once after Init().
    /// Writes one row per processed frame with feature counts, residual
    /// stats, timing, bias and extrinsic estimates. Empty path disables.
    ///
    /// Only functional when built with -DFASTER_LIO_ENABLE_DIAGNOSTICS=ON
    /// (default in non-Release builds). In Release it's a no-op with zero
    /// runtime cost.
    void EnableDiagnostics(const std::string &csv_path);

    void Finish();

   private:
    void SetPosestamp(PoseStamped &out);

    void PointBodyToWorld(PointType const *pi, PointType *const po);
    void PointBodyToWorld(const common::V3F &pi, PointType *const po);
    void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);

    void MapIncremental();

    bool LoadParamsFromYAML(const std::string &yaml);

    void PrintState(const state_ikfom &s);

    void SaveFrameWorld();

   private:
    /// modules
    IVoxType::Options ivox_options_;
    std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
    std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
    std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process

    /// local map related
    float det_range_ = 300.0f;
    double cube_len_ = 0;
    double filter_size_map_min_ = 0;
    bool localmap_initialized_ = false;

    /// params
    std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation
    std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
    std::string map_file_path_;

    /// point clouds data
    CloudPtr scan_undistort_ = std::make_shared<PointCloudType>();   // scan after undistortion
    CloudPtr scan_down_body_ = std::make_shared<PointCloudType>();   // downsampled scan in body
    CloudPtr scan_down_world_ = std::make_shared<PointCloudType>();  // downsampled scan in world
    std::vector<PointVector> nearest_points_;         // nearest points of current scan
    common::VV4F corr_pts_;                           // inlier pts
    common::VV4F corr_norm_;                          // inlier plane norms
    pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan
    std::vector<float> residuals_;                    // point-to-plane residuals
    std::vector<float> fit_quality_;                  // per-point plane fit RMS (lower = better)
    std::vector<char> point_selected_surf_;           // selected points
    common::VV4F plane_coef_;                         // plane coeffs

    /// synchronization
    mutable std::mutex mtx_state_;       // protects output state (pose, trajectory, clouds)
    bool initialized_ = false;           // set by Init(), checked by Run()

    /// Lock-free SPSC input buffers. Producer: sensor callback thread
    /// (AddIMU / AddPointCloud). Consumer: the thread that calls Run().
    /// Sized for worst-case lag: ~5s of IMU at 200Hz + ~10s of LiDAR at 10Hz.
    struct LidarItem {
        PointCloudType::Ptr cloud;
        double timestamp;
    };
    static constexpr size_t kImuQueueCapacity   = 1024;
    static constexpr size_t kLidarQueueCapacity = 128;
    static constexpr size_t kWheelQueueCapacity = 512;
    rigtorp::SPSCQueue<IMUData::Ptr>        imu_queue_{kImuQueueCapacity};
    rigtorp::SPSCQueue<LidarItem>           lidar_queue_{kLidarQueueCapacity};
    rigtorp::SPSCQueue<WheelOdomData::Ptr>  wheel_queue_{kWheelQueueCapacity};

    /// Consumer-side scratch deques. SyncPackages drains the SPSC queues
    /// into these at the start of each Run() tick, then applies the
    /// peek-without-pop sync logic against the deques (SPSC has no peek-all).
    std::deque<IMUData::Ptr>         imu_buffer_;
    std::deque<PointCloudType::Ptr>  lidar_buffer_;
    std::deque<double>               time_buffer_;
    std::deque<WheelOdomData::Ptr>   wheel_buffer_;

    /// options
    int num_max_iterations_ = 4;
    float esti_plane_threshold_ = 0.1f;
    float map_quality_threshold_ = 0.0f;  // 0 = disabled; points with fit_quality > this are excluded from map
    bool time_sync_en_ = false;
    double timediff_lidar_wrt_imu_ = 0.0;
    double last_timestamp_lidar_ = 0;
    double prev_lidar_bag_time_ = -1.0;  // previous scan time for inter-scan period estimation
    double lidar_end_time_ = 0;
    double last_timestamp_imu_ = -1.0;
    double first_lidar_time_ = 0.0;
    bool lidar_pushed_ = false;

    /// statistics and flags
    int scan_count_ = 0;
    int publish_count_ = 0;
    bool flg_first_scan_ = true;
    bool flg_EKF_inited_ = false;
    int pcd_index_ = 0;
    double lidar_mean_scantime_ = 0.0;
    int scan_num_ = 0;
    bool timediff_set_flg_ = false;
    int effect_feat_num_ = 0, frame_num_ = 0;

    /// Wheel-odometry fusion (optional; gated by yaml `wheel.enabled`).
    /// When enabled, a scalar body-frame velocity Kalman update is applied
    /// per present observation component AFTER the LiDAR IEKF update.
    bool   wheel_enabled_          = false;
    double wheel_cov_v_x_          = 0.01;
    double wheel_cov_v_y_          = 0.01;
    double wheel_cov_v_z_          = 0.001;
    double wheel_cov_omega_z_      = 0.01;
    bool   wheel_emit_nhc_v_x_     = false;   // default OFF: most robots have body-X = forward
    bool   wheel_emit_nhc_v_y_     = true;
    bool   wheel_emit_nhc_v_z_     = true;
    double wheel_nhc_cov_          = 0.001;
    double wheel_max_time_gap_     = 0.05;
    double last_timestamp_wheel_   = -1.0;

    // Accepts a body-frame scalar velocity observation on axis k∈{0,1,2}
    // and applies one Joseph-form Kalman update to kf_ (manual, manifold-aware).
    // The non-gated form passes 50.0 (≈7σ²) to the underlying helper.
    void ApplyBodyVelScalarUpdate(int axis, double z_body, double R_obs);
    // Variant for NHC pseudo-observations. The "observation" is exactly zero
    // by physics (a planar robot cannot have velocity on its vertical or
    // lateral body axis), so the Mahalanobis gate that's appropriate for
    // noisy real wheel samples only blocks the NHC from pulling a divergent
    // filter state back. Callers pass a very large gate_sq (or ~inf) to
    // accept every NHC emit.
    void ApplyBodyVelScalarUpdateGated(int axis, double z_body, double R_obs,
                                       double gate_sq);
    // Drains wheel_queue_ → wheel_buffer_; picks the sample closest to
    // `lidar_end_time` within `wheel_max_time_gap_`; applies scalar updates
    // for all present components plus NHC virtual observations.
    void DrainWheelQueue();
    WheelOdomData::Ptr FindWheelObs(double target_time) const;
    void ApplyWheelObservations(double lidar_end_time);

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    common::MeasureGroup measures_;                    // sync IMU and lidar scan
    esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
    state_ikfom state_point_;                          // ekf current state
    vect3 pos_lidar_;                                  // lidar position after eskf update
    common::V3D euler_cur_ = common::V3D::Zero();      // rotation in euler angles
    bool extrinsic_est_en_ = true;

    /////////////////////////  debug show / save /////////////////////////////////////////////////////////
    bool path_pub_en_ = true;
    bool dense_pub_en_ = false;
    bool pcd_save_en_ = false;
    bool runtime_pos_log_ = true;
    int pcd_save_interval_ = -1;
    bool path_save_en_ = false;

    PointCloudType::Ptr pcl_wait_save_ = std::make_shared<PointCloudType>();  // debug save
    std::vector<PoseStamped> path_;
    PoseStamped msg_body_pose_;

    /// Pose graph optimization (optional, gated by yaml `pose_graph.enabled`).
    bool   pose_graph_enabled_ = false;
    std::unique_ptr<PoseGraph> pose_graph_;
    PoseGraph::Options pg_opts_;
    Eigen::Isometry3d pg_correction_ = Eigen::Isometry3d::Identity();
    // Constant per-edge relative-measurement std for odometry edges. The
    // IEKF's marginal covariance grows monotonically and is not a correct
    // proxy for consecutive-keyframe relative uncertainty; a constant per
    // edge is the standard hygiene choice in pose-graph LIO stacks.
    double pg_odom_trans_std_ = 0.05;  // m per keyframe edge
    double pg_odom_rot_std_   = 0.02;  // rad per keyframe edge (~1.1 deg)
    void MaybeUpdatePoseGraph();

    /// Loop-closure detection (optional, gated by yaml `loop_closure.enabled`).
    /// Active iff pose_graph_enabled_ && loop_closer_ != nullptr. Hooks
    /// fire from inside Run() (Accumulate) and MaybeUpdatePoseGraph
    /// (AnchorKeyframe + DetectAtKeyframe + ApplyCorrection).
    bool loop_closure_enabled_ = false;
    std::unique_ptr<LoopCloser> loop_closer_;
    LoopCloser::Options loop_closer_opts_;
    int loop_closures_emitted_ = 0;

#ifdef FASTER_LIO_ENABLE_DIAGNOSTICS
    /// Diagnostics CSV (optional, enabled via EnableDiagnostics).
    std::unique_ptr<std::ofstream> diag_csv_;
    void WriteDiagnosticsRow();
    // Per-frame stage timings (microseconds) — populated inside Run()
    int64_t diag_t_undistort_us_{0};
    int64_t diag_t_downsample_us_{0};
    int64_t diag_t_iekf_us_{0};
    int64_t diag_t_mapinc_us_{0};
    int64_t diag_t_run_total_us_{0};
    // CPU-time accumulator for delta computation across rows
    int64_t diag_prev_cpu_us_{0};
#endif
};

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H
