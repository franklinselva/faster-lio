# Faster-LIO

A standalone C++ library for lightweight, tightly coupled lidar-inertial odometry using parallel sparse incremental voxels (iVox). Forked from [gaoxiang12/faster-lio](https://github.com/gaoxiang12/faster-lio) and refactored into a pure C++ library with no ROS dependency.

Faster-LIO achieves 1k–2k Hz for solid-state lidars and over 200 Hz for 32-line spinning lidars, while preserving the same level of accuracy as FastLIO2. See the [paper](./doc/faster-lio.pdf) for details.

## Key Changes from Upstream

- **No ROS dependency** — pure C++ library with a clean `Init / AddIMU / AddPointCloud / GetCurrentPose` API
- **Multi-sensor support** — Livox AVIA, Velodyne, Ouster, Hesai XT32, RoboSense, Livox PointCloud2
- **Standalone build** — CMake-based, fetches googletest and spdlog automatically
- **Comprehensive tests** — 118 test cases covering core algorithms, edge cases, and production robustness

## Dependencies

- C++17 compiler (GCC 9+, Clang 10+, or Apple Clang)
- CMake 3.14+
- Eigen3
- PCL (common, io, filters)
- TBB
- yaml-cpp
- gflags (for examples only)

### Ubuntu / Debian

```bash
sudo apt-get install libeigen3-dev libpcl-dev libtbb-dev libyaml-cpp-dev libgflags-dev
```

### macOS (Homebrew)

```bash
brew install eigen pcl tbb yaml-cpp gflags
```

## Quick Start

A [justfile](https://github.com/casey/just) is provided for common tasks:

```bash
just build    # configure and build (with tests and examples)
just test     # build and run all tests
just clean    # remove the build directory
```

### Manual Build

```bash
cmake -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
```

To use PHC iVox instead of the default linear iVox:

```bash
cmake -B build -DWITH_IVOX_NODE_TYPE_PHC=ON -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
```

### Run Tests

```bash
cd build && ctest --output-on-failure
```

### Run Example

The `run_mapping` example reads PCD files and an IMU CSV offline:

```bash
just example ./config/default.yaml

# or directly:
./build/run_mapping \
    --config_file ./config/default.yaml \
    --pcd_dir /path/to/pcds \
    --imu_file /path/to/imu.csv \
    --num_pcd 100
```

## Library API

Include the umbrella header and link against `faster_lio`:

```cpp
#include <faster_lio/faster_lio.h>

faster_lio::LaserMapping mapping;
mapping.Init("config/default.yaml");

// Feed sensor data
mapping.AddIMU(imu_data);
mapping.AddPointCloud(cloud, timestamp);
mapping.Run();

// Query results
auto pose = mapping.GetCurrentPose();
auto odom = mapping.GetCurrentOdometry();
auto trajectory = mapping.GetTrajectory();
```

### CMake Integration

```cmake
add_subdirectory(faster-lio)
target_link_libraries(your_target PRIVATE faster_lio)
```

## Configuration

A single reference config lives at `config/default.yaml`. The algorithm is
sensor-agnostic — it operates on generic `PointXYZINormal` clouds, so there
are no per-LiDAR-brand knobs. Callers override parameters (extrinsics,
blind zone, voxel sizes) in their own config files as needed.

Key parameters:

- `ivox_grid_resolution` — voxel size (default 0.5m). Increase if tracking is unstable.
- `ivox_nearby_type` — neighbor search pattern: `6`, `18`, or `26`
- `max_iteration` — IEKF iterations (default 3–4)
- `filter_size_surf` / `filter_size_map` — downsampling leaf sizes

## Project Structure

```
├── include/
│   ├── faster_lio/       # public headers
│   │   ├── faster_lio.h  # umbrella header
│   │   ├── laser_mapping.h
│   │   ├── imu_processing.h
│   │   ├── pointcloud_preprocess.h
│   │   ├── types.h
│   │   └── ...
│   └── ivox3d/           # iVox spatial data structure
├── src/                  # library implementation
├── config/               # sensor YAML configs
├── examples/             # run_mapping offline example
├── tests/                # unit tests (118 cases)
├── doc/                  # paper PDF
├── CMakeLists.txt
└── justfile
```

## Known Issues

- iVox is sensitive to voxel size. Enlarge `ivox_grid_resolution` if tracking is unstable.
- `-march=native` may cause segfaults in PCL point cloud destruction on some platforms. See [this StackOverflow thread](https://stackoverflow.com/questions/61278204/segmentation-fault-when-deallocating-pclpointcloudpclpointxyzptr).
- macOS Apple Clang does not support `std::execution` parallel policies; a TBB-based fallback is used automatically via `compat.h`.

## Acknowledgements

This project is a fork of [gaoxiang12/faster-lio](https://github.com/gaoxiang12/faster-lio), the original open-source implementation of Faster-LIO. We gratefully acknowledge the work of the original authors:

> Chunge Bai, Tao Xiao, Yajie Chen, Haoqian Wang, Fang Zhang, and Xiang Gao.
> **"Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels."**
> _IEEE Robotics and Automation Letters_, vol. 7, no. 2, pp. 4861–4868, 2022.

We also thank the authors of [FastLIO2](https://github.com/hku-mars/FAST_LIO) and [IKFoM](https://github.com/hku-mars/IKFoM), upon which the upstream project was built. The original work was supported by Idriver+ Technologies Co. Ltd.

```bibtex
@ARTICLE{9718203,
  author={Bai, Chunge and Xiao, Tao and Chen, Yajie and Wang, Haoqian and Zhang, Fang and Gao, Xiang},
  journal={IEEE Robotics and Automation Letters},
  title={Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels},
  year={2022},
  volume={7},
  number={2},
  pages={4861-4868},
  doi={10.1109/LRA.2022.3152830}
}
```

## License

This project inherits the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) from the original Faster-LIO repository.
