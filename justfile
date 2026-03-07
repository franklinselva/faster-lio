default:
    @just --list

# Configure and build the project
build:
    cmake -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# Run all tests
test: build
    cd build && ctest --output-on-failure

# Remove build directory and evaluation artifacts
clean:
    rm -rf build Log results PCD data
    rm -f scans.pcd Log/*.txt Log/*.log

# Run the mapping example (pass config file as argument)
example config:
    ./build/run_mapping --config {{config}}

# Install Python dependencies for bag extraction and evaluation
python-setup:
    uv pip install -r scripts/requirements.txt

# Extract IMU + LiDAR data from a ROS bag file
extract-bag bag output_dir *args:
    uv run scripts/extract_bag.py --bag {{bag}} --output_dir {{output_dir}} {{args}}

# Run LIO evaluation with ground truth comparison
evaluate config lidar_dir imu_file ground_truth num_scans *args:
    mkdir -p Log
    ./build/evaluate_lio \
        --config_file {{config}} \
        --lidar_dir {{lidar_dir}} \
        --imu_file {{imu_file}} \
        --ground_truth_file {{ground_truth}} \
        --num_scans {{num_scans}} \
        {{args}}

# Visualize trajectory comparison with evo
visualize estimated ground_truth *args:
    uv run scripts/evaluate_trajectory.py --estimated {{estimated}} --ground_truth {{ground_truth}} {{args}}
