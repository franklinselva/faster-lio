default:
    @just --list

alias b := build
# Configure and build the project
build:
    cmake -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

alias t := test
# Run all tests
test: build
    cd build && ctest --output-on-failure

alias c := clean
# Remove build directory and evaluation artifacts
clean:
    rm -rf build Log results PCD data
    rm -f scans.pcd Log/*.txt Log/*.log

alias ex := example
# Run the mapping example (pass config file as argument)
example config:
    ./build/run_mapping --config {{config}}

alias ps := python-setup
# Install Python dependencies for bag extraction and evaluation
python-setup:
    uv pip install -r scripts/requirements.txt

alias eb := extract-bag
# Extract IMU + LiDAR data from a ROS bag file
extract-bag bag output_dir *args:
    uv run scripts/extract_bag.py --bag {{bag}} --output_dir {{output_dir}} {{args}}

alias ev := evaluate
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

alias viz := visualize
# Visualize trajectory comparison with evo
visualize estimated ground_truth *args:
    uv run scripts/evaluate_trajectory.py --estimated {{estimated}} --ground_truth {{ground_truth}} {{args}}

alias m := metrics
# Plot performance and computation metrics from evaluation results
metrics *args:
    mkdir -p results
    .venv/bin/python3 scripts/plot_metrics.py --time_log ./Log/time.log --traj ./Log/traj.txt --gt ./Log/ground_truth_tum.txt {{args}}
