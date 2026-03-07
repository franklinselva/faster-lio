default:
    @just --list

# Configure and build the project
build:
    cmake -B build -DBUILD_TESTS=ON -DBUILD_EXAMPLES=ON
    cmake --build build -j$(sysctl -n hw.ncpu 2>/dev/null || nproc)

# Run all tests
test: build
    cd build && ctest --output-on-failure

# Remove build directory
clean:
    rm -rf build

# Run the mapping example (pass config file as argument)
example config:
    ./build/run_mapping --config {{config}}
