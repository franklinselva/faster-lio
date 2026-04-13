#!/usr/bin/env python3
"""Extract IMU and LiDAR data from a ROS1 bag file.

Supports both Livox CustomMsg and standard PointCloud2.

Outputs:
  - <output_dir>/imu.csv             — timestamp,ax,ay,az,gx,gy,gz
  - <output_dir>/livox/0.bin ...     — numbered Livox binary files (if CustomMsg)
  - <output_dir>/pcd/0.pcd ...       — numbered PCD files (if PointCloud2)
  - <output_dir>/metadata.txt        — topic info and message counts

Requires: pip install rosbags numpy
No ROS installation needed.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

try:
    from rosbags.rosbag1 import Reader as Rosbag1Reader
except ImportError:
    Rosbag1Reader = None

try:
    from rosbags.rosbag2 import Reader as Rosbag2Reader
except ImportError:
    Rosbag2Reader = None

from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.store import Nodetype

BASE = Nodetype.BASE
NAME = Nodetype.NAME
ARRAY = Nodetype.ARRAY
SEQUENCE = Nodetype.SEQUENCE


def register_livox_types(typestore):
    """Register livox_ros_driver/CustomMsg and CustomPoint with rosbags typestore.

    Covers both livox_ros_driver (v1, AVIA) and livox_ros_driver2 (v2, Mid360).
    """
    point_def = (
        [],
        [
            ("offset_time", (BASE, ("uint32", 0))),
            ("x", (BASE, ("float32", 0))),
            ("y", (BASE, ("float32", 0))),
            ("z", (BASE, ("float32", 0))),
            ("reflectivity", (BASE, ("uint8", 0))),
            ("tag", (BASE, ("uint8", 0))),
            ("line", (BASE, ("uint8", 0))),
        ],
    )
    msg_def = lambda point_type: (
        [],
        [
            ("header", (NAME, "std_msgs/msg/Header")),
            ("timebase", (BASE, ("uint64", 0))),
            ("point_num", (BASE, ("uint32", 0))),
            ("lidar_id", (BASE, ("uint8", 0))),
            ("rsvd", (ARRAY, ((BASE, ("uint8", 0)), 3))),
            ("points", (SEQUENCE, ((NAME, point_type), 0))),
        ],
    )
    typestore.register(
        {
            "livox_ros_driver/msg/CustomPoint": point_def,
            "livox_ros_driver/msg/CustomMsg": msg_def("livox_ros_driver/msg/CustomPoint"),
            "livox_ros_driver2/msg/CustomPoint": point_def,
            "livox_ros_driver2/msg/CustomMsg": msg_def("livox_ros_driver2/msg/CustomPoint"),
        }
    )


def detect_bag_format(bag_path: Path) -> str:
    if bag_path.is_dir() or bag_path.suffix in (".mcap", ".db3"):
        return "ros2"
    return "ros1"


def find_topics(reader) -> tuple[str | None, str | None, str]:
    """Auto-detect IMU and LiDAR topics. Returns (imu_topic, lidar_topic, lidar_format)."""
    imu_topic = None
    lidar_topic = None
    lidar_format = "pointcloud2"

    for conn in reader.connections:
        msgtype = conn.msgtype
        if "Imu" in msgtype and imu_topic is None:
            imu_topic = conn.topic
        if "CustomMsg" in msgtype and lidar_topic is None:
            lidar_topic = conn.topic
            lidar_format = "livox"
        if "PointCloud2" in msgtype and lidar_topic is None:
            lidar_topic = conn.topic
            lidar_format = "pointcloud2"

    return imu_topic, lidar_topic, lidar_format


def write_livox_bin(filepath: Path, msg, bag_timestamp_ns: int):
    """Write a Livox CustomMsg as a binary file matching the C++ LivoxCloud struct.

    Format:
      double   timebase      (8 bytes) — bag-level timestamp in seconds
      uint32   point_num     (4 bytes)
      uint8    lidar_id      (1 byte)
      For each point:
        float32  x, y, z     (12 bytes)
        uint8    reflectivity (1 byte)
        uint8    tag          (1 byte)
        uint8    line         (1 byte)
        uint32   offset_time  (4 bytes)
    """
    timebase = bag_timestamp_ns * 1e-9  # Use bag timestamp (Unix epoch)
    point_num = msg.point_num
    lidar_id = msg.lidar_id

    with open(filepath, "wb") as f:
        f.write(struct.pack("<d", timebase))
        f.write(struct.pack("<I", point_num))
        f.write(struct.pack("<B", lidar_id))

        for p in msg.points:
            f.write(struct.pack("<fff", p.x, p.y, p.z))
            f.write(struct.pack("<BBB", p.reflectivity, p.tag, p.line))
            f.write(struct.pack("<I", p.offset_time))


def parse_pointcloud2(msg) -> np.ndarray:
    """Extract XYZ + intensity from a PointCloud2 message into an Nx4 array."""
    field_names = [f.name for f in msg.fields]
    field_offsets = {f.name: f.offset for f in msg.fields}
    field_dtypes = {}
    for f in msg.fields:
        dt_map = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}
        field_dtypes[f.name] = dt_map.get(f.datatype, "<f4")

    point_step = msg.point_step
    n_points = msg.width * msg.height
    data = bytes(msg.data)

    has_xyz = all(f in field_names for f in ("x", "y", "z"))
    if not has_xyz:
        return np.zeros((0, 4), dtype=np.float32)

    points = np.zeros((n_points, 4), dtype=np.float32)
    for i in range(n_points):
        base = i * point_step
        for j, name in enumerate(("x", "y", "z")):
            offset = field_offsets[name]
            dt = field_dtypes[name]
            size = np.dtype(dt).itemsize
            val = np.frombuffer(data[base + offset : base + offset + size], dtype=dt)[0]
            points[i, j] = val

        for iname in ("intensity", "i", "reflectivity"):
            if iname in field_offsets:
                offset = field_offsets[iname]
                dt = field_dtypes[iname]
                size = np.dtype(dt).itemsize
                val = np.frombuffer(data[base + offset : base + offset + size], dtype=dt)[0]
                points[i, 3] = float(val)
                break

    valid = np.isfinite(points[:, :3]).all(axis=1)
    return points[valid]


def write_pcd(filepath: Path, points: np.ndarray):
    """Write points (Nx4: x,y,z,intensity) as ASCII PCD."""
    n = len(points)
    with open(filepath, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity normal_x normal_y normal_z curvature\n")
        f.write("SIZE 4 4 4 4 4 4 4 4\n")
        f.write("TYPE F F F F F F F F\n")
        f.write("COUNT 1 1 1 1 1 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {p[3]:.1f} 0 0 0 0\n")


def main():
    parser = argparse.ArgumentParser(description="Extract IMU + LiDAR from a ROS bag")
    parser.add_argument("--bag", required=True, help="Path to bag file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--imu_topic", default=None, help="IMU topic (auto-detected if not set)")
    parser.add_argument("--lidar_topic", default=None, help="LiDAR topic (auto-detected if not set)")
    parser.add_argument("--max_scans", type=int, default=0, help="Max scans to extract (0=all)")
    args = parser.parse_args()

    bag_path = Path(args.bag)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = detect_bag_format(bag_path)
    typestore = get_typestore(Stores.ROS1_NOETIC if fmt == "ros1" else Stores.ROS2_HUMBLE)
    register_livox_types(typestore)

    ReaderClass = Rosbag1Reader if fmt == "ros1" else Rosbag2Reader
    if ReaderClass is None:
        print(f"ERROR: rosbags reader for {fmt} not available.", file=sys.stderr)
        sys.exit(1)

    deserialize = typestore.deserialize_ros1 if fmt == "ros1" else typestore.deserialize_cdr

    print(f"Opening {fmt} bag: {bag_path}")
    with ReaderClass(bag_path) as reader:
        imu_topic = args.imu_topic
        lidar_topic = args.lidar_topic
        lidar_format = "pointcloud2"

        if imu_topic is None or lidar_topic is None:
            auto_imu, auto_lidar, auto_fmt = find_topics(reader)
            if imu_topic is None:
                imu_topic = auto_imu
            if lidar_topic is None:
                lidar_topic = auto_lidar
                lidar_format = auto_fmt

        # Determine format from connection msgtype if topic was specified manually
        if args.lidar_topic:
            for c in reader.connections:
                if c.topic == lidar_topic:
                    lidar_format = "livox" if "CustomMsg" in c.msgtype else "pointcloud2"
                    break

        print(f"IMU topic:    {imu_topic}")
        print(f"LiDAR topic:  {lidar_topic} ({lidar_format})")

        if imu_topic is None and lidar_topic is None:
            print("ERROR: No IMU or LiDAR topics found", file=sys.stderr)
            sys.exit(1)

        # Set up output directory for lidar data
        if lidar_format == "livox":
            lidar_dir = out_dir / "livox"
        else:
            lidar_dir = out_dir / "pcd"
        lidar_dir.mkdir(parents=True, exist_ok=True)

        # Build connection lookup
        imu_conns = {c.id for c in reader.connections if c.topic == imu_topic}
        lidar_conns = {c.id for c in reader.connections if c.topic == lidar_topic}
        imu_msgtype = None
        lidar_msgtype = None
        for c in reader.connections:
            if c.topic == imu_topic:
                imu_msgtype = c.msgtype
            if c.topic == lidar_topic:
                lidar_msgtype = c.msgtype

        imu_count = 0
        scan_count = 0
        imu_file = out_dir / "imu.csv"

        with open(imu_file, "w") as imu_f:
            imu_f.write("timestamp,ax,ay,az,gx,gy,gz\n")

            for conn, timestamp, rawdata in reader.messages():
                if conn.id in imu_conns and imu_topic:
                    msg = deserialize(rawdata, imu_msgtype)
                    # Use bag-level timestamp (Unix epoch) for cross-sensor alignment
                    t = timestamp * 1e-9
                    la = msg.linear_acceleration
                    av = msg.angular_velocity
                    imu_f.write(f"{t:.9f},{la.x},{la.y},{la.z},{av.x},{av.y},{av.z}\n")
                    imu_count += 1

                elif conn.id in lidar_conns and lidar_topic:
                    if args.max_scans > 0 and scan_count >= args.max_scans:
                        continue
                    msg = deserialize(rawdata, lidar_msgtype)

                    if lidar_format == "livox":
                        if msg.point_num > 0:
                            write_livox_bin(lidar_dir / f"{scan_count}.bin", msg, timestamp)
                            scan_count += 1
                    else:
                        points = parse_pointcloud2(msg)
                        if len(points) > 0:
                            write_pcd(lidar_dir / f"{scan_count}.pcd", points)
                            scan_count += 1

                    if scan_count % 100 == 0 and scan_count > 0:
                        print(f"  Extracted {scan_count} scans, {imu_count} IMU messages...")

        # Write metadata
        meta_file = out_dir / "metadata.txt"
        with open(meta_file, "w") as f:
            f.write(f"bag: {bag_path}\n")
            f.write(f"format: {fmt}\n")
            f.write(f"lidar_format: {lidar_format}\n")
            f.write(f"imu_topic: {imu_topic}\n")
            f.write(f"lidar_topic: {lidar_topic}\n")
            f.write(f"imu_messages: {imu_count}\n")
            f.write(f"lidar_scans: {scan_count}\n")

        print(f"\nDone! Extracted {imu_count} IMU messages and {scan_count} LiDAR scans")
        print(f"LiDAR format: {lidar_format}")
        print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
