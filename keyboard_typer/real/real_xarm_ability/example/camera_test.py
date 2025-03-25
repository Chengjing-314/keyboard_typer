import numpy as np
import pyrealsense2 as rs


def get_pointcloud():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    # Start streaming/

    align_to = rs.stream.color
    align = rs.align(align_to)

    pipeline.start(config)
    threshold_filter = rs.threshold_filter(min_dist=0.15, max_dist=1.5)
    try:
        # Skip initial frames for auto-exposure adjustment
        for _ in range(5):
            pipeline.wait_for_frames()
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_frame = threshold_filter.process(depth_frame)

        # Check if frames are valid
        if not depth_frame or not color_frame:
            print("Failed to get frames.")
            return None, None
        # Create point cloud object and map it to the color frame
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        points = pc.calculate(depth_frame)
        # Convert to numpy arrays
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3)
        colors = colors[:, ::-1]  # BGR to RGB
        colors = colors / 255.0
        return vertices, colors
    finally:
        pipeline.stop()
        
        
get_pointcloud()