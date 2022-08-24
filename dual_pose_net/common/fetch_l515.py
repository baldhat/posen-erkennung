import numpy as np
import cv2
import pyrealsense2 as rs
import os
import _pickle as cPickle

path = "E:/data/realsense_data/test/scene_1"
current_index = max([int(i[0:4]) for i in os.listdir(path)])
print(current_index)

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline = rs.pipeline()

# D435
d435_color_intrinsics = np.array([
    [384.59814453, 0., 321.45782471],
    [0., 384.23257446, 243.75810242],
    [0., 0., 1.]
])

# L515
l515_color_intrinsics = np.array([
    [599.479, 0., 325.033],
    [0., 599.497, 238.601],
    [0., 0., 1.]
])



profile = pipeline.start(config)

print(profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics)
# device = profile.get_device()
# depth_sensor = device.query_sensors()[0]
# laser_range = depth_sensor.get_option_range(rs.option.laser_power)
# depth_sensor.set_option(rs.option.laser_power, laser_range.max)

colorizer = rs.colorizer()
spatial = rs.spatial_filter()
decimation = rs.decimation_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

align_to = rs.stream.color
align = rs.align(align_to)

print(rs.intrinsics.fx)
print(rs.intrinsics.fy)
print(rs.intrinsics.ppx)
print(rs.intrinsics.ppy)

def get_frames():
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    return aligned_depth_frame, color_frame


while True:
    aligned_depth_frame, color_frame = get_frames()

    depth_frame = temporal.process(aligned_depth_frame)
    depth_frame = spatial.process(depth_frame)
    #depth_frame = hole_filling.process(depth_frame)

    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    # depth_image = cv2.resize(np.asanyarray(depth_frame.get_data()), dsize=(640, 480))
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_image = depth_image * depth_scale
    max, min = np.max(depth_image), np.min(depth_image)
    depth = ((depth_image - min) / (max - min) * 65535).astype(np.uint16)

    cv2.imshow("Depth View", colorized_depth)
    cv2.imshow("Color View", color_image)
    key = cv2.waitKey(30) & 0xFF
    if key == ord(' '):
        current_index += 1
        cv2.imwrite(os.path.join(path, "{0:04d}_color.png".format(current_index)), color_image)
        cv2.imwrite(os.path.join(path, "{0:04d}_depth.png".format(current_index)),depth)
        with open(os.path.join(path, "{0:04d}_scale.pkl".format(current_index)), 'wb') as f:
            cPickle.dump(np.array([min, max]), f)
    elif key == ord('q'):
        break

pipeline.stop()