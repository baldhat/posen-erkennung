import numpy as np
import cv2
import pyrealsense2 as rs
import segmentation
from segmentation.config import SimpleConfig
from segmentation.test_segmentation_model import show_image_detection
import os
from utils import configs
from model.dualposenet import DualPoseNet
import tensorflow as tf
from utils.evaluation_utils import transform_coordinates_3d

print(tf.__version__)

segmentation_model_epoch = 30
dpnet_epoch = 30
model_path = "segmentation/model"

FLAGS = configs.parse()
FLAGS.model = 'bsyn4'
FLAGS.test_epoch = dpnet_epoch
FLAGS.model_name = "model"
FLAGS.n_classes = 7
run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True

sess = tf.Session(config=run_config)
dpnet = DualPoseNet(FLAGS, sess)
dpnet.intrinsics = np.array([
    [599.479, 0., 325.033],
    [0., 599.497, 238.601],
    [0., 0., 1.]
])
dpnet.prepare_infer()

config = SimpleConfig()
config.DETECTION_MIN_CONFIDENCE = 0.9
model = segmentation.mrcnn.model.MaskRCNN(mode="inference", config=config, model_dir="model")
model.load_weights(
    filepath=os.path.join(
        model_path,
        "nocs_lvis_baldhatsyn/mask_rcnn_nocs_lvis_baldhatsyn_segmentation_{0:04d}.h5".format(
            segmentation_model_epoch)
    ),
    by_name=True
)

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline = rs.pipeline()

profile = pipeline.start(config)
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


def get_frames():
    # Get frameset of color and depth
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    return aligned_depth_frame, color_frame


def fetch_rgbd():
    aligned_depth_frame, color_frame = get_frames()
    depth_frame = temporal.process(aligned_depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth_image = depth_image * depth_scale
    return depth_image, color_image


def get_3d_bbox(scale):
    return np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                     [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                     [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                     [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                     [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                     [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                     [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                     [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]).transpose()


def draw_3d_bbox(img, points, color=(255, 0, 0)):
    points = np.int32(points.T).reshape(-1, 2)

    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(points[i]), tuple(points[j]), color_ground, 1)

    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(points[i]), tuple(points[j]), color_pillar, 1)

    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(points[i]), tuple(points[j]), color, 1)
    return img


def infer_poses(depth_image, color_image, masks, class_ids):
    pred_rts, pred_scales = dpnet.infer(depth_image * 1000, color_image, masks, class_ids)

    realsense_intrinsics = np.array([
        [599.479, 0., 325.033, 0],
        [0., 599.497, 238.601, 0],
        [0., 0., 1., 0]
    ])

    for i in range(len(class_ids)):
        x, y, w = realsense_intrinsics @ (pred_rts[i][:, 3])
        print(pred_rts[i][:, 3])
        if w == 0:
            continue

        x, y = x / w, y / w
        loc = (int(x), int(y))
        center = (loc[0], loc[1])
        color_image = cv2.rectangle(color_image, center, center, (255, 255, 0), 5)

        x_axis = realsense_intrinsics @ (pred_rts[i]) @ np.array([0.3, 0, 0, 1])
        x_axis = int(x_axis[0] / x_axis[2]), int(x_axis[1] / x_axis[2])
        y_axis = realsense_intrinsics @ (pred_rts[i]) @ np.array([0, 0.3, 0, 1])
        y_axis = int(y_axis[0] / y_axis[2]), int(y_axis[1] / y_axis[2])
        z_axis = realsense_intrinsics @ (pred_rts[i]) @ np.array([0, 0, 0.3, 1])
        z_axis = int(z_axis[0] / z_axis[2]), int(z_axis[1] / z_axis[2])

        color_image = cv2.line(color_image, center, x_axis, (0, 0, 255), 2)
        color_image = cv2.line(color_image, center, y_axis, (0, 255, 0), 2)
        color_image = cv2.line(color_image, center, z_axis, (255, 0, 0), 2)

        bbox = get_3d_bbox(pred_scales[i])
        bbox2d = realsense_intrinsics[:, 0:3] @ transform_coordinates_3d(bbox, pred_rts[i])
        bbox2d = bbox2d[:2, :] / bbox2d[2, :]

        color_image = draw_3d_bbox(color_image, bbox2d)

    cv2.imshow("image", color_image)


def run():
    while True:
        depth_image, color_image = fetch_rgbd()
        max, min = np.max(depth_image), np.min(depth_image)
        depth = ((depth_image - min) / (max - min) * 65535).astype(np.uint16)
        cv2.imshow("Depth image", depth)
        masks, class_ids = show_image_detection(model, np.copy(color_image))
        infer_poses(depth_image, color_image, masks, class_ids)
        cv2.waitKey(1)


if __name__ == '__main__':
    run()
