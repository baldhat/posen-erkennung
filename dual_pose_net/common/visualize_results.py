import _pickle as cPickle
import numpy as np
import cv2
import os
from utils.evaluation_utils import transform_coordinates_3d

epoch = 30
phase = 'test'
model = 'bsyn4'
dataset = 'bsyn_segmented'
pkl_path = "..\\log\\{2}\\{1}_epoch{0}\\{3}\\".format(epoch, phase, model, dataset)

synset_names = ['BG',  # 0
                'bottle',  # 1
                'bowl',  # 2
                'camera',  # 3
                'can',  # 4
                'laptop',  # 5
                'mug',  # 6
                'rubikscube']  # 7

if dataset == 'bsyn' or dataset == 'bsyn_segmented':
    intrinsics = np.array([  # bsyn
        [622.222229, 0, 320, 0],
        [0, 622.222229, 320, 0],
        [0, 0, 1., 0]
    ])
elif dataset == 'REAL275':
    intrinsics = np.array([[591.0125, 0, 322.525, 0],
                           [0, 590.16775, 244.11084, 0],
                           [0, 0, 1, 0]]) # REAL275
elif dataset == 'realsense_data':
    # d435_color_intrinsics = np.array([   # d435
    #     [387.22302246, 0, 322.73303223, 0],
    #     [0, 387.22302246, 245.33900452, 0],
    #     [0, 0, 1., 0]
    # ])
    intrinsics = np.array([  # L515
        [599.479, 0., 325.033, 0],
        [0., 599.497, 238.601, 0],
        [0., 0., 1., 0]
    ])
elif dataset == 'CAMERA25':
    intrinsics = np.array([[577.5, 0, 319.5, 0],
                           [0, 577.5, 239.5, 0],
                           [0, 0, 1, 0]])



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


def visualize_result(path, suffix):
    with open(path, 'rb') as f:
        result = cPickle.load(f)
        image_path = result['image_path']
        classes = result['pred_class_ids']
        pred_rts = result['pred_RTs']
        pred_scales = result['pred_scales']
        bboxes = result['pred_bboxes']

        image = cv2.imread(os.path.join("E:/", image_path + '_color.png'))[:, :, :3]


        for i in range(len(classes)):
            x, y, w = intrinsics @ (pred_rts[i][:, 3])
            if np.any(w == 0):
                print("w == 0")
                continue
            x, y = x / w, y / w
            loc = (int(x), int(y))
            center = (loc[0], loc[1])
            image = cv2.rectangle(image, center, center, (255, 255, 0), 5)

            x_axis = intrinsics @ (pred_rts[i]) @ np.array([0.3, 0, 0, 1])
            x_axis = int(x_axis[0] / x_axis[2]), int(x_axis[1] / x_axis[2])
            y_axis = intrinsics @ (pred_rts[i]) @ np.array([0, 0.3, 0, 1])
            y_axis = int(y_axis[0] / y_axis[2]), int(y_axis[1] / y_axis[2])
            z_axis = intrinsics @ (pred_rts[i]) @ np.array([0, 0, 0.3, 1])
            z_axis = int(z_axis[0] / z_axis[2]), int(z_axis[1] / z_axis[2])

            image = cv2.line(image, center, x_axis, (0, 0, 255), 2)
            image = cv2.line(image, center, y_axis, (0, 255, 0), 2)
            image = cv2.line(image, center, z_axis, (255, 0, 0), 2)

            bbox = get_3d_bbox(pred_scales[i])
            bbox2d = intrinsics[:, 0:3] @ transform_coordinates_3d(bbox, pred_rts[i])
            bbox2d = bbox2d[:2, :] / bbox2d[2, :]

            image = draw_3d_bbox(image, bbox2d)
            cv2.rectangle(image, (bboxes[i][1], bboxes[i][0]), (bboxes[i][3], bboxes[i][2]), (255, 255, 255), 1)

        # cv2.imwrite('..\\' + image_path + '_' + suffix + '.png', image)
        cv2.imshow("image", image)
        cv2.waitKey()


if __name__=='__main__':
    for pkl in os.listdir(pkl_path):
        full_path = os.path.join(pkl_path, pkl)
        visualize_result(full_path, "result_bsyn")

        # full_path_refine = os.path.join(real_pkl_path_refine, pkl)
        # camera_full_path = os.path.join(camera_realsense_pkl_path, pkl)
        # camera_full_path_refine = os.path.join(camera_pkl_path_refine, pkl)

        # visualize_result(full_path_refine, "refined_result_real275")
        # visualize_result(camera_full_path, "result_camera25")
        # visualize_result(camera_full_path_refine, "refined_result_camera25")
