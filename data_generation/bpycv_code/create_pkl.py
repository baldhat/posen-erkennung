import os
import glob
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from autolab_core import RigidTransform
import cmath
from visualize_dataset import get_coordinate_axes, get_3d_bbox, draw_coordinate_axes, draw_3d_bbox, transform_coordinates_3d
import cv2
from utils import backproject

ROOT_DIR = "E:"
SRC_DIR = "E:\\data\\rendered_scenes\\output_cl_id_1-6_p12"
output_path = 'D:/code/python/DualPoseNet/data/training_instance/baldhatsyn_bpycv_cl_id_1-6_p12.pkl'
sym_id = [0,1,3]
phase = "train"


def pc2sphericalmap(pc, img, resolution=64):
    n = pc.shape[0]
    assert pc.shape[1] == 3

    pc = pc.astype(np.float32)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    t = np.pi / resolution
    k = 2 * np.pi / resolution
    r = np.sqrt(np.sum(pc ** 2, axis=1) + 1e-10).astype(np.float32)

    phi = np.around(np.arccos(z / r) / t).astype('int') % resolution
    arr = np.arctan2(y, x)
    rho = np.zeros(n)
    rho[y > 0] = np.around(arr[y > 0] / k)
    rho[y < 0] = np.around((arr[y < 0] + 2 * np.pi) / k)
    rho = rho.astype('int') % resolution
    f1 = np.zeros([resolution, resolution, 1], dtype='float32')
    f2 = np.zeros([resolution, resolution, 3], dtype='float32')

    for i in range(pc.shape[0]):
        tmp = np.real(cmath.rect(r[i], 0))
        if f1[rho[i], phi[i], 0] <= tmp:
            f1[rho[i], phi[i], 0] = tmp
            f2[rho[i], phi[i], :] = img[i]

    return f1[np.newaxis, :, :, :], f2[np.newaxis, :, :, :]


def visualize(r, t, image, intrinsics, scale):
    RT = np.eye(4)
    RT[:3, :3] = r.transpose()
    RT[:3, 3] = t
    z_180_RT = np.zeros((4, 4), dtype=np.float32)
    z_180_RT[:3, :3] = np.diag([-1, -1, 1])
    z_180_RT[3, 3] = 1
    RT = z_180_RT @ RT

    bbox = get_3d_bbox(scale)
    bbox2d = intrinsics[:, :3] @ transform_coordinates_3d(bbox, RT)
    bbox2d = bbox2d[:2, :] / bbox2d[2, :]
    axis_image = draw_3d_bbox(np.copy(image), bbox2d)

    orig, x, y, z = get_coordinate_axes(np.linalg.inv(RT), intrinsics)
    draw_coordinate_axes(axis_image, orig, x, y, z)

    cv2.imshow("View", axis_image)
    cv2.waitKey()

def set_zero(map, num=64):
    idxs = np.random.randint(0, 64, size=(2, num))
    map[0, idxs[0, :], idxs[1, :], 0] = np.zeros(num)
    return map


def create_dpnet_pkl_train():
    with open(os.path.join(SRC_DIR, phase, 'metadata.pkl'), 'rb') as f:
        metadata = cPickle.load(f)
        intrinsics = metadata[0]['intrinsics']
        resolution = 640

    all_input_dis = []
    all_input_rgb = []
    all_observed_pc = []
    all_translation = []
    all_rotation = []
    all_scale = []
    all_class_id = []

    for i_scene, scene in enumerate(tqdm(metadata)):
        for i_image, image in enumerate(scene['images']):
            image_id = image['image_id']
            error = False
            img_full_path = os.path.join(SRC_DIR, phase, image_id)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_segmentation.png')
            if not all_exist:
                continue

            image = cv2.imread(img_full_path + '_color.png', -1)[:, :, :3]
            image = image[:, :, ::-1]

            if metadata[i_scene]['images'][i_image]['image_id'] != image_id:
                raise RuntimeError("Expected metadata to have same image_id")
            depth_scale = metadata[i_scene]['images'][i_image]['depth_scale']
            depth = cv2.imread(img_full_path + "_depth.png", -1)
            depth = (depth / 65535) * (depth_scale[1] - depth_scale[0]) + depth_scale[0]

            class_ids = np.array(metadata[i_scene]['images'][i_image]['class_ids'])

            mask_img = cv2.imread(img_full_path + "_segmentation.png", -1)
            masks = np.zeros((resolution, resolution, len(class_ids)), dtype=np.bool)
            scales = np.array(metadata[i_scene]['images'][i_image]['scales'])
            RTs = np.array(metadata[i_scene]['images'][i_image]['RT_cam_to_obj'])

            for i in range(RTs.shape[0]):
                z_180_RT = np.zeros((4, 4), dtype=np.float32)
                z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                z_180_RT[3, 3] = 1
                RTs[i] = z_180_RT @ RTs[i]
            rotations = RTs[:, :3, :3]
            translations = RTs[:, :3, 3]
            num_instances = len(class_ids)
            instance_pts = np.zeros((num_instances, 1024, 3))
            dis_smaps = np.zeros((num_instances, 64, 64, 1))
            rgb_smaps = np.zeros((num_instances, 64, 64, 3))
            centroids = []
            for i in range(len(class_ids)):
                if len(class_ids) != translations.shape[0]:
                    print("Error!")
                    error = True
                    break
                masks[:, :, i] = np.where(mask_img == i + 1, True, False)
                pts, idxs = backproject(depth, intrinsics, masks[:, :, i])
                centroid = np.mean(pts, axis=0)
                centroids.append(centroid)
                pts = pts - centroid[np.newaxis, :]

                translations[i] = translations[i] - centroid

                img = image[idxs[0], idxs[1], :]
                img = (img - np.array([123.7, 116.8, 103.9])[np.newaxis, :]) / 255.0
                dis_map, rgb_map = pc2sphericalmap(
                    pts, img, resolution=64)

                dis_map = set_zero(dis_map, num=128)
                rgb_map = set_zero(rgb_map, num=128)

                if pts.shape[0] > 1024:
                    pts = pts[np.random.choice(
                        pts.shape[0], 1024, replace=False), :]
                elif pts.shape[0] <= 0:
                    error = True
                    break
                else:
                    pts = pts[np.random.choice(
                        pts.shape[0], 1024), :]

                instance_pts[i, :, :] = pts
                dis_smaps[i, :, :, :] = dis_map[0]
                rgb_smaps[i, :, :, :] = rgb_map[0]

            quaternions = np.zeros((rotations.shape[0], 4))
            if not error:
                for k in range(instance_pts.shape[0]):
                    label = class_ids[k] - 1
                    r = rotations[k]
                    r = r.transpose()
                    # visualize(r, translations[k] + centroids[k], image, intrinsics, scales[k])
                    if label in sym_id:
                        theta_x = r[0, 0] + r[2, 2]
                        theta_y = r[0, 2] - r[2, 0]
                        theta_hat = np.arctan2(theta_y, theta_x)
                        s_map = np.array([
                            [np.cos(theta_hat), 0, -np.sin(theta_hat)],
                            [0, 1, 0],
                            [np.sin(theta_hat), 0, np.cos(theta_hat)]
                        ])
                        r = s_map @ r
                    # visualize(r, translations[k] + centroids[k], image, intrinsics, scales[k])
                    try:
                        quaternions[k] = RigidTransform(rotation=r).quaternion
                    except ValueError as ve:
                        error = True
                        print("Error: Invalid rotation")

                if not error:
                    all_input_dis.append(dis_smaps)
                    all_input_rgb.append(rgb_smaps)
                    all_observed_pc.append(instance_pts)
                    all_translation.append(translations)
                    all_rotation.append(quaternions)
                    all_scale.append(scales)
                    all_class_id.append(class_ids)

    dataset = {}
    dataset['input_dis'] = np.concatenate(all_input_dis, axis=0).astype(np.float32)
    dataset['input_rgb'] = np.concatenate(all_input_rgb, axis=0).astype(np.float32)
    dataset['observed_pc'] = np.concatenate(all_observed_pc, axis=0).astype(np.float32)
    dataset['translation'] = np.concatenate(all_translation, axis=0).astype(np.float32)
    dataset['rotation'] = np.concatenate(all_rotation, axis=0).astype(np.float32)
    dataset['scale'] = np.concatenate(all_scale, axis=0).astype(np.float32)
    dataset['class_id'] = np.concatenate(all_class_id, axis=0).astype(np.float32)

    with open(output_path, 'wb') as f:
        cPickle.dump(dataset, f)


if __name__ == '__main__':
    create_dpnet_pkl_train()