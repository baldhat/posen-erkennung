import os
import glob
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
from autolab_core import RigidTransform
import cmath
import cv2

ROOT_DIR = "E:"
SRC_DIR = "D:/code/python/kubric/output"
sym_id = [0,1,3]


def create_file_list(data_dir):
    for subset in ['train', 'test']:
        img_list = []
        img_dir = os.path.join(data_dir, subset)
        folder_list = [name for name in sorted(os.listdir(img_dir)) if os.path.isdir(os.path.join(img_dir, name))]
        for folder in folder_list:
            img_paths = glob.glob(os.path.join(img_dir, folder, '*_color.png'))
            img_paths = sorted(img_paths)
            for img_full_path in img_paths:
                img_name = os.path.basename(img_full_path)
                img_ind = img_name.split('_')[0]
                img_path = os.path.join(subset, folder, img_ind)
                img_list.append(img_path)
        with open(os.path.join(data_dir, subset, subset + '_list_all.txt'), 'w') as f:
            for img_path in img_list:
                f.write("%s\n" % img_path)
    print('Write all data paths to file done!')


def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    image_shape = depth.shape
    width = image_shape[1]
    height = image_shape[0]

    x = np.arange(width)
    y = np.arange(height)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs

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


def create_dpnet_pkl_train():
    baldhat_train = [line.strip() for line in open(os.path.join(SRC_DIR, 'train', 'train_list_all.txt')).readlines()]
    with open(os.path.join(SRC_DIR, 'train', 'metadata.txt'), 'rb') as f:
        metadata = cPickle.load(f)
        intrinsics = metadata['intrinsics']
        resolution = metadata['resolution']
        intrinsics = intrinsics * resolution
        intrinsics[2, 2] = intrinsics[2, 2] / resolution

    all_input_dis = []
    all_input_rgb = []
    all_observed_pc = []
    all_translation = []
    all_rotation = []
    all_scale = []

    for i, img_path in enumerate(tqdm(baldhat_train)):
        error = False
        i_scene = int(img_path.split("\\")[1][-4:])
        i_image = int(img_path.split("\\")[2])
        img_full_path = os.path.join(SRC_DIR, img_path)
        all_exist = os.path.exists(img_full_path + '_color.png') and \
                    os.path.exists(img_full_path + '_depth.png') and \
                    os.path.exists(img_full_path + '_segmentation.png')
        if not all_exist:
            continue

        image = cv2.imread(img_full_path + '_color.png')[:, :, :3]
        image = image[:, :, ::-1]
        depth_scale = metadata['depth_scale'][i_scene][i_image]
        depth = cv2.imread(img_full_path + "_depth.png", -1)
        depth = (depth / 65535) * (depth_scale['max'] - depth_scale['min']) + depth_scale['min']

        class_ids = np.array(metadata['class_ids'][i_scene][i_image])

        mask_img = cv2.imread(img_full_path + "_segmentation.png", -1)
        masks = np.zeros((resolution, resolution, len(class_ids)), dtype=np.bool)
        scales = np.array(metadata['scale'][i_scene][i_image])
        RTs = np.array(metadata['RT_cam_to_obj'][i_scene][i_image])
        rotations = RTs[:, :3, :3]
        translations = RTs[:, :3, 3]
        num_instances = len(class_ids)
        instance_pts = np.zeros((num_instances, 1024, 3))
        dis_smaps = np.zeros((num_instances, 64, 64, 1))
        rgb_smaps = np.zeros((num_instances, 64, 64, 3))
        for i in range(len(class_ids)):
            masks[:, :, i] = np.where(mask_img == i + 1, True, False)
            pts, idxs = backproject(depth, intrinsics, masks[:, :, i])
            pts = pts
            centroid = np.mean(pts, axis=0)
            pts = pts - centroid[np.newaxis, :]
            translations[i] = np.array([translations[i][0], translations[i][1], -translations[i][2]])
            translations[i] = translations[i] - centroid

            img = image[idxs[0], idxs[1], :]
            img = (img - np.array([123.7, 116.8, 103.9])[np.newaxis, :]) / 255.0
            dis_map, rgb_map = pc2sphericalmap(
                pts, img, resolution=64)

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
        for k in range(instance_pts.shape[0]):
            label = class_ids[k] - 1
            r = rotations[k]
            if label in sym_id:
                theta_x = r[0, 0] + r[2, 2]
                theta_y = r[0, 2] - r[2, 0]
                r_norm = np.sqrt(theta_x ** 2 + theta_y ** 2)
                s_map = np.array([[theta_x / r_norm, 0.0, -theta_y / r_norm],
                                  [0.0, 1.0, 0.0],
                                  [theta_y / r_norm, 0.0, theta_x / r_norm]])
                r = s_map @ r
            quaternions[k] = RigidTransform(rotation=r).quaternion
        if not error:
            all_input_dis.append(dis_smaps)
            all_input_rgb.append(rgb_smaps)
            all_observed_pc.append(instance_pts)
            all_translation.append(translations)
            all_rotation.append(quaternions)
            all_scale.append(scales)

    dataset = {}
    dataset['input_dis'] = np.concatenate(all_input_dis, axis=0).astype(np.float32)
    dataset['input_rgb'] = np.concatenate(all_input_rgb, axis=0).astype(np.float32)
    dataset['observed_pc'] = np.concatenate(all_observed_pc, axis=0).astype(np.float32)
    dataset['translation'] = np.concatenate(all_translation, axis=0).astype(np.float32)
    dataset['rotation'] = np.concatenate(all_rotation, axis=0).astype(np.float32)
    dataset['scale'] = np.concatenate(all_scale, axis=0).astype(np.float32)

    with open(os.path.join(ROOT_DIR, 'data', 'training_instance', 'baldhatsyn.pkl'), 'wb') as f:
        cPickle.dump(dataset, f)


if __name__ == '__main__':
    # create_file_list("output")
    create_dpnet_pkl_train()