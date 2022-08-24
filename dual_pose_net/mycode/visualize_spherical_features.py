import numpy as np
import cv2
import pptk
from utils.pc_utils import pc2sphericalmap
import open3d as o3d
import matplotlib.pyplot as plt

def backproject(depth, intrinsics, instance_mask):
    intrinsics_inv = np.linalg.inv(intrinsics)
    non_zero_mask = (depth > 0)
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]
    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]
    z = depth[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    pts[:, 0] = -pts[:, 0]
    pts[:, 1] = -pts[:, 1]

    return pts, idxs, final_instance_mask


if __name__ == '__main__':
    i = 26
    mask_img = cv2.imread("E:/data/realsense_data/test/scene_1/{0:04d}_mask.png".format(i), -1)
    depth = cv2.imread("E:/data/realsense_data/test/scene_1/{0:04d}_depth.png".format(i), -1)
    img = cv2.imread("E:/data/realsense_data/test/scene_1/{0:04d}_color.png".format(i))[:, :, ::-1]

    intrinsics = np.array([
        [599.479, 0., 325.033],
        [0., 599.497, 238.601],
        [0., 0., 1.]
    ])

    mask = np.where(mask_img > 0, True, False)
    mask = np.logical_and(mask, mask_img < 10)
    pts, idxs, final_instance_mask = backproject(depth, intrinsics, mask)
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid[np.newaxis, :]
    pts = pts/1000

    image = img[idxs[0], idxs[1], :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    cl, ind = pcd.remove_radius_outlier(nb_points=50, radius=0.1)
    pts = np.asarray(pcd.select_by_index(ind).points)
    image = image[ind]

    v = pptk.viewer(pts, image / 255.0)
    v.set(point_size=0.005)
    v.set(bg_color=(0, 0, 0, 1))
    v.set(floor_color=(0, 0, 0, 1))
    v.set(show_grid=False)

    dis_map, rgb_map = pc2sphericalmap(
        pts, image, resolution=64)

    dis_map = dis_map[0]
    rgb_map = rgb_map[0].astype(np.uint8)

    plt.imshow(rgb_map)
    plt.show()
    plt.imshow(dis_map)
    plt.show()


