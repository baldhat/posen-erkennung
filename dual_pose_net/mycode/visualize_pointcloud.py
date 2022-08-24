import numpy as np
import cv2
import pptk


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

    return pts, idxs


if __name__ == '__main__':
    mask_img = cv2.imread("E:/data/Real/test/scene_1/0000_mask.png", -1)
    depth = cv2.imread("E:/data/Real/test/scene_1/0000_depth.png", -1)

    print(mask_img.shape)
    print(depth.shape)

    # intrinsics = np.array([
    #     [387.22302246, 0, 322.73303223],
    #     [0, 387.22302246, 245.33900452],
    #     [0, 0, 1.]
    # ])
    intrinsics = np.array([
        [622.222229, 0, 320],
        [0, 622.222229, 320],
        [0, 0, 1.]
    ])

    mask = np.where(mask_img == 5, True, False)
    pts, idxs = backproject(depth, intrinsics, mask)
    print(pts.shape)
    v = pptk.viewer(pts)


