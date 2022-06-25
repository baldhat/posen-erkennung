import numpy as np
import cv2
import _pickle as cpickle

with open("output/metadata.txt", "rb") as f:
    metadata = cpickle.load(f)

RESOLUTION = metadata['resolution']
intrinsics = metadata['intrinsics']
intrinsics = intrinsics * RESOLUTION
intrinsics[2, 2] = intrinsics[2, 2] / RESOLUTION


def project_point(point3d, transform):
    """ Compute the image space coordinates [0, 1] for a given point in world coordinates."""
    homo_transform = np.linalg.inv(transform)
    homo_intrinsics = np.zeros((3, 4), dtype=np.float32)
    homo_intrinsics[:, :3] = intrinsics

    if point3d.shape[0] == 3:
        point4d = np.concatenate([point3d, [1.]])
    else:
        point4d = point3d
    projected = homo_intrinsics @ homo_transform @ point4d
    image_coords = projected / projected[2]
    return int(image_coords[0]), int(image_coords[1])


matrix_worlds = metadata['RT_world_cam']
obj_RTs = metadata['RT_cam_to_obj']


def draw_coordinate_axes(img, origin, xaxis, yaxis, zaxis):
    cv2.line(img, xaxis, origin, color=(0, 0, 255))
    cv2.putText(img, "X", xaxis, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    cv2.line(img, yaxis, origin, color=(0, 255, 0))
    cv2.putText(img, "Y", yaxis, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    cv2.line(img, zaxis, origin, color=(255, 0, 0))
    cv2.putText(img, "Z", zaxis, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))


def get_coordinate_axes(RT, length=0.2):
    origin = project_point(np.array([0, 0, 0, 1]), RT)
    x_axis = project_point(np.array([length, 0, 0, 1]), RT)
    y_axis = project_point(np.array([0, length, 0, 1]), RT)
    z_axis = project_point(np.array([0, 0, length, 1]), RT)
    return origin, x_axis, y_axis, z_axis


for i_scene in range(len(matrix_worlds)):
    for i_image in range(len(matrix_worlds[i_scene])):
        img = cv2.imread("output/scene{0:04d}/{1:04d}_color.png".format(i_scene, i_image))

        matrix_world = matrix_worlds[i_scene][i_image]
        obj_RT = obj_RTs[i_scene][i_image]

        orig, x, y, z = get_coordinate_axes(matrix_world)
        draw_coordinate_axes(img, orig, x, y, z)

        for i, obj_rt in enumerate(obj_RT):
            orig, x, y, z = get_coordinate_axes(np.linalg.inv(obj_rt))
            draw_coordinate_axes(img, orig, x, y, z)

        cv2.imshow("image", img)
        cv2.waitKey()
