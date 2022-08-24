import pptk
import numpy as np
import cv2
import _pickle as cPickle

from autolab_core import RigidTransform



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


def project_point(point3d, transform, intrinsics):
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


def transform_coordinates_3d(coordinates, RT):
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones(
        (1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates


def draw_coordinate_axes(img, origin, xaxis, yaxis, zaxis):
    cv2.line(img, xaxis, origin, color=(0, 0, 255))
    cv2.putText(img, "X", xaxis, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    cv2.line(img, yaxis, origin, color=(0, 255, 0))
    cv2.putText(img, "Y", yaxis, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0))
    cv2.line(img, zaxis, origin, color=(255, 0, 0))
    cv2.putText(img, "Z", zaxis, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))


def get_coordinate_axes(RT, intrinsics, length=0.2):
    origin = project_point(np.array([0, 0, 0, 1]), RT, intrinsics)
    x_axis = project_point(np.array([length, 0, 0, 1]), RT, intrinsics)
    y_axis = project_point(np.array([0, length, 0, 1]), RT, intrinsics)
    z_axis = project_point(np.array([0, 0, length, 1]), RT, intrinsics)
    return origin, x_axis, y_axis, z_axis


if __name__ == '__main__':

    # intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]]) # REAL275
    intrinsics = np.array([[622.0125, 0, 320.525], [0, 622.16775, 320.11084], [0, 0, 1]])  # baldhatsyn
    with open("D:/code/python/DualPoseNet/data/training_instance/baldhatsyn_bpycv_cl_id_1-6_p8.pkl", 'rb') as f:
        data = cPickle.load(f)

    # for key in data.keys():
    #     print(key)
    #     print(data[key].shape)
    #     print(data[key][0])

    # image = cv2.imread("E:\\data\\Real\\train\\scene_1\\0000_color.png")


    translations = data['translation'][0:200]
    rotations = data['rotation'][0:200]
    scales = data['scale'][0:200]
    try:
        class_ids = data['class_id'][0:200]
    except:
        class_ids = [0 for _ in range(20)]

    for i in range(200):
        image = cv2.imread("E:/Programming/python/untitled/output_cl_id_7/train\\0000{0:04d}_color.png".format(int(i/3)))

        RT = np.eye(4)
        RT[:3, :3] = RigidTransform.rotation_from_quaternion(rotations[i]).transpose()
        RT[:3, 3] = translations[i] + np.array([0, 0, 1.5])

        print(class_ids[i])
        print(RT)
        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1
        RT = z_180_RT @ RT
        scale = scales[i]
        print(scale)


        bbox = get_3d_bbox(scale)
        bbox2d = intrinsics[:, 0:3] @ transform_coordinates_3d(bbox, RT)
        bbox2d = bbox2d[:2, :] / bbox2d[2, :]
        axis_image = draw_3d_bbox(np.copy(image), bbox2d)

        orig, x, y, z = get_coordinate_axes(np.linalg.inv(RT), intrinsics)
        draw_coordinate_axes(axis_image, orig, x, y, z)

        cv2.imshow("View", axis_image)
        input_rgb = (data['input_rgb'][i][:, :, ::-1] * 255 + np.array([123.7, 116.8, 103.9])).astype(np.uint8)
        # input_rgb  = np.tile(input_rgb, (3, 3))
        cv2.imshow("input_rgb", input_rgb)
        v = pptk.viewer(data['observed_pc'][0:200][i])
        cv2.waitKey()
        v.close()
    #
    # v = pptk.viewer(data['observed_pc'][1] )

    # 0.018645726
    # -0.020197684
    # -0.0052067726
    # 0.24471317
    # 0.13396564