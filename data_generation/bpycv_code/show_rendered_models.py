import numpy as np
import cv2
import _pickle as cpickle

colors = [
    np.array([0, 0, 0]),       # background
    np.array([255, 0, 0]),     # bottle
    np.array([0, 255, 0]),     # bowl
    np.array([0, 0, 255]),     # camera
    np.array([128, 128, 0]),   # can
    np.array([0, 128, 128]),   # laptop
    np.array([128, 0, 128]),   # mug
    np.array([128, 128, 128])  # rubikscube
]
alpha = 0.5


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
    output_path = "rendered_models"
    with open("{0}/metadata.pkl".format(output_path), "rb") as f:
        metadata = cpickle.load(f)

    for scene in metadata:
        intrinsics = scene['intrinsics']
        for image in scene['images']:
            obj_RTs = image['RT_cam_to_obj']
            image_id = image['image_id']
            clear_img = cv2.imread("{0}/{1}_color.png".format(output_path, image_id), -1)
            cv2.putText(clear_img, str(image['asset_ids']), (30, 30),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255, 255, 255))
            axis_image = np.copy(clear_img)

            depth_image = cv2.imread("{0}/{1}_depth.png".format(output_path, image_id))

            mask = cv2.imread("{0}/{1}_segmentation.png".format(output_path, image_id))
            binary_masks = []

            final_overlay = np.zeros_like(clear_img).astype(np.uint8)
            for i in range(len(obj_RTs)):
                overlay_mask = np.where(mask == i + 1, 1, 0).astype(np.uint8)[:, :, 0]
                class_id = image['class_ids'][i]
                binary_masks.append(overlay_mask)
                overlay = np.stack(
                    (overlay_mask * colors[class_id][0], overlay_mask * colors[class_id][1],
                     overlay_mask * colors[class_id][2]),
                    axis=-1)
                final_overlay += overlay

                bbox = get_3d_bbox(image['scales'][i])
                bbox2d = intrinsics[:, 0:3] @ transform_coordinates_3d(bbox, obj_RTs[i])
                bbox2d = bbox2d[:2, :] / bbox2d[2, :]
                axis_image = draw_3d_bbox(axis_image, bbox2d)

            segmentation_image = cv2.addWeighted(clear_img, 1 - alpha, final_overlay, alpha, 0)

            for i, obj_rt in enumerate(obj_RTs):
                orig, x, y, z = get_coordinate_axes(np.linalg.inv(obj_rt), intrinsics)
                draw_coordinate_axes(axis_image, orig, x, y, z)

            depth_image = (depth_image * 255.0 / np.max(depth_image)).astype(np.uint8)
            top_image = np.hstack([clear_img, segmentation_image])
            bottom_image = np.hstack([axis_image, depth_image ])
            final_image = np.vstack((top_image, bottom_image) )

            cv2.imshow("Dataset View", final_image)
            cv2.waitKey()
