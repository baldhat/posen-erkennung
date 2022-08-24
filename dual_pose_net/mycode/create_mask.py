import numpy as np
import cv2
import _pickle as cPickle

images = range(1, 101)

synset_names = ['BG',  # 0
                'bottle',  # 1
                'bowl',  # 2
                'camera',  # 3
                'can',  # 4
                'laptop',  # 5
                'mug',  # 6
                'rubikscube']  # 7

l515_color_intrinsics = np.array([
    [599.479, 0., 325.033],
    [0., 599.497, 238.601],
    [0., 0., 1.]
])

values = [38, 75, 10, 15, 53, 90, 128]


for i in images:
    depth_image = cv2.imread("E:/data/realsense_data/test/scene_1/{0:04d}_depth.png".format(i), -1)
    with open("E:/data/realsense_data/test/scene_1/{0:04d}_scale.pkl".format(i), 'rb') as f:
        scale = cPickle.load(f)

    depth = (depth_image / 65535) * (scale[1] - scale[0]) + scale[0]
    print(np.mean(depth))

    seg_image_path = "E:/data/realsense_data/test/scene_1/{0:04d}_color.bmp".format(i)
    image = cv2.imread(seg_image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(np.unique(grayscale))

    num_class_ids = len(np.unique(grayscale)) - 1


    mask_image = np.where(grayscale == 0, 255, grayscale)
    mask_image = np.where(mask_image == 38, 1, mask_image)
    mask_image = np.where(mask_image == 75, 2, mask_image)
    mask_image = np.where(mask_image == 15, 4, mask_image)
    mask_image = np.where(mask_image == 53, 5, mask_image)
    mask_image = np.where(mask_image == 90, 6, mask_image)
    mask_image = np.where(mask_image == 128, 7, mask_image)

    cv2.imwrite("E:/data/realsense_data/test/scene_1/{0:04d}_mask.png".format(i), mask_image)

    masks = np.zeros((mask_image.shape[0], mask_image.shape[1], len(np.unique(grayscale))-1)).astype(np.bool)

    j = 0
    class_ids = []
    for index, value in enumerate(values):
        if value in grayscale:
            masks[:, :, j] = np.where(grayscale == value, True, False)
            class_ids.append(index+1)
            j += 1

    data = {}
    data['image_path'] = 'data/realsense_data/test/scene_1/{0:04d}'.format(i)
    data['pred_class_ids'] = np.array(class_ids)
    data['gt_class_ids'] = np.array(class_ids)
    data['pred_masks'] = np.array(masks)
    data['depth_scale'] = (scale[0], scale[1])

    print(data['image_path'])

    print(data['pred_masks'].shape)
    print(class_ids)

    # depth_path = "E:/" + data['image_path'] + '_depth.png'
    # depth = cv2.imread(depth_path, -1)
    #
    #
    # for i in range(masks.shape[2]):
    #     pts, idx = backproject(depth, l515_color_intrinsics, masks[:, :, i])
    #     pts = pts/1000
    #     pts = pts - np.mean(pts, axis=0)
    #     v = pptk.viewer(pts)
    #     cv2.imshow("image", image)
    #     cv2.waitKey()

    with open("E:/data/segmentation_results/realsense_data/results_{0:04d}.pkl".format(i), "wb") as f:
        cPickle.dump(data, f)
