import segmentation.mrcnn.model
from segmentation.config import SimpleConfig
from glob import glob
from segmentation.dataset import DetectorDataset

from utils.utils import class_id_to_name
import random

import cv2
import os
import numpy as np


def show_image_detection(model, image):
    results = model.detect([image])

    colors = [
        np.array([0, 0, 0]),        # background
        np.array([255, 0, 0]),      # bottle
        np.array([0, 255, 0]),      # bowl
        np.array([0, 0, 255]),      # camera
        np.array([128, 128, 0]),    # can
        np.array([0, 128, 128]),    # laptop
        np.array([128, 0, 128]),    # mug
        np.array([128, 128, 128])  # rubikscube
    ]
    alpha = 0.5

    masks = np.array(results[0]['masks']).astype(np.uint8)
    predicted_masks = np.copy(masks)
    final_overlay = np.zeros_like(image)

    class_ids = results[0]['class_ids']
    for i, class_id in enumerate(class_ids):
        roi = results[0]['rois'][i]
        score = results[0]['scores'][i]
        cv2.putText(image,
                    class_id_to_name(class_id) + ": " + "{0:1f}%".format(score),
                    (roi[1], roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        overlay_mask = np.where(masks[:, :, i] > 0, 1, 0)
        overlay_mask = overlay_mask.astype(np.uint8)
        overlay = np.stack(
            (overlay_mask * colors[class_id][0], overlay_mask * colors[class_id][1], overlay_mask * colors[class_id][2]),
            axis=-1)
        final_overlay += overlay

    image = cv2.addWeighted(image, 1 - alpha, final_overlay, alpha, 0)
    cv2.imshow("Segmentation", image)
    return predicted_masks, class_ids


def realsense_test(model):
    for image in realsense_images:
        image = cv2.imread(image)
        show_image_detection(model, image)
        cv2.waitKey()


def real275_test(model):
    random.shuffle(dataset_test.image_ids)
    for img_id in dataset_test.image_ids:
        image = dataset_test.load_image(img_id)
        show_image_detection(model, image)
        cv2.waitKey()


if __name__=='__main__':
    dataset_test = DetectorDataset(dataset="test")
    dataset_test.prepare()

    model_path = "segmentation/model"
    epoch = 30

    model = segmentation.mrcnn.model.MaskRCNN(mode="inference", config=SimpleConfig(), model_dir="model")
    model.load_weights(
        filepath=os.path.join(
            model_path,
            "nocs_lvis_baldhatsyn/mask_rcnn_nocs_lvis_baldhatsyn_segmentation_{0:04d}.h5".format(epoch)
        ),
        by_name=True
    )

    realsense_path = "data/realsense_data/test/scene_1/*_color.png"
    realsense_images = glob(realsense_path)
    # realsense_test(model)
    real275_test(model)