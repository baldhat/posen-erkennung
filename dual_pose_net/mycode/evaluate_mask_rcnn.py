from segmentation.mrcnn.utils import compute_ap
from segmentation.dataset import  DetectorDataset
from segmentation.mrcnn.model import MaskRCNN
from segmentation.config import SimpleConfig
from segmentation.mrcnn.model import data_generator
import os
import numpy as np
import cv2
from tqdm import tqdm
from utils.utils import class_id_to_name


colors = [
    np.array([0, 0, 0]),  # background
    np.array([255, 0, 0]),  # bottle
    np.array([0, 255, 0]),  # bowl
    np.array([0, 0, 255]),  # camera
    np.array([128, 128, 0]),  # can
    np.array([0, 128, 128]),  # laptop
    np.array([128, 0, 128]),  # mugp
    np.array([128, 128, 128])  # rubikscube
]
alpha = 0.5

if __name__ == '__main__':
    dataset = DetectorDataset('test', include_baldhatsyn=False)
    dataset.prepare()

    epoch = 10
    model_path = "segmentation/model"
    model_name = "nocs_only/mask_rcnn_real275_segmentation_{0:04d}.h5"
    file_path = model_path + "/" + model_name.format(epoch)

    model = MaskRCNN(mode="inference", config=SimpleConfig(), model_dir="model")
    model.load_weights(
        filepath=file_path,
        by_name=True
    )

    test_generator = data_generator(dataset, model.config, shuffle=True)
    num_images = len(dataset.image_info)
    print("Evaluating", num_images, "images")
    maps = np.zeros((num_images))


    for j in tqdm(range(num_images)):
        inputs, _ = test_generator.__next__()
        batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes,\
            batch_gt_masks = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]

        results = model.detect(batch_images)[0]
        maps[j] = compute_ap(
                        batch_gt_boxes[0], batch_gt_class_ids[0], batch_gt_masks[0],
                    results['rois'], results['class_ids'], results['scores'], results['masks']
                  )[0]
        if j < 20:
            image = np.copy(inputs[0])
            masks = np.array(results['masks']).astype(np.uint8)
            predicted_masks = np.copy(masks)
            final_overlay = np.zeros_like(image)

            class_ids = results['class_ids']
            for i, class_id in enumerate(class_ids):
                roi = results['rois'][i]
                score = results['scores'][i]
                cv2.putText(image,
                            class_id_to_name(class_id) + ": " + "{0:1f}%".format(score),
                            (roi[1], roi[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                overlay_mask = np.where(masks[:, :, i] > 0, 1, 0)
                overlay_mask = overlay_mask.astype(np.uint8)
                overlay = np.stack(
                    (overlay_mask * colors[class_id][0], overlay_mask * colors[class_id][1],
                     overlay_mask * colors[class_id][2]),
                    axis=-1)
                final_overlay += overlay

            image = cv2.addWeighted(image, 1 - alpha, final_overlay, alpha, 0)
            path = "/".join(file_path.split("/")[:-1]) + "/result{0}.png".format(j)
            print(path)
            cv2.imwrite(path, image[0] * len(class_ids))
    mmap = np.mean(maps)
    print(mmap)

