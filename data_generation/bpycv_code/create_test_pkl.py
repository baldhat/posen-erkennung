import os
import numpy as np
import _pickle as cPickle
from tqdm import tqdm
import cv2
from utils import backproject
import pptk
import sys

sys.path.append("D:/code/python/DualPoseNet")
from segmentation.config import SimpleConfig
from segmentation.mrcnn.model import MaskRCNN

ROOT_DIR = "E:"
SRC_DIR = "E:/Programming/python/untitled/output"
sym_id = [0,1,3]
phase = "test"
DETECTION_GT = False


def create_dpnet_pkl_train():
    with open(os.path.join(SRC_DIR, phase, 'metadata.pkl'), 'rb') as f:
        metadata = cPickle.load(f)
        intrinsics = metadata[0]['intrinsics']
        resolution = 640

    viewer = pptk.viewer(np.array([[0, 0, 0]]))

    epoch = 30
    model_path = "D:/code/python/DualPoseNet/segmentation/model"
    model_name = "nocs_lvis_baldhatsyn/mask_rcnn_nocs_lvis_baldhatsyn_segmentation_{0:04d}.h5"
    file_path = model_path + "/" + model_name.format(epoch)

    model = MaskRCNN(mode="inference", config=SimpleConfig(), model_dir="model")
    model.load_weights(
        filepath=file_path,
        by_name=True
    )

    for i_scene, scene in enumerate(tqdm(metadata)):
        for i_image, image in enumerate(scene['images']):
            image_id = image['image_id']
            image_path = 'Programming/python/untitled/output/test/' + image_id
            error = False
            img_full_path = os.path.join(SRC_DIR, phase, image_id)
            all_exist = os.path.exists(img_full_path + '_color.png') and \
                        os.path.exists(img_full_path + '_depth.png') and \
                        os.path.exists(img_full_path + '_segmentation.png')
            if not all_exist:
                continue

            if metadata[i_scene]['images'][i_image]['image_id'] != image_id:
                raise RuntimeError("Expected metadata to have same image_id")
            depth_scale = metadata[i_scene]['images'][i_image]['depth_scale']
            depth = cv2.imread(img_full_path + "_depth.png", -1)
            depth = (depth / 65535) * (depth_scale[1] - depth_scale[0]) + depth_scale[0]

            class_ids = np.array(metadata[i_scene]['images'][i_image]['class_ids'])

            scales = np.array(metadata[i_scene]['images'][i_image]['scales'])
            RTs = np.array(metadata[i_scene]['images'][i_image]['RT_cam_to_obj'])

            mask_img = cv2.imread(img_full_path + "_segmentation.png", -1)
            masks = np.zeros((resolution, resolution, len(class_ids)), dtype=np.bool)
            bboxes = np.zeros((len(class_ids), 4)).astype(np.uint16)
            handle_visibilities = np.ones_like(class_ids)
            img = cv2.imread(img_full_path + '_color.png')
            if DETECTION_GT:
                for i, cl_id in enumerate(class_ids):
                    if len(class_ids) != RTs.shape[0]:
                        print("Error!")
                        error = True
                        continue

                    mask = np.where(mask_img == i + 1, True, False).astype(np.uint8) * 255
                    masks[:, :, i] = mask
                    contours, _ = cv2.findContours(mask.copy()[:, :, np.newaxis],
                                                              cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                              offset=(0, 0))
                    x, y, w, h = cv2.boundingRect(contours[0])
                    bboxes[i] = np.array([y, x, y+h, x+w]).astype(np.uint16)

                    if cl_id == 6:
                        annotate_handle_visibility(depth, handle_visibilities, i, intrinsics, mask, masks, viewer)
                pred_scores = np.ones_like(class_ids)
                pred_class_ids = class_ids
            else:
                if len(class_ids) != RTs.shape[0]:
                    print("Error!")
                    error = True
                    continue
                results = model.detect([img])[0]
                bboxes = results['rois']
                pred_class_ids = results['class_ids']
                masks = results['masks']
                pred_scores = results['scores']
                sister_path = 'E:/data/segmentation_results/bsyn/results_test_scene_{0}_{1:04d}.pkl'.format(i_scene, i_image)
                with open(sister_path, 'rb') as f:
                    ds = cPickle.load(f)
                    try:
                        handle_visibilities = ds['gt_handle_visibility']
                    except KeyError:
                        pass # Just assume visible

            dataset = {}
            dataset['pred_masks'] = masks
            dataset['gt_RTs'] = RTs
            dataset['gt_scales'] = scales
            dataset['pred_class_ids'] = pred_class_ids
            dataset['gt_class_ids'] = class_ids
            dataset['pred_scores'] = pred_scores
            dataset['pred_bboxes'] = bboxes
            dataset['image_path'] = image_path
            dataset['depth_scale'] = depth_scale
            dataset['gt_handle_visibility'] = handle_visibilities

            if not error:
                if DETECTION_GT:
                    path = 'E:/data/segmentation_results/bsyn/results_test_scene_{0}_{1:04d}.pkl'.format(i_scene, i_image)
                else:
                    path = 'E:/data/segmentation_results/bsyn_segmented/results_test_scene_{0}_{1:04d}.pkl'.format(i_scene, i_image)

                with open(path, 'wb') as f:
                    cPickle.dump(dataset, f)


def annotate_handle_visibility(depth, handle_visibilities, i, intrinsics, mask, masks, viewer):
    pts, idxs = backproject(depth, intrinsics, masks[:, :, i])
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid[np.newaxis, :]
    viewer.clear()
    viewer.load(pts)
    cv2.imshow("Object", mask)
    key = cv2.waitKey() & 0xFF
    if key == ord('n'):
        handle_visibilities[i] = 0
        print("Set: handle is invisible")


if __name__ == '__main__':
    create_dpnet_pkl_train()