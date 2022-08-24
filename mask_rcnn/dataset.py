import tensorflow as tf
import os
import numpy as np
from cv2 import imread
import random
from sklearn.model_selection import train_test_split
from segmentation.mrcnn import utils
from lvis import LVIS
import _pickle as cpickle
import keras.backend as K
cfg = K.tf.ConfigProto(gpu_options={'allow_growth': True})
K.set_session(K.tf.Session(config=cfg))


class DetectorDataset(utils.Dataset):
    """Dataset class for training our dataset.
    """

    def __init__(self, dataset, include_lvis=True, include_baldhatsyn=True):
        super().__init__(self)
        random.seed(1234)

        real_275_path = "E:/data/Real"
        lvis_path = "E:/data/lvis"
        baldhatsyn_path = "D:/code/python/kubric_code/output"
        self.WIDTH = 640
        self.HEIGHT = 480
        self.dataset = dataset

        if dataset in ['train', 'test']:
            with open(os.path.join(baldhatsyn_path, dataset, "metadata.txt"), 'rb') as f:
                self.metadata = cpickle.load(f)
                self.baldhatsyn_metadata = []
                for scene_level in self.metadata['class_ids']:
                    for image_level in scene_level:
                        self.baldhatsyn_metadata.append(image_level)

        self.lvis_id_to_id = {
            133: 1,
            139: 2,
            189: 3,
            192: 4,
            631: 5,
            344: 6
        }

        self.lv_train = LVIS(os.path.join(lvis_path, 'lvis_v1_train.json'))
        self.lv_val = LVIS(os.path.join(lvis_path, 'lvis_v1_val.json'))

        self.baldhatsyn_list_train = os.path.join(baldhatsyn_path, "train", "train_list_all.txt")
        self.baldhatsyn_list_test = os.path.join(baldhatsyn_path, "test", "test_list_all.txt")

        image_paths_lvis = []
        image_paths_real275 = []
        image_paths_baldhatsyn = []

        if dataset == "train":
            with open("E:data/Real/train_list_all.txt", "r") as f:
                image_paths_real275 = [os.path.join(real_275_path, img.strip()) for img in f.readlines()]
            for lvis_id in self.lvis_id_to_id.keys():
                image_paths_lvis.extend(set([os.path.join(lvis_path, 'train2017', '{0:012d}.jpg'.format(img_id))
                                    for img_id in self.lv_train.cat_img_map[lvis_id]
                                    if os.path.exists(os.path.join(lvis_path, 'train2017', '{0:012d}.jpg'.format(img_id)))]))
            with open(self.baldhatsyn_list_train, "r") as f:
                image_paths_baldhatsyn = [os.path.join(baldhatsyn_path, img.strip()) for img in f.readlines()]
        elif dataset == 'val':
            for lvis_id in self.lvis_id_to_id.keys():
                image_paths_lvis.extend(set([os.path.join(lvis_path, 'val2017', '{0:012d}.jpg'.format(img_id))
                                    for img_id in self.lv_val.cat_img_map[lvis_id]
                                    if os.path.exists(os.path.join(lvis_path, 'val2017', '{0:012d}.jpg'.format(img_id)))]))
        elif dataset == "test":
            with open("E:data/Real/test_list_all.txt", "r") as f:
                image_paths_real275 = random.choices([os.path.join(real_275_path, img.strip()) for img in f.readlines()], k=200)
            with open(self.baldhatsyn_list_test, "r") as f:
                image_paths_baldhatsyn = [os.path.join(baldhatsyn_path, img.strip()) for img in f.readlines()]
        else:
            raise RuntimeError("Unknown phase:", dataset)

        # Add classes
        self.add_class('joined', 1, 'bottle')
        self.add_class('joined', 2, 'bowl')
        self.add_class('joined', 3, 'camera')
        self.add_class('joined', 4, 'can')
        self.add_class('joined', 5, 'laptop')
        self.add_class('joined', 6, 'mug')
        self.add_class('joined', 7, 'rubikscube')

        # add images
        for i, file_path in enumerate(image_paths_real275):
            self.add_image('joined', i, file_path + "_color.png", set='real275')
        offset = len(image_paths_real275)
        if include_lvis:
            for i, file_path in enumerate(image_paths_lvis):
                self.add_image('joined', offset+i, file_path, set='lvis')
            offset2 = offset + len(image_paths_lvis)
        else:
            offset2 = offset
        if include_baldhatsyn:
            for i, file_path in enumerate(image_paths_baldhatsyn):
                self.add_image("joined", offset2+i, file_path + "_color.png", set='baldhatsyn', img_id=i)

        print(len(image_paths_real275))
        print(len(image_paths_lvis))
        print(len(image_paths_baldhatsyn))
        print("Num images:", len(self.image_info))

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        image = imread(fp)
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]

        if info['set'] == 'lvis':
            return self.load_mask_lvis(info)
        elif info['set'] == 'real275':
            return self.load_mask_275(info)
        elif info['set'] == 'baldhatsyn':
            return self.load_mask_baldhatsyn(info)
        else:
            raise RuntimeError("Imag with id", image_id, "has no dataset info")

    def load_mask_lvis(self, info):
        if self.dataset == "train":
            lv = self.lv_train
        elif self.dataset == 'val':
            lv = self.lv_val
        else:
            raise RuntimeError("Unknown phase")

        anns = lv.get_ann_ids(img_ids=[int(info['path'].split('.')[0].split('\\')[-1])])

        masks = []
        class_ids = []
        for ann in anns:
            ann = lv.anns[ann]
            if ann['category_id'] in self.lvis_id_to_id.keys():
                masks.append(lv.ann_to_mask(ann))
                class_ids.append(self.lvis_id_to_id[ann['category_id']])
        if len(masks) > 1:
            masks = np.stack((mask for mask in masks), axis=-1)
        else:
            masks = np.array(masks)
            if masks.shape[0] < 100:
                masks = masks.reshape((masks.shape[1], masks.shape[2], masks.shape[0]))
        return masks, np.array(class_ids)

    def load_mask_275(self, info):
        path = "/".join(os.path.split(info['path'])[:-1])
        meta_file = os.path.split(info['path'])[-1].split("_")[0] + "_meta.txt"
        with open(os.path.join(path, meta_file), "r") as f:
            class_info = [line.split(" ") for line in f.readlines() if len(line) > 3]
        count = len(class_info)
        mask_file = os.path.split(info['path'])[-1].split("_")[0] + "_mask.png"
        mask_img = imread(os.path.join(path, mask_file), -1)
        mask = np.zeros((self.HEIGHT, self.WIDTH, count), dtype=np.bool)
        class_ids = np.zeros((count,), dtype=np.int32)
        for i in range(count):
            class_ids[i] = class_info[i][1]
            mask[:, :, i] = np.where(mask_img == i + 1, True, False)
        return mask, class_ids

    def load_mask_baldhatsyn(self, info):
        path = "/".join(os.path.split(info['path'])[:-1])
        class_ids = self.baldhatsyn_metadata[info['img_id']]
        resolution = self.metadata['resolution']

        mask_file = os.path.split(info['path'])[-1].split("_")[0] + "_segmentation.png"
        mask_img = imread(os.path.join(path, mask_file))[:, :, 0]
        mask = np.zeros((resolution, resolution, len(class_ids)), dtype=np.bool)
        for i in range(len(class_ids)):
            mask[:, :, i] = np.where(mask_img == i + 1, True, False)
        return mask, np.array(class_ids)
