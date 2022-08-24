import warnings
warnings.filterwarnings("ignore")

import imgaug.augmenters as iaa
import segmentation.mrcnn.model
from segmentation.config import SimpleConfig
from segmentation.dataset import DetectorDataset

dataset_train = DetectorDataset(dataset="train")
dataset_train.prepare()
dataset_val = DetectorDataset(dataset="val")
dataset_val.prepare()

augmentation = iaa.Sequential([
    iaa.OneOf([
        iaa.Affine(rotate=0),
        iaa.Affine(rotate=90),
        iaa.Affine(rotate=180),
        iaa.Affine(rotate=270),
    ]),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.OneOf([
        iaa.Multiply((0.9, 1.1)),
        iaa.ContrastNormalization((0.9, 1.1)),
    ]),
    iaa.OneOf([
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.Sharpen(alpha=(0.0, 0.1)),
    ]),
])


LR_1 = 0.001

model = segmentation.mrcnn.model.MaskRCNN(mode="training", config=SimpleConfig(), model_dir="segmentation/model")

model.load_weights(filepath="segmentation/model/nocs_lvis_baldhatsyn/mask_rcnn_nocs_lvis_baldhatsyn_segmentation_0003.h5", by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"]
                   )

model.train(dataset_train, dataset_val,
            learning_rate=LR_1,
            epochs=30,
            layers='all',
            augmentation=augmentation)
