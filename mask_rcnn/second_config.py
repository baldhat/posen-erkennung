from segmentation.mrcnn.config import Config


class SecondConfig(Config):
    NUM_CLASSES = 8  # 7 without rubikscube
    NAME = "nocs_lvis_baldhatsyn_segmentation"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    BACKBONE = "resnet50"

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    MAX_GT_INSTANCES = 7
    DETECTION_MAX_INSTANCES = 7
    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640
    USE_MINI_MASK = False