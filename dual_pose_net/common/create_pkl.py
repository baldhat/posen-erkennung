import os
import numpy as np
import glob

img_list = []
dir = os.path.join("data", "realsense_data", "test")
folder_list = [name for name in sorted(os.listdir(dir)) if os.path.isdir(os.path.join(dir, name))]
for folder in folder_list:
    img_paths = glob.glob(os.path.join(dir, folder, '*_color.png'))
    img_paths = sorted(img_paths)
    for img_full_path in img_paths:
        img_name = os.path.basename(img_full_path)
        img_ind = img_name.split('_')[0]
        img_path = os.path.join("test", folder, img_ind)
        img_list.append(img_path)
with open(os.path.join("data", 'realsense_data', "test" + '_list_all.txt'), 'w') as f:
    for img_path in img_list:
        f.write("%s\n" % img_path)

realsense_test = open(os.path.join("data", 'realsense_data/test_list_all.txt')).read().splitlines()