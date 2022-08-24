#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate and visualize 6d pose ground truth in Blender with Python API
Run Command:
    blender --background --python render_scenes.py
"""

import cv2
import bpy
import bpycv
from bpycv.camera_utils import *
import numpy as np
import os
import tqdm
import json
import _pickle as cPickle
from bpycv import HdriManager, get_cams
import logging

logging.getLogger().setLevel(logging.WARN)

data_dir = "D:/code/python/kubric"
OUTPUT_DIR = "rendered_models"
NUM_IMAGES = 3


def load_blend(obj):
    path = os.path.join(data_dir, obj['kwargs']['render_import_kwargs']['filepath'])
    bpy.ops.wm.append(
        filepath=path,
        directory=os.path.join(data_dir, obj['kwargs']['render_import_kwargs']['directory']),
        filename=obj['kwargs']['render_import_kwargs']['filename'],
        autoselect=True
    )


def load_obj(obj):
    path = os.path.join(data_dir, 'models', obj['id'], obj['kwargs']['render_filename'].split("/")[-1])
    bpy.ops.import_scene.obj(filepath=path,
                             use_split_objects=False)



def run():
    with open(os.path.join(data_dir, "model_manifest.json")) as f:
        manifest = json.load(f)

    # remove all MESH objects
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]

    hm = HdriManager(hdri_dir="E:/Programming/python/untitled/hdri")
    hdri_path = hm.sample()
    bpycv.load_hdri_world(hdri_path, random_rotate_z=True)
    cams = get_cams()
    cams[0].data.lens = 35

    bpy.context.scene.render.resolution_y = 640
    bpy.context.scene.render.resolution_x = 640

    objs = [manifest['assets'][asset_id] for asset_id in manifest['assets'].keys()]

    metadata = []

    for obj in tqdm.tqdm(objs):
        scene_metadata = {}
        bpycv.clear_all()
        matrix_intrinsics = get_cam_intrinsic(cams[0])
        scene_metadata['intrinsics'] = np.array([
            [matrix_intrinsics[0][0], 0, matrix_intrinsics[0][2]],
            [0, matrix_intrinsics[1][1], matrix_intrinsics[1][2]],
            [0, 0, matrix_intrinsics[2][2]]
        ])
        scene_metadata['images'] = []

        hdri_path = hm.sample()
        bpycv.load_hdri_world(hdri_path, random_rotate_z=True)
        scene_assets = [obj]

        for i_image in range(NUM_IMAGES):
            scene_objects = []
            obj = scene_assets[0]
            image_id = "{0}{1:04d}".format(obj['id'], i_image)
            bpycv.set_cam_pose(cam_radius=1, cam_deg=45, cam_x_deg=0)

            for i, asset in enumerate(scene_assets):
                if asset['kwargs']['render_filename'].split(".")[-1] == 'obj':
                    load_obj(asset)
                elif asset['kwargs']['render_filename'].split(".")[-1] == 'blend':
                    load_blend(asset)
                else:
                    raise RuntimeError("Unsupported file type")
                obj = bpy.context.selected_objects[0]
                obj["inst_id"] = i + 1
                obj['class_id'] = asset['metadata']['class_id']
                obj['scales'] = np.array(
                    [asset['metadata']['xscale'], asset['metadata']['yscale'], asset['metadata']['zscale']])
                obj['asset_id'] = asset['id']
                obj.location = (0, 0, 0)
                if i_image == 0:
                    obj.rotation_euler = (pi/2, 0, 1*pi/4)
                elif i_image == 1:
                    obj.rotation_euler = (pi / 2, 0, 3*pi/4)
                elif i_image == 2:
                    obj.rotation_euler = (pi / 2, pi, 3 * pi / 4)
                scene_objects.append(obj)

            result = bpycv.render_data()

            image_metadata, error = create_image_metadata(image_id, result, scene_objects)

            if not error:
                depth_scale = write_images(image_id, result)
                image_metadata['depth_scale'] = depth_scale
                scene_metadata['images'].append(image_metadata)
            [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]

        if len(scene_metadata['images']) > 0:
            metadata.append(scene_metadata)
            with open("{0}/metadata.pkl".format(OUTPUT_DIR), "wb") as f:
                cPickle.dump(metadata, f)


def scale_image(depth_image):
    max, min = np.max(depth_image), np.min(depth_image)
    if max - min != 0:
        depth = ((depth_image - min) / (max - min) * 65535).astype(np.uint16)
    else:
        depth = depth_image.astype(np.uint16)
    return depth, (min, max)


def write_images(image_id, result):
    cv2.imwrite("{1}/{0}_color.png".format(image_id, OUTPUT_DIR), cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("{1}/{0}_segmentation.png".format(image_id, OUTPUT_DIR), result['inst'])
    img, depth_scale = scale_image(result['depth'])
    cv2.imwrite("{1}/{0}_depth.png".format(image_id, OUTPUT_DIR), img)
    return depth_scale


def create_image_metadata(image_id, result, scene_objects):
    image_metadata = {}
    error = False
    image_metadata['image_id'] = image_id
    image_metadata['RT_world_cam'] = result['ycb_6d_pose']['cam_matrix_world']

    copy = np.copy(np.array([np.concatenate([rt, [[0, 0, 0, 1]]], axis=0) for rt in
                             result['ycb_6d_pose']['6ds']]))
    try:
        image_metadata['RT_cam_to_obj'] = [np.concatenate([rt, [[0, 0, 0, 1]]], axis=0) for rt in
                                           result['ycb_6d_pose']['6ds']]
        ## Reorder, because the order of the objects is not kept in the result array
        for i, inst_id in enumerate(result['ycb_6d_pose']['inst_ids']):
            image_metadata['RT_cam_to_obj'][inst_id - 1] = copy[i]
    except:
        print("Error!!!")
        error = True
    image_metadata['class_ids'] = [obj['class_id'] for obj in scene_objects]
    image_metadata['scales'] = [np.array([obj['scales'][0], obj['scales'][1], obj['scales'][2]]) for obj in
                                scene_objects]
    image_metadata['instance_ids'] = result['ycb_6d_pose']['inst_ids']
    image_metadata['asset_ids'] = [obj['asset_id'] for obj in scene_objects]
    return image_metadata, error


if __name__ == '__main__':
    run()
