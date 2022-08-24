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
import random
import numpy as np
import os
import tqdm
import json
import _pickle as cPickle
from bpycv import HdriManager, get_cams
import logging

logging.getLogger().setLevel(logging.WARN)

i = 12
data_dir = "D:/code/python/kubric"
OUTPUT_DIR = "E:\\data\\rendered_scenes\\output_cl_id_1-6_p{0}".format(i)
NUM_SCENES = 3000
NUM_IMAGES = 5
NUM_ASSETS_PER_IMAGE = 3
PHASE = 'train'
use_class_ids = [1, 2, 3, 4, 5, 6]


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


def randomize_material(obj):
    for mat_slot in obj.material_slots:
        try:
            mat = mat_slot.material
            mat.metallic = np.random.uniform(0.5, 2) * mat.metallic
            mat.roughness = np.random.uniform(0.5, 2) * mat.roughness
            mat.specular_intensity = np.random.uniform(0.5, 2) * mat.specular_intensity
            mat.diffuse_color = np.random.uniform(0.5, 2, size=4) * mat.diffuse_color
            mat.specular_color = np.random.uniform(0.5, 2, size=3) * mat.specular_color
            try:
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs["Metallic"].default_value = random.random()
                bsdf.inputs["Specular"].default_value = random.random()
                bsdf.inputs["Specular Tint"].default_value = random.random()
                bsdf.inputs["Roughness"].default_value = random.uniform(0.3, 1)
                bsdf.inputs["Anisotropic"].default_value = random.random()
                bsdf.inputs["Anisotropic Rotation"].default_value = random.random()
                bsdf.inputs["Sheen"].default_value = random.random()
                bsdf.inputs["Sheen Tint"].default_value = random.random()
                bsdf.inputs["Clearcoat"].default_value = random.random()
                bsdf.inputs["Clearcoat Roughness"].default_value = random.random()
                bsdf.inputs["Alpha"].default_value = random.uniform(0.9, 1)
                bsdf.inputs["Base Color"].default_value = bsdf.inputs["Base Color"].default_value \
                                                          * np.random.uniform(0.8, 1.2, size=(4))
                bsdf.inputs["Subsurface Color"].default_value = bsdf.inputs["Subsurface Color"].default_value \
                                                                * np.random.uniform(0.8, 1.2, size=(4))
                bsdf.inputs["Emission"].default_value = bsdf.inputs["Emission"].default_value \
                                                        * np.random.uniform(0.8, 1.2, size=(4))
            except:
                pass
            try:
                bsdf = mat.node_tree.nodes["Glossy BSDF"]
                bsdf.inputs["Color"].default_value = bsdf.inputs["Color"].default_value \
                                                     * np.random.uniform(0.8, 1.2, size=(4))
                bsdf.inputs["Roughness"].default_value = bsdf.inputs["Roughness"].default_value \
                                                         * np.random.uniform(0.5, 2)
            except:
                pass
            try:
                texs = [n for n in mat.node_tree.nodes if n.type == 'TEX_IMAGE']
                for tex in texs:
                    tex.color_mapping.brightness = tex.color_mapping.brightness * np.random.uniform(0.7, 1.5)
                    tex.color_mapping.contrast = tex.color_mapping.contrast * np.random.uniform(0.7, 1.5)
                    tex.color_mapping.saturation = tex.color_mapping.saturation * np.random.uniform(0.7, 1.5)
            except:
                pass
        except:
            pass


def randomize_image(result):
    image = result['image']

    float_image = image / 255.0  # image must be float!!
    mean = np.mean(float_image, axis=tuple(range(float_image.ndim - 1)))  # compute mean values over each channel
    image = (float_image - mean) * np.random.uniform(0.7, 1.3) + mean  # change contrast
    image = np.clip(image, a_min=0., a_max=1.)  # cut values under 0 and over 1
    image *= 255.0
    image = image.astype(np.uint8)

    image = np.where(image > 255, 255, image)
    result['image'] = np.where(image < 0, 0, image).astype(np.uint8)

    depth = result['depth']
    depth = depth * np.random.normal(1, 0.005, size=depth.shape)
    result['depth'] = np.where(depth < 0, 0, depth)


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

    objs = [manifest['assets'][asset_id] for asset_id in manifest['assets'].keys()
            if manifest['assets'][asset_id]['metadata']['type'] == PHASE
            and manifest['assets'][asset_id]['metadata']['class_id'] in use_class_ids]

    metadata = []

    for i_scene in tqdm.tqdm(range(NUM_SCENES)):
        scene_metadata = {}
        matrix_intrinsics = get_cam_intrinsic(cams[0])
        scene_metadata['intrinsics'] = np.array([
            [matrix_intrinsics[0][0], 0, matrix_intrinsics[0][2]],
            [0, matrix_intrinsics[1][1], matrix_intrinsics[1][2]],
            [0, 0, matrix_intrinsics[2][2]]
        ])
        scene_metadata['images'] = []

        hdri_path = hm.sample()
        bpycv.load_hdri_world(hdri_path, random_rotate_z=True)
        scene_assets = []
        for inst_id in range(NUM_ASSETS_PER_IMAGE):
            scene_assets.append(random.choice(objs))

        scene_objects = []
        for i_image in range(NUM_IMAGES):
            image_id = "{0:04d}{1:04d}".format(i_scene, i_image)
            bpycv.set_cam_pose(cam_radius=np.random.randint(2, 3), cam_deg=np.random.randint(-45, 45))

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
                # scale_x_z = np.random.uniform(0.7, 1.3)
                # scale_y = np.random.uniform(0.7, 1.3)
                # if obj['class_id'] == 7:
                #     scale_y = scale_x_z
                obj['sizes'] = np.array(
                    [asset['metadata']['xscale'],
                     asset['metadata']['yscale'],
                     asset['metadata']['zscale']]
                )
                # obj.scale = (scale_x_z, scale_y, scale_x_z)
                obj['asset_id'] = asset['id']
                randomize_material(obj)
                scene_objects.append(obj)
                obj.location = np.random.uniform(-0.6, 0.6, size=3)
                obj.rotation_euler = np.random.uniform(0, 2 * pi, size=3)

            result = bpycv.render_data()
            randomize_image(result)

            image_metadata, error = create_image_metadata(image_id, result, scene_objects)
            if not error:
                depth_scale = write_images(image_id, result)
                image_metadata['depth_scale'] = depth_scale
                scene_metadata['images'].append(image_metadata)
            [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
            scene_objects = []

        metadata.append(scene_metadata)
        bpycv.clear_all()
        if i_scene % 50 == 0:
            path = "{1}/{0}/metadata.pkl".format(PHASE, OUTPUT_DIR)
            with open(path, "wb") as f:
                cPickle.dump(metadata, f)


def scale_image(depth_image):
    max, min = np.max(depth_image), np.min(depth_image)
    depth = ((depth_image - min) / (max - min) * 65535).astype(np.uint16)
    return depth, (min, max)


def write_images(image_id, result):
    cv2.imwrite("{2}/{0}/{1}_color.png".format(PHASE, image_id, OUTPUT_DIR),
                cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))
    cv2.imwrite("{2}/{0}/{1}_segmentation.png".format(PHASE, image_id, OUTPUT_DIR), result['inst'])
    img, depth_scale = scale_image(result['depth'])
    cv2.imwrite("{2}/{0}/{1}_depth.png".format(PHASE, image_id, OUTPUT_DIR), img)
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
        error = True
    image_metadata['class_ids'] = [obj['class_id'] for obj in scene_objects]
    image_metadata['scales'] = [np.array([obj['sizes'][0],
                                          obj['sizes'][1],
                                          obj['sizes'][2]]) for obj in scene_objects]
    image_metadata['instance_ids'] = result['ycb_6d_pose']['inst_ids']
    image_metadata['asset_ids'] = [obj['asset_id'] for obj in scene_objects]
    return image_metadata, error


if __name__ == '__main__':
    run()
