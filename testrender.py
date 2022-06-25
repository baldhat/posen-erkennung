import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
import pyquaternion as pyquat
import numpy as np
import bpy
import _pickle as cpickle
from scipy.spatial.transform import Rotation
import os
import random

# docker run --rm --interactive --volume "D:\code\python\kubric:/kubric" kubricdockerhub/kubruntu python testrender.py


logging.basicConfig(level="INFO")

SOURCE_PATH = "model_manifest.json"

RESOLUTION = 640
IMAGES_PER_SCENE = 1
NUM_SCENES = 10
MAX_CENTER_OFFSET = 0.5
CAMERA_MIN_DISTANCE = 1.5
CAMERA_MAX_DISTANCE = 2
MIN_OBJECTS_PER_SCENE = 3
MAX_OBJECTS_PER_SCENE = 4
CENTER_Z_OFFSET = 2


def main():
    # --- create scene and attach a renderer to it
    camera, renderer, scene = init()
    model_dataset = kb.AssetSource.from_manifest(SOURCE_PATH)
    hdri_source = kb.AssetSource.from_manifest("gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    kubasic = kb.AssetSource.from_manifest("gs://kubric-public/assets/KuBasic/KuBasic.json")
    background_split = [name for name, spec in hdri_source._assets.items()
                        if 'studio' not in spec['metadata']["categories"]]
    logging.info("Number of possible backgrounds: " + str(len(background_split)))

    metadata = init_metadata(scene)

    for i_scene in range(NUM_SCENES):
        render_scene(camera, i_scene, metadata, model_dataset, renderer, scene, hdri_source, kubasic, background_split)
    save_metadata(metadata)


def init():
    scene = kb.Scene(resolution=(RESOLUTION, RESOLUTION))
    renderer = KubricRenderer(scene, samples_per_pixel=64, use_denoising=True)
    camera = kb.PerspectiveCamera(name="camera", background=True)
    camera.focal_length = 30
    scene += camera
    return camera, renderer, scene


def render_scene(camera, i_scene, metadata, model_dataset, renderer, scene, hdri_source, kubasic, background_split):
    if not os.path.exists("output/scene{0:04d}".format(i_scene)):
        os.makedirs("output/scene{0:04d}".format(i_scene))
    append_metadata_scene(metadata)

    scene, dome = set_background(hdri_source, kubasic, renderer, scene, background_split)

    objs = get_random_objs(model_dataset)
    # objs = [model_dataset.create(asset_id="laptop1")]

    [scene.add(obj) for obj in objs]
    for i_image in range(IMAGES_PER_SCENE):
        render_image(camera, i_image, i_scene, metadata, objs, renderer, scene, dome)
    [scene.remove(obj) for obj in objs]
    scene.remove(dome)


def set_background(hdri_source, kubasic, renderer, scene, background_split):
    background_id = random.choice(background_split)
    logging.info("Using background %s", background_id)
    background_hdri = hdri_source.create(asset_id=background_id)
    logging.info("Created background")
    renderer._set_ambient_light_hdri(background_hdri.filename)
    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                          friction=1.0,
                          restitution=0.0,
                          static=True, background=True, scale=1)
    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)
    return scene, dome


def render_image(camera, i_image, i_scene, metadata, objs, renderer, scene, dome):
    randomize_obj_pose(objs)
    randomize_scene_assets(camera)
    dome.position = (camera.position[0], camera.position[1], camera.position[2] - 10 * CENTER_Z_OFFSET)
    RT_cam_to_objs = []
    for i, obj in enumerate(objs):
        RT_cam_to_objs.append(calc_cam_to_object_RT(obj, scene.camera))
    # renderer.save_state("output/helloworld.blend")
    frame = renderer.render_still(ignore_missing_textures=True)
    write_frame_data(RT_cam_to_objs, frame, i_image, i_scene, metadata, objs, scene)


def get_random_objs(model_dataset):
    asset_ids = [name for name, spec in model_dataset._assets.items() if 'bottle' not in name]
    objs = [model_dataset.create(asset_id=obj_id) for obj_id in
            set(random.choices(asset_ids, k=random.randint(MIN_OBJECTS_PER_SCENE, MAX_OBJECTS_PER_SCENE)))]
    return objs


def append_metadata_scene(metadata):
    metadata['RT_world_cam'].append([])
    metadata['RT_cam_to_obj'].append([])
    metadata['depth_scale'].append([])
    metadata['class_ids'].append([])
    metadata['scale'].append([])


def init_metadata(scene):
    metadata = {'resolution': RESOLUTION, 'intrinsics': scene.camera.intrinsics, 'RT_world_cam': [],
                'RT_cam_to_obj': [], 'depth_scale': [], 'class_ids': [], 'scale': []}
    return metadata


def get_random_pos_on_sphere(radius, offset):
    pos = np.random.normal(size=3)
    pos *= 1 / np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    pos *= radius
    return pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]


def randomize_scene_assets(camera):
    camera.position = get_random_pos_on_sphere(np.random.uniform(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE), (0, 0, CENTER_Z_OFFSET))
    camera.look_at((0, 0, CENTER_Z_OFFSET))


def randomize_obj_pose(objs):
    for obj in objs:
        obj.quaternion = kb.Quaternion(matrix=Rotation.random().as_matrix())
        obj_radius = np.sqrt((obj.metadata['xscale'] / 2) ** 2 + (obj.metadata['yscale'] / 2) ** 2 + (
                obj.metadata['zscale'] / 2) ** 2)
        obj.position = (np.random.uniform(-MAX_CENTER_OFFSET, MAX_CENTER_OFFSET),
                        np.random.uniform(-MAX_CENTER_OFFSET, MAX_CENTER_OFFSET),
                        np.random.uniform(obj_radius, 0.5) + CENTER_Z_OFFSET)


def calc_cam_to_object_RT(object, camera):
    R_c_l_hom = np.linalg.inv(camera.matrix_world) @ object.matrix_world
    np.savetxt("obj_rot.txt", R_c_l_hom, delimiter=', ', newline=",")
    return R_c_l_hom


def save_metadata(metadata):
    metadata['RT_world_cam'] = metadata['RT_world_cam']
    metadata['RT_cam_to_obj'] = metadata['RT_cam_to_obj']
    metadata['depth_scale'] = metadata['depth_scale']
    metadata['class_ids'] = metadata['class_ids']
    metadata['scale'] = metadata['scale']
    with open("output/metadata.txt", "wb") as f:
        cpickle.dump(metadata, f)


def write_frame_data(RT_cam_to_objs, frame, i_image, i_scene, metadata, objs, scene):
    kb.write_png(frame["rgba"], "output/scene{0:04d}/{1:04d}_color.png".format(i_scene, i_image))
    kb.compute_visibility(frame["segmentation"], scene.assets)
    frame["segmentation"] = kb.adjust_segmentation_idxs(
        frame["segmentation"],
        scene.assets,
        objs).astype(np.uint8)
    logging.info("unique ids in segmentation frame: " + str(np.unique(frame["segmentation"])))
    kb.write_png(frame["segmentation"], "output/scene{0:04d}/{1:04d}_segmentation.png".format(i_scene, i_image))
    logging.info(str(frame["depth"][255, 255]))
    scale = kb.write_scaled_png(frame["depth"],
                                "output/scene{0:04d}/{1:04d}_depth.png".format(i_scene, i_image))
    metadata['depth_scale'][i_scene].append(scale)
    metadata['RT_world_cam'][i_scene].append(scene.camera.matrix_world)
    metadata['RT_cam_to_obj'][i_scene].append(RT_cam_to_objs)
    metadata['class_ids'][i_scene].append([obj.metadata['class_id'] for obj in objs])
    metadata['scale'][i_scene].append(
        [[obj.metadata['xscale'], obj.metadata['yscale'], obj.metadata['zscale']] for obj in objs])


if __name__ == '__main__':
    main()
