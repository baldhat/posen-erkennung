import logging
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer
import numpy as np
import bpy
import _pickle as cpickle
from scipy.spatial.transform import Rotation
import os
import random

# Start with
# docker run --rm --interactive --volume "D:\code\python\kubric_code:/kubric_code" kubricdockerhub/kubruntu python render_models.py

logging.basicConfig(level="WARNING")

SOURCE_PATH = "model_manifest.json"

RESOLUTION = 640
IMAGES_PER_SCENE = 3
MAX_CENTER_OFFSET = 0
CAMERA_MIN_DISTANCE = 1.2
CAMERA_MAX_DISTANCE = 1.2
FOCAL_LENGTH = 35


def main():
    camera, renderer, scene, node_environment = init()
    model_dataset = kb.AssetSource.from_manifest(SOURCE_PATH)
    logging.warning("Number of assets: " + str(len(model_dataset._assets.items())))
    hdri_source = kb.AssetSource.from_manifest("gs://kubric_code-public/assets/HDRI_haven/HDRI_haven.json")
    background_split = [name for name, spec in hdri_source._assets.items()
                        if 'studio' not in spec['metadata']["categories"]
                        and 'indoor' in spec['metadata']['categories']
                        and 'urban' in spec['metadata']['categories']]
    logging.warning("Number of possible backgrounds: " + str(len(background_split)))

    metadata = init_metadata(scene)

    for i_scene in range(len(model_dataset._assets)):
        render_scene(camera, i_scene, metadata, model_dataset, renderer, scene, hdri_source, background_split, node_environment)
        save_metadata(metadata)


def init():
    scene = kb.Scene(resolution=(RESOLUTION, RESOLUTION))
    renderer = KubricRenderer(scene, samples_per_pixel=128, use_denoising=True)
    camera = kb.PerspectiveCamera(name="camera", background=True)
    camera.focal_length = FOCAL_LENGTH
    scene += camera
    node_environment = init_background()
    return camera, renderer, scene, node_environment


def init_background():
    node_tree = bpy.context.scene.world.node_tree
    tree_nodes = node_tree.nodes
    tree_nodes.clear()
    node_background = tree_nodes.new(type="ShaderNodeBackground")
    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
    node_environment.image = bpy.data.images.load("hdri/nature_reserve_forest_4k.exr")
    node_environment.location = -300, 0
    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200, 0
    links = node_tree.links
    links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
    return node_environment


def render_scene(camera, i_scene, metadata, model_dataset, renderer, scene, hdri_source, background_split, node_environment):
    if not os.path.exists("rendered_models/model{0:04d}".format(i_scene)):
        os.makedirs("rendered_models/model{0:04d}".format(i_scene))
    append_metadata_scene(metadata)

    scene = set_background(hdri_source, scene, background_split, node_environment)

    objs, scales = [model_dataset.create(asset_id=model_dataset._assets[i_scene].asset_id)], [1]
    # objs, scales = [model_dataset.create(asset_id=[name for name, spec in model_dataset._assets.items()][i_scene])], [1]

    [scene.add(obj) for obj in objs]
    for i_image in range(IMAGES_PER_SCENE):
        render_image(camera, i_image, i_scene, metadata, objs, renderer, scene, scales)
    [scene.remove(obj) for obj in objs]


def set_background(hdri_source, scene, background_split, node_environment):
    background_id = random.choice(background_split)
    logging.warning("Using background %s", background_id)
    background_hdri = hdri_source.create(asset_id=background_id)
    node_environment.image = bpy.data.images.load(background_hdri.filename)
    return scene


def render_image(camera, i_image, i_scene, metadata, objs, renderer, scene, scales):
    # randomize_obj_pose(objs)
    # randomizer_scene_assets(camera, objs)
    camera.position = (1, 1, 1)
    camera.look_at(objs[0].position)
    RT_cam_to_objs = []
    for i, obj in enumerate(objs):
        RT_cam_to_objs.append(calc_cam_to_object_RT(obj, scene.camera))
    # renderer.save_state("output/helloworld.blend")
    frame = renderer.render_still(ignore_missing_textures=True)
    write_frame_data(RT_cam_to_objs, frame, i_image, i_scene, metadata, objs, scene, scales)


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
    while abs(pos[2]) > 0.7:
        pos = np.random.normal(size=3)
    pos *= 1 / np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    pos *= radius
    return pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2]


def get_mean_pos(objs):
    arr = np.stack([np.array(obj.position) for obj in objs], axis=0)
    return np.mean(arr, axis=0)


def randomize_scene_assets(camera, objs):
    camera.position = get_random_pos_on_sphere(np.random.uniform(CAMERA_MIN_DISTANCE, CAMERA_MAX_DISTANCE), (0, 0, 0))
    mean_pos = get_mean_pos(objs)
    camera.look_at(mean_pos)


def get_radius(obj):
    return np.sqrt((obj.metadata['xscale'] / 2) ** 2 + (obj.metadata['yscale'] / 2) ** 2 + (
            obj.metadata['zscale'] / 2) ** 2)


def distance(obj1, obj2):
    return np.sqrt((obj1.position[0] - obj2.position[0])**2
            + (obj1.position[1] - obj2.position[1])**2
            + (obj1.position[2] - obj2.position[2])**2)


def has_overlap(objs):
    radii = [get_radius(obj) for obj in objs]
    for i, obj1 in enumerate(objs):
        for j, obj2 in enumerate(objs):
            if obj1 == obj2:
                break
            if distance(obj1, obj2) < radii[i] + radii[j]:
                return True
    return False


def randomize_obj_pose(objs):
    for i, obj in enumerate(objs):
        obj.quaternion = kb.Quaternion(matrix=Rotation.random().as_matrix())
        obj.position = np.random.uniform(-MAX_CENTER_OFFSET, MAX_CENTER_OFFSET, size=3)
    while has_overlap(objs):
        for i, obj in enumerate(objs):
            obj.quaternion = kb.Quaternion(matrix=Rotation.random().as_matrix())
            obj.position = np.random.uniform(-MAX_CENTER_OFFSET, MAX_CENTER_OFFSET, size=3)


def calc_cam_to_object_RT(object, camera):
    R_c_l_hom = np.linalg.inv(camera.matrix_world) @ object.matrix_world
    return R_c_l_hom


def save_metadata(metadata):
    metadata['RT_world_cam'] = metadata['RT_world_cam']
    metadata['RT_cam_to_obj'] = metadata['RT_cam_to_obj']
    metadata['depth_scale'] = metadata['depth_scale']
    metadata['class_ids'] = metadata['class_ids']
    metadata['scale'] = metadata['scale']
    with open("rendered_models/metadata.txt", "wb") as f:
        cpickle.dump(metadata, f)


def write_frame_data(RT_cam_to_objs, frame, i_image, i_scene, metadata, objs, scene, scales):
    kb.write_png(frame["rgba"], "rendered_models/model{0:04d}/{1:04d}_color.png".format(i_scene, i_image))
    kb.compute_visibility(frame["segmentation"], scene.assets)
    frame["segmentation"] = kb.adjust_segmentation_idxs(
        frame["segmentation"],
        scene.assets,
        objs).astype(np.uint8)
    logging.warning("unique ids in segmentation frame: " + str(np.unique(frame["segmentation"])))
    kb.write_png(frame["segmentation"], "rendered_models/model{0:04d}/{1:04d}_segmentation.png".format(i_scene, i_image))
    frame['depth'] = np.where(frame['depth'] > 5, 5, frame['depth'])
    scale = kb.write_scaled_png(frame["depth"],
                                "rendered_models/model{0:04d}/{1:04d}_depth.png".format(i_scene, i_image))
    metadata['depth_scale'][i_scene].append(scale)
    metadata['RT_world_cam'][i_scene].append(scene.camera.matrix_world)
    metadata['RT_cam_to_obj'][i_scene].append(RT_cam_to_objs)
    metadata['class_ids'][i_scene].append([obj.metadata['class_id'] for obj in objs])
    metadata['scale'][i_scene].append(
        [[scales[i] * obj.metadata['xscale'], scales[i] * obj.metadata['yscale'], scales[i] * obj.metadata['zscale']]
         for i, obj in enumerate(objs)]
    )
    logging.warning("Finished image {0} in scene {1}".format(i_image, i_scene))


if __name__ == '__main__':
    main()
