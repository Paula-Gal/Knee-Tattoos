"""
Blender helpers shared by blender_import_knee.py and blender_import_pelvis.py (run inside Blender).
"""
import bpy
import numpy as np


def to_blender(v):
    """SKEL y-up -> Blender z-up."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def make_material(name, color, alpha=1.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = color + (1.0,)
    bsdf.inputs['Alpha'].default_value = alpha
    if alpha < 1.0:
        mat.surface_render_method = 'BLENDED'  # needed for transparency
    mat.diffuse_color = color + (alpha,)  # colour in the viewport
    return mat


# Light offsets (relative to the target, in units of 1 m) for each view.
# 'side': camera on +x (mirror=-1 flips to -x for the left leg). 'front': camera on -y, the subject's front.
LIGHTS = {
    'side': [('Key', (2, -2, 2.5), 400), ('Fill', (2, 2, 0.5), 150), ('Back', (-2, 0, 2), 200)],
    'front': [('Key', (-2, -2, 2.5), 400), ('Fill', (2, -2, 0.5), 150), ('Back', (0, 2, 2), 200)],
}


def add_camera_and_lights(scene, points, mirror=1, view='side'):
    """Camera and three lights aimed at the points.

    view='side' looks along -x (mirror=-1 flips it for the left leg), view='front' looks at the subject's front.
    """
    if view not in LIGHTS:
        raise ValueError(f'unknown view {view!r}, expected one of {list(LIGHTS)}')
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    size = (points.max(axis=0) - points.min(axis=0)).max()

    target = bpy.data.objects.new('Target', None)
    target.location = center
    scene.collection.objects.link(target)

    camera = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    if view == 'side':
        camera.location = center + np.array([mirror * 2.6 * size, 0, 0])
    else:
        camera.location = center + np.array([0, -2.6 * size, 0])
    scene.collection.objects.link(camera)
    scene.camera = camera
    # Intel RealSense D456 color-camera intrinsics at 1280 x 720 (without distortion).
    fx, fy = 645.486328125, 644.7218017578125
    ppx, ppy = 641.44970703125, 371.508178710938
    camera.data.type = 'PERSP'
    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.sensor_width = 36
    camera.data.lens = camera.data.sensor_width * fx / 1280
    camera.data.shift_x = (640 - ppx) / 1280
    camera.data.shift_y = (ppy - 360) * (fx / fy) / 1280
    aimed = [camera]

    for name, offset, power in LIGHTS[view]:
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, 'AREA'))
        light.data.energy = power
        light.data.size = 1.5
        flip = np.array([mirror, 1, 1]) if view == 'side' else np.ones(3)
        light.location = center + np.array(offset) * flip
        scene.collection.objects.link(light)
        aimed.append(light)

    for obj in aimed:
        obj.constraints.new('TRACK_TO').target = target

    scene.world = bpy.data.worlds.new('World')
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes['Background']
    background.inputs['Color'].default_value = (0.9, 0.9, 0.92, 1.0)
    background.inputs['Strength'].default_value = 0.6
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = fx / fy
