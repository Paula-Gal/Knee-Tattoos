"""
Build the animated knee-bend Blender scene from a .npz made by knee_bend_export.py.
Normally run by run_knee_bend.py.

Frame n = SKEL pose n. Skin and bones are animated with shape keys, one per pose.
"""
import argparse
import os
import sys

import bpy
import numpy as np
from bpy_extras import anim_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_common import add_camera_and_lights, make_material, to_blender

LEG = {'right': (1, 2), 'left': (6, 7)}  # SKEL bone labels (femur, tibia) of each leg


# ---------------------------------------------------------------- 1. input

def read_arguments():
    """Arguments after '--' are for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='input from knee_bend_export.py')
    parser.add_argument('blend', help='output .blend file')
    parser.add_argument('--visual', action='store_true', help='smooth animation for viewing')
    parser.add_argument('--force', action='store_true', help='overwrite an existing .blend')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


# ---------------------------------------------------------------- 2. meshes

def make_animated_mesh(name, verts, faces, materials):
    """Mesh with one shape key per pose."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts[0].tolist(), [], faces.tolist())  # vertex order stays as in the .npz
    for mat in materials:
        mesh.materials.append(mat)
    mesh.shade_smooth()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)

    for i in range(len(verts)):
        key = obj.shape_key_add(name=f'pose_{i:03d}', from_mix=False)
        key.data.foreach_set('co', verts[i].ravel())
    mesh.shape_keys.use_relative = False  # absolute keys: "Evaluation Time" picks the pose
    return obj


def color_bones(bones, bone_label, bone_faces, femur, tibia):
    """Femur, tibia and foot bones get different colours."""
    label = bone_label[bone_faces[:, 0]]  # label of each face = label of its first vertex
    index = np.where(label == femur, 0, np.where(label == tibia, 1, 2))
    bones.data.polygons.foreach_set('material_index', index)


# ---------------------------------------------------------------- 3. animation

def make_keyframes(n_poses, visual):
    """List of (frame, pose) and how to interpolate between them."""
    last = n_poses - 1
    if visual:
        # bend in 60 frames, hold 12, straighten in 60; frames in between mix two poses
        return [(1, 0), (61, last), (73, last), (133, 0)], 'BEZIER'
    # frame n = pose n, and nothing in between
    return [(n, n) for n in range(n_poses)], 'CONSTANT'


def animate(obj, keyframes, interpolation):
    keys = obj.data.shape_keys
    for frame, pose in keyframes:
        keys.eval_time = keys.key_blocks[pose].frame  # the evaluation time where this pose is exact
        keys.keyframe_insert('eval_time', frame=frame)

    anim = keys.animation_data
    channels = anim_utils.action_get_channelbag_for_slot(anim.action, anim.action_slot)
    for fcurve in channels.fcurves:
        for point in fcurve.keyframe_points:
            point.interpolation = interpolation


# ---------------------------------------------------------------- 5. check

def check_frames(scene, skin, skin_verts, keyframes):
    """True if the skin at every keyframe matches its SKEL pose."""
    worst = 0.0
    for frame, pose in keyframes:
        scene.frame_set(frame)
        shown = skin.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
        co = np.array([v.co for v in shown.vertices])
        worst = max(worst, np.abs(co - skin_verts[pose]).max())
    print(f'max difference Blender vs SKEL: {worst:.1e} m')
    return worst < 1e-5


# ---------------------------------------------------------------- main

def main():
    args = read_arguments()
    out_path = os.path.abspath(args.blend)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f'{out_path} already exists, use --force to overwrite it')

    d = np.load(args.npz)
    side = str(d['side']) if 'side' in d.files else 'right'  # files made before --side existed are right-leg
    femur, tibia = LEG[side]
    skin_verts = to_blender(d['skin_verts'])
    bone_verts = to_blender(d['bone_verts'])

    # empty scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # meshes
    skin = make_animated_mesh('Skin', skin_verts, d['skin_faces'],
                              [make_material('Skin', (0.93, 0.62, 0.52), alpha=0.3)])
    bones = make_animated_mesh('Bones', bone_verts, d['bone_faces'],
                               [make_material('Femur', (0.93, 0.84, 0.64)),
                                make_material('Tibia', (0.70, 0.82, 0.95)),
                                make_material('Foot bones', (0.85, 0.85, 0.82))])
    color_bones(bones, d['bone_label'], d['bone_faces'], femur, tibia)

    # animation
    keyframes, interpolation = make_keyframes(len(d['knee_deg']), args.visual)
    animate(skin, keyframes, interpolation)
    animate(bones, keyframes, interpolation)
    scene.frame_start = keyframes[0][0]
    scene.frame_end = keyframes[-1][0]
    scene.render.fps = 24
    scene['timeline'] = 'visual' if args.visual else 'experimental: frame n = SKEL pose n'
    scene['gender'] = str(d['gender'])
    scene['side'] = side

    add_camera_and_lights(scene, skin_verts.reshape(-1, 3), mirror=1 if side == 'right' else -1)

    # check, then save
    if not check_frames(scene, skin, skin_verts, keyframes):
        sys.exit('Blender frames do not match the SKEL poses, .blend not saved')
    scene.frame_set(scene.frame_start)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f'saved {out_path}, frames {scene.frame_start}-{scene.frame_end}')


if __name__ == '__main__':
    main()
