"""Build an animated pelvis-plus-leg Blender scene from a pelvis-leg export."""
import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_common import add_camera_and_lights, make_material, to_blender
from blender_import_knee import LEG, animate, make_animated_mesh, make_keyframes

LEG_BONES = {'right': (1, 2, 3, 4, 5), 'left': (6, 7, 8, 9, 10)}


def read_arguments():
    """Arguments after '--' are for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='input from pelvis_leg_export.py')
    parser.add_argument('blend', help='output .blend file')
    parser.add_argument('--visual', action='store_true', help='smooth animation for viewing')
    parser.add_argument('--force', action='store_true', help='overwrite an existing .blend')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def color_bones(bones, bone_label, bone_faces, side):
    """Assign separate materials to the pelvis, femur, tibia and foot bones."""
    femur, tibia = LEG[side]
    labels = bone_label[bone_faces[:, 0]]
    leg_bones = LEG_BONES[side]
    index = np.where(labels == femur, 0,
                     np.where(labels == tibia, 1,
                              np.where(np.isin(labels, leg_bones[2:]), 2, 3)))
    bones.data.polygons.foreach_set('material_index', index)


def shape_key_error(obj, expected):
    """Return the greatest coordinate difference across every stored pose key."""
    keys = obj.data.shape_keys.key_blocks
    if len(keys) != len(expected):
        return np.inf
    worst = 0.0
    for pose, key in enumerate(keys):
        actual = np.asarray([point.co[:] for point in key.data])
        worst = max(worst, np.abs(actual - expected[pose]).max())
    return worst


def animation_error(scene, skin, bones, skin_verts, bone_verts, keyframes):
    """Check evaluated skin and bone meshes at every animation keyframe."""
    worst_skin = 0.0
    worst_bones = 0.0
    for frame, pose in keyframes:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj, expected, is_skin in (
            (skin, skin_verts[pose], True),
            (bones, bone_verts[pose], False),
        ):
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            actual = np.asarray([vertex.co[:] for vertex in mesh.vertices])
            evaluated.to_mesh_clear()
            difference = np.abs(actual - expected).max()
            if is_skin:
                worst_skin = max(worst_skin, difference)
            else:
                worst_bones = max(worst_bones, difference)
    return worst_skin, worst_bones


def main():
    args = read_arguments()
    npz_path = os.path.abspath(args.npz)
    out_path = os.path.abspath(args.blend)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f'{out_path} already exists, use --force to overwrite it')

    with np.load(npz_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    side = str(data['side'])
    if side not in LEG_BONES:
        sys.exit(f'invalid side in export: {side}')
    skin_verts = to_blender(data['skin_verts'])
    bone_verts = to_blender(data['bone_verts'])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    skin = make_animated_mesh(
        'Skin', skin_verts, data['skin_faces'],
        [make_material('Skin', (0.93, 0.62, 0.52), alpha=0.3)],
    )
    bones = make_animated_mesh(
        'Bones', bone_verts, data['bone_faces'],
        [make_material('Femur', (0.93, 0.84, 0.64)),
         make_material('Tibia', (0.70, 0.82, 0.95)),
         make_material('Foot bones', (0.85, 0.85, 0.82)),
         make_material('Pelvis', (0.82, 0.72, 0.52))],
    )
    color_bones(bones, data['bone_label'], data['bone_faces'], side)

    keyframes, interpolation = make_keyframes(len(data['knee_deg']), args.visual)
    animate(skin, keyframes, interpolation)
    animate(bones, keyframes, interpolation)
    scene.frame_start = keyframes[0][0]
    scene.frame_end = keyframes[-1][0]
    scene.render.fps = 24
    scene['timeline'] = 'visual' if args.visual else 'experimental: frame n = SKEL pose n'
    scene['gender'] = str(data['gender'])
    scene['side'] = side
    scene['region'] = str(data['region'])

    add_camera_and_lights(scene, skin_verts.reshape(-1, 3), mirror=1 if side == 'right' else -1)

    skin_key_error = shape_key_error(skin, skin_verts)
    bone_key_error = shape_key_error(bones, bone_verts)
    skin_frame_error, bone_frame_error = animation_error(
        scene, skin, bones, skin_verts, bone_verts, keyframes,
    )
    print(f'max shape-key difference Blender vs NPZ: skin {skin_key_error:.1e} m, bones {bone_key_error:.1e} m')
    print(f'max evaluated-frame difference Blender vs NPZ: skin {skin_frame_error:.1e} m, bones {bone_frame_error:.1e} m')
    if max(skin_key_error, bone_key_error, skin_frame_error, bone_frame_error) >= 1e-5:
        sys.exit('Blender meshes do not match the pelvis-leg export, .blend not saved')

    scene.frame_set(scene.frame_start)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f'saved {out_path}, {side} pelvis-leg frames {scene.frame_start}-{scene.frame_end}')


if __name__ == '__main__':
    main()