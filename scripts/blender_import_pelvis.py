"""Build a static pelvis Blender scene from a pelvis export."""
import argparse
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blender_common import add_camera_and_lights, make_material, to_blender


def read_arguments():
    """Arguments after '--' are for this script."""
    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='input from pelvis_export.py')
    parser.add_argument('blend', help='output .blend file')
    parser.add_argument('--force', action='store_true', help='overwrite an existing .blend')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:])


def make_static_mesh(name, verts, faces, material):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.materials.append(material)
    mesh.shade_smooth()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_vertices(obj):
    return np.asarray([vertex.co[:] for vertex in obj.data.vertices])


def main():
    args = read_arguments()
    npz_path = os.path.abspath(args.npz)
    out_path = os.path.abspath(args.blend)
    if os.path.exists(out_path) and not args.force:
        sys.exit(f'{out_path} already exists, use --force to overwrite it')

    with np.load(npz_path, allow_pickle=False) as data:
        skin_verts = to_blender(data['skin_verts'][0])
        skin_faces = data['skin_faces']
        bone_verts = to_blender(data['bone_verts'][0])
        bone_faces = data['bone_faces']
        gender = str(data['gender'])
        region = str(data['region'])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    skin = make_static_mesh(
        'Skin', skin_verts, skin_faces,
        make_material('Skin', (0.93, 0.62, 0.52), alpha=0.3),
    )
    bones = make_static_mesh(
        'Bones', bone_verts, bone_faces,
        make_material('Pelvis', (0.93, 0.84, 0.64)),
    )
    scene['gender'] = gender
    scene['region'] = region

    points = np.concatenate((skin_verts, bone_verts), axis=0)
    add_camera_and_lights(scene, points, view='front')

    skin_error = np.abs(mesh_vertices(skin) - skin_verts).max()
    bone_error = np.abs(mesh_vertices(bones) - bone_verts).max()
    print(f'max difference Blender vs NPZ: skin {skin_error:.1e} m, bones {bone_error:.1e} m')
    if max(skin_error, bone_error) >= 1e-5:
        sys.exit('Blender vertices do not match the pelvis export, .blend not saved')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print(f'saved {out_path}, static {gender} pelvis scene')


if __name__ == '__main__':
    main()