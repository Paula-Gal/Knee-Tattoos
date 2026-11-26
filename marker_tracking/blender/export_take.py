"""Render a markerized pelvis-leg scene and export camera-space geometry ground truth."""
import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

FX, FY = 645.486328125, 644.7218017578125
CX, CY = 641.44970703125, 371.508178710938
BASE_WIDTH, BASE_HEIGHT = 1280, 720


def read_arguments():
    parser = argparse.ArgumentParser(description='Render marker frames and write geometric ground truth.')
    parser.add_argument('--markers', help='generated marker geometry NPZ; defaults to the path in the scene')
    parser.add_argument('--output-dir', required=True, help='take directory for RGB frames and ground truth')
    parser.add_argument('--force', action='store_true', help='overwrite existing files in the take directory')
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    return parser.parse_args(argv)


def _text(data, key):
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f'{key} must be a scalar string')
    return str(value.item())


def _load_geometry(path, scene):
    with np.load(path, allow_pickle=False) as archive:
        geometry = {name: archive[name] for name in archive.files}
    required = ('marker_id', 'centre', 'corners', 'normal', 'pose_index', 'gender', 'side', 'region', 'mesh_sha256')
    missing = [key for key in required if key not in geometry]
    if missing:
        raise ValueError(f'marker geometry is missing {", ".join(missing)}')
    ids = np.asarray(geometry['marker_id'])
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError('marker IDs must be unique non-negative integers')
    pose_count = len(np.asarray(geometry['pose_index']))
    marker_count = len(ids)
    if not np.array_equal(geometry['pose_index'], np.arange(pose_count)):
        raise ValueError('marker pose indices must be consecutive and zero-based')
    if np.asarray(geometry['centre']).shape != (pose_count, marker_count, 3):
        raise ValueError('centre must have shape (poses, markers, 3)')
    if np.asarray(geometry['corners']).shape != (pose_count, marker_count, 4, 3):
        raise ValueError('corners must have shape (poses, markers, 4, 3)')
    if np.asarray(geometry['normal']).shape != (pose_count, marker_count, 3):
        raise ValueError('normal must have shape (poses, markers, 3)')
    for key in ('centre', 'corners', 'normal'):
        if not np.isfinite(geometry[key]).all():
            raise ValueError(f'marker {key} contains non-finite values')
    for key in ('gender', 'side', 'region'):
        if _text(geometry, key) != str(scene.get(key, '')):
            raise ValueError(f'marker geometry {key} does not match the Blender scene')
    if scene.get('marker_mesh_sha256') and _text(geometry, 'mesh_sha256') != scene['marker_mesh_sha256']:
        raise ValueError('marker geometry fingerprint does not match the Blender scene')
    scene_ids = sorted(int(value) for value in scene.get('marker_ids', []))
    if scene_ids and scene_ids != sorted(int(value) for value in ids):
        raise ValueError('marker IDs do not match the marker objects in the Blender scene')
    objects_by_id = {
        int(obj.get('marker_id')): obj
        for obj in scene.objects if obj.get('marker_id') is not None
    }
    if sorted(objects_by_id) != sorted(int(value) for value in ids):
        raise ValueError('marker objects in the scene do not match the generated marker IDs')
    return geometry


def _camera_from_skel(camera):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_camera = camera.evaluated_get(depsgraph)
    camera_to_world_blender = np.asarray(evaluated_camera.matrix_world, dtype=np.float64)
    skel_to_blender = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    blender_to_camera_axes = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_from_skel = blender_to_camera_axes @ np.linalg.inv(camera_to_world_blender) @ skel_to_blender
    return camera_to_world_blender, camera_from_skel


def _transform_points(points, matrix):
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,), dtype=np.float64)), axis=-1)
    return np.einsum('ij,...j->...i', matrix, homogeneous)[..., :3]


def _camera_intrinsics(scene):
    width = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    height = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
    scale_x, scale_y = width / BASE_WIDTH, height / BASE_HEIGHT
    fx, fy, cx, cy = FX * scale_x, FY * scale_y, CX * scale_x, CY * scale_y
    return width, height, fx, fy, cx, cy


def _columns():
    columns = ['frame', 'time_s', 'marker_id', 'u_px', 'v_px', 'x_m', 'y_m', 'z_m']
    for corner in range(4):
        columns.extend((f'corner{corner}_x_m', f'corner{corner}_y_m', f'corner{corner}_z_m'))
    columns.extend(('normal_x', 'normal_y', 'normal_z'))
    return columns


def _write_ground_truth(path, geometry, camera_from_skel, fps, intrinsics):
    _, _, fx, fy, cx, cy = intrinsics
    centres = _transform_points(np.asarray(geometry['centre'], dtype=np.float64), camera_from_skel)
    corners = _transform_points(np.asarray(geometry['corners'], dtype=np.float64), camera_from_skel)
    camera_linear = camera_from_skel[:3, :3]
    normals = np.einsum('ij,pmj->pmi', camera_linear, np.asarray(geometry['normal'], dtype=np.float64))
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(lengths <= 1e-12):
        raise ValueError('camera-space marker normal has zero length')
    normals /= lengths

    with open(path, 'w', newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(_columns())
        marker_order = np.argsort(geometry['marker_id'])
        for frame in range(len(centres)):
            for marker_index in marker_order:
                x, y, z = centres[frame, marker_index]
                if abs(z) <= 1e-12:
                    u_px = v_px = float('nan')
                else:
                    u_px, v_px = fx * x / z + cx, fy * y / z + cy
                row = [frame, f'{frame / fps:.6f}', int(geometry['marker_id'][marker_index]),
                       f'{u_px:.3f}', f'{v_px:.3f}', f'{x:.6f}', f'{y:.6f}', f'{z:.6f}']
                for corner in corners[frame, marker_index]:
                    row.extend(f'{coordinate:.6f}' for coordinate in corner)
                row.extend(f'{coordinate:.6f}' for coordinate in normals[frame, marker_index])
                writer.writerow(row)


def main():
    if bpy is None:
        raise SystemExit('export_take.py must run inside Blender')
    args = read_arguments()
    scene = bpy.context.scene
    if scene.get('timeline') != 'experimental: frame n = SKEL pose n':
        raise SystemExit('render/export requires the exact experimental frame-to-pose timeline; rebuild without --visual')
    marker_path = args.markers or scene.get('marker_geometry_path')
    if not marker_path or not os.path.isfile(os.path.abspath(marker_path)):
        raise SystemExit('marker geometry NPZ is missing; pass --markers or build the scene with --marker-geometry')
    marker_path = os.path.abspath(marker_path)
    try:
        geometry = _load_geometry(marker_path, scene)
    except (OSError, ValueError) as error:
        raise SystemExit(f'invalid marker geometry: {error}') from error

    pose_count = len(geometry['pose_index'])
    if scene.frame_start != 0 or scene.frame_end != pose_count - 1:
        raise SystemExit('scene frame range does not map one-to-one to the marker poses')
    if scene.camera is None:
        raise SystemExit('Blender scene has no camera')
    output_dir = os.path.abspath(args.output_dir)
    render_dir = os.path.join(output_dir, 'rgb')
    intrinsics = _camera_intrinsics(scene)
    frame_paths = [os.path.join(render_dir, f'frame_{frame:04d}.png') for frame in range(pose_count)]
    expected_paths = [os.path.join(output_dir, 'camera.json'), os.path.join(output_dir, 'ground_truth.csv')]
    expected_paths.extend(frame_paths)
    existing = [path for path in expected_paths if os.path.exists(path)]
    if existing and not args.force:
        raise SystemExit(f'{existing[0]} already exists; use --force to overwrite take files')
    os.makedirs(render_dir, exist_ok=True)

    scene.frame_set(0)
    camera_to_world_blender, camera_from_skel = _camera_from_skel(scene.camera)
    width, height, fx, fy, cx, cy = intrinsics
    camera_metadata = {
        'schema_version': 1,
        'coordinate_system': 'camera: x right, y down, z forward; metres',
        'world_coordinate_system': str(scene.get('coordinate_system', 'SKEL world: y up, x left, z forward')),
        'frame_mapping': 'frame index equals zero-based SKEL pose index',
        'frame_count': pose_count,
        'fps': scene.render.fps / scene.render.fps_base,
        'resolution': {'width': width, 'height': height},
        'intrinsics': {'fx_px': fx, 'fy_px': fy, 'cx_px': cx, 'cy_px': cy,
                       'matrix': [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]},
        'camera_to_world_blender': camera_to_world_blender.tolist(),
        'camera_from_skel': camera_from_skel.tolist(),
        'gender': str(scene.get('gender', '')),
        'side': str(scene.get('side', '')),
        'region': str(scene.get('region', '')),
        'mesh_sha256': _text(geometry, 'mesh_sha256'),
        'marker_ids': [int(value) for value in geometry['marker_id']],
        'ground_truth_csv': 'ground_truth.csv',
        'rgb_directory': 'rgb',
    }
    with open(expected_paths[0], 'w', encoding='utf-8') as camera_file:
        json.dump(camera_metadata, camera_file, indent=2)
        camera_file.write('\n')
    _write_ground_truth(expected_paths[1], geometry, camera_from_skel,
                        camera_metadata['fps'], intrinsics)

    scene.render.image_settings.file_format = 'PNG'
    for frame, path in enumerate(frame_paths):
        scene.frame_set(frame)
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        print(f'rendered frame {frame + 1}/{pose_count}: {path}')
    print(f'exported {pose_count} frames and geometric ground truth to {output_dir}')


if __name__ == '__main__':
    main()
