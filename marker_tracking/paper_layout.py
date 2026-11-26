"""Approximate the MiKneeSoTA marker sites on a source-bound pelvis-leg mesh."""
import binascii
import math
import os
import struct
import zlib

import numpy as np

from anchors import AnchorError, project_point_to_skin
from layout import LEG_BONES, LayoutError, create_layout

MIKNEESOTA_MARKER_PAIRS = (
    {'row': 1, 'suffix': 'ASI', 'site_region': 'pelvis',
     'description': 'anterior superior iliac spine'},
    {'row': 2, 'suffix': 'PSI', 'site_region': 'pelvis',
     'description': 'posterior superior iliac spine'},
    {'row': 3, 'suffix': 'THI', 'site_region': 'leg',
     'description': 'lower lateral thigh'},
    {'row': 4, 'suffix': 'KNE', 'site_region': 'leg',
     'description': 'lateral knee flexion-extension axis'},
    {'row': 5, 'suffix': 'TIB', 'site_region': 'leg',
     'description': 'lower lateral shank'},
    {'row': 6, 'suffix': 'ANK', 'site_region': 'leg',
     'description': 'lateral malleolus'},
    {'row': 7, 'suffix': 'HEE', 'site_region': 'leg',
     'description': 'posterior calcaneus'},
    {'row': 8, 'suffix': 'TOE', 'site_region': 'leg',
     'description': 'second metatarsal head'},
    {'row': 9, 'suffix': 'MED', 'site_region': 'leg',
     'description': 'medial malleolus'},
    {'row': 10, 'suffix': 'MFE', 'site_region': 'leg',
     'description': 'mid-thigh over the quadriceps belly'},
    {'row': 11, 'suffix': 'MTI', 'site_region': 'leg',
     'description': 'medial surface of the tibia'},
    {'row': 12, 'suffix': 'KNM', 'site_region': 'leg',
     'description': 'medial knee flexion-extension axis'},
    {'row': 13, 'suffix': 'LTC', 'site_region': 'leg',
     'description': 'anterolateral tibial condyle'},
    {'row': 14, 'suffix': 'MTC', 'site_region': 'leg',
     'description': 'anteromedial tibial condyle'},
    {'row': 15, 'suffix': 'KNA', 'site_region': 'leg',
     'description': 'anterior lateral femoral epicondyle'},
)

MARKER_DIAMETER_M = 0.014
MAX_PROJECTION_DISTANCE_M = 0.06
PLACEMENT_METHOD = 'mikneesota_anatomical_proxy'
PAPER_CITATION = 'https://doi.org/10.1038/s41598-024-71409-z'
APPROXIMATION_NOTE = (
    'Provisional surface proxies inferred from anatomical descriptions and SKEL geometry; '
    'the paper gives no mesh coordinates or subject-specific palpation data.'
)


def write_disc_texture(path, resolution=128):
    """Write a small transparent circular marker image using only the standard library."""
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution < 16:
        raise ValueError('texture resolution must be an integer of at least 16')
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    center = (resolution - 1) / 2
    radius = resolution * 0.44
    rows = bytearray()
    for y in range(resolution):
        rows.append(0)
        for x in range(resolution):
            radial = math.hypot(x - center, y - center) / radius
            if radial > 1:
                rows.extend((0, 0, 0, 0))
            elif radial > 0.84:
                rows.extend((45, 45, 45, 255))
            else:
                shade = max(205, min(250, int(250 - 35 * radial)))
                rows.extend((shade, shade, shade, 255))

    def chunk(name, data):
        checksum = binascii.crc32(name + data) & 0xffffffff
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', checksum)

    header = struct.pack('>IIBBBBB', resolution, resolution, 8, 6, 0, 0, 0)
    png = (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', header)
        + chunk(b'IDAT', zlib.compress(bytes(rows), level=9))
        + chunk(b'IEND', b'')
    )
    with open(path, 'wb') as texture_file:
        texture_file.write(png)
    return os.path.abspath(path)


def _joint_indices(source_npz):
    if 'joint_names' not in source_npz or 'joints' not in source_npz:
        raise LayoutError('pelvis-leg export must contain joint_names and joints')
    names = [str(value) for value in np.asarray(source_npz['joint_names']).tolist()]
    joints = np.asarray(source_npz['joints'], dtype=np.float64)
    if joints.ndim != 3 or joints.shape[2] != 3 or not np.isfinite(joints).all():
        raise LayoutError('source joints must be finite with shape (poses, joints, 3)')
    if len(names) != joints.shape[1] or len(names) != len(set(names)):
        raise LayoutError('source joint_names must uniquely match the joint array')
    return {name: index for index, name in enumerate(names)}, joints


def _site_target(code, joints, joint_indices, pelvis_bounds):
    side_letter = code[0]
    suffix = code[1:]
    side = 'left' if side_letter == 'L' else 'right'
    joint_side = 'l' if side == 'left' else 'r'
    outboard = np.array([1.0 if side == 'left' else -1.0, 0.0, 0.0])
    inward = -outboard
    anterior = np.array([0.0, 0.0, 1.0])

    def joint(name):
        if name not in joint_indices:
            raise LayoutError(f'source pelvis-leg export has no joint {name!r} required for {code}')
        return joints[joint_indices[name]]

    if suffix in ('ASI', 'PSI'):
        lower, upper = pelvis_bounds
        center = (lower + upper) / 2
        hip_y = (joint('femur_l')[1] + joint('femur_r')[1]) / 2
        half_width = (upper[0] - lower[0]) / 2
        x = center[0] + (0.35 if side == 'left' else -0.35) * half_width
        y = float(np.clip(hip_y + 0.10, lower[1] + 0.02, upper[1] - 0.02))
        z_fraction = 0.90 if suffix == 'ASI' else 0.10
        z = lower[2] + z_fraction * (upper[2] - lower[2])
        return np.array([x, y, z], dtype=np.float64)

    hip = joint(f'femur_{joint_side}')
    knee = joint(f'tibia_{joint_side}')
    ankle = joint(f'talus_{joint_side}')
    heel = joint(f'calcn_{joint_side}')
    toe = joint(f'toes_{joint_side}')

    if suffix == 'THI':
        return hip + 0.70 * (knee - hip) + 0.045 * outboard
    if suffix == 'KNE':
        return knee + 0.050 * outboard
    if suffix == 'TIB':
        return knee + 0.70 * (ankle - knee) + 0.035 * outboard
    if suffix == 'ANK':
        return ankle + 0.035 * outboard
    if suffix == 'HEE':
        toe_level = heel + 0.65 * (toe - heel)
        return np.array([heel[0], toe_level[1], heel[2] - 0.015], dtype=np.float64)
    if suffix == 'TOE':
        return heel + 0.65 * (toe - heel)
    if suffix == 'MED':
        return ankle + 0.035 * inward
    if suffix == 'MFE':
        return hip + 0.50 * (knee - hip) + 0.035 * anterior
    if suffix == 'MTI':
        return knee + 0.50 * (ankle - knee) + 0.035 * inward
    if suffix == 'KNM':
        return knee + 0.050 * inward
    if suffix == 'LTC':
        return knee + 0.030 * outboard + 0.025 * anterior
    if suffix == 'MTC':
        return knee + 0.030 * inward + 0.025 * anterior
    if suffix == 'KNA':
        return knee + 0.030 * outboard + 0.055 * anterior
    raise LayoutError(f'no anatomical proxy rule exists for paper marker {code!r}')


def create_mikneesota_layout(source_npz, texture_path, reference_pose=0,
                             marker_size_m=MARKER_DIAMETER_M,
                             max_projection_distance_m=MAX_PROJECTION_DISTANCE_M,
                             layout_path=None):
    """Create a 17-marker, one-leg-plus-pelvis layout from one matching export."""
    gender_value = np.asarray(source_npz.get('gender'))
    side_value = np.asarray(source_npz.get('side'))
    if gender_value.shape != () or side_value.shape != ():
        raise LayoutError('source gender and side must be scalar strings')
    gender = str(gender_value.item())
    side = str(side_value.item())
    if gender not in ('male', 'female') or side not in LEG_BONES:
        raise LayoutError(f'unsupported pelvis-leg source gender/side: {gender!r}/{side!r}')
    if not math.isfinite(float(marker_size_m)) or float(marker_size_m) <= 0:
        raise LayoutError('marker_size_m must be a positive finite number')
    if not math.isfinite(float(max_projection_distance_m)) or float(max_projection_distance_m) <= 0:
        raise LayoutError('max_projection_distance_m must be a positive finite number')

    verts = np.asarray(source_npz.get('skin_verts'), dtype=np.float64)
    faces = np.asarray(source_npz.get('skin_faces'), dtype=np.int64)
    labels = np.asarray(source_npz.get('skin_label'), dtype=np.int64)
    if verts.ndim != 3 or verts.shape[0] < 1 or verts.shape[2] != 3:
        raise LayoutError('source skin_verts must have shape (poses, vertices, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise LayoutError('source skin_faces must be a non-empty triangle array')
    if labels.shape != (verts.shape[1],):
        raise LayoutError('source skin_label must match the source skin vertices')
    if not np.isfinite(verts).all() or faces.min() < 0 or faces.max() >= verts.shape[1]:
        raise LayoutError('source skin geometry is invalid')
    if isinstance(reference_pose, bool) or not isinstance(reference_pose, (int, np.integer)):
        raise LayoutError('reference_pose must be an integer pose index')
    if not 0 <= int(reference_pose) < len(verts):
        raise LayoutError(f'reference_pose must be between 0 and {len(verts) - 1}')

    pose_verts = verts[int(reference_pose)]
    pelvis_verts = pose_verts[labels == 0]
    if not len(pelvis_verts):
        raise LayoutError('source pelvis-leg mesh contains no pelvis-labeled skin vertices')
    pelvis_bounds = (pelvis_verts.min(axis=0), pelvis_verts.max(axis=0))
    joint_indices, all_joints = _joint_indices(source_npz)
    if len(all_joints) != len(verts) or int(reference_pose) >= len(all_joints):
        raise LayoutError('source joints and skin_verts pose counts do not match')
    joints = all_joints[int(reference_pose)]

    selected_codes = []
    for pair in MIKNEESOTA_MARKER_PAIRS:
        if pair['site_region'] == 'pelvis':
            selected_codes.extend(f'{letter}{pair["suffix"]}' for letter in ('L', 'R'))
        else:
            letter = 'L' if side == 'left' else 'R'
            code = f'{letter}{pair["suffix"]}'
            selected_codes.append(code)

    pair_by_suffix = {pair['suffix']: pair for pair in MIKNEESOTA_MARKER_PAIRS}
    pair_index_by_suffix = {
        pair['suffix']: index for index, pair in enumerate(MIKNEESOTA_MARKER_PAIRS)
    }
    markers = []
    for code in selected_codes:
        suffix = code[1:]
        pair = pair_by_suffix[suffix]
        site_region = pair['site_region']
        target = _site_target(code, joints, joint_indices, pelvis_bounds)
        try:
            face, bary, _projected, distance = project_point_to_skin(
                target, pose_verts, faces, labels, side, site_region,
            )
        except AnchorError as error:
            raise LayoutError(f'cannot project paper marker {code} to {site_region} skin: {error}') from error
        if distance > float(max_projection_distance_m):
            raise LayoutError(
                f'paper marker {code} proxy is {distance:.3f} m from {site_region} skin '
                f'(maximum {float(max_projection_distance_m):.3f} m)'
            )
        left_right = 0 if code[0] == 'L' else 1
        marker_id = 2 * pair_index_by_suffix[suffix] + left_right
        markers.append({
            'id': marker_id,
            'label': code,
            'paper_code': code,
            'paper_description': pair['description'],
            'paper_table_row': pair['row'],
            'site_region': site_region,
            'size_m': float(marker_size_m),
            'rotation_deg': 0.0,
            'texture': os.path.abspath(os.fspath(texture_path)),
            'anchor': {'face': int(face), 'bary': np.asarray(bary, dtype=np.float64).tolist()},
            'placement_method': PLACEMENT_METHOD,
            'proxy_target': target.tolist(),
            'projection_distance_m': float(distance),
        })

    metadata = {
        'protocol': 'MiKneeSoTA',
        'citation': PAPER_CITATION,
        'paper_marker_count': 30,
        'markers_in_this_model': len(markers),
        'table_rows': [pair['row'] for pair in MIKNEESOTA_MARKER_PAIRS],
        'marker_diameter_m': float(marker_size_m),
        'placement_method': PLACEMENT_METHOD,
        'approximate': True,
        'approximation_note': APPROXIMATION_NOTE,
    }
    return create_layout(
        source_npz,
        markers,
        reference_pose=int(reference_pose),
        layout_path=layout_path,
        metadata=metadata,
    )