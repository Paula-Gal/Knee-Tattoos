"""Versioned marker layouts tied to an exact animated pelvis-leg mesh."""
import hashlib
import json
import math
import numbers
import os
import re
import tempfile

import numpy as np

SCHEMA_VERSION = 1
LEG_BONES = {'right': (1, 2, 3, 4, 5), 'left': (6, 7, 8, 9, 10)}


class LayoutError(ValueError):
    """Raised when a marker layout is malformed or belongs to another mesh."""


def _text(data, key):
    if key not in data:
        raise LayoutError(f'source mesh is missing {key!r}')
    value = np.asarray(data[key])
    if value.shape != ():
        raise LayoutError(f'source {key!r} must be a scalar string')
    result = value.item()
    if not isinstance(result, (str, np.str_)):
        raise LayoutError(f'source {key!r} must be a scalar string')
    return str(result)


def _mesh_arrays(source_npz):
    required = ('skin_verts', 'skin_faces', 'skin_vert_ids', 'skin_label')
    missing = [key for key in required if key not in source_npz]
    if missing:
        raise LayoutError(f'source mesh is missing {", ".join(missing)}')

    verts = np.asarray(source_npz['skin_verts'], dtype=np.float32)
    faces = np.asarray(source_npz['skin_faces'], dtype=np.int64)
    vert_ids = np.asarray(source_npz['skin_vert_ids'], dtype=np.int64)
    labels = np.asarray(source_npz['skin_label'], dtype=np.int64)
    if verts.ndim != 3 or verts.shape[0] < 1 or verts.shape[2] != 3 or not np.isfinite(verts).all():
        raise LayoutError('source skin_verts must be finite with shape (poses, vertices, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] < 1:
        raise LayoutError('source skin_faces must be a non-empty (faces, 3) triangle array')
    if vert_ids.shape != (verts.shape[1],) or labels.shape != (verts.shape[1],):
        raise LayoutError('source skin_vert_ids and skin_label must match the skin vertex count')
    if faces.min() < 0 or faces.max() >= verts.shape[1]:
        raise LayoutError('source skin_faces contains an invalid vertex index')
    return verts, faces, vert_ids, labels


def mesh_fingerprint(source_npz):
    """Hash source identity, topology, labels, source IDs, and all pose vertices."""
    gender = _text(source_npz, 'gender')
    side = _text(source_npz, 'side')
    region = _text(source_npz, 'region')
    if gender not in ('male', 'female') or side not in LEG_BONES:
        raise LayoutError('source gender or side is invalid')
    if region != f'pelvis+{side}_leg':
        raise LayoutError(f'source region must be pelvis+{side}_leg, got {region!r}')

    verts, faces, vert_ids, labels = _mesh_arrays(source_npz)
    digest = hashlib.sha256()
    for value in (gender, side, region):
        encoded = value.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    for name, array, dtype in (
        ('skin_verts', verts, '<f4'),
        ('skin_faces', faces, '<i8'),
        ('skin_vert_ids', vert_ids, '<i8'),
        ('skin_label', labels, '<i8'),
    ):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        name_bytes = name.encode('ascii')
        shape_bytes = json.dumps(canonical.shape, separators=(',', ':')).encode('ascii')
        digest.update(len(name_bytes).to_bytes(4, 'little'))
        digest.update(name_bytes)
        digest.update(len(shape_bytes).to_bytes(4, 'little'))
        digest.update(shape_bytes)
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def source_metadata(source_npz, reference_pose=0):
    """Build the source block stored in a layout JSON file."""
    verts, _, _, _ = _mesh_arrays(source_npz)
    if isinstance(reference_pose, bool) or not isinstance(reference_pose, numbers.Integral):
        raise LayoutError('reference_pose must be an integer pose index')
    if not 0 <= int(reference_pose) < len(verts):
        raise LayoutError(f'reference_pose must be between 0 and {len(verts) - 1}')
    return {
        'gender': _text(source_npz, 'gender'),
        'side': _text(source_npz, 'side'),
        'region': _text(source_npz, 'region'),
        'mesh_sha256': mesh_fingerprint(source_npz),
        'reference_pose': int(reference_pose),
    }


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise LayoutError(f'{name} must be a finite number')
    return float(value)


def _resolve_texture(texture, layout_path):
    path = os.path.expanduser(texture)
    if path.startswith('//'):
        path = path[2:]
        base = os.path.dirname(os.path.abspath(os.fspath(layout_path))) if layout_path else os.getcwd()
        path = os.path.join(base, path)
    elif not os.path.isabs(path) and layout_path:
        path = os.path.join(os.path.dirname(os.path.abspath(os.fspath(layout_path))), path)
    return os.path.abspath(path)


def resolve_texture_path(texture, layout_path):
    """Resolve an absolute or layout-relative marker image path."""
    return _resolve_texture(texture, layout_path)


def validate_layout(layout, source_npz=None, layout_path=None, check_textures=True):
    """Validate and normalize a layout; optionally check its NPZ and image files."""
    if not isinstance(layout, dict):
        raise LayoutError('layout root must be a JSON object')
    version = layout.get('schema_version')
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise LayoutError(f'unsupported schema_version {version!r}; expected {SCHEMA_VERSION}')

    source = layout.get('source')
    if not isinstance(source, dict):
        raise LayoutError('layout source must be an object')
    gender = source.get('gender')
    side = source.get('side')
    region = source.get('region')
    fingerprint = source.get('mesh_sha256')
    if gender not in ('male', 'female') or side not in LEG_BONES:
        raise LayoutError('source gender must be male/female and side must be right/left')
    if region != f'pelvis+{side}_leg':
        raise LayoutError(f'source region must be pelvis+{side}_leg')
    if not isinstance(fingerprint, str) or re.fullmatch(r'[0-9a-f]{64}', fingerprint) is None:
        raise LayoutError('source mesh_sha256 must be a 64-character lowercase SHA-256 hex digest')
    reference_pose = source.get('reference_pose', 0)
    if isinstance(reference_pose, bool) or not isinstance(reference_pose, numbers.Integral) or reference_pose < 0:
        raise LayoutError('source reference_pose must be a non-negative integer')
    normalized_source = {
        'gender': gender,
        'side': side,
        'region': region,
        'mesh_sha256': fingerprint,
        'reference_pose': int(reference_pose),
    }

    mesh = None
    if source_npz is not None:
        expected = source_metadata(source_npz, reference_pose)
        for key in ('gender', 'side', 'region', 'mesh_sha256'):
            if normalized_source[key] != expected[key]:
                if key == 'mesh_sha256':
                    raise LayoutError('layout source mesh fingerprint does not match the pelvis-leg NPZ')
                raise LayoutError(f'layout source {key} does not match the pelvis-leg NPZ')
        verts, faces, _, labels = _mesh_arrays(source_npz)
        if reference_pose >= len(verts):
            raise LayoutError('layout reference_pose is outside the source NPZ pose range')
        mesh = (faces, labels)

    metadata = layout.get('metadata')
    if metadata is not None:
        if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
            raise LayoutError('layout metadata must be an object with string keys')
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise LayoutError(f'layout metadata must contain finite JSON values: {error}') from error

    markers = layout.get('markers')
    if not isinstance(markers, list) or not markers:
        raise LayoutError('layout markers must be a non-empty array')
    ids = set()
    normalized_markers = []
    valid_leg_labels = set(LEG_BONES[side])
    for index, marker in enumerate(markers):
        prefix = f'markers[{index}]'
        if not isinstance(marker, dict):
            raise LayoutError(f'{prefix} must be an object')
        marker_id = marker.get('id')
        if isinstance(marker_id, bool) or not isinstance(marker_id, numbers.Integral) or marker_id < 0:
            raise LayoutError(f'{prefix}.id must be a non-negative integer')
        marker_id = int(marker_id)
        if marker_id > np.iinfo(np.int64).max:
            raise LayoutError(f'{prefix}.id exceeds the supported 64-bit integer range')
        if marker_id in ids:
            raise LayoutError(f'duplicate marker id {marker_id}')
        ids.add(marker_id)

        label = marker.get('label')
        if not isinstance(label, str) or not label.strip():
            raise LayoutError(f'{prefix}.label must be a non-empty string')
        size = _finite_number(marker.get('size_m'), f'{prefix}.size_m')
        if size <= 0:
            raise LayoutError(f'{prefix}.size_m must be positive')
        rotation = _finite_number(marker.get('rotation_deg'), f'{prefix}.rotation_deg')
        texture = marker.get('texture')
        if not isinstance(texture, str) or not texture.strip() or '\0' in texture:
            raise LayoutError(f'{prefix}.texture must be a non-empty file path')
        if check_textures and not os.path.isfile(_resolve_texture(texture, layout_path)):
            raise LayoutError(f'{prefix}.texture file does not exist: {texture}')

        site_region = marker.get('site_region', 'leg')
        if site_region not in ('leg', 'pelvis'):
            raise LayoutError(f'{prefix}.site_region must be "leg" or "pelvis"')
        allowed_labels = valid_leg_labels if site_region == 'leg' else {0}

        optional_strings = {}
        for field in ('paper_code', 'paper_description', 'placement_method'):
            value = marker.get(field)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise LayoutError(f'{prefix}.{field} must be a non-empty string')
                optional_strings[field] = value.strip()

        paper_table_row = marker.get('paper_table_row')
        if paper_table_row is not None:
            if isinstance(paper_table_row, bool) or not isinstance(paper_table_row, numbers.Integral) or paper_table_row < 1:
                raise LayoutError(f'{prefix}.paper_table_row must be a positive integer')
            paper_table_row = int(paper_table_row)

        proxy_target = marker.get('proxy_target')
        if proxy_target is not None:
            if not isinstance(proxy_target, (list, tuple)) or len(proxy_target) != 3:
                raise LayoutError(f'{prefix}.proxy_target must contain three finite coordinates')
            proxy_target = [
                _finite_number(value, f'{prefix}.proxy_target') for value in proxy_target
            ]
        projection_distance = marker.get('projection_distance_m')
        if projection_distance is not None:
            projection_distance = _finite_number(projection_distance, f'{prefix}.projection_distance_m')
            if projection_distance < 0:
                raise LayoutError(f'{prefix}.projection_distance_m must be non-negative')

        anchor = marker.get('anchor')
        if not isinstance(anchor, dict):
            raise LayoutError(f'{prefix}.anchor must be an object')
        face = anchor.get('face')
        if isinstance(face, bool) or not isinstance(face, numbers.Integral) or face < 0:
            raise LayoutError(f'{prefix}.anchor.face must be a non-negative integer')
        face = int(face)
        bary = anchor.get('bary')
        if not isinstance(bary, (list, tuple)) or len(bary) != 3:
            raise LayoutError(f'{prefix}.anchor.bary must contain three weights')
        bary = [_finite_number(value, f'{prefix}.anchor.bary') for value in bary]
        if any(value < 0 or value > 1 for value in bary) or not math.isclose(sum(bary), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise LayoutError(f'{prefix}.anchor.bary must be non-negative weights summing to one')
        if mesh is not None:
            faces, labels = mesh
            if face >= len(faces):
                raise LayoutError(f'{prefix}.anchor.face {face} is outside the source mesh')
            face_labels = labels[faces[face]]
            if not all(int(value) in allowed_labels for value in face_labels):
                raise LayoutError(f'{prefix}.anchor face is outside its declared {site_region} region')

        normalized_marker = {
            'id': marker_id,
            'label': label.strip(),
            'size_m': size,
            'rotation_deg': rotation,
            'texture': texture,
            'site_region': site_region,
            'anchor': {'face': face, 'bary': bary},
        }
        normalized_marker.update(optional_strings)
        if paper_table_row is not None:
            normalized_marker['paper_table_row'] = paper_table_row
        if proxy_target is not None:
            normalized_marker['proxy_target'] = proxy_target
        if projection_distance is not None:
            normalized_marker['projection_distance_m'] = projection_distance
        normalized_markers.append(normalized_marker)

    normalized_layout = {
        'schema_version': SCHEMA_VERSION,
        'source': normalized_source,
        'markers': normalized_markers,
    }
    if metadata is not None:
        normalized_layout['metadata'] = metadata
    return normalized_layout


def create_layout(source_npz, markers, reference_pose=0, layout_path=None, metadata=None):
    """Create and validate a layout from source data and marker records."""
    layout = {
        'schema_version': SCHEMA_VERSION,
        'source': source_metadata(source_npz, reference_pose),
        'markers': list(markers),
    }
    if metadata is not None:
        layout['metadata'] = metadata
    return validate_layout(layout, source_npz=source_npz, layout_path=layout_path)


def save_layout(path, layout, source_npz=None):
    """Validate and atomically write a UTF-8 JSON layout."""
    path = os.path.abspath(os.fspath(path))
    validated = validate_layout(layout, source_npz=source_npz, layout_path=path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=os.path.dirname(path),
                                         prefix='.marker-layout-', suffix='.tmp', delete=False) as temporary:
            temporary_path = temporary.name
            json.dump(validated, temporary, ensure_ascii=False, indent=2)
            temporary.write('\n')
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
    return validated


def load_layout(path, source_npz=None):
    """Load a JSON layout and reject stale sources or missing texture files."""
    path = os.path.abspath(os.fspath(path))
    try:
        with open(path, encoding='utf-8') as layout_file:
            layout = json.load(layout_file)
    except (OSError, json.JSONDecodeError) as error:
        raise LayoutError(f'cannot read marker layout {path}: {error}') from error
    return validate_layout(layout, source_npz=source_npz, layout_path=path)