"""Fit skin patches around saved centers and evaluate their anchors per pose."""
import json
import math
import numbers

import numpy as np

from layout import LEG_BONES, LayoutError, validate_layout

DEFAULT_GRID_RESOLUTION = 9
DEFAULT_FIT_TOLERANCE_M = 0.005
_DEGENERATE_AREA_EPSILON = 1e-14


class AnchorError(ValueError):
    """Raised when a marker patch cannot be safely attached to the skin."""


def _validated_mesh(skin_verts, skin_faces, skin_label):
    verts = np.asarray(skin_verts, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int64)
    labels = np.asarray(skin_label, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
        raise AnchorError('skin_verts must be finite with shape (vertices, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise AnchorError('skin_faces must be a non-empty (faces, 3) triangle array')
    if faces.min() < 0 or faces.max() >= len(verts):
        raise AnchorError('skin_faces contains an invalid vertex index')
    if labels.shape != (len(verts),):
        raise AnchorError('skin_label must match the skin vertex count')
    return verts, faces, labels


def _triangle_normals(triangles):
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _closest_points_on_triangles(query, triangles):
    """Return closest points and barycentric coordinates for one query and many triangles."""
    count = len(triangles)
    point = np.asarray(query, dtype=np.float64)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum('ij,ij->i', ab, ap)
    d2 = np.einsum('ij,ij->i', ac, ap)
    bp = point - b
    d3 = np.einsum('ij,ij->i', ab, bp)
    d4 = np.einsum('ij,ij->i', ac, bp)
    cp = point - c
    d5 = np.einsum('ij,ij->i', ab, cp)
    d6 = np.einsum('ij,ij->i', ac, cp)

    bary = np.zeros((count, 3), dtype=np.float64)
    closest = np.empty((count, 3), dtype=np.float64)
    remaining = np.ones(count, dtype=bool)

    def assign(mask, weights):
        selected = mask & remaining
        if np.any(selected):
            bary[selected] = weights[selected]
            closest[selected] = np.einsum('ij,ijk->ik', bary[selected], triangles[selected])
            remaining[selected] = False

    ones = np.ones(count, dtype=np.float64)
    zeros = np.zeros(count, dtype=np.float64)
    assign((d1 <= 0) & (d2 <= 0), np.column_stack((ones, zeros, zeros)))
    assign((d3 >= 0) & (d4 <= d3), np.column_stack((zeros, ones, zeros)))

    vc = d1 * d4 - d3 * d2
    denominator = d1 - d3
    ab_fraction = np.divide(d1, denominator, out=np.zeros_like(d1), where=np.abs(denominator) > 1e-30)
    assign((vc <= 0) & (d1 >= 0) & (d3 <= 0),
           np.column_stack((1 - ab_fraction, ab_fraction, zeros)))

    assign((d6 >= 0) & (d5 <= d6), np.column_stack((zeros, zeros, ones)))

    vb = d5 * d2 - d1 * d6
    denominator = d2 - d6
    ac_fraction = np.divide(d2, denominator, out=np.zeros_like(d2), where=np.abs(denominator) > 1e-30)
    assign((vb <= 0) & (d2 >= 0) & (d6 <= 0),
           np.column_stack((1 - ac_fraction, zeros, ac_fraction)))

    va = d3 * d6 - d5 * d4
    denominator = (d4 - d3) + (d5 - d6)
    bc_fraction = np.divide(d4 - d3, denominator, out=np.zeros_like(d4), where=np.abs(denominator) > 1e-30)
    assign((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
           np.column_stack((zeros, 1 - bc_fraction, bc_fraction)))

    denominator = va + vb + vc
    if np.any(remaining):
        if np.any(np.abs(denominator[remaining]) <= 1e-30):
            raise AnchorError('cannot project onto degenerate skin triangles')
        second = vb / denominator
        third = vc / denominator
        assign(remaining, np.column_stack((1 - second - third, second, third)))
    return closest, bary


def _nearest_triangle(query, triangles, face_ids):
    closest, bary = _closest_points_on_triangles(query, triangles)
    distance_squared = np.einsum('ij,ij->i', closest - query, closest - query)
    nearest = int(np.argmin(distance_squared))
    return int(face_ids[nearest]), bary[nearest], closest[nearest], math.sqrt(float(distance_squared[nearest]))


def _region_labels(side, site_region):
    if site_region == 'pelvis':
        return (0,)
    if site_region == 'leg' and side in LEG_BONES:
        return LEG_BONES[side]
    raise AnchorError('site_region must be pelvis or the selected leg')


def project_point_to_skin(point, skin_verts, skin_faces, skin_label, side, site_region='leg'):
    """Project a 3D proxy point to the nearest non-degenerate face in its region."""
    verts, faces, labels = _validated_mesh(skin_verts, skin_faces, skin_label)
    query = np.asarray(point, dtype=np.float64)
    if query.shape != (3,) or not np.isfinite(query).all():
        raise AnchorError('projection point must contain three finite coordinates')
    if side not in LEG_BONES:
        raise AnchorError('side must be right or left')
    allowed_labels = _region_labels(side, site_region)
    valid_faces = np.all(np.isin(labels[faces], allowed_labels), axis=1)
    triangles = verts[faces]
    areas = np.linalg.norm(_triangle_normals(triangles), axis=1)
    candidate_ids = np.flatnonzero(valid_faces & (areas > _DEGENERATE_AREA_EPSILON))
    if not len(candidate_ids):
        raise AnchorError(f'source mesh has no non-degenerate faces in the {site_region} region')
    face, bary, projected, distance = _nearest_triangle(query, triangles[candidate_ids], candidate_ids)
    return face, bary, projected, distance


def _valid_barycentric(value, name):
    bary = np.asarray(value, dtype=np.float64)
    if bary.shape[-1:] != (3,) or not np.isfinite(bary).all():
        raise AnchorError(f'{name} must contain three finite barycentric weights')
    if np.any(bary < 0) or np.any(bary > 1) or not np.allclose(bary.sum(axis=-1), 1.0, rtol=0.0, atol=1e-6):
        raise AnchorError(f'{name} must be non-negative barycentric weights summing to one')
    return bary


def fit_marker_patch(skin_verts, skin_faces, skin_label, side, marker,
                     grid_resolution=DEFAULT_GRID_RESOLUTION,
                     fit_tolerance_m=DEFAULT_FIT_TOLERANCE_M):
    """Project a square tangent-plane grid to the selected-leg reference surface."""
    verts, faces, labels = _validated_mesh(skin_verts, skin_faces, skin_label)
    if side not in LEG_BONES:
        raise AnchorError('side must be right or left')
    if isinstance(grid_resolution, bool) or not isinstance(grid_resolution, numbers.Integral):
        raise AnchorError('grid_resolution must be an odd integer of at least 3')
    grid_resolution = int(grid_resolution)
    if grid_resolution < 3 or grid_resolution % 2 == 0:
        raise AnchorError('grid_resolution must be an odd integer of at least 3')
    if isinstance(fit_tolerance_m, bool) or not isinstance(fit_tolerance_m, numbers.Real):
        raise AnchorError('fit_tolerance_m must be a positive finite number')
    fit_tolerance_m = float(fit_tolerance_m)
    if not math.isfinite(fit_tolerance_m) or fit_tolerance_m <= 0:
        raise AnchorError('fit_tolerance_m must be a positive finite number')

    try:
        size = float(marker['size_m'])
        rotation = float(marker.get('rotation_deg', 0.0))
        anchor = marker['anchor']
        raw_face = anchor['face']
        if isinstance(raw_face, bool) or not isinstance(raw_face, numbers.Integral):
            raise AnchorError('center anchor face must be an integer')
        center_face = int(raw_face)
        center_bary = _valid_barycentric(anchor['bary'], 'center anchor barycentric weights')
    except AnchorError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise AnchorError(f'invalid marker anchor or size: {error}') from error
    if not math.isfinite(size) or size <= 0:
        raise AnchorError('marker size_m must be a positive finite number')
    if not math.isfinite(rotation):
        raise AnchorError('marker rotation_deg must be finite')
    if not 0 <= center_face < len(faces):
        raise AnchorError('center anchor face is outside the source mesh')
    center_bary = center_bary.reshape(3)

    site_region = marker.get('site_region', 'leg')
    allowed_labels = _region_labels(side, site_region)
    valid_faces = np.all(np.isin(labels[faces], allowed_labels), axis=1)
    if not valid_faces[center_face]:
        raise AnchorError(f'center anchor face is not entirely on the declared {site_region} region')
    all_triangles = verts[faces]
    all_normals = _triangle_normals(all_triangles)
    all_areas = np.linalg.norm(all_normals, axis=1)
    center_normal = all_normals[center_face]
    center_normal_length = all_areas[center_face]
    if center_normal_length <= _DEGENERATE_AREA_EPSILON:
        raise AnchorError('center anchor face is degenerate')
    center_normal /= center_normal_length

    candidate_ids = np.flatnonzero(valid_faces & (all_areas > _DEGENERATE_AREA_EPSILON))
    if not len(candidate_ids):
        raise AnchorError(f'declared {site_region} region has no non-degenerate skin faces')
    candidate_triangles = all_triangles[candidate_ids]
    boundary_ids = np.flatnonzero(~valid_faces & (all_areas > _DEGENERATE_AREA_EPSILON))
    boundary_triangles = all_triangles[boundary_ids]

    triangle = all_triangles[center_face]
    center = center_bary @ triangle
    tangent = triangle[1] - triangle[0]
    tangent -= np.dot(tangent, center_normal) * center_normal
    tangent_length = np.linalg.norm(tangent)
    if tangent_length <= 1e-12:
        raise AnchorError('center anchor face has a degenerate tangent')
    tangent /= tangent_length
    bitangent = np.cross(center_normal, tangent)
    angle = math.radians(rotation)
    rotated_tangent = math.cos(angle) * tangent + math.sin(angle) * bitangent
    rotated_bitangent = -math.sin(angle) * tangent + math.cos(angle) * bitangent

    coordinates = np.linspace(-size / 2, size / 2, grid_resolution)
    middle = grid_resolution // 2
    surface_face = np.empty((grid_resolution, grid_resolution), dtype=np.int32)
    surface_bary = np.empty((grid_resolution, grid_resolution, 3), dtype=np.float64)
    grid_points = np.empty((grid_resolution, grid_resolution, 3), dtype=np.float64)
    for row, y in enumerate(coordinates):
        for column, x in enumerate(coordinates):
            if row == middle and column == middle:
                face_id, bary, projected = center_face, center_bary, center
            else:
                query = center + x * rotated_tangent + y * rotated_bitangent
                face_id, bary, projected, distance = _nearest_triangle(query, candidate_triangles, candidate_ids)
                if len(boundary_ids):
                    _boundary_face, _boundary_bary, _boundary_point, boundary_distance = _nearest_triangle(
                        query, boundary_triangles, boundary_ids,
                    )
                    if boundary_distance <= distance + 1e-9:
                        raise AnchorError(f'marker patch crosses or touches the declared {site_region} region boundary')
                if distance > fit_tolerance_m:
                    raise AnchorError(
                        f'marker grid sample is {distance:.4g} m from declared {site_region} skin '
                        f'(fit tolerance {fit_tolerance_m:.4g} m)'
                    )
            surface_face[row, column] = face_id
            surface_bary[row, column] = bary
            grid_points[row, column] = projected

    for row in range(grid_resolution - 1):
        for column in range(grid_resolution - 1):
            edge_x = grid_points[row, column + 1] - grid_points[row, column]
            edge_y = grid_points[row + 1, column] - grid_points[row, column]
            cell_normal = np.cross(edge_x, edge_y)
            cell_area = np.linalg.norm(cell_normal)
            if cell_area <= 1e-12 or np.dot(cell_normal, center_normal) <= 1e-8 * cell_area:
                raise AnchorError('fitted marker patch folds or contains a degenerate grid cell')

    return surface_face, surface_bary.astype(np.float32)


def evaluate_barycentric(skin_verts, skin_faces, face_indices, barycentric):
    """Evaluate one or more face+barycentric anchors against every pose."""
    verts = np.asarray(skin_verts, dtype=np.float64)
    faces = np.asarray(skin_faces, dtype=np.int64)
    anchor_faces = np.asarray(face_indices)
    bary = _valid_barycentric(barycentric, 'barycentric weights')
    if verts.ndim != 3 or verts.shape[2] != 3 or not np.isfinite(verts).all():
        raise AnchorError('skin_verts must be finite with shape (poses, vertices, 3)')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise AnchorError('skin_faces must be a non-empty triangle array')
    if faces.min() < 0 or faces.max() >= verts.shape[1]:
        raise AnchorError('skin_faces contains an invalid vertex index')
    if anchor_faces.dtype.kind not in 'iu' or np.any(anchor_faces < 0) or np.any(anchor_faces >= len(faces)):
        raise AnchorError('anchor face indices are invalid')
    if bary.shape != anchor_faces.shape + (3,):
        raise AnchorError('barycentric array shape must match face indices plus a final dimension of 3')
    triangles = verts[:, faces[anchor_faces]]
    return np.sum(triangles * bary[None, ..., :, None], axis=-2)


def make_marker_geometry(source_npz, layout, grid_resolution=DEFAULT_GRID_RESOLUTION,
                         fit_tolerance_m=DEFAULT_FIT_TOLERANCE_M, layout_path=None):
    """Fit all layout markers and return per-pose grids and ground-truth geometry."""
    try:
        layout = validate_layout(layout, source_npz=source_npz, layout_path=layout_path)
    except LayoutError as error:
        raise AnchorError(str(error)) from error
    skin_verts = np.asarray(source_npz['skin_verts'], dtype=np.float64)
    skin_faces = np.asarray(source_npz['skin_faces'], dtype=np.int64)
    skin_label = np.asarray(source_npz['skin_label'], dtype=np.int64)
    reference_pose = layout['source']['reference_pose']
    markers = layout['markers']
    fitted = []
    for marker in markers:
        try:
            fitted.append(fit_marker_patch(
                skin_verts[reference_pose], skin_faces, skin_label, layout['source']['side'], marker,
                grid_resolution=grid_resolution, fit_tolerance_m=fit_tolerance_m,
            ))
        except AnchorError as error:
            raise AnchorError(f'marker {marker.get("paper_code", marker["label"])}: {error}') from error
    surface_face = np.stack([item[0] for item in fitted]).astype(np.int32)
    surface_bary = np.stack([item[1] for item in fitted]).astype(np.float32)
    centre_face = np.asarray([marker['anchor']['face'] for marker in markers], dtype=np.int32)
    centre_bary = np.asarray([marker['anchor']['bary'] for marker in markers], dtype=np.float32)
    grid_verts = evaluate_barycentric(skin_verts, skin_faces, surface_face, surface_bary)
    centre = evaluate_barycentric(skin_verts, skin_faces, centre_face, centre_bary)
    corners = np.stack((grid_verts[:, :, 0, 0], grid_verts[:, :, 0, -1],
                        grid_verts[:, :, -1, -1], grid_verts[:, :, -1, 0]), axis=2)

    middle = grid_resolution // 2
    tangent_x = grid_verts[:, :, middle, middle + 1] - grid_verts[:, :, middle, middle - 1]
    tangent_y = grid_verts[:, :, middle + 1, middle] - grid_verts[:, :, middle - 1, middle]
    normal = np.cross(tangent_x, tangent_y)
    normal_length = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(normal_length <= 1e-12):
        raise AnchorError('marker grid has a degenerate normal in one or more poses')
    normal /= normal_length

    anchor_triangles = skin_verts[:, skin_faces[centre_face]]
    face_normals = _triangle_normals(anchor_triangles.reshape(-1, 3, 3)).reshape(len(skin_verts), len(markers), 3)
    face_normal_length = np.linalg.norm(face_normals, axis=-1, keepdims=True)
    if np.any(face_normal_length <= _DEGENERATE_AREA_EPSILON):
        raise AnchorError('center anchor face becomes degenerate in one or more poses')
    face_normals /= face_normal_length
    if np.any(np.sum(normal * face_normals, axis=-1) <= 0):
        raise AnchorError('marker patch normal flips relative to its center face in one or more poses')

    return {
        'marker_id': np.asarray([marker['id'] for marker in markers], dtype=np.int64),
        'marker_label': np.asarray([marker['label'] for marker in markers]),
        'paper_code': np.asarray([marker.get('paper_code', '') for marker in markers]),
        'paper_description': np.asarray([marker.get('paper_description', '') for marker in markers]),
        'paper_table_row': np.asarray([marker.get('paper_table_row', 0) for marker in markers], dtype=np.int32),
        'site_region': np.asarray([marker['site_region'] for marker in markers]),
        'placement_method': np.asarray([marker.get('placement_method', 'manual_surface_pick') for marker in markers]),
        'projection_distance_m': np.asarray([
            marker.get('projection_distance_m', 0.0) for marker in markers
        ], dtype=np.float32),
        'size_m': np.asarray([marker['size_m'] for marker in markers], dtype=np.float32),
        'rotation_deg': np.asarray([marker['rotation_deg'] for marker in markers], dtype=np.float32),
        'texture': np.asarray([marker['texture'] for marker in markers]),
        'centre_face': centre_face,
        'centre_bary': centre_bary,
        'surface_face': surface_face,
        'surface_bary': surface_bary,
        'grid_verts': grid_verts.astype(np.float32),
        'centre': centre.astype(np.float32),
        'corners': corners.astype(np.float32),
        'normal': normal.astype(np.float32),
        'pose_index': np.arange(len(skin_verts), dtype=np.int32),
        'grid_resolution': np.asarray(grid_resolution, dtype=np.int32),
        'fit_tolerance_m': np.asarray(fit_tolerance_m, dtype=np.float32),
        'reference_pose': np.asarray(reference_pose, dtype=np.int32),
        'gender': np.asarray(layout['source']['gender']),
        'side': np.asarray(layout['source']['side']),
        'region': np.asarray(layout['source']['region']),
        'mesh_sha256': np.asarray(layout['source']['mesh_sha256']),
        'layout_metadata_json': np.asarray(json.dumps(layout.get('metadata', {}), allow_nan=False, sort_keys=True)),
        'units': np.asarray(str(np.asarray(source_npz.get('units', 'metres')).item())),
        'coordinate_system': np.asarray(str(np.asarray(source_npz.get('coordinate_system', 'SKEL world')).item())),
    }