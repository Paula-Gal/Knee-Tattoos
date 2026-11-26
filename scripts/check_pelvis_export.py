"""Validate a pelvis .npz before it is sent to Blender."""
import argparse
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def main():
    parser = argparse.ArgumentParser(description='Validate a SKEL pelvis export.')
    parser.add_argument('npz', help='path to the .npz file')
    parser.add_argument('--recompute', action='store_true', help='re-run SKEL and compare the vertices')
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    failed = []

    def check(name, ok, detail=''):
        print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
        if not ok:
            failed.append(name)

    expected = ['poses', 'pose_param_names', 'gender', 'betas', 'trans', 'region',
                'skin_verts', 'skin_faces', 'skin_vert_ids', 'skin_label',
                'bone_verts', 'bone_faces', 'bone_vert_ids', 'bone_label',
                'joints', 'joint_names', 'bone_transforms', 'units', 'coordinate_system']
    missing = [name for name in expected if name not in data]
    check('all arrays present', not missing, f'missing: {missing}' if missing else f'{len(expected)} arrays')
    if missing:
        sys.exit(1)

    skin_verts, bone_verts = data['skin_verts'], data['bone_verts']
    skin_faces, bone_faces = data['skin_faces'], data['bone_faces']
    skin_ids, bone_ids = data['skin_vert_ids'], data['bone_vert_ids']
    skin_labels, bone_labels = data['skin_label'], data['bone_label']
    shape_ok = (
        data['poses'].shape == (1, 46)
        and data['pose_param_names'].shape == (46,)
        and data['betas'].shape == (10,)
        and data['trans'].shape == (1, 3)
        and skin_verts.ndim == 3 and skin_verts.shape[0] == 1 and skin_verts.shape[2] == 3
        and bone_verts.ndim == 3 and bone_verts.shape[0] == 1 and bone_verts.shape[2] == 3
        and skin_faces.ndim == 2 and skin_faces.shape[1] == 3
        and bone_faces.ndim == 2 and bone_faces.shape[1] == 3
        and skin_ids.ndim == 1 and skin_ids.shape[0] == skin_verts.shape[1]
        and bone_ids.ndim == 1 and bone_ids.shape[0] == bone_verts.shape[1]
        and skin_labels.shape == skin_ids.shape
        and bone_labels.shape == bone_ids.shape
        and data['joints'].shape == (1, 24, 3)
        and data['joint_names'].shape == (24,)
        and data['bone_transforms'].shape == (1, 24, 4, 4)
        and skin_verts.shape[1] > 0 and bone_verts.shape[1] > 0
    )
    skin_count = skin_verts.shape[1] if skin_verts.ndim == 3 else 0
    bone_count = bone_verts.shape[1] if bone_verts.ndim == 3 else 0
    check('shapes consistent (P=1)', shape_ok,
          f'{data["poses"].shape[0] if data["poses"].ndim else 0} poses, '
          f'skin {skin_count} verts, bones {bone_count} verts')

    float_arrays = ['poses', 'betas', 'trans', 'skin_verts', 'bone_verts', 'joints', 'bone_transforms']
    bad_values = [name for name in float_arrays
                  if not np.issubdtype(data[name].dtype, np.number) or not np.isfinite(data[name]).all()]
    check('no NaN / inf', not bad_values, f'bad: {bad_values}' if bad_values else '')

    coords = [array for array in (skin_verts, bone_verts) if array.size]
    coords_numeric = all(np.issubdtype(array.dtype, np.number) and np.isrealobj(array) for array in coords)
    extent = max((np.abs(array).max() for array in coords), default=np.inf) if coords_numeric else np.inf
    check('coordinates plausible (< 2 m from origin)', extent < 2.0, f'max |coord| = {extent:.3f} m')

    for part, vertices, faces, ids in (
        ('skin', skin_verts, skin_faces, skin_ids),
        ('bone', bone_verts, bone_faces, bone_ids),
    ):
        vertex_count = vertices.shape[1] if vertices.ndim == 3 else 0
        face_shape_ok = faces.ndim == 2 and faces.shape[1] == 3 and faces.shape[0] > 0
        integer_faces = np.issubdtype(faces.dtype, np.integer)
        faces_valid = (face_shape_ok and integer_faces and faces.min() >= 0
                       and faces.max() < vertex_count)
        if faces_valid:
            detail = f'{len(faces)} faces, indices 0..{faces.max()}'
        else:
            detail = f'{len(faces)} faces; expected non-empty triangles within 0..{vertex_count - 1}'
        check(f'{part} faces valid', faces_valid, detail)

        ids_valid = (ids.ndim == 1 and len(ids) == vertex_count
                     and np.issubdtype(ids.dtype, np.integer)
                     and (ids >= 0).all() and (np.diff(ids) > 0).all())
        check(f'{part} vertex ids unique and sorted', ids_valid)

    labels_ok = (np.all(skin_labels == 0) and np.all(bone_labels == 0))
    check('all skin and bone labels are pelvis (0)', labels_ok)
    check('all pose parameters are zero', np.all(data['poses'] == 0))
    check('betas and translation are zero', np.all(data['betas'] == 0) and np.all(data['trans'] == 0))
    check('region is pelvis', data['region'].shape == () and str(data['region']) == 'pelvis')
    check('gender is male or female', data['gender'].shape == () and str(data['gender']) in ('male', 'female'))

    geometry_arrays = (skin_verts, bone_verts, data['joints'])
    geometry_ok = (shape_ok and all(np.issubdtype(array.dtype, np.number) and np.isrealobj(array)
                                    and np.isfinite(array).all() for array in geometry_arrays))
    if geometry_ok:
        joint = data['joints'][0, 0]
        bone_min, bone_max = bone_verts[0].min(axis=0), bone_verts[0].max(axis=0)
        joint_inside = np.all(joint >= bone_min - 1e-5) and np.all(joint <= bone_max + 1e-5)
        check('pelvis joint inside bone bounding box', joint_inside)

        bone_center = (bone_min + bone_max) / 2
        skin_min, skin_max = skin_verts[0].min(axis=0), skin_verts[0].max(axis=0)
        center_inside = np.all(bone_center >= skin_min - 1e-5) and np.all(bone_center <= skin_max + 1e-5)
        check('bone bounding-box centre inside skin bounding box', center_inside)
    else:
        check('pelvis joint inside bone bounding box', False, 'invalid geometry or shapes')
        check('bone bounding-box centre inside skin bounding box', False, 'invalid geometry or shapes')

    skin_faces_valid = (skin_faces.ndim == 2 and skin_faces.shape[1:] == (3,)
                        and skin_faces.shape[0] > 0 and np.issubdtype(skin_faces.dtype, np.integer)
                        and skin_faces.min() >= 0 and skin_faces.max() < skin_count)
    if skin_faces_valid and shape_ok:
        edge_pairs = np.concatenate((skin_faces[:, [0, 1]], skin_faces[:, [1, 2]],
                                     skin_faces[:, [2, 0]]), axis=0)
        rows = np.concatenate((edge_pairs[:, 0], edge_pairs[:, 1]))
        cols = np.concatenate((edge_pairs[:, 1], edge_pairs[:, 0]))
        graph = coo_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                           shape=(skin_count, skin_count))
        components = connected_components(graph, directed=False, return_labels=False)
        print(f'[INFO] skin connected components: {components}')
    else:
        print('[INFO] skin connected components: not computed (invalid skin faces)')

    if args.recompute and shape_ok:
        import torch

        from skel_mesh import dominant_bone, submesh
        from skel_setup import load_skel

        gender = str(data['gender'])
        skel = load_skel(gender)
        with torch.no_grad():
            out = skel(torch.as_tensor(data['poses']),
                       torch.as_tensor(data['betas'])[None].expand(1, -1),
                       torch.as_tensor(data['trans']))

        expected_skin_ids, expected_skin_faces = submesh(skel.skin_f, dominant_bone(skel.skin_weights), [0])
        expected_bone_ids, expected_bone_faces = submesh(skel.skel_f, dominant_bone(skel.skel_weights_rigid), [0])
        skin_selection_ok = (np.array_equal(expected_skin_ids, skin_ids)
                             and np.array_equal(expected_skin_faces, skin_faces))
        bone_selection_ok = (np.array_equal(expected_bone_ids, bone_ids)
                             and np.array_equal(expected_bone_faces, bone_faces))
        check('recomputed skin selection matches', skin_selection_ok)
        check('recomputed bone selection matches', bone_selection_ok)

        skin_in_range = skin_ids.size and skin_ids.min() >= 0 and skin_ids.max() < out.skin_verts.shape[1]
        bone_in_range = bone_ids.size and bone_ids.min() >= 0 and bone_ids.max() < out.skel_verts.shape[1]
        if skin_in_range:
            skin_diff = np.abs(out.skin_verts.numpy()[:, skin_ids] - skin_verts).max()
            check('recomputed skin vertices match', skin_diff < 1e-5, f'max diff {skin_diff:.2e} m')
        else:
            check('recomputed skin vertices match', False, 'vertex ids exceed full SKEL skin mesh')
        if bone_in_range:
            bone_diff = np.abs(out.skel_verts.numpy()[:, bone_ids] - bone_verts).max()
            check('recomputed bone vertices match', bone_diff < 1e-5, f'max diff {bone_diff:.2e} m')
        else:
            check('recomputed bone vertices match', False, 'vertex ids exceed full SKEL bone mesh')

    print(f'\n{args.npz}: gender={data["gender"]}, region={data["region"]}, '
          f'1 pose, units: {data["units"]}')
    print(f'coordinate system: {data["coordinate_system"]}')
    if failed:
        print(f'{len(failed)} check(s) FAILED: {failed}')
        sys.exit(1)
    print('All checks passed.')


if __name__ == '__main__':
    main()