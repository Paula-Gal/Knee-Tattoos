"""
Check a .npz made by knee_bend_export.py before it goes to Blender.
Normally run by run_knee_bend.py. Exits with an error if a check fails.

--recompute also re-runs SKEL and compares the vertices (needs the model files).
"""
import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser(description='Validate a SKEL knee-bend export.')
parser.add_argument('npz', help='path to the .npz file')
parser.add_argument('--recompute', action='store_true', help='re-run SKEL and compare the vertices')
args = parser.parse_args()

d = np.load(args.npz)
failed = []

side = str(d['side']) if 'side' in d.files else 'right'  # files made before --side existed are right-leg
LEG_BONES = {'right': [1, 2, 3, 4, 5], 'left': [6, 7, 8, 9, 10]}
FEMUR, TIBIA = (1, 2) if side == 'right' else (6, 7)  # SKEL bone indices
HIP, KNEE, ANKLE = FEMUR, TIBIA, TIBIA + 1  # joint index = index of the bone that starts there
KNEE_PARAM = f'knee_angle_{side[0]}'


def check(name, ok, detail=''):
    """Print PASS/FAIL and remember failures."""
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  ({detail})' if detail else ''))
    if not ok:
        failed.append(name)


# 1. All expected arrays are there
expected = ['knee_deg', 'poses', 'pose_param_names', 'gender', 'betas', 'trans',
            'skin_verts', 'skin_faces', 'skin_vert_ids', 'skin_label',
            'bone_verts', 'bone_faces', 'bone_vert_ids', 'bone_label',
            'joints', 'joint_names', 'bone_transforms', 'units', 'coordinate_system']
missing = [k for k in expected if k not in d.files]
check('all arrays present', not missing, f'missing: {missing}' if missing else f'{len(expected)} arrays')
if missing:
    sys.exit(1)

# 2. Shapes agree
P = len(d['knee_deg'])
Vs, Vb = d['skin_verts'].shape[1], d['bone_verts'].shape[1]
shapes_ok = (d['poses'].shape == (P, 46) and d['skin_verts'].shape == (P, Vs, 3)
             and d['bone_verts'].shape == (P, Vb, 3) and d['joints'].shape[0] == P
             and d['bone_transforms'].shape[0] == P and d['trans'].shape == (P, 3)
             and d['skin_label'].shape == (Vs,) and d['bone_label'].shape == (Vb,))
check('shapes consistent', shapes_ok, f'{P} poses, skin {Vs} verts, bones {Vb} verts')

# 3. No invalid values
floats = ['knee_deg', 'poses', 'skin_verts', 'bone_verts', 'joints', 'bone_transforms']
bad = [k for k in floats if not np.isfinite(d[k]).all()]
check('no NaN / inf', not bad, f'bad: {bad}' if bad else '')
extent = max(np.abs(d['skin_verts']).max(), np.abs(d['bone_verts']).max())
check('coordinates plausible (< 2 m from origin)', extent < 2.0, f'max |coord| = {extent:.3f} m')

# 4. Faces and vertex ids are valid
for part, V in (('skin', Vs), ('bone', Vb)):
    f, ids = d[f'{part}_faces'], d[f'{part}_vert_ids']
    check(f'{part} faces valid', f.min() >= 0 and f.max() < V, f'{len(f)} faces, indices 0..{f.max()}')
    check(f'{part} vertex ids unique and sorted', len(ids) == V and (np.diff(ids) > 0).all())

# 5. Only the chosen knee moves, by knee_deg
names = list(d['pose_param_names'])
k = names.index(KNEE_PARAM)
check(f'{KNEE_PARAM} column == knee_deg', np.allclose(np.degrees(d['poses'][:, k]), d['knee_deg'], atol=1e-3),
      f'{d["knee_deg"][0]:g} -> {d["knee_deg"][-1]:g} deg')
others = np.delete(d['poses'], k, axis=1)
check('all other pose columns are 0', np.abs(others).max() == 0)

# 5b. Pelvis-attached exports contain only the pelvis and selected leg.
if 'region' in d.files:
    region = str(d['region'])
    expected_region = f'pelvis+{side}_leg'
    check('region matches pelvis and selected side', d['region'].shape == () and region == expected_region,
          f'{region}, expected {expected_region}')
    allowed_labels = [0] + LEG_BONES[side]
    skin_labels_ok = (np.isin(d['skin_label'], allowed_labels).all()
                      and np.any(d['skin_label'] == 0))
    bone_labels_ok = (np.isin(d['bone_label'], allowed_labels).all()
                      and {0, *LEG_BONES[side]}.issubset(set(d['bone_label'].tolist())))
    check('skin labels are pelvis or selected leg', skin_labels_ok)
    check('bone labels contain pelvis and selected leg only', bone_labels_ok)

# 6. Bone lengths stay constant
J = d['joints']
femur_len = np.linalg.norm(J[:, KNEE] - J[:, HIP], axis=1)
tibia_len = np.linalg.norm(J[:, ANKLE] - J[:, KNEE], axis=1)
check('femur length constant', np.ptp(femur_len) < 1e-4, f'{femur_len.mean():.4f} m')
check('tibia length constant', np.ptp(tibia_len) < 1e-4, f'{tibia_len.mean():.4f} m')

# 7. bone_transforms match the bone mesh (tibia doesn't move in its own frame)
T = d['bone_transforms'][:, TIBIA]  # (P, 4, 4)
tib = d['bone_verts'][:, d['bone_label'] == TIBIA]  # (P, n, 3)
local = np.einsum('pji,pnj->pni', T[:, :3, :3], tib - T[:, None, :3, 3])  # world -> tibia frame
drift = np.abs(local - local[0]).max()
check('tibia mesh rigid in its bone frame', drift < 1e-4, f'max drift {drift * 1000:.4f} mm')

# 8. Tibia rotates relative to femur by the knee angle
Rf, Rt = d['bone_transforms'][:, FEMUR, :3, :3], d['bone_transforms'][:, TIBIA, :3, :3]
rel = np.einsum('pji,pjk->pik', Rf, Rt)  # tibia orientation seen from the femur
rel0 = rel[0]
rel = np.einsum('ji,pjk->pik', rel0, rel)  # rotation since pose 0
cos = (np.trace(rel, axis1=1, axis2=2) - 1) / 2
sin = np.linalg.norm([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], axis=0) / 2
ang = np.degrees(np.arctan2(sin, cos))  # robust near 0 deg, unlike arccos
check('femur-tibia rotation == knee_deg', np.allclose(ang, d['knee_deg'], atol=0.05),
      f'max error {np.abs(ang - d["knee_deg"]).max():.4f} deg')

# 9. Skin moves with the bones
skin_tib = d['skin_verts'][:, d['skin_label'] == TIBIA].mean(axis=1)  # centre of shin skin, per pose
skin_move = np.linalg.norm(skin_tib[-1] - skin_tib[0])
ankle_move = np.linalg.norm(J[-1, ANKLE] - J[0, ANKLE])
check('skin moves with the bones', skin_move > 0.1 and ankle_move > 0.1,
      f'shin skin moved {skin_move:.3f} m, ankle {ankle_move:.3f} m')

# 10. Optional: re-running SKEL gives the same result
if args.recompute:
    import torch

    sys.path.insert(0, 'scripts')
    from skel_mesh import dominant_bone, submesh
    from skel_setup import load_skel

    skel = load_skel(str(d['gender']))
    with torch.no_grad():
        out = skel(torch.tensor(d['poses']), torch.tensor(d['betas'])[None].expand(P, -1), torch.tensor(d['trans']))
    ds = np.abs(out.skin_verts.numpy()[:, d['skin_vert_ids']] - d['skin_verts']).max()
    db = np.abs(out.skel_verts.numpy()[:, d['bone_vert_ids']] - d['bone_verts']).max()
    check('recomputed skin matches', ds < 1e-5, f'max diff {ds:.2e} m')
    check('recomputed bones match', db < 1e-5, f'max diff {db:.2e} m')
    selected_bones = [0] + LEG_BONES[side] if 'region' in d.files else LEG_BONES[side]
    skin_ids, skin_faces = submesh(skel.skin_f, dominant_bone(skel.skin_weights), selected_bones)
    bone_ids, bone_faces = submesh(skel.skel_f, dominant_bone(skel.skel_weights_rigid), selected_bones)
    check('recomputed skin selection matches', np.array_equal(skin_ids, d['skin_vert_ids'])
          and np.array_equal(skin_faces, d['skin_faces']))
    check('recomputed bone selection matches', np.array_equal(bone_ids, d['bone_vert_ids'])
          and np.array_equal(bone_faces, d['bone_faces']))

print(f'\n{args.npz}: gender={d["gender"]}, side={side}, {P} poses, units: {d["units"]}')
print(f'coordinate system: {d["coordinate_system"]}')
if failed:
    print(f'{len(failed)} check(s) FAILED: {failed}')
    sys.exit(1)
print('All checks passed.')