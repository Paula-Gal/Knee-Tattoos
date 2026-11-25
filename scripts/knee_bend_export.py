"""Export the SKEL knee flexion (0 -> 90 deg) of one leg for Blender."""
import argparse
import os

import numpy as np
import torch

from skel.kin_skel import pose_param_names, skel_joints_name
from skel_setup import load_skel
from skel_mesh import bone_transforms, dominant_bone, submesh

N_POSES = 46  # 0, 2, 4, ..., 90 degrees
MAX_DEG = 90
# SKEL bone indices of each leg: femur, tibia, talus, calcn, toes
LEG_BONES = {'right': [1, 2, 3, 4, 5], 'left': [6, 7, 8, 9, 10]}


def export_knee_bend(gender='male', side='right', output_path=None, include_pelvis=False):
    """Export the existing knee-bend motion, optionally retaining the pelvis too."""
    bones = LEG_BONES[side]
    selected_bones = ([0] + bones) if include_pelvis else bones
    knee_param = f'knee_angle_{side[0]}'  # SKEL pose names end in _r or _l
    ankle = bones[2]  # the ankle joint is where the talus starts
    if output_path is None:
        output_path = f'output/knee_bend/knee_bend_anim_{gender}_{side}.npz'

    skel = load_skel(gender)

    # Knee angle of every pose, evenly spaced from 0 to 90 degrees: (46,)
    knee_deg = np.linspace(0, MAX_DEG, N_POSES)

    # One row per pose: (46, 46). Only the chosen knee column is non-zero.
    pose = torch.zeros(N_POSES, skel.num_q_params)
    pose[:, pose_param_names.index(knee_param)] = torch.tensor(np.radians(knee_deg), dtype=torch.float32)
    betas = torch.zeros(N_POSES, skel.num_betas)
    trans = torch.zeros(N_POSES, 3)

    with torch.no_grad():
        out = skel(pose, betas, trans)

    print('knee angles:', knee_deg[:4], '...', knee_deg[-1])
    print('skin_verts:', out.skin_verts.shape)
    print('ankle first / last pose:', out.joints[0, ankle].numpy().round(3), out.joints[-1, ankle].numpy().round(3))

    # Skinning weights are fixed, so the same vertices and order are used for every pose.
    skin_bone = dominant_bone(skel.skin_weights)
    bone_bone = dominant_bone(skel.skel_weights_rigid)
    skin_vid, skin_f = submesh(skel.skin_f, skin_bone, selected_bones)
    bone_vid, bone_f = submesh(skel.skel_f, bone_bone, selected_bones)

    region = f'pelvis+{side}_leg' if include_pelvis else None
    region_prefix = 'pelvis+' if include_pelvis else ''
    print(f'{region_prefix}{side}-leg skin:', len(skin_vid), 'verts,', len(skin_f), 'faces')
    print(f'{region_prefix}{side}-leg bone:', len(bone_vid), 'verts,', len(bone_f), 'faces')

    bone_T = bone_transforms(out.joints, out.joints_ori)

    arrays = dict(
        knee_deg=knee_deg.astype(np.float32),
        poses=pose.numpy(),
        pose_param_names=np.array(pose_param_names),
        gender=np.array(gender),
        side=np.array(side),
        betas=betas[0].numpy(),
        trans=trans.numpy(),
        skin_verts=out.skin_verts.numpy()[:, skin_vid],
        skin_faces=skin_f.astype(np.int32),
        skin_vert_ids=skin_vid.astype(np.int32),
        skin_label=skin_bone[skin_vid].astype(np.int32),
        bone_verts=out.skel_verts.numpy()[:, bone_vid],
        bone_faces=bone_f.astype(np.int32),
        bone_vert_ids=bone_vid.astype(np.int32),
        bone_label=bone_bone[bone_vid].astype(np.int32),
        joints=out.joints.numpy(),
        joint_names=np.array(skel_joints_name),
        bone_transforms=bone_T,
        units=np.array('metres; angles in poses = radians, knee_deg = degrees'),
        coordinate_system=np.array('SKEL world: y up, x = subject left (right leg is -x), z = forward'),
    )
    if region is not None:
        arrays['region'] = np.array(region)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Export the SKEL knee flexion of one leg for Blender.')
    parser.add_argument('--gender', choices=['male', 'female'], default='male')
    parser.add_argument('--side', choices=['right', 'left'], default='right')
    parser.add_argument('--output', help='override the default output path')
    args = parser.parse_args()
    export_knee_bend(args.gender, args.side, output_path=args.output)


if __name__ == '__main__':
    main()
