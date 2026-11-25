"""Export the static SKEL pelvis and its associated skin for Blender."""
import argparse
import os

import numpy as np
import torch

from skel.kin_skel import pose_param_names, skel_joints_name
from skel_setup import load_skel
from skel_mesh import bone_transforms, dominant_bone, submesh


def main():
    parser = argparse.ArgumentParser(description='Export the static SKEL pelvis for Blender.')
    parser.add_argument('--gender', choices=['male', 'female'], default='male')
    args = parser.parse_args()

    skel = load_skel(args.gender)
    pose = torch.zeros(1, skel.num_q_params)
    betas = torch.zeros(1, skel.num_betas)
    trans = torch.zeros(1, 3)

    with torch.no_grad():
        out = skel(pose, betas, trans)

    skin_bone = dominant_bone(skel.skin_weights)
    bone_bone = dominant_bone(skel.skel_weights_rigid)
    skin_vid, skin_faces = submesh(skel.skin_f, skin_bone, [0])
    bone_vid, bone_faces = submesh(skel.skel_f, bone_bone, [0])
    bone_T = bone_transforms(out.joints, out.joints_ori)

    output = f'output/pelvis/pelvis_{args.gender}.npz'
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(
        output,
        poses=pose.numpy(),
        pose_param_names=np.array(pose_param_names),
        gender=np.array(args.gender),
        betas=betas[0].numpy(),
        trans=trans.numpy(),
        skin_verts=out.skin_verts.numpy()[:, skin_vid],
        skin_faces=skin_faces.astype(np.int32),
        skin_vert_ids=skin_vid.astype(np.int32),
        skin_label=skin_bone[skin_vid].astype(np.int32),
        bone_verts=out.skel_verts.numpy()[:, bone_vid],
        bone_faces=bone_faces.astype(np.int32),
        bone_vert_ids=bone_vid.astype(np.int32),
        bone_label=bone_bone[bone_vid].astype(np.int32),
        joints=out.joints.numpy(),
        joint_names=np.array(skel_joints_name),
        bone_transforms=bone_T,
        region=np.array('pelvis'),
        units=np.array('metres; angles in poses = radians'),
        coordinate_system=np.array('SKEL world: y up, x = subject left (right leg is -x), z = forward'),
    )

    print(f'pelvis skin: {len(skin_vid)} verts, {len(skin_faces)} faces')
    print(f'pelvis bone: {len(bone_vid)} verts, {len(bone_faces)} faces')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()