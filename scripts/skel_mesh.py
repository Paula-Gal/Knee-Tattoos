import numpy as np


def dominant_bone(weights):
    """For each vertex, the index of the bone that moves it most: (V,)."""
    w = weights.to_dense() if weights.is_sparse else weights
    return w.argmax(dim=1).numpy()


def submesh(faces, vert_bone, bones):
    """Keep faces whose 3 vertices all belong to `bones`, and renumber them.

    Returns (vert_ids, new_faces):
        vert_ids   original indices of the kept vertices, in order
        new_faces  faces using the new numbering 0..len(vert_ids)-1
    """
    faces = faces.numpy()
    keep_f = np.isin(vert_bone, bones)[faces].all(axis=1)
    faces = faces[keep_f]
    vert_ids = np.unique(faces)
    remap = -np.ones(len(vert_bone), dtype=np.int64)
    remap[vert_ids] = np.arange(len(vert_ids))
    return vert_ids, remap[faces]


def bone_transforms(joints, joints_ori):
    """Build rigid bone-to-world transforms from SKEL joints and orientations."""
    transforms = np.zeros((joints.shape[0], joints.shape[1], 4, 4), dtype=np.float32)
    transforms[..., :3, :3] = joints_ori.numpy()
    transforms[..., :3, 3] = joints.numpy()
    transforms[..., 3, 3] = 1.0
    return transforms