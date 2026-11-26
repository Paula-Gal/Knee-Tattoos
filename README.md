# Surgical Tattoos for Knee Arthroplasty

**Topic area:** Computer vision, fiducial markers

## Problem

Robotic-assisted knee surgeries (e.g. MAKO) are more precise than manual surgeries, but require dynamic **tracking of the leg bones via optical markers** which are pinned into the bones. This is necessary for the robotic assistance, but harms the bone.

## Background and motivation

Less invasive would be **visual markers that are applied to the skin**, such as tattoos. However, soft tissue displacement occurs here, and the positions of the markers relative to the bone are not rigid. This makes it tricky to estimate the bone axes.

## Goal

Experimental evaluation of a non-invasive skin-based tracking method for tibia and femur axes.

## Proposed solution

Explore and compare skin-based marker approaches to ground truth data (e.g. using OptiTrack).

## Tasks and objectives

Using an artificial knee, create a prototype which compares skin-based marker precision to ground truth femur and tibia axes.

## Synthetic knee bend

First, in-silico experiment: the [SKEL](https://github.com/MarilynKeller/SKEL) body model bends one knee from 0° to 90°, and the skin and bones are exported to an animated Blender scene.

### How it works

1. **Pose the body.** SKEL takes 46 joint angles. All are 0 (standing) except the chosen knee, which goes from 0° to 90° in 2° steps: 46 poses. *(`scripts/knee_bend_export.py`)*
2. **Generate skin and bones.** For each pose SKEL returns the skin mesh, the bone meshes, the joint centres and each bone's position and orientation (average body shape). *(`scripts/knee_bend_export.py`, model loaded by `scripts/skel_setup.py`)*
3. **Keep one leg.** Only the skin and bones of the chosen leg are kept. The same vertices are kept in every pose, in the same order. *(`scripts/knee_bend_export.py`)*
4. **Save the ground truth.** Everything is saved in a `.npz` file and checked automatically. *(`scripts/knee_bend_export.py`, checked by `scripts/check_export.py`)*
5. **Build the Blender scene.** Blender reads the `.npz`, creates the `Skin` and `Bones` objects, and stores each pose as a shape key: frame n = pose n. *(`scripts/blender_import_knee.py`)*

`run_knee_bend.py` runs all of these in order.

### Setup

Needs Python 3.12, git and [Blender](https://www.blender.org/download/)

```bash
python3.12 -m venv .venv            # Windows: py -3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

SKEL model files: register at https://skel.is.tue.mpg.de, download `skel_models_v1.1.zip`, and put `skel_female.pkl` and `skel_male.pkl` in `models/skel/`. They are licensed for non-commercial research only, so `models/` is git-ignored and each person downloads their own copy.

### Run

From the repo root, with the environment active:

```bash
python run_knee_bend.py male right      # one case: male|female, right|left
python run_knee_bend.py all             # all four cases
```

Each case runs three steps and stops at the first error:

1. `scripts/knee_bend_export.py`: SKEL poses → `output/knee_bend/knee_bend_anim_<gender>_<side>.npz`
2. `scripts/check_export.py`: validates the `.npz` (shapes, NaNs, topology, angles, rigid bones)
3. `scripts/blender_import_knee.py`: `.npz` → `output/knee_bend/knee_bend_<gender>_<side>.blend`

`output/` is git-ignored; everything in it can be regenerated.

`scripts/knee_bend.py` is a separate quick test (one pose → `.obj` meshes) and isn't part of the pipeline.

### Outputs

**Blender scene:** objects `Skin` and `Bones` (femur beige, tibia blue), camera and lights. The animation uses absolute **shape keys**, one per SKEL pose, not an armature. **Frame n = SKEL pose n** (frames 0–45 = knee 0°, 2°, …, 90°), with no interpolation between frames. Vertex order is the same in every frame, so a point on the skin keeps its index throughout the motion.

**Open the scene:** From the terminal:

```bash
open output/knee_bend/knee_bend_male_right.blend                                                           # macOS
```

**Ground truth (`.npz`, one row per pose):**

| Array | Shape | Content |
|---|---|---|
| `knee_deg` | (46,) | knee flexion [deg] |
| `poses` | (46, 46) | full SKEL pose vector [rad], names in `pose_param_names` |
| `skin_verts`, `skin_faces` | (46, V, 3), (F, 3) | skin of the chosen leg [m] |
| `bone_verts`, `bone_faces`, `bone_label` | (46, V, 3), (F, 3), (V,) | bones of the chosen leg, SKEL bone index per vertex |
| `joints` | (46, 24, 3) | all SKEL joint centres [m], names in `joint_names` |
| `bone_transforms` | (46, 24, 4, 4) | rigid bone-to-world transform of every SKEL bone (femur 1/6, tibia 2/7 for right/left) |
| `gender`, `side`, `betas`, `trans` | | model configuration (`betas` = 0: average body) |

Units: metres. Coordinates: SKEL's y up, x towards the subject's left, z forward. The Blender scene uses z up: Blender (x, y, z) = SKEL (x, −z, y).

### Known limitations

The following limitations are based on the SKEL model described by [Keller et al. (2023)](https://doi.org/10.1145/3618381):

- **Simplified knee kinematics.** SKEL uses a simplified hinge joint for the knee (Sec. 5.2). Consequently, it does not explicitly reproduce the full coupled rotations and translations of a real knee joint.
- **Limited soft-tissue realism.** The skin mesh deforms with the skeletal pose using linear blend skinning and pose-dependent corrections. However, muscles, fat, and skin sliding relative to the bones are not explicitly simulated using a physical soft-tissue model.
- **Skin deformation artefacts.** SKEL inherits pose-dependent deformation corrections from SMPL, which the authors acknowledge can produce artefacts in extreme poses (Sec. 5.2).

## Pelvis (static)

The static pelvis case exports the SKEL pelvis bone (bone 0) and the skin vertices dominated by that bone. From the repository root, run:

```bash
python run_pelvis.py male       # or female
python run_pelvis.py all        # both genders
python run_pelvis.py all --force  # overwrite existing Blender scenes
```

The runner exports and checks `output/pelvis/pelvis_<gender>.npz`, then builds `output/pelvis/pelvis_<gender>.blend`. The scene contains static `Skin` and `Bones` objects, a front-facing camera, and lights. This remains the pelvis-only reference-pose export.

## Animated pelvis + leg

To keep the pelvis attached to a bending leg, run the separate pelvis-leg pipeline:

```bash
python run_pelvis_leg.py all                 # male/female, right/left
python run_pelvis_leg.py male left            # one selected case
python run_pelvis_leg.py all --force          # overwrite existing Blender scenes
python run_pelvis_leg.py all --visual         # smooth playback for viewing
```

Each case selects the pelvis and one complete leg from SKEL and uses the same 46-pose, 0–90° knee-flexion motion as `run_knee_bend.py`. It writes `output/pelvis/pelvis_leg_anim_<gender>_<side>.npz` and `output/pelvis/pelvis_leg_<gender>_<side>.blend`; both `Skin` and `Bones` are animated with shape keys. By default, frame `n` is SKEL pose `n`, just like the normal knee scene. The original static pelvis and knee-bend outputs remain separate and unchanged.

Skin markers are deferred. The planned approach is to anchor marker surface points barycentrically to skin triangles and recompute their 3D positions from the deformed skin vertices, preserving exact ground truth without reading positions back from Blender. ArUco, STag, and CCTag support and experiment sweeps are not implemented yet.

## References

- Keller, M., Werling, K., Shin, S., Delp, S., Pujades, S., Liu, C. K., & Black, M. J. (2023). From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans. *ACM Transactions on Graphics*, 42(6), Article 553. https://doi.org/10.1145/3618381. Code: https://github.com/MarilynKeller/SKEL
- Coburn, K., Velhal, M., Goenezen, S., & Liang, H. (2023). Digital image correlation through surface patterning of skins to monitor knee motion. *Biotribology*, 35, 100251.
- Zamora, R., Punt, S. E., Christman-Skieller, C., Yildirim, C., Shapton, J. C., & Conrad 3rd, E. U. (2019). Are skin fiducials comparable to bone fiducials for registration when planning navigation-assisted musculoskeletal tumor resections in a cadaveric simulated tumor model? *Clinical Orthopaedics and Related Research*, 477(12), 2692.
- Tsukada, S., Ogawa, H., Nishino, M., Kurosaka, K., & Hirasawa, N. (2019). Augmented reality-based navigation system applied to tibial bone resection in total knee arthroplasty. *Journal of Experimental Orthopaedics*, 6(1), 44.
- Tsukada, S., Ogawa, H., Nishino, M., Kurosaka, K., & Hirasawa, N. (2021). Augmented reality-assisted femoral bone resection in total knee arthroplasty. *JBJS Open Access*, 6(3), e21-00001.
- Ancillao, A., Aertbeliën, E., & De Schutter, J. (2021). Effect of the soft tissue artifact on marker measurements and on the calculation of the helical axis of the knee during a gait cycle: A study on the CAMS-Knee data set. *Human Movement Science*, 80, 102866.
- Rath, B., Huber, T., Kamper, M., Beckmann, J., Hirschmann, M. T., Hammer, N., & Ullmann, D. (2026). Feasibility and accuracy of intraoperative 3D surface scanning to assess femoral component alignment relative to the native distal femoral articular surface in total knee arthroplasty. *Knee Surgery, Sports Traumatology, Arthroscopy*.
- Agarwal, B. B. (2009). Tattoo: a navigational tool, a lighthouse for biomolecular level precision in surgical endoscopy. Newton's apple or Eve's? *Surgical Endoscopy*, 23(5), 1166-1167.