# Marker tracking

Video of a body with markers in, marker positions (camera frame, metres) out.

## Place markers on a pelvis-leg mesh

The picker authors a reusable, source-bound JSON layout. It uses Blender's interactive viewport, so do not pass `-b` when launching it.

```bash
python run_pelvis_leg.py male right
blender --python marker_tracking/blender/place_markers.py -- \
  --npz output/pelvis/pelvis_leg_anim_male_right.npz \
  --output output/pelvis/layout_shin_experiment_male_right.json --pose 0
```

In the `Markers` sidebar, choose a reference pose and click **Set Reference Pose**, then click **Pick Marker Center** and select a visible leg-skin face. Pelvis faces and mixed pelvis/leg faces are rejected for manual picks. Set the marker ID, label, size in metres, rotation, and an existing texture image; use **Preview Marker** to see a textured tangent-plane preview and **Add Marker to Layout** to keep it. Add more markers, then save the JSON layout. The preview is for authoring; the generated patch is fitted to the skin surface.

The saved face index is local to `skin_faces`, with barycentric weights in that face's vertex order. The layout stores gender, side, region, reference pose, and a SHA-256 fingerprint of the complete exported mesh/pose data. Relative texture paths resolve from the layout file. Layouts are not interchangeable between different gender/side exports.

## Generate the MiKneeSoTA marker set

The paper's Table 1 defines 30 retroreflective marker sites in 15 bilateral pairs; it does not specify their physical diameter. The generator uses a configurable 14 mm default. Each one-leg-plus-pelvis model receives 17 markers: bilateral LASI/RASI and LPSI/RPSI on pelvis-labeled skin, plus the 13 markers for its selected leg: THI, KNE, TIB, ANK, HEE, TOE, MFE, MTI, KNM, LTC, MTC, KNA, and MED. The hip joint centers (LFEP/RFEP) illustrated in Fig. 1 are not physical markers and are not added as decals.

The paper gives anatomical descriptions and schematic views, not coordinates on these SKEL meshes. Therefore the generator places **provisional anatomical proxies** using the exported joints, pelvis bounds, and side-specific directions, then projects each target onto the nearest triangle in its permitted pelvis/leg region. These are not exact palpated clinical locations; inspect the layouts and adjust them in Blender if exact sites are required. The marker size is configurable, and the generated transparent disc texture is only a visualization, not a model of retroreflective optics.

Ensure all four pelvis-leg NPZ exports exist, then run:

```bash
python scripts/generate_mikneesota_markers.py
```

This writes four source-bound layout files (`mikneesota_layout_<gender>_<side>.json`), four animated marker geometry files (`markers_mikneesota_<gender>_<side>.npz`), and `mikneesota_marker_disc.png` under `output/pelvis/`. Each geometry NPZ contains 17 stable marker IDs, center/corner/normal data, and a 9 × 9 barycentric grid for every exported pose. All grid samples are recomputed from the source skin triangles in each pose, so the patches follow the surface; the default fit tolerance is 10 mm to accommodate the curved heel proxy. Use `--marker-size-mm`, `--grid-resolution`, `--fit-tolerance-m`, `--max-projection-distance-m`, and `--force` to configure or regenerate the outputs.

The generated marker NPZ is then passed to the existing Blender importer as in the preview section below. The per-model marker IDs are stable across genders and sides; `paper_code` and `paper_table_row` retain the paper's anatomical code and Table 1 row.

## Generate animated marker geometry

```bash
python scripts/generate_marker_geometry.py \
  --npz output/pelvis/pelvis_leg_anim_male_right.npz \
  --layout output/pelvis/layout_shin_experiment_male_right.json
```

The generator validates provenance and marker fields, fits a configurable odd-resolution surface grid (default `9 × 9`), and rejects patches that cross their declared pelvis/leg region, leave the skin, exceed the fit tolerance, or fold. Use `--grid-resolution` and `--fit-tolerance-m` to adjust the geometry checks. By default it writes `output/pelvis/markers_<layout>_<gender>_<side>.npz`; `--force` allows replacing an existing file.

## Preview and render a take

Build a separate markerized scene; the regular pelvis-leg runner and its no-marker output remain unchanged:

```bash
blender -b --python scripts/blender_import_pelvis_leg.py -- \
  output/pelvis/pelvis_leg_anim_male_right.npz \
  output/pelvis/pelvis_leg_male_right_markers.blend \
  --marker-geometry output/pelvis/markers_mikneesota_male_right.npz

blender -b output/pelvis/pelvis_leg_male_right_markers.blend \
  --python marker_tracking/blender/export_take.py -- \
  --output-dir output/pelvis/takes/shin_experiment
```

The marker meshes use the same absolute shape keys and frame-to-pose mapping as `Skin`, with only a 0.1 mm display offset; ground truth remains on the true surface. Render/export requires the exact experimental timeline (do not build with `--visual`). Each take contains `rgb/frame_0000.png` onward, `camera.json` with intrinsics/extrinsics and source metadata, and `ground_truth.csv` with one row per frame/marker. Its common `frame`, `time_s`, `marker_id`, `u_px`, `v_px`, `x_m`, `y_m`, and `z_m` fields follow the tracking table's naming and camera coordinate convention; the remaining columns contain four camera-space corners and the camera-space normal. `marker_tracking.table` remains the observation CSV contract, including its `detected` and `valid` flags; this geometric ground-truth file does not fabricate detector observations.

## Project layout and tests

- `layout.py` and `anchors.py`: versioned layouts, mesh provenance, patch fitting, and per-pose barycentric evaluation.
- `paper_layout.py`: MiKneeSoTA table definitions and approximate, region-constrained surface proxy placement.
- `scripts/generate_mikneesota_markers.py`: creates all four paper-preset layouts and animated geometry files.
- `blender/place_markers.py`: interactive surface picker and marker preview.
- `blender/export_take.py`: rendered frames, camera metadata, and camera-space geometry ground truth.
- `marker_tracking/`: tracking pipeline modules (`io`, `detect`, `lift`, `table`, `evaluate`, `run`).
- `tests/`: synthetic layout/geometry and ground-truth checks, plus tracking table tests.

From the project root, install `pytest` if needed and run all marker-tracking tests:


```powershell
python -m pip install pytest
python -m pytest marker_tracking\tests
```
