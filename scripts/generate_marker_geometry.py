"""Generate surface-attached marker grids and per-pose ground truth from a layout."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'marker_tracking'))

from anchors import AnchorError, DEFAULT_FIT_TOLERANCE_M, DEFAULT_GRID_RESOLUTION, make_marker_geometry
from layout import LayoutError, load_layout, resolve_texture_path


def main():
    parser = argparse.ArgumentParser(description='Generate per-pose skin-conforming marker geometry.')
    parser.add_argument('--npz', required=True, help='pelvis_leg_anim_<gender>_<side>.npz')
    parser.add_argument('--layout', required=True, help='source-bound marker layout JSON')
    parser.add_argument('--output', help='override the generated marker NPZ path')
    parser.add_argument('--grid-resolution', type=int, default=DEFAULT_GRID_RESOLUTION,
                        help='odd number of surface samples per marker edge (default: 9)')
    parser.add_argument('--fit-tolerance-m', type=float, default=DEFAULT_FIT_TOLERANCE_M,
                        help='maximum tangent-plane projection distance in metres')
    parser.add_argument('--force', action='store_true', help='overwrite an existing marker NPZ')
    args = parser.parse_args()

    npz_path = os.path.abspath(args.npz)
    layout_path = os.path.abspath(args.layout)
    if not os.path.isfile(npz_path):
        parser.error(f'pelvis-leg NPZ does not exist: {npz_path}')
    if not os.path.isfile(layout_path):
        parser.error(f'marker layout does not exist: {layout_path}')
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            source = {name: archive[name] for name in archive.files}
        layout = load_layout(layout_path, source_npz=source)
    except (OSError, ValueError, LayoutError) as error:
        parser.error(str(error))

    gender = layout['source']['gender']
    side = layout['source']['side']
    output_path = args.output or os.path.join(
        ROOT, 'output', 'pelvis', f'markers_{os.path.splitext(os.path.basename(layout_path))[0]}_{gender}_{side}.npz',
    )
    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path) and not args.force:
        parser.error(f'{output_path} already exists; use --force to overwrite it')

    try:
        geometry = make_marker_geometry(
            source, layout,
            grid_resolution=args.grid_resolution,
            fit_tolerance_m=args.fit_tolerance_m,
            layout_path=layout_path,
        )
        geometry['texture'] = np.asarray([
            resolve_texture_path(marker['texture'], layout_path) for marker in layout['markers']
        ])
    except (AnchorError, LayoutError, ValueError) as error:
        parser.error(str(error))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **geometry)
    print(
        f'Saved {output_path}: {len(geometry["marker_id"])} markers, '
        f'{len(geometry["pose_index"])} poses, grid {int(geometry["grid_resolution"])}x'
        f'{int(geometry["grid_resolution"])}'
    )


if __name__ == '__main__':
    main()