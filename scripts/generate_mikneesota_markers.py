"""Generate paper-derived marker layouts and animated geometry for all four models."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'marker_tracking'))

from anchors import AnchorError, DEFAULT_GRID_RESOLUTION, make_marker_geometry
from layout import LayoutError, resolve_texture_path, save_layout
from paper_layout import (
    MARKER_DIAMETER_M,
    MAX_PROJECTION_DISTANCE_M,
    create_mikneesota_layout,
    write_disc_texture,
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate MiKneeSoTA marker layouts and per-pose geometry for male/female, left/right.'
    )
    parser.add_argument('--input-dir', default=os.path.join(ROOT, 'output', 'pelvis'),
                        help='directory containing the four pelvis-leg NPZ exports')
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'output', 'pelvis'),
                        help='destination for layouts, texture and marker geometry')
    parser.add_argument('--marker-size-mm', type=float, default=MARKER_DIAMETER_M * 1000,
                        help='provisional marker diameter in millimetres (default: 14)')
    parser.add_argument('--grid-resolution', type=int, default=DEFAULT_GRID_RESOLUTION,
                        help='odd number of samples per marker edge (default: 9)')
    parser.add_argument('--fit-tolerance-m', type=float, default=0.01,
                        help='maximum tangent-plane projection distance in metres (default: 0.01)')
    parser.add_argument('--max-projection-distance-m', type=float, default=MAX_PROJECTION_DISTANCE_M,
                        help='maximum distance from a proxy target to its allowed skin region')
    parser.add_argument('--force', action='store_true', help='overwrite existing generated outputs')
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    sources = []
    outputs = []
    for gender in ('male', 'female'):
        for side in ('right', 'left'):
            stem = f'{gender}_{side}'
            source_path = os.path.join(input_dir, f'pelvis_leg_anim_{stem}.npz')
            layout_path = os.path.join(output_dir, f'mikneesota_layout_{stem}.json')
            geometry_path = os.path.join(output_dir, f'markers_mikneesota_{stem}.npz')
            if not os.path.isfile(source_path):
                parser.error(f'missing source export: {source_path}; run the pelvis-leg export first')
            sources.append((gender, side, source_path))
            outputs.extend((layout_path, geometry_path))

    texture_path = os.path.join(output_dir, 'mikneesota_marker_disc.png')
    if not args.force:
        existing = [path for path in outputs if os.path.exists(path)]
        if existing:
            parser.error(f'{existing[0]} already exists; use --force to overwrite generated files')
        if os.path.exists(texture_path):
            try:
                with open(texture_path, 'rb') as texture_file:
                    valid_texture = texture_file.read(8) == b'\x89PNG\r\n\x1a\n'
            except OSError:
                valid_texture = False
            if not valid_texture:
                parser.error(f'{texture_path} exists but is not a PNG; use --force to replace it')

    try:
        loaded_sources = []
        for gender, side, source_path in sources:
            with np.load(source_path, allow_pickle=False) as archive:
                source = {name: archive[name] for name in archive.files}
            if str(source.get('gender')) != gender or str(source.get('side')) != side:
                raise LayoutError(f'{source_path} metadata does not match its filename')
            loaded_sources.append((gender, side, source))

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.isfile(texture_path) or args.force:
            texture_path = write_disc_texture(texture_path)

        generated = []
        for gender, side, source in loaded_sources:
            stem = f'{gender}_{side}'
            layout_path = os.path.join(output_dir, f'mikneesota_layout_{stem}.json')
            geometry_path = os.path.join(output_dir, f'markers_mikneesota_{stem}.npz')
            layout = create_mikneesota_layout(
                source,
                texture_path,
                marker_size_m=args.marker_size_mm / 1000.0,
                max_projection_distance_m=args.max_projection_distance_m,
                layout_path=layout_path,
            )
            geometry = make_marker_geometry(
                source,
                layout,
                grid_resolution=args.grid_resolution,
                fit_tolerance_m=args.fit_tolerance_m,
                layout_path=layout_path,
            )
            geometry['texture'] = np.asarray([
                resolve_texture_path(marker['texture'], layout_path) for marker in layout['markers']
            ])
            generated.append((layout_path, geometry_path, source, layout, geometry))

        for layout_path, geometry_path, source, layout, geometry in generated:
            save_layout(layout_path, layout, source_npz=source)
            np.savez_compressed(geometry_path, **geometry)
            worst_projection = float(np.max(geometry['projection_distance_m']))
            print(
                f'Saved {len(layout["markers"])} markers / {len(geometry["pose_index"])} poses: '
                f'{layout_path} and {geometry_path} '
                f'(max proxy-to-skin distance {worst_projection:.3f} m)'
            )
    except (OSError, ValueError, LayoutError, AnchorError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()