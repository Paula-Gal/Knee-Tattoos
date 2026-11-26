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
    MAX_PROJECTION_DISTANCE_M,
    create_mikneesota_layout,
    write_disc_texture,
)
from styles import DEFAULT_STYLE_PRESET_PATH, StylePresetError, load_style_preset


def main():
    parser = argparse.ArgumentParser(
        description='Generate MiKneeSoTA marker layouts and per-pose geometry for male/female, left/right.'
    )
    parser.add_argument('--input-dir', default=os.path.join(ROOT, 'output', 'pelvis'),
                        help='directory containing the four pelvis-leg NPZ exports')
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'output', 'pelvis'),
                        help='destination for layouts, texture and marker geometry')
    parser.add_argument('--style-preset', default=DEFAULT_STYLE_PRESET_PATH,
                        help='JSON marker style preset (default: bundled MiKneeSoTA example)')
    parser.add_argument('--marker-size-mm', type=float,
                        help='override the preset size for every site, in millimetres')
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

    if not args.force:
        existing = [path for path in outputs if os.path.exists(path)]
        if existing:
            parser.error(f'{existing[0]} already exists; use --force to overwrite generated files')

    try:
        style_preset_path = os.path.abspath(os.path.expanduser(args.style_preset))
        default_texture_path = os.path.abspath(os.path.join(
            os.path.dirname(DEFAULT_STYLE_PRESET_PATH), '..', '..', 'output', 'pelvis',
            'mikneesota_marker_disc.png',
        ))
        if (style_preset_path == os.path.abspath(DEFAULT_STYLE_PRESET_PATH)
                and not os.path.isfile(default_texture_path)):
            write_disc_texture(default_texture_path)
        style_preset = load_style_preset(style_preset_path)
        loaded_sources = []
        for gender, side, source_path in sources:
            with np.load(source_path, allow_pickle=False) as archive:
                source = {name: archive[name] for name in archive.files}
            if str(source.get('gender')) != gender or str(source.get('side')) != side:
                raise LayoutError(f'{source_path} metadata does not match its filename')
            loaded_sources.append((gender, side, source))

        os.makedirs(output_dir, exist_ok=True)

        generated = []
        for gender, side, source in loaded_sources:
            stem = f'{gender}_{side}'
            layout_path = os.path.join(output_dir, f'mikneesota_layout_{stem}.json')
            geometry_path = os.path.join(output_dir, f'markers_mikneesota_{stem}.npz')
            layout = create_mikneesota_layout(
                source,
                marker_size_m=None if args.marker_size_mm is None else args.marker_size_mm / 1000.0,
                max_projection_distance_m=args.max_projection_distance_m,
                layout_path=layout_path,
                style_preset=style_preset,
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
    except (OSError, ValueError, LayoutError, AnchorError, StylePresetError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()