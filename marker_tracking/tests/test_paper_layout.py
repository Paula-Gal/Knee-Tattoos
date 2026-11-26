import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchors import make_marker_geometry
from layout import LEG_BONES
from paper_layout import (
    MAX_PROJECTION_DISTANCE_M,
    MIKNEESOTA_MARKER_PAIRS,
    create_mikneesota_layout,
    write_disc_texture,
)
from styles import load_style_preset


class TestMikneeSoTALayout(unittest.TestCase):
    def test_paper_protocol_defines_thirty_bilateral_markers(self):
        self.assertEqual(len(MIKNEESOTA_MARKER_PAIRS), 15)
        self.assertEqual(2 * len(MIKNEESOTA_MARKER_PAIRS), 30)
        self.assertEqual({pair['site_region'] for pair in MIKNEESOTA_MARKER_PAIRS[:2]}, {'pelvis'})

    def test_all_four_exports_generate_valid_surface_anchored_markers(self):
        project_root = Path(__file__).resolve().parents[2]
        exports = [
            project_root / 'output' / 'pelvis' / f'pelvis_leg_anim_{gender}_{side}.npz'
            for gender in ('male', 'female')
            for side in ('right', 'left')
        ]
        if not all(path.is_file() for path in exports):
            self.skipTest('the four pelvis-leg exports have not all been generated')

        with tempfile.TemporaryDirectory() as temporary_directory:
            texture_path = write_disc_texture(Path(temporary_directory) / 'marker.png')
            self.assertEqual(Path(texture_path).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            style_path = Path(temporary_directory) / 'styles.json'
            style_path.write_text(json.dumps({
                'schema_version': 1,
                'defaults': {
                    'family': 'aruco', 'shape': 'square', 'texture': 'marker.png',
                    'size_m': 0.014, 'rotation_deg': 0,
                },
                'sites': {'RASI': {'family': 'stag', 'shape': 'circle'}},
            }), encoding='utf-8')
            style_preset = load_style_preset(style_path)

            for export_path in exports:
                with self.subTest(export=export_path.name):
                    with np.load(export_path, allow_pickle=False) as archive:
                        source = {name: archive[name] for name in archive.files}
                    side = str(source['side'])
                    layout = create_mikneesota_layout(source, texture_path)
                    styled_layout = create_mikneesota_layout(source, style_preset=style_preset)
                    markers = layout['markers']
                    styled_markers = styled_layout['markers']
                    self.assertEqual(len(markers), 17)
                    self.assertEqual([m['id'] for m in styled_markers], [m['id'] for m in markers])
                    self.assertEqual([m['paper_code'] for m in styled_markers], [m['paper_code'] for m in markers])
                    for marker_default, marker_styled in zip(markers, styled_markers):
                        self.assertEqual(marker_default['anchor'], marker_styled['anchor'])
                    self.assertEqual(styled_markers[0]['family'], 'aruco')
                    self.assertEqual(styled_markers[0]['shape'], 'square')
                    if side == 'right':
                        rasi = next(marker for marker in styled_markers if marker['paper_code'] == 'RASI')
                        self.assertEqual(rasi['family'], 'stag')
                        self.assertEqual(rasi['shape'], 'circle')
                        self.assertEqual(rasi['size_m'], 0.014)
                    self.assertTrue(layout['metadata']['approximate'])
                    self.assertEqual(layout['metadata']['paper_marker_count'], 30)
                    self.assertEqual(sum(marker['site_region'] == 'pelvis' for marker in markers), 4)
                    self.assertEqual(sum(marker['site_region'] == 'leg' for marker in markers), 13)
                    self.assertTrue(all(marker['size_m'] == 0.014 for marker in markers))
                    self.assertTrue(all(
                        marker['projection_distance_m'] <= MAX_PROJECTION_DISTANCE_M
                        for marker in markers
                    ))
                    self.assertEqual(len({marker['id'] for marker in markers}), len(markers))
                    self.assertTrue(all(0 <= marker['id'] < 30 for marker in markers))
                    codes = {marker['paper_code'] for marker in markers}
                    self.assertTrue({'LASI', 'RASI', 'LPSI', 'RPSI'} <= codes)
                    side_letter = 'L' if side == 'left' else 'R'
                    self.assertTrue({f'{side_letter}{suffix}' for suffix in (
                        'THI', 'KNE', 'TIB', 'ANK', 'HEE', 'TOE', 'MFE', 'MTI',
                        'KNM', 'LTC', 'MTC', 'KNA', 'MED',
                    )} <= codes)

                    for marker in markers:
                        face = source['skin_faces'][marker['anchor']['face']]
                        labels = source['skin_label'][face]
                        if marker['site_region'] == 'pelvis':
                            self.assertTrue(np.all(labels == 0), marker['paper_code'])
                        else:
                            self.assertTrue(np.all(np.isin(labels, LEG_BONES[side])), marker['paper_code'])

                    geometry = make_marker_geometry(
                        source, styled_layout, grid_resolution=9, fit_tolerance_m=0.01,
                    )
                    self.assertEqual(geometry['grid_verts'].shape, (46, 17, 9, 9, 3))
                    self.assertEqual(geometry['centre'].shape, (46, 17, 3))
                    self.assertEqual(geometry['corners'].shape, (46, 17, 4, 3))
                    self.assertEqual(geometry['normal'].shape, (46, 17, 3))
                    self.assertTrue(np.isfinite(geometry['grid_verts']).all())
                    self.assertTrue(np.isfinite(geometry['corners']).all())
                    self.assertTrue(np.isfinite(geometry['normal']).all())
                    np.testing.assert_array_equal(geometry['paper_code'], [m['paper_code'] for m in styled_markers])
                    np.testing.assert_array_equal(geometry['family'], [m['family'] for m in styled_markers])
                    np.testing.assert_array_equal(geometry['shape'], [m['shape'] for m in styled_markers])


if __name__ == '__main__':
    unittest.main()