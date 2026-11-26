import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from layout import LayoutError, create_layout, load_layout, mesh_fingerprint, save_layout, validate_layout


def source_mesh():
    return {
        'gender': np.array('male'),
        'side': np.array('right'),
        'region': np.array('pelvis+right_leg'),
        'skin_verts': np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        ], dtype=np.float32),
        'skin_faces': np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        'skin_vert_ids': np.array([10, 11, 12, 13], dtype=np.int32),
        'skin_label': np.array([1, 1, 1, 1], dtype=np.int32),
    }


def marker(texture, **overrides):
    value = {
        'id': 7,
        'label': 'front_shin',
        'size_m': 0.03,
        'rotation_deg': 12.5,
        'texture': str(texture),
        'anchor': {'face': 0, 'bary': [0.2, 0.3, 0.5]},
    }
    value.update(overrides)
    return value


def texture_file(tmp_path):
    path = tmp_path / 'marker.png'
    path.write_bytes(b'placeholder image')
    return path


class TestMarkerLayout(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.tmp_path = Path(self.temporary_directory.name)
        self.texture = texture_file(self.tmp_path)
        self.source = source_mesh()

    def assert_layout_error(self, message, function, *args, **kwargs):
        with self.assertRaisesRegex(LayoutError, message):
            function(*args, **kwargs)

    def test_layout_round_trip_preserves_marker_and_source(self):
        layout = create_layout(
            self.source,
            [marker(self.texture, family='stag', shape='square')],
            reference_pose=1,
        )
        path = self.tmp_path / 'layout.json'

        save_layout(path, layout, source_npz=self.source)
        loaded = load_layout(path, source_npz=self.source)

        self.assertEqual(loaded, layout)
        self.assertEqual(loaded['schema_version'], 1)
        self.assertEqual(loaded['source']['reference_pose'], 1)
        self.assertEqual(loaded['source']['mesh_sha256'], mesh_fingerprint(self.source))
        self.assertEqual(loaded['markers'][0]['family'], 'stag')
        self.assertEqual(loaded['markers'][0]['shape'], 'square')

    def test_layout_defaults_legacy_style_and_rejects_invalid_style_fields(self):
        layout = create_layout(self.source, [marker(self.texture)])
        self.assertEqual(layout['markers'][0]['family'], 'custom')
        self.assertEqual(layout['markers'][0]['shape'], 'texture_alpha')
        self.assert_layout_error('family', create_layout, self.source,
                                 [marker(self.texture, family=' ')])
        self.assert_layout_error('shape', create_layout, self.source,
                                 [marker(self.texture, shape='triangle')])

    def test_rejects_malformed_barycentric_weights(self):
        for bad_bary in ([0.2, 0.2, 0.2], [-0.1, 0.5, 0.6], [0.5, float('nan'), 0.5]):
            with self.subTest(bary=bad_bary):
                layout = create_layout(self.source, [marker(self.texture)])
                layout['markers'][0]['anchor']['bary'] = bad_bary
                self.assert_layout_error(
                    'bary', validate_layout, layout, source_npz=self.source,
                    layout_path=self.tmp_path / 'layout.json',
                )

    def test_rejects_duplicate_ids_and_non_positive_sizes(self):
        first = marker(self.texture)
        duplicate = marker(self.texture, label='second')
        self.assert_layout_error('duplicate', create_layout, self.source, [first, duplicate])
        self.assert_layout_error('size', create_layout, self.source, [marker(self.texture, size_m=0)])

    def test_rejects_invalid_face_and_pelvis_anchor(self):
        self.assert_layout_error(
            'face', create_layout, self.source,
            [marker(self.texture, anchor={'face': 20, 'bary': [1, 0, 0]})],
        )
        pelvis_source = source_mesh()
        pelvis_source['skin_label'] = np.zeros(4, dtype=np.int32)
        self.assert_layout_error('leg', create_layout, pelvis_source, [marker(self.texture)])

    def test_accepts_declared_pelvis_anchor_and_preserves_site_metadata(self):
        pelvis_source = source_mesh()
        pelvis_source['skin_label'] = np.zeros(4, dtype=np.int32)
        pelvic_marker = marker(
            self.texture,
            site_region='pelvis',
            paper_code='LASI',
            paper_description='left anterior superior iliac spine',
        )

        layout = create_layout(pelvis_source, [pelvic_marker])

        self.assertEqual(layout['markers'][0]['site_region'], 'pelvis')
        self.assertEqual(layout['markers'][0]['paper_code'], 'LASI')
        self.assertEqual(layout['markers'][0]['paper_description'], 'left anterior superior iliac spine')

    def test_rejects_anchor_outside_declared_site_region(self):
        pelvis_source = source_mesh()
        pelvis_source['skin_label'] = np.zeros(4, dtype=np.int32)
        self.assert_layout_error(
            'declared leg region', create_layout, pelvis_source,
            [marker(self.texture, site_region='leg')],
        )

    def test_rejects_stale_mesh_fingerprint(self):
        layout = create_layout(self.source, [marker(self.texture)])
        changed_source = source_mesh()
        changed_source['skin_verts'][0, 0, 0] = 0.01
        self.assert_layout_error(
            'fingerprint', validate_layout, layout, source_npz=changed_source,
            layout_path=self.tmp_path / 'layout.json',
        )

    def test_rejects_missing_texture_file(self):
        self.assert_layout_error(
            'texture', create_layout, self.source, [marker(self.tmp_path / 'missing.png')],
            layout_path=self.tmp_path / 'layout.json',
        )

    def test_load_rejects_wrong_side_even_with_other_valid_layout(self):
        path = self.tmp_path / 'layout.json'
        save_layout(path, create_layout(self.source, [marker(self.texture)]), source_npz=self.source)
        other_side = source_mesh()
        other_side['side'] = np.array('left')
        other_side['region'] = np.array('pelvis+left_leg')
        self.assert_layout_error('source|fingerprint', load_layout, path, source_npz=other_side)


if __name__ == '__main__':
    unittest.main()