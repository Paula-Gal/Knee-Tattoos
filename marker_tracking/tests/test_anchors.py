import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anchors import AnchorError, evaluate_barycentric, fit_marker_patch, make_marker_geometry
from layout import LEG_BONES, create_layout, save_layout


def source_mesh(labels=None):
    return {
        'gender': np.array('male'),
        'side': np.array('right'),
        'region': np.array('pelvis+right_leg'),
        'skin_verts': np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]],
        ], dtype=np.float32),
        'skin_faces': np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        'skin_vert_ids': np.array([10, 11, 12, 13], dtype=np.int32),
        'skin_label': np.array(labels if labels is not None else [1, 1, 1, 1], dtype=np.int32),
        'units': np.array('metres'),
        'coordinate_system': np.array('synthetic: x right, y up, z forward'),
    }


def marker(texture_path):
    return {
        'id': 3,
        'label': 'test_patch',
        'size_m': 0.2,
        'rotation_deg': 0.0,
        'texture': str(texture_path),
        'anchor': {'face': 0, 'bary': [0.5, 0.25, 0.25]},
    }


class TestMarkerAnchors(unittest.TestCase):
    def test_patch_is_fitted_and_follows_pose_deformation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            texture = Path(temporary_directory) / 'marker.png'
            texture.write_bytes(b'placeholder')
            source = source_mesh()
            layout = create_layout(source, [marker(texture)])

            geometry = make_marker_geometry(source, layout, grid_resolution=5, fit_tolerance_m=0.001)

        self.assertEqual(geometry['surface_face'].shape, (1, 5, 5))
        self.assertEqual(geometry['surface_bary'].shape, (1, 5, 5, 3))
        self.assertEqual(geometry['grid_verts'].shape, (2, 1, 5, 5, 3))
        self.assertEqual(geometry['centre'].shape, (2, 1, 3))
        self.assertEqual(geometry['corners'].shape, (2, 1, 4, 3))
        self.assertEqual(geometry['normal'].shape, (2, 1, 3))
        self.assertTrue(np.isfinite(geometry['grid_verts']).all())
        np.testing.assert_allclose(geometry['centre'][0, 0], [0.5, 0.25, 0.0], atol=1e-7)
        np.testing.assert_allclose(geometry['centre'][1, 0], [0.5, 0.25, 0.05], atol=1e-7)
        np.testing.assert_allclose(geometry['grid_verts'][:, 0, 2, 2], geometry['centre'][:, 0], atol=1e-7)
        np.testing.assert_allclose(geometry['normal'][0, 0], [0, 0, 1], atol=1e-6)

    def test_patch_that_reaches_pelvis_face_is_rejected(self):
        source = source_mesh(labels=[1, 1, 1, 0])
        boundary_marker = marker('unused.png')
        boundary_marker['anchor'] = {'face': 0, 'bary': [0.5, 0, 0.5]}
        with self.assertRaisesRegex(AnchorError, 'boundary'):
            fit_marker_patch(
                source['skin_verts'][0], source['skin_faces'], source['skin_label'],
                'right', boundary_marker, grid_resolution=5, fit_tolerance_m=0.01,
            )

    def test_pelvis_marker_patch_is_constrained_to_pelvis_faces(self):
        source = source_mesh(labels=[0, 0, 0, 0])
        pelvic_marker = marker('unused.png')
        pelvic_marker['site_region'] = 'pelvis'

        surface_face, surface_bary = fit_marker_patch(
            source['skin_verts'][0], source['skin_faces'], source['skin_label'],
            'right', pelvic_marker, grid_resolution=5, fit_tolerance_m=0.001,
        )

        self.assertEqual(surface_face.shape, (5, 5))
        self.assertTrue(np.all(source['skin_label'][source['skin_faces'][surface_face]] == 0))
        np.testing.assert_allclose(surface_bary.sum(axis=-1), 1.0, atol=1e-7)

    def test_degenerate_anchor_triangle_is_rejected(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        labels = np.array([1, 1, 1], dtype=np.int32)
        with self.assertRaisesRegex(AnchorError, 'degenerate'):
            fit_marker_patch(
                verts, faces, labels, 'right', marker('unused.png'),
                grid_resolution=3, fit_tolerance_m=0.01,
            )

    def test_evaluate_barycentric_rejects_invalid_weights(self):
        source = source_mesh()
        with self.assertRaisesRegex(AnchorError, 'barycentric'):
            evaluate_barycentric(
                source['skin_verts'], source['skin_faces'], np.array([0]),
                np.array([[0.2, 0.2, 0.2]]),
            )

    def test_even_grid_resolution_is_rejected(self):
        source = source_mesh()
        with self.assertRaisesRegex(AnchorError, 'odd'):
            fit_marker_patch(
                source['skin_verts'][0], source['skin_faces'], source['skin_label'],
                'right', marker('unused.png'), grid_resolution=4,
            )

    def test_generator_cli_writes_marker_npz(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = source_mesh()
            source_path = directory / 'pelvis_leg.npz'
            np.savez_compressed(source_path, **source)
            texture = directory / 'marker.png'
            texture.write_bytes(b'placeholder')
            layout_path = directory / 'layout.json'
            layout = create_layout(source, [marker('marker.png')], layout_path=layout_path)
            save_layout(layout_path, layout, source_npz=source)
            output_path = directory / 'generated_markers.npz'
            result = subprocess.run(
                [
                    sys.executable,
                    str(root / 'scripts' / 'generate_marker_geometry.py'),
                    '--npz', str(source_path), '--layout', str(layout_path), '--output', str(output_path),
                    '--grid-resolution', '5', '--fit-tolerance-m', '0.001',
                ],
                cwd=root, capture_output=True, text=True, check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with np.load(output_path, allow_pickle=False) as generated:
                self.assertEqual(generated['grid_verts'].shape, (2, 1, 5, 5, 3))
                self.assertEqual(generated['marker_id'].tolist(), [3])
                self.assertEqual(str(generated['texture'][0]), str(texture.resolve()))

    def test_real_46_pose_export_if_available(self):
        export_path = Path(__file__).resolve().parents[2] / 'output' / 'pelvis' / 'pelvis_leg_anim_male_right.npz'
        if not export_path.is_file():
            self.skipTest('male/right pelvis-leg export has not been generated')
        with np.load(export_path, allow_pickle=False) as archive:
            source = {name: archive[name] for name in archive.files}

        faces = source['skin_faces']
        labels = source['skin_label']
        side = str(source['side'])
        leg_faces = np.flatnonzero(np.all(np.isin(labels[faces], LEG_BONES[side]), axis=1))
        triangles = source['skin_verts'][0, faces[leg_faces]]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
        candidate = marker('unused.png')
        candidate['size_m'] = 0.01
        chosen_face = None
        for index in leg_faces[np.argsort(areas)[::-1]]:
            candidate['anchor'] = {'face': int(index), 'bary': [1 / 3, 1 / 3, 1 / 3]}
            try:
                fit_marker_patch(
                    source['skin_verts'][0], faces, labels, side, candidate,
                    grid_resolution=5, fit_tolerance_m=0.005,
                )
            except AnchorError:
                continue
            chosen_face = int(index)
            break
        self.assertIsNotNone(chosen_face, 'no safe leg-only patch found in the real export')

        with tempfile.TemporaryDirectory() as temporary_directory:
            texture = Path(temporary_directory) / 'marker.png'
            texture.write_bytes(b'placeholder')
            real_marker = marker(texture)
            real_marker['size_m'] = 0.01
            real_marker['anchor'] = {'face': chosen_face, 'bary': [1 / 3, 1 / 3, 1 / 3]}
            layout = create_layout(source, [real_marker])
            geometry = make_marker_geometry(source, layout, grid_resolution=5, fit_tolerance_m=0.005)

        self.assertEqual(geometry['grid_verts'].shape, (46, 1, 5, 5, 3))
        self.assertTrue(np.isfinite(geometry['grid_verts']).all())
        tri = source['skin_verts'][0, faces[chosen_face]]
        expected_center = np.asarray(real_marker['anchor']['bary']) @ tri
        np.testing.assert_allclose(geometry['centre'][0, 0], expected_center, atol=1e-6)


if __name__ == '__main__':
    unittest.main()