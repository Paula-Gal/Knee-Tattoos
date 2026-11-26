import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'blender'))
from export_take import _columns, _transform_points, _write_ground_truth


class TestGroundTruthExport(unittest.TestCase):
    def test_camera_transform_handles_points_and_directions(self):
        points = np.array([[1.0, 2.0, 3.0]])
        transform = np.array([
            [1, 0, 0, 4],
            [0, 1, 0, 5],
            [0, 0, 1, 6],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        np.testing.assert_allclose(_transform_points(points, transform), [[5, 7, 9]])

    def test_ground_truth_rows_are_sorted_and_include_camera_geometry(self):
        geometry = {
            'marker_id': np.array([4, 1], dtype=np.int64),
            'centre': np.array([
                [[0.2, 0.1, 1.0], [0.1, 0.2, 1.0]],
                [[0.3, 0.1, 1.0], [0.2, 0.2, 1.0]],
            ]),
            'corners': np.zeros((2, 2, 4, 3), dtype=np.float64),
            'normal': np.tile([0.0, 0.0, 1.0], (2, 2, 1)),
        }
        geometry['corners'][:, :, :, 2] = 1.0
        intrinsics = (1280, 720, 645.486328125, 644.7218017578125, 641.44970703125, 371.508178710938)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / 'ground_truth.csv'
            _write_ground_truth(path, geometry, np.eye(4), 24, intrinsics)
            with path.open(newline='', encoding='utf-8') as input_file:
                rows = list(csv.DictReader(input_file))

        self.assertEqual(len(_columns()), 23)
        self.assertEqual(len(rows), 4)
        self.assertEqual([(int(row['frame']), int(row['marker_id'])) for row in rows],
                         [(0, 1), (0, 4), (1, 1), (1, 4)])
        self.assertEqual(rows[0]['time_s'], '0.000000')
        self.assertAlmostEqual(float(rows[0]['x_m']), 0.1)
        self.assertAlmostEqual(float(rows[0]['y_m']), 0.2)
        self.assertEqual(float(rows[0]['z_m']), 1.0)
        self.assertAlmostEqual(float(rows[0]['u_px']), 645.486328125 * 0.1 + 641.44970703125, places=2)
        self.assertAlmostEqual(float(rows[0]['v_px']), 644.7218017578125 * 0.2 + 371.508178710938, places=2)
        self.assertEqual((rows[0]['normal_x'], rows[0]['normal_y'], rows[0]['normal_z']), ('0.000000', '0.000000', '1.000000'))


if __name__ == '__main__':
    unittest.main()