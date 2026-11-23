import csv
import os

from marker_tracking.table import COLUMNS, make_rows, write_csv

EXAMPLE = os.path.join(os.path.dirname(__file__), '..', 'examples', 'markers_example.csv')

FOUND = {'u_px': 612.4, 'v_px': 355.1, 'x_m': 0.02, 'y_m': -0.01, 'z_m': 0.64, 'n_depth_px': 25, 'valid': True}
NO_DEPTH = {'u_px': 700.0, 'v_px': 401.3, 'x_m': 9.0, 'y_m': 9.0, 'z_m': 9.0, 'n_depth_px': 2, 'valid': False}


def read(path):
    with open(path, newline='') as f:
        return list(csv.reader(f))


def test_one_row_per_frame_and_marker_sorted():
    rows = make_rows(3, 30, [2, 0, 1], {})
    assert len(rows) == 9
    assert [(r['frame'], r['marker_id']) for r in rows] == [(f, m) for f in range(3) for m in (0, 1, 2)]


def test_time_is_frame_over_fps():
    rows = make_rows(31, 30, [0], {})
    assert rows[30]['time_s'] == 1.0


def test_flags_and_zeros():
    rows = make_rows(1, 30, [0, 1, 2], {(0, 0): FOUND, (0, 1): NO_DEPTH})
    found, no_depth, missing = rows

    assert (found['detected'], found['valid'], found['z_m']) == (1, 1, 0.64)

    # detected without depth: pixel position kept, xyz zeroed even if the input had values
    assert (no_depth['detected'], no_depth['valid']) == (1, 0)
    assert (no_depth['u_px'], no_depth['n_depth_px']) == (700.0, 2)
    assert (no_depth['x_m'], no_depth['y_m'], no_depth['z_m']) == (0.0, 0.0, 0.0)

    assert (missing['detected'], missing['valid']) == (0, 0)
    assert all(missing[c] == 0 for c in COLUMNS[5:])


def test_written_csv_header_and_format(tmp_path):
    path = tmp_path / 'markers.csv'
    write_csv(make_rows(1, 30, [0, 1], {(0, 0): FOUND}), path)
    lines = read(path)
    assert lines[0] == COLUMNS
    assert lines[1] == ['0', '0.000000', '0', '1', '1', '612.400', '355.100', '0.020000', '-0.010000', '0.640000', '25']
    assert lines[2] == ['0', '0.000000', '1', '0', '0', '0.000', '0.000', '0.000000', '0.000000', '0.000000', '0']


def test_rerun_gives_identical_file(tmp_path):
    rows = make_rows(5, 30, [0, 1], {(2, 1): FOUND, (3, 0): NO_DEPTH})
    write_csv(rows, tmp_path / 'a.csv')
    write_csv(rows, tmp_path / 'b.csv')
    assert (tmp_path / 'a.csv').read_bytes() == (tmp_path / 'b.csv').read_bytes()


def test_committed_example_follows_contract():
    lines = read(EXAMPLE)
    assert lines[0] == COLUMNS
    body = lines[1:]
    frames = {r[0] for r in body}
    markers = {r[2] for r in body}
    assert len(body) == len(frames) * len(markers)
    for r in body:
        detected, valid, z = int(r[3]), int(r[4]), float(r[9])
        assert valid <= detected
        assert (z > 0) == bool(valid)
