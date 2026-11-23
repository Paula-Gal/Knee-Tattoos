"""Assemble the fixed frame x marker table with flags and write markers.csv.

One row per frame and marker, every marker in every frame. Values that are not
available are 0, and the 0/1 flags say why:
    detected = 0             marker not found: all numeric fields 0
    detected = 1, valid = 0  found, but too little depth: u, v kept, x, y, z = 0
    detected = 1, valid = 1  everything filled in
"""
import csv

COLUMNS = ['frame', 'time_s', 'marker_id', 'detected', 'valid',
           'u_px', 'v_px', 'x_m', 'y_m', 'z_m', 'n_depth_px']


def make_rows(n_frames, fps, marker_ids, results):
    """Build all rows, sorted by frame, then marker id.

    results: {(frame, marker_id): {'u_px', 'v_px', 'x_m', 'y_m', 'z_m', 'n_depth_px', 'valid'}}
             one entry per detected marker; missing keys = not detected.
    """
    rows = []
    for frame in range(n_frames):
        for marker_id in sorted(marker_ids):
            row = {'frame': frame, 'time_s': frame / fps, 'marker_id': marker_id,
                   'detected': 0, 'valid': 0, 'u_px': 0.0, 'v_px': 0.0,
                   'x_m': 0.0, 'y_m': 0.0, 'z_m': 0.0, 'n_depth_px': 0}
            r = results.get((frame, marker_id))
            if r is not None:
                row['detected'] = 1
                row['u_px'], row['v_px'] = r['u_px'], r['v_px']
                row['n_depth_px'] = r['n_depth_px']
                if r['valid']:
                    row['valid'] = 1
                    row['x_m'], row['y_m'], row['z_m'] = r['x_m'], r['y_m'], r['z_m']
            rows.append(row)
    return rows


def write_csv(rows, path):
    """Write rows to path with fixed decimals: time 1 us, pixels 0.001 px, xyz 1 um."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r['frame'], f"{r['time_s']:.6f}", r['marker_id'], r['detected'], r['valid'],
                        f"{r['u_px']:.3f}", f"{r['v_px']:.3f}",
                        f"{r['x_m']:.6f}", f"{r['y_m']:.6f}", f"{r['z_m']:.6f}", r['n_depth_px']])
