"""Write examples/markers_example.csv: fake tracker output in the real format.

For teammates building on markers.csv before the tracker works. The values are
made up, but consistent: xyz is projected with the D456 K to get u, v.

    4 markers on a shank that swings 0 -> 90 deg about a "knee" 0.7 m in front of
    the camera, 46 frames at 30 fps.
    marker 2: not detected in frames 20-29 (as if occluded)
    marker 3: detected but too few depth pixels in frames 10-14 (valid = 0)

Run from the marker_tracking folder:
    python examples/make_example.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from marker_tracking.table import make_rows, write_csv  # noqa: E402

FX, FY, CX, CY = 645.486328125, 644.7218017578125, 641.44970703125, 371.508178710938  # D456 colour
FPS = 30
N_FRAMES = 46
MARKER_IDS = [0, 1, 2, 3]
KNEE = (0.0, -0.05, 0.70)                 # knee position in the camera frame [m]
DIST_BELOW_KNEE = [0.05, 0.10, 0.15, 0.20]  # marker distance along the shank [m]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'markers_example.csv')

results = {}
for frame in range(N_FRAMES):
    angle = math.radians(90 * frame / (N_FRAMES - 1))  # 0 = shank straight down, 90 = towards camera
    for marker_id, d in zip(MARKER_IDS, DIST_BELOW_KNEE):
        if marker_id == 2 and 20 <= frame <= 29:
            continue  # not detected
        x = KNEE[0]
        y = KNEE[1] + d * math.cos(angle)   # y down
        z = KNEE[2] - d * math.sin(angle)   # z forward, shank swings towards the camera
        valid = not (marker_id == 3 and 10 <= frame <= 14)
        results[(frame, marker_id)] = {
            'u_px': CX + FX * x / z, 'v_px': CY + FY * y / z,
            'x_m': x, 'y_m': y, 'z_m': z,
            'n_depth_px': 25 if valid else 2, 'valid': valid,
        }

write_csv(make_rows(N_FRAMES, FPS, MARKER_IDS, results), OUT)
print(f'wrote {OUT}')
