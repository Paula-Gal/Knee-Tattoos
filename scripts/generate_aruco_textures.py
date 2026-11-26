"""Generate the ArUco texture images referenced by the MiKneeSoTA style preset."""
import argparse
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / 'marker_tracking' / 'presets' / 'textures'
MARKER_COUNT = 30
DEFAULT_SIZE_PX = 512


def generate_textures(output_dir, size_px=DEFAULT_SIZE_PX, force=False):
    """Write one distinct DICT_4X4_50 image for every possible paper site code."""
    if isinstance(size_px, bool) or not isinstance(size_px, int) or size_px < 32:
        raise ValueError('size_px must be an integer of at least 32')
    output_dir = Path(output_dir).resolve()
    paths = [output_dir / f'aruco_4x4_50_{marker_id:02d}.png'
             for marker_id in range(MARKER_COUNT)]
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        raise FileExistsError(f'{existing[0]} already exists; pass --force to overwrite textures')

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if MARKER_COUNT > dictionary.bytesList.shape[0]:
        raise ValueError('DICT_4X4_50 does not contain enough marker IDs')
    output_dir.mkdir(parents=True, exist_ok=True)
    for marker_id, path in enumerate(paths):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
        if not cv2.imwrite(str(path), image):
            raise OSError(f'could not write ArUco marker image: {path}')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                        help='destination for generated pattern PNGs')
    parser.add_argument('--size-px', type=int, default=DEFAULT_SIZE_PX,
                        help='width and height of each generated image')
    parser.add_argument('--force', action='store_true', help='overwrite existing pattern images')
    args = parser.parse_args()
    try:
        paths = generate_textures(args.output_dir, args.size_px, args.force)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Generated {len(paths)} DICT_4X4_50 textures in {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()