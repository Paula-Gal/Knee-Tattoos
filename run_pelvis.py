"""Static pelvis pipeline: SKEL export -> check -> Blender scene."""
import argparse
import os
import sys

from run_knee_bend import find_blender, run

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join('output', 'pelvis')


def run_case(gender, blender, force):
    npz = os.path.join(OUT_DIR, f'pelvis_{gender}.npz')
    blend = os.path.join(OUT_DIR, f'pelvis_{gender}.blend')
    print(f'\n===== {gender} pelvis')
    if os.path.exists(os.path.join(ROOT, blend)) and not force:
        sys.exit(f'{blend} already exists (maybe edited by a teammate). Use --force to overwrite it.')

    run('1/3 export SKEL', [sys.executable, os.path.join('scripts', 'pelvis_export.py'),
                            '--gender', gender])
    run('2/3 check export', [sys.executable, os.path.join('scripts', 'check_pelvis_export.py'), npz])
    run('3/3 build Blender scene', [blender, '-b', '--python',
                                    os.path.join('scripts', 'blender_import_pelvis.py'),
                                    '--', npz, blend] + (['--force'] if force else []))


def main():
    parser = argparse.ArgumentParser(description='Build a static SKEL pelvis scene for one or both genders.')
    parser.add_argument('gender', choices=['male', 'female', 'all'])
    parser.add_argument('--force', action='store_true', help='overwrite existing .blend files')
    args = parser.parse_args()

    blender = find_blender()
    genders = ('male', 'female') if args.gender == 'all' else (args.gender,)
    for gender in genders:
        run_case(gender, blender, args.force)
    print(f'\nDone: {len(genders)} pelvis scene(s) in {OUT_DIR}')


if __name__ == '__main__':
    main()