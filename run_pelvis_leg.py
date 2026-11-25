"""Animated pelvis-plus-leg pipeline: SKEL export -> check -> Blender scene."""
import argparse
import os
import sys

from run_knee_bend import find_blender, run

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join('output', 'pelvis')


def run_case(gender, side, blender, force, visual):
    stem = f'pelvis_leg_{gender}_{side}'
    npz = os.path.join(OUT_DIR, f'pelvis_leg_anim_{gender}_{side}.npz')
    blend = os.path.join(OUT_DIR, f'{stem}.blend')
    print(f'\n===== {gender} {side} pelvis-leg')
    if os.path.exists(os.path.join(ROOT, blend)) and not force:
        sys.exit(f'{blend} already exists (maybe edited by a teammate). Use --force to overwrite it.')

    run('1/3 export SKEL', [sys.executable, os.path.join('scripts', 'pelvis_leg_export.py'),
                            '--gender', gender, '--side', side])
    run('2/3 check export', [sys.executable, os.path.join('scripts', 'check_export.py'), npz])
    blender_args = [blender, '-b', '--python', os.path.join('scripts', 'blender_import_pelvis_leg.py'),
                    '--', npz, blend]
    if visual:
        blender_args.append('--visual')
    if force:
        blender_args.append('--force')
    run('3/3 build Blender scene', blender_args)


def main():
    parser = argparse.ArgumentParser(description='Build animated SKEL pelvis-plus-leg scenes.')
    parser.add_argument('gender', choices=['male', 'female', 'all'])
    parser.add_argument('side', nargs='?', choices=['right', 'left'], default='right')
    parser.add_argument('--force', action='store_true', help='overwrite existing Blender scenes')
    parser.add_argument('--visual', action='store_true', help='use smooth interpolation for viewing')
    args = parser.parse_args()

    blender = find_blender()
    if args.gender == 'all':
        cases = [(gender, side) for gender in ('male', 'female') for side in ('right', 'left')]
    else:
        cases = [(args.gender, args.side)]
    for gender, side in cases:
        run_case(gender, side, blender, args.force, args.visual)
    print(f'\nDone: {len(cases)} pelvis-leg scene(s) in {OUT_DIR}')


if __name__ == '__main__':
    main()