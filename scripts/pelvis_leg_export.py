"""Export the pelvis and one leg through the standard SKEL knee-bend motion."""
import argparse
import os

from knee_bend_export import export_knee_bend


def main():
    parser = argparse.ArgumentParser(description='Export an animated SKEL pelvis and one leg for Blender.')
    parser.add_argument('--gender', choices=['male', 'female'], default='male')
    parser.add_argument('--side', choices=['right', 'left'], default='right')
    args = parser.parse_args()

    output = os.path.join('output', 'pelvis', f'pelvis_leg_anim_{args.gender}_{args.side}.npz')
    export_knee_bend(args.gender, args.side, output_path=output, include_pelvis=True)


if __name__ == '__main__':
    main()