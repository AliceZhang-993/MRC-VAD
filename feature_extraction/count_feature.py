"""Write the per-video cumulative frame counts of a feature directory.

main.py reads <data_root>/<dataset>/{train,test}_clip_pt_lengths.npy to know how
many frames each video contributes to the concatenated score array. Those
lengths must match the extracted feature files, so they are counted from the
feature directory itself rather than from the raw frames.

Example:
  python count_feature.py --root DATA_ROOT --dataset avenue
"""

import argparse
import glob
import os

import numpy as np


def count_split(feature_dir):
    """Cumulative .pt counts per video, in sorted video order."""
    lengths = []
    total = 0
    for video in sorted(glob.glob(os.path.join(feature_dir, '*'))):
        if not os.path.isdir(video):
            continue
        total += len(glob.glob(os.path.join(video, '*.pt')))
        lengths.append(total)
    return lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../dataset',
                        help='dataset root containing <dataset>/<split>/<feature>/')
    parser.add_argument('--dataset', type=str, default='ped2',
                        choices=['ped2', 'avenue', 'shanghaitech'])
    parser.add_argument('--feature', type=str, default='features_RGB_L_offi_gpu',
                        help='feature directory the frame grid is taken from')
    args = parser.parse_args()

    dataset_root = os.path.join(os.path.abspath(args.root), args.dataset)

    for split, name in (('training', 'train_clip_pt_lengths.npy'),
                        ('testing', 'test_clip_pt_lengths.npy')):
        feature_dir = os.path.join(dataset_root, split, args.feature)
        if not os.path.isdir(feature_dir):
            raise FileNotFoundError(f'feature directory not found: {feature_dir}')
        lengths = count_split(feature_dir)
        np.save(os.path.join(dataset_root, name), lengths)
        print(f'{args.dataset}/{split}: {len(lengths)} videos, '
              f'{lengths[-1]} frames -> {name}')


if __name__ == '__main__':
    main()
