"""Build the continuity representation by concatenating the I3D backbone
embedding and the Continuity-head embedding written by exps/exp3/evaluate.py.

Input (output of "python main_epoch.py --mode test --exp exp3"):
  <root>/<dataset>/<split>/continuity/<seq>/<frame>.pt
  <root>/<dataset>/<split>/cjhead_base_512d/<seq>/<frame>.pt

Output:
  <root>/<dataset>/<split>/continuity_cjhead_base_512d/<seq>/<frame>.pt

Only frames present in both source dirs are written.

Example:
  python data/combine_backbone_head.py --root DATA_ROOT --dataset avenue
"""

import argparse
import glob
import os

import torch
from tqdm import tqdm

BACKBONE_DIR = "continuity"
HEAD_DIR = "cjhead_base_512d"
TARGET_DIR = "continuity_cjhead_base_512d"


def combine_split(dataset_root, dataset_name, split):
    split_path = os.path.join(dataset_root, dataset_name, split)
    backbone_root = os.path.join(split_path, BACKBONE_DIR)
    head_root = os.path.join(split_path, HEAD_DIR)
    target_root = os.path.join(split_path, TARGET_DIR)

    for src in (backbone_root, head_root):
        if not os.path.isdir(src):
            raise FileNotFoundError(f"missing source feature dir: {src}")

    video_dirs = sorted(
        d for d in os.listdir(backbone_root)
        if os.path.isdir(os.path.join(backbone_root, d)))

    total = 0
    for video in tqdm(video_dirs, desc=f"combining {dataset_name}/{split}"):
        backbone_files = sorted(glob.glob(os.path.join(backbone_root, video, "*.pt")))
        target_video = os.path.join(target_root, video)
        os.makedirs(target_video, exist_ok=True)

        for bf in backbone_files:
            name = os.path.basename(bf)
            hf = os.path.join(head_root, video, name)
            if not os.path.exists(hf):
                continue
            backbone = torch.load(bf)["feature"]
            head = torch.load(hf)["feature"]
            combined = torch.cat([backbone, head], dim=0)
            meta = torch.load(bf)["metadata"]
            torch.save({"feature": combined, "metadata": meta},
                       os.path.join(target_video, name))
            total += 1
    print(f"[{dataset_name}/{split}] wrote {total} frames -> {TARGET_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../../../../../dataset",
                        help="dataset root containing "
                             "<root>/<dataset>/{training,testing}/<feature>/<seq>/*.pt")
    parser.add_argument("--dataset", type=str, default="avenue",
                        choices=["ped2", "avenue", "shanghaitech"])
    parser.add_argument("--splits", type=str, nargs="*",
                        default=["training", "testing"])
    args = parser.parse_args()
    for split in args.splits:
        combine_split(os.path.abspath(args.root), args.dataset, split)


if __name__ == "__main__":
    main()
