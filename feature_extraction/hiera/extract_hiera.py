"""Frame-level RGB features from a Hiera backbone.

Output: <root>/<dataset>/<split>/features_RGB_L_offi_gpu/<video>/<frame>.pt
with {"feature": Tensor[D], "metadata": {"video_seq", "frame_idx"}}.

Example:
  python extract_hiera.py --root DATA_ROOT --dataset avenue --split training --gpu 0
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

import hiera

NORMALIZE_MEAN = (0.45, 0.45, 0.45)
NORMALIZE_STD = (0.225, 0.225, 0.255)
OUTPUT_FEATURE_DIR = "features_RGB_L_offi_gpu"


def list_samples(frame_dir, sliding_window=False):
    samples = []
    for seq in sorted(os.listdir(frame_dir)):
        seq_dir = os.path.join(frame_dir, seq)
        if not os.path.isdir(seq_dir):
            continue
        paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))

        if sliding_window:
            for start in range(len(paths) - 16 + 1):
                clip = paths[start:start + 16]
                samples.append((seq, clip, clip[8]))
        else:
            for path in paths:
                samples.append((seq, [path], path))
    return samples


class HieraExtractor:
    def __init__(self, gpu=0):
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        self.model = hiera.hiera_large_16x224(pretrained=True, checkpoint="mae_k400_ft_k400")
        self.model.head = nn.Identity()
        self.model.eval().to(self.device)

        self.gpu_transforms = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float16, scale=True),
            transforms.Resize((224, 224),
                              interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.Normalize(list(NORMALIZE_MEAN), list(NORMALIZE_STD)),
        ])

    def _load_clip(self, clip_paths):
        images = []
        for path in clip_paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Missing frame: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(img, (256, 256)))

        clip = np.stack(images)
        if len(clip_paths) == 1:
            clip = np.repeat(clip, 16, axis=0)
        return clip

    def extract_clip(self, clip_paths):
        arr = self._load_clip(clip_paths)
        tensor = torch.as_tensor(arr).to(self.device)
        tensor = self.gpu_transforms(tensor.permute(0, 3, 1, 2))
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda"):
            feat = self.model(tensor)
        return feat[0].float().cpu()


def save_feature(feature, seq, frame_idx, save_path):
    torch.save({
        "feature": feature,
        "metadata": {"video_seq": seq, "frame_idx": frame_idx},
    }, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="../../dataset/",
                        help="dataset root containing <root>/<dataset>/<split>/frames/<video>/*.jpg")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ped2", "avenue", "shanghaitech"])
    parser.add_argument("--split", type=str, required=True, choices=["training", "testing"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sliding-window", action="store_true", default=False)
    args = parser.parse_args()

    frames_root = os.path.join(args.root, args.dataset, args.split, "frames")
    if not os.path.isdir(frames_root):
        raise FileNotFoundError(f"frames dir not found: {frames_root}")

    out_root = os.path.join(args.root, args.dataset, args.split, OUTPUT_FEATURE_DIR)
    samples = list_samples(frames_root, sliding_window=args.sliding_window)
    print(f"{len(samples)} samples | out -> {out_root}")

    extractor = HieraExtractor(args.gpu)

    n_saved = 0
    for i, (seq, clip_paths, labelled_path) in enumerate(samples):
        feat = extractor.extract_clip(clip_paths)
        seq_dir = os.path.join(out_root, seq)
        os.makedirs(seq_dir, exist_ok=True)
        frame_name = os.path.splitext(os.path.basename(labelled_path))[0]
        save_feature(feat, seq, int(frame_name), os.path.join(seq_dir, f"{frame_name}.pt"))
        n_saved += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(samples)} | saved {n_saved}")
    print(f"Done. {n_saved} features saved under {out_root}")


if __name__ == "__main__":
    main()
