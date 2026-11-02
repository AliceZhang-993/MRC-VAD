import glob
import os
import random
from typing import Any, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class ContinuityDataset(Dataset):
    """Yields the first-stage continuity sample for every clip start.

    Per clip start the dataset returns a continuous clip (Vc), a discontinuous
    clip with one frame dropped (Vd), the dropped frame on its own (Im) and a
    clip taken from another video (Vco), each with its validity mask and the
    labels needed by the three proxy losses.
    """

    def __init__(self, video_dir: str, config: Dict[str, Any]):
        self.cl_cfg = config.get("continuity_learning", {})
        self.clip_length = self.cl_cfg["clip_length"]

        self.video_sequences = sorted(os.listdir(video_dir))
        self.seq_to_frames = {
            seq: sorted(glob.glob(os.path.join(video_dir, seq, "*.jpg")))
            for seq in self.video_sequences
        }

        self.samples = []
        for seq in self.video_sequences:
            frames = self.seq_to_frames[seq]
            for start in range(len(frames) - self.clip_length + 1):
                self.samples.append((seq, start))

        gm_cfg = self.cl_cfg.get("gradient_masking", {})
        self.padding_enabled = gm_cfg.get("enable", True)
        self.vd_padding_mode = gm_cfg.get("padding", {}).get("discontinuous", "front")
        self.im_padding_mode = gm_cfg.get("padding", {}).get("missing_frame", "duplicate")

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frames(self, frame_paths):
        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Missing frame: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            frames.append(img)
        return np.stack(frames)

    def _to_tensor_5d(self, x: np.ndarray):
        tensor = torch.from_numpy(x).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(224, 224), mode='bilinear', align_corners=False)
        return tensor

    def _build_discontinuous(self, frames):
        t = len(frames)
        missing_idx = random.randint(1, t - 2)
        vd_indices = list(range(t))
        vd_indices.pop(missing_idx)
        return vd_indices, missing_idx

    def __getitem__(self, idx: int):
        seq, start = self.samples[idx]

        vc_frames = self.seq_to_frames[seq][start: start + self.clip_length]
        vc_imgs = self._read_frames(vc_frames)
        vc_mask = np.ones((1, self.clip_length, 1, 1), dtype=np.float32)

        # a second, non-overlapping clip of the same video provides the basis for
        # the negative samples
        possible_starts = [
            s for s in range(len(self.seq_to_frames[seq]) - self.clip_length + 1)
            if abs(s - start) >= self.clip_length
        ]
        if not possible_starts:
            possible_starts = [
                s for s in range(len(self.seq_to_frames[seq]) - self.clip_length + 1)
                if s != start
            ]
        rand_start = random.choice(possible_starts)
        initial_frames = self.seq_to_frames[seq][rand_start: rand_start + self.clip_length]

        vd_indices, im_idx = self._build_discontinuous(initial_frames)
        vd_imgs = self._read_frames([initial_frames[i] for i in vd_indices])
        if self.padding_enabled:
            vd_padded_imgs = np.concatenate([vd_imgs[0:1], vd_imgs], axis=0)
            vd_mask = np.ones((1, self.clip_length, 1, 1), dtype=np.float32)
            vd_mask[:, 0, :, :] = 0
        else:
            vd_padded_imgs = vd_imgs
            vd_mask = np.ones((1, self.clip_length - 1, 1, 1), dtype=np.float32)

        im_imgs = self._read_frames([initial_frames[im_idx]])
        if self.padding_enabled:
            im_padded_imgs = np.repeat(im_imgs, self.clip_length, axis=0)
            im_mask = np.zeros((1, self.clip_length, 1, 1), dtype=np.float32)
            im_mask[:, 0, :, :] = 1
        else:
            im_padded_imgs = im_imgs
            im_mask = np.ones((1, 1, 1, 1), dtype=np.float32)

        other_seq = random.choice([s for s in self.video_sequences if s != seq])
        other_start = random.randint(0, len(self.seq_to_frames[other_seq]) - self.clip_length)
        vco_imgs = self._read_frames(
            self.seq_to_frames[other_seq][other_start: other_start + self.clip_length])
        vco_mask = np.ones((1, self.clip_length, 1, 1), dtype=np.float32)

        vc = self._to_tensor_5d(vc_imgs).permute(1, 0, 2, 3)
        vd = self._to_tensor_5d(vd_padded_imgs).permute(1, 0, 2, 3)
        im = self._to_tensor_5d(im_padded_imgs).permute(1, 0, 2, 3)
        vco = self._to_tensor_5d(vco_imgs).permute(1, 0, 2, 3)

        return (
            vc, vd, im, vco,
            torch.from_numpy(vc_mask), torch.from_numpy(vd_mask),
            torch.from_numpy(im_mask), torch.from_numpy(vco_mask),
            torch.tensor(1), torch.tensor(0),
            torch.tensor(im_idx - 1),
            seq, start, im_idx
        )
