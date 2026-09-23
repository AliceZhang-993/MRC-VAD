import glob
import os

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset


class Label_loader:
    """Per-video frame ground truth, ordered by video folder name."""

    def __init__(self, cfg, video_folders):
        self.cfg = cfg
        self.dataset_name = cfg["dataset_name"]
        self.video_folders = sorted(video_folders)
        if self.dataset_name == 'shanghaitech':
            self.label_dir = os.path.join(
                cfg["data_root"], 'shanghaitech', 'testing', 'test_frame_mask')
        else:
            self.mat_path = os.path.join(
                cfg["data_root"], self.dataset_name, f'{self.dataset_name}.mat')

    def __call__(self):
        if self.dataset_name == 'shanghaitech':
            return self._load_shanghaitech()
        return self._load_ucsd_avenue()

    def _load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']
        assert len(self.video_folders) == abnormal_events.shape[0], \
            f"numbers of videos not match: {len(self.video_folders)} vs {abnormal_events.shape[0]}"

        all_gt = []
        for video_idx, video_path in enumerate(self.video_folders):
            frame_count = len(os.listdir(video_path))
            video_gt = np.zeros(frame_count, dtype=np.int8)

            abnormal_ranges = abnormal_events[video_idx]
            if abnormal_ranges.ndim == 1:
                abnormal_ranges = np.expand_dims(abnormal_ranges, axis=1)

            for range_idx in range(abnormal_ranges.shape[1]):
                start = abnormal_ranges[0, range_idx] - 1  # MATLAB -> Python indexing
                end = abnormal_ranges[1, range_idx]
                video_gt[start:end] = 1

            all_gt.append(video_gt)
        return all_gt

    def _load_shanghaitech(self):
        npy_files = sorted(glob.glob(os.path.join(self.label_dir, '*.npy')))
        assert len(self.video_folders) == len(npy_files), \
            f"video count mismatch: {len(self.video_folders)} vs {len(npy_files)}"

        for npy_path, video_path in zip(npy_files, self.video_folders):
            video_name = os.path.basename(video_path)
            npy_name = os.path.basename(npy_path).replace('label', '').replace('.npy', '')
            assert video_name == npy_name, f"video order mismatch: {video_name} vs {npy_name}"

        return [np.load(npy) for npy in npy_files]


class FeatureAnomalyDataset(Dataset):
    """Features with their labels, optionally standardized."""

    def __init__(self, feature_dir, config, mode, test_video_dir=None, mean=None, std=None):
        self.feature_root = feature_dir
        self.config = config
        self.mode = mode
        self.mean = mean
        self.std = std
        self.test_video_dir = test_video_dir

        self.feature_paths = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(self.feature_root)
            for f in files if f.endswith(".pt")
        ])

        if self.mode == "test":
            self._dynamic_label_mapping()

        self._auto_standardize = (mean is not None) and (std is not None)

    def _dynamic_label_mapping(self):
        original_labels = Label_loader(self.config, self.test_video_dir)()
        self.label_dict = {
            os.path.basename(d): original_labels[i]
            for i, d in enumerate(self.test_video_dir)
        }

        self.label_map = []
        for fp in self.feature_paths:
            data = torch.load(fp)
            meta = data['metadata']
            vs = meta['video_seq']
            fidx = meta['frame_idx']
            if fidx < len(self.label_dict[vs]):
                self.label_map.append(int(self.label_dict[vs][fidx]))
            else:
                self.label_map.append(0)
            del data

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        data = torch.load(self.feature_paths[idx])
        meta = data['metadata']
        feature = data['feature']

        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature, dtype=torch.float32)

        if self._auto_standardize:
            assert feature.shape == self.mean.shape, "feature / mean dimension mismatch"
            safe_std = torch.maximum(self.std, torch.tensor(1e-8))
            feature = (feature - self.mean) / safe_std

        item = {"feature": feature, "metadata": meta}
        if self.mode == "test":
            item["label"] = self.label_map[idx]
        return item


def _compute_stats(feature_dir, config, mode):
    dataset = FeatureAnomalyDataset(
        feature_dir=feature_dir, config=config, mode=mode, mean=None, std=None)
    features = torch.stack([dataset[i]["feature"] for i in range(len(dataset))])
    return features.mean(dim=0), features.std(dim=0)


def build_dataloader(config, mode="train"):
    """Build the frame-level DataLoader.

    In train mode the mean/std are computed from the training split; in test
    mode the caller supplies those training statistics via config['mean']/['std'].
    """
    dataset_name = config["dataset_name"]
    features_name = config["features_name"]
    data_root = config["data_root"]

    feature_dir = os.path.join(
        data_root, dataset_name,
        "training" if mode == "train" else "testing", features_name)

    test_video_dir = None
    if mode == "test":
        video_dir = os.path.join(data_root, dataset_name, "testing", "frames")
        test_video_dir = sorted([
            os.path.join(video_dir, v) for v in os.listdir(video_dir)
            if os.path.isdir(os.path.join(video_dir, v))])

    if mode == "test" and "mean" in config and "std" in config:
        mean, std = config["mean"], config["std"]
    else:
        mean, std = _compute_stats(feature_dir, config, mode)

    dataset = FeatureAnomalyDataset(
        feature_dir=feature_dir,
        config=config,
        mode=mode,
        test_video_dir=test_video_dir,
        mean=mean,
        std=std,
    )

    sampler = (torch.utils.data.RandomSampler(dataset) if mode == "train"
               else torch.utils.data.SequentialSampler(dataset))

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"] if mode == "train" else config["val_batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ), mean, std
