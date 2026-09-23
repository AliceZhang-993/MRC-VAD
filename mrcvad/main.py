"""MRC-VAD: train / evaluate the per-stream log-density networks.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from sklearn import mixture

from models import MLPs, ScoreOrLogDensityNetwork
from dataset import get_dataset

SEED = 1


BRANCHES = ["semantic", "motion", "continuity"]
FEATURE_DIRS = {
    "semantic": "features_RGB_L_offi_gpu",
    "continuity": "continuity_cjhead_base_512d",
}
CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_video_frame_counts(data_root, dataset_name):
    frame_dir = os.path.join(data_root, dataset_name, 'testing/frames')
    video_names = sorted(os.listdir(frame_dir))
    video_frames = []
    for v in video_names:
        video_path = os.path.join(frame_dir, v)
        if os.path.isdir(video_path):
            num_frames = len([f for f in os.listdir(video_path)
                              if f.endswith(('.jpg', '.png'))])
            video_frames.append(num_frames)
    return video_names, video_frames


def get_global_frame_indices(data_root, dataset_name):
    video_names, video_frames = get_video_frame_counts(data_root, dataset_name)

    start_indices = [0]
    for i in range(len(video_frames) - 1):
        start_indices.append(start_indices[-1] + video_frames[i])

    grid_dir = os.path.join(
        data_root, dataset_name, 'testing', FEATURE_DIRS['continuity'])
    if not os.path.isdir(grid_dir):
        return None

    video_start = {name: start_indices[i] for i, name in enumerate(video_names)}
    global_indices = []
    for seq in sorted(os.listdir(grid_dir)):
        seq_dir = os.path.join(grid_dir, seq)
        if seq not in video_start or not os.path.isdir(seq_dir):
            continue
        base = video_start[seq]
        for name in sorted(os.listdir(seq_dir)):
            if name.endswith('.pt'):
                global_indices.append(base + int(os.path.splitext(name)[0]))
    return np.array(global_indices)

def gaussian_video(video, lengths, sigma=3):
    scores = np.zeros_like(video)
    prev = 0
    for cur in lengths:
        scores[prev: cur] = gaussian_filter1d(video[prev: cur], sigma)
        prev = cur
    return scores


def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for cur in lengths:
        cur_auc = roc_auc_score(
            np.concatenate(([0], test_labels[prev: cur], [1])),
            np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)

def object2frame_score(object_score, feature_object_count):
    frame_score = np.zeros(len(feature_object_count))
    pre_count = 0
    for i, count in enumerate(feature_object_count):
        if count > pre_count:
            frame_score[i] = object_score[pre_count:count].max()
        pre_count = count
    return frame_score


def standardize_scores(train_scores, test_scores):
    train_scores = np.asarray(train_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    mean = train_scores.mean(axis=0, keepdims=True)
    std = train_scores.std(axis=0, keepdims=True)
    train_z = (train_scores - mean) / (std + 1e-8)
    test_z = (test_scores - mean) / (std + 1e-8)
    return train_z, test_z


def fit_gmm(data, n_components, reg_covar=1e-6):
    data = np.asarray(data, dtype=float)
    for reg in (reg_covar, max(reg_covar * 1e2, 1e-5), 1e-3, 1e-2):
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type='full', reg_covar=reg)
        try:
            gmm.fit(data)
            return gmm
        except ValueError:
            continue
    raise ValueError(
        f"GMM({n_components}) fit failed: covariance not positive definite for any reg_covar.")


def aggregate_multiscale(train_scores, test_scores, agg_type):
    if agg_type.startswith('gmm'):
        n_components = int(agg_type.split('(')[1].split(')')[0])
        gmm = fit_gmm(train_scores, n_components=n_components)
        return -gmm.score_samples(train_scores), -gmm.score_samples(test_scores)
    if agg_type == 'max':
        return train_scores.max(axis=1), test_scores.max(axis=1)
    if agg_type == 'median':
        return np.median(train_scores, axis=1), np.median(test_scores, axis=1)
    if agg_type == 'mean':
        return train_scores.mean(axis=1), test_scores.mean(axis=1)
    raise ValueError(f"Unknown aggregation: {agg_type}")


def frame_and_video_auc(scores, labels, video_lengths):
    """Frame-level (micro) AUC over all test frames and mean per-video (macro) AUC."""
    return {
        'micro': roc_auc_score(labels, scores),
        'macro': macro_auc(scores, labels, video_lengths),
    }


class Trainer:
    def __init__(self, args):
        self.args = args
        self.features = args.features_list
        self.device = args.device
        self.start_epoch = 0
        self.infer_batch_size = args.infer_batch_size

        self.is_pretrained = {
            feature: feature in args.pretrained_features
            for feature in self.features
        }

        self.load_data()

        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.lr = {}
        for feature in self.features:
            self.models[feature] = self.init_model(self.input_dim[feature])
            if feature in args.pretrained_features:
                self.load_checkpoint(feature, args.pretrained_features[feature])
                self.optimizers[feature] = None
                self.schedulers[feature] = None
            else:
                self.lr[feature] = args.feature_lr[feature]
                self.optimizers[feature] = optim.Adam(
                    self.models[feature].parameters(), lr=self.lr[feature], betas=(0.5, 0.9))
                self.schedulers[feature] = optim.lr_scheduler.StepLR(
                    self.optimizers[feature], step_size=50, gamma=0.9)

    def init_model(self, input_dim):
        return ScoreOrLogDensityNetwork(
            MLPs(
                input_dim=input_dim + 1,  # +1 for the noise conditioning input
                units=self.args.units,
                dropout=self.args.dropout,
                layernorm=self.args.layernorm,
            ),
            score_network=False,
        ).to(self.device)

    def load_data(self):
        self.train_datasets = {}
        self.test_datasets = {}
        self.dataloader_train = {}
        self.dataloader_test = {}
        self.test_video_lengths = {}
        self.labels_test = {}
        self.input_dim = {}
        self.test_object_count = {}
        self.frame_ids = None

        # The continuity branch defines the frame grid (labels, video lengths,
        # frame indices) that the object-level branch is aligned to, so it is
        # loaded first.
        frame_level = [f for f in self.features if f != "motion"]
        grid_branch = "continuity" if "continuity" in frame_level else (
            frame_level[0] if frame_level else None)
        features_order = ([grid_branch] if grid_branch else []) \
            + [f for f in frame_level if f != grid_branch] \
            + (["motion"] if "motion" in self.features else [])

        for feature in features_order:
            print(f"Loading datasets for {feature}...")
            start_time = time.time()
            if feature == "motion":
                feature_path = self.args.motion_root if self.args.motion_root else os.path.join(
                    self.args.data_root, self.args.dataset_name, "motion")
                train_motion = np.load(os.path.join(
                    feature_path, '{}/train/motion.npy'.format(self.args.dataset_name)), allow_pickle=True)
                train_motion = np.concatenate(train_motion, 0)
                train_motion = torch.Tensor(train_motion)

                train_motion_mean = train_motion.mean(dim=0)
                train_motion_std = train_motion.std(dim=0)
                train_motion = (train_motion - train_motion_mean) / (train_motion_std + 1e-8)

                dataset_train_motion = TensorDataset(
                    train_motion, torch.zeros(train_motion.shape[0], dtype=torch.int))
                dataloader_train_motion = DataLoader(
                    dataset_train_motion, shuffle=True, batch_size=self.args.batch_size)

                test_motion = np.load(os.path.join(
                    feature_path, '{}/test/motion.npy'.format(self.args.dataset_name)), allow_pickle=True)
                count = 0
                test_motion_object_count = []
                normalized_test_motion = []
                train_motion_mean_np = train_motion_mean.numpy()
                train_motion_std_np = train_motion_std.numpy()
                for data in test_motion:
                    count += data.shape[0]
                    test_motion_object_count.append(count)
                    if data.ndim == 1:
                        data = data.reshape(1, -1)
                    normalized_data = (data - train_motion_mean_np) / (train_motion_std_np + 1e-8)
                    normalized_test_motion.append(normalized_data)
                self.test_object_count['motion'] = test_motion_object_count
                test_motion = np.concatenate(normalized_test_motion, 0)

                test_motion = torch.Tensor(test_motion)
                dataset_test_motion = TensorDataset(
                    test_motion, torch.zeros(test_motion.shape[0], dtype=torch.int))
                dataloader_test_motion = DataLoader(
                    dataset_test_motion, shuffle=False, batch_size=self.args.batch_size)

                self.input_dim['motion'] = train_motion.reshape(train_motion.shape[0], -1).shape[1]
                self.train_datasets['motion'] = dataset_train_motion
                self.dataloader_train['motion'] = dataloader_train_motion
                self.test_datasets['motion'] = dataset_test_motion
                self.dataloader_test['motion'] = dataloader_test_motion

                print(f"{feature} Dataset loaded in {time.time() - start_time:.2f} seconds")
            else:
                data_train, labels_train, data_test, labels_test, _, _ = get_dataset(
                    data_root=self.args.data_root,
                    dataset_name=self.args.dataset_name,
                    features_name=feature,
                    batchsize=self.args.batch_size,
                )

                data_train = torch.Tensor(data_train).float()
                data_test = torch.Tensor(data_test).float()
                self.input_dim[feature] = data_train.shape[1]

                self.train_datasets[feature] = TensorDataset(
                    data_train, torch.Tensor(labels_train))
                self.dataloader_train[feature] = DataLoader(
                    self.train_datasets[feature], shuffle=True, batch_size=self.args.batch_size)
                self.test_datasets[feature] = TensorDataset(
                    data_test, torch.Tensor(labels_test))
                self.dataloader_test[feature] = DataLoader(
                    self.test_datasets[feature], shuffle=False, batch_size=self.args.batch_size)

                self.test_video_lengths[feature] = np.load(
                    os.path.join(self.args.data_root, self.args.dataset_name, 'test_clip_pt_lengths.npy'))
                self.labels_test[feature] = labels_test

                # the object-level branch shares the frame grid of the grid branch
                if feature == grid_branch:
                    self.frame_ids = get_global_frame_indices(
                        self.args.data_root, self.args.dataset_name)
                    self.labels_test['motion'] = labels_test
                    self.test_video_lengths['motion'] = self.test_video_lengths[feature]

                print(f"{feature} Dataset loaded in {time.time() - start_time:.2f} seconds")
                print(f"Input feature dimension: {data_train.shape[1]}")
                print(f"Number of training samples: {len(data_train)}")
                print(f"Number of test samples: {len(data_test)}")

        missing = [feature for feature in self.features if feature not in self.labels_test]
        if missing:
            raise ValueError(
                "The object-level branch alone has no frame grid; include at least one "
                "frame-level branch (semantic / continuity)."
            )

    def _train_single_feature(self, epoch, feature):
        model = self.models[feature]
        optimizer = self.optimizers[feature]
        dataloader = self.dataloader_train[feature]

        model.train()
        torch.cuda.empty_cache()

        loss_sum = 0.0
        num_batches = 0
        for data in dataloader:
            x = data[0].to(self.device)
            x = x.reshape(x.shape[0], -1)
            dimension = x.shape[1]

            sigma = torch.Tensor(np.exp(
                np.random.uniform(
                    np.log(self.args.sigma_low),
                    np.log(self.args.sigma_high),
                    x.size(0),
                )
            )).unsqueeze(1).to(self.device)

            noise = torch.randn_like(x, device=self.device) * sigma
            x = x.requires_grad_()
            x_ = x + noise

            lambda_factor = (sigma ** 2).ravel()
            score_, log_density_ = model.score(
                torch.hstack([x_, sigma]), return_log_density=True)

            loss = torch.norm(score_[:, :-1] + noise / (sigma ** 2), dim=-1) ** 2
            loss = lambda_factor * loss
            loss = loss.mean() / dimension

            if self.args.beta:
                _, log_density_noise_free = model.score(
                    torch.hstack([x, sigma]), return_log_density=True)
                loss_regularizer = self.args.beta * (log_density_noise_free ** 2).mean() / dimension
                loss = loss + loss_regularizer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            num_batches += 1

        self.schedulers[feature].step()
        avg_loss = loss_sum / max(num_batches, 1)
        print(f"  [{feature}] epoch {epoch}  dsm loss {avg_loss:.4f}")

    def _infer_loader(self, feature, split):
        dataset = self.train_datasets[feature] if split == 'train' else self.test_datasets[feature]
        return DataLoader(dataset, shuffle=False, batch_size=self.infer_batch_size)

    def calculate_scores(self, dataloader, feature):
        """Both score families on the noise ladder -> {name: [N, L]}."""
        model = self.models[feature]
        sigma_L = np.linspace(self.args.sigma_low, self.args.sigma_high, self.args.L)
        sigma_L = torch.tensor(sigma_L, device=self.device, dtype=torch.float32)

        total_samples = len(dataloader.dataset)
        log_density = torch.empty((total_samples, len(sigma_L)), device=self.device)
        score_norm = torch.empty((total_samples, len(sigma_L)), device=self.device)

        model.eval()
        start_idx = 0
        with torch.set_grad_enabled(True):
            for batch in dataloader:
                x = batch[0].to(self.device).reshape(-1, self.input_dim[feature])
                batch_size = x.shape[0]

                x_expanded = x.repeat_interleave(len(sigma_L), dim=0)
                sigmas = sigma_L.repeat(batch_size).unsqueeze(1)
                inputs = torch.cat([x_expanded, sigmas], dim=1)

                score, ld = model.score(inputs, return_log_density=True)
                # norm of the score vector over the feature dimensions, ignoring
                # the noise-conditioning dimension
                sn = torch.norm(score[:, :-1], dim=1)

                end_idx = start_idx + batch_size
                log_density[start_idx:end_idx] = ld.reshape(batch_size, len(sigma_L))
                score_norm[start_idx:end_idx] = sn.reshape(batch_size, len(sigma_L))
                start_idx = end_idx

        return {
            'log_density': log_density.cpu().numpy(),
            'score_norm': score_norm.cpu().numpy(),
        }

    def evaluate_branch(self, feature, scores):
        """Aggregate one stream over both score families and all aggregations.

        The object-level stream is max-pooled to frames and sliced to the frame
        grid of the frame-level streams before the AUC is computed.
        """
        labels = self.labels_test[feature]
        video_lengths = self.test_video_lengths[feature]
        results = {}

        for score_type in ('log_density', 'score_norm'):
            train_scores = np.asarray(scores[score_type]['train'], dtype=float)
            test_scores = np.asarray(scores[score_type]['test'], dtype=float)
            train_std, test_std = standardize_scores(train_scores, test_scores)
            results[score_type] = {}
            for agg_type in ('max', 'median', 'mean', 'gmm(1)', 'gmm(3)', 'gmm(5)'):
                if agg_type.startswith('gmm'):
                    train_agg, test_agg = aggregate_multiscale(
                        train_scores, test_scores, agg_type)
                else:
                    train_agg, test_agg = aggregate_multiscale(
                        train_std, test_std, agg_type)

                if feature == "motion":
                    test_agg = object2frame_score(test_agg, self.test_object_count[feature])
                    if self.frame_ids is not None:
                        test_agg = test_agg[self.frame_ids]

                smoothed = gaussian_video(test_agg, video_lengths, self.args.sigma_len)
                results[score_type][agg_type] = {
                    'train': train_agg,
                    'test': test_agg,
                    'auc': frame_and_video_auc(smoothed, labels, video_lengths),
                }
        return results

    def evaluate_combined(self, branches):
        """Calibrate every stream to a common range with its training
        distribution, sum across streams, and temporally smooth."""
        feature0 = self.features[0]
        video_lengths = self.test_video_lengths[feature0]
        labels = self.labels_test[feature0]
        results = {}

        for score_type in ('log_density', 'score_norm'):
            results[score_type] = {}
            for agg_type in ('max', 'median', 'mean', 'gmm(1)', 'gmm(3)', 'gmm(5)'):
                normalized_scores = []
                for feature in self.features:
                    test_scores = np.asarray(
                        branches[feature][score_type][agg_type]['test'], dtype=float)
                    train_scores = np.asarray(
                        branches[feature][score_type][agg_type]['train'], dtype=float)
                    train_scores = train_scores[np.isfinite(train_scores)]

                    if train_scores.size == 0:
                        normalized_scores.append(np.zeros_like(test_scores))
                        continue

                    min_val = float(np.min(train_scores))
                    max_val = float(np.percentile(train_scores, 99.9))
                    if max_val > min_val:
                        normalized = (test_scores - min_val) / (max_val - min_val)
                    else:
                        normalized = np.zeros_like(test_scores)
                    normalized = np.clip(np.nan_to_num(normalized, nan=0.0), 0.0, 1.0)
                    normalized_scores.append(normalized)

                combined = np.column_stack(normalized_scores).sum(axis=1)
                smoothed = gaussian_video(combined, video_lengths, self.args.sigma_len)
                results[score_type][agg_type] = frame_and_video_auc(smoothed, labels, video_lengths)
        return results

    def evaluate(self):
        branches = {}
        for feature in self.features:
            scores = {
                'train': self.calculate_scores(self._infer_loader(feature, 'train'), feature),
                'test': self.calculate_scores(self._infer_loader(feature, 'test'), feature),
            }
            branches[feature] = self.evaluate_branch(feature, scores)

        combined = None
        if len(self.features) > 1:
            combined = self.evaluate_combined(branches)
        return branches, combined

    def load_checkpoint(self, feature, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        self.models[feature].load_state_dict(torch.load(checkpoint_path)['model'])
        print(f"Loaded weights for stream: {feature}")

    def train(self):
        print("\n" + "=" * 60)
        print(f"Starting training for {self.args.epochs} epochs...")
        print(f"Dataset: {self.args.dataset_name}, features: {self.features}")
        print("=" * 60)
        start_time = time.time()

        for epoch in range(self.start_epoch, self.args.epochs + 1):
            self.train_epoch(epoch)

        print(f"Training completed in {(time.time() - start_time) / 60:.1f} minutes")

    def train_epoch(self, epoch):
        for feature in self.features:
            if not self.is_pretrained[feature]:
                self._train_single_feature(epoch, feature)

    def report(self, branches, combined):
        for feature, results in branches.items():
            print(f"\n[{feature}]")
            for score_type, by_agg in results.items():
                for agg_type, entry in by_agg.items():
                    auc = entry['auc']
                    print(f"  {score_type:<11} {agg_type:<8} "
                          f"micro {auc['micro']:.4f}  macro {auc['macro']:.4f}")
        if combined is not None:
            print("\n[combined]")
            for score_type, by_agg in combined.items():
                for agg_type, auc in by_agg.items():
                    print(f"  {score_type:<11} {agg_type:<8} "
                          f"micro {auc['micro']:.4f}  macro {auc['macro']:.4f}")

    def evaluate_only(self):
        print("Running in evaluation-only mode...")
        branches, combined = self.evaluate()
        self.report(branches, combined)
        print("\nEvaluation completed.")


def main():
    parser = argparse.ArgumentParser(description="MRC-VAD (frame-level VAD)")
    parser.add_argument("--evaluate_only", action="store_true", default=False,
                        help="Run in evaluation-only mode (requires per-stream checkpoints)")

    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, nargs='+', default=[1e-4, 5e-4, 1e-4],
                        help="Per-branch learning rate, in branch order")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--infer_batch_size", type=int, default=8192,
                        help="Inference DataLoader batch size. Does not affect results.")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Regularization factor on the log-density")

    parser.add_argument("--units", nargs='+', default=[4096, 4096], type=int)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--layernorm", action="store_true", default=False)

    parser.add_argument("--data_root", type=str, default="../../dataset/",
                        help="Root containing <dataset>/{training,testing}/...")
    parser.add_argument("--motion_root", type=str, default=None,
                        help="Root holding the object-level branch under "
                             "<root>/<dataset>/{train,test}/motion.npy")
    parser.add_argument("--dataset_name", type=str, default="ped2",
                        choices=["shanghaitech", "ped2", "avenue"])
    parser.add_argument("--features", nargs='+', default=list(BRANCHES),
                        help="Branches to use. Optionally override a branch's checkpoint "
                             "with name:path; otherwise the default under checkpoints/ "
                             "is used in evaluate_only mode.")
    parser.add_argument("--sigma_low", type=float, default=1e-3)
    parser.add_argument("--sigma_high", type=float, default=1.0)
    parser.add_argument("--sigma_len", type=int, default=3,
                        help="Gaussian smoothing sigma for the temporal smoothing. "
                             "ped2: 3; avenue / shanghaitech: 7")
    parser.add_argument("--L", type=int, default=16, help="Number of noise levels")

    args = parser.parse_args()

    args.features_list = []
    args.pretrained_features = {}
    requested = []
    for feature_spec in args.features:
        branch, _, checkpoint_path = feature_spec.partition(':')
        feature_name = FEATURE_DIRS.get(branch, branch)
        requested.append((branch, feature_name))
        args.features_list.append(feature_name)

        if not checkpoint_path and args.evaluate_only:
            checkpoint_path = os.path.join(
                CHECKPOINT_ROOT, args.dataset_name, f"{branch}.pth")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint for branch '{branch}' not found: {checkpoint_path}. "
                    f"Download the pretrained weights (see README) or pass "
                    f"--features {branch}:<path>.")
        if checkpoint_path:
            args.pretrained_features[feature_name] = checkpoint_path

    # --lr is positional over the branch order, independently of which subset
    # of branches is requested.
    lr_values = args.lr if len(args.lr) == len(BRANCHES) else [args.lr[0]] * len(BRANCHES)
    branch_lr = dict(zip(BRANCHES, lr_values))
    args.feature_lr = {feat: branch_lr.get(branch, args.lr[0])
                       for branch, feat in requested}

    missing = [f for f in args.features_list if f not in args.pretrained_features]
    if args.evaluate_only and missing:
        raise ValueError(f"Evaluation mode requires weights for: {', '.join(missing)}")

    set_seed(SEED)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = "cuda:0"

    trainer = Trainer(args)
    if args.evaluate_only:
        trainer.evaluate_only()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
