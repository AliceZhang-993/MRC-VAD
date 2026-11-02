import os
import time

import torch
import torch.nn.functional as F

from data.dataloader import ContinuityDataset
from models.continuity_model import ContinuityModel
from utils.compatibility import pt_compat
from utils.distributed import dist_manager

BACKBONE_DIR = "continuity"
CJ_HEAD_DIR = "cjhead_base_512d"


def build_dataloader(config, mode="test", split="testing"):
    video_dir = config["data_root"] + config["dataset_name"] + f"/{split}/frames/"
    dataset = ContinuityDataset(video_dir, config)
    sampler = dist_manager.get_sampler(dataset, shuffle=False, drop_last=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )


class InferenceEngine:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.use_amp = self.config["test"]["mixed_precision"] == "fp16"
        self.split = self.config["test"].get("infer_split", "testing")
        self.test_loader = build_dataloader(self.config["data"], mode="test", split=self.split)

    def _save_feature(self, feature_tensor, video_seq, mid_frame, save_path):
        torch.save({
            'feature': feature_tensor.detach().cpu(),
            'metadata': {'video_seq': video_seq, 'frame_idx': mid_frame},
        }, save_path)

    def _infer_pad_width(self, seq_dir):
        files = [f for f in os.listdir(seq_dir) if f.endswith('.jpg')]
        if not files:
            return 4
        name = os.path.splitext(sorted(files)[0])[0]
        digits = ''.join(ch for ch in name if ch.isdigit())
        return len(digits) if digits else 4

    def _derive_feature_path(self, split, head_dir, seq, frame_idx):
        base = os.path.join(self.config['data']['data_root'],
                            self.config['data']['dataset_name'], split)
        pad_w = self._infer_pad_width(os.path.join(base, 'frames', seq))
        feat_dir = os.path.join(base, head_dir, seq)
        os.makedirs(feat_dir, exist_ok=True)
        return os.path.join(feat_dir, f"{frame_idx:0{pad_w}d}.pt")

    def validate(self, checkpoint_path=None, model=None):
        if dist_manager.is_master:
            print(time.strftime('[%Y-%m-%d %H:%M:%S] continuity extraction start', time.localtime()))

        if model is None:
            model = self.load_model(checkpoint_path)
        model = pt_compat.move_data_to_device(model, dist_manager.device)
        raw_model = model
        model.eval()

        clip_length = self.config["continuity_learning"]["clip_length"]

        with torch.no_grad():
            for batch in self.test_loader:
                (vc, _, _, _, vc_mask, _, _, _, _, _, _, seqs, starts, _) = batch
                vc = pt_compat.move_data_to_device(vc, dist_manager.device)
                vc_mask = pt_compat.move_data_to_device(vc_mask, dist_manager.device)

                feat = raw_model.forward_backbone(vc)

                B = feat.size(0)
                mids = [starts[i].item() + clip_length // 2 for i in range(B)]

                backbone_feat = F.adaptive_avg_pool3d(feat, (1, 1, 1)).flatten(start_dim=1)
                cj_feat = raw_model.cj_head.base(feat, vc_mask)

                for i in range(B):
                    video_seq = seqs[i]
                    mid_frame = mids[i]
                    self._save_feature(
                        backbone_feat[i], video_seq, mid_frame,
                        self._derive_feature_path(self.split, BACKBONE_DIR, video_seq, mid_frame))
                    self._save_feature(
                        cj_feat[i], video_seq, mid_frame,
                        self._derive_feature_path(self.split, CJ_HEAD_DIR, video_seq, mid_frame))

        return None, None, None, None

    def load_model(self, checkpoint_path=None):
        model = ContinuityModel(self.config)
        model = pt_compat.move_data_to_device(model, dist_manager.device)
        if checkpoint_path:
            checkpoint = pt_compat.load_checkpoint(checkpoint_path, model)
            return checkpoint['model']
        return model
