import os
import sys
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.distributed import dist_manager


class BaseProjectionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 512):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(512)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            t, h, w = x.shape[2], x.shape[3], x.shape[4]
            if mask.shape[2] != t or mask.shape[3] != h or mask.shape[4] != w:
                mask = F.interpolate(mask, size=(t, h, w), mode='nearest')
            x = x * mask
        x = self.relu(self.bn(self.conv(x)))
        return self.pool(x).flatten(1)


class ContinuityHead(nn.Module):
    """Continuity proxy: is the clip temporally continuous."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.base = BaseProjectionHead(in_channels)
        self.fc = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fc(self.base(x, mask))


class DiscontinuityHead(nn.Module):
    """Discontinuity proxy: which position was dropped from the clip."""

    def __init__(self, in_channels: int, clip_length: int):
        super().__init__()
        self.base = BaseProjectionHead(in_channels)
        self.fc = nn.Linear(512, clip_length - 2)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fc(self.base(x, mask))


class MissingFrameHead(nn.Module):
    """Missing-frame proxy: embed the dropped frame and the padded clip."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.base = BaseProjectionHead(in_channels)
        self.fc = nn.Linear(512, 128)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fc(self.base(x, mask))


class I3DBackbone(nn.Module):
    """I3D-RGB backbone, returning the feature map before global pooling."""

    LAYERS = [
        'Conv3d_1a_7x7', 'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 'MaxPool3d_3a_3x3',
        'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3',
        'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f',
        'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c',
    ]

    def __init__(self, pretrained: bool = True, output_channels: int = 1024,
                 repo_path: Optional[str] = None, pretrained_path: Optional[str] = None):
        super().__init__()
        self.output_channels = output_channels

        if repo_path and os.path.isdir(repo_path):
            sys.path.insert(0, repo_path)
        from i3d import InceptionI3d

        model = InceptionI3d(400, in_channels=3)
        if pretrained and pretrained_path and os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location="cpu")
            state_dict = state.get('state_dict', state)
            cleaned = {}
            for k, v in state_dict.items():
                k2 = k.replace('module.', '')
                if k2.startswith('logits') or 'Logits' in k2 or 'AvgPool' in k2:
                    continue
                cleaned[k2] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if dist_manager.is_master:
                print(f"[I3D] loaded pretrained weights; "
                      f"missing={len(missing)}, unexpected={len(unexpected)}")
        self.backbone = model
        self.feature_channels = 1024

        if self.feature_channels != output_channels:
            self.proj = nn.Conv3d(self.feature_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

        self.register_buffer("norm_mean", torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.225, 0.225, 0.255]).view(1, 3, 1, 1, 1))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        y = x
        for name in self.LAYERS:
            if not hasattr(self.backbone, name):
                raise RuntimeError(f'I3D backbone missing module: {name}')
            y = getattr(self.backbone, name)(y)
        return self.proj(y)


class ContinuityModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.exp_cfg = cfg["model"] if "model" in cfg else cfg
        cl_cfg = cfg.get("continuity_learning", {})
        self.clip_length = cl_cfg["clip_length"]

        self.backbone = I3DBackbone(
            pretrained=True,
            output_channels=1024,
            repo_path=cl_cfg.get("i3d_repo_path"),
            pretrained_path=cl_cfg.get("pretrained_path"),
        )

        self.enable_cj = cl_cfg.get("ContinuityHead", True)
        self.enable_dl = cl_cfg.get("DiscontinuityHead", True)
        self.enable_mfe = cl_cfg.get("MissingFrameHead", True)

        in_ch = 1024
        if self.enable_cj:
            self.cj_head = ContinuityHead(in_ch)
        if self.enable_dl:
            self.dl_head = DiscontinuityHead(in_ch, self.clip_length)
        if self.enable_mfe:
            self.mfe_head = MissingFrameHead(in_ch)

        self.learning_rate = self.exp_cfg.get("learning_rate", 3e-4)
        self.optimizer = None
        self.unfreeze_cfg = cl_cfg.get("unfreeze_schedule", {})

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate,
            weight_decay=self.exp_cfg.get("weight_decay", 0.01))
        lr = dist_manager.scaled_lr(self.learning_rate)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return self.optimizer

    def apply_unfreeze_policy(self, current_epoch: int, dataset_name: str):
        """Stage 1 unfreezes the higher layers, stage 2 the rest on the listed
        datasets."""
        stage1_epoch = self.unfreeze_cfg.get("stage1_epoch", 1)
        stage2_epoch = self.unfreeze_cfg.get("stage2_epoch", 11)
        stage1_layers = self.unfreeze_cfg.get("stage1_layers", [])
        stage2_layers = self.unfreeze_cfg.get("stage2_layers", [])
        stage2_datasets = set(self.unfreeze_cfg.get("stage2_datasets", []))

        for p in self.backbone.parameters():
            p.requires_grad = False

        if current_epoch >= stage1_epoch:
            for name, p in self.backbone.named_parameters():
                if any(layer in name for layer in stage1_layers):
                    p.requires_grad = True

        if current_epoch >= stage2_epoch and dataset_name in stage2_datasets:
            for name, p in self.backbone.named_parameters():
                if any(layer in name for layer in stage2_layers):
                    p.requires_grad = True

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward_heads(self, feat_vc, feat_vd, feat_im=None, feat_vco=None,
                      mask_vc=None, mask_vd=None, mask_im=None, mask_vco=None) -> dict:
        outputs = {}
        if self.enable_cj:
            outputs["cj_logits_pos"] = self.cj_head(feat_vc, mask_vc)
            outputs["cj_logits_neg"] = self.cj_head(feat_vd, mask_vd)

        if self.enable_dl:
            outputs["dl_logits"] = self.dl_head(feat_vd, mask_vd)

        if self.enable_mfe and feat_im is not None:
            outputs["mfe_vd"] = self.mfe_head(feat_vd, mask_vd)
            outputs["mfe_im"] = self.mfe_head(feat_im, mask_im)
            outputs["mfe_vc"] = self.mfe_head(feat_vc, mask_vc)
            if feat_vco is not None:
                outputs["mfe_vco"] = self.mfe_head(feat_vco, mask_vco)
        return outputs
