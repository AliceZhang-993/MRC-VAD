import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEAverageLoss(nn.Module):
    """Binary cross-entropy over logits, averaged over the batch."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class MultiClassCEAverageLoss(nn.Module):
    """Multi-class cross-entropy over logits, averaged over the batch."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets)


class TripletContrastiveLoss(nn.Module):
    """Triplet loss (Vd, Im, Vc) plus a contrastive term that pulls Vd towards
    the same-video clip Vc and pushes it away from the other-video clip Vco."""

    def __init__(self, omega: float = 0.5, temperature: float = 0.1):
        super().__init__()
        self.omega = omega
        self.tau = temperature
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1)

    def forward(self, anchor_vd, pos_im, neg_vc, pos_vc, neg_vco) -> torch.Tensor:
        loss_triplet = self.triplet(anchor_vd, pos_im, neg_vc)

        a = self._normalize(anchor_vd)
        p = self._normalize(pos_vc)
        n = self._normalize(neg_vco)
        logits = torch.stack([
            (a * p).sum(dim=-1) / self.tau,
            (a * n).sum(dim=-1) / self.tau,
        ], dim=-1)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss_contrast = F.cross_entropy(logits, targets, reduction='mean')

        return self.omega * loss_triplet + (1 - self.omega) * loss_contrast
