import torch
import torch.nn as nn


class MLPs(nn.Module):
    """Feed-forward MLP with GELU activations."""

    def __init__(self, input_dim=2, output_dim=1, units=(4096, 4096),
                 layernorm=False, dropout=None, last_activation=nn.Identity()):
        super().__init__()
        layers = []
        in_dim = input_dim
        self.layernorm = layernorm

        def block(in_, out_):
            return nn.Sequential(
                nn.Linear(in_, out_),
                nn.LayerNorm(out_) if self.layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout) if dropout else nn.Identity(),
            )

        for out_dim in units:
            layers.append(block(in_dim, out_dim))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(last_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ScoreOrLogDensityNetwork(nn.Module):
    """Noise-conditioned log-density network; score = -grad_x(log-density)."""

    def __init__(self, net, score_network=False):
        super().__init__()
        self.network = net
        self.is_score_network = score_network

    def forward(self, x):
        return self.network(x)

    def score(self, x, return_log_density=False):
        score, log_density = None, None
        if self.is_score_network:
            score = self.network(x)
            if return_log_density:
                log_density = torch.zeros_like(score[:, 0][:, None])
        else:
            if not x.requires_grad:
                x = x.requires_grad_(True)
            log_density = self.network(x)
            if self.training:
                logp = -log_density.sum()
                score = torch.autograd.grad(
                    logp, x, create_graph=True, retain_graph=True)[0]
            else:
                with torch.enable_grad():
                    logp = -log_density.sum()
                    score = torch.autograd.grad(
                        logp, x, create_graph=False, retain_graph=False)[0]
        if not self.training:
            score = score.detach()
            log_density = log_density.detach()

        if return_log_density:
            return score, log_density
        return score
