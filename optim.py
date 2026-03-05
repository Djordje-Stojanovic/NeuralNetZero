"""Muon optimizer + optimizer builder for NeuralNetZero."""

import torch
from torch.optim import AdamW


class Muon(torch.optim.Optimizer):
    """Muon optimizer: Nesterov momentum + Newton-Schulz orthogonalization.

    Only for 2D weight matrices. Uses orthogonalized updates for better
    compute efficiency (from modded-nanogpt / Moonlight).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01,
                 ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(G: torch.Tensor, steps: int) -> torch.Tensor:
        """Approximate matrix square root inverse via Newton-Schulz iteration.

        Optimized coefficients from modded-nanogpt.
        """
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.float()
        # Normalize
        X = X / (X.norm() + 1e-7)
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True
        else:
            transposed = False
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                # Nesterov momentum
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum)

                # Newton-Schulz orthogonalization
                g = self._newton_schulz(g, ns_steps)
                g = g.to(p.data.dtype)

                # Scale by dimensions
                scale = max(g.shape[0], g.shape[1]) ** 0.5
                p.data.mul_(1 - lr * wd)
                p.data.add_(g, alpha=-lr * scale)

        return loss


def build_optimizer(model, train_config):
    """Split params: Muon for 2D weights, AdamW for embeddings/norms."""
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and "tok_emb" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    optimizers = []

    if muon_params:
        optimizers.append(Muon(
            muon_params,
            lr=train_config.muon_lr,
            weight_decay=train_config.muon_weight_decay,
        ))

    if adamw_params:
        optimizers.append(AdamW(
            adamw_params,
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95),
        ))

    return optimizers
