"""Reconstruction training for the top-k SAE.

Task labels never enter this loop: the primary SAE experiment tests whether
reconstruction-only feature learning happens to preserve the task class.
Includes dead-feature monitoring and optional resampling.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from models.sae import TopKSAE
from utils.seeds import child_seed


@dataclass
class ReconstructionTrainConfig:
    epochs: int = 200
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    eval_every: int = 10
    patience: int = 10
    dead_window_epochs: int = 10
    resample_dead: bool = True
    resample_every: int = 25
    seed: int = 0
    device: str = "cpu"  # resolved via utils.device; results always saved from CPU


@dataclass
class ReconstructionTrainResult:
    status: str  # completed | diverged
    best_val_mse: float
    history: list[dict] = field(default_factory=list)
    dead_feature_count: int = 0
    epochs_run: int = 0


@torch.no_grad()
def _val_mse(model: torch.nn.Module, x_val: torch.Tensor, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(x_val), batch_size):
        batch = x_val[start : start + batch_size]
        _, recon = model(batch)
        total += float(((recon - batch) ** 2).mean()) * len(batch)
        count += len(batch)
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def _resample_dead_features(model: TopKSAE, x: torch.Tensor, dead: np.ndarray, seed: int) -> None:
    """Reinitialize dead dictionary elements toward badly reconstructed inputs."""
    if not dead.any():
        return
    _, recon = model(x)
    errors = ((recon - x) ** 2).mean(dim=1)
    order = errors.argsort(descending=True)
    gen = torch.Generator().manual_seed(child_seed(seed, "resample"))
    pool = order[: max(len(dead), 32)]
    choice = pool[torch.randint(len(pool), (int(dead.sum()),), generator=gen).to(pool.device)]
    directions = x[choice] - model.b_dec
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
    dead_idx = torch.from_numpy(np.flatnonzero(dead)).to(model.w_enc.device)
    alive_mask = torch.from_numpy(~dead).to(model.w_enc.device)
    alive_norm = model.w_enc[:, alive_mask].norm(dim=0).mean().clamp_min(1e-3)
    model.w_enc[:, dead_idx] = directions.T * 0.2 * alive_norm
    model.b_enc[dead_idx] = 0.0
    model.w_dec[dead_idx] = directions


def train_reconstruction(
    model: TopKSAE,
    x_train: np.ndarray,
    x_val: np.ndarray,
    cfg: ReconstructionTrainConfig,
) -> ReconstructionTrainResult:
    torch.manual_seed(child_seed(cfg.seed, "train_reconstruction"))
    device = torch.device(cfg.device)
    model.to(device)
    xt = torch.from_numpy(np.asarray(x_train, dtype=np.float32)).to(device)
    xv = torch.from_numpy(np.asarray(x_val, dtype=np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    perm_gen = torch.Generator().manual_seed(child_seed(cfg.seed, "recon_perm"))

    fire_window: list[np.ndarray] = []
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []
    status = "completed"
    dead_count = 0
    epochs_run = 0
    model.train()

    for epoch in range(cfg.epochs):
        epochs_run = epoch + 1
        # permutation stays on CPU so seeding is device-independent
        order = torch.randperm(len(xt), generator=perm_gen)
        epoch_loss = 0.0
        n_batches = 0
        fired = np.zeros(model.dictionary_width, dtype=bool)
        diverged = False
        for start in range(0, len(xt), cfg.batch_size):
            batch = xt[order[start : start + cfg.batch_size].to(device)]
            z, recon = model(batch)
            loss = ((recon - batch) ** 2).mean()
            if not torch.isfinite(loss):
                diverged = True
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.normalize_decoder_()
            with torch.no_grad():
                fired |= (z > 0).any(dim=0).cpu().numpy()
            epoch_loss += float(loss.detach())
            n_batches += 1
        if diverged:
            status = "diverged"
            break

        record = {"epoch": epoch, "train_mse": epoch_loss / max(n_batches, 1)}
        fire_window.append(fired)
        fire_window = fire_window[-cfg.dead_window_epochs :]
        dead = ~np.logical_or.reduce(fire_window)
        dead_count = int(dead.sum())
        record["dead_features"] = dead_count
        if (
            cfg.resample_dead
            and dead_count > 0
            and (epoch + 1) % cfg.resample_every == 0
            and len(fire_window) >= cfg.dead_window_epochs
        ):
            sample = xt[: min(len(xt), 4096)]
            _resample_dead_features(model, sample, dead, cfg.seed)
            record["resampled_features"] = dead_count
            fire_window = []

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            val_mse = _val_mse(model, xv, cfg.batch_size)
            record["val_mse"] = val_mse
            if val_mse < best_val - 1e-12:
                best_val = val_mse
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale > cfg.patience:
                history.append(record)
                break
        history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return ReconstructionTrainResult(
        status=status,
        best_val_mse=best_val,
        history=history,
        dead_feature_count=dead_count,
        epochs_run=epochs_run,
    )
