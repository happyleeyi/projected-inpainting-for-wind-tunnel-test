from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_save import load_processed_data, save_processed_data
from model_DM import DiffusionSchedule, build_model
from params import CFG


class WindowConditionDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        face_ids: np.ndarray,
        left_dists: np.ndarray,
        bottom_dists: np.ndarray,
        indices: np.ndarray,
    ):
        selected = np.asarray(indices, dtype=np.int64)
        self.windows = torch.from_numpy(windows[selected]).float()
        self.face_ids = torch.from_numpy(face_ids[selected]).long()
        self.left_dists = torch.from_numpy(left_dists[selected]).float()
        self.bottom_dists = torch.from_numpy(bottom_dists[selected]).float()

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x0": self.windows[idx],
            "face_id": self.face_ids[idx],
            "left_dist": self.left_dists[idx],
            "bottom_dist": self.bottom_dists[idx],
        }


def _resolve_device() -> torch.device:
    configured = str(CFG.runtime.device).lower()
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CFG.runtime.device requests CUDA but CUDA is not available.")
    return device


def _configure_determinism() -> None:
    seed = int(CFG.data.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if CFG.runtime.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _load_or_build_processed() -> dict[str, Any]:
    try:
        return load_processed_data()
    except FileNotFoundError:
        save_processed_data()
        return load_processed_data()


def _write_metrics(metrics_path: Path, payload: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    sequence_length: int,
    data_dict: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "sequence_length": int(sequence_length),
        "normalization_stats": {
            "mean": float(data_dict["mean"]),
            "std": float(data_dict["std"]),
        },
        "window_len": int(data_dict["window_len"]),
        "sample_frequency_hz": int(data_dict["sample_frequency_hz"]),
    }
    torch.save(checkpoint, path)


def build_dataloader(data_dict: dict[str, Any], split: str = "train") -> DataLoader:
    if split == "train":
        indices = data_dict["train_indices"]
        batch_size = int(CFG.train.batch_size)
        shuffle = True
    elif split == "test":
        indices = data_dict["test_indices"]
        batch_size = int(CFG.test.batch_size)
        shuffle = False
    else:
        raise ValueError(f"Unsupported split: {split}")

    dataset = WindowConditionDataset(
        windows=np.asarray(data_dict["windows"], dtype=np.float32),
        face_ids=np.asarray(data_dict["face_ids"], dtype=np.int64),
        left_dists=np.asarray(data_dict["left_dists"], dtype=np.float32),
        bottom_dists=np.asarray(data_dict["bottom_dists"], dtype=np.float32),
        indices=np.asarray(indices, dtype=np.int64),
    )

    use_pin_memory = bool(CFG.runtime.pin_memory and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(CFG.runtime.num_workers),
        pin_memory=use_pin_memory,
        drop_last=False,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    diffusion: DiffusionSchedule,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    max_steps = CFG.train.max_steps_per_epoch
    use_amp = bool(CFG.runtime.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader, desc=f"Epoch {epoch}", total=len(dataloader)):
        x0 = batch["x0"].to(device, non_blocking=True)
        cond = {
            "face_id": batch["face_id"].to(device, non_blocking=True),
            "left_dist": batch["left_dist"].to(device, non_blocking=True),
            "bottom_dist": batch["bottom_dist"].to(device, non_blocking=True),
        }

        t = torch.randint(
            low=0,
            high=diffusion.diffusion_steps,
            size=(x0.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x0)
        x_t = diffusion.q_sample(x0=x0, t=t, noise=noise)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_noise = model(x_t, t, cond)
            loss = F.mse_loss(pred_noise, noise)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, step={total_steps + 1}.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CFG.train.grad_clip))
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.detach().item())
        total_steps += 1

        if max_steps is not None and total_steps >= int(max_steps):
            break

    mean_loss = total_loss / max(1, total_steps)
    return {
        "epoch": float(epoch),
        "loss": float(mean_loss),
        "steps": float(total_steps),
    }


def train() -> dict[str, Any]:
    CFG.ensure_artifact_dirs()
    _configure_determinism()

    data_dict = _load_or_build_processed()
    train_loader = build_dataloader(data_dict, split="train")

    sequence_length = int(data_dict["window_len"])
    device = _resolve_device()
    model = build_model(sequence_length=sequence_length).to(device)
    diffusion = DiffusionSchedule(device=device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(CFG.train.lr),
        weight_decay=float(CFG.train.weight_decay),
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, int(CFG.train.epochs) + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            diffusion=diffusion,
            device=device,
            epoch=epoch,
        )
        history.append(metrics)
        _save_checkpoint(
            path=CFG.paths.dm_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            sequence_length=sequence_length,
            data_dict=data_dict,
        )
        print(
            f"epoch={epoch}/{CFG.train.epochs} "
            f"train_loss={metrics['loss']:.6f} steps={int(metrics['steps'])}"
        )

    payload = {
        "history": history,
        "num_epochs": int(CFG.train.epochs),
        "max_steps_per_epoch": CFG.train.max_steps_per_epoch,
        "checkpoint_path": str(CFG.paths.dm_checkpoint_path),
        "window_len": int(data_dict["window_len"]),
        "sample_frequency_hz": int(data_dict["sample_frequency_hz"]),
    }
    _write_metrics(CFG.paths.dm_metrics_path, payload)

    return {
        "checkpoint_path": str(CFG.paths.dm_checkpoint_path),
        "metrics_path": str(CFG.paths.dm_metrics_path),
        "epochs_completed": int(CFG.train.epochs),
        "last_epoch_loss": float(history[-1]["loss"]) if history else None,
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
