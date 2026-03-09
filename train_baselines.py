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

from baseline_models import build_baseline_model
from data_save import load_processed_data, save_processed_data
from params import CFG


class FrontBackDataset(Dataset):
    def __init__(self, windows: np.ndarray, indices: np.ndarray, front_len: int):
        selected = np.asarray(indices, dtype=np.int64)
        chunk = np.asarray(windows[selected], dtype=np.float32)
        self.inputs = torch.from_numpy(chunk[:, :front_len]).float()
        self.targets = torch.from_numpy(chunk[:, front_len:]).float()

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_front": self.inputs[idx],
            "y_back": self.targets[idx],
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


def _build_train_loader(
    windows: np.ndarray,
    train_indices: np.ndarray,
    front_len: int,
) -> DataLoader:
    dataset = FrontBackDataset(windows=windows, indices=train_indices, front_len=front_len)
    use_pin_memory = bool(CFG.runtime.pin_memory and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=max(1, int(CFG.baseline.batch_size)),
        shuffle=True,
        num_workers=int(CFG.runtime.num_workers),
        pin_memory=use_pin_memory,
        drop_last=False,
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_name: str,
    epoch: int,
    window_len: int,
    front_len: int,
    back_len: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": str(model_name),
        "epoch": int(epoch),
        "window_len": int(window_len),
        "front_len": int(front_len),
        "back_len": int(back_len),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_name: str,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    max_steps = CFG.baseline.max_steps_per_epoch
    use_amp = bool(CFG.runtime.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader, desc=f"{model_name.upper()} Epoch {epoch}", total=len(dataloader)):
        x_front = batch["x_front"].to(device, non_blocking=True)
        y_back = batch["y_back"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_back = model(x_front)
            loss = F.mse_loss(pred_back, y_back)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite baseline loss at model={model_name}, epoch={epoch}, step={total_steps + 1}."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CFG.baseline.grad_clip))
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


def _train_single_model(
    model_name: str,
    checkpoint_path: Path,
    train_loader: DataLoader,
    device: torch.device,
    front_len: int,
    back_len: int,
    window_len: int,
) -> dict[str, Any]:
    model = build_baseline_model(model_name=model_name, input_len=front_len, output_len=back_len).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(CFG.baseline.lr),
        weight_decay=float(CFG.baseline.weight_decay),
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, int(CFG.baseline.epochs) + 1):
        metrics = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            model_name=model_name,
        )
        history.append(metrics)
        _save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            model_name=model_name,
            epoch=epoch,
            window_len=window_len,
            front_len=front_len,
            back_len=back_len,
        )
        print(
            f"model={model_name} epoch={epoch}/{CFG.baseline.epochs} "
            f"train_loss={metrics['loss']:.6f} steps={int(metrics['steps'])}"
        )

    return {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "epochs_completed": int(CFG.baseline.epochs),
        "last_epoch_loss": float(history[-1]["loss"]) if history else None,
        "history": history,
    }


def train() -> dict[str, Any]:
    CFG.ensure_artifact_dirs()
    _configure_determinism()

    data_dict = _load_or_build_processed()
    windows = np.asarray(data_dict["windows"], dtype=np.float32)
    train_indices = np.asarray(data_dict["train_indices"], dtype=np.int64)

    window_len = int(windows.shape[1])
    front_len = int(round(window_len * float(CFG.data.front_keep_ratio)))
    front_len = max(1, min(window_len - 1, front_len))
    back_len = int(window_len - front_len)

    train_loader = _build_train_loader(windows=windows, train_indices=train_indices, front_len=front_len)
    device = _resolve_device()

    ann_result = _train_single_model(
        model_name="ann",
        checkpoint_path=CFG.paths.ann_checkpoint_path,
        train_loader=train_loader,
        device=device,
        front_len=front_len,
        back_len=back_len,
        window_len=window_len,
    )
    cnn_result = _train_single_model(
        model_name="cnn",
        checkpoint_path=CFG.paths.cnn_checkpoint_path,
        train_loader=train_loader,
        device=device,
        front_len=front_len,
        back_len=back_len,
        window_len=window_len,
    )

    payload = {
        "config": {
            "device": str(device),
            "window_len": int(window_len),
            "front_len": int(front_len),
            "back_len": int(back_len),
            "batch_size": int(CFG.baseline.batch_size),
            "epochs": int(CFG.baseline.epochs),
            "lr": float(CFG.baseline.lr),
            "weight_decay": float(CFG.baseline.weight_decay),
        },
        "ann": ann_result,
        "cnn": cnn_result,
    }
    CFG.paths.baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    CFG.paths.baseline_metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return {
        "ann_checkpoint_path": str(CFG.paths.ann_checkpoint_path),
        "cnn_checkpoint_path": str(CFG.paths.cnn_checkpoint_path),
        "baseline_metrics_path": str(CFG.paths.baseline_metrics_path),
        "ann_last_epoch_loss": ann_result["last_epoch_loss"],
        "cnn_last_epoch_loss": cnn_result["last_epoch_loss"],
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))

