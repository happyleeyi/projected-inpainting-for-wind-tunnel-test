from __future__ import annotations

import importlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_save import load_processed_data, save_processed_data
from params import CFG


class WindowConditionDataset(Dataset):
    def __init__(
        self,
        windows: np.ndarray,
        face_ids: np.ndarray,
        left_dists: np.ndarray,
        bottom_dists: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        selected = np.asarray(indices, dtype=np.int64)
        self.windows = torch.from_numpy(np.asarray(windows[selected], dtype=np.float32)).float()
        self.face_ids = torch.from_numpy(np.asarray(face_ids[selected], dtype=np.int64)).long()
        self.left_dists = torch.from_numpy(np.asarray(left_dists[selected], dtype=np.float32)).float()
        self.bottom_dists = torch.from_numpy(
            np.asarray(bottom_dists[selected], dtype=np.float32)
        ).float()

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
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

    if bool(CFG.runtime.deterministic):
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


def _load_fm_module() -> Any:
    try:
        return importlib.import_module("model_FM")
    except ImportError as exc:
        raise ImportError(
            "Unable to import model_FM. Ensure model_FM.py exists and is importable."
        ) from exc


def _build_fm_model(sequence_length: int) -> nn.Module:
    mod = _load_fm_module()
    builder_names = ("build_fm_model", "build_model", "build_flow_model")
    for name in builder_names:
        builder = getattr(mod, name, None)
        if callable(builder):
            try:
                return builder(sequence_length=sequence_length)
            except TypeError:
                try:
                    return builder(window_len=sequence_length)
                except TypeError:
                    try:
                        return builder(sequence_length)
                    except TypeError:
                        continue

    for class_name in ("FlowMatchingModel", "ConditionalFlowModel", "FMModel"):
        klass = getattr(mod, class_name, None)
        if isinstance(klass, type) and issubclass(klass, nn.Module):
            try:
                return klass(sequence_length=sequence_length)
            except TypeError:
                try:
                    return klass(window_len=sequence_length)
                except TypeError:
                    continue

    raise AttributeError(
        "model_FM.py does not expose a supported builder/class. "
        "Expected one of build_fm_model/build_model/build_flow_model."
    )


def _is_signature_mismatch_type_error(exc: TypeError) -> bool:
    message = str(exc)
    markers = (
        "missing",
        "required positional argument",
        "unexpected keyword argument",
        "positional arguments but",
        "positional argument but",
        "takes",
        "given",
    )
    return any(marker in message for marker in markers)


def _forward_velocity(model: nn.Module, x_t: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
    attempts: list[tuple[str, Any]] = [
        ("(x_t, t, cond)", lambda: model(x_t, t, cond)),
        ("(x_t=x_t, t=t, cond=cond)", lambda: model(x_t=x_t, t=t, cond=cond)),
        (
            "(x_t=x_t, t=t, face_id=..., left_dist=..., bottom_dist=...)",
            lambda: model(
                x_t=x_t,
                t=t,
                face_id=cond["face_id"],
                left_dist=cond["left_dist"],
                bottom_dist=cond["bottom_dist"],
            ),
        ),
        (
            "(x_t, t, face_id, left_dist, bottom_dist)",
            lambda: model(x_t, t, cond["face_id"], cond["left_dist"], cond["bottom_dist"]),
        ),
        ("(x_t, t)", lambda: model(x_t, t)),
    ]

    last_signature_error: TypeError | None = None
    for signature_name, attempt in attempts:
        try:
            pred = attempt()
            if not isinstance(pred, torch.Tensor):
                raise TypeError("Flow model forward must return a torch.Tensor.")
            return pred
        except TypeError as exc:
            if _is_signature_mismatch_type_error(exc):
                last_signature_error = exc
                continue
            raise TypeError(
                f"Flow model forward raised TypeError while using signature {signature_name}."
            ) from exc

    raise TypeError(
        "Unsupported model_FM forward signature. "
        "Expected one of: (x_t, t, cond), (x_t=x_t, t=t, cond=cond), "
        "(x_t=x_t, t=t, face_id=..., left_dist=..., bottom_dist=...), "
        "(x_t, t, face_id, left_dist, bottom_dist), or (x_t, t)."
    ) from last_signature_error


def _build_train_loader(data_dict: dict[str, Any]) -> DataLoader:
    dataset = WindowConditionDataset(
        windows=np.asarray(data_dict["windows"], dtype=np.float32),
        face_ids=np.asarray(data_dict["face_ids"], dtype=np.int64),
        left_dists=np.asarray(data_dict["left_dists"], dtype=np.float32),
        bottom_dists=np.asarray(data_dict["bottom_dists"], dtype=np.float32),
        indices=np.asarray(data_dict["train_indices"], dtype=np.int64),
    )

    use_pin_memory = bool(CFG.runtime.pin_memory and torch.cuda.is_available())
    return DataLoader(
        dataset,
        batch_size=int(CFG.flow.batch_size),
        shuffle=True,
        num_workers=int(CFG.runtime.num_workers),
        pin_memory=use_pin_memory,
        drop_last=False,
    )


def _resolve_fm_checkpoint_path() -> Path:
    checkpoint_path = getattr(CFG.paths, "fm_checkpoint_path", None)
    if checkpoint_path is None:
        raise AttributeError(
            "CFG.paths.fm_checkpoint_path is required for flow-matching training checkpoints."
        )
    return Path(checkpoint_path)


def _resolve_fm_metrics_path() -> Path:
    configured = getattr(CFG.paths, "fm_metrics_path", None)
    if configured is not None:
        return Path(configured)
    return Path(CFG.paths.metrics_dir) / "fm_train_metrics.json"


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    sequence_length: int,
    data_dict: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
        "objective": "conditional_flow_matching",
    }
    torch.save(payload, path)


def _write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_steps = 0
    max_steps = CFG.flow.max_steps_per_epoch
    use_amp = bool(CFG.runtime.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch in tqdm(dataloader, desc=f"FM Epoch {epoch}", total=len(dataloader)):
        x0 = batch["x0"].to(device, non_blocking=True)
        cond = {
            "face_id": batch["face_id"].to(device, non_blocking=True),
            "left_dist": batch["left_dist"].to(device, non_blocking=True),
            "bottom_dist": batch["bottom_dist"].to(device, non_blocking=True),
        }

        t = torch.rand((x0.shape[0],), device=device, dtype=x0.dtype)
        noise = torch.randn_like(x0)
        t_expanded = t.unsqueeze(-1)
        # Match model_FM sampling convention:
        # t=0 -> data, t=1 -> noise, and reverse-time sampling integrates 1->0.
        x_t = (1.0 - t_expanded) * x0 + t_expanded * noise
        v_target = noise - x0

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            v_pred = _forward_velocity(model=model, x_t=x_t, t=t, cond=cond)
            if v_pred.shape != v_target.shape:
                raise RuntimeError(
                    f"Velocity shape mismatch: pred={tuple(v_pred.shape)} target={tuple(v_target.shape)}"
                )
            loss = F.mse_loss(v_pred, v_target)

        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite FM loss at epoch={epoch}, step={total_steps + 1}.")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CFG.flow.grad_clip))
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
    train_loader = _build_train_loader(data_dict)

    sequence_length = int(data_dict["window_len"])
    device = _resolve_device()
    model = _build_fm_model(sequence_length=sequence_length).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(CFG.flow.lr),
        weight_decay=float(CFG.flow.weight_decay),
    )

    checkpoint_path = _resolve_fm_checkpoint_path()
    metrics_path = _resolve_fm_metrics_path()

    history: list[dict[str, float]] = []
    for epoch in range(1, int(CFG.flow.epochs) + 1):
        metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )
        history.append(metrics)
        _save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            sequence_length=sequence_length,
            data_dict=data_dict,
        )
        print(
            f"epoch={epoch}/{CFG.flow.epochs} "
            f"train_loss={metrics['loss']:.6f} steps={int(metrics['steps'])}"
        )

    payload = {
        "history": history,
        "num_epochs": int(CFG.flow.epochs),
        "max_steps_per_epoch": CFG.flow.max_steps_per_epoch,
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "window_len": int(data_dict["window_len"]),
        "sample_frequency_hz": int(data_dict["sample_frequency_hz"]),
        "objective": "conditional_flow_matching",
    }
    _write_metrics(metrics_path, payload)

    return {
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "epochs_completed": int(CFG.flow.epochs),
        "last_epoch_loss": float(history[-1]["loss"]) if history else None,
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
