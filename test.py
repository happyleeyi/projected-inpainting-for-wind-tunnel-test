from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from scipy.linalg import sqrtm

from baseline_models import build_baseline_model
from data_save import load_processed_data
from model_DM import DiffusionSchedule, build_model
from params import CFG

ProjectionMode = Literal["none", "step", "data_roundtrip"]


def _resolve_device() -> torch.device:
    cfg_device = str(CFG.runtime.device).lower()
    if cfg_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CFG.runtime.device requests CUDA but CUDA is not available.")
    return device


def _set_runtime_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if CFG.runtime.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _sanitize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    has_module_prefix = any(key.startswith("module.") for key in state_dict)
    if not has_module_prefix:
        return state_dict
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _safe_torch_load(path: Path, device: torch.device) -> Any:
    # Prefer restricted loading when supported by the installed torch version.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except RuntimeError as exc:
        if "Weights only load failed" in str(exc):
            return torch.load(path, map_location=device)
        raise


def _load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "random_init_used": True,
        "error": None,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if not path.exists():
        return report

    try:
        raw = _safe_torch_load(path=path, device=device)
        if isinstance(raw, dict):
            if "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
                state_dict = raw["model_state_dict"]
            elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
                state_dict = raw["state_dict"]
            elif "model" in raw and isinstance(raw["model"], dict):
                state_dict = raw["model"]
            elif all(isinstance(v, torch.Tensor) for v in raw.values()):
                state_dict = raw
            else:
                raise ValueError("Checkpoint dict format is not recognized.")
        else:
            raise ValueError(f"Unsupported checkpoint object type: {type(raw)}")

        state_dict = _sanitize_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        report["loaded"] = True
        report["random_init_used"] = False
        report["missing_keys"] = list(missing)
        report["unexpected_keys"] = list(unexpected)
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)

    return report


def _load_baseline_checkpoint(
    model_name: str,
    path: Path,
    device: torch.device,
    front_len: int,
    back_len: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_baseline_model(model_name=model_name, input_len=front_len, output_len=back_len).to(device)
    report: dict[str, Any] = {
        "model_name": str(model_name),
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "error": None,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if not path.exists():
        return model, report

    try:
        raw = _safe_torch_load(path=path, device=device)
        if isinstance(raw, dict):
            if "model_state_dict" in raw and isinstance(raw["model_state_dict"], dict):
                state_dict = raw["model_state_dict"]
            elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
                state_dict = raw["state_dict"]
            elif "model" in raw and isinstance(raw["model"], dict):
                state_dict = raw["model"]
            elif all(isinstance(v, torch.Tensor) for v in raw.values()):
                state_dict = raw
            else:
                raise ValueError("Checkpoint dict format is not recognized.")
        else:
            raise ValueError(f"Unsupported checkpoint object type: {type(raw)}")

        state_dict = _sanitize_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        report["loaded"] = True
        report["missing_keys"] = list(missing)
        report["unexpected_keys"] = list(unexpected)
    except Exception as exc:  # noqa: BLE001
        report["error"] = repr(exc)

    model.eval()
    return model, report


def _resolve_flow_checkpoint_path() -> Path:
    configured = getattr(CFG.paths, "fm_checkpoint_path", None)
    if configured:
        return Path(configured)
    legacy = getattr(CFG.paths, "flow_checkpoint_path", None)
    if legacy:
        return Path(legacy)
    return CFG.paths.checkpoints_dir / "flow_model_last.pt"


def _load_flow_runtime(
    device: torch.device,
    window_len: int,
    front_len: int,
) -> tuple[Any | None, Callable[..., Any] | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "enabled": bool(getattr(CFG.test, "compare_flow", True)),
        "module_available": False,
        "model_built": False,
        "path": str(_resolve_flow_checkpoint_path()),
        "exists": _resolve_flow_checkpoint_path().exists(),
        "loaded": False,
        "random_init_used": True,
        "error": None,
        "build_helper": "build_flow_model",
        "load_helper": "_load_checkpoint",
        "inference_helper": "euler_flow_sample",
        "inference_helper_available": True,
    }
    if not report["enabled"]:
        report["error"] = "Flow benchmark disabled by CFG.test.compare_flow."
        return None, None, report

    try:
        from model_FM import build_flow_model, euler_flow_sample

        report["module_available"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"model_FM import failed: {exc!r}"
        return None, None, report

    try:
        flow_model = build_flow_model(sequence_length=int(window_len))
        if flow_model is None:
            raise RuntimeError("Flow builder returned None.")
        report["model_built"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"Flow model build failed: {exc!r}"
        return None, None, report

    if isinstance(flow_model, torch.nn.Module):
        flow_model = flow_model.to(device)
        flow_model.eval()

    ckpt_path = _resolve_flow_checkpoint_path()
    if isinstance(flow_model, torch.nn.Module):
        ckpt_report = _load_checkpoint(model=flow_model, path=ckpt_path, device=device)
        report.update(ckpt_report)

    report["random_init_used"] = not bool(report.get("loaded", False))
    return flow_model, euler_flow_sample, report


@torch.no_grad()
def _predict_baseline_back_half(
    model: torch.nn.Module,
    front_windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    front_windows = np.asarray(front_windows, dtype=np.float32)
    if front_windows.ndim != 2:
        raise ValueError(f"front_windows must be rank-2 [N, L_front], got {front_windows.shape}.")

    n_samples = int(front_windows.shape[0])
    preds: np.ndarray | None = None

    for start in range(0, n_samples, max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), n_samples)
        x_front = torch.from_numpy(front_windows[start:end]).to(device=device, dtype=torch.float32)
        pred_back = model(x_front)
        pred_np = pred_back.detach().cpu().numpy().astype(np.float32, copy=False)
        if preds is None:
            preds = np.zeros((n_samples, pred_np.shape[1]), dtype=np.float32)
        preds[start:end] = pred_np

    if preds is None:
        raise RuntimeError("Baseline prediction produced no outputs.")
    return preds


def _flow_idx_to_tau(total_steps: int, idx: int) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1.")
    if idx < 0:
        return 0.0
    if idx >= total_steps:
        raise ValueError(f"index {idx} out of range for total_steps={total_steps}.")
    if total_steps == 1:
        return 1.0
    return float(idx) / float(total_steps - 1)


def _blend_known_region_flow(
    x_unknown: torch.Tensor,
    known_clean: torch.Tensor,
    known_mask: torch.Tensor,
    tau_next: float,
) -> torch.Tensor:
    tau = float(np.clip(float(tau_next), 0.0, 1.0))
    known_noisy = (1.0 - tau) * known_clean + tau * torch.randn_like(known_clean)
    return known_mask * known_noisy + (1.0 - known_mask) * x_unknown


@torch.no_grad()
def _repaint_flow_inpaint_batch(
    flow_model: Any,
    known_front: torch.Tensor,
    cond: dict[str, torch.Tensor | int],
    front_len: int,
    window_len: int,
    device: torch.device,
    projection_mode: ProjectionMode = "none",
) -> torch.Tensor:
    if projection_mode not in ("none", "step", "data_roundtrip"):
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")

    flow_steps = int(getattr(CFG.test, "flow_sample_steps", CFG.model.diffusion_steps))
    flow_steps = max(1, flow_steps)
    jump_length = max(1, int(CFG.test.repaint_jump_length))
    jump_n_sample = max(1, int(CFG.test.repaint_jump_n_sample))
    projection_epsilon = float(getattr(CFG.test, "projection_epsilon", 1e-6))
    timesteps = _build_repaint_timesteps(
        total_steps=flow_steps,
        jump_length=jump_length,
        jump_n_sample=jump_n_sample,
    )

    batch_size = int(known_front.shape[0])
    target_device = torch.device(device)
    x_t = torch.randn(batch_size, int(window_len), device=target_device, dtype=torch.float32)

    known_front_target = known_front.to(device=target_device, dtype=x_t.dtype)
    if known_front_target.ndim != 2:
        raise ValueError(f"known_front must be rank-2 [B, L_front], got {tuple(known_front_target.shape)}")
    if known_front_target.shape[0] != batch_size:
        raise ValueError(f"known_front batch mismatch: expected {batch_size}, got {known_front_target.shape[0]}")
    if front_len <= 0 or front_len > known_front_target.shape[1]:
        raise ValueError(f"front_len={front_len} is invalid for known_front shape {tuple(known_front_target.shape)}")
    if front_len > window_len:
        raise ValueError(f"front_len={front_len} exceeds window_len={window_len}")

    known_mask = torch.zeros(batch_size, int(window_len), device=target_device, dtype=x_t.dtype)
    known_mask[:, :front_len] = 1.0
    known_clean = torch.zeros(batch_size, int(window_len), device=target_device, dtype=x_t.dtype)
    known_clean[:, :front_len] = known_front_target[:, :front_len]

    def _as_batch_1d_local(value: torch.Tensor | int, dtype: torch.dtype, name: str) -> torch.Tensor:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        tensor = tensor.to(device=target_device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        elif tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D [B] (or scalar), got {tuple(tensor.shape)}")

        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.expand(batch_size)
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
        return tensor

    cond_model = {
        "face_id": _as_batch_1d_local(cond["face_id"], torch.long, "face_id"),
        "left_dist": _as_batch_1d_local(cond["left_dist"], torch.float32, "left_dist"),
        "bottom_dist": _as_batch_1d_local(cond["bottom_dist"], torch.float32, "bottom_dist"),
    }

    model_callable = flow_model
    model_module: torch.nn.Module | None = flow_model if isinstance(flow_model, torch.nn.Module) else None
    was_training = False
    if model_module is not None:
        model_callable = model_module.to(target_device)
        was_training = model_module.training
        model_module.eval()
    elif not callable(model_callable):
        raise TypeError("flow_model must be callable for FM repaint inpainting.")

    try:
        for t_idx, t_next_idx in zip(timesteps[:-1], timesteps[1:]):
            tau_cur = _flow_idx_to_tau(total_steps=flow_steps, idx=int(t_idx))
            tau_next = _flow_idx_to_tau(total_steps=flow_steps, idx=int(t_next_idx))
            t_cur_batch = torch.full((batch_size,), tau_cur, device=target_device, dtype=x_t.dtype)

            v_t = model_callable(x_t, t_cur_batch, cond_model)
            if not isinstance(v_t, torch.Tensor):
                raise TypeError("Flow model output must be a torch.Tensor.")
            if v_t.shape != x_t.shape:
                raise RuntimeError(f"Model output shape {tuple(v_t.shape)} must match state {tuple(x_t.shape)}")

            x_candidate = x_t + (tau_next - tau_cur) * v_t
            if projection_mode == "step":
                x_candidate = _project_with_front_stats(
                    x=x_candidate,
                    known_front=known_front_target,
                    front_len=front_len,
                    epsilon=projection_epsilon,
                )
            elif projection_mode == "data_roundtrip":
                x_data = x_candidate - tau_next * v_t
                x_data_projected = _project_with_front_stats(
                    x=x_data,
                    known_front=known_front_target,
                    front_len=front_len,
                    epsilon=projection_epsilon,
                )
                x_candidate = x_data_projected + tau_next * v_t

            if t_next_idx >= 0:
                x_t = _blend_known_region_flow(
                    x_unknown=x_candidate,
                    known_clean=known_clean,
                    known_mask=known_mask,
                    tau_next=tau_next,
                )
            else:
                x_t = x_candidate
    finally:
        if model_module is not None and was_training:
            model_module.train()

    x_t[:, :front_len] = known_front_target[:, :front_len]
    return x_t


@torch.no_grad()
def _predict_flow_inpaint_batch(
    flow_model: Any,
    infer_func: Callable[..., Any],
    known_front: torch.Tensor,
    cond: dict[str, torch.Tensor | int],
    front_len: int,
    window_len: int,
    device: torch.device,
    projection_mode: ProjectionMode = "none",
) -> np.ndarray:
    if projection_mode not in ("none", "step", "data_roundtrip"):
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")
    _ = infer_func  # Kept for API compatibility; FM inpaint uses repaint loop here.

    out = _repaint_flow_inpaint_batch(
        flow_model=flow_model,
        known_front=known_front,
        cond=cond,
        front_len=front_len,
        window_len=window_len,
        device=device,
        projection_mode=projection_mode,
    )

    if isinstance(out, tuple):
        tensor_like = None
        for item in out:
            if isinstance(item, (torch.Tensor, np.ndarray)):
                tensor_like = item
                break
        if tensor_like is None:
            raise RuntimeError("Flow inference helper returned tuple without tensor-like predictions.")
        out = tensor_like

    if isinstance(out, torch.Tensor):
        pred_np = out.detach().cpu().numpy().astype(np.float32, copy=False)
    elif isinstance(out, np.ndarray):
        pred_np = out.astype(np.float32, copy=False)
    else:
        pred_np = np.asarray(out, dtype=np.float32)

    if pred_np.ndim != 2:
        raise ValueError(f"Flow prediction must be rank-2 [B, L] or [B, back_len], got {pred_np.shape}.")

    batch_size = int(known_front.shape[0])
    back_len = int(window_len - front_len)
    if pred_np.shape[0] != batch_size:
        raise ValueError(
            f"Flow prediction batch size mismatch: expected {batch_size}, got {pred_np.shape[0]}."
        )

    if pred_np.shape[1] == window_len:
        full_window = pred_np.copy()
    elif pred_np.shape[1] == back_len:
        known_front_np = known_front.detach().cpu().numpy().astype(np.float32, copy=False)
        full_window = np.concatenate([known_front_np, pred_np], axis=1).astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"Flow prediction length mismatch: expected {window_len} (full) or {back_len} (back-half), got {pred_np.shape[1]}."
        )

    full_window[:, :front_len] = known_front.detach().cpu().numpy().astype(np.float32, copy=False)
    return full_window.astype(np.float32, copy=False)


def _project_with_front_stats(
    x: torch.Tensor,
    known_front: torch.Tensor,
    front_len: int,
    epsilon: float,
) -> torch.Tensor:
    if front_len <= 0 or front_len >= x.shape[1]:
        return x

    def _project_unknown_to_mean_var(unknown: torch.Tensor, target_mean: float, target_var: float) -> torch.Tensor:
        n = int(unknown.numel())
        if n == 0:
            return unknown
        m = float(target_mean)
        if n == 1:
            return torch.full_like(unknown, m)

        v = max(0.0, float(target_var))
        target_radius = float((v * n) ** 0.5)

        # Closest point with mean/variance constraints:
        # project onto zero-mean sphere by rescaling zero-mean component.
        centered = unknown - unknown.mean()
        centered_norm = float(torch.linalg.vector_norm(centered).item())

        if centered_norm > float(epsilon):
            y = centered * (target_radius / centered_norm)
        else:
            if target_radius <= float(epsilon):
                return torch.full_like(unknown, m)
            direction = torch.zeros_like(unknown)
            direction[0] = 1.0
            direction[1] = -1.0
            direction_norm = float(torch.linalg.vector_norm(direction).item())
            y = direction * (target_radius / max(direction_norm, float(epsilon)))

        projected_unknown = y + m

        # Numerical cleanup for tiny drift.
        projected_unknown = projected_unknown + (m - float(projected_unknown.mean().item()))
        centered_projected = projected_unknown - m
        centered_projected_norm = float(torch.linalg.vector_norm(centered_projected).item())
        if target_radius <= float(epsilon):
            projected_unknown = torch.full_like(projected_unknown, m)
        elif centered_projected_norm > float(epsilon):
            projected_unknown = m + centered_projected * (target_radius / centered_projected_norm)

        return projected_unknown

    projected = x.clone()
    target_mean = known_front.mean(dim=1)
    target_var = known_front.var(dim=1, unbiased=False)

    for b in range(projected.shape[0]):
        unknown = projected[b, front_len:]
        projected[b, front_len:] = _project_unknown_to_mean_var(
            unknown=unknown,
            target_mean=float(target_mean[b].item()),
            target_var=float(target_var[b].item()),
        )

    return projected


def _build_repaint_timesteps(
    total_steps: int,
    jump_length: int,
    jump_n_sample: int,
) -> list[int]:
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1 for RePaint sampling.")

    if total_steps == 1:
        return [0, -1]

    jump_length = max(1, min(int(jump_length), total_steps - 1))
    jump_n_sample = max(1, int(jump_n_sample))
    jumps: dict[int, int] = {}
    if jump_n_sample > 1:
        for t in range(0, total_steps - jump_length, jump_length):
            jumps[t] = jump_n_sample - 1

    t = total_steps - 1
    timesteps = [t]
    while t >= 0:
        t -= 1
        timesteps.append(t)
        if t < 0:
            break
        if jumps.get(t, 0) > 0:
            jumps[t] -= 1
            for _ in range(jump_length):
                if t + 1 >= total_steps:
                    break
                t += 1
                timesteps.append(t)
    return timesteps


def _forward_diffuse_one_step(schedule: DiffusionSchedule, x_t: torch.Tensor, t_next: int) -> torch.Tensor:
    alpha = schedule.alphas[t_next].to(device=x_t.device, dtype=x_t.dtype)
    beta = schedule.betas[t_next].to(device=x_t.device, dtype=x_t.dtype)
    return torch.sqrt(alpha) * x_t + torch.sqrt(beta.clamp(min=1e-20)) * torch.randn_like(x_t)


def _blend_known_region(
    schedule: DiffusionSchedule,
    x_unknown: torch.Tensor,
    known_clean: torch.Tensor,
    known_mask: torch.Tensor,
    t: int,
) -> torch.Tensor:
    t_batch = torch.full((x_unknown.shape[0],), t, device=x_unknown.device, dtype=torch.long)
    known_noisy = schedule.q_sample(x0=known_clean, t=t_batch, noise=torch.randn_like(known_clean))
    return known_mask * known_noisy + (1.0 - known_mask) * x_unknown


@torch.no_grad()
def _project_dm_sample_via_data_roundtrip(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    x_t: torch.Tensor,
    model_cond: dict[str, torch.Tensor],
    known_front: torch.Tensor,
    front_len: int,
    t_ref: int,
    epsilon: float,
) -> torch.Tensor:
    if t_ref < 0:
        return _project_with_front_stats(
            x=x_t,
            known_front=known_front,
            front_len=front_len,
            epsilon=epsilon,
        )

    t_batch = torch.full((x_t.shape[0],), int(t_ref), device=x_t.device, dtype=torch.long)
    _, _, x0_pred = schedule.p_mean_variance(model=model, x_t=x_t, t=t_batch, cond=model_cond)
    x0_projected = _project_with_front_stats(
        x=x0_pred,
        known_front=known_front,
        front_len=front_len,
        epsilon=epsilon,
    )
    if int(t_ref) == 0:
        return x0_projected

    sqrt_ab = schedule.sqrt_alpha_bars[int(t_ref)].to(device=x_t.device, dtype=x_t.dtype)
    sqrt_omb = schedule.sqrt_one_minus_alpha_bars[int(t_ref)].to(device=x_t.device, dtype=x_t.dtype).clamp(min=1e-20)
    noise_hat = (x_t - sqrt_ab * x0_pred) / sqrt_omb
    return sqrt_ab * x0_projected + sqrt_omb * noise_hat


@torch.no_grad()
def _repaint_inpaint_batch(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    known_front: torch.Tensor,
    cond: dict[str, torch.Tensor],
    front_len: int,
    projection_mode: ProjectionMode = "step",
) -> torch.Tensor:
    if projection_mode not in ("none", "step", "data_roundtrip"):
        raise ValueError(f"Unsupported projection_mode: {projection_mode}")

    batch_size = known_front.shape[0]
    total_len = known_front.shape[1] + cond["target_back_len"]
    total_steps = int(schedule.diffusion_steps)
    jump_length = max(1, int(CFG.test.repaint_jump_length))
    jump_n_sample = max(1, int(CFG.test.repaint_jump_n_sample))
    timesteps = _build_repaint_timesteps(
        total_steps=total_steps,
        jump_length=jump_length,
        jump_n_sample=jump_n_sample,
    )

    model_cond = {
        "face_id": cond["face_id"],
        "left_dist": cond["left_dist"],
        "bottom_dist": cond["bottom_dist"],
    }
    known_mask = torch.zeros(batch_size, total_len, device=known_front.device, dtype=known_front.dtype)
    known_mask[:, :front_len] = 1.0
    known_clean = torch.zeros(batch_size, total_len, device=known_front.device, dtype=known_front.dtype)
    known_clean[:, :front_len] = known_front

    # Start from pure noise at the highest diffusion step.
    x_t = torch.randn(batch_size, total_len, device=known_front.device, dtype=known_front.dtype)

    for t, t_next in zip(timesteps[:-1], timesteps[1:]):
        if t_next < t:
            t_batch = torch.full((batch_size,), t, device=known_front.device, dtype=torch.long)
            mean, var, _ = schedule.p_mean_variance(model=model, x_t=x_t, t=t_batch, cond=model_cond)

            if t > 0:
                x_candidate = mean + torch.sqrt(var.clamp(min=1e-20)) * torch.randn_like(x_t)
            else:
                x_candidate = mean

            if projection_mode == "step":
                x_candidate = _project_with_front_stats(
                    x=x_candidate,
                    known_front=known_front,
                    front_len=front_len,
                    epsilon=float(CFG.test.projection_epsilon),
                )
            elif projection_mode == "data_roundtrip":
                if t_next < 0:
                    x_candidate = _project_with_front_stats(
                        x=x_candidate,
                        known_front=known_front,
                        front_len=front_len,
                        epsilon=float(CFG.test.projection_epsilon),
                    )
                else:
                    x_candidate = _project_dm_sample_via_data_roundtrip(
                        model=model,
                        schedule=schedule,
                        x_t=x_candidate,
                        model_cond=model_cond,
                        known_front=known_front,
                        front_len=front_len,
                        t_ref=t_next,
                        epsilon=float(CFG.test.projection_epsilon),
                    )

            if t_next >= 0:
                x_t = _blend_known_region(
                    schedule=schedule,
                    x_unknown=x_candidate,
                    known_clean=known_clean,
                    known_mask=known_mask,
                    t=t_next,
                )
            else:
                x_t = x_candidate
        elif t_next > t:
            x_t = _forward_diffuse_one_step(schedule=schedule, x_t=x_t, t_next=t_next)
        else:
            raise RuntimeError(f"Invalid RePaint timestep transition: {t} -> {t_next}.")

    x_t[:, :front_len] = known_front
    return x_t


def _handcrafted_feature_extractor(back_half: np.ndarray) -> np.ndarray:
    x = back_half.astype(np.float64, copy=False)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [N, L], got shape {x.shape}.")
    if x.shape[1] < 2:
        raise ValueError("Back-half length must be at least 2 for feature extraction.")

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    minimum = x.min(axis=1, keepdims=True)
    maximum = x.max(axis=1, keepdims=True)
    median = np.median(x, axis=1, keepdims=True)
    q25 = np.quantile(x, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(x, 0.75, axis=1, keepdims=True)
    energy = np.mean(x * x, axis=1, keepdims=True)
    abs_mean = np.mean(np.abs(x), axis=1, keepdims=True)

    dx = np.diff(x, axis=1)
    dx_abs_mean = np.mean(np.abs(dx), axis=1, keepdims=True)
    dx_std = dx.std(axis=1, keepdims=True)

    fft_mag = np.abs(np.fft.rfft(x, axis=1))
    fft_bins = min(8, fft_mag.shape[1])
    fft_slice = fft_mag[:, :fft_bins]
    fft_norm = fft_slice / (np.linalg.norm(fft_slice, axis=1, keepdims=True) + 1e-12)

    return np.concatenate(
        [
            mean,
            std,
            minimum,
            maximum,
            median,
            q25,
            q75,
            energy,
            abs_mean,
            dx_abs_mean,
            dx_std,
            fft_norm,
        ],
        axis=1,
    )


def _covariance(feats: np.ndarray) -> np.ndarray:
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D features array [N, D], got {feats.shape}.")
    d = feats.shape[1]
    if feats.shape[0] <= 1:
        return np.eye(d, dtype=np.float64) * 1e-6
    cov = np.cov(feats, rowvar=False).astype(np.float64)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    return cov


def _fid_from_features(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    real_feats = real_feats.astype(np.float64, copy=False)
    fake_feats = fake_feats.astype(np.float64, copy=False)

    mu_real = real_feats.mean(axis=0)
    mu_fake = fake_feats.mean(axis=0)
    cov_real = _covariance(real_feats)
    cov_fake = _covariance(fake_feats)

    eps_values = (0.0, 1e-8, 1e-6, 1e-4)
    covmean = None
    for eps in eps_values:
        jitter = np.eye(cov_real.shape[0], dtype=np.float64) * eps
        candidate = sqrtm((cov_real + jitter) @ (cov_fake + jitter))
        if np.all(np.isfinite(candidate)):
            covmean = candidate
            break

    if covmean is None:
        covmean = sqrtm((cov_real + np.eye(cov_real.shape[0]) * 1e-3) @ (cov_fake + np.eye(cov_fake.shape[0]) * 1e-3))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu_real - mu_fake
    fid = float(diff @ diff + np.trace(cov_real + cov_fake - 2.0 * covmean))
    return max(0.0, fid)


def _fid_proxy_1d(real_back: np.ndarray, fake_back: np.ndarray) -> float:
    real_feats = _handcrafted_feature_extractor(real_back)
    fake_feats = _handcrafted_feature_extractor(fake_back)
    return _fid_from_features(real_feats, fake_feats)


def _mean_var_diff_vs_known(
    generated_back: np.ndarray,
    known_front: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    generated_back = np.asarray(generated_back, dtype=np.float64)
    known_front = np.asarray(known_front, dtype=np.float64)
    if generated_back.ndim != 2 or known_front.ndim != 2:
        raise ValueError(
            f"Expected rank-2 arrays for generated_back/known_front, got {generated_back.shape} and {known_front.shape}."
        )
    if generated_back.shape[0] != known_front.shape[0]:
        raise ValueError(
            f"Batch size mismatch between generated_back and known_front: {generated_back.shape[0]} vs {known_front.shape[0]}."
        )

    gen_mean = generated_back.mean(axis=1, dtype=np.float64)
    gen_var = generated_back.var(axis=1, dtype=np.float64)
    known_mean = known_front.mean(axis=1, dtype=np.float64)
    known_var = known_front.var(axis=1, dtype=np.float64)
    return gen_mean - known_mean, gen_var - known_var


def _plot_single_case(
    sample_index: int,
    original_window: np.ndarray,
    projected_reconstructed_window: np.ndarray,
    plain_reconstructed_window: np.ndarray,
    flow_projected_reconstructed_window: np.ndarray | None,
    flow_plain_reconstructed_window: np.ndarray | None,
    ann_reconstructed_window: np.ndarray | None,
    cnn_reconstructed_window: np.ndarray | None,
    front_len: int,
    show_flow_curve: bool = True,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"Plot skipped: matplotlib import failed: {exc}")
        return

    x_axis = np.arange(original_window.shape[0], dtype=np.int64)
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

    axes[0].plot(x_axis, original_window, label="Actual", linewidth=2.0)
    axes[0].plot(
        x_axis,
        projected_reconstructed_window,
        label="Repainted (Projected)",
        linewidth=1.8,
        alpha=0.9,
    )
    axes[0].axvline(
        front_len - 1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Known/Predicted Boundary",
    )
    axes[0].set_title(f"Projected Repaint vs Actual (sample_index={sample_index})")
    axes[0].set_ylabel("Signal Value")
    axes[0].legend()

    axes[1].plot(x_axis, original_window, label="Actual", linewidth=2.0)
    axes[1].plot(
        x_axis,
        plain_reconstructed_window,
        label="Repainted (Plain)",
        linewidth=1.8,
        alpha=0.9,
    )
    axes[1].axvline(
        front_len - 1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Known/Predicted Boundary",
    )
    axes[1].set_title("Plain Repaint vs Actual")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Signal Value")
    axes[1].legend()

    axes[2].plot(x_axis, original_window, label="Actual", linewidth=2.0)
    if show_flow_curve and flow_projected_reconstructed_window is not None:
        axes[2].plot(
            x_axis,
            flow_projected_reconstructed_window,
            label="Flow Repaint (Projected)",
            linewidth=1.8,
            alpha=0.9,
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "Flow projected result not available",
            transform=axes[2].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    axes[2].axvline(
        front_len - 1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Known/Predicted Boundary",
    )
    axes[2].set_title("Flow Repaint (Projected) vs Actual")
    axes[2].set_xlabel("Time Step")
    axes[2].set_ylabel("Signal Value")
    axes[2].legend()

    axes[3].plot(x_axis, original_window, label="Actual", linewidth=2.0)
    if show_flow_curve and flow_plain_reconstructed_window is not None:
        axes[3].plot(
            x_axis,
            flow_plain_reconstructed_window,
            label="Flow Repaint (Plain)",
            linewidth=1.8,
            alpha=0.9,
        )
    else:
        axes[3].text(
            0.5,
            0.5,
            "Flow plain result not available",
            transform=axes[3].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    axes[3].axvline(
        front_len - 1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Known/Predicted Boundary",
    )
    axes[3].set_title("Flow Repaint (Plain) vs Actual")
    axes[3].set_xlabel("Time Step")
    axes[3].set_ylabel("Signal Value")
    axes[3].legend()

    axes[4].plot(x_axis, original_window, label="Actual", linewidth=2.0)
    if ann_reconstructed_window is not None:
        axes[4].plot(
            x_axis,
            ann_reconstructed_window,
            label="ANN Baseline",
            linewidth=1.8,
            alpha=0.9,
        )
    if cnn_reconstructed_window is not None:
        axes[4].plot(
            x_axis,
            cnn_reconstructed_window,
            label="CNN Baseline",
            linewidth=1.8,
            alpha=0.9,
        )
    axes[4].axvline(
        front_len - 1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        label="Known/Predicted Boundary",
    )
    if ann_reconstructed_window is None and cnn_reconstructed_window is None:
        axes[4].text(
            0.5,
            0.5,
            "ANN/CNN checkpoints not available",
            transform=axes[4].transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
    axes[4].set_title("ANN/CNN Baselines vs Actual")
    axes[4].set_xlabel("Time Step")
    axes[4].set_ylabel("Signal Value")
    axes[4].legend()

    fig.suptitle("Single-case Reconstruction Comparison", fontsize=12)
    plt.tight_layout()

    try:
        plt.show()
    except Exception as exc:  # noqa: BLE001
        fallback_path = CFG.paths.recon_outputs_dir / f"single_case_plot_compare_{sample_index}.png"
        plt.savefig(fallback_path, dpi=150)
        print(f"Plot show failed ({exc}); saved fallback image to: {fallback_path}")
    finally:
        plt.close()


def evaluate() -> dict[str, Any]:
    CFG.ensure_artifact_dirs()
    _set_runtime_seed(int(CFG.data.seed))

    device = _resolve_device()
    processed = load_processed_data()

    windows = np.asarray(processed["windows"], dtype=np.float32)
    test_indices = np.asarray(processed["test_indices"], dtype=np.int64)
    if CFG.test.max_eval_samples is not None:
        test_indices = test_indices[: int(CFG.test.max_eval_samples)]

    if test_indices.size == 0:
        raise ValueError("No test indices available for evaluation.")

    test_windows = windows[test_indices]
    window_len = int(test_windows.shape[1])
    front_ratio = float(CFG.data.front_keep_ratio)
    front_len = int(round(window_len * front_ratio))
    front_len = max(1, min(window_len - 1, front_len))

    model = build_model(sequence_length=window_len).to(device)
    ckpt_report = _load_checkpoint(model=model, path=CFG.paths.dm_checkpoint_path, device=device)
    model.eval()
    schedule = DiffusionSchedule(device=device)
    flow_model, flow_infer_func, flow_report = _load_flow_runtime(
        device=device,
        window_len=window_len,
        front_len=front_len,
    )
    if bool(getattr(CFG.test, "compare_flow", True)) and bool(getattr(CFG.test, "require_flow_checkpoint", False)):
        if not bool(flow_report.get("loaded", False)):
            detail = flow_report.get("error", "checkpoint missing or failed to load")
            raise FileNotFoundError(
                f"Flow checkpoint is required but not usable: {flow_report.get('path')} ({detail}). "
                "Run train_FM.py first."
            )
    flow_enabled = bool(flow_report.get("loaded", False)) and flow_model is not None and flow_infer_func is not None

    face_ids_all = np.asarray(processed["face_ids"], dtype=np.int64)
    left_dists_all = np.asarray(processed["left_dists"], dtype=np.float32)
    bottom_dists_all = np.asarray(processed["bottom_dists"], dtype=np.float32)

    batch_size = max(1, int(CFG.test.batch_size))
    recon_windows = np.zeros_like(test_windows, dtype=np.float32)
    recon_windows_plain = np.zeros_like(test_windows, dtype=np.float32)
    recon_windows_projected_2 = np.zeros_like(test_windows, dtype=np.float32)
    recon_windows_flow_projected = np.zeros_like(test_windows, dtype=np.float32) if flow_enabled else None
    recon_windows_flow_plain = np.zeros_like(test_windows, dtype=np.float32) if flow_enabled else None
    recon_windows_flow_projected_2 = np.zeros_like(test_windows, dtype=np.float32) if flow_enabled else None

    for start in range(0, test_indices.size, batch_size):
        end = min(start + batch_size, test_indices.size)
        batch_idx = test_indices[start:end]
        batch_windows = torch.from_numpy(test_windows[start:end]).to(device=device, dtype=torch.float32)

        known_front = batch_windows[:, :front_len].clone()
        cond = {
            "face_id": torch.from_numpy(face_ids_all[batch_idx]).to(device=device, dtype=torch.long),
            "left_dist": torch.from_numpy(left_dists_all[batch_idx]).to(device=device, dtype=torch.float32),
            "bottom_dist": torch.from_numpy(bottom_dists_all[batch_idx]).to(device=device, dtype=torch.float32),
            "target_back_len": window_len - front_len,
        }

        recon_batch = _repaint_inpaint_batch(
            model=model,
            schedule=schedule,
            known_front=known_front,
            cond=cond,
            front_len=front_len,
            projection_mode="step",
        )
        recon_batch_plain = _repaint_inpaint_batch(
            model=model,
            schedule=schedule,
            known_front=known_front,
            cond=cond,
            front_len=front_len,
            projection_mode="none",
        )
        recon_batch_projected_2 = _repaint_inpaint_batch(
            model=model,
            schedule=schedule,
            known_front=known_front,
            cond=cond,
            front_len=front_len,
            projection_mode="data_roundtrip",
        )
        recon_windows[start:end] = recon_batch.detach().cpu().numpy().astype(np.float32)
        recon_windows_plain[start:end] = recon_batch_plain.detach().cpu().numpy().astype(np.float32)
        recon_windows_projected_2[start:end] = recon_batch_projected_2.detach().cpu().numpy().astype(np.float32)
        if (
            flow_enabled
            and recon_windows_flow_projected is not None
            and recon_windows_flow_plain is not None
            and recon_windows_flow_projected_2 is not None
            and flow_infer_func is not None
            and flow_model is not None
        ):
            try:
                recon_batch_flow_projected = _predict_flow_inpaint_batch(
                    flow_model=flow_model,
                    infer_func=flow_infer_func,
                    known_front=known_front,
                    cond=cond,
                    front_len=front_len,
                    window_len=window_len,
                    device=device,
                    projection_mode="step",
                )
                recon_batch_flow_plain = _predict_flow_inpaint_batch(
                    flow_model=flow_model,
                    infer_func=flow_infer_func,
                    known_front=known_front,
                    cond=cond,
                    front_len=front_len,
                    window_len=window_len,
                    device=device,
                    projection_mode="none",
                )
                recon_batch_flow_projected_2 = _predict_flow_inpaint_batch(
                    flow_model=flow_model,
                    infer_func=flow_infer_func,
                    known_front=known_front,
                    cond=cond,
                    front_len=front_len,
                    window_len=window_len,
                    device=device,
                    projection_mode="data_roundtrip",
                )
                recon_windows_flow_projected[start:end] = recon_batch_flow_projected.astype(np.float32, copy=False)
                recon_windows_flow_plain[start:end] = recon_batch_flow_plain.astype(np.float32, copy=False)
                recon_windows_flow_projected_2[start:end] = recon_batch_flow_projected_2.astype(np.float32, copy=False)
            except Exception as exc:  # noqa: BLE001
                flow_enabled = False
                recon_windows_flow_projected = None
                recon_windows_flow_plain = None
                recon_windows_flow_projected_2 = None
                flow_report["loaded"] = False
                flow_report["error"] = f"Flow inference failed: {exc!r}"

    recon_windows[:, :front_len] = test_windows[:, :front_len]
    recon_windows_plain[:, :front_len] = test_windows[:, :front_len]
    recon_windows_projected_2[:, :front_len] = test_windows[:, :front_len]
    if (
        flow_enabled
        and recon_windows_flow_projected is not None
        and recon_windows_flow_plain is not None
        and recon_windows_flow_projected_2 is not None
    ):
        recon_windows_flow_projected[:, :front_len] = test_windows[:, :front_len]
        recon_windows_flow_plain[:, :front_len] = test_windows[:, :front_len]
        recon_windows_flow_projected_2[:, :front_len] = test_windows[:, :front_len]

    true_back = test_windows[:, front_len:]
    pred_back = recon_windows[:, front_len:]
    pred_back_plain = recon_windows_plain[:, front_len:]
    pred_back_projected_2 = recon_windows_projected_2[:, front_len:]
    known_front_eval = test_windows[:, :front_len]
    pred_back_flow_projected = (
        recon_windows_flow_projected[:, front_len:]
        if (flow_enabled and recon_windows_flow_projected is not None)
        else None
    )
    pred_back_flow_plain = (
        recon_windows_flow_plain[:, front_len:]
        if (flow_enabled and recon_windows_flow_plain is not None)
        else None
    )
    pred_back_flow_projected_2 = (
        recon_windows_flow_projected_2[:, front_len:]
        if (flow_enabled and recon_windows_flow_projected_2 is not None)
        else None
    )

    mse_per_sample = np.mean((pred_back - true_back) ** 2, axis=1, dtype=np.float64)
    mse_mean = float(np.mean(mse_per_sample, dtype=np.float64))
    fid_proxy = float(_fid_proxy_1d(true_back, pred_back))
    mean_diff_per_sample, var_diff_per_sample = _mean_var_diff_vs_known(
        generated_back=pred_back,
        known_front=known_front_eval,
    )

    mse_per_sample_plain = np.mean((pred_back_plain - true_back) ** 2, axis=1, dtype=np.float64)
    mse_mean_plain = float(np.mean(mse_per_sample_plain, dtype=np.float64))
    fid_proxy_plain = float(_fid_proxy_1d(true_back, pred_back_plain))
    mean_diff_per_sample_plain, var_diff_per_sample_plain = _mean_var_diff_vs_known(
        generated_back=pred_back_plain,
        known_front=known_front_eval,
    )

    mse_per_sample_projected_2 = np.mean((pred_back_projected_2 - true_back) ** 2, axis=1, dtype=np.float64)
    mse_mean_projected_2 = float(np.mean(mse_per_sample_projected_2, dtype=np.float64))
    fid_proxy_projected_2 = float(_fid_proxy_1d(true_back, pred_back_projected_2))
    mean_diff_per_sample_projected_2, var_diff_per_sample_projected_2 = _mean_var_diff_vs_known(
        generated_back=pred_back_projected_2,
        known_front=known_front_eval,
    )

    mse_per_sample_flow_projected = (
        np.mean((pred_back_flow_projected - true_back) ** 2, axis=1, dtype=np.float64)
        if pred_back_flow_projected is not None
        else None
    )
    mse_mean_flow_projected = (
        float(np.mean(mse_per_sample_flow_projected, dtype=np.float64))
        if mse_per_sample_flow_projected is not None
        else None
    )
    fid_proxy_flow_projected = (
        float(_fid_proxy_1d(true_back, pred_back_flow_projected)) if pred_back_flow_projected is not None else None
    )
    if pred_back_flow_projected is not None:
        mean_diff_per_sample_flow_projected, var_diff_per_sample_flow_projected = _mean_var_diff_vs_known(
            generated_back=pred_back_flow_projected,
            known_front=known_front_eval,
        )
    else:
        mean_diff_per_sample_flow_projected, var_diff_per_sample_flow_projected = None, None

    mse_per_sample_flow_plain = (
        np.mean((pred_back_flow_plain - true_back) ** 2, axis=1, dtype=np.float64)
        if pred_back_flow_plain is not None
        else None
    )
    mse_mean_flow_plain = (
        float(np.mean(mse_per_sample_flow_plain, dtype=np.float64)) if mse_per_sample_flow_plain is not None else None
    )
    fid_proxy_flow_plain = float(_fid_proxy_1d(true_back, pred_back_flow_plain)) if pred_back_flow_plain is not None else None
    if pred_back_flow_plain is not None:
        mean_diff_per_sample_flow_plain, var_diff_per_sample_flow_plain = _mean_var_diff_vs_known(
            generated_back=pred_back_flow_plain,
            known_front=known_front_eval,
        )
    else:
        mean_diff_per_sample_flow_plain, var_diff_per_sample_flow_plain = None, None
    mse_per_sample_flow_projected_2 = (
        np.mean((pred_back_flow_projected_2 - true_back) ** 2, axis=1, dtype=np.float64)
        if pred_back_flow_projected_2 is not None
        else None
    )
    mse_mean_flow_projected_2 = (
        float(np.mean(mse_per_sample_flow_projected_2, dtype=np.float64))
        if mse_per_sample_flow_projected_2 is not None
        else None
    )
    fid_proxy_flow_projected_2 = (
        float(_fid_proxy_1d(true_back, pred_back_flow_projected_2)) if pred_back_flow_projected_2 is not None else None
    )
    if pred_back_flow_projected_2 is not None:
        mean_diff_per_sample_flow_projected_2, var_diff_per_sample_flow_projected_2 = _mean_var_diff_vs_known(
            generated_back=pred_back_flow_projected_2,
            known_front=known_front_eval,
        )
    else:
        mean_diff_per_sample_flow_projected_2, var_diff_per_sample_flow_projected_2 = None, None

    # Backward-compatible alias: "flow" => projected flow repaint.
    mse_per_sample_flow = mse_per_sample_flow_projected
    mse_mean_flow = mse_mean_flow_projected
    fid_proxy_flow = fid_proxy_flow_projected
    mean_diff_per_sample_flow = mean_diff_per_sample_flow_projected
    var_diff_per_sample_flow = var_diff_per_sample_flow_projected

    baseline_reports: dict[str, Any] = {}
    baseline_predictions: dict[str, np.ndarray] = {}
    baseline_mse_per_sample: dict[str, np.ndarray] = {}
    baseline_mse_mean: dict[str, float] = {}
    baseline_fid: dict[str, float] = {}
    baseline_mean_diff_per_sample: dict[str, np.ndarray] = {}
    baseline_var_diff_per_sample: dict[str, np.ndarray] = {}

    if bool(CFG.test.compare_baselines):
        baseline_pairs = (
            ("ann", CFG.paths.ann_checkpoint_path),
            ("cnn", CFG.paths.cnn_checkpoint_path),
        )
        front_windows = test_windows[:, :front_len]
        for model_name, checkpoint_path in baseline_pairs:
            baseline_model, baseline_report = _load_baseline_checkpoint(
                model_name=model_name,
                path=checkpoint_path,
                device=device,
                front_len=front_len,
                back_len=window_len - front_len,
            )
            baseline_reports[model_name] = baseline_report

            if not baseline_report["loaded"]:
                if bool(CFG.test.require_baseline_checkpoints):
                    detail = baseline_report["error"] if baseline_report["error"] is not None else "checkpoint missing"
                    raise FileNotFoundError(
                        f"Baseline checkpoint for {model_name} is required but not usable: "
                        f"{checkpoint_path} ({detail}). Run train_baselines.py first."
                    )
                continue

            pred_back_baseline = _predict_baseline_back_half(
                model=baseline_model,
                front_windows=front_windows,
                batch_size=batch_size,
                device=device,
            )
            baseline_predictions[model_name] = pred_back_baseline
            baseline_mse_per_sample[model_name] = np.mean(
                (pred_back_baseline - true_back) ** 2,
                axis=1,
                dtype=np.float64,
            )
            baseline_mse_mean[model_name] = float(np.mean(baseline_mse_per_sample[model_name], dtype=np.float64))
            baseline_fid[model_name] = float(_fid_proxy_1d(true_back, pred_back_baseline))
            baseline_mean_diff_per_sample[model_name], baseline_var_diff_per_sample[model_name] = _mean_var_diff_vs_known(
                generated_back=pred_back_baseline,
                known_front=known_front_eval,
            )

    ann_reconstructed_window: np.ndarray | None = None
    cnn_reconstructed_window: np.ndarray | None = None
    flow_projected_reconstructed_window: np.ndarray | None = None
    flow_plain_reconstructed_window: np.ndarray | None = None
    if flow_enabled and recon_windows_flow_projected is not None and recon_windows_flow_plain is not None:
        flow_projected_reconstructed_window = recon_windows_flow_projected[0]
        flow_plain_reconstructed_window = recon_windows_flow_plain[0]
    if "ann" in baseline_predictions:
        ann_reconstructed_window = np.concatenate(
            [test_windows[0, :front_len], baseline_predictions["ann"][0]],
            axis=0,
        ).astype(np.float32, copy=False)
    if "cnn" in baseline_predictions:
        cnn_reconstructed_window = np.concatenate(
            [test_windows[0, :front_len], baseline_predictions["cnn"][0]],
            axis=0,
        ).astype(np.float32, copy=False)

    _plot_single_case(
        sample_index=int(test_indices[0]),
        original_window=test_windows[0],
        projected_reconstructed_window=recon_windows[0],
        plain_reconstructed_window=recon_windows_plain[0],
        flow_projected_reconstructed_window=flow_projected_reconstructed_window,
        flow_plain_reconstructed_window=flow_plain_reconstructed_window,
        ann_reconstructed_window=ann_reconstructed_window,
        cnn_reconstructed_window=cnn_reconstructed_window,
        front_len=front_len,
        show_flow_curve=bool(getattr(CFG.test, "plot_flow_curve", True)),
    )

    masked_windows = test_windows.copy()
    masked_windows[:, front_len:] = 0.0

    per_sample_metrics: list[dict[str, Any]] = []
    for idx, sample_index in enumerate(test_indices.tolist()):
        row: dict[str, Any] = {
            "sample_index": int(sample_index),
            "mse_back_half_projected": float(mse_per_sample[idx]),
            "mse_back_half_plain": float(mse_per_sample_plain[idx]),
            "mse_back_half_dm_projected_2": float(mse_per_sample_projected_2[idx]),
            "mean_diff_known_projected": float(mean_diff_per_sample[idx]),
            "var_diff_known_projected": float(var_diff_per_sample[idx]),
            "mean_diff_known_plain": float(mean_diff_per_sample_plain[idx]),
            "var_diff_known_plain": float(var_diff_per_sample_plain[idx]),
            "mean_diff_known_dm_projected_2": float(mean_diff_per_sample_projected_2[idx]),
            "var_diff_known_dm_projected_2": float(var_diff_per_sample_projected_2[idx]),
            # Human-readable aliases for dm_projected_2.
            "mse_back_half_projected_2": float(mse_per_sample_projected_2[idx]),
            "mean_diff_known_projected_2": float(mean_diff_per_sample_projected_2[idx]),
            "var_diff_known_projected_2": float(var_diff_per_sample_projected_2[idx]),
            # Backward-compatible alias for projected repaint.
            "mse_back_half": float(mse_per_sample[idx]),
        }
        if "ann" in baseline_mse_per_sample:
            row["mse_back_half_ann"] = float(baseline_mse_per_sample["ann"][idx])
            row["mean_diff_known_ann"] = float(baseline_mean_diff_per_sample["ann"][idx])
            row["var_diff_known_ann"] = float(baseline_var_diff_per_sample["ann"][idx])
        if "cnn" in baseline_mse_per_sample:
            row["mse_back_half_cnn"] = float(baseline_mse_per_sample["cnn"][idx])
            row["mean_diff_known_cnn"] = float(baseline_mean_diff_per_sample["cnn"][idx])
            row["var_diff_known_cnn"] = float(baseline_var_diff_per_sample["cnn"][idx])
        if mse_per_sample_flow_projected is not None:
            row["mse_back_half_flow_projected"] = float(mse_per_sample_flow_projected[idx])
            row["mean_diff_known_flow_projected"] = float(mean_diff_per_sample_flow_projected[idx])
            row["var_diff_known_flow_projected"] = float(var_diff_per_sample_flow_projected[idx])
            # Backward-compatible alias.
            row["mse_back_half_flow"] = float(mse_per_sample_flow_projected[idx])
            row["mean_diff_known_flow"] = float(mean_diff_per_sample_flow_projected[idx])
            row["var_diff_known_flow"] = float(var_diff_per_sample_flow_projected[idx])
        if mse_per_sample_flow_plain is not None:
            row["mse_back_half_flow_plain"] = float(mse_per_sample_flow_plain[idx])
            row["mean_diff_known_flow_plain"] = float(mean_diff_per_sample_flow_plain[idx])
            row["var_diff_known_flow_plain"] = float(var_diff_per_sample_flow_plain[idx])
        if mse_per_sample_flow_projected_2 is not None:
            row["mse_back_half_flow_projected_2"] = float(mse_per_sample_flow_projected_2[idx])
            row["mean_diff_known_flow_projected_2"] = float(mean_diff_per_sample_flow_projected_2[idx])
            row["var_diff_known_flow_projected_2"] = float(var_diff_per_sample_flow_projected_2[idx])
        per_sample_metrics.append(row)

    metrics_payload: dict[str, Any] = {
        "checkpoint": ckpt_report,
        "flow_checkpoint": flow_report,
        "baseline_checkpoints": baseline_reports,
        "config": {
            "device": str(device),
            "num_test_samples": int(test_indices.size),
            "window_len": int(window_len),
            "front_keep_ratio": front_ratio,
            "front_len": int(front_len),
            "back_len": int(window_len - front_len),
            "diffusion_steps": int(schedule.diffusion_steps),
            "repaint_jump_length": int(CFG.test.repaint_jump_length),
            "repaint_jump_n_sample": int(CFG.test.repaint_jump_n_sample),
        },
        "aggregate_metrics": {
            "mse_back_half_projected": mse_mean,
            "fid_back_half_proxy_projected": fid_proxy,
            "mse_back_half_std_projected": float(np.std(mse_per_sample, dtype=np.float64)),
            "mean_diff_known_projected": float(np.mean(mean_diff_per_sample, dtype=np.float64)),
            "var_diff_known_projected": float(np.mean(var_diff_per_sample, dtype=np.float64)),
            "mean_abs_diff_known_projected": float(np.mean(np.abs(mean_diff_per_sample), dtype=np.float64)),
            "var_abs_diff_known_projected": float(np.mean(np.abs(var_diff_per_sample), dtype=np.float64)),
            "mse_back_half_plain": mse_mean_plain,
            "fid_back_half_proxy_plain": fid_proxy_plain,
            "mse_back_half_std_plain": float(np.std(mse_per_sample_plain, dtype=np.float64)),
            "mean_diff_known_plain": float(np.mean(mean_diff_per_sample_plain, dtype=np.float64)),
            "var_diff_known_plain": float(np.mean(var_diff_per_sample_plain, dtype=np.float64)),
            "mean_abs_diff_known_plain": float(np.mean(np.abs(mean_diff_per_sample_plain), dtype=np.float64)),
            "var_abs_diff_known_plain": float(np.mean(np.abs(var_diff_per_sample_plain), dtype=np.float64)),
            "mse_back_half_dm_projected_2": mse_mean_projected_2,
            "fid_back_half_proxy_dm_projected_2": fid_proxy_projected_2,
            "mse_back_half_std_dm_projected_2": float(np.std(mse_per_sample_projected_2, dtype=np.float64)),
            "mean_diff_known_dm_projected_2": float(np.mean(mean_diff_per_sample_projected_2, dtype=np.float64)),
            "var_diff_known_dm_projected_2": float(np.mean(var_diff_per_sample_projected_2, dtype=np.float64)),
            "mean_abs_diff_known_dm_projected_2": float(np.mean(np.abs(mean_diff_per_sample_projected_2), dtype=np.float64)),
            "var_abs_diff_known_dm_projected_2": float(np.mean(np.abs(var_diff_per_sample_projected_2), dtype=np.float64)),
            # Human-readable aliases for dm_projected_2.
            "mse_back_half_projected_2": mse_mean_projected_2,
            "fid_back_half_proxy_projected_2": fid_proxy_projected_2,
            "mse_back_half_flow_projected": mse_mean_flow_projected,
            "fid_back_half_proxy_flow_projected": fid_proxy_flow_projected,
            "mse_back_half_std_flow_projected": float(np.std(mse_per_sample_flow_projected, dtype=np.float64))
            if mse_per_sample_flow_projected is not None
            else None,
            "mean_diff_known_flow_projected": float(np.mean(mean_diff_per_sample_flow_projected, dtype=np.float64))
            if mean_diff_per_sample_flow_projected is not None
            else None,
            "var_diff_known_flow_projected": float(np.mean(var_diff_per_sample_flow_projected, dtype=np.float64))
            if var_diff_per_sample_flow_projected is not None
            else None,
            "mean_abs_diff_known_flow_projected": float(
                np.mean(np.abs(mean_diff_per_sample_flow_projected), dtype=np.float64)
            )
            if mean_diff_per_sample_flow_projected is not None
            else None,
            "var_abs_diff_known_flow_projected": float(
                np.mean(np.abs(var_diff_per_sample_flow_projected), dtype=np.float64)
            )
            if var_diff_per_sample_flow_projected is not None
            else None,
            "mse_back_half_flow_plain": mse_mean_flow_plain,
            "fid_back_half_proxy_flow_plain": fid_proxy_flow_plain,
            "mse_back_half_std_flow_plain": float(np.std(mse_per_sample_flow_plain, dtype=np.float64))
            if mse_per_sample_flow_plain is not None
            else None,
            "mean_diff_known_flow_plain": float(np.mean(mean_diff_per_sample_flow_plain, dtype=np.float64))
            if mean_diff_per_sample_flow_plain is not None
            else None,
            "var_diff_known_flow_plain": float(np.mean(var_diff_per_sample_flow_plain, dtype=np.float64))
            if var_diff_per_sample_flow_plain is not None
            else None,
            "mean_abs_diff_known_flow_plain": float(np.mean(np.abs(mean_diff_per_sample_flow_plain), dtype=np.float64))
            if mean_diff_per_sample_flow_plain is not None
            else None,
            "var_abs_diff_known_flow_plain": float(np.mean(np.abs(var_diff_per_sample_flow_plain), dtype=np.float64))
            if var_diff_per_sample_flow_plain is not None
            else None,
            "mse_back_half_flow_projected_2": mse_mean_flow_projected_2,
            "fid_back_half_proxy_flow_projected_2": fid_proxy_flow_projected_2,
            "mse_back_half_std_flow_projected_2": float(np.std(mse_per_sample_flow_projected_2, dtype=np.float64))
            if mse_per_sample_flow_projected_2 is not None
            else None,
            "mean_diff_known_flow_projected_2": float(np.mean(mean_diff_per_sample_flow_projected_2, dtype=np.float64))
            if mean_diff_per_sample_flow_projected_2 is not None
            else None,
            "var_diff_known_flow_projected_2": float(np.mean(var_diff_per_sample_flow_projected_2, dtype=np.float64))
            if var_diff_per_sample_flow_projected_2 is not None
            else None,
            "mean_abs_diff_known_flow_projected_2": float(
                np.mean(np.abs(mean_diff_per_sample_flow_projected_2), dtype=np.float64)
            )
            if mean_diff_per_sample_flow_projected_2 is not None
            else None,
            "var_abs_diff_known_flow_projected_2": float(
                np.mean(np.abs(var_diff_per_sample_flow_projected_2), dtype=np.float64)
            )
            if var_diff_per_sample_flow_projected_2 is not None
            else None,
            "mse_back_half_flow": mse_mean_flow,
            "fid_back_half_proxy_flow": fid_proxy_flow,
            "mse_back_half_std_flow": float(np.std(mse_per_sample_flow, dtype=np.float64))
            if mse_per_sample_flow is not None
            else None,
            "mean_diff_known_flow": float(np.mean(mean_diff_per_sample_flow, dtype=np.float64))
            if mean_diff_per_sample_flow is not None
            else None,
            "var_diff_known_flow": float(np.mean(var_diff_per_sample_flow, dtype=np.float64))
            if var_diff_per_sample_flow is not None
            else None,
            "mean_abs_diff_known_flow": float(np.mean(np.abs(mean_diff_per_sample_flow), dtype=np.float64))
            if mean_diff_per_sample_flow is not None
            else None,
            "var_abs_diff_known_flow": float(np.mean(np.abs(var_diff_per_sample_flow), dtype=np.float64))
            if var_diff_per_sample_flow is not None
            else None,
            "mse_back_half_ann": baseline_mse_mean.get("ann"),
            "fid_back_half_proxy_ann": baseline_fid.get("ann"),
            "mse_back_half_std_ann": float(np.std(baseline_mse_per_sample["ann"], dtype=np.float64))
            if "ann" in baseline_mse_per_sample
            else None,
            "mean_diff_known_ann": float(np.mean(baseline_mean_diff_per_sample["ann"], dtype=np.float64))
            if "ann" in baseline_mean_diff_per_sample
            else None,
            "var_diff_known_ann": float(np.mean(baseline_var_diff_per_sample["ann"], dtype=np.float64))
            if "ann" in baseline_var_diff_per_sample
            else None,
            "mean_abs_diff_known_ann": float(np.mean(np.abs(baseline_mean_diff_per_sample["ann"]), dtype=np.float64))
            if "ann" in baseline_mean_diff_per_sample
            else None,
            "var_abs_diff_known_ann": float(np.mean(np.abs(baseline_var_diff_per_sample["ann"]), dtype=np.float64))
            if "ann" in baseline_var_diff_per_sample
            else None,
            "mse_back_half_cnn": baseline_mse_mean.get("cnn"),
            "fid_back_half_proxy_cnn": baseline_fid.get("cnn"),
            "mse_back_half_std_cnn": float(np.std(baseline_mse_per_sample["cnn"], dtype=np.float64))
            if "cnn" in baseline_mse_per_sample
            else None,
            "mean_diff_known_cnn": float(np.mean(baseline_mean_diff_per_sample["cnn"], dtype=np.float64))
            if "cnn" in baseline_mean_diff_per_sample
            else None,
            "var_diff_known_cnn": float(np.mean(baseline_var_diff_per_sample["cnn"], dtype=np.float64))
            if "cnn" in baseline_var_diff_per_sample
            else None,
            "mean_abs_diff_known_cnn": float(np.mean(np.abs(baseline_mean_diff_per_sample["cnn"]), dtype=np.float64))
            if "cnn" in baseline_mean_diff_per_sample
            else None,
            "var_abs_diff_known_cnn": float(np.mean(np.abs(baseline_var_diff_per_sample["cnn"]), dtype=np.float64))
            if "cnn" in baseline_var_diff_per_sample
            else None,
            # Backward-compatible aliases for projected repaint.
            "mse_back_half": mse_mean,
            "fid_back_half_proxy": fid_proxy,
            "mse_back_half_std": float(np.std(mse_per_sample, dtype=np.float64)),
        },
        "per_sample_metrics": per_sample_metrics,
    }

    CFG.paths.test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    CFG.paths.recon_output_path.parent.mkdir(parents=True, exist_ok=True)

    CFG.paths.test_metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    recon_payload: dict[str, np.ndarray] = {
        "test_indices": test_indices.astype(np.int64),
        "front_len": np.array([front_len], dtype=np.int64),
        "original_windows": test_windows.astype(np.float32),
        "masked_windows": masked_windows.astype(np.float32),
        "recon_windows": recon_windows.astype(np.float32),
        "recon_windows_plain": recon_windows_plain.astype(np.float32),
        "recon_windows_projected_2": recon_windows_projected_2.astype(np.float32),
        "true_back_half": true_back.astype(np.float32),
        "pred_back_half": pred_back.astype(np.float32),
        "pred_back_half_plain": pred_back_plain.astype(np.float32),
        "pred_back_half_dm_projected_2": pred_back_projected_2.astype(np.float32),
        "pred_back_half_projected_2": pred_back_projected_2.astype(np.float32),
        "mse_back_half_per_sample": mse_per_sample.astype(np.float64),
        "mse_back_half_per_sample_plain": mse_per_sample_plain.astype(np.float64),
        "mse_back_half_per_sample_dm_projected_2": mse_per_sample_projected_2.astype(np.float64),
        "mse_back_half_per_sample_projected_2": mse_per_sample_projected_2.astype(np.float64),
        "mean_diff_known_per_sample": mean_diff_per_sample.astype(np.float64),
        "var_diff_known_per_sample": var_diff_per_sample.astype(np.float64),
        "mean_diff_known_per_sample_plain": mean_diff_per_sample_plain.astype(np.float64),
        "var_diff_known_per_sample_plain": var_diff_per_sample_plain.astype(np.float64),
        "mean_diff_known_per_sample_dm_projected_2": mean_diff_per_sample_projected_2.astype(np.float64),
        "var_diff_known_per_sample_dm_projected_2": var_diff_per_sample_projected_2.astype(np.float64),
        "mean_diff_known_per_sample_projected_2": mean_diff_per_sample_projected_2.astype(np.float64),
        "var_diff_known_per_sample_projected_2": var_diff_per_sample_projected_2.astype(np.float64),
    }
    if (
        pred_back_flow_projected is not None
        and recon_windows_flow_projected is not None
        and mse_per_sample_flow_projected is not None
    ):
        recon_payload["recon_windows_flow_projected"] = recon_windows_flow_projected.astype(np.float32)
        recon_payload["pred_back_half_flow_projected"] = pred_back_flow_projected.astype(np.float32)
        recon_payload["mse_back_half_per_sample_flow_projected"] = mse_per_sample_flow_projected.astype(np.float64)
        if mean_diff_per_sample_flow_projected is not None and var_diff_per_sample_flow_projected is not None:
            recon_payload["mean_diff_known_per_sample_flow_projected"] = mean_diff_per_sample_flow_projected.astype(
                np.float64
            )
            recon_payload["var_diff_known_per_sample_flow_projected"] = var_diff_per_sample_flow_projected.astype(
                np.float64
            )
        # Backward-compatible aliases.
        recon_payload["recon_windows_flow"] = recon_windows_flow_projected.astype(np.float32)
        recon_payload["pred_back_half_flow"] = pred_back_flow_projected.astype(np.float32)
        recon_payload["mse_back_half_per_sample_flow"] = mse_per_sample_flow_projected.astype(np.float64)
        if mean_diff_per_sample_flow_projected is not None and var_diff_per_sample_flow_projected is not None:
            recon_payload["mean_diff_known_per_sample_flow"] = mean_diff_per_sample_flow_projected.astype(np.float64)
            recon_payload["var_diff_known_per_sample_flow"] = var_diff_per_sample_flow_projected.astype(np.float64)
    if pred_back_flow_plain is not None and recon_windows_flow_plain is not None and mse_per_sample_flow_plain is not None:
        recon_payload["recon_windows_flow_plain"] = recon_windows_flow_plain.astype(np.float32)
        recon_payload["pred_back_half_flow_plain"] = pred_back_flow_plain.astype(np.float32)
        recon_payload["mse_back_half_per_sample_flow_plain"] = mse_per_sample_flow_plain.astype(np.float64)
        if mean_diff_per_sample_flow_plain is not None and var_diff_per_sample_flow_plain is not None:
            recon_payload["mean_diff_known_per_sample_flow_plain"] = mean_diff_per_sample_flow_plain.astype(np.float64)
            recon_payload["var_diff_known_per_sample_flow_plain"] = var_diff_per_sample_flow_plain.astype(np.float64)
    if (
        pred_back_flow_projected_2 is not None
        and recon_windows_flow_projected_2 is not None
        and mse_per_sample_flow_projected_2 is not None
    ):
        recon_payload["recon_windows_flow_projected_2"] = recon_windows_flow_projected_2.astype(np.float32)
        recon_payload["pred_back_half_flow_projected_2"] = pred_back_flow_projected_2.astype(np.float32)
        recon_payload["mse_back_half_per_sample_flow_projected_2"] = mse_per_sample_flow_projected_2.astype(np.float64)
        if mean_diff_per_sample_flow_projected_2 is not None and var_diff_per_sample_flow_projected_2 is not None:
            recon_payload["mean_diff_known_per_sample_flow_projected_2"] = mean_diff_per_sample_flow_projected_2.astype(
                np.float64
            )
            recon_payload["var_diff_known_per_sample_flow_projected_2"] = var_diff_per_sample_flow_projected_2.astype(
                np.float64
            )
    if "ann" in baseline_predictions:
        recon_payload["pred_back_half_ann"] = baseline_predictions["ann"].astype(np.float32)
        recon_payload["mse_back_half_per_sample_ann"] = baseline_mse_per_sample["ann"].astype(np.float64)
        recon_payload["mean_diff_known_per_sample_ann"] = baseline_mean_diff_per_sample["ann"].astype(np.float64)
        recon_payload["var_diff_known_per_sample_ann"] = baseline_var_diff_per_sample["ann"].astype(np.float64)
    if "cnn" in baseline_predictions:
        recon_payload["pred_back_half_cnn"] = baseline_predictions["cnn"].astype(np.float32)
        recon_payload["mse_back_half_per_sample_cnn"] = baseline_mse_per_sample["cnn"].astype(np.float64)
        recon_payload["mean_diff_known_per_sample_cnn"] = baseline_mean_diff_per_sample["cnn"].astype(np.float64)
        recon_payload["var_diff_known_per_sample_cnn"] = baseline_var_diff_per_sample["cnn"].astype(np.float64)
    np.savez_compressed(CFG.paths.recon_output_path, **recon_payload)

    return {
        "metrics_path": str(CFG.paths.test_metrics_path),
        "recon_output_path": str(CFG.paths.recon_output_path),
        "checkpoint_loaded": bool(ckpt_report["loaded"]),
        "random_init_used": bool(ckpt_report["random_init_used"]),
        "mse_back_half_projected": mse_mean,
        "fid_back_half_proxy_projected": fid_proxy,
        "mean_abs_diff_known_projected": float(np.mean(np.abs(mean_diff_per_sample), dtype=np.float64)),
        "var_abs_diff_known_projected": float(np.mean(np.abs(var_diff_per_sample), dtype=np.float64)),
        "mse_back_half_plain": mse_mean_plain,
        "fid_back_half_proxy_plain": fid_proxy_plain,
        "mean_abs_diff_known_plain": float(np.mean(np.abs(mean_diff_per_sample_plain), dtype=np.float64)),
        "var_abs_diff_known_plain": float(np.mean(np.abs(var_diff_per_sample_plain), dtype=np.float64)),
        "mse_back_half_dm_projected_2": mse_mean_projected_2,
        "fid_back_half_proxy_dm_projected_2": fid_proxy_projected_2,
        "mean_abs_diff_known_dm_projected_2": float(np.mean(np.abs(mean_diff_per_sample_projected_2), dtype=np.float64)),
        "var_abs_diff_known_dm_projected_2": float(np.mean(np.abs(var_diff_per_sample_projected_2), dtype=np.float64)),
        "mse_back_half_projected_2": mse_mean_projected_2,
        "fid_back_half_proxy_projected_2": fid_proxy_projected_2,
        "mse_back_half_flow_projected": mse_mean_flow_projected,
        "fid_back_half_proxy_flow_projected": fid_proxy_flow_projected,
        "mean_abs_diff_known_flow_projected": float(np.mean(np.abs(mean_diff_per_sample_flow_projected), dtype=np.float64))
        if mean_diff_per_sample_flow_projected is not None
        else None,
        "var_abs_diff_known_flow_projected": float(np.mean(np.abs(var_diff_per_sample_flow_projected), dtype=np.float64))
        if var_diff_per_sample_flow_projected is not None
        else None,
        "mse_back_half_flow_plain": mse_mean_flow_plain,
        "fid_back_half_proxy_flow_plain": fid_proxy_flow_plain,
        "mean_abs_diff_known_flow_plain": float(np.mean(np.abs(mean_diff_per_sample_flow_plain), dtype=np.float64))
        if mean_diff_per_sample_flow_plain is not None
        else None,
        "var_abs_diff_known_flow_plain": float(np.mean(np.abs(var_diff_per_sample_flow_plain), dtype=np.float64))
        if var_diff_per_sample_flow_plain is not None
        else None,
        "mse_back_half_flow_projected_2": mse_mean_flow_projected_2,
        "fid_back_half_proxy_flow_projected_2": fid_proxy_flow_projected_2,
        "mean_abs_diff_known_flow_projected_2": float(
            np.mean(np.abs(mean_diff_per_sample_flow_projected_2), dtype=np.float64)
        )
        if mean_diff_per_sample_flow_projected_2 is not None
        else None,
        "var_abs_diff_known_flow_projected_2": float(
            np.mean(np.abs(var_diff_per_sample_flow_projected_2), dtype=np.float64)
        )
        if var_diff_per_sample_flow_projected_2 is not None
        else None,
        "mse_back_half_flow": mse_mean_flow,
        "fid_back_half_proxy_flow": fid_proxy_flow,
        "mean_abs_diff_known_flow": float(np.mean(np.abs(mean_diff_per_sample_flow), dtype=np.float64))
        if mean_diff_per_sample_flow is not None
        else None,
        "var_abs_diff_known_flow": float(np.mean(np.abs(var_diff_per_sample_flow), dtype=np.float64))
        if var_diff_per_sample_flow is not None
        else None,
        "mse_back_half_ann": baseline_mse_mean.get("ann"),
        "fid_back_half_proxy_ann": baseline_fid.get("ann"),
        "mean_abs_diff_known_ann": float(np.mean(np.abs(baseline_mean_diff_per_sample["ann"]), dtype=np.float64))
        if "ann" in baseline_mean_diff_per_sample
        else None,
        "var_abs_diff_known_ann": float(np.mean(np.abs(baseline_var_diff_per_sample["ann"]), dtype=np.float64))
        if "ann" in baseline_var_diff_per_sample
        else None,
        "mse_back_half_cnn": baseline_mse_mean.get("cnn"),
        "fid_back_half_proxy_cnn": baseline_fid.get("cnn"),
        "mean_abs_diff_known_cnn": float(np.mean(np.abs(baseline_mean_diff_per_sample["cnn"]), dtype=np.float64))
        if "cnn" in baseline_mean_diff_per_sample
        else None,
        "var_abs_diff_known_cnn": float(np.mean(np.abs(baseline_var_diff_per_sample["cnn"]), dtype=np.float64))
        if "cnn" in baseline_var_diff_per_sample
        else None,
        # Backward-compatible aliases for projected repaint.
        "mse_back_half": mse_mean,
        "fid_back_half_proxy": fid_proxy,
        "num_test_samples": int(test_indices.size),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
