from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from params import CFG


ConditionDict = Mapping[str, Tensor]


def continuous_timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be 1D [B].")
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    exponent = -math.log(float(max_period)) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    ) / max(half - 1, 1)
    freqs = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def patch_position_embedding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32)
    half = dim // 2
    if half == 0:
        return pos.view(1, length, 1).to(dtype=dtype)
    exponent = -math.log(10_000.0) * torch.arange(half, device=device, dtype=torch.float32) / max(
        half - 1, 1
    )
    freqs = torch.exp(exponent)
    args = pos.unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb.unsqueeze(0).to(dtype=dtype)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _as_batch_1d(
    value: Any,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=dtype)
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


def _prepare_condition(cond: Mapping[str, Any], batch_size: int, device: torch.device) -> dict[str, Tensor]:
    required = ("face_id", "left_dist", "bottom_dist")
    for key in required:
        if key not in cond:
            raise KeyError(f"Missing condition key: {key}")
    return {
        "face_id": _as_batch_1d(cond["face_id"], batch_size, device, torch.long, "face_id"),
        "left_dist": _as_batch_1d(cond["left_dist"], batch_size, device, torch.float32, "left_dist"),
        "bottom_dist": _as_batch_1d(cond["bottom_dist"], batch_size, device, torch.float32, "bottom_dist"),
    }


class ContinuousTimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, time_scale: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_scale = float(time_scale)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        scaled_t = timesteps.float() * self.time_scale
        return self.mlp(continuous_timestep_embedding(scaled_t, self.hidden_dim))


class FlowConditionEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int, time_scale: float):
        super().__init__()
        self.face_embedding = nn.Embedding(5, cond_dim)
        self.distance_mlp = nn.Sequential(
            nn.Linear(2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.label_fuse = nn.Sequential(
            nn.Linear(2 * cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_embed = ContinuousTimestepEmbedder(hidden_dim=hidden_dim, time_scale=time_scale)
        self.final_fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor, cond: ConditionDict) -> Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [B].")
        required = ("face_id", "left_dist", "bottom_dist")
        for key in required:
            if key not in cond:
                raise KeyError(f"Missing condition key: {key}")

        face_id = cond["face_id"].long().clamp(min=1, max=4)
        left = cond["left_dist"].float()
        bottom = cond["bottom_dist"].float()
        if face_id.shape != left.shape or left.shape != bottom.shape:
            raise ValueError("Condition tensors must all have shape [B].")

        face_emb = self.face_embedding(face_id)
        dist_emb = self.distance_mlp(torch.stack([left, bottom], dim=-1))
        label_emb = self.label_fuse(torch.cat([face_emb, dist_emb], dim=-1))
        time_emb = self.time_embed(timesteps)
        return self.final_fuse(time_emb + label_emb)


class DiTBlock1D(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ada_ln = nn.Linear(hidden_dim, hidden_dim * 6)
        nn.init.zeros_(self.ada_ln.weight)
        nn.init.zeros_(self.ada_ln.bias)

    def forward(self, x: Tensor, cond_vec: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(cond_vec).chunk(
            6, dim=-1
        )
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h

        h = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class DiT1DConditionalFlowMatching(nn.Module):
    """
    Forward contract:
    - x_t: [B, L] noisy/interpolated sequence at time t
    - t: [B] or scalar in [0, 1], where 1.0 ~ noise end and 0.0 ~ data end
    - cond: {"face_id": [B], "left_dist": [B], "bottom_dist": [B]}
    Returns:
    - velocity v_t with shape [B, L]
    """

    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        heads: int,
        patch_size: int,
        cond_dim: int,
        time_scale: float,
        sequence_length: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_size = int(patch_size)
        self.sequence_length = sequence_length

        self.patch_embed = nn.Conv1d(1, self.hidden_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cond_embed = FlowConditionEmbedder(
            hidden_dim=self.hidden_dim,
            cond_dim=int(cond_dim),
            time_scale=float(time_scale),
        )
        self.blocks = nn.ModuleList([DiTBlock1D(hidden_dim=self.hidden_dim, heads=int(heads)) for _ in range(int(depth))])
        self.final_norm = nn.LayerNorm(self.hidden_dim)
        self.final_ada_ln = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        nn.init.zeros_(self.final_ada_ln.weight)
        nn.init.zeros_(self.final_ada_ln.bias)
        self.unpatchify = nn.ConvTranspose1d(
            self.hidden_dim, 1, kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x_t: Tensor, t: Tensor, cond: ConditionDict) -> Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"x_t must have shape [B, L], got {tuple(x_t.shape)}")
        batch_size, length = x_t.shape

        x_t = x_t.float()
        t_vec = _as_batch_1d(t, batch_size, x_t.device, torch.float32, "t")

        cond_prepared = _prepare_condition(cond, batch_size=batch_size, device=x_t.device)

        pad_right = (self.patch_size - (length % self.patch_size)) % self.patch_size
        if pad_right > 0:
            x_t = F.pad(x_t, (0, pad_right))

        x = self.patch_embed(x_t.unsqueeze(1)).transpose(1, 2)
        x = x + patch_position_embedding(x.shape[1], self.hidden_dim, x.device, x.dtype)

        cond_vec = self.cond_embed(timesteps=t_vec, cond=cond_prepared)
        for block in self.blocks:
            x = block(x, cond_vec)

        shift, scale = self.final_ada_ln(cond_vec).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        v = self.unpatchify(x.transpose(1, 2)).squeeze(1)
        return v[:, :length]


def sample_flow_path(x0: Tensor, noise: Tensor, t: Tensor | float) -> Tensor:
    """
    Linear interpolation path used in flow matching:
    x_t = (1 - t) * x0 + t * noise
    """
    if x0.shape != noise.shape:
        raise ValueError(f"x0 and noise must match shape, got {tuple(x0.shape)} vs {tuple(noise.shape)}")
    t_vec = _as_batch_1d(t, x0.shape[0], x0.device, x0.dtype, "t")
    if torch.any((t_vec < 0.0) | (t_vec > 1.0)):
        raise ValueError("t must be in [0, 1] for sample_flow_path.")
    t_view = t_vec.view(-1, *([1] * (x0.ndim - 1)))
    return (1.0 - t_view) * x0 + t_view * noise


def euler_flow_sample(
    model: nn.Module,
    cond: Mapping[str, Any],
    shape: tuple[int, int],
    steps: int,
    device: torch.device | str,
    known_front: Tensor | None = None,
    front_len: int | None = None,
    use_projection: bool = False,
    projection_fn: Callable[[Tensor], Tensor] | Callable[[Tensor, ConditionDict], Tensor] | None = None,
) -> Tensor:
    if len(shape) != 2:
        raise ValueError(f"shape must be (B, L), got {shape}")
    batch_size, length = int(shape[0]), int(shape[1])
    if batch_size <= 0 or length <= 0:
        raise ValueError(f"Invalid shape: {shape}")
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if use_projection and projection_fn is None:
        raise ValueError("projection_fn must be provided when use_projection=True.")
    if known_front is None and front_len is not None:
        raise ValueError("front_len requires known_front.")

    target_device = torch.device(device)
    x = torch.randn(batch_size, length, device=target_device, dtype=torch.float32)
    cond_prepared = _prepare_condition(cond, batch_size=batch_size, device=target_device)

    known_front_target: Tensor | None = None
    front_keep = 0
    if known_front is not None:
        kf = known_front.to(device=target_device, dtype=x.dtype)
        if kf.ndim == 1:
            kf = kf.unsqueeze(0)
        if kf.ndim != 2:
            raise ValueError(f"known_front must be [B, L_front] or [L_front], got {tuple(kf.shape)}")
        if kf.shape[0] == 1 and batch_size > 1:
            kf = kf.expand(batch_size, -1)
        elif kf.shape[0] != batch_size:
            raise ValueError(f"known_front batch mismatch: expected {batch_size}, got {kf.shape[0]}")
        inferred_len = int(kf.shape[1]) if front_len is None else int(front_len)
        if inferred_len <= 0:
            raise ValueError(f"front_len must be > 0, got {inferred_len}")
        if inferred_len > length:
            raise ValueError(f"front_len={inferred_len} exceeds sample length={length}")
        if inferred_len > kf.shape[1]:
            raise ValueError(f"front_len={inferred_len} exceeds known_front length={kf.shape[1]}")
        front_keep = inferred_len
        known_front_target = kf[:, :front_keep]

    model = model.to(target_device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        time_grid = torch.linspace(1.0, 0.0, steps + 1, device=target_device, dtype=x.dtype)
        for i in range(steps):
            t_cur = torch.full((batch_size,), float(time_grid[i]), device=target_device, dtype=x.dtype)
            v_t = model(x, t_cur, cond_prepared)
            if v_t.shape != x.shape:
                raise RuntimeError(f"Model output shape {tuple(v_t.shape)} must match state {tuple(x.shape)}")

            dt = time_grid[i + 1] - time_grid[i]
            x = x + dt * v_t

            if known_front_target is not None:
                x[:, :front_keep] = known_front_target

            if use_projection and projection_fn is not None:
                try:
                    projected = projection_fn(x)
                except TypeError:
                    projected = projection_fn(x, cond_prepared)
                if not isinstance(projected, Tensor):
                    raise TypeError("projection_fn must return a torch.Tensor.")
                if projected.shape != x.shape:
                    raise ValueError(
                        f"projection_fn returned shape {tuple(projected.shape)}; expected {tuple(x.shape)}"
                    )
                x = projected.to(device=target_device, dtype=x.dtype)
                if known_front_target is not None:
                    x[:, :front_keep] = known_front_target

    if was_training:
        model.train()
    return x


def build_flow_model(sequence_length: int | None = None) -> nn.Module:
    return DiT1DConditionalFlowMatching(
        hidden_dim=CFG.model.hidden_dim,
        depth=CFG.model.depth,
        heads=CFG.model.heads,
        patch_size=CFG.model.patch_size,
        cond_dim=CFG.model.cond_dim,
        time_scale=float(CFG.model.diffusion_steps),
        sequence_length=sequence_length,
    )


def run_dummy_forward_flow() -> Tensor:
    sequence_length = int(CFG.model.patch_size) * 4
    batch_size = 2
    model = build_flow_model(sequence_length=sequence_length)
    x_t = torch.randn(batch_size, sequence_length)
    t = torch.rand(batch_size)
    cond = {
        "face_id": torch.randint(1, 5, (batch_size,), dtype=torch.long),
        "left_dist": torch.randn(batch_size),
        "bottom_dist": torch.randn(batch_size),
    }
    with torch.no_grad():
        v = model(x_t, t, cond)
    if v.shape != x_t.shape:
        raise RuntimeError(f"Unexpected output shape: {tuple(v.shape)}, expected {tuple(x_t.shape)}")
    return v


__all__ = [
    "DiT1DConditionalFlowMatching",
    "build_flow_model",
    "sample_flow_path",
    "euler_flow_sample",
    "run_dummy_forward_flow",
]

