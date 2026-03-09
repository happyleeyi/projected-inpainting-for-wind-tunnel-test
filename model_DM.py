from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from params import CFG


def extract(a: Tensor, t: Tensor, x_shape: torch.Size) -> Tensor:
    if t.ndim != 1:
        raise ValueError("t must be 1D [B].")
    out = a.to(device=t.device).gather(0, t)
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


def timestep_embedding(timesteps: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be 1D [B].")
    half = dim // 2
    if half == 0:
        return timesteps.float().unsqueeze(-1)
    exponent = -math.log(max_period) * torch.arange(
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


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.mlp(timestep_embedding(timesteps.long(), self.hidden_dim))


class ConditionEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        # Face ids are defined as 1..4 in the TPU dataset.
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
        self.time_embed = TimestepEmbedder(hidden_dim)
        self.final_fuse = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        required = ("face_id", "left_dist", "bottom_dist")
        for key in required:
            if key not in cond:
                raise KeyError(f"Missing condition key: {key}")

        face_id = cond["face_id"].long().clamp(min=1, max=4)
        face_emb = self.face_embedding(face_id)

        left = cond["left_dist"].float()
        bottom = cond["bottom_dist"].float()
        dist_feat = torch.stack([left, bottom], dim=-1)
        dist_emb = self.distance_mlp(dist_feat)

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


class DiT1DConditionalDiffusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        depth: int,
        heads: int,
        patch_size: int,
        cond_dim: int,
        sequence_length: int | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.sequence_length = sequence_length

        self.patch_embed = nn.Conv1d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cond_embed = ConditionEmbedder(hidden_dim=hidden_dim, cond_dim=cond_dim)
        self.blocks = nn.ModuleList([DiTBlock1D(hidden_dim=hidden_dim, heads=heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_ada_ln = nn.Linear(hidden_dim, hidden_dim * 2)
        nn.init.zeros_(self.final_ada_ln.weight)
        nn.init.zeros_(self.final_ada_ln.bias)
        self.unpatchify = nn.ConvTranspose1d(  # inverse of patchify in shape
            hidden_dim, 1, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x_t: Tensor, timesteps: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape [B, L].")
        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [B].")
        if x_t.shape[0] != timesteps.shape[0]:
            raise ValueError("Batch size mismatch between x_t and timesteps.")

        batch_size, length = x_t.shape
        x_t = x_t.float()
        timesteps = timesteps.long()

        pad_right = (self.patch_size - (length % self.patch_size)) % self.patch_size
        if pad_right > 0:
            x_t = F.pad(x_t, (0, pad_right))

        x = self.patch_embed(x_t.unsqueeze(1)).transpose(1, 2)
        pos = patch_position_embedding(x.shape[1], self.hidden_dim, x.device, x.dtype)
        x = x + pos

        cond_vec = self.cond_embed(timesteps=timesteps, cond=cond)
        for block in self.blocks:
            x = block(x, cond_vec)

        shift, scale = self.final_ada_ln(cond_vec).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)

        x = self.unpatchify(x.transpose(1, 2)).squeeze(1)
        if x.shape[0] != batch_size:
            raise RuntimeError("Unexpected batch dimension change.")
        return x[:, :length]


class DiffusionSchedule:
    def __init__(
        self,
        diffusion_steps: int | None = None,
        beta_start: float | None = None,
        beta_end: float | None = None,
        device: torch.device | str | None = None,
    ):
        self.diffusion_steps = diffusion_steps or CFG.model.diffusion_steps
        beta_start = beta_start if beta_start is not None else CFG.model.beta_start
        beta_end = beta_end if beta_end is not None else CFG.model.beta_end

        betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, dtype=torch.float32)
        if device is not None:
            betas = betas.to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, device=alpha_bars.device), alpha_bars[:-1]], dim=0)

        denom = (1.0 - alpha_bars).clamp(min=1e-20)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / denom
        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_bars_prev) / denom
        self.posterior_mean_coef2 = (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / denom

    def to(self, device: torch.device | str) -> "DiffusionSchedule":
        for name in (
            "betas",
            "alphas",
            "alpha_bars",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
            "posterior_variance",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def q_sample(self, x0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        noise = torch.randn_like(x0) if noise is None else noise
        return extract(self.sqrt_alpha_bars, t, x0.shape) * x0 + extract(
            self.sqrt_one_minus_alpha_bars, t, x0.shape
        ) * noise

    def predict_x0_from_eps(self, x_t: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        sqrt_ab = extract(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_omb = extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        return (x_t - sqrt_omb * eps) / sqrt_ab

    def p_mean_variance(
        self, model: nn.Module, x_t: Tensor, t: Tensor, cond: Mapping[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor]:
        eps = model(x_t, t, cond)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps)
        mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x0_pred + extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        var = extract(self.posterior_variance, t, x_t.shape)
        return mean, var, x0_pred


def build_model(sequence_length: int | None = None) -> nn.Module:
    return DiT1DConditionalDiffusion(
        hidden_dim=CFG.model.hidden_dim,
        depth=CFG.model.depth,
        heads=CFG.model.heads,
        patch_size=CFG.model.patch_size,
        cond_dim=CFG.model.cond_dim,
        sequence_length=sequence_length,
    )


def run_dummy_forward() -> Tensor:
    sequence_length = CFG.model.patch_size * 4
    batch_size = 2
    model = build_model(sequence_length=sequence_length)
    x_t = torch.randn(batch_size, sequence_length)
    timesteps = torch.randint(0, CFG.model.diffusion_steps, (batch_size,), dtype=torch.long)
    cond = {
        "face_id": torch.randint(1, 5, (batch_size,), dtype=torch.long),
        "left_dist": torch.randn(batch_size),
        "bottom_dist": torch.randn(batch_size),
    }
    with torch.no_grad():
        pred = model(x_t, timesteps, cond)
    if pred.shape != x_t.shape:
        raise RuntimeError(f"Unexpected output shape: {pred.shape}, expected {x_t.shape}.")
    return pred
