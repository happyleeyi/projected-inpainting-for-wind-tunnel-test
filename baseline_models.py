from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from params import CFG


class ANNBackPredictor(nn.Module):
    def __init__(self, input_len: int, output_len: int):
        super().__init__()
        h1 = int(CFG.baseline.ann_hidden_dim_1)
        h2 = int(CFG.baseline.ann_hidden_dim_2)
        p = float(CFG.baseline.ann_dropout)
        self.net = nn.Sequential(
            nn.Linear(input_len, h1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(h2, output_len),
        )

    def forward(self, x_front: Tensor) -> Tensor:
        if x_front.ndim != 2:
            raise ValueError(f"x_front must be [B, L_front], got {tuple(x_front.shape)}")
        return self.net(x_front)


class CNNBackPredictor(nn.Module):
    def __init__(self, input_len: int, output_len: int):
        super().__init__()
        c1 = int(CFG.baseline.cnn_channels_1)
        c2 = int(CFG.baseline.cnn_channels_2)
        c3 = int(CFG.baseline.cnn_channels_3)
        pool_out = max(1, int(CFG.baseline.cnn_pool_out_len))
        p = float(CFG.baseline.cnn_dropout)

        self.feature = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(pool_out),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(c3 * pool_out, output_len),
        )
        self.input_len = int(input_len)

    def forward(self, x_front: Tensor) -> Tensor:
        if x_front.ndim != 2:
            raise ValueError(f"x_front must be [B, L_front], got {tuple(x_front.shape)}")
        if x_front.shape[1] != self.input_len:
            raise ValueError(
                f"Unexpected front length: expected {self.input_len}, got {int(x_front.shape[1])}"
            )
        h = self.feature(x_front.unsqueeze(1))
        return self.head(h)


def build_baseline_model(
    model_name: Literal["ann", "cnn"],
    input_len: int,
    output_len: int,
) -> nn.Module:
    name = str(model_name).lower()
    if name == "ann":
        return ANNBackPredictor(input_len=input_len, output_len=output_len)
    if name == "cnn":
        return CNNBackPredictor(input_len=input_len, output_len=output_len)
    raise ValueError(f"Unsupported baseline model: {model_name}")


def run_baseline_dummy_forward() -> dict[str, tuple[int, ...]]:
    input_len = 32
    output_len = 32
    x = torch.randn(2, input_len)

    ann = build_baseline_model("ann", input_len=input_len, output_len=output_len)
    cnn = build_baseline_model("cnn", input_len=input_len, output_len=output_len)

    with torch.no_grad():
        y_ann = ann(x)
        y_cnn = cnn(x)

    if y_ann.shape != (2, output_len):
        raise RuntimeError(f"ANN output shape mismatch: {tuple(y_ann.shape)}")
    if y_cnn.shape != (2, output_len):
        raise RuntimeError(f"CNN output shape mismatch: {tuple(y_cnn.shape)}")

    return {
        "input": tuple(x.shape),
        "ann_output": tuple(y_ann.shape),
        "cnn_output": tuple(y_cnn.shape),
    }

