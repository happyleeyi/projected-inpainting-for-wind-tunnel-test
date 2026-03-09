from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class PathsConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    raw_tpu_data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    recon_outputs_dir: Path = field(init=False)
    processed_npz_path: Path = field(init=False)
    processed_json_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    dm_checkpoint_path: Path = field(init=False)
    ann_checkpoint_path: Path = field(init=False)
    cnn_checkpoint_path: Path = field(init=False)
    fm_checkpoint_path: Path = field(init=False)
    test_metrics_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    dm_metrics_path: Path = field(init=False)
    baseline_metrics_path: Path = field(init=False)
    fm_metrics_path: Path = field(init=False)
    recon_output_path: Path = field(init=False)
    spec_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.raw_tpu_data_dir = self.project_root / "tpu_data"
        self.artifacts_dir = self.project_root / "artifacts"
        self.processed_dir = self.artifacts_dir / "processed"
        self.checkpoints_dir = self.artifacts_dir / "checkpoints"
        self.metrics_dir = self.artifacts_dir / "metrics"
        self.recon_outputs_dir = self.artifacts_dir / "recon_outputs"
        self.processed_npz_path = self.processed_dir / "dataset.npz"
        self.processed_json_path = self.processed_dir / "dataset_meta.json"
        self.dm_checkpoint_path = self.checkpoints_dir / "model_last.pt"
        self.checkpoint_path = self.dm_checkpoint_path
        self.ann_checkpoint_path = self.checkpoints_dir / "ann_baseline_last.pt"
        self.cnn_checkpoint_path = self.checkpoints_dir / "cnn_baseline_last.pt"
        self.fm_checkpoint_path = self.checkpoints_dir / "flow_model_last.pt"
        self.test_metrics_path = self.metrics_dir / "test_metrics.json"
        self.dm_metrics_path = self.metrics_dir / "metrics.json"
        self.metrics_path = self.dm_metrics_path
        self.baseline_metrics_path = self.metrics_dir / "baseline_train_metrics.json"
        self.fm_metrics_path = self.metrics_dir / "fm_train_metrics.json"
        self.recon_output_path = self.recon_outputs_dir / "reconstruction.npz"
        self.spec_path = self.project_root / "spec" / "spec.json"

    def artifact_dirs(self) -> Tuple[Path, ...]:
        return (
            self.artifacts_dir,
            self.processed_dir,
            self.checkpoints_dir,
            self.metrics_dir,
            self.recon_outputs_dir,
        )

    def ensure_artifact_dirs(self) -> None:
        for directory in self.artifact_dirs():
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    window_seconds: float = 3.0
    train_ratio: float = 0.8
    seed: int = 42
    front_keep_ratio: float = 0.5
    sample_frequency_hz: int = 1000


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    depth: int = 4
    heads: int = 4
    patch_size: int = 16
    diffusion_steps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    cond_dim: int = 64


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_steps_per_epoch: Optional[int] = 10000


@dataclass
class TestConfig:
    batch_size: int = 16
    max_eval_samples: Optional[int] = 64
    compare_baselines: bool = True
    require_baseline_checkpoints: bool = True
    compare_flow: bool = True
    require_flow_checkpoint: bool = False
    flow_sample_steps: int = 200
    flow_use_projection: bool = True
    plot_flow_curve: bool = True
    repaint_jump_length: int = 10
    repaint_jump_n_sample: int = 10
    # Legacy knob kept for backward compatibility with older checkpoints/config dumps.
    repaint_resample_steps: int = 8
    projection_epsilon: float = 1e-6


@dataclass
class BaselineConfig:
    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_steps_per_epoch: Optional[int] = 10000
    ann_hidden_dim_1: int = 512
    ann_hidden_dim_2: int = 256
    ann_dropout: float = 0.1
    cnn_channels_1: int = 32
    cnn_channels_2: int = 64
    cnn_channels_3: int = 128
    cnn_dropout: float = 0.1
    cnn_pool_out_len: int = 8


@dataclass
class FlowTrainConfig:
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_steps_per_epoch: Optional[int] = 10000


@dataclass
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    deterministic: bool = True
    amp: bool = False


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    flow: FlowTrainConfig = field(default_factory=FlowTrainConfig)
    test: TestConfig = field(default_factory=TestConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def ensure_artifact_dirs(self) -> None:
        self.paths.ensure_artifact_dirs()


CFG = Config()
