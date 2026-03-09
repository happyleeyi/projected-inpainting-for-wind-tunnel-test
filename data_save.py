from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from data_preprocess import preprocess_data
from params import CFG


SCHEMA_VERSION = 1
REQUIRED_KEYS = (
    "windows",
    "face_ids",
    "left_dists",
    "bottom_dists",
    "train_indices",
    "test_indices",
    "mean",
    "std",
    "window_len",
    "sample_frequency_hz",
    "file_ids",
    "tap_indices",
    "point_ids",
)


def _to_int64_1d(name: str, arr: Any, expected_len: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {out.shape}.")
    if expected_len is not None and out.shape[0] != expected_len:
        raise ValueError(f"{name} length mismatch: expected {expected_len}, got {out.shape[0]}.")
    return out


def _to_float32_1d(name: str, arr: Any, expected_len: int | None = None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim != 1:
        raise ValueError(f"{name} must be rank-1, got shape {out.shape}.")
    if expected_len is not None and out.shape[0] != expected_len:
        raise ValueError(f"{name} length mismatch: expected {expected_len}, got {out.shape[0]}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values.")
    return out


def _normalize_bundle(bundle: dict[str, Any]) -> dict[str, np.ndarray]:
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise KeyError(f"Bundle missing required keys: {', '.join(missing)}.")

    windows = np.asarray(bundle["windows"], dtype=np.float32)
    if windows.ndim != 2:
        raise ValueError(f"windows must be rank-2 [N, L], got {windows.shape}.")
    if windows.shape[0] <= 0 or windows.shape[1] <= 0:
        raise ValueError(f"windows must have positive shape, got {windows.shape}.")
    if not np.all(np.isfinite(windows)):
        raise ValueError("windows contains non-finite values.")
    n_samples = int(windows.shape[0])

    arrays: dict[str, np.ndarray] = {
        "windows": windows,
        "face_ids": _to_int64_1d("face_ids", bundle["face_ids"], expected_len=n_samples),
        "left_dists": _to_float32_1d("left_dists", bundle["left_dists"], expected_len=n_samples),
        "bottom_dists": _to_float32_1d(
            "bottom_dists", bundle["bottom_dists"], expected_len=n_samples
        ),
        "train_indices": _to_int64_1d("train_indices", bundle["train_indices"]),
        "test_indices": _to_int64_1d("test_indices", bundle["test_indices"]),
        "mean": np.asarray(bundle["mean"], dtype=np.float32),
        "std": np.asarray(bundle["std"], dtype=np.float32),
        "window_len": np.asarray(bundle["window_len"], dtype=np.int64),
        "sample_frequency_hz": np.asarray(bundle["sample_frequency_hz"], dtype=np.int64),
        "file_ids": _to_int64_1d("file_ids", bundle["file_ids"], expected_len=n_samples),
        "tap_indices": _to_int64_1d("tap_indices", bundle["tap_indices"], expected_len=n_samples),
        "point_ids": _to_int64_1d("point_ids", bundle["point_ids"], expected_len=n_samples),
    }
    _validate_arrays(arrays)
    return arrays


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        raise KeyError(f"Missing required persisted keys: {', '.join(missing)}.")

    windows = arrays["windows"]
    if windows.ndim != 2:
        raise ValueError(f"windows must be rank-2 [N, L], got {windows.shape}.")
    if windows.dtype != np.float32:
        raise ValueError(f"windows dtype must be float32, got {windows.dtype}.")
    if not np.all(np.isfinite(windows)):
        raise ValueError("windows contains non-finite values.")

    n_samples, window_len = windows.shape
    if n_samples <= 0 or window_len <= 0:
        raise ValueError(f"windows must have positive shape, got {windows.shape}.")

    for key in ("face_ids", "train_indices", "test_indices", "file_ids", "tap_indices", "point_ids"):
        arr = arrays[key]
        if arr.dtype != np.int64:
            raise ValueError(f"{key} dtype must be int64, got {arr.dtype}.")
        if arr.ndim != 1:
            raise ValueError(f"{key} must be rank-1, got shape {arr.shape}.")

    for key in ("left_dists", "bottom_dists"):
        arr = arrays[key]
        if arr.dtype != np.float32:
            raise ValueError(f"{key} dtype must be float32, got {arr.dtype}.")
        if arr.ndim != 1:
            raise ValueError(f"{key} must be rank-1, got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key} contains non-finite values.")

    for key in ("face_ids", "left_dists", "bottom_dists", "file_ids", "tap_indices", "point_ids"):
        if arrays[key].shape[0] != n_samples:
            raise ValueError(
                f"{key} length mismatch: expected {n_samples}, got {arrays[key].shape[0]}."
            )

    if np.any((arrays["face_ids"] < 1) | (arrays["face_ids"] > 4)):
        raise ValueError("face_ids must be in [1, 4].")
    if np.any(arrays["file_ids"] < 0):
        raise ValueError("file_ids must be non-negative.")
    if np.any(arrays["tap_indices"] < 0):
        raise ValueError("tap_indices must be non-negative.")

    train_idx = arrays["train_indices"]
    test_idx = arrays["test_indices"]
    all_idx = np.concatenate((train_idx, test_idx), axis=0)
    if all_idx.size != n_samples:
        raise ValueError(
            "train_indices + test_indices must cover all samples exactly once: "
            f"got {all_idx.size} for {n_samples} samples."
        )
    if np.unique(all_idx).size != all_idx.size:
        raise ValueError("train_indices and test_indices overlap or contain duplicates.")
    if np.any(all_idx < 0) or np.any(all_idx >= n_samples):
        raise ValueError("train_indices/test_indices contain out-of-range values.")

    for key in ("mean", "std", "window_len", "sample_frequency_hz"):
        if arrays[key].ndim != 0:
            raise ValueError(f"{key} must be scalar, got shape {arrays[key].shape}.")

    mean = float(arrays["mean"])
    std = float(arrays["std"])
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError("mean/std must be finite scalars.")
    if float(arrays["window_len"]) != window_len:
        raise ValueError(
            f"window_len scalar mismatch: {float(arrays['window_len'])} vs windows.shape[1]={window_len}."
        )
    if int(arrays["sample_frequency_hz"]) <= 0:
        raise ValueError("sample_frequency_hz must be positive.")


def _json_summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "saved_keys": list(REQUIRED_KEYS),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "num_samples": int(arrays["windows"].shape[0]),
        "window_len": int(arrays["window_len"]),
        "sample_frequency_hz": int(arrays["sample_frequency_hz"]),
        "num_train": int(arrays["train_indices"].shape[0]),
        "num_test": int(arrays["test_indices"].shape[0]),
        "mean": float(arrays["mean"]),
        "std": float(arrays["std"]),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_processed_data(bundle: dict[str, Any] | None = None) -> dict[str, str]:
    if bundle is None:
        bundle = preprocess_data()

    arrays = _normalize_bundle(bundle)

    npz_path = CFG.paths.processed_npz_path
    json_path = CFG.paths.processed_json_path
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(npz_path, **arrays)
    _write_json(json_path, _json_summary(arrays))

    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
    }


def load_processed_data() -> dict[str, Any]:
    npz_path = CFG.paths.processed_npz_path
    json_path = CFG.paths.processed_json_path

    if not npz_path.exists():
        raise FileNotFoundError(f"Processed npz not found: {npz_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Processed json not found: {json_path}")

    with np.load(npz_path, allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    _validate_arrays(arrays)

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if int(meta.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema version in json: {meta.get('schema_version')} (expected {SCHEMA_VERSION})."
        )
    recorded_shapes = meta.get("shapes", {})
    recorded_dtypes = meta.get("dtypes", {})
    for key in REQUIRED_KEYS:
        if key in recorded_shapes and list(recorded_shapes[key]) != list(arrays[key].shape):
            raise ValueError(f"JSON shape mismatch for {key}.")
        if key in recorded_dtypes and str(recorded_dtypes[key]) != str(arrays[key].dtype):
            raise ValueError(f"JSON dtype mismatch for {key}.")

    return {
        "windows": arrays["windows"].astype(np.float32, copy=False),
        "face_ids": arrays["face_ids"].astype(np.int64, copy=False),
        "left_dists": arrays["left_dists"].astype(np.float32, copy=False),
        "bottom_dists": arrays["bottom_dists"].astype(np.float32, copy=False),
        "train_indices": arrays["train_indices"].astype(np.int64, copy=False),
        "test_indices": arrays["test_indices"].astype(np.int64, copy=False),
        "mean": np.float32(arrays["mean"]),
        "std": np.float32(arrays["std"]),
        "window_len": np.int64(arrays["window_len"]),
        "sample_frequency_hz": np.int64(arrays["sample_frequency_hz"]),
        "file_ids": arrays["file_ids"].astype(np.int64, copy=False),
        "tap_indices": arrays["tap_indices"].astype(np.int64, copy=False),
        "point_ids": arrays["point_ids"].astype(np.int64, copy=False),
    }

