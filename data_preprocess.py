from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from params import CFG


REQUIRED_MAT_KEYS = (
    "Wind_pressure_coefficients",
    "Location_of_measured_points",
    "Sample_frequency",
)
ROUND_DECIMALS = 8
STD_EPS = 1e-12


def _list_mat_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw TPU directory does not exist: {raw_dir}")
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Raw TPU path is not a directory: {raw_dir}")

    mat_files = sorted(raw_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {raw_dir}")
    return mat_files


def _as_scalar_positive_int(value: np.ndarray, key: str, file_path: Path) -> int:
    arr = np.asarray(value).squeeze()
    if arr.size != 1:
        raise ValueError(
            f"{key} must be scalar in {file_path.name}; got shape {np.asarray(value).shape}."
        )

    raw = float(arr.item())
    if not np.isfinite(raw) or raw <= 0:
        raise ValueError(f"{key} must be finite and positive in {file_path.name}; got {raw}.")

    parsed = int(round(raw))
    if abs(raw - parsed) > 1e-6:
        raise ValueError(f"{key} must be integer-like in {file_path.name}; got {raw}.")
    return parsed


def _validate_mat_payload(
    wind_pressure: np.ndarray, locations: np.ndarray, file_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    wind = np.asarray(wind_pressure)
    loc = np.asarray(locations)

    if wind.ndim != 2:
        raise ValueError(
            f"Wind_pressure_coefficients must be rank-2 [T, N] in {file_path.name}; got {wind.shape}."
        )
    if loc.ndim != 2 or loc.shape[0] != 4:
        raise ValueError(
            "Location_of_measured_points must be rank-2 [4, N] in "
            f"{file_path.name}; got {loc.shape}."
        )
    if wind.shape[1] != loc.shape[1]:
        raise ValueError(
            f"Tap dimension mismatch in {file_path.name}: wind {wind.shape}, loc {loc.shape}."
        )
    if wind.shape[0] <= 0 or wind.shape[1] <= 0:
        raise ValueError(
            f"Wind_pressure_coefficients must be non-empty in {file_path.name}; got {wind.shape}."
        )
    if not np.all(np.isfinite(wind)) or not np.all(np.isfinite(loc)):
        raise ValueError(f"Non-finite values found in {file_path.name}.")

    return wind.astype(np.float32, copy=False), loc.astype(np.float64, copy=False)


def _extract_location_rows(loc: np.ndarray, file_path: Path) -> tuple[np.ndarray, ...]:
    left = loc[0, :].astype(np.float32, copy=False)
    bottom = loc[1, :].astype(np.float32, copy=False)
    point_raw = loc[2, :]
    face_raw = loc[3, :]

    point_int = np.rint(point_raw)
    face_int = np.rint(face_raw)
    if not np.allclose(point_raw, point_int, atol=1e-6):
        raise ValueError(f"point_id values are not integer-like in {file_path.name}.")
    if not np.allclose(face_raw, face_int, atol=1e-6):
        raise ValueError(f"face values are not integer-like in {file_path.name}.")

    point_ids = point_int.astype(np.int64, copy=False)
    face_ids = face_int.astype(np.int64, copy=False)
    if np.any((face_ids < 1) | (face_ids > 4)):
        raise ValueError(f"face ids must be in [1, 4] in {file_path.name}.")

    return face_ids, left, bottom, point_ids


def _window_non_overlap(
    series: np.ndarray, window_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    time_steps, tap_count = series.shape
    windows_per_tap = time_steps // window_len
    if windows_per_tap <= 0:
        empty = np.empty((0, window_len), dtype=np.float32)
        return (
            empty,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            0,
        )

    trimmed = series[: windows_per_tap * window_len, :]
    windows = (
        trimmed.reshape(windows_per_tap, window_len, tap_count)
        .transpose(2, 0, 1)
        .reshape(tap_count * windows_per_tap, window_len)
        .astype(np.float32, copy=False)
    )
    tap_indices = np.repeat(np.arange(tap_count, dtype=np.int64), windows_per_tap)
    tap_window_indices = np.tile(np.arange(windows_per_tap, dtype=np.int64), tap_count)
    return windows, tap_indices, tap_window_indices, windows_per_tap


def _split_group_train_count(group_size: int, train_ratio: float) -> int:
    if group_size <= 0:
        return 0
    if group_size == 1:
        return 1

    count = int(np.floor(train_ratio * group_size))
    count = max(1, count)
    count = min(group_size - 1, count)
    return count


def _stratified_split(
    face_ids: np.ndarray,
    left_dists: np.ndarray,
    bottom_dists: np.ndarray,
    train_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(face_ids) == len(left_dists) == len(bottom_dists)):
        raise ValueError("Stratification inputs must have equal lengths.")

    groups: dict[tuple[int, float, float], list[int]] = {}
    for idx in range(face_ids.shape[0]):
        key = (
            int(face_ids[idx]),
            float(np.round(float(left_dists[idx]), ROUND_DECIMALS)),
            float(np.round(float(bottom_dists[idx]), ROUND_DECIMALS)),
        )
        groups.setdefault(key, []).append(int(idx))

    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []

    for key in sorted(groups.keys()):
        group_idx = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(group_idx)
        cut = _split_group_train_count(group_idx.size, train_ratio=train_ratio)
        train.extend(group_idx[:cut].tolist())
        test.extend(group_idx[cut:].tolist())

    train_idx = np.asarray(sorted(train), dtype=np.int64)
    test_idx = np.asarray(sorted(test), dtype=np.int64)
    total = face_ids.shape[0]

    if np.intersect1d(train_idx, test_idx).size != 0:
        raise ValueError("Train/test overlap detected.")
    if train_idx.size + test_idx.size != total:
        raise ValueError("Train/test split does not cover all samples.")

    # Guarantee both sides have at least one sample when the total sample count allows it.
    if total >= 2 and train_idx.size == 0:
        train_idx = np.asarray([int(test_idx[0])], dtype=np.int64)
        test_idx = np.asarray(sorted(test_idx[1:].tolist()), dtype=np.int64)
    if total >= 2 and test_idx.size == 0:
        test_idx = np.asarray([int(train_idx[-1])], dtype=np.int64)
        train_idx = np.asarray(sorted(train_idx[:-1].tolist()), dtype=np.int64)

    return train_idx, test_idx


def preprocess_data() -> dict[str, Any]:
    mat_files = _list_mat_files(CFG.paths.raw_tpu_data_dir)

    window_seconds = float(CFG.data.window_seconds)
    if window_seconds <= 0:
        raise ValueError(f"CFG.data.window_seconds must be > 0; got {window_seconds}.")

    train_ratio = float(getattr(CFG.data, "train_ratio", 0.8))
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"CFG.data.train_ratio must be in (0, 1); got {train_ratio}.")

    seed = int(CFG.data.seed)

    windows_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    left_parts: list[np.ndarray] = []
    bottom_parts: list[np.ndarray] = []
    file_id_parts: list[np.ndarray] = []
    tap_parts: list[np.ndarray] = []
    point_parts: list[np.ndarray] = []

    sample_frequency_hz: int | None = None
    window_len: int | None = None

    for file_id, file_path in enumerate(mat_files):
        mat = loadmat(file_path)
        missing = [key for key in REQUIRED_MAT_KEYS if key not in mat]
        if missing:
            raise KeyError(f"{file_path.name} missing required keys: {', '.join(missing)}")

        fs = _as_scalar_positive_int(mat["Sample_frequency"], "Sample_frequency", file_path)
        local_window_len = int(fs * window_seconds)
        if local_window_len <= 0:
            raise ValueError(
                f"Computed window length must be > 0 in {file_path.name}; got {local_window_len}."
            )

        if sample_frequency_hz is None:
            sample_frequency_hz = fs
            window_len = local_window_len
        elif fs != sample_frequency_hz:
            raise ValueError(
                "Inconsistent Sample_frequency across files: "
                f"expected {sample_frequency_hz}, got {fs} in {file_path.name}."
            )
        elif local_window_len != window_len:
            raise ValueError(
                f"Inconsistent window length across files: expected {window_len}, got {local_window_len}."
            )

        wind, loc = _validate_mat_payload(
            mat["Wind_pressure_coefficients"],
            mat["Location_of_measured_points"],
            file_path,
        )
        face_per_tap, left_per_tap, bottom_per_tap, point_per_tap = _extract_location_rows(
            loc, file_path
        )
        windows, tap_indices, _, windows_per_tap = _window_non_overlap(
            wind, window_len=local_window_len
        )
        if windows.shape[0] == 0:
            continue

        windows_parts.append(windows)
        face_parts.append(np.repeat(face_per_tap, windows_per_tap).astype(np.int64, copy=False))
        left_parts.append(np.repeat(left_per_tap, windows_per_tap).astype(np.float32, copy=False))
        bottom_parts.append(
            np.repeat(bottom_per_tap, windows_per_tap).astype(np.float32, copy=False)
        )
        point_parts.append(
            np.repeat(point_per_tap, windows_per_tap).astype(np.int64, copy=False)
        )
        tap_parts.append(tap_indices.astype(np.int64, copy=False))
        file_id_parts.append(np.full(windows.shape[0], file_id, dtype=np.int64))

    if not windows_parts:
        raise ValueError("No valid windows were produced from tpu_data/*.mat.")
    if sample_frequency_hz is None or window_len is None:
        raise RuntimeError("Internal error: sample frequency/window length unresolved.")

    windows_raw = np.concatenate(windows_parts, axis=0).astype(np.float32, copy=False)
    face_ids = np.concatenate(face_parts, axis=0).astype(np.int64, copy=False)
    left_dists = np.concatenate(left_parts, axis=0).astype(np.float32, copy=False)
    bottom_dists = np.concatenate(bottom_parts, axis=0).astype(np.float32, copy=False)
    point_ids = np.concatenate(point_parts, axis=0).astype(np.int64, copy=False)
    tap_indices = np.concatenate(tap_parts, axis=0).astype(np.int64, copy=False)
    file_ids = np.concatenate(file_id_parts, axis=0).astype(np.int64, copy=False)

    train_indices, test_indices = _stratified_split(
        face_ids=face_ids,
        left_dists=left_dists,
        bottom_dists=bottom_dists,
        train_ratio=train_ratio,
        seed=seed,
    )
    if train_indices.size == 0:
        raise ValueError("Train split is empty; cannot fit normalization.")

    train_windows = windows_raw[train_indices]
    mean = float(np.mean(train_windows, dtype=np.float64))
    std = float(np.std(train_windows, dtype=np.float64))
    std_applied = std if std > STD_EPS else 1.0
    windows = ((windows_raw - mean) / std_applied).astype(np.float32, copy=False)

    return {
        "windows": windows,
        "face_ids": face_ids,
        "left_dists": left_dists,
        "bottom_dists": bottom_dists,
        "train_indices": train_indices.astype(np.int64, copy=False),
        "test_indices": test_indices.astype(np.int64, copy=False),
        "mean": np.float32(mean),
        "std": np.float32(std),
        "window_len": np.int64(window_len),
        "sample_frequency_hz": np.int64(sample_frequency_hz),
        "file_ids": file_ids,
        "tap_indices": tap_indices,
        "point_ids": point_ids,
    }

