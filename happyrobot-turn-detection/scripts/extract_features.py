#!/usr/bin/env python3
"""Extract deterministic audio-only features for the EoT baseline."""

from __future__ import annotations

import argparse
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "acoustic-v1"
TARGET_SAMPLE_RATE = 16000
FRAME_LENGTH_MS = 25
FRAME_HOP_MS = 10
FFT_SIZE = 512
MEL_FILTER_COUNT = 26
MFCC_COUNT = 13
ROLLOFF_FRACTION = 0.85

_SUMMARY_NAMES = [
    "duration_seconds",
    "rms_mean",
    "rms_std",
    "rms_p10",
    "rms_p90",
    "short_time_energy_mean",
    "short_time_energy_std",
    "short_time_energy_p90",
    "low_energy_fraction",
    "zcr_mean",
    "zcr_std",
    "spectral_centroid_mean_hz",
    "spectral_centroid_std_hz",
    "spectral_bandwidth_mean_hz",
    "spectral_bandwidth_std_hz",
    "spectral_rolloff_mean_hz",
    "spectral_rolloff_std_hz",
    "spectral_flux_mean",
    "spectral_flux_std",
]
FEATURE_NAMES = _SUMMARY_NAMES + [
    f"mfcc_{index:02d}_{stat}"
    for index in range(MFCC_COUNT)
    for stat in ("mean", "std")
]


class FeatureExtractionError(ValueError):
    """Raised when an audio file cannot produce a complete feature vector."""


def resolve_audio_path(audio_path: str | Path, project_root: Path = ROOT) -> Path:
    path = Path(audio_path)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    return path


def _frame_signal(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate * FRAME_LENGTH_MS / 1000)))
    frame_hop = max(1, int(round(sample_rate * FRAME_HOP_MS / 1000)))
    frame_count = max(1, 1 + math.ceil(max(0, len(signal) - frame_length) / frame_hop))
    padded_length = (frame_count - 1) * frame_hop + frame_length
    padded = np.pad(signal, (0, max(0, padded_length - len(signal))))
    return np.lib.stride_tricks.sliding_window_view(padded, frame_length)[::frame_hop][:frame_count]


@lru_cache(maxsize=None)
def _frequency_grid(sample_rate: int) -> np.ndarray:
    frequencies = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)
    frequencies.setflags(write=False)
    return frequencies


@lru_cache(maxsize=None)
def _hanning_window(frame_length: int) -> np.ndarray:
    window = np.hanning(frame_length).astype(np.float32)
    window.setflags(write=False)
    return window


@lru_cache(maxsize=None)
def _mel_filterbank(sample_rate: int) -> np.ndarray:
    frequencies = _frequency_grid(sample_rate)
    max_mel = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
    mel_points = np.linspace(0.0, max_mel, MEL_FILTER_COUNT + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    filters = np.zeros((MEL_FILTER_COUNT, len(frequencies)), dtype=np.float64)
    for index in range(MEL_FILTER_COUNT):
        left, center, right = hz_points[index:index + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).eps)
        falling = (right - frequencies) / max(right - center, np.finfo(float).eps)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    filters.setflags(write=False)
    return filters


def extract_audio_features(audio_path: str | Path | BinaryIO | sf.SoundFile) -> Dict[str, float]:
    """Load, mono-mix, resample, and summarize one audio file."""
    is_sound_file = isinstance(audio_path, sf.SoundFile)
    is_path = isinstance(audio_path, (str, Path))
    source_label = str(audio_path) if is_path else "<audio stream>"
    try:
        if is_sound_file:
            audio_path.seek(0)
            audio = audio_path.read(dtype="float32", always_2d=True)
            source_rate = audio_path.samplerate
        else:
            source = str(audio_path) if is_path else audio_path
            if not is_path:
                audio_path.seek(0)
            audio, source_rate = sf.read(source, dtype="float32", always_2d=True)
    except (AttributeError, OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise FeatureExtractionError(f"Could not decode audio {source_label}: {exc}") from exc

    if source_rate <= 0 or audio.shape[0] == 0 or audio.shape[1] == 0:
        raise FeatureExtractionError(f"Audio is empty or has an invalid sample rate: {source_label}")
    duration = audio.shape[0] / float(source_rate)
    mono = np.mean(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    if source_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(source_rate) // divisor,
        ).astype(np.float32, copy=False)

    if mono.size == 0:
        raise FeatureExtractionError(f"Audio became empty after resampling: {source_label}")
    mono = np.clip(mono, -1.0, 1.0)
    frames = _frame_signal(mono, TARGET_SAMPLE_RATE)
    window = _hanning_window(frames.shape[1])
    windowed = frames * window[None, :]

    short_time_energy = np.mean(np.square(frames, dtype=np.float64), axis=1)
    rms = np.sqrt(short_time_energy)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    spectrum = np.abs(np.fft.rfft(windowed, n=FFT_SIZE, axis=1))
    power = np.square(spectrum)
    frequencies = _frequency_grid(TARGET_SAMPLE_RATE)
    magnitude_sum = np.maximum(np.sum(spectrum, axis=1), np.finfo(float).eps)
    centroid = np.sum(spectrum * frequencies[None, :], axis=1) / magnitude_sum
    bandwidth = np.sqrt(
        np.sum(power * np.square(frequencies[None, :] - centroid[:, None]), axis=1)
        / np.maximum(np.sum(power, axis=1), np.finfo(float).eps)
    )
    cumulative_power = np.cumsum(power, axis=1)
    rolloff_target = ROLLOFF_FRACTION * cumulative_power[:, -1]
    rolloff_bins = np.argmax(cumulative_power >= rolloff_target[:, None], axis=1)
    rolloff = frequencies[rolloff_bins]

    normalized_spectrum = spectrum / magnitude_sum[:, None]
    spectral_change = np.diff(normalized_spectrum, axis=0, prepend=normalized_spectrum[:1])
    flux = np.sqrt(np.mean(np.square(spectral_change), axis=1))

    mel_energy = power @ _mel_filterbank(TARGET_SAMPLE_RATE).T
    log_mel_energy = np.log(np.maximum(mel_energy, 1e-12))
    mfcc = dct(log_mel_energy, type=2, axis=1, norm="ortho")[:, :MFCC_COUNT]

    features: Dict[str, float] = {
        "duration_seconds": float(duration),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_p10": float(np.percentile(rms, 10)),
        "rms_p90": float(np.percentile(rms, 90)),
        "short_time_energy_mean": float(np.mean(short_time_energy)),
        "short_time_energy_std": float(np.std(short_time_energy)),
        "short_time_energy_p90": float(np.percentile(short_time_energy, 90)),
        "low_energy_fraction": float(np.mean(rms < 1e-4)),
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
        "spectral_centroid_mean_hz": float(np.mean(centroid)),
        "spectral_centroid_std_hz": float(np.std(centroid)),
        "spectral_bandwidth_mean_hz": float(np.mean(bandwidth)),
        "spectral_bandwidth_std_hz": float(np.std(bandwidth)),
        "spectral_rolloff_mean_hz": float(np.mean(rolloff)),
        "spectral_rolloff_std_hz": float(np.std(rolloff)),
        "spectral_flux_mean": float(np.mean(flux)),
        "spectral_flux_std": float(np.std(flux)),
    }
    for index in range(MFCC_COUNT):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(mfcc[:, index]))
        features[f"mfcc_{index:02d}_std"] = float(np.std(mfcc[:, index]))

    if list(features) != FEATURE_NAMES:
        raise RuntimeError("Feature names or ordering differ from the declared feature schema")
    if not np.isfinite(np.asarray(list(features.values()), dtype=np.float64)).all():
        raise FeatureExtractionError(f"Non-finite features were generated for {source_label}")
    return features


def extract_frame_features(
    frame: pd.DataFrame,
    project_root: Path = ROOT,
) -> pd.DataFrame:
    """Extract audio features while returning only IDs plus numeric features."""
    rows: List[Dict[str, object]] = []
    for record in frame.to_dict("records"):
        sample_id = str(record["sample_id"])
        path = resolve_audio_path(str(record["audio_path"]), project_root)
        rows.append({"sample_id": sample_id, **extract_audio_features(path)})
    return pd.DataFrame(rows, columns=["sample_id", *FEATURE_NAMES])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio-only EoT baseline features from a split CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Input split CSV containing sample_id and audio_path.")
    parser.add_argument("--output", type=Path, required=True, help="Output feature CSV (does not include target columns).")
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    required = {"sample_id", "audio_path"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input CSV must include {sorted(required)}")
    features = extract_frame_features(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    logging.info("Wrote %s audio-only feature rows to %s", len(features), args.output)


if __name__ == "__main__":
    main()
