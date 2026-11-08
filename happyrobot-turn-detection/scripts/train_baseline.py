#!/usr/bin/env python3
"""Fit lightweight audio-only EoT baselines using train and validation only."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import soundfile as sf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from extract_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    ROOT,
    TARGET_SAMPLE_RATE,
    extract_audio_features,
    extract_frame_features,
    resolve_audio_path,
)

SEED = 42
LABEL_COLUMN = "endpoint_bool"
MODEL_DIR = ROOT / "models" / "baseline"
REPORT_DIR = ROOT / "reports" / "baseline"
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VALIDATION_CSV = ROOT / "data" / "processed" / "validation.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_labeled_split(path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"sample_id", "audio_path", LABEL_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name} CSV is missing required columns: {sorted(missing)}")
    if not pd.api.types.is_bool_dtype(frame[LABEL_COLUMN]) or frame[LABEL_COLUMN].isna().any():
        raise ValueError(f"{split_name} {LABEL_COLUMN} must contain only nonmissing boolean values")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{split_name} sample IDs must be present and unique")
    return frame


def make_model_candidates(seed: int = SEED) -> Dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def positive_class_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    if True not in classes:
        return np.zeros(len(features), dtype=np.float64)
    positive_index = classes.index(True)
    return model.predict_proba(features)[:, positive_index]


def metrics_at_predictions(
    y_true: Iterable[bool],
    predictions: Iterable[bool],
    probabilities: np.ndarray | None = None,
    threshold: float | None = None,
) -> Dict[str, Any]:
    actual = np.asarray(list(y_true), dtype=bool)
    predicted = np.asarray(list(predictions), dtype=bool)
    matrix = confusion_matrix(actual, predicted, labels=[False, True])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "false_positive_count": fp,
        "false_negative_count": fn,
        "threshold": threshold,
    }
    if probabilities is not None and len(np.unique(actual)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(actual, probabilities))
        metrics["precision_recall_auc"] = float(average_precision_score(actual, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["precision_recall_auc"] = None
    return metrics


def select_validation_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> Tuple[float, Dict[str, Any]]:
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 0.5, 1.0]),
                np.asarray(probabilities, dtype=np.float64),
                np.asarray([np.nextafter(np.max(probabilities), 1.0)]),
            )
        )
    )
    evaluated = []
    for threshold in candidates:
        predictions = probabilities >= threshold
        metrics = metrics_at_predictions(y_true, predictions, probabilities, float(threshold))
        evaluated.append(metrics)
    best = max(
        evaluated,
        key=lambda item: (
            item["f1"],
            item["precision"],
            item["threshold"],
        ),
    )
    return float(best["threshold"]), best


def fit_and_compare(
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
    seed: int = SEED,
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]], str]:
    candidates = make_model_candidates(seed)
    comparison: Dict[str, Dict[str, Any]] = {}
    for name, model in candidates.items():
        model.fit(train_features, train_labels)
        probabilities = positive_class_probabilities(model, validation_features)
        if name == "dummy_most_frequent":
            predictions = model.predict(validation_features).astype(bool)
            metrics = metrics_at_predictions(validation_labels, predictions, probabilities)
        else:
            _, metrics = select_validation_threshold(validation_labels, probabilities)
        comparison[name] = metrics
        logging.info(
            "%s validation: F1=%.4f precision=%.4f recall=%.4f FP=%s FN=%s threshold=%s",
            name,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["false_positive_count"],
            metrics["false_negative_count"],
            metrics["threshold"],
        )

    selected_name = max(
        ("logistic_regression", "random_forest"),
        key=lambda name: (
            comparison[name]["f1"],
            comparison[name]["precision"],
        ),
    )
    return candidates, comparison, selected_name


def summarize_groups(frame: pd.DataFrame, error_mask: pd.Series, field: str) -> Dict[str, Any]:
    working = pd.DataFrame(
        {
            "group": frame[field].astype("string").fillna("<missing>"),
            "is_error": error_mask.astype(bool),
            "is_fp": frame["false_positive"].astype(bool),
            "is_fn": frame["false_negative"].astype(bool),
        }
    )
    output: Dict[str, Any] = {}
    for group, rows in working.groupby("group", dropna=False, sort=True):
        if len(rows) < 10:
            continue
        output[str(group)] = {
            "samples": int(len(rows)),
            "errors": int(rows["is_error"].sum()),
            "error_rate": float(rows["is_error"].mean()),
            "false_positives": int(rows["is_fp"].sum()),
            "false_negatives": int(rows["is_fn"].sum()),
        }
    return output


def validation_error_analysis(
    validation_frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    predicted = np.asarray(predictions, dtype=bool)
    actual = validation_frame[LABEL_COLUMN].to_numpy(dtype=bool)
    errors = predicted != actual
    rows = validation_frame.copy()
    rows["predicted_endpoint_bool"] = predicted
    rows["endpoint_probability"] = probabilities
    rows["decision_threshold"] = threshold
    rows["false_positive"] = (~actual) & predicted
    rows["false_negative"] = actual & (~predicted)
    rows["is_error"] = errors
    errors_frame = rows.loc[errors].copy()

    duration_bins = pd.cut(
        rows["duration_seconds"],
        bins=[-np.inf, 2.0, 5.0, 10.0, np.inf],
        labels=["under_2s", "2_to_5s", "5_to_10s", "over_10s"],
    )
    rows["duration_group"] = duration_bins.astype("string").fillna("<missing>")
    report: Dict[str, Any] = {"validation_errors": int(errors.sum()), "groups": {}}
    for field in (
        "duration_group",
        "low_energy_fraction",
        "sample_rate_hz",
        "language",
        "synthetic",
        "midfiller",
        "endfiller",
    ):
        if field in rows:
            if field == "low_energy_fraction":
                rows["low_energy_group"] = pd.cut(
                    rows[field],
                    bins=[-np.inf, 0.1, 0.5, 0.9, np.inf],
                    labels=["under_10pct", "10_to_50pct", "50_to_90pct", "over_90pct"],
                ).astype("string").fillna("<missing>")
                group_field = "low_energy_group"
            else:
                group_field = field
            report["groups"][group_field] = summarize_groups(rows, errors, group_field)
    return errors_frame, report


def percentile_summary(values: list[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(array)),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def benchmark_latency(
    model: Pipeline,
    threshold: float,
    validation_frame: pd.DataFrame,
    validation_features: pd.DataFrame,
    feature_names: list[str],
) -> Dict[str, Any]:
    feature_matrix = validation_features.loc[:, feature_names]
    model.predict_proba(feature_matrix.iloc[:1])

    model_only_ms: list[float] = []
    for index in range(len(feature_matrix)):
        row = feature_matrix.iloc[[index]]
        started = time.perf_counter()
        model.predict_proba(row)
        model_only_ms.append((time.perf_counter() - started) * 1000.0)

    total_ms: list[float] = []
    for audio_path in validation_frame["audio_path"]:
        resolved = resolve_audio_path(str(audio_path), ROOT)
        started = time.perf_counter()
        features = extract_audio_features(resolved)
        row = pd.DataFrame([features], columns=feature_names)
        probability = positive_class_probabilities(model, row)[0]
        _ = probability >= threshold
        total_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "measurement_split": "validation",
        "measurements": len(total_ms),
        "warm_file_cache": True,
        "model_only_prediction": percentile_summary(model_only_ms),
        "audio_load_feature_extraction_and_prediction": percentile_summary(total_ms),
        "p95_under_100ms_target": percentile_summary(total_ms)["p95_ms"] < 100.0,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unspecified",
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
        },
        "measurement_notes": (
            "End-to-end includes file open/decode, feature extraction, and one-row prediction; "
            "model-only excludes audio I/O and feature extraction. Model startup/load time is excluded."
        ),
    }


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight audio-only EoT baselines on train/validation splits."
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42).")
    args = parser.parse_args()
    seed = args.seed

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_frame = load_labeled_split(TRAIN_CSV, "train")
    validation_frame = load_labeled_split(VALIDATION_CSV, "validation")
    logging.info("Extracting features from train (%s) and validation (%s); test is not opened.", len(train_frame), len(validation_frame))
    train_features_with_id = extract_frame_features(train_frame, ROOT)
    validation_features_with_id = extract_frame_features(validation_frame, ROOT)
    train_features = train_features_with_id.loc[:, FEATURE_NAMES]
    validation_features = validation_features_with_id.loc[:, FEATURE_NAMES]
    train_labels = train_frame[LABEL_COLUMN]
    validation_labels = validation_frame[LABEL_COLUMN]

    models, comparison, selected_name = fit_and_compare(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        seed,
    )
    selected_model = models[selected_name]
    validation_probabilities = positive_class_probabilities(selected_model, validation_features)
    threshold, selected_metrics = select_validation_threshold(validation_labels, validation_probabilities)
    validation_predictions = validation_probabilities >= threshold

    model_path = MODEL_DIR / "selected_model.joblib"
    joblib.dump(selected_model, model_path)
    config = {
        "finalized": True,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_settings": {
            "target_sample_rate_hz": TARGET_SAMPLE_RATE,
            "frame_length_ms": 25,
            "frame_hop_ms": 10,
            "fft_size": 512,
            "mel_filter_count": 26,
            "mfcc_count": 13,
        },
        "random_seed": seed,
        "selected_model": selected_name,
        "model_parameters": {
            "logistic_regression": {"max_iter": 2000, "random_state": seed},
            "random_forest": {"n_estimators": 300, "min_samples_leaf": 2, "random_state": seed},
        }.get(selected_name),
        "threshold": threshold,
        "threshold_selection": "validation F1 maximization; ties prefer higher precision, then higher threshold",
        "validation_metrics": selected_metrics,
        "train_samples": len(train_frame),
        "validation_samples": len(validation_frame),
        "test_used_for_model_selection": False,
    }
    write_json(MODEL_DIR / "model_config.json", config)
    write_json(REPORT_DIR / "validation_model_comparison.json", comparison)
    write_json(REPORT_DIR / "validation_selected_model_metrics.json", selected_metrics)

    prediction_frame = validation_frame.copy()
    for feature_name in FEATURE_NAMES:
        prediction_frame[feature_name] = validation_features[feature_name].to_numpy()
    prediction_frame["endpoint_probability"] = validation_probabilities
    prediction_frame["predicted_endpoint_bool"] = validation_predictions
    prediction_frame["decision_threshold"] = threshold
    errors_frame, error_report = validation_error_analysis(
        prediction_frame,
        validation_predictions,
        validation_probabilities,
        threshold,
    )
    prediction_frame.to_csv(REPORT_DIR / "validation_predictions.csv", index=False)
    errors_frame.to_csv(REPORT_DIR / "validation_errors.csv", index=False)
    write_json(REPORT_DIR / "validation_error_analysis.json", error_report)

    latency = benchmark_latency(
        selected_model,
        threshold,
        validation_frame,
        validation_features,
        FEATURE_NAMES,
    )
    write_json(REPORT_DIR / "latency.json", latency)
    logging.info("Selected validation model: %s at threshold %.6f", selected_name, threshold)
    logging.info("Validation metrics: %s", selected_metrics)
    logging.info("Latency: %s", latency)
    logging.info("Saved finalized model artifact and validation reports.")


if __name__ == "__main__":
    main()
