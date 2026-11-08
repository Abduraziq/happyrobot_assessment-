#!/usr/bin/env python3
"""Evaluate the already-finalized baseline once on the held-out test split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from extract_features import FEATURE_NAMES, FEATURE_VERSION, ROOT, extract_frame_features
from train_baseline import LABEL_COLUMN, metrics_at_predictions, positive_class_probabilities, write_json

MODEL_DIR = ROOT / "models" / "baseline"
REPORT_DIR = ROOT / "reports" / "baseline"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_finalized_bundle(model_dir: Path) -> tuple[Any, Dict[str, Any]]:
    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))
    if config.get("finalized") is not True:
        raise ValueError("Refusing test evaluation: model/threshold are not finalized from validation")
    if config.get("feature_version") != FEATURE_VERSION or config.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Refusing test evaluation: feature configuration differs from the saved model")
    if "threshold" not in config or "selected_model" not in config:
        raise ValueError("Refusing test evaluation: finalized model configuration is incomplete")
    return joblib.load(model_dir / "selected_model.joblib"), config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final evaluation on the held-out test split.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    model, config = load_finalized_bundle(args.model_dir)
    test_frame = pd.read_csv(args.test_csv)
    required = {"sample_id", "audio_path", LABEL_COLUMN}
    missing = required - set(test_frame.columns)
    if missing:
        raise ValueError(f"Test CSV is missing required columns: {sorted(missing)}")
    if not pd.api.types.is_bool_dtype(test_frame[LABEL_COLUMN]) or test_frame[LABEL_COLUMN].isna().any():
        raise ValueError("Test labels must contain only nonmissing boolean values")
    if test_frame["sample_id"].isna().any() or test_frame["sample_id"].duplicated().any():
        raise ValueError("Test sample IDs must be present and unique")

    logging.info("Finalized model and threshold loaded; extracting features for one test evaluation.")
    feature_rows = extract_frame_features(test_frame, ROOT)
    test_features = feature_rows.loc[:, FEATURE_NAMES]
    probabilities = positive_class_probabilities(model, test_features)
    threshold = float(config["threshold"])
    predictions = probabilities >= threshold
    metrics = metrics_at_predictions(
        test_frame[LABEL_COLUMN],
        predictions,
        probabilities,
        threshold,
    )
    report = {
        "split": "test",
        "model": config["selected_model"],
        "threshold": threshold,
        "test_samples": len(test_frame),
        "metrics": metrics,
        "test_used_for_model_selection_or_threshold": False,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.report_dir / "test_metrics.json", report)
    predictions_frame = pd.DataFrame(
        {
            "sample_id": test_frame["sample_id"],
            "expected_endpoint_bool": test_frame[LABEL_COLUMN],
            "predicted_endpoint_bool": predictions,
            "endpoint_probability": probabilities,
            "decision_threshold": threshold,
        }
    )
    predictions_frame.to_csv(args.report_dir / "test_predictions.csv", index=False)
    logging.info("Final test metrics: %s", report)


if __name__ == "__main__":
    main()
