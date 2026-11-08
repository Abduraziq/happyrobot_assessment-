#!/usr/bin/env python3
"""Load the finalized baseline and predict the EoT label for one audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from extract_features import FEATURE_NAMES, FEATURE_VERSION, extract_audio_features

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "baseline"


def load_model_bundle(model_dir: Path = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    config_path = model_dir / "model_config.json"
    model_path = model_dir / "selected_model.joblib"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("finalized") is not True:
        raise ValueError("Model configuration is not marked finalized")
    if config.get("feature_version") != FEATURE_VERSION or config.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Model feature configuration does not match this inference code")
    threshold = config.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Model configuration contains an invalid decision threshold")
    return {"model": joblib.load(model_path), "config": config}


def predict_audio_file(audio_path: str | Path, bundle: Dict[str, Any]) -> Dict[str, Any]:
    features = extract_audio_features(audio_path)
    return predict_feature_values(features, bundle)


def predict_feature_values(features: Dict[str, float], bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the saved model and threshold to an existing feature vector."""
    feature_frame = pd.DataFrame([features], columns=FEATURE_NAMES)
    model = bundle["model"]
    config = bundle["config"]
    classes = list(model.classes_)
    positive_index = classes.index(True) if True in classes else None
    probability = 0.0 if positive_index is None else float(model.predict_proba(feature_frame)[0, positive_index])
    threshold = float(config["threshold"])
    return {
        "endpoint_bool": probability >= threshold,
        "endpoint_probability": probability,
        "decision_threshold": threshold,
        "model": config["selected_model"],
        "feature_version": config["feature_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an audio segment is an end of turn.")
    parser.add_argument("--audio", type=Path, required=True, help="Audio file to classify.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    result = predict_audio_file(args.audio, load_model_bundle(args.model_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
