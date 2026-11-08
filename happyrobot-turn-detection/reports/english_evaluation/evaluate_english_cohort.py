#!/usr/bin/env python3
"""Build and evaluate the approved, disjoint English-only holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Audio, load_dataset
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
DATASET = "pipecat-ai/smart-turn-data-v3-train"
REVISION = "909e604f990651631a596f5cb03358b3ee8fbad0"
SCAN_LIMIT = 20_000
SELECTION_SEED = 42
PER_CLASS = 400
EXPECTED_ID_HASH = "5683af05ba24d8ade049e631662e240dce072865a64ac2043ed0c047820727f4"
THRESHOLD = 0.3691340835714273
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_REPLICATES = 10_000

sys.path.insert(0, str(ROOT / "scripts"))
from extract_features import extract_audio_features  # noqa: E402
from predict import load_model_bundle, predict_feature_values  # noqa: E402
from prepare_dataset import (  # noqa: E402
    build_record_metadata,
    save_audio_record,
    validate_audio_payload,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_paths() -> list[Path]:
    paths = [
        ROOT / "models" / "baseline" / "selected_model.joblib",
        ROOT / "models" / "baseline" / "model_config.json",
        ROOT / "reports" / "baseline" / "test_metrics.json",
        ROOT / "reports" / "baseline" / "test_predictions.csv",
    ]
    for split in ("train", "validation", "test"):
        paths.extend(
            [
                ROOT / "data" / "metadata" / f"{split}.csv",
                ROOT / "data" / "processed" / f"{split}.csv",
            ]
        )
    return paths


def input_fingerprints() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in input_paths()}


def existing_split_ids() -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        ids_by_source: list[set[str]] = []
        for source in ("metadata", "processed"):
            path = ROOT / "data" / source / f"{split}.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                ids_by_source.append({row["sample_id"] for row in csv.DictReader(stream)})
        if ids_by_source[0] != ids_by_source[1]:
            raise RuntimeError(f"Existing {split} metadata and processed IDs differ")
        result[split] = ids_by_source[0]
    return result


def select_cohort(
    split_ids: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, int], int, int]:
    existing = set().union(*split_ids.values())
    pools: dict[str, dict[str, dict[str, Any]]] = {"positive": {}, "negative": {}}
    seen_english: set[str] = set()
    invalid_rows = 0
    duplicate_rows = 0
    dataset = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    dataset = dataset.select_columns(["id", "endpoint_bool", "language"])

    scanned = 0
    for row in islice(iter(dataset), SCAN_LIMIT):
        scanned += 1
        if str(row.get("language") or "").strip().lower() not in {"eng", "en", "english"}:
            continue
        sample_id = str(row.get("id") or "").strip()
        label = row.get("endpoint_bool")
        if not sample_id or not isinstance(label, bool):
            invalid_rows += 1
            continue
        if sample_id in seen_english:
            duplicate_rows += 1
            continue
        seen_english.add(sample_id)
        if sample_id in existing:
            continue
        label_name = "positive" if label else "negative"
        pools[label_name][sample_id] = {
            "sample_id": sample_id,
            "endpoint_bool": label,
            "language": "eng",
        }

    eligible = {label: len(rows) for label, rows in pools.items()}
    if scanned != SCAN_LIMIT:
        raise RuntimeError(f"Source ended after {scanned} rows, expected {SCAN_LIMIT}")
    if min(eligible.values()) < PER_CLASS:
        raise RuntimeError(f"Insufficient eligible candidates: {eligible}")

    rng = random.Random(SELECTION_SEED)
    selected: list[dict[str, Any]] = []
    for label in ("positive", "negative"):
        rows = [pools[label][sample_id] for sample_id in sorted(pools[label])]
        selected.extend(rng.sample(rows, PER_CLASS))

    selected_ids = {row["sample_id"] for row in selected}
    overlaps = {split: selected_ids & ids for split, ids in split_ids.items()}
    label_counts = Counter(row["endpoint_bool"] for row in selected)
    id_hash = hashlib.sha256(
        ("\n".join(sorted(selected_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        len(selected) != 2 * PER_CLASS
        or len(selected_ids) != len(selected)
        or label_counts != Counter({True: PER_CLASS, False: PER_CLASS})
        or any(overlaps.values())
        or id_hash != EXPECTED_ID_HASH
    ):
        raise RuntimeError(
            "Approved cohort verification failed: "
            f"eligible={eligible}, selected={len(selected)}, "
            f"overlaps={ {split: len(ids) for split, ids in overlaps.items()} }, "
            f"id_hash={id_hash}"
        )

    print(
        "Pre-extraction verification passed: "
        f"eligible positive={eligible['positive']}, negative={eligible['negative']}; "
        f"selected positive={label_counts[True]}, negative={label_counts[False]}; "
        f"overlaps={{'train': 0, 'validation': 0, 'test': 0}}; ID hash={id_hash}",
        flush=True,
    )
    return selected, eligible, invalid_rows, duplicate_rows


def extract_cohort(
    selected: list[dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=False)
    by_id = {row["sample_id"]: row for row in selected}
    optional_fields = ("synthetic", "midfiller", "endfiller", "dataset")
    dataset = load_dataset(DATASET, split="train", revision=REVISION, streaming=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=None, decode=False))
    available = set(dataset.features)
    columns = ["id", "endpoint_bool", "language", "audio"]
    columns.extend(field for field in optional_fields if field in available)
    dataset = dataset.select_columns(columns)

    metadata_rows: list[dict[str, Any]] = []
    found: set[str] = set()
    for row in islice(iter(dataset), SCAN_LIMIT):
        sample_id = str(row.get("id") or "").strip()
        expected = by_id.get(sample_id)
        if expected is None:
            continue
        if sample_id in found:
            raise RuntimeError(f"Duplicate selected source ID during extraction: {sample_id}")
        if (
            row.get("endpoint_bool") is not expected["endpoint_bool"]
            or str(row.get("language") or "").strip().lower() not in {"eng", "en", "english"}
        ):
            raise RuntimeError(f"Source metadata changed for selected ID {sample_id}")
        record = dict(row)
        record["_validated_audio"] = validate_audio_payload(row.get("audio"))
        audio_info = save_audio_record(record, audio_dir)
        metadata = build_record_metadata(record, audio_info)
        metadata["evaluation_split"] = "english_holdout"
        metadata_rows.append(metadata)
        found.add(sample_id)
        if len(found) % 100 == 0:
            print(f"Extracted and validated {len(found)}/{len(selected)} audio files", flush=True)

    missing = set(by_id) - found
    if missing:
        raise RuntimeError(
            f"Source did not provide {len(missing)} selected audio records; "
            "no replacement samples were selected"
        )
    return sorted(metadata_rows, key=lambda row: row["sample_id"])


def evaluate(
    metadata_rows: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    bundle = load_model_bundle(model_dir)
    if float(bundle["config"]["threshold"]) != THRESHOLD:
        raise RuntimeError("Frozen model threshold differs from the approved threshold")
    if bundle["config"].get("selected_model") != "logistic_regression":
        raise RuntimeError("Loaded model is not the expected frozen Logistic Regression baseline")

    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(metadata_rows, start=1):
        audio_path = ROOT / row["audio_path"]
        features = extract_audio_features(audio_path)
        prediction = predict_feature_values(features, bundle)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "expected_endpoint_bool": bool(row["endpoint_bool"]),
                "predicted_endpoint_bool": bool(prediction["endpoint_bool"]),
                "endpoint_probability": float(prediction["endpoint_probability"]),
                "decision_threshold": float(prediction["decision_threshold"]),
                "model": prediction["model"],
                "feature_version": prediction["feature_version"],
            }
        )
        if index % 100 == 0:
            print(f"Evaluated {index}/{len(metadata_rows)} examples", flush=True)

    y = np.asarray([row["expected_endpoint_bool"] for row in predictions], dtype=bool)
    probability = np.asarray([row["endpoint_probability"] for row in predictions], dtype=float)
    predicted = probability >= THRESHOLD
    tp = int(np.sum(y & predicted))
    tn = int(np.sum(~y & ~predicted))
    fp = int(np.sum(~y & predicted))
    fn = int(np.sum(y & ~predicted))
    point = {
        "accuracy": float((tp + tn) / len(y)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "average_precision": float(average_precision_score(y, probability)),
    }
    matrix = {
        "label_order": ["negative", "positive"],
        "rows": "actual",
        "columns": "predicted",
        "matrix": [[tn, fp], [fn, tp]],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    return predictions, point, matrix


def bootstrap_intervals(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    y = np.asarray([row["expected_endpoint_bool"] for row in predictions], dtype=bool)
    probability = np.asarray([row["endpoint_probability"] for row in predictions], dtype=float)
    positive = np.flatnonzero(y)
    negative = np.flatnonzero(~y)
    if not len(positive) or not len(negative):
        raise RuntimeError("Cannot bootstrap a cohort without both label classes")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = {
        name: [] for name in ("accuracy", "precision", "recall", "f1", "average_precision")
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        y_sample = y[indices]
        probability_sample = probability[indices]
        predicted_sample = probability_sample >= THRESHOLD
        tp = int(np.sum(y_sample & predicted_sample))
        tn = int(np.sum(~y_sample & ~predicted_sample))
        fp = int(np.sum(~y_sample & predicted_sample))
        fn = int(np.sum(y_sample & ~predicted_sample))
        samples["accuracy"].append((tp + tn) / len(indices))
        samples["precision"].append(tp / (tp + fp) if tp + fp else 0.0)
        samples["recall"].append(tp / (tp + fn) if tp + fn else 0.0)
        samples["f1"].append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        samples["average_precision"].append(
            float(average_precision_score(y_sample, probability_sample))
        )

    return {
        name: {
            "lower_95_percentile": float(np.quantile(values, 0.025)),
            "upper_95_percentile": float(np.quantile(values, 0.975)),
        }
        for name, values in samples.items()
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = BASE / "run_seed42_909e604f"
    parser.add_argument("--output-dir", type=Path, default=default_output)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(BASE.resolve()):
        parser.error("--output-dir must be under reports/english_evaluation")
    if output_dir.exists():
        parser.error(f"Refusing to overwrite an existing output directory: {output_dir}")

    split_ids = existing_split_ids()
    fingerprints_before = input_fingerprints()
    selected, eligible, invalid_rows, duplicate_rows = select_cohort(split_ids)

    # This point is reached only after the approved sample IDs pass every overlap check.
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_rows = extract_cohort(selected, output_dir)
    predictions, point_metrics, matrix = evaluate(
        metadata_rows, ROOT / "models" / "baseline"
    )
    intervals = bootstrap_intervals(predictions)
    fingerprints_after = input_fingerprints()
    if fingerprints_before != fingerprints_after:
        raise RuntimeError("A protected input file changed during cohort extraction/evaluation")

    selected_ids = sorted(row["sample_id"] for row in selected)
    cohort_hash = hashlib.sha256(("\n".join(selected_ids) + "\n").encode()).hexdigest()
    synthetic_counts = Counter(
        str(row.get("synthetic")).lower() for row in metadata_rows
    )
    manifest = {
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "stream_rows_scanned": SCAN_LIMIT,
        "selection_seed": SELECTION_SEED,
        "selection_method": (
            "Filter to unique English metadata rows after excluding all IDs in existing "
            "train/validation/test splits; sort eligible IDs within each label and sample "
            "400 per class using random.Random(42)."
        ),
        "eligible_additional_candidates": eligible,
        "invalid_english_rows_skipped": invalid_rows,
        "duplicate_english_ids_skipped": duplicate_rows,
        "cohort_size": len(selected_ids),
        "positive": sum(bool(row["endpoint_bool"]) for row in selected),
        "negative": sum(not bool(row["endpoint_bool"]) for row in selected),
        "selected_id_sha256_sorted_newline_delimited": cohort_hash,
        "approved_id_sha256": EXPECTED_ID_HASH,
        "overlap_counts": {"train": 0, "validation": 0, "test": 0},
        "synthetic_flag_counts": dict(synthetic_counts),
        "frozen_input_sha256": fingerprints_after,
    }
    write_json(output_dir / "cohort_manifest.json", manifest)
    write_csv(
        output_dir / "cohort_metadata.csv",
        metadata_rows,
        [
            "sample_id",
            "label",
            "endpoint_bool",
            "language",
            "synthetic",
            "midfiller",
            "endfiller",
            "source_dataset",
            "source_identifier",
            "audio_path",
            "audio_duration_seconds",
            "sample_rate_hz",
            "raw_audio_bytes",
            "evaluation_split",
        ],
    )
    write_csv(
        output_dir / "predictions.csv",
        predictions,
        [
            "sample_id",
            "expected_endpoint_bool",
            "predicted_endpoint_bool",
            "endpoint_probability",
            "decision_threshold",
            "model",
            "feature_version",
        ],
    )
    metrics = {
        "cohort_size": len(predictions),
        "class_balance": {
            "positive": int(np.sum([row["expected_endpoint_bool"] for row in predictions])),
            "negative": int(np.sum([not row["expected_endpoint_bool"] for row in predictions])),
        },
        "model": "logistic_regression",
        "feature_version": "acoustic-v1",
        "decision_threshold": THRESHOLD,
        "metrics": point_metrics,
        "confidence_intervals_95": {
            "method": (
                f"Stratified nonparametric bootstrap, {BOOTSTRAP_REPLICATES} replicates; "
                "resample with replacement within each true-label class; percentile interval."
            ),
            "seed": BOOTSTRAP_SEED,
            "metrics": intervals,
        },
        "evaluation_scope": "Separate English-only holdout; original test report was not read for tuning.",
    }
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "confusion_matrix.json", matrix)

    methodology = f"""# English-only held-out evaluation

## Cohort selection

- Source: `{DATASET}` at immutable revision `{REVISION}`.
- A metadata-only streaming scan inspected the first {SCAN_LIMIT:,} rows; it did not download the full corpus.
- Existing train, validation, and test IDs were read from both `data/metadata/` and `data/processed/`; the two copies were required to match.
- Existing IDs were excluded before candidates were pooled or sampled. There were {eligible['positive']} eligible additional positive and {eligible['negative']} eligible additional negative English examples in the bounded scan.
- English was identified by `language == eng` (also accepting equivalent `en`/`english` spellings).
- Unique eligible IDs were sorted within each label and sampled with `random.Random({SELECTION_SEED})`: {PER_CLASS} positive and {PER_CLASS} negative.
- The selected-ID SHA-256 is `{cohort_hash}`. Verified overlap counts: train 0, validation 0, test 0.
- Synthetic metadata: `{synthetic_counts.get("true", 0)}` flagged true and `{synthetic_counts.get("false", 0)}` flagged false ({synthetic_counts.get("true", 0) / len(metadata_rows):.2%} flagged true).
- Source audio was extracted only for the approved cohort and stored under this run's `audio/` directory. No source train/validation/test CSV or audio was modified.

## Frozen model and evaluation

The existing finalized `{metrics['model']}` artifact and saved threshold `{THRESHOLD:.16f}` were loaded unchanged. No training, model selection, or threshold tuning was performed. The evaluation uses the existing shared feature extraction and inference functions. Average precision is reported for this balanced 50/50 cohort and should not be interpreted as production-prevalence precision.

The 95% intervals are percentile intervals from {BOOTSTRAP_REPLICATES:,} stratified bootstrap replicates, resampling with replacement separately within positive and negative examples. Bootstrap seed: `{BOOTSTRAP_SEED}`.

## Results

| Metric | Estimate | 95% interval |
|---|---:|---:|
"""
    for name in ("precision", "recall", "f1", "accuracy", "average_precision"):
        ci = intervals[name]
        methodology += (
            f"| {name.replace('_', ' ').title()} | {point_metrics[name]:.4f} | "
            f"{ci['lower_95_percentile']:.4f}–{ci['upper_95_percentile']:.4f} |\n"
        )
    methodology += f"""

## Limitations

This is a separate English-only evaluation from the same source dataset, not an independent external-domain test. The cohort is balanced by design, so prevalence-sensitive metrics (especially precision and average precision) do not estimate production prevalence. The source is predominantly synthetic; language labels and sample IDs do not provide speaker-level independence. The results are evidence for this fixed cohort only and do not establish robust English generalization.
"""
    (output_dir / "methodology.md").write_text(methodology, encoding="utf-8", newline="\n")

    reproduction = f"""# Reproduction

Run from the project root in an environment with the serving requirements and `datasets` installed:

```powershell
python -m pip install -r requirements-serving.txt
python -m pip install datasets=={importlib.metadata.version('datasets')}
python reports\\english_evaluation\\evaluate_english_cohort.py --output-dir reports\\english_evaluation\\reproduction_run
```

The script pins dataset revision `{REVISION}`, scans at most {SCAN_LIMIT:,} streamed rows for metadata, excludes existing split IDs before sampling, uses seed `{SELECTION_SEED}`, and refuses to overwrite an existing output directory. It verifies the approved selected-ID hash before audio extraction. The script writes no files outside its selected output directory and checks hashes of the existing splits, model/config, and original test reports before and after execution.

The frozen model threshold is `{THRESHOLD:.16f}`. The bootstrap confidence intervals use {BOOTSTRAP_REPLICATES:,} stratified replicates and seed `{BOOTSTRAP_SEED}`. To reproduce in another fresh output location, choose a new directory under `reports\\english_evaluation\\`.
"""
    (output_dir / "reproduction.md").write_text(reproduction, encoding="utf-8", newline="\n")

    print(f"Completed English evaluation in: {output_dir}")
    print(json.dumps({"eligible": eligible, "selected": metrics["class_balance"], "metrics": point_metrics, "confidence_intervals_95": intervals}, indent=2))


if __name__ == "__main__":
    main()
