#!/usr/bin/env python3
"""Inspect the smart-turn dataset schema and generate a markdown report.

This script intentionally uses Hugging Face streaming mode so it never materializes the
entire dataset into memory or downloads the full corpus. It inspects a bounded number of
streamed records, summarizes the schema, and writes a readable Markdown report to
`docs/dataset_inspection.md`.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Audio, load_dataset

DATASET_NAME = "pipecat-ai/smart-turn-data-v3-train"
DATASET_REVISION = "909e604f990651631a596f5cb03358b3ee8fbad0"
MAX_EXAMPLES = 1000

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_streamed_dataset() -> Any:
    dataset = load_dataset(DATASET_NAME, split="train", revision=DATASET_REVISION, streaming=True)
    return dataset.cast_column("audio", Audio(sampling_rate=None, decode=False))


def find_audio_column(features: Dict[str, Any]) -> str:
    for name, value in features.items():
        if isinstance(value, Audio):
            return name
    raise ValueError(f"No audio column found in dataset features; available columns: {list(features.keys())}")


def is_boolean_feature(feature: Any) -> bool:
    dtype = getattr(feature, "dtype", None)
    return dtype == "bool" or str(feature).strip().lower() in {"bool", "value('bool')"}


def find_label_column(features: Dict[str, Any]) -> str:
    bool_candidates = [name for name, value in features.items() if is_boolean_feature(value)]
    for name in ["endpoint_bool", "label", "target"]:
        if name in bool_candidates:
            return name
    if bool_candidates:
        return bool_candidates[0]
    raise ValueError(f"No boolean label column found; available columns: {list(features.keys())}")


def get_examples(stream: Iterable[Dict[str, Any]], limit: int = MAX_EXAMPLES) -> List[Dict[str, Any]]:
    return list(islice(stream, limit))


def summarize_missing_values(
    examples: List[Dict[str, Any]],
    fields: Iterable[str] | None = None,
) -> Dict[str, int]:
    reported: Dict[str, int] = {}
    field_names = set(fields or ())
    for example in examples:
        field_names.update(example.keys())
    for field in sorted(field_names):
        missing = 0
        for ex in examples:
            value = ex.get(field)
            if value is None or (isinstance(value, str) and value == ""):
                missing += 1
        reported[field] = missing
    return reported


def summarize_audio_payload(audio: Any) -> Dict[str, Any]:
    if not isinstance(audio, dict):
        return {"payload_type": type(audio).__name__, "bytes_available": False, "path_available": False}
    encoded = audio.get("bytes")
    path = audio.get("path")
    return {
        "payload_type": type(audio).__name__,
        "bytes_available": isinstance(encoded, (bytes, bytearray, memoryview)) and len(encoded) > 0,
        "path_available": isinstance(path, str) and bool(path.strip()),
    }


def duplicate_counts(examples: List[Dict[str, Any]], id_field: str) -> Tuple[int, List[str]]:
    counts: Counter[str] = Counter()
    for ex in examples:
        identifier = ex.get(id_field)
        if identifier is not None and str(identifier).strip():
            counts[str(identifier)] += 1
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    return len(duplicates), duplicates


def detect_grouping_identifiers(example_keys: Iterable[str]) -> List[str]:
    known = {"speaker_id", "speakerid", "recording_id", "recordingid", "conversation_id", "conversationid"}
    return [key for key in example_keys if key.lower() in known]


def build_report(max_examples: int = MAX_EXAMPLES) -> str:
    dataset = load_streamed_dataset()
    features = dict(dataset.features)
    audio_col = find_audio_column(features)
    label_col = find_label_column(features)

    examples = get_examples(dataset, limit=max_examples)
    if not examples:
        raise RuntimeError("No examples were found in the streamed dataset.")

    valid_labels = [example.get(label_col) for example in examples if isinstance(example.get(label_col), bool)]
    label_counts = Counter(valid_labels)
    missing_labels = sum(1 for example in examples if example.get(label_col) is None)
    invalid_labels = sum(
        1 for example in examples
        if example.get(label_col) is not None and not isinstance(example.get(label_col), bool)
    )
    missing_by_field = summarize_missing_values(examples, features.keys())
    duplicate_id_count, duplicate_ids = duplicate_counts(examples, "id")
    example_keys = sorted({key for ex in examples for key in ex.keys()})
    grouping_ids = detect_grouping_identifiers(features.keys())

    report_lines: List[str] = []
    report_lines.append("# Dataset Inspection Report")
    report_lines.append("")
    report_lines.append("## Dataset source")
    report_lines.append(f"- Dataset: `{DATASET_NAME}`")
    report_lines.append(f"- Dataset revision: `{DATASET_REVISION}`")
    report_lines.append(f"- Streamed examples inspected: {len(examples)}")
    report_lines.append("- Streaming mode: enabled")
    report_lines.append("- Audio decoding: disabled during inspection; sampled payload structure is inspected without decoding audio")
    report_lines.append("")
    report_lines.append("## Schema summary")
    report_lines.append("")
    report_lines.append("| Column | Type | Present in schema | Notes |")
    report_lines.append("| --- | --- | --- | --- |")
    for key, value in features.items():
        notes = ""
        if key == audio_col:
            notes = "Audio payload column"
        elif key == label_col:
            notes = "Target column"
        elif key in {"language", "synthetic", "midfiller", "endfiller", "dataset"}:
            notes = "Metadata column"
        report_lines.append(f"| `{key}` | `{value}` | yes | {notes} |")
    report_lines.append("")
    report_lines.append(f"- Actual audio column: `{audio_col}`")
    report_lines.append(f"- Actual target column: `{label_col}`")
    report_lines.append(f"- Target schema feature: `{features[label_col]}`")
    report_lines.append(
        f"- Sampled target values: {len(valid_labels)} valid booleans, "
        f"{missing_labels} missing, {invalid_labels} present but non-boolean."
    )
    audio_feature = features[audio_col]
    report_lines.append(
        f"- Audio feature settings: sampling rate `{getattr(audio_feature, 'sampling_rate', None)}`, "
        f"decode `{getattr(audio_feature, 'decode', None)}`."
    )
    report_lines.append("")
    report_lines.append("## First 3 examples")
    report_lines.append("```python")
    for example in examples[:3]:
        preview = {
            "id": example.get("id"),
            "language": example.get("language"),
            label_col: example.get(label_col),
            "midfiller": example.get("midfiller"),
            "endfiller": example.get("endfiller"),
            "synthetic": example.get("synthetic"),
            "dataset": example.get("dataset"),
            "audio_payload": summarize_audio_payload(example.get(audio_col)),
        }
        report_lines.append(json.dumps(preview, ensure_ascii=False, sort_keys=True))
    report_lines.append("```")
    report_lines.append("")
    report_lines.append(f"## Class distribution (first {len(examples)} examples)")
    report_lines.append("")
    true_count = label_counts[True]
    false_count = label_counts[False]
    report_lines.append(f"- Positive (`True`): {true_count}")
    report_lines.append(f"- Negative (`False`): {false_count}")
    report_lines.append(f"- Missing labels: {missing_labels}")
    report_lines.append(f"- Invalid non-boolean labels: {invalid_labels}")
    valid_label_total = true_count + false_count
    report_lines.append(f"- Positive share among valid boolean labels: {true_count / max(1, valid_label_total):.2%}")
    report_lines.append(f"- Negative share among valid boolean labels: {false_count / max(1, valid_label_total):.2%}")
    report_lines.append("")
    report_lines.append("## Available metadata")
    report_lines.append("")
    report_metadata = [
        "language",
        "synthetic",
        "midfiller",
        "endfiller",
        "dataset",
        "id",
    ]
    for field in report_metadata:
        present = field in features
        report_lines.append(f"- `{field}`: {'present' if present else 'not present'}")
    report_lines.append("- `audio_duration` and `sample_rate`: not present as top-level schema fields; these are derived from successfully opened audio during preparation.")
    report_lines.append(
        "- Speaker / recording / conversation identifiers: "
        + (", ".join(f"`{key}`" for key in grouping_ids) if grouping_ids else "not present in the available schema")
    )
    payloads = [summarize_audio_payload(example.get(audio_col)) for example in examples]
    bytes_payloads = sum(1 for payload in payloads if payload["bytes_available"])
    path_payloads = sum(1 for payload in payloads if payload["path_available"])
    missing_audio_payloads = sum(
        1 for payload in payloads
        if not payload["bytes_available"] and not payload["path_available"]
    )
    report_lines.append(
        f"- Sampled audio payloads with bytes: {bytes_payloads}; with paths: {path_payloads}; "
        f"with neither: {missing_audio_payloads}. "
        "Payload storage is reported from observed records and is not assumed to be bytes-only."
    )
    report_lines.append("")
    report_lines.append("## Missing values and duplicate IDs")
    report_lines.append("")
    report_lines.append("| Field | Missing count |")
    report_lines.append("| --- | ---: |")
    for field in sorted(missing_by_field):
        report_lines.append(f"| `{field}` | {missing_by_field[field]} |")
    report_lines.append("")
    report_lines.append(f"- Duplicate `id` records in the first {len(examples)} examples: {duplicate_id_count}")
    report_lines.append(f"- Duplicate identifiers: {duplicate_ids[:10] if duplicate_ids else 'none'}")
    report_lines.append("")
    report_lines.append("## Observations")
    report_lines.append("")
    report_lines.append(f"1. The dataset is streamed as an iterable dataset with an `{audio_col}` column and a top-level target column.")
    report_lines.append("2. Audio payload availability was checked in sampled records; payloads may expose encoded bytes, a file path, both, or neither.")
    report_lines.append("3. Class balance includes only valid boolean target values; missing and non-boolean values are counted separately.")
    report_lines.append("4. No speaker-level grouping identifiers are available in the inspected schema, so the resulting splits cannot enforce speaker-level separation.")
    report_lines.append("5. Because of streaming and bounded inspection, the inspection result is a careful sample of the corpus rather than a full dataset scan.")
    report_lines.append("")
    report_lines.append("## Recommended next step")
    report_lines.append("")
    report_lines.append("Use `scripts/prepare_dataset.py` to perform a bounded-stream, class-aware collection, then save validated audio and metadata under `data/processed/`.")
    return "\n".join(report_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a bounded sample of the streamed smart-turn dataset.")
    parser.add_argument("--max-inspection", type=int, default=MAX_EXAMPLES, help="Maximum number of streamed rows to inspect.")
    args = parser.parse_args()
    if args.max_inspection <= 0:
        parser.error("--max-inspection must be greater than zero")
    report = build_report(max_examples=args.max_inspection)
    out_path = DOCS_DIR / "dataset_inspection.md"
    out_path.write_text(report, encoding="utf-8")
    logging.info("Inspection report saved to %s", out_path)


if __name__ == "__main__":
    main()
