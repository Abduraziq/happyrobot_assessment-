#!/usr/bin/env python3
"""Prepare a balanced, streaming-selected End-of-Turn dataset.

The script performs a bounded stream inspection first, then a deterministic
class-balanced sampling pass. It saves selected audio to `data/processed/audio/`
and metadata CSVs under `data/processed/` and `data/metadata/`.

This task intentionally does not train a model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import random
import re
import shutil
import tempfile
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

DATASET_NAME = "pipecat-ai/smart-turn-data-v3-train"
DATASET_REVISION = "909e604f990651631a596f5cb03358b3ee8fbad0"
LABEL_COLUMN = "endpoint_bool"
ID_COLUMN = "id"
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = PROCESSED_DIR / "audio"
METADATA_DIR = DATA_DIR / "metadata"
DOCS_DIR = ROOT / "docs"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

AUDIO_EXTENSIONS = {
    "FLAC": ".flac",
    "WAV": ".wav",
    "WAVEX": ".wav",
    "RF64": ".wav",
    "OGG": ".ogg",
    "AIFF": ".aiff",
    "AIFC": ".aifc",
    "AU": ".au",
    "CAF": ".caf",
    "MAT4": ".mat",
    "MAT5": ".mat",
    "NIST": ".nist",
    "W64": ".w64",
}


def load_streamed_dataset() -> Any:
    dataset = load_dataset(DATASET_NAME, split="train", revision=DATASET_REVISION, streaming=True)
    return dataset.cast_column("audio", Audio(sampling_rate=None, decode=False))


def inspect_label_distribution(
    dataset: Iterable[Dict[str, Any]],
    max_examples: int = 20000,
) -> Tuple[Counter, int]:
    labels: Counter = Counter()
    scanned = 0
    invalid = 0
    missing = 0
    for example in islice(dataset, max_examples):
        scanned += 1
        label = example.get(LABEL_COLUMN)
        if isinstance(label, bool):
            labels[label] += 1
        elif label is None:
            missing += 1
        else:
            invalid += 1
    logging.info(
        "Inspected %s examples for label balance; valid distribution=%s, missing labels=%s, invalid labels=%s",
        scanned,
        dict(sorted(labels.items())),
        missing,
        invalid,
    )
    return labels, scanned


def get_audio_source(audio: Any) -> bytes | Path:
    """Return the first structurally usable source in a Hugging Face audio payload."""
    sources = get_audio_sources(audio)
    if not sources:
        raise ValueError("Audio payload contains neither usable bytes nor an existing file path")
    return sources[0]


def get_audio_sources(audio: Any) -> List[bytes | Path]:
    if not isinstance(audio, dict):
        return []

    sources: List[bytes | Path] = []
    encoded = audio.get("bytes")
    if isinstance(encoded, (bytes, bytearray, memoryview)) and encoded:
        sources.append(bytes(encoded))

    path = audio.get("path")
    if isinstance(path, str) and path.strip():
        audio_path = Path(path)
        if audio_path.is_file():
            sources.append(audio_path)

    return sources


def inspect_audio_source(
    source: bytes | Path,
    fully_decode: bool = True,
) -> Tuple[str, int, int]:
    """Inspect an audio source and optionally decode all frames."""
    handle: Any = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    with sf.SoundFile(handle) as audio_file:
        audio_format = str(audio_file.format).upper()
        sample_rate = int(audio_file.samplerate)
        frames = int(audio_file.frames)
        if sample_rate <= 0 or frames <= 0 or audio_file.channels <= 0:
            raise ValueError("Audio has invalid sample rate, frame count, or channel count")
        if fully_decode:
            decoded = audio_file.read(dtype="float32")
            if len(decoded) != frames:
                raise ValueError(f"Audio decoded {len(decoded)} frames but its header declares {frames}")
    return audio_format, sample_rate, frames


def validate_audio_payload(audio: Any) -> Tuple[bytes | Path, str, int, int]:
    errors: List[str] = []
    for source in get_audio_sources(audio):
        try:
            audio_format, sample_rate, frames = inspect_audio_source(source)
            if audio_format in AUDIO_EXTENSIONS:
                return source, audio_format, sample_rate, frames
            errors.append(f"unsupported audio container {audio_format!r}")
        except (OSError, RuntimeError, TypeError, ValueError, sf.LibsndfileError) as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no encoded bytes or existing file path"
    raise ValueError(f"Audio payload has no valid supported source: {detail}")


def estimate_storage_requirements(dataset: Iterable[Dict[str, Any]], max_examples: int = 1000) -> Tuple[float, float, float, int]:
    sizes: List[int] = []
    for example in islice(dataset, max_examples):
        try:
            source = get_audio_source(example.get("audio"))
        except (OSError, ValueError):
            continue
        sizes.append(len(source) if isinstance(source, bytes) else source.stat().st_size)
    if not sizes:
        return 0.0, 0.0, 0.0, 0
    avg_bytes = sum(sizes) / len(sizes)
    max_bytes = max(sizes)
    min_bytes = min(sizes)
    estimated_mb = (avg_bytes * 5000) / (1024 * 1024)
    return avg_bytes, min_bytes, max_bytes, estimated_mb


def select_balanced_records(
    dataset: Iterable[Dict[str, Any]],
    target_total: int,
    seed: int = SEED,
    max_scan_examples: int | None = None,
    validate_audio_payloads: bool = False,
) -> List[Dict[str, Any]]:
    """Select a balanced subset using a bounded streaming scan.

    Each class uses an independent reservoir and random-number generator. The
    stream is scanned to its end or the configured bound even after both
    reservoirs reach capacity.
    """
    if target_total <= 0:
        raise ValueError("target_total must be greater than zero")
    if max_scan_examples is not None and max_scan_examples < 0:
        raise ValueError("max_scan_examples cannot be negative")

    target_per_label = {True: target_total // 2, False: target_total - target_total // 2}
    selected_by_label: Dict[bool, List[Dict[str, Any]]] = {True: [], False: []}
    seen_by_label = {True: 0, False: 0}
    rng_by_label = {True: random.Random(seed), False: random.Random(seed + 1)}
    seen_ids: set[str] = set()
    scanned = 0
    invalid_labels = 0
    missing_labels = 0
    missing_ids = 0
    duplicate_ids = 0
    invalid_audio = 0

    scan = dataset if max_scan_examples is None else islice(dataset, max_scan_examples)
    for example in scan:
        scanned += 1

        label = example.get(LABEL_COLUMN)
        if not isinstance(label, bool):
            if label is None:
                missing_labels += 1
            else:
                invalid_labels += 1
            continue

        raw_id = example.get(ID_COLUMN)
        if raw_id is None or str(raw_id).strip() == "":
            missing_ids += 1
            continue
        sample_id = str(raw_id).strip()
        if sample_id in seen_ids:
            duplicate_ids += 1
            continue
        if validate_audio_payloads:
            try:
                validated_audio = validate_audio_payload(example.get("audio"))
            except (OSError, RuntimeError, TypeError, ValueError, sf.LibsndfileError):
                invalid_audio += 1
                continue
            selected_record = dict(example)
            selected_record["_validated_audio"] = validated_audio
        else:
            selected_record = example

        seen_ids.add(sample_id)
        label_key = label
        pool = selected_by_label[label_key]
        seen_by_label[label_key] += 1
        seen = seen_by_label[label_key]

        if len(pool) < target_per_label[label_key]:
            pool.append(selected_record)
        else:
            j = rng_by_label[label_key].randrange(seen)
            if j < target_per_label[label_key]:
                pool[j] = selected_record

    selected = selected_by_label[True] + selected_by_label[False]
    logging.info(
        "Inspected %s stream records; valid class records: positive=%s, negative=%s; selected: positive=%s, negative=%s",
        scanned,
        seen_by_label[True],
        seen_by_label[False],
        len(selected_by_label[True]),
        len(selected_by_label[False]),
    )
    logging.info(
        "Selection exclusions: missing labels=%s, invalid labels=%s, missing IDs=%s, duplicate IDs=%s, invalid audio=%s",
        missing_labels,
        invalid_labels,
        missing_ids,
        duplicate_ids,
        invalid_audio,
    )
    return selected


def sanitize_filename(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw)).strip("._")
    return cleaned or "sample"


def save_audio_record(record: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    raw_id = record.get(ID_COLUMN)
    sample_id = "" if raw_id is None else str(raw_id).strip()
    if not sample_id:
        raise ValueError("Cannot save audio for a record without a sample ID")
    cached_validation = record.get("_validated_audio")
    if cached_validation is None:
        source, expected_format, expected_rate, expected_frames = validate_audio_payload(record.get("audio"))
    else:
        source, expected_format, expected_rate, expected_frames = cached_validation
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    file_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            if isinstance(source, bytes):
                temporary.write(source)
            else:
                with source.open("rb") as source_file:
                    shutil.copyfileobj(source_file, temporary)

        audio_format, sample_rate, frames = inspect_audio_source(temporary_path, fully_decode=False)
        extension = AUDIO_EXTENSIONS.get(audio_format)
        if extension is None:
            raise ValueError(f"Unsupported audio container {audio_format!r} for sample {sample_id}")
        if (audio_format, sample_rate, frames) != (expected_format, expected_rate, expected_frames):
            raise ValueError(f"Audio changed while saving sample {sample_id}")

        id_digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        stem = f"{sanitize_filename(sample_id)}-{id_digest}"
        file_path = output_dir / f"{stem}{extension}"
        temporary_path.replace(file_path)

        duration_seconds = expected_frames / float(expected_rate)
    except (OSError, RuntimeError, TypeError, ValueError, sf.LibsndfileError) as exc:
        if file_path is not None and file_path.exists():
            file_path.unlink()
        raise ValueError(f"Audio file could not be reopened for {sample_id}: {exc}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if file_path is None:
        raise RuntimeError(f"Audio file path was not created for sample {sample_id}")
    metadata = {
        "sample_id": sample_id,
        "audio_path": str(file_path.relative_to(ROOT)),
        "audio_duration_seconds": round(duration_seconds, 6),
        "sample_rate_hz": sample_rate,
    }
    return metadata


def deduplicate_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_ids: set[str] = set()
    unique_records: List[Dict[str, Any]] = []
    for record in records:
        raw_id = record.get(ID_COLUMN)
        sample_id = "" if raw_id is None else str(raw_id).strip()
        if not sample_id:
            continue
        if sample_id in seen_ids:
            logging.warning("Skipping duplicate sample id %s", sample_id)
            continue
        seen_ids.add(sample_id)
        unique_records.append(record)
    return unique_records


def build_record_metadata(record: Dict[str, Any], audio_info: Dict[str, Any]) -> Dict[str, Any]:
    label = record.get(LABEL_COLUMN)
    if not isinstance(label, bool):
        raise ValueError(f"Invalid {LABEL_COLUMN} value for sample {record.get(ID_COLUMN)!r}: {label!r}")
    raw_id = record.get(ID_COLUMN)
    sample_id = "" if raw_id is None else str(raw_id).strip()
    metadata = {
        "sample_id": sample_id,
        "label": label,
        LABEL_COLUMN: label,
        "language": record.get("language"),
        "synthetic": record.get("synthetic"),
        "midfiller": record.get("midfiller"),
        "endfiller": record.get("endfiller"),
        "source_dataset": record.get("dataset"),
        "source_identifier": sample_id,
        "audio_path": audio_info.get("audio_path"),
        "audio_duration_seconds": audio_info.get("audio_duration_seconds"),
        "sample_rate_hz": audio_info.get("sample_rate_hz"),
        "raw_audio_bytes": len(record["audio"]["bytes"])
        if isinstance(record.get("audio"), dict)
        and isinstance(record["audio"].get("bytes"), (bytes, bytearray, memoryview))
        else None,
    }
    return metadata


def split_records_stratified(records: Sequence[Dict[str, Any]], seed: int = SEED) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[bool, List[Dict[str, Any]]] = {True: [], False: []}
    seen_ids: set[str] = set()
    for record in records:
        label = record.get(LABEL_COLUMN)
        if not isinstance(label, bool):
            raise ValueError(f"Invalid {LABEL_COLUMN} value for sample {record.get(ID_COLUMN)!r}: {label!r}")
        sample_id = record.get("sample_id")
        if sample_id is None or str(sample_id).strip() == "":
            raise ValueError("Cannot split a record without a sample ID")
        sample_id = str(sample_id)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID found before splitting: {sample_id}")
        seen_ids.add(sample_id)
        grouped[label].append(record)

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for label, label_records in grouped.items():
        rng = random.Random(seed + (0 if label else 1))
        shuffled = list(label_records)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = int(n * 0.7)
        val_end = int(n * 0.15)
        # keep the split sizes consistent with the requested 70/15/15 ratio while preserving the full set.
        if train_end + val_end > n:
            val_end = n - train_end
        splits["train"].extend(shuffled[:train_end])
        splits["validation"].extend(shuffled[train_end:train_end + val_end])
        splits["test"].extend(shuffled[train_end + val_end:])
    validate_split_ids(splits)
    return splits


def validate_split_ids(split_map: Dict[str, Sequence[Dict[str, Any]]]) -> None:
    ids_by_split = {
        name: [str(record["sample_id"]) for record in records]
        for name, records in split_map.items()
    }
    for name, ids in ids_by_split.items():
        if any(not sample_id.strip() for sample_id in ids):
            raise ValueError(f"Missing sample ID found in {name} split")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs found in {name} split")
    split_names = list(ids_by_split)
    for index, name in enumerate(split_names):
        for other_name in split_names[index + 1:]:
            overlap = set(ids_by_split[name]) & set(ids_by_split[other_name])
            if overlap:
                raise ValueError(f"Sample IDs overlap between {name} and {other_name}: {sorted(overlap)[:5]}")


def validate_saved_records(
    records: Sequence[Dict[str, Any]],
    split_map: Dict[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Any]:
    invalid_labels = [
        record.get(LABEL_COLUMN)
        for record in records
        if not isinstance(record.get(LABEL_COLUMN), bool)
    ]
    if invalid_labels:
        raise RuntimeError(f"Final metadata contains invalid labels: {invalid_labels[:5]}")
    ids = [str(record["sample_id"]) for record in records]
    filenames = [str(record["audio_path"]) for record in records]
    duplicate_ids = len(ids) - len(set(ids))
    duplicate_filenames = len(filenames) - len(set(filenames))
    missing_files = 0
    corrupted_files = 0
    valid_audio_count = 0
    for record in records:
        path = ROOT / str(record["audio_path"])
        if not path.is_file():
            missing_files += 1
            continue
        try:
            inspect_audio_source(path)
            valid_audio_count += 1
        except (OSError, RuntimeError, TypeError, ValueError, sf.LibsndfileError):
            corrupted_files += 1

    validate_split_ids(split_map)
    split_ids = [str(record["sample_id"]) for rows in split_map.values() for record in rows]
    split_overlap_count = len(split_ids) - len(set(split_ids))
    expected_ids = set(ids)
    split_coverage_errors = len(expected_ids.symmetric_difference(set(split_ids)))
    label_counts = Counter(record.get(LABEL_COLUMN) for record in records)
    results = {
        "valid_audio_files": valid_audio_count,
        "metadata_records": len(records),
        "positive_labels": label_counts[True],
        "negative_labels": label_counts[False],
        "duplicate_ids": duplicate_ids,
        "duplicate_audio_filenames": duplicate_filenames,
        "missing_audio_files": missing_files,
        "corrupted_audio_files": corrupted_files,
        "split_overlap": split_overlap_count,
        "split_coverage_errors": split_coverage_errors,
        "split_counts": {name: len(rows) for name, rows in split_map.items()},
        "split_label_counts": {
            name: {
                "positive": sum(1 for record in rows if record.get(LABEL_COLUMN) is True),
                "negative": sum(1 for record in rows if record.get(LABEL_COLUMN) is False),
            }
            for name, rows in split_map.items()
        },
    }
    if duplicate_ids or duplicate_filenames or missing_files or corrupted_files or split_overlap_count or split_coverage_errors:
        raise RuntimeError(f"Final dataset validation failed: {results}")
    return results


def verify_reproducible_selection(
    first_selection: Sequence[Dict[str, Any]],
    second_selection: Sequence[Dict[str, Any]],
) -> None:
    first_ids = [str(record.get(ID_COLUMN)) for record in first_selection]
    second_ids = [str(record.get(ID_COLUMN)) for record in second_selection]
    if first_ids != second_ids:
        raise RuntimeError("Selection is not reproducible with the configured seed and scan limit")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_summary(records: Sequence[Dict[str, Any]], split_map: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary = {
        "total_selected": len(records),
        "positive": sum(1 for r in records if r[LABEL_COLUMN] is True),
        "negative": sum(1 for r in records if r[LABEL_COLUMN] is False),
        "duration_seconds": [float(r["audio_duration_seconds"]) for r in records],
        "sample_rates": [int(r["sample_rate_hz"]) for r in records],
        "languages": Counter(str(r.get("language")) for r in records),
        "synthetic": Counter(str(r.get("synthetic")) for r in records),
        "split_counts": {name: len(rows) for name, rows in split_map.items()},
        "split_label_counts": {
            name: {
                "positive": sum(1 for r in rows if r[LABEL_COLUMN] is True),
                "negative": sum(1 for r in rows if r[LABEL_COLUMN] is False),
            }
            for name, rows in split_map.items()
        },
    }
    return summary


def write_summary_document(summary: Dict[str, Any], dataset_name: str, inspected_total: int, methodology: str) -> None:
    duration_values = summary["duration_seconds"]
    sample_rate_values = summary["sample_rates"]
    language_counts = summary["languages"]
    synthetic_counts = summary["synthetic"]

    doc = [
        "# Dataset Summary",
        "",
        f"- Dataset source: `{dataset_name}`",
        f"- Dataset revision: `{DATASET_REVISION}`",
        f"- Samples inspected: {inspected_total}",
        f"- Samples selected: {summary['total_selected']}",
        f"- Positive labels: {summary['positive']}",
        f"- Negative labels: {summary['negative']}",
        "",
        "## Label distribution",
        "",
        f"- Positive share: {summary['positive'] / max(1, summary['total_selected']):.2%}",
        f"- Negative share: {summary['negative'] / max(1, summary['total_selected']):.2%}",
        "",
        "## Audio statistics",
        "",
        f"- Audio duration min: {min(duration_values):.2f}s",
        f"- Audio duration max: {max(duration_values):.2f}s",
        f"- Audio duration mean: {sum(duration_values) / len(duration_values):.2f}s",
        f"- Sample-rate min: {min(sample_rate_values)} Hz",
        f"- Sample-rate max: {max(sample_rate_values)} Hz",
        f"- Sample-rate mode: {Counter(sample_rate_values).most_common(1)[0][0]} Hz",
        "",
        "## Language and synthetic mix",
        "",
        "### Language distribution",
        "",
    ]
    for key, value in sorted(language_counts.items()):
        doc.append(f"- `{key}`: {value}")
    doc.extend(["", "### Real vs synthetic distribution", ""])
    for key, value in sorted(synthetic_counts.items()):
        doc.append(f"- `{key}`: {value}")
    doc.extend(["", "## Split counts", ""])
    for split_name, split_rows in summary["split_counts"].items():
        doc.append(f"- `{split_name}`: {split_rows}")
    doc.extend(["", "## Split class distribution", ""])
    for split_name, split_counts in summary["split_label_counts"].items():
        doc.append(f"- `{split_name}`: positive={split_counts['positive']}, negative={split_counts['negative']}")
    doc.extend(["", "## Sampling methodology", ""])
    doc.append(methodology)
    doc.extend(["", "## Known limitations", ""])
    doc.append("- Streaming mode was used to avoid a full-dataset download; the selected subset is therefore a bounded streaming sample rather than a complete scan of every row.")
    doc.append("- No speaker-level grouping identifier is available in the inspected schema; these splits cannot enforce speaker-level separation and must not be described as speaker-independent.")
    doc.append("- Audio duration and sample rate are computed after saving and reopening each audio file in its detected container format.")
    (DOCS_DIR / "dataset_summary.md").write_text("\n".join(doc) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a 5,000-sample end-of-turn dataset from the smart-turn train split.")
    parser.add_argument("--target-samples", type=int, default=5000, help="Total valid samples to keep before splitting.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed used for deterministic sampling.")
    parser.add_argument("--max-inspection", type=int, default=20000, help="Bounded inspection limit for streaming label distribution estimation.")
    parser.add_argument("--max-scan", type=int, default=20000, help="Maximum number of streamed rows to scan for class-aware selection.")
    args = parser.parse_args()
    if args.target_samples <= 0:
        parser.error("--target-samples must be greater than zero")
    if args.max_inspection < 0:
        parser.error("--max-inspection cannot be negative")
    if args.max_scan <= 0:
        parser.error("--max-scan must be greater than zero")

    dataset = load_streamed_dataset()
    _, inspected_total = inspect_label_distribution(
        dataset,
        max_examples=args.max_inspection,
    )
    logging.info(
        "Inspection pass found %s valid boolean labels (limit=%s); selection scan is separately limited to %s rows",
        inspected_total,
        args.max_inspection,
        args.max_scan,
    )

    dataset = load_streamed_dataset()
    selected_records = select_balanced_records(
        dataset,
        target_total=args.target_samples,
        seed=args.seed,
        max_scan_examples=args.max_scan,
        validate_audio_payloads=True,
    )
    selected_records = deduplicate_records(selected_records)

    if len(selected_records) < args.target_samples:
        logging.warning(
            "The bounded stream selected %s of %s requested records; one or both valid classes were underrepresented "
            "within the %s-row scan window.",
            len(selected_records),
            args.target_samples,
            args.max_scan,
        )

    processed_rows: List[Dict[str, Any]] = []
    for record in selected_records:
        raw_id = record.get(ID_COLUMN)
        sample_id = "" if raw_id is None else str(raw_id).strip()
        if not sample_id:
            logging.warning("Skipping record without id: %s", record)
            continue
        try:
            audio_info = save_audio_record(record, AUDIO_DIR)
            row = build_record_metadata(record, audio_info)
            processed_rows.append(row)
        except (OSError, RuntimeError, TypeError, ValueError, sf.LibsndfileError) as exc:
            logging.warning("Skipping unusable audio for sample %s: %s", sample_id, exc)

    if not processed_rows:
        raise RuntimeError(
            f"No valid audio records were saved (0 of {args.target_samples} requested); "
            "the bounded scan produced no usable records or selected audio failed validation."
        )
    if len(processed_rows) < args.target_samples:
        logging.warning(
            "Only %s of %s requested records were saved successfully after audio validation.",
            len(processed_rows),
            args.target_samples,
        )

    selected_df = pd.DataFrame(processed_rows)
    write_csv(selected_df, PROCESSED_DIR / "selected_samples.csv")
    write_csv(selected_df, METADATA_DIR / "selected_samples.csv")

    split_map = split_records_stratified(processed_rows, seed=args.seed)
    if len(processed_rows) != args.target_samples:
        logging.warning(
            "Preparation completed below target: saved %s of %s requested samples; class availability within the "
            "scan window and/or audio validation limited the final count.",
            len(processed_rows),
            args.target_samples,
        )
    split_dfs = {name: pd.DataFrame(rows) for name, rows in split_map.items()}
    for name, frame in split_dfs.items():
        write_csv(frame, PROCESSED_DIR / f"{name}.csv")
        write_csv(frame, METADATA_DIR / f"{name}.csv")

    persisted_metadata = pd.read_csv(PROCESSED_DIR / "selected_samples.csv")
    if len(persisted_metadata) != len(processed_rows):
        raise RuntimeError(
            f"Metadata CSV contains {len(persisted_metadata)} records; expected {len(processed_rows)}"
        )
    persisted_label_counts = persisted_metadata[LABEL_COLUMN].value_counts(dropna=False).to_dict()
    if persisted_metadata[LABEL_COLUMN].isna().any() or set(persisted_label_counts) - {True, False}:
        raise RuntimeError("Persisted metadata contains missing or invalid target labels")
    if not persisted_metadata["label"].equals(persisted_metadata[LABEL_COLUMN]):
        raise RuntimeError("Persisted label alias differs from the original endpoint_bool target")
    persisted_split_map = {
        name: pd.read_csv(PROCESSED_DIR / f"{name}.csv").to_dict("records")
        for name in split_map
    }
    for name, persisted_rows in persisted_split_map.items():
        if len(persisted_rows) != len(split_map[name]):
            raise RuntimeError(
                f"Persisted {name} CSV contains {len(persisted_rows)} records; expected {len(split_map[name])}"
            )
        if {str(row["sample_id"]) for row in persisted_rows} != {
            str(row["sample_id"]) for row in split_map[name]
        }:
            raise RuntimeError(f"Persisted {name} CSV sample IDs differ from the generated split")
        if any(not isinstance(row[LABEL_COLUMN], bool) for row in persisted_rows):
            raise RuntimeError(f"Persisted {name} CSV contains invalid target labels")
    validation_results = validate_saved_records(
        persisted_metadata.to_dict("records"),
        persisted_split_map,
    )
    validation_results["metadata_records"] = len(persisted_metadata)
    validation_results["positive_labels"] = int(persisted_label_counts.get(True, 0))
    validation_results["negative_labels"] = int(persisted_label_counts.get(False, 0))
    validation_results["split_counts"] = {
        name: len(rows) for name, rows in persisted_split_map.items()
    }
    validation_results["split_label_counts"] = {
        name: {
            "positive": sum(1 for row in rows if row[LABEL_COLUMN] is True),
            "negative": sum(1 for row in rows if row[LABEL_COLUMN] is False),
        }
        for name, rows in persisted_split_map.items()
    }

    validation = {
        "total_selected": len(processed_rows),
        "positive": sum(1 for r in processed_rows if r[LABEL_COLUMN] is True),
        "negative": sum(1 for r in processed_rows if r[LABEL_COLUMN] is False),
        "duration_seconds": [float(r["audio_duration_seconds"]) for r in processed_rows],
        "sample_rates": [int(r["sample_rate_hz"]) for r in processed_rows],
        "languages": Counter(str(r.get("language")) for r in processed_rows),
        "synthetic": Counter(str(r.get("synthetic")) for r in processed_rows),
        "split_counts": {name: len(rows) for name, rows in split_map.items()},
        "split_label_counts": {
            name: {
                "positive": sum(1 for r in rows if r[LABEL_COLUMN] is True),
                "negative": sum(1 for r in rows if r[LABEL_COLUMN] is False),
            }
            for name, rows in split_map.items()
        },
    }

    methodology = (
        "1. The dataset was opened in streaming mode and a bounded inspection pass was used to estimate label balance without downloading the full corpus.\n"
        f"2. A class-aware reservoir pass scanned up to {args.max_scan} stream rows with seed {args.seed}; it continued through the entire bound and selected a balanced positive/negative mix where available.\n"
        "3. Usable audio bytes or audio file paths were saved in their detected container format, then reopened and decoded before metadata was accepted.\n"
        "4. Records were split into train/validation/test sets using a deterministic, label-stratified 70/15/15 split; sample IDs were checked for duplicates and overlap."
    )
    write_summary_document(validation, DATASET_NAME, inspected_total=inspected_total, methodology=methodology)

    logging.info("Final validation results: %s", validation_results)
    logging.info(
        "Prepared %s valid audio files / metadata records and saved metadata CSVs under %s",
        validation_results["valid_audio_files"],
        PROCESSED_DIR,
    )
    logging.info("Train: %s | Validation: %s | Test: %s", len(split_map["train"]), len(split_map["validation"]), len(split_map["test"]))


if __name__ == "__main__":
    main()
