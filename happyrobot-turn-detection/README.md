# HappyRobot Turn Detection Baseline

This project prepares a streaming-selected dataset and develops an audio-only baseline for HappyRobot end-of-turn (EoT) detection.

## Dataset scripts

- `scripts/inspect_dataset.py` inspects the Hugging Face dataset schema at a pinned revision and writes `docs/dataset_inspection.md`.
- `scripts/prepare_dataset.py` performs bounded streaming inspection at the same pinned revision, selects ~5,000 valid examples, saves audio locally, and writes train/validation/test CSVs.

## Baseline workflow

```bash
cd happyrobot-turn-detection
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
python scripts/predict.py --audio path/to/audio.wav
```

Training reads only the existing train and validation CSVs. The selected model and decision threshold are finalized on validation before `evaluate_baseline.py` opens the held-out test split. Do not use test results to tune the model or threshold.

The first baseline uses only deterministic acoustic features. Metadata such as language, filler flags, and synthetic/source markers is reserved for validation error analysis and is not passed to a model.

Both data scripts pin `pipecat-ai/smart-turn-data-v3-train` to revision
`909e604f990651631a596f5cb03358b3ee8fbad0`. The existing prepared split
artifacts predate recording that revision and are not rewritten by installing
dependencies.

## Local serving

The finalized model can be served without retraining:

```bash
python -m pip install -r requirements-serving.txt
python -m uvicorn serve:app --app-dir scripts --host 127.0.0.1 --port 8000
python scripts/serve_client.py --audio path/to/audio.wav
```

See [`docs/serving_and_performance.md`](docs/serving_and_performance.md) for API details, Docker usage, and the concurrency benchmark.

## Output directories

- `data/processed/audio/` and `data/processed/*.csv` are the prepared dataset inputs and are not modified by the baseline scripts.
- `models/baseline/` stores the finalized model and feature/threshold configuration.
- `reports/baseline/` stores validation comparisons, validation error analysis, latency, and final test reports.
- `reports/english_evaluation/` stores a frozen-model English-only cohort evaluation.
- `reports/serving/` stores HTTP benchmark output.
- `docs/baseline_model.md` documents the baseline, evaluation, and limitations.

## Notes

- Dataset preparation uses Hugging Face streaming with explicit scan limits; baseline development uses only the prepared local files.
- This initial audio-only baseline does not use conversational context and is not a production-ready endpointing system.
- Audio payloads are intentionally excluded from Git to keep the submission
  manageable. Re-run `scripts/prepare_dataset.py` to materialize the base
  audio in a fresh checkout; the English cohort evaluator retrieves its
  approved cohort from the pinned source revision.
