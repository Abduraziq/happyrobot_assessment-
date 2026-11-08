# Submission Checklist

## Assessment deliverables

- [ ] Record a concise solution video no longer than 10 minutes; target
  8-9 minutes using [`final_video_outline.md`](final_video_outline.md).
- [x] Short technical document:
  [`final_technical_report.md`](final_technical_report.md).
- [x] Documented implementation and reproduction instructions.
- [x] Monitoring/improvement proposal:
  [`monitoring_and_improvement.md`](monitoring_and_improvement.md).
- [x] Local throughput/latency findings and Dockerfile included.
- [x] Follow-up serving reports are separate from the original performance
  benchmark and include a representative 6.75-second input.
- [ ] Review the original `HappyRobot ML Case Study.pdf` discussion topics
  before the interview: model/data limits, audio versus text/multimodal/VAD,
  Whisper/Parketeer, and voice-agent integration.

## Important source and evidence files

- Dataset: `scripts/inspect_dataset.py`,
  `scripts/prepare_dataset.py`, `docs/dataset_inspection.md`,
  `docs/dataset_summary.md`, and `data/processed/{selected_samples,train,validation,test}.csv`.
- Features and baseline: `scripts/extract_features.py`,
  `scripts/train_baseline.py`, `scripts/evaluate_baseline.py`,
  `scripts/predict.py`.
- Serving: `scripts/serve.py`, `scripts/serve_client.py`,
  `scripts/benchmark_serving.py`, `tests/test_serving.py`.
- Tests: `tests/test_baseline.py`, `tests/test_serving.py`,
  `tests/test_benchmark_serving.py`.
- Existing model/config: `models/baseline/selected_model.joblib`,
  `models/baseline/model_config.json`.
- Evaluation evidence: `reports/baseline/validation_model_comparison.json`,
  `validation_selected_model_metrics.json`, `test_metrics.json`,
  `test_predictions.csv`, `validation_error_analysis.json`, and `latency.json`.
- English-only frozen-model evaluation:
  `reports/english_evaluation/run_seed42_909e604f/`.
- HTTP benchmark: `reports/serving/performance.json`.
- Deployment: `Dockerfile`, `.dockerignore`, `requirements-serving.txt`,
  and Docker verification in `docs/serving_and_performance.md`.

## Dataset preparation and reproducibility

The current prepared split CSVs contain 3,500 train, 750 validation, and 750
test rows; their sample IDs are disjoint, and all referenced files exist.
The audio directory contains 4,998 files not referenced by those CSVs. They
were not deleted or changed; investigate their provenance before packaging.

The Git package intentionally excludes audio payloads to avoid adding about
735 MiB of referenced FLAC files. A fresh checkout must run dataset
preparation before training or offline test evaluation; the English cohort
evaluator retrieves its approved cohort from the pinned dataset source. Local
audio remains present but ignored by `.gitignore`.

`requirements-training.txt` includes the pinned serving/model packages and
`datasets==5.1.0`. Dataset inspection and preparation scripts pin the dataset
source revision to `909e604f990651631a596f5cb03358b3ee8fbad0`. The source
revision of the already prepared split artifacts was not recorded at creation.

Run preparation or training only in a clean copy if preserving the current
artifacts:

```powershell
python -m pip install -r requirements-training.txt
python scripts\inspect_dataset.py --max-inspection 1000
python scripts\prepare_dataset.py --target-samples 5000 --seed 42 --max-inspection 20000 --max-scan 20000
python scripts\train_baseline.py --seed 42
python scripts\evaluate_baseline.py
```

Preparation streams bounded rows rather than downloading the full corpus,
but rewrites prepared audio/CSV files and `docs/dataset_summary.md`. Training
overwrites the selected model/config and validation reports; evaluation
overwrites held-out test reports. Do not run these commands against the
submission artifacts unless intentionally reproducing in a separate copy.

## Tests and verification

- [x] Existing unit suite: `python -m unittest discover -s tests -v`
  passed 17 tests in the latest audit. Its model-fitting tests use synthetic
  in-memory data and do not retrain or overwrite the saved baseline.
- [x] Read-only inference with the saved model reproduced all 750 persisted
  test predictions (maximum probability difference `8.33e-17`); recomputed
  metrics were precision 0.5121, recall 0.9040, F1 0.6538, 323 FP, and 36 FN.
- [x] Docker image `happyrobot-eot:latest` built; container startup, `/ready`,
  healthy status, prepared-FLAC inference, malformed-audio HTTP 422, and
  stop/start health were verified on 2026-11-07.
- [x] Docker verification container was left stopped. Do not rebuild or
  restart Docker solely for this checklist.
- [ ] Before delivery, confirm no output changes from local commands and
  retain the model, threshold, split CSVs, reports, and data unchanged.

## Final package and Git review

- [ ] Include the technical report and final video/recording link or file.
- [ ] Check the actual submission package for required source, model/config,
  reports, and any dataset artifacts the submission channel permits.
- [ ] Decide deliberately whether the 4,998 unreferenced audio files belong
  in the package; do not delete them without provenance/approval.
- [ ] Review `git status --short` and `git diff --check` before submission.
  At audit time, the parent repository reported the entire
  `happyrobot-turn-detection/` directory as untracked and had zero tracked
  project files. Do not stage the whole tree blindly: it includes prepared
  data/audio and model artifacts. Select the intended submission files
  explicitly; no files were staged, committed, or pushed in this audit.
