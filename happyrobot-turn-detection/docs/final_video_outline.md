# Final Video Outline (8-9 minutes)

Target length: approximately 8 minutes 45 seconds, below the assessment's
10-minute maximum. Keep the walkthrough focused on decisions, evidence, and
limitations rather than reading code line by line.

| Time | Talking points | Show |
|---|---|---|
| 0:00-0:40 | Define EoT and the assessment goals: English support, ideal sub-100 ms latency, evaluation, API, Docker, and monitoring plan. State that this is a baseline, not a production-ready quality claim. | `docs/final_technical_report.md`, executive summary; PDF requirements if useful |
| 0:40-1:30 | Explain the dataset choice and bounded streaming preparation. Call out 5,000 balanced samples, 97.84% marked synthetic, multilingual but uneven coverage, and no speaker IDs. | `docs/dataset_summary.md`; `scripts/prepare_dataset.py` |
| 1:30-2:20 | Explain the deterministic 70/15/15 stratified split, seed 42, unique/disjoint IDs, and lack of speaker-group isolation or cross-validation. | `data/processed/{train,validation,test}.csv` headers/counts; `scripts/prepare_dataset.py` |
| 2:20-3:25 | Walk through the 45 acoustic features and candidate baselines. Explain why Logistic Regression was selected and why the 0.369134 threshold came from validation F1, not the test set. | `scripts/extract_features.py`; `scripts/train_baseline.py`; `models/baseline/model_config.json` |
| 3:25-4:20 | Show held-out test metrics and interpret the confusion matrix. Emphasize precision 0.512, 323 false positives, interruption risk, and the small English subgroup (82 examples, F1 0.631). | `reports/baseline/test_metrics.json`; `reports/baseline/test_predictions.csv`; report error analysis |
| 4:20-5:15 | Explain inference consistency: one-time model load, same feature/prediction functions, validation, readiness, and timing fields. Contrast EoT with VAD; VAD locates speech, while EoT estimates conversational turn completion. | `scripts/serve.py`; `scripts/predict.py`; `scripts/serve_client.py` |
| 5:15-6:20 | Present the HTTP stress-test table. Separate client-observed HTTP P95 from model-only prediction P95. Note the 0.5-second synthetic clip, warm connection pool, one machine, and 111.52 ms P95 at concurrency 8. | `reports/serving/performance.json`; HTTP benchmark table in `docs/serving_and_performance.md` |
| 6:20-7:00 | Summarize Docker evidence: build, port mapping, `/ready`, healthy status, prepared-FLAC inference, invalid audio 422, and stop/start. Do not rebuild during the recording unless a live demo is explicitly desired. | Docker verification table in `docs/serving_and_performance.md`; optionally show saved terminal evidence |
| 7:00-8:05 | Explain monitoring as a proposal, not a deployed stack: service/error/latency/resource signals, delayed ground truth, FP/FN cohort quality, drift, privacy, alerts, human review, and shadow/canary rollout. | `docs/monitoring_and_improvement.md` |
| 8:05-8:45 | Discuss future model choices: incremental ASR (Whisper/Parketeer), text/audio fusion, conversational context, integration behind a turn manager, and production feedback. Close with the limits: synthetic-heavy data, small English cohort, false positives, and no speaker grouping. | `docs/final_technical_report.md`, discussion and limitations sections |

## Short discussion answers

- **Why this baseline?** It is a small, explainable, low-cost starting point
  that can be trained and served within the time budget. Its results are
  transparent, including the high false-positive rate.
- **Why audio-only first?** It uses the input already present in a voice
  pipeline and avoids adding ASR latency and transcription errors. It omits
  lexical and conversational cues, so it is not claimed to be the best final
  architecture.
- **How is EoT different from VAD?** VAD detects speech activity; EoT decides
  whether the user intends to yield the turn. Pauses and fillers can occur
  without turn completion.
- **How could transcription help?** Incremental Whisper/Parketeer text could
  contribute lexical completion, discourse markers, and partial-transcript
  confidence. This remains a proposal and must fit latency/privacy limits.
- **How would it integrate?** Feed audio/VAD and optional ASR signals to a
  turn manager that combines endpoint confidence with conversation state and
  preserves continuation/barge-in behavior.
- **How would it improve?** Collect privacy-approved, human-reviewed,
  representative delayed labels; evaluate by speaker/time/language where
  possible; compare offline; then shadow/canary before rollout.
