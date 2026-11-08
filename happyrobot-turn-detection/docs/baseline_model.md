# Audio-Only EoT Baseline

## Dataset and protocol

This experiment uses the existing 5,000-sample prepared dataset from
`pipecat-ai/smart-turn-data-v3-train`: 3,500 training, 750 validation, and 750
test records, with balanced positive and negative labels in each split. All
prepared audio is 16 kHz, mono FLAC. The baseline scripts read the existing
CSVs and audio; they do not regenerate or modify them.

The models are fit on training data. Candidate models and the decision
threshold are compared using validation data only. The selected model and
threshold were finalized before the single test evaluation. The test split is
not used for feature selection, tuning, or threshold selection.

There are no speaker-level grouping identifiers in the inspected dataset
schema, so these splits are not speaker-independent.

## Features and models

The first baseline is audio-only and uses a fixed feature schema: duration,
RMS and short-time-energy summaries, low-energy fraction, zero-crossing rate,
spectral centroid/bandwidth/rolloff/flux, and mean/standard deviation summaries
for 13 MFCCs. Audio is mixed to mono and resampled to 16 kHz as needed.
Short-time features use 25 ms frames, a 10 ms hop, and a 512-point FFT.
Non-finite samples are replaced with zero; empty or undecodable audio fails
feature extraction rather than producing a prediction.

The compared candidates are a most-frequent `DummyClassifier`, Logistic
Regression, and a 300-tree Random Forest. Both learned models use median
imputation; Logistic Regression and Random Forest also use a `StandardScaler`
in their scikit-learn pipelines. The fixed random seed is 42.

No sample IDs, labels, language, filler flags, synthetic markers, or source
metadata are model inputs. Those metadata fields are used only for validation
error analysis.

## Validation results and threshold

The decision threshold was selected on validation by maximizing F1, breaking
ties by precision and then by preferring the higher threshold. Logistic
Regression was selected: its validation F1 (0.6705) slightly exceeded Random
Forest (0.6673), while the DummyClassifier is a majority-class sanity check.

| Model | Threshold | Accuracy | Precision | Recall | F1 | FP | FN | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dummy (most frequent) | n/a | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 375 | 0.5000 | 0.5000 |
| Logistic Regression | 0.3691 | 0.5427 | 0.5240 | 0.9307 | 0.6705 | 317 | 26 | 0.6084 | 0.6116 |
| Random Forest | 0.2454 | 0.5013 | 0.5007 | 1.0000 | 0.6673 | 374 | 0 | 0.5971 | 0.5707 |

Validation confusion matrix for Logistic Regression, rows actual
`[negative, positive]` and columns predicted `[negative, positive]`:
`[[58, 317], [26, 349]]`.

The selected threshold favors recall, but produces many false positives. A
false positive predicts that the user has finished when they are actually
continuing, which can cut off a turn prematurely. A false negative predicts
that the user is continuing after they have finished, which can add response
delay or require extra silence. Threshold choice therefore reflects a product
trade-off; optimizing accuracy alone would not represent these costs.

## Final test results

The finalized Logistic Regression and validation-selected threshold were
evaluated once on the held-out test split:

| Accuracy | Precision | Recall | F1 | FP | FN | ROC-AUC | Average precision |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5213 | 0.5121 | 0.9040 | 0.6538 | 323 | 36 | 0.5750 | 0.5751 |

Test confusion matrix (actual rows `[negative, positive]`, predicted columns
`[negative, positive]`): `[[52, 323], [36, 339]]`.

The saved `precision_recall_auc` field is computed with scikit-learn's
`average_precision_score`; it is average precision, not trapezoidal area under
the precision-recall curve.

The test metrics are close to validation performance but remain weak for
reliable turn-taking: the high false-positive count is a material limitation.
The test results were not used to make further model or threshold changes.

## Validation error analysis

There were 343 validation errors (317 false positives and 26 false negatives).
The saved report groups these errors by duration, low-energy fraction, sample
rate, language, synthetic marker, and filler metadata. For example, 199 of 233
examples with `endfiller=True` were errors, all false positives; the error rate
was 26.4% for the 478 examples with `endfiller=False`. These are observed
associations in this validation subset, not evidence that the metadata caused
the errors. Every audio file in these splits has a 16 kHz sample rate, so this
dataset provides no sample-rate comparison. The present analysis measures
low-energy fraction but does not establish that pauses caused individual
errors.

Representative and complete validation error rows, including IDs, actual
labels, predictions, probabilities, and metadata for later review, are saved
to `reports/baseline/validation_errors.csv`.

## Inference latency

Latency was measured on the development machine using all 750 validation
examples, with a warm file cache. End-to-end includes audio loading/decoding,
feature extraction, and one-row prediction; model-only prediction excludes
audio I/O and feature extraction. Model startup/load time is excluded.

| Measurement | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|
| Model-only prediction | 1.21 ms | 1.03 ms | 2.18 ms | 2.89 ms |
| Audio load, features, and prediction | 17.17 ms | 16.23 ms | 28.45 ms | 35.77 ms |

Environment: Windows 10, Python 3.10.11, 8 logical CPUs, Intel Family 6 Model
140 processor; NumPy 2.2.3, pandas 2.2.3, scikit-learn 1.6.1, SciPy 1.15.2,
and soundfile 0.14.0. The measured warm-cache P95 is below 100 ms in this
benchmark, but this does not establish production latency or reliability;
cold I/O, deployment hardware, concurrency, and model startup were not tested.

## Artifacts and reproduction

- `models/baseline/selected_model.joblib`: selected scikit-learn pipeline.
- `models/baseline/model_config.json`: feature schema/version, seed, model,
  threshold, and validation metrics.
- `reports/baseline/validation_model_comparison.json`,
  `validation_selected_model_metrics.json`, `validation_error_analysis.json`,
  `validation_errors.csv`, and `latency.json`: training-stage reports.
- `reports/baseline/test_metrics.json` and `test_predictions.csv`: final
  held-out test evaluation.

From `happyrobot-turn-detection/`, reproduce the workflow with:

```bash
python -m unittest discover -s tests -v
python scripts/train_baseline.py --seed 42
python scripts/evaluate_baseline.py
python scripts/predict.py --audio path/to/audio.wav
```

Training reads train and validation only; the evaluator reads test only after
the finalized model configuration exists. Re-running training replaces the
baseline model and validation reports, not the prepared dataset.

## Limitations

This is a small, interpretable first baseline, not a production-ready EoT
system. It uses fixed acoustic summaries and does not model conversational
context, speaker turns, or temporal sequence behavior. EoT decisions can
benefit from conversational context, which this audio-only baseline omits.
The dataset is bounded and lacks speaker grouping identifiers; the reported
results do not prove speaker-independent generalization.
