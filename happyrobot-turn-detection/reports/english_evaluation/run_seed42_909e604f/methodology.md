# English-only held-out evaluation

## Cohort selection

- Source: `pipecat-ai/smart-turn-data-v3-train` at immutable revision `909e604f990651631a596f5cb03358b3ee8fbad0`.
- A metadata-only streaming scan inspected the first 20,000 rows; it did not download the full corpus.
- Existing train, validation, and test IDs were read from both `data/metadata/` and `data/processed/`; the two copies were required to match.
- Existing IDs were excluded before candidates were pooled or sampled. There were 866 eligible additional positive and 834 eligible additional negative English examples in the bounded scan.
- English was identified by `language == eng` (also accepting equivalent `en`/`english` spellings).
- Unique eligible IDs were sorted within each label and sampled with `random.Random(42)`: 400 positive and 400 negative.
- The selected-ID SHA-256 is `5683af05ba24d8ade049e631662e240dce072865a64ac2043ed0c047820727f4`. Verified overlap counts: train 0, validation 0, test 0.
- Synthetic metadata: `true` flagged 682 and `false` flagged 118 (85.25% flagged true).
- Source audio was extracted only for the approved cohort and stored under this run's `audio/` directory. No source train/validation/test CSV or audio was modified.

## Frozen model and evaluation

The existing finalized `logistic_regression` artifact and saved threshold `0.3691340835714273` were loaded unchanged. No training, model selection, or threshold tuning was performed. The evaluation uses the existing shared feature extraction and inference functions. Average precision is reported for this balanced 50/50 cohort and should not be interpreted as production-prevalence precision.

The 95% intervals are percentile intervals from 10,000 stratified bootstrap replicates, resampling with replacement separately within positive and negative examples. Bootstrap seed: `20261107`.

## Results

| Metric | Estimate | 95% interval |
|---|---:|---:|
| Precision | 0.5045 | 0.4888–0.5201 |
| Recall | 0.8400 | 0.8025–0.8750 |
| F1 | 0.6304 | 0.6089–0.6510 |
| Accuracy | 0.5075 | 0.4813–0.5337 |
| Average Precision | 0.5654 | 0.5293–0.6084 |


## Limitations

This is a separate English-only evaluation from the same source dataset, not an independent external-domain test. The cohort is balanced by design, so prevalence-sensitive metrics (especially precision and average precision) do not estimate production prevalence. The source is predominantly synthetic; language labels and sample IDs do not provide speaker-level independence. The results are evidence for this fixed cohort only and do not establish robust English generalization.
