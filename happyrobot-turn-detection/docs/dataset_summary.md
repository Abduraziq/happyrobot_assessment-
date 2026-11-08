# Dataset Summary

- Dataset source: `pipecat-ai/smart-turn-data-v3-train`
- Samples inspected: 20000
- Samples selected: 5000
- Positive labels: 2500
- Negative labels: 2500

## Label distribution

- Positive share: 50.00%
- Negative share: 50.00%

## Audio statistics

- Audio duration min: 0.43s
- Audio duration max: 25.56s
- Audio duration mean: 7.25s
- Sample-rate min: 16000 Hz
- Sample-rate max: 16000 Hz
- Sample-rate mode: 16000 Hz

## Language and synthetic mix

### Language distribution

- `ara`: 173
- `ben`: 181
- `dan`: 142
- `deu`: 260
- `eng`: 539
- `fin`: 166
- `fra`: 255
- `hin`: 286
- `ind`: 193
- `ita`: 157
- `jpn`: 150
- `kor`: 176
- `mar`: 151
- `nld`: 263
- `nor`: 179
- `pol`: 205
- `por`: 290
- `rus`: 303
- `spa`: 277
- `tur`: 170
- `ukr`: 134
- `vie`: 179
- `zho`: 171

### Real vs synthetic distribution

- `False`: 108
- `True`: 4892

## Split counts

- `train`: 3500
- `validation`: 750
- `test`: 750

## Split class distribution

- `train`: positive=1750, negative=1750
- `validation`: positive=375, negative=375
- `test`: positive=375, negative=375

## Sampling methodology

1. The dataset was opened in streaming mode and a bounded inspection pass was used to estimate label balance without downloading the full corpus.
2. A class-aware reservoir pass scanned up to 20000 stream rows with seed 42; it continued through the entire bound and selected a balanced positive/negative mix where available.
3. Usable audio bytes or audio file paths were saved in their detected container format, then reopened and decoded before metadata was accepted.
4. Records were split into train/validation/test sets using a deterministic, label-stratified 70/15/15 split; sample IDs were checked for duplicates and overlap.

## Known limitations

- Streaming mode was used to avoid a full-dataset download; the selected subset is therefore a bounded streaming sample rather than a complete scan of every row.
- No speaker-level grouping identifier is available in the inspected schema; these splits cannot enforce speaker-level separation and must not be described as speaker-independent.
- Audio duration and sample rate are computed after saving and reopening each audio file in its detected container format.
