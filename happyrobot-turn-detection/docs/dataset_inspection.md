# Dataset Inspection Report

## Dataset source
- Dataset: `pipecat-ai/smart-turn-data-v3-train`
- Streamed examples inspected: 1000
- Streaming mode: enabled
- Audio decoding: disabled during inspection to avoid unnecessary decode overhead and Windows-specific audio decoder issues

## Schema summary

| Column | Type | Present in schema | Notes |
| --- | --- | --- | --- |
| `audio` | `Audio(sampling_rate=None, decode=False, num_channels=None, stream_index=None)` | yes | Audio payload column |
| `id` | `Value('string')` | yes |  |
| `language` | `Value('string')` | yes | Metadata column |
| `endpoint_bool` | `Value('bool')` | yes | Target column |
| `midfiller` | `Value('bool')` | yes | Metadata column |
| `endfiller` | `Value('bool')` | yes | Metadata column |
| `synthetic` | `Value('bool')` | yes | Metadata column |
| `spoken_text` | `Value('null')` | yes |  |
| `dataset` | `Value('string')` | yes | Metadata column |

- Actual audio column: `audio`
- Actual target column: `endpoint_bool`
- Target dtype: `Value('bool')`
- Endpoint validation: this boolean field is the end-of-turn target used by the dataset; it is stored as a boolean, not as a string or float.

## First 3 examples
```python
{"audio_bytes": 107512, "dataset": "chirp3_1", "endfiller": false, "endpoint_bool": true, "id": "00006fe3-5331-4db2-b30e-65affe00cfbe", "language": "zho", "midfiller": true, "synthetic": true}
{"audio_bytes": 92833, "dataset": "chirp3_2", "endfiller": true, "endpoint_bool": false, "id": "0000abee-a101-4a2f-bdb5-aecae1187830", "language": "vie", "midfiller": false, "synthetic": true}
{"audio_bytes": 101506, "dataset": "chirp3_2", "endfiller": false, "endpoint_bool": false, "id": "0001e47c-ff52-46ca-8368-380f2a052bed", "language": "fin", "midfiller": true, "synthetic": true}
```

## Class distribution (first 1,000 examples)

- Positive (`True`): 520
- Negative (`False`): 480
- Positive share: 52.00%
- Negative share: 48.00%

## Available metadata

- `language`: present
- `synthetic`: present
- `midfiller`: present
- `endfiller`: present
- `dataset`: present
- `id`: present
- `audio_duration` and `sample_rate`: not present as top-level schema fields; these are derived when the selected audio bytes are opened during dataset preparation.
- Speaker / recording / conversation identifiers: not present in the available schema

## Missing values and duplicate IDs

| Field | Missing count |
| --- | ---: |
| `endfiller` | 66 |
| `midfiller` | 66 |
| `spoken_text` | 1000 |
| `id` | 0 |

- Duplicate `id` records in the first 1000 examples: 0
- Duplicate identifiers: none

## Observations

1. The dataset is streamed as an iterable dataset with an `audio` column and a top-level boolean target column.
2. The audio payload is stored as raw bytes in the `audio` column and can be saved directly as a local FLAC or WAV file when selected.
3. The label column is a boolean target, so class balance can be measured without any label conversion.
4. Metadata such as `language`, `synthetic`, `midfiller`, `endfiller`, and `dataset` are present in the schema, but grouping identifiers such as speaker or conversation IDs are not exposed in this dataset snapshot.
5. Because of streaming and bounded inspection, the inspection result is a careful sample of the corpus rather than a full dataset scan.

## Recommended next step

Use `scripts/prepare_dataset.py` to perform a bounded-stream, class-aware collection of ~5,000 examples, then save audio and metadata under `data/processed/`.
