# Serving and Performance

## Architecture

`scripts/serve.py` provides a synchronous FastAPI inference endpoint. The app
loads `models/baseline/selected_model.joblib` and `model_config.json` once in
its lifespan and refuses startup if the finalized feature configuration or
threshold differs from the baseline. Each request uses the existing
`extract_audio_features()` implementation and `predict_feature_values()`;
offline and HTTP inference therefore share the same features and threshold
(`0.3691340835714273`).

The service accepts WAV and FLAC only, with a 20 MiB file limit, duration from
0.1 through 30 seconds, sample rate from 8 through 96 kHz, and 1-8 channels.
Container content is checked against its filename extension and audio must be
readable. Valid stereo/non-16-kHz audio follows the existing mono mix and
resampling logic. Rejections return an HTTP error with a short `detail`.

The prediction response timing distinguishes upload inspection/opening,
feature extraction, model-only inference, total inference, and server
processing. `upload_ms` begins after FastAPI has parsed the multipart body, so
it does not include client transfer or FastAPI's multipart parsing/spooling.
The benchmark separately measures client-side request latency, which includes
the complete HTTP request and response. The model is CPU-only and inference
runs in FastAPI's synchronous worker thread pool.

## Endpoints

- `GET /health`: process health.
- `GET /ready`: readiness; returns 503 when the model bundle is not loaded.
- `POST /predict`: multipart form with an `audio` WAV or FLAC file.

Example:

```bash
curl.exe -F "audio=@sample.wav" http://127.0.0.1:8000/predict
```

Example response:

```json
{
  "endpoint_bool": false,
  "endpoint_probability": 0.21,
  "decision_threshold": 0.3691340835714273,
  "model": "logistic_regression",
  "feature_version": "acoustic-v1",
  "audio_duration_seconds": 4.2,
  "sample_rate_hz": 16000,
  "channels": 1,
  "timing": {
    "upload_ms": 3.1,
    "feature_extraction_ms": 18.4,
    "model_inference_ms": 1.2,
    "inference_ms": 19.6,
    "server_processing_ms": 23.0
  }
}
```

Values above illustrate the response schema; request timing and prediction
values vary by audio and machine.

## Run locally

From `happyrobot-turn-detection/`:

```bash
python -m pip install -r requirements-serving.txt
python -m uvicorn serve:app --app-dir scripts --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
python scripts/serve_client.py --audio path/to/audio.wav
```

The standalone client prints the model response and measured total request
latency. It reports HTTP errors, connection errors, and timeouts explicitly.

## Docker

Build and run from `happyrobot-turn-detection/`:

```bash
docker build -t happyrobot-eot:latest .
docker run -d --name happyrobot-eot-verification -p 8000:8000 happyrobot-eot:latest
```

The build context ignores dataset audio, CSVs, reports, and tests. The image
copies only the serving modules, pinned serving dependencies, and finalized
model/configuration. It does not download the Hugging Face dataset. The
container has a `/ready` Docker health check.

### Docker verification (2026-11-07)

| Check | Result |
|---|---|
| Docker Server | **VERIFIED** — Docker Desktop 4.39.0 (184744), Linux Engine 28.0.1, `desktop-linux` context |
| WSL 2 backend | **VERIFIED** — configured with `WslEngineEnabled: true`; `docker-desktop` was registered and running |
| Image build | **VERIFIED** — `happyrobot-eot:latest` built successfully (467,206,428 bytes) |
| Container startup | **VERIFIED** — started with port 8000 mapped; Uvicorn completed application startup |
| Readiness and health check | **VERIFIED** — `GET /ready` returned 200; Docker health status became `healthy` |
| Real inference | **VERIFIED** — prepared 6.16-second FLAC returned a prediction from `logistic_regression` (`acoustic-v1`) at the saved threshold |
| Invalid audio | **VERIFIED** — malformed WAV upload returned HTTP 422 with an unreadable-audio detail |
| Stop/restart | **VERIFIED** — container restarted, returned ready, and became healthy again |
| Dataset download or retraining | **NOT PERFORMED** — the image uses the existing selected model/config and serving code only |

The inference used the existing file
`data/processed/audio/00006fe3-5331-4db2-b30e-65affe00cfbe-ff406d327d5a668cd05993fca6d99fd20a5a1d65c476bed5edd9f02b7a8301c9.flac`.
The container was stopped after verification and left in the stopped state.

At inspection time, the Docker Desktop UI/backend processes and Linux engine
were running even though the Windows `com.docker.service` was stopped. Docker
CLI requests to the Linux server succeeded, the WSL 2 backend setting was
enabled, and the Docker-managed WSL distribution was running. The original
missing named-pipe failure could not be reproduced, so its historical trigger
is **NOT CONFIRMED**. Current startup logs included WSL bootstrap messages,
including a device-offline/JBD2 I/O warning for `sde`; the daemon nevertheless
started and the image/container checks passed. No Docker settings, WSL data,
images, or volumes were reset or deleted.

Reproduce from `happyrobot-turn-detection/` in PowerShell:

```powershell
$container = "happyrobot-eot-verification-$([guid]::NewGuid().ToString('N'))"
docker build -t happyrobot-eot:latest .
docker run -d --name $container -p 8000:8000 happyrobot-eot:latest
Start-Sleep -Seconds 10
curl.exe --fail http://127.0.0.1:8000/ready
python scripts\serve_client.py --audio "data\processed\audio\00006fe3-5331-4db2-b30e-65affe00cfbe-ff406d327d5a668cd05993fca6d99fd20a5a1d65c476bed5edd9f02b7a8301c9.flac"
docker stop $container
docker start $container
Start-Sleep -Seconds 10
curl.exe --fail http://127.0.0.1:8000/ready
$deadline = (Get-Date).AddSeconds(90)
do {
    $health = docker inspect --format '{{.State.Health.Status}}' $container
    if ($health -eq 'healthy') { break }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)
if ($health -ne 'healthy') { throw "Container health check failed: $health" }
docker inspect --format '{{.State.Health.Status}}' $container
$invalidAudio = Join-Path $env:TEMP ("happyrobot-invalid-" + [guid]::NewGuid().ToString('N') + ".wav")
[IO.File]::WriteAllText($invalidAudio, 'not a valid WAV file')
curl.exe -sS -o NUL -w "HTTP %{http_code}`n" -F "audio=@${invalidAudio};type=audio/wav" http://127.0.0.1:8000/predict
Remove-Item -LiteralPath $invalidAudio
docker stop $container
```

The invalid-audio request should return HTTP 422. The container name is unique
so the sequence can be repeated without removing the prior stopped test
container.

## HTTP benchmark

The reproducible benchmark script sends the same caller-provided WAV/FLAC
sample sequentially and at increasing concurrency levels. It records
successful/failed requests, requests per second, client-observed latency
(mean/median/P95/P99), and server feature/model-only timing distributions:

```bash
python scripts/benchmark_serving.py --audio path/to/sample.wav --requests 100 --concurrency 1,2,4,8 --server-workers 1
```

`--requests` is the number of requests at each concurrency level. The script
reuses an HTTP connection pool and the same audio payload. Results are saved
to a uniquely timestamped file under `reports/serving/benchmark_runs/` by
default; it refuses to overwrite an existing report. Use `--server-workers`
to record the server's Uvicorn worker count. To choose a separate explicit
path, pass `--output reports/serving/benchmark_runs/my_run.json`. The original
`reports/serving/performance.json` is preserved. Generate an independent
synthetic audio file if you do not want to use any prepared dataset file:

```bash
python -c "import numpy as np, soundfile as sf; x=np.arange(8000)/16000; sf.write('sample.wav', 0.1*np.sin(2*np.pi*440*x), 16000)"
```

### Original baseline results

The original local benchmark used a synthetic, mono 16 kHz WAV file of 0.5 seconds and
16,044 bytes. Each run sent 100 requests; all 400 requests succeeded. Local
connection reuse was enabled. Client latency includes request/response
handling; server-only timing is instrumented separately. This short synthetic
clip is not representative of the dataset's longer average audio, so these
throughput figures should not be extrapolated to typical or production traffic.

| Mode / concurrency | Requests | Success / failed | Requests/s | HTTP mean | HTTP median | HTTP P95 | HTTP P99 | Model-only P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sequential / 1 | 100 | 100 / 0 | 51.45 | 19.41 ms | 18.69 ms | 23.82 ms | 33.87 ms | 3.08 ms |
| Concurrent / 2 | 100 | 100 / 0 | 60.92 | 32.41 ms | 31.73 ms | 39.45 ms | 61.69 ms | 2.61 ms |
| Concurrent / 4 | 100 | 100 / 0 | 76.55 | 51.66 ms | 54.42 ms | 66.60 ms | 71.94 ms | 3.39 ms |
| Concurrent / 8 | 100 | 100 / 0 | 95.28 | 81.80 ms | 79.63 ms | 111.52 ms | 118.35 ms | 3.11 ms |

Observed server-side feature extraction P95 increased from 3.34 ms sequential
to 52.54 ms at concurrency 8; model-only prediction P95 stayed around 2.6-3.4
ms. This indicates CPU contention is primarily in decoding/feature extraction,
not the lightweight classifier.

The environment was Windows 10, Python 3.10.11, 8 logical CPUs, Intel Family 6
Model 140 processor; FastAPI 0.115.0, Uvicorn 0.30.6, and httpx 0.27.2. Results
are in `reports/serving/performance.json`.

The assignment's ideal target is below 100 ms per request. Sequential and
concurrency 2/4 P95 were below 100 ms for this short clip; concurrency 8 P95
exceeded 100 ms. These measurements do not establish that the target is met
for typical audio, cold caches, other hardware, or production loads.

### Follow-up optimization results

The original benchmark above remains unchanged. A later run used eight
Uvicorn workers and measured both a 0.5-second WAV and a 6.75-second FLAC,
with 100 requests at each client concurrency. All requests succeeded:

| Audio | Concurrency | Throughput | Mean | Median | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|
| 0.5-second WAV | 1 | 41.10 req/s | 24.29 ms | 23.50 ms | 31.29 ms | 34.72 ms |
| 0.5-second WAV | 2 | 50.22 req/s | 39.59 ms | 29.25 ms | 104.82 ms | 281.05 ms |
| 0.5-second WAV | 4 | 100.58 req/s | 39.05 ms | 38.34 ms | 46.27 ms | 83.61 ms |
| 0.5-second WAV | 8 | 122.46 req/s | 62.04 ms | 57.57 ms | 89.19 ms | 122.23 ms |
| 6.75-second FLAC | 1 | 22.29 req/s | 44.81 ms | 43.82 ms | 58.19 ms | 64.73 ms |
| 6.75-second FLAC | 2 | 37.49 req/s | 52.91 ms | 52.14 ms | 66.76 ms | 72.86 ms |
| 6.75-second FLAC | 4 | 54.06 req/s | 72.71 ms | 70.39 ms | 94.72 ms | 119.21 ms |
| 6.75-second FLAC | 8 | 69.60 req/s | 111.38 ms | 107.40 ms | 146.41 ms | 156.66 ms |

The short clip meets 110 ms P95 at concurrency 8; the longer clip does not.
The latter's measured feature-extraction P95 was 96.63 ms, with model-only
inference P95 16.83 ms. The full reports are
`phase2_followup_20261107/http_short_8workers.json` and
`phase2_followup_20261107/http_average_flac_8workers.json`. Those existing
files predate the `--server-workers` field, so the worker configuration is
documented here; new runs record it explicitly.

The follow-up correctness report confirms exact feature-vector and prediction
agreement for three representative inputs. The tests additionally cover WAV,
FLAC, malformed input, non-seekable fallback, concurrent requests, and
non-overwriting report behavior. All 17 tests pass.

## Tests

From the project directory:

```bash
python -m unittest discover -s tests -v
```

The suite covers baseline compatibility, cached-constant immutability,
model startup/readiness, health, valid synthetic audio, unsupported/unreadable/
missing files, duration and sample-rate validation, saved-threshold application,
response fields, upload-spool reuse/fallback, FLAC, concurrent requests, and
non-overwriting benchmark report writes. The current suite has 17 tests.

## Limitations

This reuses the selected Logistic Regression artifact and does not retrain or
read the held-out test split. Its test F1 was 0.6538 with 323 false positives
and 36 false negatives. A false positive may cut off a user who is still
speaking; the high false-positive count is a serious limitation. The model is
audio-only and lacks conversational context, speaker-turn context, and
temporal sequence modeling. These artifacts and local load tests do not make
the system production-ready; deployment reliability, real traffic, security
controls, observability, cold-start behavior, and longer typical audio still
need evaluation.
