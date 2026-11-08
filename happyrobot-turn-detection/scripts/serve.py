#!/usr/bin/env python3
"""FastAPI service for the finalized audio-only EoT baseline."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, Optional

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, ConfigDict
from starlette.responses import JSONResponse
from threadpoolctl import threadpool_limits

from predict import DEFAULT_MODEL_DIR, load_model_bundle, predict_feature_values
from extract_features import FeatureExtractionError, extract_audio_features

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_DURATION_SECONDS = 0.1
MAX_DURATION_SECONDS = 30.0
MIN_SAMPLE_RATE_HZ = 8000
MAX_SAMPLE_RATE_HZ = 96000
ALLOWED_MEDIA_TYPES = {
    ".wav": {"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave", "application/octet-stream"},
    ".flac": {"audio/flac", "audio/x-flac", "application/octet-stream"},
}


class PredictionTiming(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    upload_ms: float
    feature_extraction_ms: float
    model_inference_ms: float
    inference_ms: float
    server_processing_ms: float


class PredictionResponse(BaseModel):
    endpoint_bool: bool
    endpoint_probability: float
    decision_threshold: float
    model: str
    feature_version: str
    audio_duration_seconds: float
    sample_rate_hz: int
    channels: int
    timing: PredictionTiming


def _load_bundle() -> Dict[str, Any]:
    model_dir = Path(os.environ.get("HAPPYROBOT_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    bundle = load_model_bundle(model_dir)
    if float(bundle["config"]["threshold"]) != 0.3691340835714273:
        raise ValueError("Loaded model threshold does not match the finalized baseline")
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI) -> Iterator[None]:
    with threadpool_limits(limits=1, user_api="blas"):
        app.state.model_bundle = _load_bundle()
        logger.info(
            "Loaded %s feature version %s with threshold %.10f",
            app.state.model_bundle["config"]["selected_model"],
            app.state.model_bundle["config"]["feature_version"],
            app.state.model_bundle["config"]["threshold"],
        )
        try:
            yield
        finally:
            app.state.model_bundle = None


app = FastAPI(
    title="HappyRobot EoT Baseline",
    version="1.0.0",
    description="Audio-only end-of-turn prediction using the finalized baseline artifact.",
    lifespan=lifespan,
)


@app.middleware("http")
async def reject_oversized_requests(request: Request, call_next: Any) -> Any:
    if request.url.path == "/predict":
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                body_size = int(content_length)
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
            if body_size > MAX_UPLOAD_BYTES + 64 * 1024:
                return JSONResponse(status_code=413, content={"detail": "Audio upload exceeds the 20 MiB limit"})
    return await call_next(request)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness(request: Request) -> Dict[str, str]:
    if getattr(request.app.state, "model_bundle", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready"}


def _open_and_inspect_upload(upload: UploadFile) -> tuple[sf.SoundFile, float, Optional[Path]]:
    started = time.perf_counter()
    filename = upload.filename or ""
    suffix = Path(filename).suffix.lower()
    accepted_types = ALLOWED_MEDIA_TYPES.get(suffix)
    if accepted_types is None:
        raise HTTPException(status_code=415, detail="Only WAV and FLAC audio files are supported")
    media_type = (upload.content_type or "").split(";", 1)[0].strip().lower()
    if media_type not in accepted_types:
        raise HTTPException(status_code=415, detail="File extension and audio content type are unsupported")

    temporary_path: Optional[Path] = None
    audio_file: Optional[sf.SoundFile] = None
    try:
        audio_source: Path | BinaryIO
        try:
            if not upload.file.seekable():
                raise OSError("Upload stream is not seekable")
            upload.file.seek(0, os.SEEK_END)
            total_bytes = upload.file.tell()
            upload.file.seek(0)
            audio_source = upload.file
        except (AttributeError, OSError, ValueError):
            with tempfile.NamedTemporaryFile(prefix="happyrobot-eot-", suffix=suffix, delete=False) as output:
                temporary_path = Path(output.name)
                total_bytes = 0
                while True:
                    chunk = upload.file.read(64 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Audio upload exceeds the 20 MiB limit")
                    output.write(chunk)
            audio_source = temporary_path

        if total_bytes == 0:
            raise HTTPException(status_code=422, detail="Audio upload is empty")
        if total_bytes > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio upload exceeds the 20 MiB limit")
        try:
            audio_file = sf.SoundFile(str(audio_source) if isinstance(audio_source, Path) else audio_source)
        except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exc:
            raise HTTPException(status_code=422, detail="Uploaded file is not readable WAV or FLAC audio") from exc

        expected_format = {".wav": "WAV", ".flac": "FLAC"}[suffix]
        if str(audio_file.format).upper() != expected_format:
            raise HTTPException(status_code=422, detail="Audio container does not match the file extension")
        if not MIN_SAMPLE_RATE_HZ <= int(audio_file.samplerate) <= MAX_SAMPLE_RATE_HZ:
            raise HTTPException(
                status_code=422,
                detail=f"Sample rate must be between {MIN_SAMPLE_RATE_HZ} and {MAX_SAMPLE_RATE_HZ} Hz",
            )
        duration = int(audio_file.frames) / float(audio_file.samplerate)
        if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=f"Audio duration must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g} seconds",
            )
        if not 1 <= int(audio_file.channels) <= 8:
            raise HTTPException(status_code=422, detail="Audio must have between 1 and 8 channels")
        return audio_file, (time.perf_counter() - started) * 1000.0, temporary_path
    except Exception:
        if audio_file is not None:
            audio_file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@app.post("/predict", response_model=PredictionResponse)
def predict(request: Request, audio: UploadFile = File(...)) -> PredictionResponse:
    request_started = time.perf_counter()
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    audio_file: Optional[sf.SoundFile] = None
    temporary_path: Optional[Path] = None
    try:
        audio_file, upload_ms, temporary_path = _open_and_inspect_upload(audio)
        feature_started = time.perf_counter()
        try:
            features = extract_audio_features(audio_file)
        except FeatureExtractionError as exc:
            raise HTTPException(status_code=422, detail="Audio could not be decoded for inference") from exc
        feature_ms = (time.perf_counter() - feature_started) * 1000.0

        model_started = time.perf_counter()
        prediction = predict_feature_values(features, bundle)
        model_ms = (time.perf_counter() - model_started) * 1000.0
        inference_ms = feature_ms + model_ms
        return PredictionResponse(
            **prediction,
            audio_duration_seconds=int(audio_file.frames) / float(audio_file.samplerate),
            sample_rate_hz=int(audio_file.samplerate),
            channels=int(audio_file.channels),
            timing=PredictionTiming(
                upload_ms=upload_ms,
                feature_extraction_ms=feature_ms,
                model_inference_ms=model_ms,
                inference_ms=inference_ms,
                server_processing_ms=(time.perf_counter() - request_started) * 1000.0,
            ),
        )
    finally:
        if audio_file is not None:
            audio_file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
