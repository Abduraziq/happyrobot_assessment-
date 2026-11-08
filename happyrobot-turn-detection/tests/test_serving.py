from __future__ import annotations

import io
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from threadpoolctl import threadpool_info

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import serve
from extract_features import FEATURE_NAMES
from predict import predict_feature_values


class NonSeekableFile:
    def __init__(self, content: bytes):
        self.content = io.BytesIO(content)

    def read(self, size: int = -1) -> bytes:
        return self.content.read(size)

    def seekable(self) -> bool:
        return False


def wav_bytes(sample_rate: int = 16000, duration: float = 0.5) -> bytes:
    signal = np.sin(
        np.linspace(0.0, duration * 440.0 * 2.0 * np.pi, int(sample_rate * duration), endpoint=False)
    ).astype(np.float32) * 0.1
    buffer = io.BytesIO()
    sf.write(buffer, signal, sample_rate, format="WAV")
    return buffer.getvalue()


class FixedProbabilityModel:
    classes_ = [False, True]

    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, features: Any) -> np.ndarray:
        return np.asarray([[1.0 - self.probability, self.probability]])


class ServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_context = TestClient(serve.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_model_loads_once_and_readiness_is_reported(self) -> None:
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/ready").json(), {"status": "ready"})
        blas_pools = [pool for pool in threadpool_info() if pool["user_api"] == "blas"]
        self.assertTrue(all(pool["num_threads"] == 1 for pool in blas_pools))
        bundle = self.client.app.state.model_bundle
        self.assertEqual(bundle["config"]["selected_model"], "logistic_regression")
        self.assertAlmostEqual(bundle["config"]["threshold"], 0.3691340835714273)

    def test_valid_audio_returns_prediction_and_timing_schema(self) -> None:
        response = self.client.post(
            "/predict",
            files={"audio": ("sample.wav", wav_bytes(), "audio/wav")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(
            set(result),
            {
                "endpoint_bool",
                "endpoint_probability",
                "decision_threshold",
                "model",
                "feature_version",
                "audio_duration_seconds",
                "sample_rate_hz",
                "channels",
                "timing",
            },
        )
        self.assertIsInstance(result["endpoint_bool"], bool)
        self.assertGreaterEqual(result["endpoint_probability"], 0.0)
        self.assertLessEqual(result["endpoint_probability"], 1.0)
        self.assertEqual(result["decision_threshold"], 0.3691340835714273)
        self.assertEqual(result["sample_rate_hz"], 16000)
        self.assertEqual(result["channels"], 1)
        self.assertEqual(
            set(result["timing"]),
            {
                "upload_ms",
                "feature_extraction_ms",
                "model_inference_ms",
                "inference_ms",
                "server_processing_ms",
            },
        )
        self.assertGreaterEqual(result["timing"]["server_processing_ms"], result["timing"]["inference_ms"])

    def test_seekable_upload_spool_is_reused_without_a_second_temporary_file(self) -> None:
        payload = wav_bytes()
        upload = serve.UploadFile(
            file=io.BytesIO(payload),
            filename="sample.wav",
            headers=Headers({"content-type": "audio/wav"}),
        )
        with patch("serve.tempfile.NamedTemporaryFile") as temporary_file:
            audio_file, _, temporary_path = serve._open_and_inspect_upload(upload)

        try:
            self.assertIsInstance(audio_file, sf.SoundFile)
            self.assertIsNone(temporary_path)
            self.assertEqual(audio_file.format, "WAV")
            temporary_file.assert_not_called()
        finally:
            audio_file.close()
        self.assertFalse(upload.file.closed)

    def test_nonseekable_upload_uses_and_cleans_temporary_file_fallback(self) -> None:
        upload = serve.UploadFile(
            file=NonSeekableFile(wav_bytes()),
            filename="sample.wav",
            headers=Headers({"content-type": "audio/wav"}),
        )
        audio_file, _, temporary_path = serve._open_and_inspect_upload(upload)
        try:
            self.assertIsInstance(audio_file, sf.SoundFile)
            self.assertIsInstance(temporary_path, Path)
            self.assertTrue(temporary_path.is_file())
            self.assertEqual(audio_file.format, "WAV")
        finally:
            audio_file.close()
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def test_valid_flac_audio_is_accepted(self) -> None:
        signal = np.sin(np.linspace(0.0, 0.5 * 440.0 * 2.0 * np.pi, 8000, endpoint=False))
        buffer = io.BytesIO()
        sf.write(buffer, signal, 16000, format="FLAC")
        response = self.client.post(
            "/predict",
            files={"audio": ("sample.flac", buffer.getvalue(), "audio/flac")},
        )
        self.assertEqual(response.status_code, 200, response.text)

    def test_invalid_audio_type_and_unreadable_audio_are_rejected(self) -> None:
        unsupported = self.client.post(
            "/predict",
            files={"audio": ("sample.mp3", b"not audio", "audio/mpeg")},
        )
        self.assertEqual(unsupported.status_code, 415)

        unreadable = self.client.post(
            "/predict",
            files={"audio": ("broken.wav", b"not a wav file", "audio/wav")},
        )
        self.assertEqual(unreadable.status_code, 422)

        mismatched_container = self.client.post(
            "/predict",
            files={"audio": ("mismatched.flac", wav_bytes(), "audio/flac")},
        )
        self.assertEqual(mismatched_container.status_code, 422)

    def test_missing_audio_and_invalid_duration_or_sample_rate_are_rejected(self) -> None:
        missing = self.client.post("/predict")
        self.assertEqual(missing.status_code, 422)

        too_short = self.client.post(
            "/predict",
            files={"audio": ("short.wav", wav_bytes(duration=0.05), "audio/wav")},
        )
        self.assertEqual(too_short.status_code, 422)

        rate_too_low = self.client.post(
            "/predict",
            files={"audio": ("low-rate.wav", wav_bytes(sample_rate=4000), "audio/wav")},
        )
        self.assertEqual(rate_too_low.status_code, 422)

    def test_saved_threshold_is_applied_to_model_probability(self) -> None:
        config = {
            "threshold": 0.3691340835714273,
            "selected_model": "logistic_regression",
            "feature_version": "acoustic-v1",
        }
        features = {name: 0.0 for name in FEATURE_NAMES}
        below_threshold = predict_feature_values(
            features,
            {"model": FixedProbabilityModel(0.30), "config": config},
        )
        above_threshold = predict_feature_values(
            features,
            {"model": FixedProbabilityModel(0.40), "config": config},
        )
        self.assertFalse(below_threshold["endpoint_bool"])
        self.assertTrue(above_threshold["endpoint_bool"])

    def test_multiple_concurrent_requests_succeed(self) -> None:
        payload = wav_bytes()

        def send_request(_: int) -> int:
            response = self.client.post(
                "/predict",
                files={"audio": ("parallel.wav", payload, "audio/wav")},
            )
            return response.status_code

        with ThreadPoolExecutor(max_workers=4) as executor:
            statuses = list(executor.map(send_request, range(8)))
        self.assertEqual(statuses, [200] * 8)


if __name__ == "__main__":
    unittest.main()
