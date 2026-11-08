from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from extract_features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    FeatureExtractionError,
    _frequency_grid,
    _hanning_window,
    _mel_filterbank,
    extract_audio_features,
)
from predict import load_model_bundle, predict_audio_file
from train_baseline import fit_and_compare, make_model_candidates


class AudioFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_cached_feature_constants_are_reused_and_read_only(self) -> None:
        constants = (
            (_frequency_grid, 16000),
            (_hanning_window, 400),
            (_mel_filterbank, 16000),
        )
        for factory, argument in constants:
            with self.subTest(factory=factory.__name__):
                first = factory(argument)
                self.assertIs(first, factory(argument))
                self.assertFalse(first.flags.writeable)
                with self.assertRaises(ValueError):
                    first.flat[0] = 0.0

    def test_mono_stereo_and_sample_rates_have_consistent_finite_features(self) -> None:
        mono_path = self.root / "mono.wav"
        stereo_path = self.root / "stereo.wav"
        mono = np.sin(np.linspace(0.0, 40.0, 16000, endpoint=False)).astype(np.float32) * 0.1
        stereo = np.column_stack((mono[:8000], mono[:8000] * 0.5))
        sf.write(mono_path, mono, 16000, format="WAV")
        sf.write(stereo_path, stereo, 22050, format="WAV")

        mono_features = extract_audio_features(mono_path)
        stereo_features = extract_audio_features(stereo_path)
        self.assertEqual(list(mono_features), FEATURE_NAMES)
        self.assertEqual(list(stereo_features), FEATURE_NAMES)
        self.assertEqual(len(mono_features), len(stereo_features))
        self.assertTrue(np.isfinite(list(mono_features.values())).all())
        self.assertTrue(np.isfinite(list(stereo_features.values())).all())
        self.assertAlmostEqual(mono_features["duration_seconds"], 1.0, places=2)

    def test_empty_corrupt_and_nonfinite_samples_are_handled(self) -> None:
        empty_path = self.root / "empty.wav"
        corrupt_path = self.root / "corrupt.wav"
        nonfinite_path = self.root / "nonfinite.wav"
        empty_path.write_bytes(b"")
        corrupt_path.write_bytes(b"not an audio stream")
        sf.write(
            nonfinite_path,
            np.asarray([0.0, np.nan, np.inf, -np.inf] * 100, dtype=np.float32),
            16000,
            format="WAV",
            subtype="FLOAT",
        )

        for path in (empty_path, corrupt_path):
            with self.subTest(path=path.name):
                with self.assertRaises(FeatureExtractionError):
                    extract_audio_features(path)
        result = extract_audio_features(nonfinite_path)
        self.assertTrue(np.isfinite(list(result.values())).all())


class BaselineTests(unittest.TestCase):
    def test_training_is_reproducible_on_synthetic_features(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(size=(48, len(FEATURE_NAMES)))
        labels = np.asarray([False, True] * 24)
        values[:, 0] += labels.astype(float) * 1.5
        train_x = pd.DataFrame(values[:36], columns=FEATURE_NAMES)
        train_y = pd.Series(labels[:36])
        validation_x = pd.DataFrame(values[36:], columns=FEATURE_NAMES)
        validation_y = pd.Series(labels[36:])

        first_models, first_metrics, first_selected = fit_and_compare(
            train_x, train_y, validation_x, validation_y, seed=42
        )
        second_models, second_metrics, second_selected = fit_and_compare(
            train_x, train_y, validation_x, validation_y, seed=42
        )
        self.assertEqual(first_selected, second_selected)
        self.assertEqual(first_metrics.keys(), second_metrics.keys())
        for model_name in first_metrics:
            self.assertEqual(first_metrics[model_name].keys(), second_metrics[model_name].keys())
            for metric_name, first_value in first_metrics[model_name].items():
                second_value = second_metrics[model_name][metric_name]
                if isinstance(first_value, float) and isinstance(second_value, float):
                    self.assertAlmostEqual(first_value, second_value, places=12)
                else:
                    self.assertEqual(first_value, second_value)
        np.testing.assert_allclose(
            first_models[first_selected].predict_proba(validation_x),
            second_models[second_selected].predict_proba(validation_x),
            rtol=1e-14,
            atol=1e-15,
        )

    def test_saved_pipeline_loads_and_returns_prediction_schema(self) -> None:
        rng = np.random.default_rng(7)
        features = pd.DataFrame(rng.normal(size=(24, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
        labels = pd.Series([False, True] * 12)
        model = make_model_candidates(seed=42)["logistic_regression"]
        model.fit(features, labels)
        config = {
            "finalized": True,
            "feature_version": FEATURE_VERSION,
            "feature_names": FEATURE_NAMES,
            "selected_model": "logistic_regression",
            "threshold": 0.5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            joblib.dump(model, model_dir / "selected_model.joblib")
            (model_dir / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
            audio_path = model_dir / "new.wav"
            sf.write(audio_path, np.zeros(4000, dtype=np.float32), 16000, format="WAV")
            bundle = load_model_bundle(model_dir)
            output = predict_audio_file(audio_path, bundle)

        self.assertEqual(
            set(output),
            {"endpoint_bool", "endpoint_probability", "decision_threshold", "model", "feature_version"},
        )
        self.assertIsInstance(output["endpoint_bool"], bool)
        self.assertGreaterEqual(output["endpoint_probability"], 0.0)
        self.assertLessEqual(output["endpoint_probability"], 1.0)


if __name__ == "__main__":
    unittest.main()
