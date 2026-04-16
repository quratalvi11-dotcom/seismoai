"""Tests for seismoai_model - run with: pytest seismoai_model/tests/test_model.py -v"""

import numpy as np
import pytest
from seismoai_model import extract_features, train_classifier, predict_traces


class TestExtractFeatures:

    def test_output_shape(self):
        traces = np.random.randn(10, 4001).astype(np.float32)
        features = extract_features(traces)
        assert features.shape == (10, 6)

    def test_dead_trace_gives_zeros(self):
        traces = np.zeros((1, 4001), dtype=np.float32)
        features = extract_features(traces)
        assert features[0, 0] == 0.0
        assert features[0, 1] == 0.0

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            extract_features(np.zeros(100))


class TestTrainClassifier:

    def test_returns_model_dict(self):
        traces = np.random.randn(30, 4001).astype(np.float32)
        labels = np.array(['good'] * 15 + ['noisy'] * 15)
        result = train_classifier(traces, labels)
        assert 'model' in result
        assert 'classes' in result
        assert 'report' in result

    def test_classes_correct(self):
        traces = np.random.randn(30, 4001).astype(np.float32)
        labels = np.array(['good'] * 15 + ['noisy'] * 15)
        result = train_classifier(traces, labels)
        assert 'good' in result['classes']
        assert 'noisy' in result['classes']


class TestPredictTraces:

    def test_predictions_length(self):
        traces = np.random.randn(30, 4001).astype(np.float32)
        labels = np.array(['good'] * 15 + ['noisy'] * 15)
        model_dict = train_classifier(traces, labels)
        preds, probs = predict_traces(traces[:10], model_dict)
        assert len(preds) == 10

    def test_predictions_are_strings(self):
        traces = np.random.randn(30, 4001).astype(np.float32)
        labels = np.array(['good'] * 15 + ['noisy'] * 15)
        model_dict = train_classifier(traces, labels)
        preds, probs = predict_traces(traces[:5], model_dict)
        assert all(isinstance(p, str) for p in preds)

    def test_probs_shape(self):
        traces = np.random.randn(30, 4001).astype(np.float32)
        labels = np.array(['good'] * 15 + ['noisy'] * 15)
        model_dict = train_classifier(traces, labels)
        preds, probs = predict_traces(traces[:5], model_dict)
        assert probs.shape[0] == 5