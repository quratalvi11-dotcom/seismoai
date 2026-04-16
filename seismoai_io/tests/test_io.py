"""Tests for seismoai_io module."""
import os
import pytest
import numpy as np
from seismoai_io import load_sgy, load_folder, normalize_traces

SAMPLE_SGY = os.path.join("data", "27_1511546140_30100_50100_20171127_150416_752.sgy")


class TestLoadSgy:

    def test_correct_shape(self):
        if not os.path.isfile(SAMPLE_SGY):
            pytest.skip("SGY file not available")
        traces, meta = load_sgy(SAMPLE_SGY)
        assert traces.shape == (167, 4001)

    def test_correct_meta(self):
        if not os.path.isfile(SAMPLE_SGY):
            pytest.skip("SGY file not available")
        traces, meta = load_sgy(SAMPLE_SGY)
        assert meta["n_traces"] == 167
        assert meta["n_samples"] == 4001
        assert meta["sample_rate_ms"] == 1.0

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_sgy("nonexistent.sgy")


class TestLoadFolder:

    def test_loads_folder(self):
        if not os.path.isdir("data"):
            pytest.skip("Data folder not available")
        results = load_folder("data")
        assert len(results) > 0
        first_traces, first_meta = results[0]
        assert first_traces.shape[0] == 167

    def test_folder_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_folder("no_such_folder")


class TestNormalizeTraces:

    def test_zscore_mean_zero(self):
        traces = np.random.randn(5, 100).astype(np.float32)
        normed = normalize_traces(traces, method="zscore")
        for i in range(5):
            assert abs(normed[i].mean()) < 1e-10
            assert abs(normed[i].std() - 1.0) < 1e-10

    def test_minmax_range(self):
        traces = np.random.randn(5, 100).astype(np.float32)
        normed = normalize_traces(traces, method="minmax")
        assert normed.min() >= -1e-10
        assert normed.max() <= 1.0 + 1e-10

    def test_dead_trace_no_nan(self):
        traces = np.zeros((1, 100), dtype=np.float32)
        normed = normalize_traces(traces, method="zscore")
        assert np.all(normed == 0.0)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            normalize_traces(np.zeros((2, 10)), method="bad")