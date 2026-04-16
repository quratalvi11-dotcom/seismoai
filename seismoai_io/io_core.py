"""SeismoAI I/O module - load and prepare SGY seismic files."""

import os
import glob
import numpy as np
import pandas as pd
import segyio


def load_sgy(filepath: str):
    """Load a single SGY file and return traces and metadata.

    Parameters
    ----------
    filepath : str
        Path to the .sgy file.

    Returns
    -------
    tuple: (traces, meta)
        traces : np.ndarray of shape (n_traces, n_samples)
        meta   : dict with n_traces, n_samples, sample_rate_ms, filepath

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If the file cannot be parsed as SEG-Y.

    Examples
    --------
    >>> traces, meta = load_sgy("data/file.sgy")
    >>> traces.shape
    (167, 4001)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"SGY file not found: {filepath}")

    for endian in ("little", "big"):
        try:
            with segyio.open(filepath, ignore_geometry=True, endian=endian) as f:
                traces = segyio.tools.collect(f.trace[:])
                sample_rate_ms = segyio.tools.dt(f) / 1000.0
                headers = pd.DataFrame([
                    {str(k): v for k, v in f.header[i].items()}
                    for i in range(f.tracecount)
                ])
            meta = {
                "n_traces": traces.shape[0],
                "n_samples": traces.shape[1],
                "sample_rate_ms": sample_rate_ms,
                "filepath": filepath,
                "headers": headers,
            }
            return traces, meta
        except RuntimeError:
            continue

    raise RuntimeError(f"Cannot open SGY file (tried both endians): {filepath}")


def load_folder(folder_path: str) -> list:
    """Load all .sgy files from a folder.

    Parameters
    ----------
    folder_path : str
        Path to folder containing .sgy files.

    Returns
    -------
    list of (traces, meta) tuples, one per file.

    Raises
    ------
    FileNotFoundError
        If folder doesn't exist or has no .sgy files.

    Examples
    --------
    >>> results = load_folder("data/")
    >>> len(results)
    2
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(glob.glob(os.path.join(folder_path, "*.sgy")))
    if not files:
        raise FileNotFoundError(f"No .sgy files found in: {folder_path}")

    results = []
    for fp in files:
        try:
            results.append(load_sgy(fp))
            print(f"  Loaded: {os.path.basename(fp)}")
        except RuntimeError as e:
            print(f"  WARNING: Skipping {os.path.basename(fp)}: {e}")

    return results


def normalize_traces(traces: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize seismic trace amplitudes.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples).
    method : str
        One of: 'zscore', 'minmax', 'trace_max'.

    Returns
    -------
    np.ndarray
        Normalized traces, same shape, dtype float64.

    Raises
    ------
    ValueError
        If method is invalid or traces is not 2D.

    Examples
    --------
    >>> import numpy as np
    >>> traces = np.array([[0, 5, -10, 3]], dtype=np.float32)
    >>> normalize_traces(traces, method='trace_max')
    array([[ 0. ,  0.5, -1. ,  0.3]])
    """
    if traces.ndim != 2:
        raise ValueError(f"Expected 2D array, got {traces.ndim}D")

    valid = ("zscore", "minmax", "trace_max")
    if method not in valid:
        raise ValueError(f"method must be one of {valid}, got '{method}'")

    result = np.zeros_like(traces, dtype=np.float64)

    for i in range(traces.shape[0]):
        tr = traces[i].astype(np.float64)

        if method == "minmax":
            mn, mx = tr.min(), tr.max()
            result[i] = 0.0 if mx == mn else (tr - mn) / (mx - mn)

        elif method == "zscore":
            mu, sigma = tr.mean(), tr.std()
            result[i] = 0.0 if sigma == 0 else (tr - mu) / sigma

        elif method == "trace_max":
            mx = np.max(np.abs(tr))
            result[i] = 0.0 if mx == 0 else tr / mx

    return result