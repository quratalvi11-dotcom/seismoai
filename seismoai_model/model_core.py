"""SeismoAI Model module - feature extraction, training, prediction."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def extract_features(traces):
    """Extract 6 statistical features from each seismic trace.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples).

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_traces, 6).

    Examples
    --------
    >>> import numpy as np
    >>> extract_features(np.random.randn(10, 4001).astype(np.float32)).shape
    (10, 6)
    """
    if traces.ndim != 2:
        raise ValueError(f"Expected 2D array, got {traces.ndim}D")
    out = []
    for tr in traces:
        tr = tr.astype(float)
        std = float(tr.std())
        k = float((((tr - tr.mean()) / std) ** 4).mean()) if std > 0 else 0.0
        out.append([
            float(abs(tr).mean()),
            std,
            float(abs(tr).max()),
            float((tr ** 2).sum()),
            float((np.diff(np.sign(tr)) != 0).sum()),
            k
        ])
    return np.array(out)


def train_classifier(traces, labels):
    """Train a Random Forest classifier on trace features.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples) - raw trace data.
    labels : np.ndarray
        1D array of string labels e.g. 'good', 'noisy', 'dead'.

    Returns
    -------
    dict with keys:
        'model'   : trained RandomForestClassifier
        'classes' : list of class name strings
        'report'  : classification report string

    Examples
    --------
    >>> import numpy as np
    >>> traces = np.random.randn(30, 4001).astype(np.float32)
    >>> labels = np.array(['good'] * 15 + ['noisy'] * 15)
    >>> result = train_classifier(traces, labels)
    >>> 'model' in result
    True
    """
    if len(traces) != len(labels):
        raise ValueError("traces and labels must have the same length")
    if len(traces) < 2:
        raise ValueError("Need at least 2 samples to train")

    features = extract_features(traces)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)
    report = classification_report(labels, clf.predict(features), zero_division=0)

    return {
        'model': clf,
        'classes': list(clf.classes_),
        'report': report
    }


def predict_traces(traces, model_dict):
    """Predict labels for new traces using a trained classifier.

    Parameters
    ----------
    traces : np.ndarray
        2D array of shape (n_traces, n_samples) - raw trace data.
    model_dict : dict
        Output from train_classifier() containing 'model' key.

    Returns
    -------
    tuple : (predictions, probabilities)
        predictions : np.ndarray of string labels e.g. 'good', 'noisy'
        probabilities : np.ndarray of shape (n_traces, n_classes)

    Examples
    --------
    >>> import numpy as np
    >>> traces = np.random.randn(30, 4001).astype(np.float32)
    >>> labels = np.array(['good'] * 15 + ['noisy'] * 15)
    >>> model_dict = train_classifier(traces, labels)
    >>> preds, probs = predict_traces(traces[:5], model_dict)
    >>> len(preds)
    5
    """
    clf = model_dict['model']
    features = extract_features(traces)
    preds = clf.predict(features)
    probs = clf.predict_proba(features)
    return preds, probs