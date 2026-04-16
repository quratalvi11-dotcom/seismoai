"""SeismoAI Model module — train and use a noise classifier."""
from .model_core import extract_features, train_classifier, predict_traces
__all__ = ["extract_features", "train_classifier", "predict_traces"]
__version__ = "0.1.0"
