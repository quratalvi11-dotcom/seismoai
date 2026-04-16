# seismoai
# SeismoAI

A Python library for seismic data processing built on the Utah FORGE 2D Survey dataset.

## Modules

### seismoai_io — Data Loading
Loads and prepares SGY seismic files.

- `load_sgy(filepath)` — Load a single .sgy file
- `load_folder(folder_path)` — Load all .sgy files from a folder  
- `normalize_traces(traces, method)` — Normalize trace amplitudes

### seismoai_model — ML Classifier
Trains and uses a noise classifier on seismic traces.

- `extract_features(traces)` — Extract 6 statistical features per trace
- `train_classifier(traces, labels)` — Train a Random Forest classifier
- `predict_traces(traces, model_dict)` — Predict labels for new traces

## Dataset

- **Source:** Utah FORGE 2D Seismic Survey, 2017
- **Files:** 166 SGY files
- **Traces per file:** 167
- **Samples per trace:** 4,001 at 1ms interval

## Installation

```bash
pip install segyio numpy pandas matplotlib scikit-learn shap
```

## Usage

```python
from seismoai_io import load_sgy, normalize_traces
from seismoai_model import extract_features, train_classifier, predict_traces

# Load data
data = load_sgy("data/your_file.sgy")
traces = normalize_traces(data['traces'], method='zscore')

# Extract features and predict
features = extract_features(traces)
```

## Running Tests

```bash
pytest seismoai_io/tests/test_io.py -v
pytest seismoai_model/tests/test_model.py -v
```

## Team

MLOps Course — SeismoAI Group Project