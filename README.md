# Student Dropout Prediction

Predicts student dropout (churn) from the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) using a Logistic Regression pipeline with SMOTE oversampling.

Features are built from student demographics, VLE interaction logs, and assessment submission history up to a configurable prediction window (default: day 28).

---

## Project Structure

```
churn_prediction/
├── churn_prediction.py        # Entry point
├── pyproject.toml             # Project metadata and dependencies
├── config/
│   └── config_churn.yaml      # Model and pipeline configuration
├── data/                      # Raw OULAD CSV files (not tracked in git)
│   ├── assessments.csv
│   ├── courses.csv
│   ├── studentAssessment.csv
│   ├── studentInfo.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   └── vle.csv
├── models/                    # Saved model artifacts
└── src/
    └── utils/
        └── churn_predictor.py # ChurnPredictor class
```

---

## Data

The project uses the OULAD dataset. Download the CSV files from the [UCI ML Repository (ID 349)](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) and place them in the `data/` directory.

Required files:
- `studentInfo.csv`
- `studentVle.csv`
- `studentAssessment.csv`
- `assessments.csv`
- `studentRegistration.csv`

---

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

### Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create environment and install dependencies

```bash
uv sync
```

This reads `pyproject.toml`, creates a virtual environment in `.venv/`, and installs all required packages.

To include optional development dependencies (Jupyter, matplotlib, seaborn):

```bash
uv sync --extra dev
```

---

## Configuration

All settings are in `config/config_churn.yaml`.

| Parameter | Default | Description |
|---|---|---|
| `step` | `test` | Default execution step (`train` or `test`) |
| `preprocessing.window_days` | `28` | Prediction cutoff in days from course start |
| `preprocessing.smote.enabled` | `true` | Apply SMOTE oversampling during training |
| `preprocessing.scaling.enabled` | `true` | Apply StandardScaler |
| `model.hyperparameters.C` | `0.001` | Logistic Regression regularisation strength |
| `model.threshold` | `0.40` | Decision threshold for dropout classification |
| `data.test_size` | `0.2` | Proportion of data held out for evaluation |
| `output.model_path` | `models/churn_model.pkl` | Path to save/load the trained model |

---

## Running

### Train

Trains the model on the OULAD data, evaluates on the held-out test set, and saves the pipeline to `models/churn_model.pkl`.

```bash
uv run churn_prediction.py step=train
```

### Evaluate a saved model

Loads the saved model and runs evaluation on the test split without retraining.

```bash
uv run churn_prediction.py step=test
```

The `step` argument overrides the value set in `config_churn.yaml`.

---

## Pipeline

The training pipeline consists of three sequential steps:

1. **StandardScaler** - normalises all numeric features
2. **SMOTE** - oversamples the minority class (withdrawn students) to address class imbalance
3. **LogisticRegression** - binary classifier

---

## Output

Evaluation prints the following metrics to the console:

- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix
- Full classification report (Stayed / Dropped)

The trained pipeline is serialised with `joblib` to `models/churn_model.pkl`.

---

## Requirements

- Python >= 3.10, < 3.13
- uv >= 0.4 (for `uv sync` and `uv run`)

Core dependencies (managed via `pyproject.toml`):

| Package | Version |
|---|---|
| pandas | 2.2.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.3.2 |
| imbalanced-learn | 0.11.0 |
| pyyaml | 6.0.1 |
| joblib | 1.3.2 |
