# AI-Powered Soil Analytics and Advisory System

An Infosys Springboard Internship 7.0 project for soil-image classification and
future agricultural advisory support.

> **Current milestone:** Milestone 2 — AI/ML Soil Analysis Engine
> **Current focus:** Week 3 — CNN Soil Image Analysis

## Project overview

The planned system combines soil images with structured laboratory measurements
to support soil assessment. The modalities remain separate:

- Soil images can support soil-type classification.
- Laboratory data is required for N, P, K, pH, moisture, and organic-matter
  analysis.
- An image classifier must not be presented as a laboratory nutrient predictor.

The repository currently prioritizes the Week 3 image-classification pipeline.
Structured-data modelling, explainability, hybrid fusion, deployment, and farmer
advisory features are future milestones.

## Week 3 scope

The canonical pipeline is available in:

- `notebooks/week3_cnn_pipeline.py`
- `notebooks/week3_cnn_pipeline.ipynb`

It performs:

1. Dataset verification and class normalization
2. Corrupted-image detection
3. Exact duplicate detection
4. Class and split distribution reporting
5. RGB conversion and `224x224` resizing
6. Stratified train/validation preparation from training data only
7. Test-set isolation
8. EfficientNetB0 preprocessing
9. Training augmentation
10. Frozen-base transfer learning
11. Upper-layer fine-tuning with a smaller learning rate
12. Validation-based best-model selection
13. Test prediction and evaluation
14. Accuracy, macro precision, macro recall, and macro F1
15. Classification report and confusion matrix
16. Incorrect-prediction analysis
17. Saved models and evaluation artifacts

### EfficientNetB0 preprocessing

The TensorFlow/Keras EfficientNetB0 implementation used by the pipeline includes
its own input rescaling and expects image tensors in the `0-255` range. The
pipeline intentionally does **not** add another `Rescaling(1./255)` layer, which
avoids double normalization.

### Fine-tuning

Stage 1 freezes the pretrained EfficientNetB0 base and trains the classification
head. Stage 2 unfreezes an upper portion of the base, keeps batch-normalization
layers frozen for stability, and uses a learning rate of `1e-5`. The final model
is selected using validation accuracy before the test set is evaluated.

## Dataset

The expected image classes are:

| Class |
| --- |
| Red Soil |
| Black Soil |
| Alluvial Soil |
| Clay Soil |

The dataset is not committed to GitHub. The pipeline expects a source directory
with this shape:

```text
soil_dataset/
├── train_data/
│   ├── Alluvial Soil/
│   ├── Black Soil/
│   ├── Clay Soil/
│   └── Red Soil/
└── test/
    ├── Alluvial Soil/
    ├── Black Soil/
    ├── Clay Soil/
    └── Red Soil/
```

Any historical image count must be regenerated from the current dataset before
being reported. The pipeline records verified, corrupted, duplicate, class, and
split counts in its output reports.

## Artifacts

After a successful run, the output directory contains:

```text
artifacts/
├── models/
│   ├── stage1_best.keras
│   ├── stage2_best.keras
│   └── soil_cnn_final.keras
├── metrics/
│   ├── test_metrics.json
│   ├── classification_report.json
│   └── confusion_matrix.csv
├── plots/
│   ├── stage1_accuracy.png
│   ├── stage1_loss.png
│   ├── stage2_accuracy.png
│   ├── stage2_loss.png
│   └── confusion_matrix.png
├── predictions/
│   ├── incorrect_predictions.csv
│   └── incorrect_predictions.png
└── reports/
    ├── dataset_verification.json
    ├── split_counts.json
    ├── class_names.json
    ├── classification_report.txt
    └── run_summary.json
```

Precision, recall, and F1 use **macro averaging**, which gives every soil class
equal weight. No result is written into this README until the training pipeline
has actually been executed.

## Run in Google Colab

Install the Week 3 dependencies:

```python
!pip install -r requirements-colab.txt
```

Upload and extract the dataset so that `/content/soil_dataset` has the structure
shown above, then run:

```python
!python notebooks/week3_cnn_pipeline.py \
  --dataset-root /content/soil_dataset \
  --artifact-root /content/artifacts \
  --stage1-epochs 20 \
  --stage2-epochs 10
```

Alternatively, open `notebooks/week3_cnn_pipeline.ipynb` and execute it from
top to bottom. TensorFlow training requires a suitable Colab runtime and may
take substantial time.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-colab.txt
python notebooks/week3_cnn_pipeline.py `
  --dataset-root C:\path\to\soil_dataset `
  --artifact-root artifacts
```

## Validation and leakage controls

- Files are verified before preprocessing.
- Exact duplicates are excluded from the working set.
- The validation set is split only from `train_data`.
- The source `test` set is never used for fitting or tuning.
- Validation performance chooses between Stage 1 and Stage 2.
- The selected model is evaluated on the test set only after selection.
- The saved `.keras` model is loaded before final evaluation.

See [`docs/week3-validation.md`](docs/week3-validation.md) for the detailed
validation rules and artifact contract.

## Current results

No Week 3 training run has been executed in this repository environment because
Python and TensorFlow were unavailable. Therefore, accuracy, precision, recall,
F1-score, confusion-matrix values, and fine-tuning improvement are currently
**not available** and must not be fabricated.

## Repository structure

```text
soil-analytics-system/
├── README.md
├── requirements-colab.txt
├── data/
├── notebooks/
│   ├── week3_cnn_pipeline.py
│   ├── week3_cnn_pipeline.ipynb
│   ├── week3_soil_dataset_and_efficientnet.py
│   └── week3_soil_dataset_and_efficientnet.ipynb
├── src/
├── backend/
├── frontend/
├── tests/
└── docs/
```

The supplied original Week 3 files are retained as reference. New runs should
use `week3_cnn_pipeline.py` or its matching notebook.

## Future milestones

These items are intentionally not part of the current Week 3 implementation:

- Structured soil-data model using Gradient Boosting or XGBoost
- SHAP explanations
- Grad-CAM explanations
- Hybrid image and structured-data fusion
- Soil-health scoring
- PostgreSQL persistence
- FastAPI deployment
- Frontend/PWA integration
- Multilingual and voice features

They must only be started after the Week 3 pipeline has been executed, reviewed,
and documented with real results.

## Limitations and disclaimer

Image classification does not automatically provide laboratory measurements of
N, P, K, pH, moisture, or organic matter. Model performance depends on dataset
quality, class balance, geographic coverage, and real-world representativeness.
Predictions are for academic and prototype use and do not replace laboratory
soil testing or qualified agricultural advice.

## Security and data handling

Do not commit API keys, passwords, database credentials, private tokens, or
private datasets. Large datasets and generated model artifacts should remain
outside Git unless deliberately reviewed and documented.

## Project status

| Component | Status |
| --- | --- |
| Dataset verification | Implemented; requires execution on the real dataset |
| Duplicate and corruption checks | Implemented; requires execution |
| Image preprocessing | Implemented |
| EfficientNetB0 | Implemented |
| Transfer learning | Implemented |
| Fine-tuning | Implemented; requires execution and comparison |
| CNN evaluation | Implemented; requires execution |
| Actual Week 3 metrics | Not available yet |
| Structured-data model | Planned |
| SHAP | Planned |
| Grad-CAM | Planned |
| Hybrid analysis | Planned |
| Soil-health scoring | Planned |
| PostgreSQL | Planned |
| FastAPI/PWA | Planned |

## Contributors

Infosys Springboard Internship 7.0 project team.
