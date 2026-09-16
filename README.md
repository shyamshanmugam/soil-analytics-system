# 🌱 AI-Powered Soil Analytics and Advisory System

An AI-powered soil analytics system that combines **soil image analysis**, **structured soil-test data**, **machine learning**, and **explainable AI** to support soil assessment and agricultural decision-making.

> **Project:** Infosys Springboard Internship 7.0
> **Current Focus:** Milestone 2 — AI/ML Soil Analysis Engine
> **Current Stage:** Week 3 — CNN Soil Image Analysis

---

## 📌 Project Overview

Soil quality plays an important role in agricultural productivity. Traditional soil analysis generally depends on laboratory testing, while visual soil inspection can provide useful information about soil appearance and classification.

This project aims to develop an AI-based system that combines:

* 📷 Soil image analysis
* 🧠 CNN-based deep learning
* 📊 Structured soil-test data analysis
* 🔍 Explainable AI
* 🔗 Hybrid image + structured-data analysis
* 🌱 Soil-health assessment
* 💡 Agricultural recommendations
* 🚀 API-based deployment
* 📱 Progressive Web App (PWA)

The system is being developed incrementally through multiple project milestones.

---

# 🎯 Project Objectives

The major objectives are:

1. Collect and prepare soil-image datasets.
2. Clean and preprocess soil images.
3. Build a CNN-based soil-image classification model.
4. Use transfer learning with EfficientNetB0.
5. Evaluate the CNN using standard classification metrics.
6. Build a structured soil-data machine-learning model.
7. Implement explainable AI using Grad-CAM and SHAP.
8. Combine image-based and structured-data predictions.
9. Develop a soil-health assessment mechanism.
10. Provide the ML functionality through an API.
11. Develop a mobile-friendly PWA interface.
12. Provide understandable soil-analysis results and recommendations.

---

# 🏗️ System Architecture

The planned system architecture is:

```text
                    ┌──────────────────────┐
                    │      User / Farmer   │
                    └──────────┬───────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       ┌─────────────────┐          ┌──────────────────┐
       │   Soil Image    │          │ Structured Soil  │
       │     Input       │          │   Test Data      │
       └────────┬────────┘          └─────────┬────────┘
                │                             │
                ▼                             ▼
       ┌─────────────────┐          ┌──────────────────┐
       │ CNN / Efficient │          │ Structured ML    │
       │    NetB0        │          │ Model            │
       └────────┬────────┘          └─────────┬────────┘
                │                             │
                ▼                             ▼
       ┌─────────────────┐          ┌──────────────────┐
       │ Image Prediction│          │ Soil Parameter   │
       │ + Confidence    │          │ Predictions      │
       └────────┬────────┘          └─────────┬────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                     ┌────────────────────┐
                     │  Hybrid Analysis   │
                     └─────────┬──────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          Soil Status      XAI Results    Health Score
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                     ┌────────────────────┐
                     │ Recommendations    │
                     └────────────────────┘
```

---

# 📅 Project Milestones

## Milestone 1 — Data Ingestion & Preparation

### Week 1 — Requirement & Dataset Collection

Tasks include:

* Understand project requirements.
* Identify required soil parameters:

  * Nitrogen (N)
  * Phosphorus (P)
  * Potassium (K)
  * pH
  * Moisture
  * Organic Matter
* Collect suitable soil-image datasets.
* Identify reliable soil-test datasets.
* Review dataset licenses and restrictions.
* Analyze dataset size, classes, labels, and quality.
* Set up project structure.
* Set up Git/GitHub.
* Set up Python/Google Colab.
* Prepare dataset documentation.

### Week 2 — Data Cleaning & Preprocessing

#### Image Processing

* Detect corrupted images.
* Detect/remove duplicate images.
* Verify image labels.
* Resize images.
* Apply appropriate preprocessing.
* Apply augmentation.
* Organize training, validation, and test data.

#### Structured Data

* Load CSV/Excel data.
* Check missing values.
* Check invalid values.
* Identify outliers.
* Remove duplicates.
* Perform encoding/scaling where required.
* Perform exploratory data analysis.

#### Database

Design PostgreSQL tables for:

* Farms
* Fields
* Soil samples
* Soil-test records

---

# 🧠 Milestone 2 — AI/ML Soil Analysis Engine

## Week 3 — CNN Soil Image Analysis

The current implementation focuses on classifying soil images using a CNN.

### Dataset

The current image dataset contains four soil classes:

| Class         |
| ------------- |
| Red Soil      |
| Black Soil    |
| Alluvial Soil |
| Clay Soil     |

The dataset previously contained approximately **529 images** before duplicate removal.

The dataset should always be rechecked from the current data before publishing final statistics.

---

# 🔬 Week 3 CNN Pipeline

The current CNN pipeline follows:

```text
Raw Soil Images
       ↓
Dataset Verification
       ↓
Corrupted Image Detection
       ↓
Duplicate Detection
       ↓
Label Verification
       ↓
RGB Conversion
       ↓
Image Resizing
       ↓
Train / Validation / Test Split
       ↓
Image Augmentation
       ↓
EfficientNetB0 Transfer Learning
       ↓
Initial Training
       ↓
Fine-Tuning
       ↓
Prediction
       ↓
Evaluation
       ↓
Model Selection
       ↓
Saved CNN Model
```

---

# 🧹 Image Data Preparation

The preprocessing pipeline performs:

### Corrupted Image Detection

Images are checked to ensure they can be opened and processed correctly.

### Duplicate Detection

Exact duplicate files can be identified using file hashing.

### Image Standardization

Images are converted to a consistent RGB representation where required.

### Image Resizing

Images are resized to the input dimensions required by the selected CNN.

Current EfficientNetB0 configuration:

```text
224 × 224 pixels
```

### Data Augmentation

Training images can be augmented using transformations such as:

* Random horizontal flip
* Random rotation
* Random zoom

Augmentation is applied to training data rather than the final test set.

---

# 🧠 CNN Model

The project currently uses:

## EfficientNetB0

EfficientNetB0 is used through transfer learning.

The general architecture is:

```text
Input Image
     ↓
EfficientNetB0
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense Classification Layer
     ↓
Softmax
     ↓
Soil Class
```

---

# 🔄 Transfer Learning

Transfer learning uses a pretrained neural network as a starting point rather than training the entire network from random initialization.

The workflow is:

```text
Pretrained EfficientNetB0
          ↓
Freeze pretrained layers
          ↓
Train classification head
          ↓
Evaluate
          ↓
Unfreeze selected layers
          ↓
Fine-tune with smaller learning rate
          ↓
Evaluate again
```

This approach is useful when the available domain-specific image dataset is relatively limited.

---

# ⚙️ Model Training

The training pipeline includes:

* Training dataset
* Validation dataset
* Optimizer
* Loss function
* Learning rate
* Batch size
* Training epochs
* Early stopping
* Model checkpointing
* Learning-rate reduction

The best model should be selected using validation performance rather than simply taking the last training epoch.

---

# 📊 Model Evaluation

The CNN is evaluated using:

### Accuracy

Percentage of correctly classified test samples.

### Precision

Measures how many samples predicted as a particular class actually belong to that class.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

Harmonic mean of precision and recall.

### Confusion Matrix

Shows the relationship between actual and predicted classes and helps identify which soil classes are being confused.

---

# 📈 Week 3 Results

> ⚠️ These values must be replaced with the actual results produced by the final executed experiment.

| Metric        |          Result |
| ------------- | --------------: |
| Test Accuracy | `[RUN RESULTS]` |
| Precision     | `[RUN RESULTS]` |
| Recall        | `[RUN RESULTS]` |
| F1-Score      | `[RUN RESULTS]` |

### Training Curves

The project generates:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

These curves are used to investigate:

* Model convergence
* Overfitting
* Underfitting
* Training stability

---

# 🔎 Incorrect Prediction Analysis

Incorrect test predictions are inspected to understand model limitations.

The analysis includes:

* Original image
* Actual class
* Predicted class
* Prediction confidence

Potential sources of errors are investigated rather than assuming a specific cause.

---

# 💾 Model Artifacts

The project aims to save:

```text
artifacts/
├── models/
│   ├── best_model.keras
│   └── final_model.keras
│
├── metrics/
│   ├── metrics.json
│   └── classification_report.csv
│
├── plots/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
│
├── predictions/
│   └── incorrect_predictions.csv
│
└── reports/
    └── week3_evaluation_report.md
```

Actual filenames may differ depending on the implementation.

---

# 📊 Week 4 — Structured Soil Data Model

Week 4 will extend the project to structured soil-test data.

Required parameters include:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* pH
* Moisture
* Organic Matter

The structured-data pipeline will include:

```text
CSV / Excel
     ↓
Data Validation
     ↓
Missing Value Handling
     ↓
Invalid Value Detection
     ↓
Outlier Analysis
     ↓
Duplicate Removal
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Gradient Boosting / XGBoost
     ↓
Evaluation
     ↓
Hyperparameter Tuning
     ↓
Saved Model
```

The exact prediction target must be determined from the actual structured dataset.

---

# 🔍 Explainable AI

The planned project includes two explainability approaches.

## Grad-CAM

Grad-CAM will be applied to the CNN to visualize image regions contributing to a prediction.

Expected output:

```text
Original Soil Image
        +
Grad-CAM Heatmap
        ↓
Highlighted Important Regions
```

Grad-CAM should be interpreted as a model-attribution visualization rather than proof of causal reasoning.

## SHAP

SHAP will be used for the structured-data model where appropriate.

It can provide:

* Feature importance
* Global explanations
* Individual prediction explanations

Potential features include:

* N
* P
* K
* pH
* Moisture
* Organic Matter

The actual feature list will depend on the final structured dataset.

---

# 🔗 Hybrid Analysis

The planned hybrid system combines:

```text
CNN Image Prediction
        +
Structured ML Prediction
        ↓
Hybrid Analysis
        ↓
Combined Soil Assessment
```

The combination logic will be explicitly documented.

The system should account for:

* Available input modalities
* Model confidence
* Prediction outputs
* Missing modalities

No confidence value should be fabricated.

---

# 🌱 Soil Health Assessment

A future component will derive a soil-health assessment from available soil information.

The scoring methodology must be:

* Transparent
* Configurable
* Documented
* Clearly distinguished from scientifically validated laboratory measurements

The system is not intended to replace professional laboratory soil testing.

---

# 🗄️ Database

The planned PostgreSQL database will contain structures for:

```text
Farm
  │
  └── Field
        │
        └── Soil Sample
               │
               ├── Soil Test Results
               ├── Image Sample
               └── Predictions
```

Potential tables include:

* `farms`
* `fields`
* `soil_samples`
* `soil_test_results`
* `image_samples`
* `predictions`
* `model_versions`

Database credentials must never be committed to GitHub.

---

# 🚀 Backend

A future FastAPI backend will expose the machine-learning functionality through APIs.

Planned endpoints include:

```text
GET  /health

POST /predict/image

POST /predict/structured

POST /predict/hybrid

GET  /models
```

The exact API design may change during implementation.

---

# 📱 Frontend / PWA

The planned PWA will provide a mobile-friendly interface where users can:

* Upload soil images.
* Enter soil-test parameters.
* Request soil analysis.
* View soil classification.
* View nutrient status.
* View confidence values.
* View explainability results.
* View soil-health assessment.
* View recommendations.

---

# 🛠️ Technology Stack

## Machine Learning

* Python
* TensorFlow / Keras
* EfficientNetB0
* Scikit-learn
* XGBoost
* SHAP

## Data Processing

* Pandas
* NumPy
* Pillow
* Matplotlib

## Backend

* FastAPI
* Pydantic

## Database

* PostgreSQL

## Frontend

* PWA-compatible web technologies

## Development

* Google Colab
* Git
* GitHub

---

# 📁 Project Structure

The target project structure is:

```text
soil-analytics/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── structured/
│
├── notebooks/
│   ├── week1_dataset_analysis.ipynb
│   ├── week2_preprocessing.ipynb
│   ├── week3_cnn.ipynb
│   └── week4_structured_ml_xai.ipynb
│
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   ├── explainability/
│   ├── hybrid/
│   └── utils/
│
├── models/
│
├── artifacts/
│   ├── metrics/
│   ├── plots/
│   ├── predictions/
│   └── reports/
│
├── database/
│   └── schema/
│
├── backend/
│   └── app/
│
├── frontend/
│
├── tests/
│
└── docs/
    ├── dataset.md
    ├── architecture.md
    ├── week3.md
    └── week4.md
```

---

# 💻 Google Colab

The CNN training pipeline can be executed using Google Colab.

General workflow:

```text
Open Colab
   ↓
Upload dataset
   ↓
Install dependencies
   ↓
Run preprocessing
   ↓
Create dataset splits
   ↓
Build EfficientNetB0
   ↓
Train
   ↓
Fine-tune
   ↓
Evaluate
   ↓
Save artifacts
```

The notebook should be executed from beginning to end to reproduce the results.

---

# 🔐 Data & Security

The repository should never contain:

* API keys
* Passwords
* Database credentials
* Private tokens
* Personal credentials
* Unnecessary private datasets

Large datasets should normally be stored outside GitHub and documented through dataset-access instructions.

---

# ⚠️ Limitations

Important limitations include:

1. Image classification does not automatically provide laboratory measurements of N, P, K, pH, moisture, or organic matter.
2. Model performance depends strongly on dataset quality and representativeness.
3. A small or imbalanced dataset may limit generalization.
4. Predictions should not be treated as a replacement for laboratory soil testing.
5. Explainability methods such as Grad-CAM and SHAP describe model behavior but do not establish causality.
6. Soil-health scores require clearly documented agronomic assumptions.
7. Results should be validated with appropriate real-world soil-test data before practical agricultural deployment.

---

# 🔮 Future Improvements

Possible future improvements include:

* Larger and more diverse soil-image datasets.
* More geographic locations and soil conditions.
* Additional soil parameters.
* Better calibration of prediction confidence.
* More robust structured-data models.
* Improved hybrid fusion.
* Advanced explainability.
* PostgreSQL integration.
* FastAPI deployment.
* PWA deployment.
* Model monitoring.
* Continuous evaluation with new field data.

---

# 📌 Current Development Status

| Component                         | Status                        |
| --------------------------------- | ----------------------------- |
| Repository                        | ✅ Created                     |
| Soil image dataset                | ✅ Available                   |
| Image cleaning                    | ✅ Implemented                 |
| Duplicate detection               | ✅ Implemented                 |
| Image preprocessing               | ✅ Implemented                 |
| Data augmentation                 | ✅ Implemented                 |
| Train/validation/test preparation | ✅ Implemented                 |
| EfficientNetB0                    | ✅ Implemented                 |
| Transfer learning                 | ✅ Implemented                 |
| CNN training                      | ✅ Implemented                 |
| CNN evaluation                    | 🟡 Verify final execution     |
| Fine-tuning                       | 🟡 To be verified/implemented |
| Week 3 report                     | 🟡 In progress                |
| Structured soil-data model        | ⏳ Planned                     |
| XGBoost/Gradient Boosting         | ⏳ Planned                     |
| SHAP                              | ⏳ Planned                     |
| Grad-CAM                          | ⏳ Planned                     |
| Hybrid analysis                   | ⏳ Planned                     |
| Soil-health scoring               | ⏳ Planned                     |
| PostgreSQL                        | ⏳ Planned                     |
| FastAPI                           | ⏳ Planned                     |
| PWA                               | ⏳ Planned                     |

---

# 👥 Contributors

**Project Team**

Infosys Springboard Internship 7.0

Individual contributors and responsibilities should be added here as the project develops.

---

# 📚 Disclaimer

This project is an academic/internship machine-learning project.

The system is intended for research, learning, and prototype purposes.

AI predictions should not be considered a substitute for professional laboratory soil testing or qualified agricultural advice.

---

# ⭐ Project Status

**Current milestone:** Milestone 2 — AI/ML Soil Analysis Engine

**Current focus:** Week 3 — CNN Soil Image Analysis

The immediate objective is to complete and validate the CNN image-classification pipeline and produce a reproducible Week 3 evaluation report before moving to the Week 4 structured-data and explainability components.
