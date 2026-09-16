# Week 3 validation

## Scope

This milestone covers only soil-image dataset verification and EfficientNetB0
classification. It does not include structured nutrient models, hybrid analysis,
SHAP, Grad-CAM, soil-health scoring, persistence, FastAPI, or frontend work.

## Leakage controls

The source `test` directory is copied directly to the prepared test directory.
Only records from `train_data` are stratified into `train` and `validation`.
Validation accuracy selects between the frozen-head and fine-tuned checkpoints;
the test set is evaluated once after selection and never used for fitting or
tuning.

## EfficientNet preprocessing

TensorFlow/Keras EfficientNetB0 includes a preprocessing/rescaling layer and
expects image tensors in the 0-255 range. The pipeline therefore does not add a
second `Rescaling(1./255)` layer. The augmentation layers operate before the
EfficientNet base model and are inactive during evaluation.

## Artifacts

After a successful run, `/content/artifacts` contains:

```text
artifacts/
├── models/       stage checkpoints and soil_cnn_final.keras
├── metrics/      JSON metrics and confusion_matrix.csv
├── plots/        stage curves and confusion matrix
├── predictions/  incorrect_predictions.csv and PNG examples
└── reports/      dataset verification, class names, report, and run summary
```

The run summary records whether stage 1 or stage 2 was selected and the observed
validation difference. It does not claim fine-tuning improved performance unless
the executed values demonstrate that.
