# Week 3 notebooks

`week3_soil_dataset_and_efficientnet.py` and
`week3_soil_dataset_and_efficientnet.ipynb` are the supplied Infosys project
workflow and are retained unchanged as the original reference. The corrected
Week 3 implementation is `week3_cnn_pipeline.py` and
`week3_cnn_pipeline.ipynb`; use those files for new runs.

The workflow performs:

- ZIP upload and extraction in Google Colab
- soil-class discovery and normalization
- corruption and duplicate checks
- RGB conversion and 224x224 resizing
- train/validation/test directory creation
- EfficientNetB0 transfer-learning setup
- training, evaluation, confusion matrix, and artifact export

The corrected pipeline additionally:

- verifies file content and removes duplicate content from the working set
- creates a stratified validation split from training data only
- keeps the test set isolated until final evaluation
- uses EfficientNetB0's built-in preprocessing without a second rescaling layer
- runs frozen-head training followed by upper-layer fine-tuning at `1e-5`
- selects the best stage by validation accuracy before using the test set
- saves macro precision, recall, F1, accuracy, reports, plots, and incorrect examples

Macro averaging is used for multiclass precision, recall, and F1 so every soil
class contributes equally regardless of class size.

## Running in Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload the dataset ZIP when prompted.
3. Run cells from top to bottom.
4. Keep generated models and reports outside Git unless they are intentionally
   reviewed and documented.

The workflow uses Colab paths such as `/content/soil_dataset` and expects the
uploaded archive to contain `train_data` and `test` directories. Until the
training cells are actually executed and the resulting model is reviewed, the
API correctly reports the image classifier as `not_trained`.

Do not copy printed accuracy, precision, recall, F1, or confusion-matrix values
into documentation unless they come from a completed run with a known dataset and
configuration.
