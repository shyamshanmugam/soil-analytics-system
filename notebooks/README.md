# Week 3 notebooks

`week3_soil_dataset_and_efficientnet.py` and
`week3_soil_dataset_and_efficientnet.ipynb` are the supplied Infosys project
workflow. They are kept together so the notebook and exported Colab script remain
consistent.

The workflow performs:

- ZIP upload and extraction in Google Colab
- soil-class discovery and normalization
- corruption and duplicate checks
- RGB conversion and 224x224 resizing
- train/validation/test directory creation
- EfficientNetB0 transfer-learning setup
- training, evaluation, confusion matrix, and artifact export

## Running in Colab

1. Open the `.ipynb` file in Google Colab.
2. Upload the dataset ZIP when prompted.
3. Run cells from top to bottom.
4. Keep generated models and reports outside Git unless they are intentionally
   reviewed and documented.

The attached workflow uses Colab paths such as `/content/soil_dataset` and expects
the uploaded archive to be named `archive.zip`. It is not imported by the API
automatically. Until the training cells are actually executed and the resulting
model is reviewed, the API correctly reports the image classifier as
`not_trained`.

Do not copy printed accuracy, precision, recall, F1, or confusion-matrix values
into documentation unless they come from a completed run with a known dataset and
configuration.
