"""Week 3 soil image classification pipeline.

Run this file in Google Colab after setting DATASET_ROOT, or run it locally with
the same dependencies from requirements-colab.txt. The source dataset must have
class folders below train_data and test. No laboratory nutrient values are used.

The pipeline deliberately does not include SHAP, Grad-CAM, hybrid analysis,
structured-data models, API code, or soil-health scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
CLASS_NAMES = ("Alluvial Soil", "Black Soil", "Clay Soil", "Red Soil")


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def normalize_class_name(folder_name: str) -> str | None:
    name = folder_name.strip().lower()
    for key, label in (
        ("alluvial", "Alluvial Soil"),
        ("black", "Black Soil"),
        ("clay", "Clay Soil"),
        ("red", "Red Soil"),
    ):
        if key in name:
            return label
    return None


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def verify_source_dataset(dataset_root: Path) -> dict:
    """Verify files, remove duplicate content, and report class/split counts."""
    records: list[dict[str, str]] = []
    corrupted: list[str] = []
    duplicates: list[str] = []
    hashes: set[str] = set()

    for path in find_images(dataset_root):
        label = normalize_class_name(path.parent.name)
        parts = {part.lower() for part in path.parts}
        split = "test" if "test" in parts else "train" if "train_data" in parts else None
        if label is None or split is None:
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        except (OSError, UnidentifiedImageError):
            corrupted.append(str(path))
            continue
        if digest in hashes:
            duplicates.append(str(path))
            continue
        hashes.add(digest)
        records.append({"path": str(path), "label": label, "split": split})

    if not records:
        raise ValueError("No valid labelled train_data/test images were found")
    return {
        "valid_records": records,
        "corrupted": corrupted,
        "duplicates": duplicates,
        "class_distribution": dict(Counter(record["label"] for record in records)),
        "split_distribution": dict(Counter(record["split"] for record in records)),
    }


def prepare_dataset(verification: dict, output_root: Path) -> dict[str, int]:
    """Create RGB 224x224 train/validation/test folders without test leakage."""
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "validation", "test"):
        for label in CLASS_NAMES:
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    train_records = [r for r in verification["valid_records"] if r["split"] == "train"]
    test_records = [r for r in verification["valid_records"] if r["split"] == "test"]
    if not train_records or not test_records:
        raise ValueError("Both train_data and test splits are required")

    labels = [record["label"] for record in train_records]
    train_records, validation_records = train_test_split(
        train_records, test_size=0.2, random_state=SEED, stratify=labels
    )

    def copy_records(records: list[dict[str, str]], split: str) -> None:
        for index, record in enumerate(records):
            source = Path(record["path"])
            destination = output_root / split / record["label"] / f"{index:06d}.jpg"
            with Image.open(source) as image:
                image.convert("RGB").resize(IMG_SIZE).save(destination, format="JPEG", quality=95)

    copy_records(train_records, "train")
    copy_records(validation_records, "validation")
    copy_records(test_records, "test")
    return {
        split: sum(1 for _ in find_images(output_root / split))
        for split in ("train", "validation", "test")
    }


def load_datasets(dataset_root: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list[str]]:
    common = {"image_size": IMG_SIZE, "batch_size": BATCH_SIZE}
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / "train", shuffle=True, seed=SEED, **common
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / "validation", shuffle=False, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_root / "test", shuffle=False, **common
    )
    class_names = train_ds.class_names
    if class_names != list(CLASS_NAMES):
        raise ValueError(f"Unexpected class names: {class_names}; expected {list(CLASS_NAMES)}")
    autotune = tf.data.AUTOTUNE
    return (
        train_ds.prefetch(autotune),
        validation_ds.prefetch(autotune),
        test_ds.prefetch(autotune),
        class_names,
    )


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="training_augmentation",
    )
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    # tf.keras EfficientNetB0 includes its own rescaling layer and expects 0-255
    # image tensors. Adding layers.Rescaling(1./255) here would normalize twice.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


def plot_history(history: tf.keras.callbacks.History, output_dir: Path, prefix: str) -> None:
    for metric, title, filename in (
        ("accuracy", "Training and validation accuracy", "accuracy.png"),
        ("loss", "Training and validation loss", "loss.png"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.title(f"{prefix}: {title}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix.lower()}_{filename}", dpi=150)
        plt.close()


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    artifact_root: Path,
) -> dict:
    y_true: list[int] = []
    y_pred: list[int] = []
    images: list[np.ndarray] = []
    for batch_images, labels in test_ds:
        predictions = model.predict(batch_images, verbose=0)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        images.extend(batch_images.numpy())

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "precision_macro": float(precision_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true_array, y_pred_array, average="macro", zero_division=0)),
        "averaging": "macro: unweighted mean across the four soil classes",
    }
    report = classification_report(
        y_true_array, y_pred_array, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true_array, y_pred_array)
    artifact_root.joinpath("metrics").mkdir(parents=True, exist_ok=True)
    artifact_root.joinpath("plots").mkdir(parents=True, exist_ok=True)
    artifact_root.joinpath("predictions").mkdir(parents=True, exist_ok=True)
    artifact_root.joinpath("reports").mkdir(parents=True, exist_ok=True)
    (artifact_root / "metrics" / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    (artifact_root / "metrics" / "classification_report.json").write_text(json.dumps(report, indent=2))
    np.savetxt(artifact_root / "metrics" / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    with (artifact_root / "reports" / "classification_report.txt").open("w") as report_file:
        report_file.write(classification_report(y_true_array, y_pred_array, target_names=class_names, zero_division=0))
        report_file.write("\nMacro averaging is used for precision, recall, and F1.\n")
    ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(xticks_rotation=25)
    plt.tight_layout()
    plt.savefig(artifact_root / "plots" / "confusion_matrix.png", dpi=150)
    plt.close()

    wrong = np.flatnonzero(y_true_array != y_pred_array)
    with (artifact_root / "predictions" / "incorrect_predictions.csv").open("w") as file:
        file.write("index,actual,predicted\n")
        for index in wrong:
            file.write(f"{index},{class_names[y_true_array[index]]},{class_names[y_pred_array[index]]}\n")
    figure = plt.figure(figsize=(12, 9))
    for position, index in enumerate(wrong[:12]):
        axis = figure.add_subplot(3, 4, position + 1)
        axis.imshow(images[index].astype("uint8"))
        axis.set_title(f"Actual: {class_names[y_true_array[index]]}\nPredicted: {class_names[y_pred_array[index]]}")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(artifact_root / "predictions" / "incorrect_predictions.png", dpi=150)
    plt.close(figure)
    return metrics | {"test_samples": len(y_true_array), "incorrect_predictions": int(len(wrong))}


def run(dataset_root: Path, artifact_root: Path, stage1_epochs: int, stage2_epochs: int) -> dict:
    set_seed()
    verification = verify_source_dataset(dataset_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "reports" ).mkdir(exist_ok=True)
    (artifact_root / "reports" / "dataset_verification.json").write_text(json.dumps(verification, indent=2))
    prepared_root = artifact_root / "prepared_dataset"
    split_counts = prepare_dataset(verification, prepared_root)
    (artifact_root / "reports" / "split_counts.json").write_text(json.dumps(split_counts, indent=2))
    train_ds, validation_ds, test_ds, class_names = load_datasets(prepared_root)

    model, base_model = build_model(len(class_names))
    compile_model(model, 1e-3)
    stage1 = model.fit(train_ds, validation_data=validation_ds, epochs=stage1_epochs,
                       callbacks=callbacks(artifact_root / "models" / "stage1_best.keras"))
    plot_history(stage1, artifact_root / "plots", "stage1")

    # Fine-tune only the upper portion. BatchNorm remains frozen for stability.
    base_model.trainable = True
    freeze_until = int(len(base_model.layers) * 0.8)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    stage2 = model.fit(train_ds, validation_data=validation_ds, epochs=stage2_epochs,
                       callbacks=callbacks(artifact_root / "models" / "stage2_best.keras"))
    plot_history(stage2, artifact_root / "plots", "stage2")

    # Validation selects the final checkpoint; test is evaluated once afterward.
    stage1_best = max(stage1.history["val_accuracy"])
    stage2_best = max(stage2.history["val_accuracy"])
    best_path = artifact_root / "models" / ("stage2_best.keras" if stage2_best >= stage1_best else "stage1_best.keras")
    selected_model = tf.keras.models.load_model(best_path)
    selected_model.save(artifact_root / "models" / "soil_cnn_final.keras")
    metrics = evaluate_model(selected_model, test_ds, class_names, artifact_root)
    summary = {
        "selected_stage": "stage2" if stage2_best >= stage1_best else "stage1",
        "stage1_best_validation_accuracy": float(stage1_best),
        "stage2_best_validation_accuracy": float(stage2_best),
        "fine_tuning_improvement": float(stage2_best - stage1_best),
        "split_counts": split_counts,
        "metrics": metrics,
    }
    (artifact_root / "reports" / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_root / "reports" / "class_names.json").write_text(json.dumps(class_names, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default=os.getenv("DATASET_ROOT", "/content/soil_dataset"))
    parser.add_argument("--artifact-root", default=os.getenv("ARTIFACT_ROOT", "/content/artifacts"))
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(Path(args.dataset_root), Path(args.artifact_root), args.stage1_epochs, args.stage2_epochs), indent=2))
