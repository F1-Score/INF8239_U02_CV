"""LAB07 · Baseline denso frente a CNN con Fashion-MNIST.

El entrenamiento es el del proyecto base (mismos modelos, partición 54,000 /
6,000 / 10,000, Adam, lotes de 128, EarlyStopping con paciencia 2, semilla 42).
Se agregó lo que pide el manual para comparar de forma justa y documentar:

- reports/data_audit.json: formas, rango de píxeles y distribución por clase.
- models/best_dense.keras y models/best_cnn.keras: mejor época de cada modelo.
- reports/cv_metrics.json: F1 macro, accuracy, parámetros, tamaño del archivo,
  tiempo de entrenamiento e inferencia (5 repeticiones sobre las mismas
  10,000 imágenes; se informa la mediana).
- reports/per_class_metrics.csv: precisión, recall y F1 por clase de ambos modelos.
- reports/confusion_cnn.png, reports/cnn_errors.png (16 errores elegidos al
  azar con semilla fija), reports/loss_curves.png y reports/f1_por_clase.png.

Con un número de épocas distinto de 8, los resultados se guardan en
reports/experimento_e<N>/ y models/experimento_e<N>/.

    uv run python scripts/train_cv.py --epochs 8
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_recall_fscore_support)

from inf8239_u02_cv.data import normalize_images, validate_images
from inf8239_u02_cv.models import build_cnn, build_dense

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TIMING_REPEATS = 5
CLASSES = ["Camiseta/top", "Pantalón", "Suéter", "Vestido", "Abrigo",
           "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"]
LAVANDA, LAVANDA_BORDE = "#A593D3", "#6B58A6"
MENTA, MENTA_BORDE = "#A8D8C2", "#4F9478"
TINTA, TINTA_2, REJILLA = "#0B0B0B", "#52514E", "#E1E0D9"

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def compile_model(model):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def distribution(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {CLASSES[int(v)]: int(c) for v, c in zip(values, counts)}


def timed_inference(model, images) -> tuple[np.ndarray, list[float]]:
    model.predict(images[:256], verbose=0)  # calentamiento: no se mide
    times = []
    for _ in range(TIMING_REPEATS):
        start = perf_counter()
        probabilities = model.predict(images, batch_size=256, verbose=0)
        times.append(perf_counter() - start)
    return probabilities, times


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=REJILLA, linewidth=0.8, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    reports = ROOT / "reports"
    models_dir = ROOT / "models"
    if args.epochs != 8:
        # Experimentos con otro número de épocas: carpetas aparte para no reemplazar la corrida del laboratorio.
        reports = reports / f"experimento_e{args.epochs}"
        models_dir = models_dir / f"experimento_e{args.epochs}"
    reports.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    (x_train_all, y_train_all), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    raw_range = [int(x_train_all.min()), int(x_train_all.max())]
    x_train_all = normalize_images(x_train_all)
    x_test = normalize_images(x_test)
    x_train, y_train = x_train_all[:-6000], y_train_all[:-6000]
    x_valid, y_valid = x_train_all[-6000:], y_train_all[-6000:]
    validate_images(x_train, y_train)
    validate_images(x_test, y_test)

    audit = {
        "train_shape": list(x_train.shape), "valid_shape": list(x_valid.shape), "test_shape": list(x_test.shape),
        "raw_pixel_range": raw_range,
        "normalized_pixel_range": [float(x_train.min()), float(x_train.max())],
        "classes": CLASSES,
        "train_distribution": distribution(y_train),
        "valid_distribution": distribution(y_valid),
        "test_distribution": distribution(y_test),
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__,
                        "platform": platform.platform(), "gpu": [d.name for d in tf.config.list_physical_devices("GPU")]},
    }
    (reports / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Auditoría:", audit["train_shape"], audit["valid_shape"], audit["test_shape"])

    metrics, predictions, histories, per_class = {}, {}, {}, []
    for name, factory in {"dense": build_dense, "cnn": build_cnn}.items():
        model = compile_model(factory(tf))
        path = models_dir / f"best_{name}.keras"
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(path, save_best_only=True),
        ]
        start = perf_counter()
        history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=args.epochs,
                            batch_size=128, callbacks=callbacks, verbose=2)
        train_seconds = perf_counter() - start
        probabilities, times = timed_inference(model, x_test)
        pred = probabilities.argmax(axis=1)
        median = float(np.median(times))
        metrics[name] = {
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, pred)),
            "parameters": int(model.count_params()),
            "model_size_bytes": path.stat().st_size,
            "epochs_run": len(history.history["loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "train_seconds": train_seconds,
            "inference_seconds_runs": [round(t, 4) for t in times],
            "inference_ms_per_image_median": 1000 * median / len(x_test),
        }
        histories[name] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        predictions[name] = pred
        p, r, f, s = precision_recall_fscore_support(y_test, pred, zero_division=0)
        per_class += [{"model": name, "class": CLASSES[i], "precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in range(len(CLASSES))]
        print(name, json.dumps(metrics[name], indent=2))
        print(classification_report(y_test, pred, target_names=CLASSES, digits=3, zero_division=0))

    metrics["histories"] = histories
    (reports / "cv_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    import csv
    with (reports / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in per_class)

    # Matriz de confusión de la CNN
    pastel = LinearSegmentedColormap.from_list("lavanda", ["#F7F5FC", LAVANDA, LAVANDA_BORDE])
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions["cnn"], display_labels=CLASSES, cmap=pastel,
                                            colorbar=False, ax=ax, xticks_rotation=45)
    ax.set_title(f"Matriz de confusión · CNN ({len(y_test):,} imágenes de prueba)", loc="left", fontsize=12, color=TINTA)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    fig.tight_layout()
    fig.savefig(reports / "confusion_cnn.png", dpi=200)
    plt.close(fig)

    # 16 errores de la CNN elegidos al azar (semilla fija), no los primeros del conjunto
    wrong = np.where(predictions["cnn"] != y_test)[0]
    chosen = np.sort(np.random.default_rng(SEED).choice(wrong, size=min(16, len(wrong)), replace=False))
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.8))
    for axis, index in zip(axes.ravel(), chosen):
        axis.imshow(x_test[index].squeeze(), cmap="gray")
        axis.set_title(f"#{index}\nReal: {CLASSES[y_test[index]]}\nCNN: {CLASSES[predictions['cnn'][index]]}",
                       fontsize=8, color=TINTA)
        axis.axis("off")
    fig.suptitle(f"16 errores de la CNN (muestra aleatoria de {len(wrong)})", x=0.02, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(reports / "cnn_errors.png", dpi=170)
    plt.close(fig)

    # Curvas de pérdida de ambos modelos
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name, title in [(axes[0], "dense", "Baseline denso"), (axes[1], "cnn", "CNN")]:
        h = histories[name]
        ep = np.arange(1, len(h["loss"]) + 1)
        ax.plot(ep, h["loss"], color=LAVANDA_BORDE, marker="o", markerfacecolor=LAVANDA, linewidth=2, label="Entrenamiento")
        ax.plot(ep, h["val_loss"], color=MENTA_BORDE, marker="s", markerfacecolor=MENTA, linewidth=2, label="Validación")
        ax.axvline(metrics[name]["best_epoch"], color=TINTA_2, linestyle=":", linewidth=1)
        ax.set_title(f"{title} · mejor época {metrics[name]['best_epoch']}", loc="left", fontsize=11, color=TINTA)
        ax.set_xlabel("Época", color=TINTA_2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        style(ax)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Pérdida (entropía cruzada)", color=TINTA_2)
    fig.tight_layout()
    fig.savefig(reports / "loss_curves.png", dpi=200)
    plt.close(fig)

    # F1 por clase: denso frente a CNN
    f1 = {m: [row["f1"] for row in per_class if row["model"] == m] for m in ("dense", "cnn")}
    y = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.barh(y - 0.2, f1["dense"], height=0.38, color=LAVANDA, edgecolor=LAVANDA_BORDE, label="Denso", zorder=3)
    ax.barh(y + 0.2, f1["cnn"], height=0.38, color=MENTA, edgecolor=MENTA_BORDE, label="CNN", zorder=3)
    for i in range(len(CLASSES)):
        ax.text(f1["dense"][i] + 0.005, i - 0.2, f"{f1['dense'][i]:.2f}", va="center", fontsize=8, color=TINTA)
        ax.text(f1["cnn"][i] + 0.005, i + 0.2, f"{f1['cnn'][i]:.2f}", va="center", fontsize=8, color=TINTA)
    ax.set_yticks(y, CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 en prueba", color=TINTA_2)
    ax.grid(axis="x", color=REJILLA, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.set_title("F1 por clase · baseline denso frente a CNN", loc="left", fontsize=12, color=TINTA, pad=28)
    fig.tight_layout()
    fig.savefig(reports / "f1_por_clase.png", dpi=200)
    plt.close(fig)
    print("Artefactos guardados en reports/ y models/")


if __name__ == "__main__":
    main()
