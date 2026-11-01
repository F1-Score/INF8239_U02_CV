"""LAB07 · Figuras de análisis a partir de los resultados guardados (no entrena).

- reports/f1_por_clase.png: F1 por clase del baseline denso y la CNN (8 épocas).
- reports/comparacion_epocas.csv y reports/comparacion_epocas.png: experimento
  propio que compara la corrida del laboratorio (8 épocas) con 20 épocas para
  comprobar si la CNN estaba subentrenada.

    uv run python scripts/lab07_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LAVANDA, LAVANDA_BORDE = "#A593D3", "#6B58A6"
MENTA, MENTA_BORDE = "#A8D8C2", "#4F9478"
TINTA, TINTA_2, REJILLA = "#0B0B0B", "#52514E", "#E1E0D9"
NOMBRES = {"dense": "Denso", "cnn": "CNN"}


def read_per_class(path: Path) -> dict[str, list[tuple[str, float]]]:
    rows: dict[str, list[tuple[str, float]]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["model"], []).append((row["class"], float(row["f1"])))
    return rows


def f1_chart() -> None:
    per_class = read_per_class(REPORTS / "per_class_metrics.csv")
    classes = [c for c, _ in per_class["dense"]]
    y = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for offset, model, fill, edge in [(-0.2, "dense", LAVANDA, LAVANDA_BORDE), (0.2, "cnn", MENTA, MENTA_BORDE)]:
        values = [v for _, v in per_class[model]]
        ax.barh(y + offset, values, height=0.38, color=fill, edgecolor=edge, label=NOMBRES[model], zorder=3)
        for i, v in enumerate(values):
            ax.text(v + 0.006, i + offset, f"{v:.2f}", va="center", fontsize=8, color=TINTA)
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 en prueba (10,000 imágenes)", color=TINTA_2)
    ax.grid(axis="x", color=REJILLA, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    ax.set_title("F1 por clase · baseline denso frente a CNN (8 épocas)", loc="left", fontsize=12, color=TINTA, pad=28)
    fig.tight_layout()
    fig.savefig(REPORTS / "f1_por_clase.png", dpi=200)
    plt.close(fig)


def epochs_comparison() -> None:
    runs = {8: REPORTS / "cv_metrics.json", 20: REPORTS / "experimento_e20" / "cv_metrics.json"}
    table = []
    for limit, path in runs.items():
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for model in ("dense", "cnn"):
            m = metrics[model]
            table.append({
                "max_epochs": limit, "model": model, "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"],
                "f1_macro": round(m["f1_macro"], 4), "accuracy": round(m["accuracy"], 4),
                "train_seconds": round(m["train_seconds"], 1),
                "inference_ms_per_image": round(m["inference_ms_per_image_median"], 4),
                "final_val_loss": round(metrics["histories"][model]["val_loss"][-1], 4),
            })
    with (REPORTS / "comparacion_epocas.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(2)
    for offset, model, fill, edge in [(-0.18, "dense", LAVANDA, LAVANDA_BORDE), (0.18, "cnn", MENTA, MENTA_BORDE)]:
        rows = [r for r in table if r["model"] == model]
        f1 = [r["f1_macro"] for r in rows]
        secs = [r["train_seconds"] for r in rows]
        a1.bar(x + offset, f1, width=0.34, color=fill, edgecolor=edge, label=NOMBRES[model], zorder=3)
        a2.bar(x + offset, secs, width=0.34, color=fill, edgecolor=edge, label=NOMBRES[model], zorder=3)
        for xi, v, r in zip(x, f1, rows):
            a1.text(xi + offset, v + 0.015, f"{v:.3f}", ha="center", fontsize=9, color=TINTA)
        for xi, v, r in zip(x, secs, rows):
            a2.text(xi + offset, v + 3, f"{v:.0f} s\n({r['epochs_run']} ép.)", ha="center", fontsize=9, color=TINTA)
    a1.set_ylim(0, 1.05)
    a1.set_title("F1 macro en prueba", loc="left", fontsize=11, color=TINTA)
    a2.set_ylim(0, max(r["train_seconds"] for r in table) * 1.25)
    a2.set_title("Tiempo de entrenamiento en CPU", loc="left", fontsize=11, color=TINTA)
    for ax in (a1, a2):
        ax.set_xticks(x, ["Límite 8 épocas", "Límite 20 épocas"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color=REJILLA, zorder=0)
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("LAB07 · ¿La CNN mejora con más épocas?", x=0.02, ha="left", fontsize=13, fontweight="bold", color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(REPORTS / "comparacion_epocas.png", dpi=200)
    plt.close(fig)
    for r in table:
        print(r)


def main() -> None:
    f1_chart()
    epochs_comparison()
    print("Guardado: reports/f1_por_clase.png, reports/comparacion_epocas.csv y reports/comparacion_epocas.png")


if __name__ == "__main__":
    main()
