# INF-8239 · Unidad 02 · Proyecto de visión (LAB07)

Proyecto base del curso Ciencia de Datos II (autor del proyecto base: Edwin Ramón José Nolasco), ejecutado y documentado por **Emma Carolina Cabrera Valverde**. Compara un baseline denso y una red convolucional (CNN) sobre Fashion-MNIST con la misma partición, y analiza desempeño, errores y costo.

## Instalación (CPU)

```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check_runtime.py
```

La ruta GPU (`uv sync --extra gpu`) solo aplica en WSL2/Linux con NVIDIA configurada. Este proyecto no comparte entorno con el de NLP.

## Datos

Fashion-MNIST (Zalando Research, licencia MIT): 70,000 imágenes de prendas en escala de grises, 28 × 28 píxeles, 10 clases. TensorFlow lo descarga a su caché. Partición: 54,000 entrenamiento, 6,000 validación (las últimas del conjunto de entrenamiento) y 10,000 prueba. Píxeles normalizados a [0, 1].

## Ejecución

```bash
uv run python scripts/train_cv.py --epochs 8      # corrida del laboratorio
uv run python scripts/train_cv.py --epochs 20     # experimento propio (carpetas experimento_e20/)
uv run python scripts/lab07_figures.py            # F1 por clase y comparación de épocas
uv run pytest -q
```

| Archivo | Contenido |
|---|---|
| `reports/data_audit.json` | Formas, rango de píxeles, distribución por clase y entorno |
| `reports/cv_metrics.json` | F1 macro, accuracy, parámetros, tamaño, tiempos e historial de cada modelo |
| `reports/per_class_metrics.csv` | Precisión, recall y F1 por clase de ambos modelos |
| `reports/confusion_cnn.png` | Matriz de confusión de la CNN |
| `reports/cnn_errors.png` | 16 errores de la CNN elegidos al azar con semilla fija |
| `reports/loss_curves.png` | Pérdida de entrenamiento y validación de ambos modelos |
| `reports/f1_por_clase.png` | F1 por clase, denso frente a CNN |
| `reports/comparacion_epocas.csv`, `reports/comparacion_epocas.png` | Límite de 8 frente a 20 épocas |
| `reports/experimento_e20/` | Resultados completos de la corrida con 20 épocas |
| `reports/lab07_analisis.md` | Interpretación de resultados, errores y costo |
| `models/best_dense.keras`, `models/best_cnn.keras` | Mejor época de cada modelo |

El entrenamiento es el del proyecto base (Adam, lotes de 128, `EarlyStopping` con paciencia 2, semilla 42). El tiempo de inferencia se mide 5 veces sobre las mismas 10,000 imágenes y se informa la mediana. Los tiempos dependen del equipo.

## Resultados principales

| | Denso | CNN |
|---|---:|---:|
| F1 macro (8 épocas) | **0.863** | 0.789 |
| F1 macro (límite 20 épocas) | **0.872** | 0.837 |
| Parámetros | 50,890 | **19,466** |
| Entrenamiento (8 épocas, CPU) | **7.9 s** | 84.9 s |
| Inferencia | **0.014 ms/imagen** | 0.080 ms/imagen |

La CNN pequeña del proyecto base no supera al baseline denso y cuesta más tiempo; la clase más difícil para ambos es Camisa.

## Documentación

- `MODEL_CARD.md`: uso previsto, datos, métricas, costo, limitaciones y supervisión.
- `AI_USE_DECLARATION.md`: asistencia de IA y verificaciones.

## Referencia

Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms*. arXiv:1708.07747.
