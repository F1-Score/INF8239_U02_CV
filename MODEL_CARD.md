# Model Card · Fashion-MNIST: baseline denso y CNN

## Modelo y versión
| Modelo | Archivo | Arquitectura | Parámetros |
|---|---|---|---:|
| Baseline denso (modelo de referencia) | `models/best_dense.keras` | Aplanado → Dense 64 ReLU → Dense 10 softmax | 50,890 |
| CNN pequeña | `models/best_cnn.keras` | Conv 32 (3×3) → MaxPool → Conv 64 (3×3) → MaxPool → promedio global → Dropout 0.25 → Dense 10 softmax | 19,466 |

Entrenados el 31 de octubre de 2026 por Emma Carolina Cabrera Valverde con el proyecto base del curso INF-8239. Entorno: Windows 11, Python 3.12.16, TensorFlow 2.21.0 en CPU, semilla 42, dependencias fijadas en `uv.lock`.

## Uso previsto
Material didáctico para comparar una red densa y una convolucional con la misma partición, y para analizar desempeño, errores y costo computacional en un benchmark de imágenes pequeñas.

## Usos fuera de alcance
- Clasificar fotografías reales de ropa, catálogos o inventarios.
- Imágenes médicas, documentos (OCR), vigilancia o cualquier dominio distinto.
- Decisiones sobre personas o compras.

## Dataset y particiones
Fashion-MNIST (Xiao, Rasul y Vollgraf, 2017; Zalando Research, licencia MIT), descargado por TensorFlow. 10 clases equilibradas.

| Partición | Imágenes | Origen |
|---|---:|---|
| Entrenamiento | 54,000 | Primeras 54,000 del conjunto de entrenamiento |
| Validación | 6,000 | Últimas 6,000 del conjunto de entrenamiento |
| Prueba | 10,000 | Conjunto de prueba oficial (1,000 por clase) |

## Preprocesamiento
Píxeles de 0–255 divididos entre 255 (rango [0, 1]) y un canal añadido (28 × 28 × 1). `validate_images` comprueba forma, rango y que imágenes y etiquetas coincidan.

## Métricas globales y por clase
Prueba, 10,000 imágenes, límite de 8 épocas (`reports/cv_metrics.json`, `reports/per_class_metrics.csv`):

| Métrica | Denso | CNN |
|---|---:|---:|
| F1 macro | 0.863 | 0.789 |
| Accuracy | 0.865 | 0.793 |
| Mejor clase (F1) | Pantalón 0.974 | Pantalón 0.959 |
| Peor clase (F1) | Camisa 0.652 | Camisa 0.427 |

Experimento con límite de 20 épocas (`reports/experimento_e20/`): denso 0.872, CNN 0.837 (la CNN seguía mejorando en la época 20).

## Comparación de costo
- **Hardware:** CPU de una computadora personal con Windows 11; sin GPU.
- **Parámetros:** denso 50,890; CNN 19,466 (2.6 veces menos).
- **Tamaño en disco:** denso 634 KB; CNN 268 KB.
- **Tiempo de entrenamiento (8 épocas):** denso 7.9 s; CNN 84.9 s.
- **Tiempo de inferencia:** denso 0.014 ms/imagen; CNN 0.080 ms/imagen (mediana de 5 repeticiones sobre las mismas 10,000 imágenes).

## Limitaciones y riesgos
- Las clases de prendas superiores se confunden mucho entre sí (Camisa, Camiseta/top, Suéter, Abrigo); en baja resolución algunos casos son ambiguos incluso para una persona.
- Una sola semilla: diferencias pequeñas entre modelos pueden no repetirse.
- Imágenes centradas, de fondo negro y una sola prenda; el desempeño en imágenes reales sería mucho menor.
- No se midió consumo energético ni uso de memoria.

## Supervisión y monitoreo
- Mostrar siempre las predicciones de Camisa con baja confianza para revisión humana.
- Si se usara en otro contexto, medir F1 por clase con datos de ese contexto antes de cualquier uso y repetir la medición cuando cambien los datos.
- Mantener el baseline denso como referencia: cualquier modelo nuevo debe superarlo con la misma partición y justificar su costo adicional.
