# LAB07 · Baseline denso frente a CNN en Fashion-MNIST

## Protocolo
- Datos: Fashion-MNIST, 54,000 imágenes de entrenamiento, 6,000 de validación y 10,000 de prueba (1,000 por clase), 28 × 28 en escala de grises, píxeles en [0, 1] (`data_audit.json`).
- Ambos modelos usan la misma partición, Adam, lotes de 128, como máximo 8 épocas, `EarlyStopping` (paciencia 2, mejores pesos) y semilla 42.
- **Denso:** aplanado → 64 neuronas ReLU → 10 salidas softmax.
- **CNN:** dos convoluciones (32 y 64 filtros 3 × 3) con max pooling, promedio global, dropout 0.25 y 10 salidas softmax.
- Hardware: CPU (Windows 11, Python 3.12.16, TensorFlow 2.21.0, sin GPU).

## Resultados (8 épocas, `cv_metrics.json`)

| Indicador | Denso | CNN |
|---|---:|---:|
| F1 macro | **0.863** | 0.789 |
| Accuracy | **86.5 %** | 79.3 % |
| Parámetros | 50,890 | **19,466** |
| Tamaño del archivo `.keras` | 634 KB | **268 KB** |
| Tiempo de entrenamiento | **7.9 s** | 84.9 s |
| Inferencia (mediana de 5 repeticiones, 10,000 imágenes) | **0.014 ms/imagen** | 0.080 ms/imagen |
| Pérdida de validación final | **0.354** | 0.587 |

**La CNN no superó al baseline.** Con ocho épocas obtuvo 7.4 puntos menos de F1 macro, tardó casi 11 veces más en entrenar y es unas 6 veces más lenta en inferencia. Su única ventaja de costo es el tamaño: tiene 2.6 veces menos parámetros porque el promedio global reemplaza a las capas densas grandes.

## Entrenamiento y validación (`loss_curves.png`)
En el modelo denso ambas pérdidas bajan y quedan muy cerca (entrenamiento 0.313, validación 0.354): aprendió sin sobreajuste visible. En la CNN las dos pérdidas siguen bajando con fuerza en la octava época (validación 0.635 → 0.611 → 0.587) y la pérdida de entrenamiento es todavía mayor que la de validación por efecto del dropout. Es el patrón de **subajuste**: el modelo no terminó de aprender en el número de épocas fijado.

## Experimento propio: 20 épocas (`comparacion_epocas.csv`)
Hipótesis: si la CNN estaba subentrenada, más épocas deberían cerrar la brecha. Se repitió el entrenamiento con un límite de 20 épocas y los resultados se guardaron aparte (`reports/experimento_e20/`).

| Límite | Modelo | Épocas usadas | F1 macro | Entrenamiento |
|---|---|---:|---:|---:|
| 8 | Denso | 8 | 0.863 | 7.9 s |
| 8 | CNN | 8 | 0.789 | 84.9 s |
| 20 | Denso | 16 (mejor: 14) | 0.872 | 14.4 s |
| 20 | CNN | 20 (mejor: 20) | 0.837 | 216.1 s |

La CNN ganó 4.8 puntos de F1 y su pérdida de validación siguió bajando hasta la época 20, así que la hipótesis se confirma en parte: estaba subentrenada. Aun así no alcanzó al modelo denso, que también mejoró un poco y se detuvo solo en la época 16. Con 20 épocas, la CNN supera al denso únicamente en Suéter (F1 0.783 frente a 0.766).

## Clase más difícil y errores (`confusion_cnn.png`, `cnn_errors.png`)
La clase más difícil para ambos modelos es **Camisa** (F1 0.43 en la CNN y 0.65 en el denso). En la matriz de la CNN, 274 camisas se predijeron como Camiseta/top, 132 como Abrigo y 123 como Suéter. Otras confusiones frecuentes: Suéter → Abrigo (155) y Suéter → Camisa (142), Abrigo → Camisa (122) y, en calzado, Sandalia → Zapatilla (88) y Botín → Zapatilla (70).

En los 16 errores elegidos al azar (semilla 42), 8 son camisas reales. Comentarios:
- **Camisa → Camiseta/top** (5 casos, por ejemplo #971 y #9692): a 28 × 28 píxeles el cuello y los botones casi no se ven; queda una silueta de manga corta igual a la de una camiseta.
- **Suéter ↔ Abrigo** (#4255, #6511): prendas de manga larga con la misma silueta; la diferencia está en la textura o el cierre, que se pierden en baja resolución.
- **Vestido → Camisa** (#7342, #7646): estampados recargados y siluetas largas parecidas a una camisa larga.
- **Camiseta/top → Bolso** (#930) y **Camisa → Bolso** (#7910): formas anchas sin mangas visibles.
- No hubo errores de calzado en la muestra; las confusiones entre calzado existen (Sandalia → Zapatilla) pero son menos frecuentes que entre prendas superiores.

Muchos errores son ambiguos incluso para una persona: el problema está en la resolución y en clases que se solapan, no solo en el modelo.

## Costo comparado y decisión (Green AI)
Con el protocolo del laboratorio, el modelo denso es mejor en desempeño y en tiempo; la CNN solo es más pequeña en disco (268 KB frente a 634 KB), lo que no compensa su peor F1 ni su inferencia 6 veces más lenta. Con 20 épocas la brecha baja a 3.5 puntos, pero el costo de entrenamiento de la CNN sube a 216 s (15 veces el del denso). **La mejora de la CNN no justifica su costo en este experimento.** Se mantiene el baseline denso como modelo de referencia.

## Limitaciones
- Una sola semilla y una sola partición; las diferencias pequeñas pueden cambiar con otra inicialización.
- Los tiempos son de una computadora con CPU y varían entre equipos y ejecuciones.
- La CNN del proyecto base es deliberadamente pequeña y no se ajustaron sus hiperparámetros; una CNN más grande o con más épocas probablemente superaría al denso, a mayor costo.
- Fashion-MNIST es un benchmark didáctico: imágenes centradas, fondo negro y una sola prenda. Un buen resultado aquí no demuestra que el modelo sirva para fotografías reales, imágenes médicas, OCR ni vigilancia.

## Decisión antes de usar otro dominio
No se usaría ninguno de los dos modelos fuera de este benchmark. Antes de otro dominio habría que reunir y etiquetar imágenes de ese dominio, repetir la comparación con varias semillas, medir el desempeño por clase y revisar los errores con personas que conozcan el dominio.
