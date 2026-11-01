# Declaración de uso de IA · Proyecto de visión (LAB07)

**Estudiante:** Emma Carolina Cabrera Valverde
**Asignatura:** INF-8239 · Ciencia de Datos II
**Herramienta:** Claude (Anthropic), asistente conversacional con acceso a los archivos del proyecto

## Procedencia del código
- El proyecto base (`src/inf8239_u02_cv`, `check_runtime.py`, `train_cv.py`, pruebas, `pyproject.toml` y `uv.lock`) es material del profesor Edwin Ramón José Nolasco, descargado del aula virtual. Los modelos y el procedimiento de entrenamiento no se modificaron.
- Se utilizó como referencia el material compartido en el curso; la ejecución, los resultados y la redacción de esta entrega son propios.

## En qué ayudó la IA
1. Ampliar `train_cv.py` para guardar la auditoría de datos, ambos modelos, métricas por clase, tamaño de archivo, tiempo de inferencia repetido 5 veces, historial de pérdidas, matriz de confusión con nombres de clase, 16 errores elegidos al azar con semilla fija y gráficos en colores pastel; y para guardar en carpetas aparte las corridas con un número de épocas distinto de 8.
2. Proponer el experimento de 20 épocas para comprobar si la CNN estaba subentrenada, y escribir `lab07_figures.py` para comparar ambas corridas.
3. Redactar borradores del README, Model Card, análisis (`reports/lab07_analisis.md`) e informe Word a partir de mis resultados.

## Qué hice y verifiqué yo
- Ejecuté en mi computadora la verificación del entorno, el entrenamiento con 8 y con 20 épocas, las figuras y las pruebas (3 aprobadas).
- Revisé la matriz de confusión y los 16 errores, y contrasté las cifras de los documentos con `cv_metrics.json` y `per_class_metrics.csv`.

## Límites
La IA probó los scripts con datos de prueba en un entorno aparte, pero todos los resultados reportados provienen de las ejecuciones en mi computadora. Asumo la responsabilidad del código, de la verificación de los resultados y de las conclusiones.
