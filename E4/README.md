# Experimento E4: sensibilidad a la métrica Minkowski

Este paquete evalúa cómo cambian la estructura de densidad y la calidad del
clustering cuando se usa DBSCAN con distintas distancias Minkowski y con o sin
normalización L2.

## Objetivo

Comparar los resultados con exponentes

- `p = 0.5`
- `p = 1.0`
- `p = 2.0`
- `p = inf`

y analizar si la normalización del espacio vectorial modifica la densidad
observada por DBSCAN.

## Requisitos

Antes de ejecutar este experimento, la carpeta `vectorizations` debe contener las
vectorizaciones generadas con archivos `X.npy` y `metadata.csv`.

Ejemplos de carpetas válidas:

- `vectorizations/message__all__minilm`
- `vectorizations/user__all__minilm-mean`

## Ejecución

Desde la raíz del proyecto:

```bash
python -m E4.run_e4 --vectorization message__all__minilm --max-samples 5000
```

También se puede indicar explícitamente el conjunto de exponentes:

```bash
python -m E4.run_e4 \
  --vectorization message__all__minilm \
  --p-values 0.5 1.0 2.0 inf \
  --min-samples 5 \
  --max-samples 5000
```

## Salida

El comando guarda dos CSVs en la carpeta `E4/resultados` por defecto:

- `e4_minkowski_detail.csv`: resultados completos para cada eps y cada
  combinación de `p` + `normalized`.
- `e4_minkowski_summary.csv`: mejor configuración por cada `p` y cada
  valor de `normalized`, usando `dbcv` como criterio principal.

## Métricas evaluadas

El experimento usa el módulo `evaluation` para calcular:

- `structure_summary` para ruido, número de clusters y fracción del cluster más
  grande.
- `internal_metrics` para DBCV, silhouette, Davies-Bouldin y Calinski-Harabasz.
- `external_metrics` para ARI y AMI frente a etiquetas del corpus.

## Nota importante

La evaluación puede ser costosa en memoria y tiempo si se usa todo el conjunto de
instancias. Por eso se recomienda usar `--max-samples` para reducir la muestra
sin perder la comparativa entre métricas.
