# E4 — barrido de DBSCAN para una vectorización

Este módulo ejecuta un barrido de `eps` y `min_samples` sobre una
vectorización concreta y evalúa cada partición con el paquete `evaluation`.

## Uso

Desde la raíz del repositorio:

```bash
python -m E4.run_e4 \
  --vectorization /ruta/a/vectorizacion
```

La ruta de vectorización puede ser:

- una carpeta que contenga `X.npy` y `metadata.csv`, o
- la ruta directa al archivo `X.npy` (con `metadata.csv` en la misma carpeta).

Opciones útiles:

```bash
python -m E4.run_e4 \
  --vectorization /ruta/a/vectorizacion \
  --metric cosine \
  --min-samples 5,10,20 \
  --percentiles 10,20,30,40,50
```

## Salidas

Por defecto guarda los CSV en `E4/resultados/`:

- `e4_grid.csv`: todas las configuraciones evaluadas.
- `e4_best.csv`: mejor configuración según DBCV, ruido y tamaño del cluster mayor.
