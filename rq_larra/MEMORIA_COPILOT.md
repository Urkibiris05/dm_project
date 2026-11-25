# MEMORIA_COPILOT

## Objetivo operativo
- Ejecutar RQ1 y RQ2 desde notebooks en `rq_larra/ipynb`.
- Usar particiones canónicas desde `particiones/`.
- Guardar resultados por `run_id` en `rq_larra/csv` y `rq_larra/figuras`.

## Rutas canónicas
- Particiones: `particiones/messages_train.csv`, `particiones/messages_dev.csv`, `particiones/messages_test.csv`
- Scripts base: `supervised_splits/rama_b_relabeling/`
- Notebooks: `rq_larra/ipynb/rq1.ipynb`, `rq_larra/ipynb/rq2.ipynb`
- Salidas: `rq_larra/csv/{rq1,rq2,rq_baseline}/{run_id}` y `rq_larra/figuras/{rq1,rq2,rq_baseline}/{run_id}`

## Contrato anti-leakage
- Split por usuario (`subject_id`) y trastorno.
- Relabeling solo con train.
- Seleccion de hiperparametros solo con dev.
- Test congelado para evaluacion final.

## Estado actual
- Estructura de carpetas creada para RQ y baseline.
- Corrida smoke inicial creada: `run_20261125_205316_smoke`.
- Bloqueo actual: entorno local sin `pip`/`ensurepip`; no se pudieron instalar dependencias para ejecutar scripts Python.

## Comandos de ejecucion (cuando el entorno tenga dependencias)
```bash
python3 supervised_splits/rama_b_relabeling/audit_splits.py \
  --input-dir particiones \
  --output-dir rq_larra/csv/rq_baseline/${RUN_ID}/audit

python3 supervised_splits/rama_b_relabeling/make_tfidf_cache.py \
  --train-csv particiones/messages_train.csv \
  --disorder Depresion \
  --components 64 \
  --random-state 2023 \
  --output rq_larra/csv/rq_baseline/${RUN_ID}/embeddings_train_tfidf_dep.npz

python3 supervised_splits/rama_b_relabeling/run_relabeling.py \
  --train-csv particiones/messages_train.csv \
  --disorder Depresion \
  --embedding-cache rq_larra/csv/rq_baseline/${RUN_ID}/embeddings_train_tfidf_dep.npz \
  --neighbors 10 \
  --min-control-neighbors 5 \
  --mode single_pass \
  --run-name ${RUN_ID}
```

## Decisiones fijas
- `run_id` obligatorio por corrida.
- Registrar cada corrida en `rq_larra/csv/index_runs.csv`.
- No sobrescribir artefactos previos.
