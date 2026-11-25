# MEMORIA_AGENTE — Registro verificado del proyecto (rq_larra)

> **Propósito:** archivo de memoria persistente para consultar hechos **verificados directamente** en archivos y ejecuciones del repositorio, evitando suposiciones entre sesiones. Actualizar cada vez que se confirme un nuevo resultado, decisión del usuario, ruta o cambio de código.

---

## 1. Estructura verificada del repositorio y rutas canónicas

- **Raíz del proyecto:** `/home/larra/dm_project` (en Windows/WSL: `\\wsl.localhost\Ubuntu\home\larra\dm_project`).
- **Particiones canónicas (`particiones/`):**
  - `particiones/messages_train.csv` (19.805 filas de mensajes)
  - `particiones/messages_dev.csv` (5.452 filas de mensajes)
  - `particiones/messages_test.csv` (2.828 filas de mensajes — **CONGELADO, no usar para selección ni ajuste**)
  - `particiones/subjects_train.csv`, `particiones/subjects_dev.csv`, `particiones/subjects_test.csv`, `particiones/subjects.csv`
  - `particiones/manifest.json`: `seed=2023`, ratios `train=0.7`, `dev=0.2`, `test=0.1`, estratificado por `["trastorno", "label"]`, unidad `"subject_id within each disorder"`, `0=control`, `1=suffer`.
  - `particiones/split_summary.csv`: conteo exacto de sujetos por `(trastorno, label, partition)`:
    - `Ansiedad`: train (40 control, 65 suffer = 105), dev (11 control, 19 suffer = 30), test (6 control, 9 suffer = 15).
    - `Depresion`: train (116 control, 118 suffer = 234), dev (33 control, 34 suffer = 67), test (17 control, 17 suffer = 34).
    - `TCA`: train (134 control, 100 suffer = 234), dev (39 control, 29 suffer = 68), test (19 control, 14 suffer = 33).
  - **Columnas verificadas en `particiones/messages_train.csv`:**
    `subject_id,trastorno,source_partition,label,label_name,class_name,id_message,message,date,partition`
  - **Importante sobre `subject_id`:** un mismo `subject_id` puede aparecer en más de un trastorno (p. ej. `embedding_manifest.json` reporta 361 `subject_id` únicos en train frente a 105+234+234=573 pares `(trastorno, subject_id)`). Siempre agrupar/filtrar por `(trastorno, subject_id)` o dentro de cada trastorno por separado.
  - **Orden temporal:** ordenar por `date` ascendente y desempatar por `id_message` (hay timestamps repetidos).

- **Carpeta de trabajo experimental (`rq_larra/`):**
  - `rq_larra/rq.md` — especificación de RQ1 y RQ2 y protocolo común.
  - `rq_larra/ipynb/` — notebooks (`rq1.ipynb`, `rq2.ipynb`).
  - `rq_larra/csv/` — resultados tabulares por `{rq1, rq2, rq_baseline}/{run_id}/` + `index_runs.csv`.
  - `rq_larra/figuras/` — figuras por `{rq1, rq2, rq_baseline}/{run_id}/`.
  - `rq_larra/MEMORIA_COPILOT.md` — memoria previa dejada por Copilot.
  - `rq_larra/MEMORIA_AGENTE.md` — este archivo de memoria verificada.

---

## 2. Código existente en `supervised_splits/rama_b_relabeling/` (comportamiento exacto verificado)

Todos los scripts tienen como ruta por defecto `MODULE_DIR.parent / "resultados"`, carpeta que **NO existe** en el repo actual. Por tanto, **siempre** hay que pasar explícitamente `particiones/` por CLI o argumento.

1. **`audit_splits.py`**
   - CLI: `--input-dir particiones --output-dir <dir>`
   - Valida columnas (`subject_id, trastorno, label, label_name, id_message, message, date, partition`), consistencia `label`/`label_name`, ausencia de solapamiento de sujetos entre splits dentro de cada trastorno, duplicados y fechas.
   - Escribe `split_audit.csv` y `split_audit_checks.json` sin texto de mensajes.

2. **`relabeling.py`**
   - Implementa `relabel_messages(embeddings, labels, subject_ids, *, neighbors, min_control_neighbors, mode="single_pass"|"iterative", exclude_same_subject=False, max_neighbors_per_subject=None, max_rounds=None, block_size=128)`.
   - Similitud coseno exacta (normaliza filas L2). Excluye siempre el propio mensaje (`query_index`).
   - Asimetría estricta: **solo cambia `suffer` (1) -> `control` (0)** si al menos `K` (`min_control_neighbors`) de los `J` (`neighbors`) vecinos tienen etiqueta `0`. Los mensajes `control` nunca cambian.
   - En modo `iterative`, el grafo de vecinos se calcula una sola vez y las etiquetas se actualizan síncronamente por rondas hasta convergencia o `max_rounds`.

3. **`run_relabeling.py`**
   - CLI: `--train-csv particiones/messages_train.csv --disorder {Ansiedad,Depresion,TCA} --embedding-cache <.npz> --neighbors J --min-control-neighbors K --mode {single_pass,iterative} [--exclude-same-subject] [--max-neighbors-per-subject N] [--max-rounds R] --run-name <nombre>`
   - **Limitación/detalle verificado:** siempre escribe la salida en `supervised_splits/rama_b_relabeling/reports/relabeling/<run-name>/` (no tiene `--output-dir` en CLI) y falla si ya existe.
   - Genera `weak_message_labels.csv` (`subject_id, id_message, initial_label, weak_label, was_relabelled`), `round_history.csv`, `run_manifest.json`.
   - Requiere que el `.npz` tenga exactamente el mismo orden de filas `(subject_id, id_message)` que `messages_train.csv` filtrado por `--disorder`, más los campos `embeddings`, `subject_id`, `id_message`, `disorder` (y opcional `model_id`).

4. **`summarize_relabeling.py`**
   - CLI: `--train-csv particiones/messages_train.csv --run-dir <dir> --disorder <trastorno>`
   - Genera `summary_by_disorder.csv`, `summary_by_position.csv`, `summary_by_subject.csv` sin texto.

5. **`make_tfidf_cache.py`**
   - Genera caché `.npz` TF-IDF (1-2 gramas) + TruncatedSVD ajustado solo en `train` para un trastorno. Es solo para pruebas rápidas, **no** es el encoder del baseline.

6. **`make_beto_sts_embeddings.py`**
   - Modelo: `hiiamsid/sentence_similarity_spanish_es`, revisión `66ab46adac3910bb6ea6085b962a25e49513b981`, 768 dims, `normalize_embeddings=False`, sin filtro `min_tokens`.
   - CLI: `--data-dir particiones --output-dir <dir> --partitions train dev --batch-size 32 --device {cpu,cuda}`
   - Guarda `<partition>.npz` con `embeddings, subject_id, id_message, disorder, label, date, partition, model_id` y `embedding_manifest.json`. No sobrescribe si `output_dir` ya existe.
   - **GOTCHA CRÍTICO (línea 71):** `_load_encoder` llama a `SentenceTransformer(..., local_files_only=True)`. En un entorno nuevo sin caché local previa de Hugging Face (como Google Colab o un venv limpio), fallará al intentar cargar el modelo si no se descarga antes o se permite descarga online (`local_files_only=False`).

7. **`baseline_mlp.py`**
   - Lee `train.npz` y `dev.npz` de `--embedding-dir` y escribe en `--output-root/<run-name>/`.
   - Arquitectura MLP: `Linear(d, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 1)`.
   - Entrenamiento: `BCEWithLogitsLoss` (sin pesos de clase), `AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=128`, `max_epochs=50`, `patience=8` (early stopping por `macro_f1` en dev). Estandarización media/std calculada **solo en train** por `(trastorno, condición)`.
   - Semillas por defecto: `(11, 23, 37, 51, 79)`.
   - Para cada trastorno ejecuta 3 condiciones: `initial_labels`, `knn_include_same_subject`, `knn_exclude_same_subject` (`J=10, K=5`, `single_pass`).
   - Decisión online de sujeto: media acumulada cronológica de `P(suffer)` por mensaje; dispara alarma en el primer cruce `>= threshold` (si no cruza, predice `control` al final). El umbral y la mejor época se eligen en `dev` maximizando `macro_f1` de sujeto.
   - Artefactos que guarda por `(disorder, condition)`:
     - `weak_labels_include_same_subject.csv`, `weak_labels_exclude_same_subject.csv` y sus `round_history_*.csv`.
     - `seed-{seed}.npz` con los pesos de PyTorch (`0_weight`, `0_bias`, `3_weight`, `3_bias`), `scaler_mean`, `scaler_scale`, `threshold`, `best_epoch`.
     - `seed_metrics.csv`, `dev_subject_predictions.csv`.
     - Globales: `dev_seed_metrics.csv`, `dev_summary.csv`, `run_manifest.json`.
   - **LIMITACIONES CRÍTICAS para RQ1 y RQ2:**
     1. En línea 511 hace `details.pop("dev_probabilities")`: **NO guarda las probabilidades ni los logits por mensaje**. Sin embargo, como sí guarda `seed-{seed}.npz` (pesos + `scaler_mean` + `scaler_scale`), los logits y probabilidades por mensaje en `dev` (y en `test` cuando se congele) se pueden **reconstruir exactamente** aplicando el scaler y el MLP guardado sobre `dev.npz` / `test.npz` (o exportándolos directamente).
     2. `baseline_mlp.py` solo carga `train.npz` y `dev.npz`; no tiene modo de inferencia congelada sobre `test.npz`.

---

## 3. Estado real en disco observado (2026-11-25)

- **Artefactos previos en `supervised_splits/rama_b_relabeling/reports/`:**
  - Existen únicamente los resúmenes ligeros (`.csv` y `.json`) de `baseline-beto-sts-dev-j10-k5-5seeds` y `beto_sts_full_baseline/embedding_manifest.json`.
  - **NO existen archivos `.npz` en el repositorio** (`train.npz`, `dev.npz`, `seed-*.npz`, `vectorizations/`, `vectorizations2/` no están presentes en disco; `.gitignore` ignora `.venv/`, `vectorizations/`, `vectorizations2/`, `.cache/`).
- **Artefactos previos en `rq_larra/`:**
  - Carpetas creadas: `csv/{_tmp,rq1,rq2,rq_baseline}`, `figuras/{rq1,rq2,rq_baseline}`, `ipynb/`.
  - `rq_larra/csv/index_runs.csv` tiene una fila de estado `blocked_env` (`run_20261125_205316_smoke`).
  - `rq_larra/ipynb/rq1.ipynb` tiene un intento fallido de ejecución en Google Colab (`ROOT = Path('/home/larra/dm_project')` no contenía el repo en la VM de Colab -> `[Errno 2] No such file or directory`).
  - `rq_larra/ipynb/rq2.ipynb` es un borrador con un bug: apunta `INPUT_CSV` a `dev_subject_predictions.csv` e intenta leer la columna `message_prob`, que no existe en ese CSV.
- **Entorno de ejecución observado:**
  - En Windows sandbox (`run_command`), `wsl` devuelve `Wsl/E_ACCESSDENIED`.
  - Python instalado en Windows: `C:\Users\PC\AppData\Local\Programs\Python\Python314\python.exe` (Python 3.14.0, solo `pip 25.2`, sin `numpy`, `pandas`, `scikit-learn`, `torch` ni `sentence-transformers`).

---

## 4. Contrato metodológico y anti-leakage (de `rq.md` y `rama_b_relabeling`)

1. Separación estricta por usuario dentro de cada trastorno (`train` / `dev` / `test` desde `particiones/`).
2. Relabeling k-NN únicamente con `train` y por trastorno separado.
3. Estandarización, calibradores (Temperature / Platt / Isotonic), selección de umbrales, hiperparámetros de agregación temporal y modelos de alarma (Logistic/Ridge sobre características temporales) ajustados **solo en `train`/`dev`**, nunca en `test`.
4. `test` permanece **congelado** hasta tener protocolo, hiperparámetros y umbrales cerrados en `dev`.
5. No usar información futura en prefijos temporales de RQ2.
6. Las pseudoetiquetas k-NN y las etiquetas heredadas por mensaje son **supervisión débil**, nunca gold labels clínicas de mensaje. No reportar métricas por mensaje como si fueran gold labels clínicas.
7. No volcar texto crudo de mensajes en informes, CSVs de resultados ni manifiestos.
8. Nunca sobrescribir corridas previas: usar `run_id` único y registrar en `rq_larra/csv/index_runs.csv`.

---

## 5. Decisiones confirmadas por el usuario (2026-11-25)

1. **Entorno y persistencia:** ejecución en **Google Colab conectado a Google Drive**. Los artefactos pesados (embeddings `.npz` y checkpoints MLP `seed-*.npz`) se guardan en una carpeta persistente en Drive (`STORAGE_ROOT`, fuera de Git o ignorados por `.gitignore`), mientras que los CSVs de resultados, figuras, manifiestos e `index_runs.csv` se guardan en `rq_larra/csv/` y `rq_larra/figuras/` (sincronizados/exportables).
2. **Estructura de notebooks (3 notebooks en `rq_larra/ipynb/`):**
   - `00_baseline_relabeling.ipynb`: auditoría de `particiones/`, generación/caché de embeddings BETO-STS, relabeling k-NN, entrenamiento del baseline MLP, guardado de checkpoints y exportación de logits/probabilidades por mensaje (`dev_message_predictions.csv`).
   - `rq1.ipynb`: análisis RQ1 (umbrales fijos, confianza, entropía normalizada, calibración Temperature/Platt/Isotonic, clasificación selectiva y curvas coverage vs selective risk).
   - `rq2.ipynb`: análisis RQ2 (agregación temporal causal: mean pooling, moving window, EWMA, consecutive threshold, características de tendencia + Logistic/Ridge Regression, evaluación con Macro-F1, ERDE5, ERDE30 y latencia).
3. **Alcance del baseline:** los **3 trastornos** (`Ansiedad`, `Depresion`, `TCA`) con BETO-STS + MLP (`J=10, K=5`, `single_pass`), comparando `initial_labels` y `knn_include_same_subject` (`exclude_same_subject=False`). **NO ejecutar `knn_exclude_same_subject`** (pertenece a las RQs de otro compañero).
4. **Nivel de evaluación principal en RQ1:** principalmente **nivel mensaje** usando la etiqueta heredada del sujeto como proxy débil (documentando explícitamente que no son gold labels clínicas de mensaje).

---

## 6. Implementación creada en `rq_larra/` (2026-11-25)

- `rq_larra/rq_utils.py`:
  - Detección automática de raíz (`PROJECT_ROOT`) y carpeta persistente en Google Drive (`STORAGE_ROOT = /content/drive/MyDrive/dm_project_artifacts` en Colab, o `rq_larra/artifacts` en local).
  - `ensure_beto_sts_embeddings`: envuelve `make_beto_sts_embeddings.encode_partitions` permitiendo descarga online fallback cuando el runtime de Colab no tiene el modelo pre-cacheado.
  - `run_baseline_and_export`: ejecuta `relabeling.py` (`exclude_same_subject=False`, `J=10, K=5`) y `baseline_mlp.py` para `initial_labels` y `knn_include_same_subject` en `Ansiedad`, `Depresion` y `TCA` (5 semillas `11, 23, 37, 51, 79`), guarda checkpoints `seed-*.npz` en `STORAGE_ROOT` y exporta `dev_message_predictions.csv` (ordenado cronológicamente con `logit` y `prob`, sin texto).
  - `save_latest_baseline_pointer` / `load_latest_baseline_pointer`: puntero `rq_larra/csv/rq_baseline/latest_baseline.json` para conectar `00_baseline_relabeling.ipynb` con `rq1.ipynb` y `rq2.ipynb`.
  - Helpers RQ1: `compute_binary_metrics`, `compute_ece`, `compute_calibration_metrics`, `fit_subject_oof_calibrators` (OOF agrupado por `subject_id` en `dev` para Temperature, Platt e Isotonic), `compute_entropy_norm`, `evaluate_selective_thresholds`, `compute_risk_coverage_curve`.
  - Helpers RQ2: `compute_erde` (ERDE5 y ERDE30 oficiales), `build_prefix_features_df` (estrictamente causal), `evaluate_alarm_rule`, `fit_predict_oof_temporal_alarm` (Logistic y Ridge agrupados por `subject_id` con peso `1 / subject_n_messages`).
- `rq_larra/ipynb/00_baseline_relabeling.ipynb`, `rq_larra/ipynb/rq1.ipynb`, `rq_larra/ipynb/rq2.ipynb`: listos para Colab + Google Drive, con evaluación `test` bloqueada tras `RUN_FROZEN_TEST = False`.
- `.gitignore`: añadidos `rq_larra/artifacts/`, `rq_larra/**/*.npz`, `rq_larra/**/*.pt`.
