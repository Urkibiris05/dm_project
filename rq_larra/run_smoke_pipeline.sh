#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-run_$(date +%Y%m%d_%H%M%S)_smoke}"
DISORDER="${2:-Depresion}"

BASE_DIR="rq_larra/csv/rq_baseline/${RUN_ID}"
mkdir -p "$BASE_DIR/audit" "rq_larra/figuras/rq_baseline/${RUN_ID}"

python3 supervised_splits/rama_b_relabeling/audit_splits.py \
  --input-dir particiones \
  --output-dir "$BASE_DIR/audit"

python3 supervised_splits/rama_b_relabeling/make_tfidf_cache.py \
  --train-csv particiones/messages_train.csv \
  --disorder "$DISORDER" \
  --components 64 \
  --random-state 2023 \
  --output "$BASE_DIR/embeddings_train_tfidf_${DISORDER}.npz"

python3 supervised_splits/rama_b_relabeling/run_relabeling.py \
  --train-csv particiones/messages_train.csv \
  --disorder "$DISORDER" \
  --embedding-cache "$BASE_DIR/embeddings_train_tfidf_${DISORDER}.npz" \
  --neighbors 10 \
  --min-control-neighbors 5 \
  --mode single_pass \
  --run-name "$RUN_ID"

cp -r "supervised_splits/rama_b_relabeling/reports/relabeling/${RUN_ID}" "$BASE_DIR/relabeling"

echo "run_id,rq,stage,status,notebook,seed,disorder,embeddings,relabel_neighbors,relabel_min_control,source_partitions,artifacts_dir,notes" > "$BASE_DIR/run_manifest_row.csv"
echo "${RUN_ID},rq_baseline,smoke,completed,,2023,${DISORDER},tfidf_svd_64,10,5,particiones,rq_larra/csv/rq_baseline/${RUN_ID},smoke_ok" >> "$BASE_DIR/run_manifest_row.csv"

echo "Done: $BASE_DIR"
