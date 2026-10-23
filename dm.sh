#!/usr/bin/env bash
# CLI unificada para los flujos del proyecto.

set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

usage() {
    cat <<'EOF'
Uso: ./dm.sh <comando> [opciones]

Vectorización:
  vectorize [tfidf|transformer|features] [opciones]
  vectorize-all

Experimentos y clustering:
  e0 [--rapido] [--for-the-impatient]
  e1 [--rapido] [--for-the-impatient]
  cluster-users
  cluster-messages [opciones] [--for-the-impatient]
  cluster-hierarchical [opciones]
  cluster-message-pca

Evaluación y asignación:
  evaluate-message-dbcv
  evaluate-message-candidates
  evaluate-shortlist
  validate-dbcv
  evaluate-e4 [opciones] [--for-the-impatient]
  evaluate-distances
  assign-test [opciones]
  prepare-assigner [opciones] (guarda DBSCAN BETO-STS entrenado en Train)
  assign --input mensajes.txt --model E8/modelos/modelo.joblib
  assign --help

Otros:
  word-doc-embeddings
  profile [opciones]
  profile-plots
  describe [opciones]
  tests [opciones de pytest]
  for-the-impatient [opciones de pytest] (alias: quick)
  help

Las opciones se pasan al ejecutor existente. Ejemplos:
  ./dm.sh vectorize tfidf --unit message --analyzer word
  ./dm.sh cluster-messages --all
  ./dm.sh evaluate-e4 --vectorization message__all__e5
  ./dm.sh assign-test --representation message__all__e5
  ./dm.sh e0 --for-the-impatient

El perfil --for-the-impatient usa --rapido para e0/e1 y submuestras de
1.000 instancias para cluster-messages/evaluate-e4. Las opciones explícitas
del usuario prevalecen sobre los valores de submuestreo predeterminados.
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 2
}

contains_arg() {
    local wanted="$1"
    shift
    local arg
    for arg in "$@"; do
        [[ "$arg" == "$wanted" ]] && return 0
    done
    return 1
}

if (($# == 0)); then
    usage
    exit 2
fi

command="$1"
shift
args=("$@")
module=""
script=""
accepts_args=0

case "$command" in
    -h|--help|help)
        usage
        exit 0
        ;;
    quick|for-the-impatient|tests|test)
        ;;
    vectorize)
        module="vectorization"
        accepts_args=1
        ;;
    e0)
        module="E0.run_e0"
        accepts_args=1
        ;;
    e1)
        module="E1.run_e1"
        accepts_args=1
        ;;
    cluster-users)
        module="E2.run_e2_user"
        ;;
    cluster-messages)
        module="E2.run_e2_messages"
        accepts_args=1
        ;;
    cluster-hierarchical)
        module="E2.experiment_message_hierarchical"
        accepts_args=1
        ;;
    cluster-message-pca)
        module="E2.experiment_message_pca"
        ;;
    evaluate-message-dbcv)
        module="E2.calculate_message_dbcv"
        ;;
    evaluate-message-candidates)
        module="E2.evaluate_message_candidates"
        ;;
    evaluate-shortlist)
        module="E2.validate_message_shortlist"
        ;;
    validate-dbcv)
        module="E2.validate_dbcv"
        ;;
    word-doc-embeddings)
        module="E2.generate_word_doc_embeddings"
        ;;
    evaluate-e4)
        module="E4.run_e4"
        accepts_args=1
        ;;
    evaluate-distances)
        module="E4.experiment_distances"
        ;;
    assign-test)
        module="E8.experiment_assignment"
        accepts_args=1
        ;;
      prepare-assigner)
          module="E8.assign_messages"
          args=(prepare "${args[@]}")
          accepts_args=1
          ;;
      assign)
          module="E8.assign_messages"
          args=(assign "${args[@]}")
          accepts_args=1
          ;;
    profile)
        module="dbscan_implementation.profiling.benchmark"
        accepts_args=1
        ;;
    profile-plots)
        module="dbscan_implementation.profiling.plots"
        ;;
    describe)
        module="resultados_analisis_descriptivo.analisis_descriptivo"
        accepts_args=1
        ;;
    vectorize-all)
        script="run_vectorizations.sh"
        ;;
    *)
        fail "comando desconocido: $command (usa './dm.sh help')."
        ;;
esac

if [[ "$command" == "quick" || "$command" == "for-the-impatient" ]]; then
    if ((${#args[@]} == 0)); then
        args=(-q dbscan_implementation/tests evaluation/tests \
            vectorization/tests E4/tests E8/tests)
    fi
    cd "$ROOT" || exit 1
    exec "$PYTHON" -m pytest "${args[@]}"
fi

if [[ "$command" == "tests" || "$command" == "test" ]]; then
    ((${#args[@]} > 0)) || args=(.)
    cd "$ROOT" || exit 1
    exec "$PYTHON" -m pytest "${args[@]}"
fi

if contains_arg --for-the-impatient "${args[@]}"; then
    filtered=()
    for arg in "${args[@]}"; do
        [[ "$arg" == "--for-the-impatient" ]] || filtered+=("$arg")
    done
    args=("${filtered[@]}")
    case "$command" in
        e0|e1)
            contains_arg --rapido "${args[@]}" || args+=(--rapido)
            ;;
        evaluate-e4)
            contains_arg --max-samples "${args[@]}" ||
                args+=(--max-samples 1000)
            contains_arg --min-samples "${args[@]}" ||
                args+=(--min-samples 5)
            ;;
        cluster-messages)
            contains_arg --sample-size "${args[@]}" ||
                args+=(--sample-size 1000)
            ;;
        *)
            fail "--for-the-impatient no está disponible para $command."
            ;;
    esac
fi

if ((${#args[@]} > 0)) && ((accepts_args == 0)); then
    if contains_arg -h "${args[@]}" || contains_arg --help "${args[@]}"; then
        printf '%s no define opciones de línea de comandos.\n' "$command"
        exit 0
    fi
    fail "$command no acepta opciones adicionales."
fi

cd "$ROOT" || exit 1
if [[ -n "$module" ]]; then
    exec "$PYTHON" -m "$module" "${args[@]}"
fi
exec bash "$ROOT/$script" "${args[@]}"
