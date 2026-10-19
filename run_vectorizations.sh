#!/usr/bin/env bash
# Lanza las vectorizaciones del corpus MentalRiskES y anota cuánto tarda cada
# una en vectorizations/times.txt.
#
# Uso (desde WSL, con el entorno activado):  bash run_vectorizations.sh
#
# Cada vectorización es UNA línea que llama a run_vectorization (sin bucles y
# sin "\" de continuación). Para saltarte una, comenta esa línea con "#": las
# demás se ejecutan igual. Si una falla, se anota FALLO y el script sigue.
#
# Los tiempos son de pared e incluyen cargar el corpus y el modelo. Los
# embeddings se guardan en .cache/, así que una repetición con la caché llena
# es mucho más rápida. Para medir en frío, borra .cache/ o añade --no-cache
# a la línea correspondiente.
#
# Opciones de cada método: python -m vectorization <tfidf|transformer|
# features> --help

cd "$(dirname "$0")" || exit 1

# Variables de entorno opcionales (útiles en Colab, ver el README del equipo):
#   PYTHON                interprete a usar (def. python)
#   OUT_DIR               carpeta de salida (def. vectorizations)
#   TIMES_FILE            archivo de tiempos (def. $OUT_DIR/times.txt)
#   VECTORIZATION_ARGS    opciones extra para TODAS las líneas, por ejemplo
#                         "--device cuda --cache-dir /ruta/cache"
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-vectorizations}"
TIMES_FILE="${TIMES_FILE:-$OUT_DIR/times.txt}"

mkdir -p "$OUT_DIR"
if [ ! -f "$TIMES_FILE" ]; then
    printf '# fecha\tvectorizacion\tsegundos\testado\n' > "$TIMES_FILE"
fi

# Ejecuta una vectorización y añade una línea a times.txt con la fecha, el
# nombre de la carpeta creada, los segundos que ha tardado y OK o FALLO.
# Los argumentos son los de "python -m vectorization".
run_vectorization() {
    local name start end seconds status
    if ! name=$("$PYTHON" -m vectorization "$@" --print-name); then
        printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" \
            "argumentos invalidos: $*" "0" "FALLO" >> "$TIMES_FILE"
        return 0
    fi
    echo ">>> $name"
    start=$(date +%s.%N)
    # shellcheck disable=SC2086  # VECTORIZATION_ARGS se separa por palabras
    if "$PYTHON" -m vectorization "$@" --output-dir "$OUT_DIR" --overwrite $VECTORIZATION_ARGS; then
        status="OK"
    else
        status="FALLO"
    fi
    end=$(date +%s.%N)
    seconds=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.1f", e - s }')
    printf '%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$name" \
        "$seconds" "$status" >> "$TIMES_FILE"
    return 0
}

# --- E1: TF-IDF (palabra y caracteres), por mensaje y por usuario ---
# Una vectorización por trastorno: el vocabulario y la SVD se ajustan solo
# con los textos de ese trastorno.
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer word
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer char_wb
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer word
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer char_wb
run_vectorization tfidf --disorders Depresion --unit message --analyzer word
run_vectorization tfidf --disorders Depresion --unit message --analyzer char_wb
run_vectorization tfidf --disorders Depresion --unit user --analyzer word
run_vectorization tfidf --disorders Depresion --unit user --analyzer char_wb
run_vectorization tfidf --disorders TCA --unit message --analyzer word
run_vectorization tfidf --disorders TCA --unit message --analyzer char_wb
run_vectorization tfidf --disorders TCA --unit user --analyzer word
run_vectorization tfidf --disorders TCA --unit user --analyzer char_wb
# E1: barrido de la dimension de la SVD (la de 128 son las 12 lineas de arriba)
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer word --svd 32
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer char_wb --svd 32
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer word --svd 32
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer char_wb --svd 32
run_vectorization tfidf --disorders Depresion --unit message --analyzer word --svd 32
run_vectorization tfidf --disorders Depresion --unit message --analyzer char_wb --svd 32
run_vectorization tfidf --disorders Depresion --unit user --analyzer word --svd 32
run_vectorization tfidf --disorders Depresion --unit user --analyzer char_wb --svd 32
run_vectorization tfidf --disorders TCA --unit message --analyzer word --svd 32
run_vectorization tfidf --disorders TCA --unit message --analyzer char_wb --svd 32
run_vectorization tfidf --disorders TCA --unit user --analyzer word --svd 32
run_vectorization tfidf --disorders TCA --unit user --analyzer char_wb --svd 32
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer word --svd 64
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer char_wb --svd 64
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer word --svd 64
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer char_wb --svd 64
run_vectorization tfidf --disorders Depresion --unit message --analyzer word --svd 64
run_vectorization tfidf --disorders Depresion --unit message --analyzer char_wb --svd 64
run_vectorization tfidf --disorders Depresion --unit user --analyzer word --svd 64
run_vectorization tfidf --disorders Depresion --unit user --analyzer char_wb --svd 64
run_vectorization tfidf --disorders TCA --unit message --analyzer word --svd 64
run_vectorization tfidf --disorders TCA --unit message --analyzer char_wb --svd 64
run_vectorization tfidf --disorders TCA --unit user --analyzer word --svd 64
run_vectorization tfidf --disorders TCA --unit user --analyzer char_wb --svd 64
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer word --svd 300
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer char_wb --svd 300
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer word --svd 300
run_vectorization tfidf --disorders Ansiedad --unit user --analyzer char_wb --svd 300
run_vectorization tfidf --disorders Depresion --unit message --analyzer word --svd 300
run_vectorization tfidf --disorders Depresion --unit message --analyzer char_wb --svd 300
run_vectorization tfidf --disorders Depresion --unit user --analyzer word --svd 300
run_vectorization tfidf --disorders Depresion --unit user --analyzer char_wb --svd 300
run_vectorization tfidf --disorders TCA --unit message --analyzer word --svd 300
run_vectorization tfidf --disorders TCA --unit message --analyzer char_wb --svd 300
run_vectorization tfidf --disorders TCA --unit user --analyzer word --svd 300
run_vectorization tfidf --disorders TCA --unit user --analyzer char_wb --svd 300

# --- E2: encoders por mensaje ---
run_vectorization transformer --model minilm --unit message
run_vectorization transformer --model e5 --unit message
run_vectorization transformer --model beto-sts --unit message

# --- E3 / P2: construcción del vector de usuario ---
run_vectorization transformer --model minilm --unit user --user-pooling concat
run_vectorization transformer --model minilm --unit user --user-pooling mean
run_vectorization transformer --model minilm --unit user --user-pooling max
run_vectorization transformer --model minilm --unit user --user-pooling blocks
run_vectorization transformer --model e5 --unit user --user-pooling concat
run_vectorization transformer --model e5 --unit user --user-pooling mean
run_vectorization transformer --model e5 --unit user --user-pooling max
run_vectorization transformer --model e5 --unit user --user-pooling blocks

# --- E6 / P3: bloque afectivo (pysentimiento) y de hora, con peso ---
run_vectorization transformer --model minilm --unit message --affect 0.1
run_vectorization transformer --model minilm --unit message --affect 0.2
run_vectorization transformer --model minilm --unit message --affect 0.3
run_vectorization transformer --model minilm --unit message --time 0.1
run_vectorization transformer --model minilm --unit message --affect 0.2 --time 0.1
run_vectorization transformer --model minilm --unit user --user-pooling mean --affect 0.2 --time 0.1
run_vectorization features --unit user --affect 0.5 --time 0.5

# --- Variantes de preprocesado, de trastorno y de ajuste ---
run_vectorization transformer --model minilm --unit message --emojis remove
run_vectorization tfidf --disorders Ansiedad --unit message --analyzer char_wb --emojis remove
run_vectorization tfidf --disorders Depresion --unit message --analyzer char_wb --emojis remove
run_vectorization tfidf --disorders TCA --unit message --analyzer char_wb --emojis remove
run_vectorization transformer --model minilm --unit message --disorders Depresion
# --fit-on train (E8): Ansiedad solo tiene Test, así que no se puede.
run_vectorization tfidf --disorders Depresion --unit message --analyzer char_wb --fit-on train
run_vectorization tfidf --disorders TCA --unit message --analyzer char_wb --fit-on train

echo "Tiempos en $TIMES_FILE"
