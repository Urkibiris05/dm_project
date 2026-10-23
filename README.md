# dm_project

Herramientas para vectorizar mensajes y usuarios, experimentar con DBSCAN,
evaluar agrupamientos y asignar nuevos mensajes a clusters previamente
ajustados. El repositorio incluye una CLI Bash, módulos Python, experimentos
reproducibles y pruebas.

## Requisitos e instalación

- Bash
- Python 3
- Dependencias declaradas en `requirements.txt`
- Para vectorización con transformers, afecto y asignación BETO-STS:
  dependencias adicionales de `vectorization/requirements.txt`
- Los flujos que usan corpus requieren que los datos estén disponibles
  localmente; los experimentos de transformers pueden necesitar descargar
  modelos de Hugging Face la primera vez.

Se recomienda crear un entorno virtual e instalar las dependencias en el
mismo intérprete que utilizará `dm.sh`:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r vectorization/requirements.txt
```

Si se usa otro intérprete, se puede especificar con `PYTHON`:

```bash
PYTHON=.venv/bin/python ./dm.sh help
```

## Interfaz de línea de comandos

Ejecuta los comandos desde la raíz del repositorio. La ayuda general muestra
el catálogo de tareas y cada ejecutor conserva sus opciones específicas:

```bash
./dm.sh help
./dm.sh vectorize --help
./dm.sh vectorize transformer --help
./dm.sh assign --help
```

Las opciones de los comandos se pasan después de su nombre. Por ejemplo:

```bash
./dm.sh vectorize transformer --model beto-sts --unit message
./dm.sh cluster-messages --all
./dm.sh evaluate-e4 --vectorization message__all__e5 --max-samples 1000
```

## Datos de entrada y salidas

### Corpus

El cargador del corpus espera una carpeta raíz organizada por trastorno y
partición, por ejemplo:

```text
corpus/
├── Ansiedad/
│   ├── Train/
│   └── Test/
├── Depresion/
└── TCA/
```

Las particiones contienen archivos JSON de sujetos con mensajes que incluyen
texto y fecha. Las etiquetas de referencia se leen de los archivos `Gold`
cuando están disponibles. Se indican mediante `--corpus-root`, `--disorders`
y `--splits`.

### Representaciones vectoriales

Una vectorización se guarda en una carpeta que contiene, entre otros archivos:

- `X.npy`: matriz con una fila por unidad de análisis;
- `metadata.csv`: metadatos alineados por fila;
- `config.json`: parámetros y versiones usados al generar la representación.

El `--name` identifica la carpeta de salida. Por defecto las vectorizaciones
se guardan en `vectorizations/`; varios ejecutores de clustering de mensajes
y asignación usan `vectorizations2/` como directorio predeterminado.

### Resultados y figuras de experimentos

Los experimentos organizan sus salidas en carpetas propias, normalmente
`resultados/` para tablas, métricas y asignaciones, y `figuras/` para gráficos.
Por ejemplo, E0 guarda resultados y figuras bajo `E0/`, mientras que E1 usa
`E1/resultados/` y `E1/figuras/`. Otros módulos pueden generar solo una de
esas carpetas o usar nombres específicos; consulta el README o la ayuda del
experimento para conocer las rutas exactas.

En general, no se escriben todas las salidas en una carpeta global: cada
experimento conserva sus resultados junto a su código. Las excepciones con
rutas explícitas, como las asignaciones de `assign-test` en `E8/resultados/`,
se describen en la sección del comando correspondiente. Las vectorizaciones
generadas se guardan aparte, bajo el directorio indicado por `--output-dir`.

## Vectorización

El subcomando `vectorize` ofrece `tfidf`, `transformer` y `features`.
`--unit message` genera un vector por mensaje; `--unit user`, uno por usuario.

Ejemplo: embeddings BETO-STS por mensaje, con Train y Test:

```bash
./dm.sh vectorize transformer \
  --corpus-root /ruta/al/corpus \
  --disorders Ansiedad \
  --splits Train Test \
  --unit message \
  --model beto-sts \
  --output-dir vectorizations2 \
  --name message__ansiedad__beto-sts
```

Ejemplo: TF-IDF de caracteres reducido con SVD:

```bash
./dm.sh vectorize tfidf \
  --corpus-root /ruta/al/corpus \
  --disorders Depresion \
  --splits Train Test \
  --unit message \
  --analyzer char_wb \
  --svd 128 \
  --fit-on train \
  --output-dir vectorizations2 \
  --name message__depresion__tfidf-char
```

Ejemplo: representación de usuario agregando embeddings MiniLM:

```bash
./dm.sh vectorize transformer \
  --corpus-root /ruta/al/corpus \
  --disorders Ansiedad \
  --splits Train Test \
  --unit user \
  --model minilm \
  --user-pooling mean \
  --output-dir vectorizations2 \
  --name user__ansiedad__minilm
```

Para conocer las opciones disponibles:

```bash
./dm.sh vectorize tfidf --help
./dm.sh vectorize transformer --help
./dm.sh vectorize features --help
```

El lote de vectorizaciones del proyecto se puede ejecutar con
`./dm.sh vectorize-all`. Puede tardar, requerir modelos externos y generar
múltiples representaciones.

## Clustering y evaluación

### Barrido de mensajes

`cluster-messages` recibe los nombres de carpetas de representaciones
existentes, no la ruta directa a `X.npy`:

```bash
./dm.sh cluster-messages \
  --representation message__ansiedad__beto-sts \
  --vector-dir vectorizations2
```

Se pueden indicar varias representaciones o usar `--all` para descubrir las
carpetas `message__all__*` compatibles:

```bash
./dm.sh cluster-messages \
  --representation message__all__beto-sts message__all__beto-raw \
  --vector-dir vectorizations2

./dm.sh cluster-messages --all --vector-dir vectorizations2
```

`--sample-size N` permite realizar un cribado sobre una submuestra
reproducible. Los resultados se guardan como CSV bajo `E2/resultados/`.

### Otros experimentos

```bash
# Clustering jerárquico de mensajes y perfiles de usuario
./dm.sh cluster-hierarchical --help

# Experimento E4 para distancias Minkowski
./dm.sh evaluate-e4 \
  --vectorization message__all__beto-sts \
  --max-samples 1000

# Validación de DBCV sobre datos sintéticos
./dm.sh validate-dbcv

# Experimentos de la implementación y de TF-IDF
./dm.sh e0 --rapido
./dm.sh e1 --rapido
```

Algunos experimentos tienen entradas preconfiguradas en sus módulos y
requieren que sus archivos de datos o resultados intermedios existan.
Consulta la ayuda de cada comando y la documentación de la carpeta
correspondiente.

## Asignación a clusters

Hay dos flujos distintos:

- `assign-test` ajusta DBSCAN con las filas Train de una representación
  existente y asigna las filas Test incluidas en esa misma representación.
- `prepare-assigner` y `assign` permiten persistir el modelo DBSCAN y asignar
  mensajes nuevos leídos desde un archivo de texto.

### Persistir un asignador BETO-STS

La representación debe haber sido generada con BETO-STS, una fila por
mensaje, vectores normalizados y sin bloques adicionales. Por defecto se usa
`vectorizations2/message__all__beto-sts/`.

```bash
./dm.sh prepare-assigner \
  --representation message__all__beto-sts \
  --vector-dir vectorizations2 \
  --min-samples 5 \
  --percentile 50
```

El comando ajusta DBSCAN únicamente con las filas Train y guarda el artefacto
en `E8/modelos/`. El nombre predeterminado para los parámetros anteriores es
`message__all__beto-sts_dbscan_m5_p50.joblib`. Se puede cambiar la ubicación
con `--model-out`.

### Asignar mensajes desde TXT

El archivo de entrada debe estar codificado en UTF-8 y contener un mensaje por
línea. Las líneas vacías se ignoran; los mensajes de menos de cuatro tokens
no se vectorizan, siguiendo el preprocesado del corpus. BETO-STS genera los
embeddings y el DBSCAN guardado asigna cada mensaje a un cluster existente o
lo marca como ruido (`cluster_assigned = -1`).

```bash
./dm.sh assign \
  --input demo_mensajes.txt \
  --model E8/modelos/message__all__beto-sts_dbscan_m5_p50.joblib \
  --output demo_asignaciones.csv
```

El CSV incluye `input_line`, `text`, `cluster_assigned`, `is_noise`,
`nearest_core_cluster`, `nearest_core_distance`, `distance_over_eps` y el
tamaño del cluster en Train. La salida de terminal resume los parámetros del
modelo, su número de clusters, el porcentaje asignado/ruido y la distribución
de mensajes por cluster. `distance_over_eps < 1` indica que el mensaje quedó
dentro del radio de asignación; un valor mayor que 1 corresponde a ruido.
Esta distancia describe proximidad geométrica, no es una probabilidad ni una
confianza de clasificación.

El artefacto `.joblib` contiene DBSCAN y sus puntos núcleo, pero no los pesos
del encoder BETO-STS. Para asignar textos nuevos se requieren las dependencias
de `vectorization/requirements.txt` y acceso local al encoder de Hugging Face.
Los embeddings calculados se guardan en `.cache/` para reutilizarlos.

`assign-test` procesa las instancias de Test que ya forman parte de la
representación; no recibe un TXT. Ejemplo:

```bash
./dm.sh assign-test \
  --representation message__all__beto-sts \
  --min-samples 5 \
  --percentile 50 \
  --output-prefix beto_sts
```

Los resultados de este flujo se escriben bajo `E8/resultados/`.

## Pruebas y perfiles rápidos

```bash
# Pruebas locales rápidas o suite completa
./dm.sh for-the-impatient
./dm.sh tests
./dm.sh tests vectorization/tests/test_cli.py -q
```

`--for-the-impatient` también está disponible en algunos experimentos para
usar rejillas o submuestras reducidas:

```bash
./dm.sh e0 --for-the-impatient
./dm.sh cluster-messages --all --for-the-impatient
./dm.sh evaluate-e4 --for-the-impatient
```

## Catálogo

Para ver todos los comandos disponibles, ejecuta `./dm.sh help`. Entre ellos
se incluyen `vectorize`, `vectorize-all`, `e0`, `e1`, `cluster-users`,
`cluster-messages`, `cluster-hierarchical`, `cluster-message-pca`,
`evaluate-message-dbcv`, `evaluate-message-candidates`, `evaluate-shortlist`,
`validate-dbcv`, `word-doc-embeddings`, `evaluate-e4`, `evaluate-distances`,
`assign-test`, `prepare-assigner`, `assign`, `profile`, `profile-plots`,
`describe` y `tests`.
