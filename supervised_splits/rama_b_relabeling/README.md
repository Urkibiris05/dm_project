# Rama B: relabeling débil de mensajes

El plan operativo paso a paso, con estado, RQs, matriz de experimentos y
criterios de interpretación, está en
[`PLAN_DE_TRABAJO.md`](./PLAN_DE_TRABAJO.md). Este README conserva el
protocolo técnico y los resultados ejecutados.

## Objetivo y límites de las conclusiones

Estudiar cómo generar etiquetas débiles de mensaje a partir de `suffer` /
`control` a nivel de sujeto y si esas etiquetas ayudan a la detección
temprana en sujetos no vistos.

El plan **no depende de obtener etiquetas humanas**. Las etiquetas iniciales
heredadas del sujeto y las etiquetas cambiadas por k-NN son *silver labels*,
no anotaciones verdaderas. Un mensaje de un sujeto `suffer` puede no contener
indicios y un mensaje de un sujeto `control` puede mencionarlos. Sin gold
labels independientes, no se puede medir exactitud/F1 real a nivel de
mensaje. Las conclusiones se limitarán a:

- cuántas etiquetas modifica el algoritmo y cómo se distribuyen esos cambios;
- estabilidad/sensibilidad de la propagación ante parámetros y decisiones;
- si el entrenamiento con esas etiquetas mejora la detección del sujeto en
  dev/test, manteniendo separados los sujetos.

Una mejora downstream no demuestra que las pseudoetiquetas de mensaje sean
correctas: solo muestra utilidad para el objetivo de detección bajo este
protocolo de supervisión débil.

## Datos reutilizados

Se reutilizan los splits compartidos por sujeto, semilla y trastorno:

- [`messages_train.csv`](../resultados/messages_train.csv)
- [`messages_dev.csv`](../resultados/messages_dev.csv)
- [`messages_test.csv`](../resultados/messages_test.csv)

Cada fila tiene `subject_id`, `trastorno`, `label` (`0=control`,
`1=suffer`), `id_message`, `message`, `date` y `partition`. `subject_id` e
`id_message` sirven para agrupar y ordenar; no se usan como características.
El orden temporal se fija por `date` y, en caso de empate, `id_message`.
El enunciado formateado está en
[`tarea_individual_enunciado.md`](./tarea_individual_enunciado.md).

No cambiar el split ni volver a generar los CSV para esta rama. Se pueden
reutilizar las reglas de reproducibilidad descritas en
[`../prepare_splits.py`](../prepare_splits.py).

## Herramientas iniciales

Desde la raíz del repositorio:

```bash
python supervised_splits/rama_b_relabeling/audit_splits.py
python -m pytest supervised_splits/rama_b_relabeling -q
```

El auditor escribe `reports/split_audit.csv` y
`reports/split_audit_checks.json` dentro de esta carpeta. No copia los textos
de los mensajes al informe.

Primera auditoría de los CSV disponibles: no encontró solapamiento de sujetos
entre particiones, etiquetas inconsistentes, mensajes vacíos, fechas ausentes
ni IDs de mensaje duplicados dentro de un sujeto. Hay marcas temporales
repetidas, por lo que se mantiene `id_message` como desempate cronológico.
Los splits contienen 19.805 mensajes de train, 5.452 de dev y 2.828 de test
(28.085 en total). La auditoría también resume longitudes, duplicados exactos
de texto por sujeto y mensajes por sujeto en el CSV.

`relabeling.py` implementa búsqueda exacta por similitud coseno y relabeling
de una pasada o iterativo, con exclusión opcional del sujeto consultado y
límite opcional de vecinos por sujeto. La lista de vecinos se calcula una
vez a partir de embeddings fijos; en cada ronda se actualizan de forma
síncrona las etiquetas usadas en el voto. El cálculo exacto es cuadrático en
el número de mensajes de train: es útil para empezar y comparar en un
subconjunto, pero para todo el corpus puede ser necesario sustituir la
búsqueda por un índice ANN manteniendo la misma interfaz.

`make_tfidf_cache.py` y `run_relabeling.py` requieren seleccionar un
trastorno. Esto evita que vecinos de Anxiety, Depression y TCA se mezclen en
una tarea binaria. `make_tfidf_cache.py` construye una representación
exploratoria local TF-IDF + SVD para probar el relabeler mientras no haya un
Sentence Transformer disponible; no es una representación contextual ni el
baseline definitivo. Ejemplo:

```bash
python supervised_splits/rama_b_relabeling/make_tfidf_cache.py \
  --disorder Ansiedad \
  --output supervised_splits/rama_b_relabeling/reports/embeddings/train_ansiedad_tfidf_svd.npz \
  --components 64 \
  --max-features 20000 \
  --min-df 2
```

La caché de cualquier encoder debe ser un `.npz` con:

- `embeddings`: matriz `(n_mensajes_train, dimensión)`;
- `subject_id`: IDs en el mismo orden que el CSV;
- `id_message`: IDs en el mismo orden que el CSV;
- `disorder`: trastorno de los mensajes de la caché;
- `model_id`: opcional, un escalar con el nombre/revisión del encoder.

Ejemplo de creación de caché una vez que haya embeddings:

```python
import numpy as np

np.savez_compressed(
    "train_embeddings.npz",
    embeddings=embeddings.astype(np.float32),
    subject_id=train["subject_id"].to_numpy(dtype=str),
    id_message=train["id_message"].to_numpy(dtype=str),
    disorder=np.array("Ansiedad"),
    model_id=np.array("nombre-y-revision-del-encoder"),
)
```

El runner comprueba que la caché coincide fila por fila con el CSV de train y
rechaza otros splits o trastornos. Ejemplo de ejecución:

```bash
python supervised_splits/rama_b_relabeling/run_relabeling.py \
  --embedding-cache /ruta/local/train_embeddings.npz \
  --disorder Ansiedad \
  --neighbors 10 \
  --min-control-neighbors 5 \
  --mode single_pass \
  --run-name knn-j10-k5
```

El CSV de etiquetas de salida solo contiene `subject_id`, `id_message`,
etiqueta inicial/final y un indicador de cambio; no copia los textos. Cada
ejecución se escribe en `reports/relabeling/<run-name>/` dentro de esta
carpeta y no sobrescribe otra ejecución existente.

## Primer resultado exploratorio

Se ejecutó una pasada con `J=10`, `K=5`, exclusión del mensaje consultado
pero sin excluir los otros mensajes del mismo sujeto. Para cada trastorno se
ajustó por separado un TF-IDF de unigramas/bigramas y SVD de 64 dimensiones,
solo con sus mensajes de train (semilla SVD `2023`). Los recuentos fueron:

| Trastorno | Mensajes `suffer` iniciales | Relabelados a `control` | Fracción relabelada |
|---|---:|---:|---:|
| Ansiedad | 2.307 | 828 | 35,9 % |
| Depresión | 3.951 | 2.466 | 62,4 % |
| TCA | 2.866 | 2.174 | 75,9 % |

Los mensajes relabelados proceden de todos los sujetos `suffer` de train:
65/65 en Ansiedad, 118/118 en Depresión y 100/100 en TCA tuvieron al menos
un cambio. La mediana de la fracción de mensajes positivos relabelados por
sujeto fue 34,8 %, 62,5 % y 76,2 %, respectivamente.

Estos resultados son **solo una prueba de funcionamiento**, no una selección
de `J/K` ni una medida de calidad de mensaje. La representación es léxica
TF-IDF/LSA y el detector downstream aún no se ha evaluado. El porcentaje
elevado en Depresión/TCA indica que conviene inspeccionar distribución por
sujeto y sensibilidad a parámetros antes de continuar; no demuestra que
tantos mensajes carezcan de indicios.

La primera corrida, previa a separar por trastorno, mezcló vecinos de las tres
tareas y no es válida para interpretación; se conserva únicamente como
artefacto de depuración y se excluye de los resultados anteriores.

Los CSV de etiquetas, historial y resúmenes por trastorno/posición/sujeto
están en:

- [`exploratory-ansiedad-tfidf-svd64-j10-k5/`](./reports/relabeling/exploratory-ansiedad-tfidf-svd64-j10-k5/)
- [`exploratory-depresion-tfidf-svd64-j10-k5/`](./reports/relabeling/exploratory-depresion-tfidf-svd64-j10-k5/)
- [`exploratory-tca-tfidf-svd64-j10-k5/`](./reports/relabeling/exploratory-tca-tfidf-svd64-j10-k5/)

Para regenerar el resumen de cada corrida, ejecutar
`summarize_relabeling.py` con `--disorder` y `--run-dir`.

## Comparación inicial de encoders Transformer

### Antecedentes y selección

Los antecedentes respaldan comparar **representaciones congeladas de mensaje**
antes de comprometerse con un encoder:

- En MentalRiskES 2023, Peix et al. compararon RoBERTa-base-bne,
  `roberta-suicide-es` y BETO para representar textos en español; eligieron
  `roberta-suicide-es` como encoder más útil para sus regresores. Sus
  instancias eran historiales de usuario, no mensajes etiquetados
  independientemente. [Artículo](https://arxiv.org/abs/2306.16125).
- En MentalRiskES 2024, UNED-GELP empleó embeddings de mensajes de Universal
  Sentence Encoder y Annoy en su expansión de etiquetas. Su encoder exacto
  no está entre las cachés locales, así que los ensayos siguientes no son una
  reproducción literal. [Artículo](https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper2.pdf).
- ELiRF-VRAIN probó `AIMH/mental-roberta-large` y
  `AIMH/mental-longformer-base-4096`, ajustados para sus tareas de detección
  temprana. Son candidatos para un ensayo posterior, pero no había embeddings
  locales de esos modelos y su entrada/objetivo no equivale a similitud entre
  mensajes independientes.
  [Artículo](https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper3.pdf).
- UMUTeam comparó MarIA, BETO y BERTIN e incorporó sentimiento en su enfoque
  de fine-tuning. Esto motiva una RQ aparte sobre sentimiento, pero no permite
  inferir que sentimiento sea una señal de trastorno.
  [Artículo](https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper4.pdf).

Se reutilizaron seis matrices que ya existían en `vectorizations2/`:

| Alias local | Modelo | Motivo de inclusión |
|---|---|---|
| `beto-sts` | `hiiamsid/sentence_similarity_spanish_es` | BETO ajustado para similitud semántica; candidato principal para k-NN y el más cercano al propósito de similitud de frases de antecedentes locales. |
| `beto-raw` | `dccuchile/bert-base-spanish-wwm-uncased` | Control para contrastar BETO general frente a BETO ajustado a similitud. |
| `minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Alternativa compacta y rápida ya utilizada en el proyecto anterior. |
| `e5` | `intfloat/multilingual-e5-small` | Encoder de frases general, útil como comparación semántica; se conserva el prefijo `query: ` de la vectorización anterior. |
| `robertalex` | `PlanTL-GOB-ES/RoBERTalex` | RoBERTa preentrenado en español. |
| `robertuito-base-uncased` | `pysentimiento/robertuito-base-uncased` | Encoder adaptado al lenguaje de redes sociales en español. |

BETO-STS es una **aproximación disponible**, no el Universal Sentence
Encoder del sistema UNED-GELP ni el mismo checkpoint BETO de ese artículo.
También se incluyó BETO sin ajuste de similitud para poder separar, en una
comparación exploratoria, la familia del encoder de su objetivo de ajuste.

### Alineación, protocolo y límites

`export_previous_embeddings.py` alinea las seis matrices por
`(trastorno, subject, id_message)`, verifica las etiquetas de sujeto, y
exporta una lista común de filas de train y cachés `.npz` sin texto. Las
cachés antiguas aplicaron `min_tokens=4`: los seis modelos tienen embeddings
para los mismos 18.998 de los 19.805 mensajes de train; los 807 ausentes se
excluyen de **todos** estos ensayos para mantener idéntico el universo.
La distribución común es Ansiedad 3.701, Depresión 7.988 y TCA 7.309 mensajes.

Los vectores se generaron previamente con encoders congelados, sin ajuste a
las etiquetas; el relabeling usa únicamente las filas actuales de train y
procesa cada trastorno por separado. Sin embargo, las cachés originales se
calcularon sobre todo el corpus. No se usaron vectores de dev/test en estas
corridas, pero conviene volver a generar los embeddings de train únicamente
si el protocolo final exige evitar incluso la inferencia previa sobre esos
textos.

Para regenerar:

```bash
python supervised_splits/rama_b_relabeling/export_previous_embeddings.py

for model in beto-sts beto-raw minilm e5 robertalex robertuito-base-uncased; do
  for disorder in Ansiedad Depresion TCA; do
    run="exploratory-${model}-${disorder,,}-common-j10-k5"
    python supervised_splits/rama_b_relabeling/run_relabeling.py \
      --train-csv supervised_splits/rama_b_relabeling/reports/embeddings/transformer_common_train/common_train.csv \
      --embedding-cache "supervised_splits/rama_b_relabeling/reports/embeddings/transformer_common_train/${model}/${disorder}.npz" \
      --disorder "$disorder" \
      --neighbors 10 \
      --min-control-neighbors 5 \
      --mode single_pass \
      --run-name "$run"
    python supervised_splits/rama_b_relabeling/summarize_relabeling.py \
      --train-csv supervised_splits/rama_b_relabeling/reports/embeddings/transformer_common_train/common_train.csv \
      --run-dir "supervised_splits/rama_b_relabeling/reports/relabeling/${run}" \
      --disorder "$disorder"
  done
done

python supervised_splits/rama_b_relabeling/summarize_encoder_comparison.py
```

Estos comandos generan la primera copia y no sobrescriben resultados
existentes. Para repetir una configuración, usa un `--output-dir` nuevo al
exportar y un `--run-name` nuevo al relabelar.

La exportación requiere las cachés preexistentes en `vectorizations2/`. El
directorio `reports/embeddings/transformer_common_train/` ocupa alrededor de
261 MB y contiene las matrices recortadas necesarias para repetir estas
corridas sin volver a instalar los Transformers.

### Resultado de esta comparación exploratoria

Se mantuvieron `J=10`, `K=5`, una sola pasada, coseno, el mensaje consultado
excluido y los demás mensajes del mismo sujeto permitidos. Los porcentajes
son la fracción de mensajes iniciales `suffer` que el algoritmo cambió a
`control`; no miden acierto de mensaje.

| Encoder | Ansiedad | Depresión | TCA |
|---|---:|---:|---:|
| BETO-STS | 22,0 % (495/2.249) | 40,2 % (1.549/3.852) | 48,7 % (1.331/2.732) |
| BETO raw | 19,6 % (440/2.249) | 45,0 % (1.733/3.852) | 61,5 % (1.681/2.732) |
| MiniLM | 28,7 % (645/2.249) | 51,2 % (1.974/3.852) | 56,9 % (1.555/2.732) |
| E5-small | 23,2 % (522/2.249) | 44,3 % (1.705/3.852) | 50,8 % (1.388/2.732) |
| RoBERTalex | 22,4 % (504/2.249) | 47,7 % (1.836/3.852) | 71,9 % (1.963/2.732) |
| RoBERTuito | 21,3 % (480/2.249) | 42,0 % (1.618/3.852) | 51,3 % (1.401/2.732) |

Al menos el 90,8 % de los sujetos `suffer` de cada trastorno tuvo algún
mensaje relabelado en cada encoder. Hay diferencias claras entre
representaciones —en especial para TCA—, pero **más relabeling no significa
mejor calidad**. Sin etiquetas gold de mensaje ni evaluación downstream, no
se puede elegir un encoder ganador. BETO-STS es el candidato inicial más
justificado por similitud semántica; MiniLM es una comparación ligera y
RoBERTuito una comparación adaptada a redes sociales. RoBERTalex merece
seguimiento en TCA por su mayor cobertura, no porque se haya probado más
preciso.

El informe completo y reproducible está en
[`reports/encoder_comparison_j10_k5.csv`](./reports/encoder_comparison_j10_k5.csv);
cada corrida, sus etiquetas débiles e informes están en
[`reports/relabeling/`](./reports/relabeling/).

La comparación con los primeros resultados TF-IDF/SVD no es directa: esos
resultados usaron los 19.805 mensajes de train, mientras que los Transformers
se ejecutaron sobre la intersección común de 18.998. Los resultados TF-IDF
previos siguen siendo una prueba de funcionamiento separada.

## Baseline provisional: BETO-STS + MLP + k-NN

`make_beto_sts_embeddings.py` crea un embedding para cada mensaje de train y
dev con `hiiamsid/sentence_similarity_spanish_es`. Esta versión usa los
pesos locales fijados a una revisión del modelo, los devuelve sin normalizar
y usa el mean pooling del propio Sentence Transformer (768 dimensiones).
Procesa textos cortos también: no aplica el antiguo filtro `min_tokens=4`.
Las salidas NPZ guardan embeddings, etiquetas de sujeto, IDs, fechas y
trastorno, pero nunca el texto. El split test no se usa en esta corrida.

El MLP inicial es `Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 1)`,
AdamW, `lr=1e-3`, `weight_decay=1e-4`, batch 128, hasta 50 épocas, early
stopping de 8 épocas y cinco semillas (`11, 23, 37, 51, 79`). Las
características se estandarizan con media/desviación ajustadas solo en train.
La pérdida inicial es BCE sin pesos de clase. El mismo MLP se entrena con:

1. etiquetas iniciales heredadas del sujeto (`initial_labels`);
2. k-NN `J=10, K=5`, una pasada, mensaje propio excluido pero otros mensajes
   del mismo sujeto permitidos (`knn_include_same_subject`);
3. el mismo k-NN excluyendo todos los mensajes del sujeto consultado
   (`knn_exclude_same_subject`).

La predicción online de sujeto es la primera vez que la **media acumulada**
de `P(suffer)` cruza el umbral. El umbral y la mejor época se seleccionan en
dev usando macro-F1 por sujeto. Si no hay cruce, el sistema decide `control`
al final. Por tanto, las métricas de dev presentadas son de selección y no
una estimación final no sesgada; test queda intacto para la evaluación
congelada.

Para regenerar los vectores y ejecutar el baseline desde la raíz:

```bash
venv/bin/python supervised_splits/rama_b_relabeling/make_beto_sts_embeddings.py \
  --partitions train dev \
  --batch-size 32

venv/bin/python supervised_splits/rama_b_relabeling/baseline_mlp.py \
  --run-name baseline-beto-sts-dev-j10-k5-5seeds
```

La generación de embeddings requiere `torch` y `sentence-transformers`,
disponibles en el entorno del proyecto por las dependencias de
`vectorization/requirements.txt`. El baseline guarda manifiestos,
pseudoetiquetas train sin texto, checkpoint/scaler por seed y predicciones
por sujeto en `reports/baseline/<run-name>/`.

Los hiperparámetros son un **protocolo provisional** para entregar una
primera comparación reproducible a los compañeros. No se comparan aún
alternativas de encoder ni la ablación de pesos de clase; hacerlo después
requiere mantener fijo el resto del protocolo. Resultados y limitaciones
concretos, tablas y lectura cautelosa están en
[`BASELINE_RESULTADOS.md`](./BASELINE_RESULTADOS.md). La corrida usa las 19.805
filas de train y 5.452 de dev, incluye mensajes cortos y deja test intacto.
Como resumen, el macro-F1 de sujeto (media ± desviación entre cinco semillas)
para MLP-Init / k-NN excluyendo al sujeto fue 0,8253 / 0,8295 en Ansiedad,
0,7691 / 0,7857 en Depresión y 0,9285 / 0,9260 en TCA. Por tanto, el efecto
del relabeling no es uniforme ni demuestra que las etiquetas de mensaje sean
correctas. Los CSV y el manifiesto están en
[`reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/`](./reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/).

## Pipeline experimental provisional

El pipeline está expresado de forma modular para que se pueda avanzar antes
de que el grupo acuerde encoder, clasificador y regla online.

1. **Embeddings:** codificar cada mensaje de forma independiente con un
   Sentence Transformer congelado. Guardar una fila por `id_message` y los
   metadatos exactos del modelo, tokenizer, librerías, dimensión, dispositivo
   y truncamientos.
2. **Etiquetas iniciales de train:** asignar a cada mensaje la clase de su
   sujeto, solo como etiqueta débil de partida.
3. **Relabeling:** para cada mensaje inicialmente `suffer`, recuperar `J`
   vecinos en el banco de train. Si al menos `K` vecinos tienen etiqueta
   `control`, cambiar su pseudoetiqueta a `control`. Los mensajes
   inicialmente `control` no cambian. Exigir `1 <= K <= J`.
4. **Student:** entrenar el clasificador acordado por el grupo con las
   pseudoetiquetas resultantes. Como control, entrenar el mismo modelo con
   las etiquetas iniciales sin relabeling. El encoder, los datos, el
   clasificador y el decisor deben mantenerse iguales en esta comparación.
5. **Early detection:** inferir scores para mensajes de dev/test en orden
   temporal y usar una regla online común al grupo. Elegir la regla y sus
   umbrales en dev.

La regla `suffer → control` con `J` vecinos y umbral `K` adapta el
procedimiento iterativo de relabeling de UNED-GELP en MentalRiskES 2024. La
adaptación binaria y el student del proyecto no son una reproducción exacta
del sistema participante. Véase la referencia [18] en
[`../plan_proyecto_individual.md`](../plan_proyecto_individual.md).

### Controles contra fugas

- El índice y el relabeling se ajustan solo con train. Dev/test no se añaden
  al banco ni se usan para generar etiquetas de entrenamiento.
- Al consultar vecinos para un mensaje de train, excluir como mínimo el
  propio mensaje. Excluir todos los mensajes del mismo sujeto y limitar
  cuántos vecinos puede aportar un sujeto son comparaciones experimentales.
- Validar parámetros con grupos de sujetos, no separando mensajes de una
  misma persona entre folds.
- Dev se usa para seleccionar configuración y regla de detección; test se
  evalúa con la configuración congelada.
- Ninguna feature puede contener la etiqueta del sujeto, el ID o mensajes
  posteriores a la posición que se está evaluando.

## Research Questions

### RQ1 — Selectividad y utilidad de las etiquetas débiles

**¿Cómo afectan `J`, `K` y la representación de similitud al número y
distribución de mensajes `suffer` relabelados como `control`, y al
rendimiento de detección temprana posterior? ¿Qué aportan sentimiento local
o un teacher LLM local como fuentes alternativas/complementarias de
supervisión débil?**

Comparaciones escalonadas, no un producto cartesiano de todas las opciones:

1. **Núcleo k-NN:** variar pares válidos `(J, K)`; medir la fracción de
   mensajes positivos iniciales relabelados, por sujeto y por posición
   temporal.
2. **Sentimiento:** comparar embeddings semánticos con una similitud que
   incorpore una señal de sentimiento local. El sentimiento es una feature
   auxiliar, no una etiqueta de trastorno.
3. **Teacher local (opcional):** comparar zero-shot y, si el coste lo
   permite, few-shot. Congelar el modelo, prompt, ejemplos, decodificación y
   formato de salida. Sus respuestas son otra fuente de pseudoetiquetas, no
   una referencia verdadera. No enviar mensajes del corpus a servicios
   remotos no autorizados.
4. **Impacto downstream:** entrenar el mismo student con cada configuración
   seleccionada y comparar la detección temprana en los mismos sujetos,
   prefijos y regla online.

**Medidas de relabeling:** cantidad/fracción de cambios; distribución de
clases antes y después; proporción de sujetos afectados; cambios por posición
temporal; sensibilidad a parámetros, semilla y encoder; tiempo y memoria.
Estas medidas describen el comportamiento del método, no la corrección
semántica de cada etiqueta.

**Medidas downstream:** macro-F1 y F1 de `suffer` a nivel sujeto/prefijo,
recall de positivos, falsas alarmas en controles y latencia en cantidad de
mensajes y tiempo natural. Reportar conteos absolutos e intervalos agrupados
por sujeto.

### RQ2 — Iteración, diversidad de vecinos y deriva

**¿La propagación iterativa hasta convergencia aporta más utilidad downstream
que una única pasada, y qué controles de diversidad de sujetos limitan la
deriva de pseudoetiquetas?**

Comparar:

- una pasada frente a iteración hasta convergencia, con un límite máximo de
  rondas como control;
- permitir vecinos del mismo sujeto frente a excluir todo el sujeto
  consultado;
- sin límite frente a un máximo fijo de vecinos por sujeto.

Mantener el encoder y los valores seleccionados de `J/K`. Cuando sea
posible, comparar variantes a cobertura de relabeling semejante para separar
el efecto de la política de vecinos del mero número de cambios. Registrar
por ronda los cambios absolutos/relativos, distribución de clases, sujetos
que aportan vecinos y solapamiento entre mensajes modificados.

Evaluar downstream con el mismo student, sujetos de dev/test, regla online y
presupuesto de entrenamiento. Concluir que una variante es útil si mejora
las métricas de detección o la latencia sin aumentar de forma inaceptable las
falsas alarmas. Más mensajes relabelados o convergencia no significan por sí
mismos una mejora ni una mayor corrección semántica.

## Qué puedes empezar a hacer ya

Estas tareas no requieren decidir el baseline del grupo:

1. **Auditar splits y mensajes.** Crear un resumen por trastorno y partición:
   sujetos, mensajes por sujeto, balance de clase, mensajes vacíos/duplicados,
   fechas ausentes, longitudes y empates temporales. Guardar el informe y no
   cambiar el split.
2. **Definir el contrato del relabeler.** Implementar una función que reciba
   embeddings, pseudoetiquetas iniciales, IDs de sujeto, `J`, `K` y la
   política de vecinos; que devuelva pseudoetiquetas y un log por ronda.
   Separar claramente modo una pasada y modo iterativo.
3. **Escribir pruebas sintéticas.** Comprobar `1 <= K <= J`, auto-vecino
   excluido, `control` inmutable, solo cambios `suffer → control`, ausencia
   de vecinos de dev/test y desempates deterministas.
4. **Preparar una caché de embeddings intercambiable.** Separar extracción y
   k-NN; guardar embeddings alineados con IDs y metadatos del encoder. Así
   podrá sustituirse el encoder sin reescribir ni el algoritmo ni las
   métricas.
5. **Construir diagnósticos.** Graficar cambios frente a `J/K`, por ronda,
   sujeto y posición temporal; medir distribución de clases, sujetos
   contribuyentes y sensibilidad a limitar vecinos por sujeto.
6. **Probar sentimiento local de forma aislada.** Ejecutarlo primero en una
   muestra pequeña local, guardar los scores por ID y medir coste. No
   interpretar polaridad como evidencia clínica.
7. **Inventariar LLM locales sin procesar todo el corpus.** Comparar
   candidatos por licencia, idioma, memoria requerida y formato de salida.
   Probar instalación e inferencia con texto sintético/permitido; fijar un
   prompt y esquema de salida antes de experimentar con el corpus.
8. **Preparar evaluación downstream modular.** Definir una interfaz que
   reciba scores por mensaje y calcule métricas por sujeto/prefijo. La
   agregación online concreta puede quedar como parámetro hasta el acuerdo
   del grupo.

## Orden de trabajo sugerido

1. Completar auditoría de datos, pruebas e interfaz de caché/relabeler.
2. Cuando esté disponible una representación vectorial, comparar etiquetas
   iniciales frente a una pasada k-NN y verificar los logs.
3. Responder RQ1 empezando por `J/K`; probar sentimiento y teacher local solo
   si el tiempo y los recursos lo permiten.
4. Responder RQ2 con una pasada/iteración y las ablaciones de diversidad.
5. Elegir baseline y decisor con el grupo, fijar configuración usando dev y
   evaluar test una sola vez.

## Límites de interpretación

Sin gold labels de mensaje, no reportar precisión, recall o F1 de mensaje
como si midieran verdad. Sí se puede informar de cobertura, distribución,
estabilidad, coste y rendimiento downstream en sujetos no vistos. Si el
relabeling mejora el detector, la conclusión será que mejora la utilidad de
la supervisión débil bajo este protocolo, no que haya identificado con
certeza qué mensajes contienen indicios.
