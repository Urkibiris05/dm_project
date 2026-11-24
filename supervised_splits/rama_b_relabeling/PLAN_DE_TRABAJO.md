# Plan de trabajo experimental — Rama B: relabeling de mensajes

> **Guía operativa para ejecutar el proyecto paso a paso.** Este documento
> registra el objetivo, el estado real, las decisiones que faltan, el diseño
> experimental, cómo leer los resultados y qué debe hacer la siguiente persona
> (o LLM) que continúe el trabajo. Mantener el código, los informes y los
> resultados nuevos dentro de `supervised_splits/rama_b_relabeling/`.

## 1. Resumen ejecutivo

El proyecto estudia cómo convertir etiquetas de trastorno a nivel de sujeto
en etiquetas débiles a nivel de mensaje mediante k-NN, y si entrenar un MLP
con esas etiquetas ayuda a detectar antes el trastorno en sujetos no vistos.
La unidad de representación y predicción es **un mensaje individual**; no se
debe asignar sin más la etiqueta del sujeto a todos sus mensajes como si
fuera una verdad de mensaje.

La comparación ya realizada muestra que la decisión de encoder cambia mucho
la cobertura del relabeling. **No demuestra que unos mensajes estén mejor
etiquetados que otros** y no permite elegir un encoder ganador. El resultado
principal del trabajo todavía está por obtener: comparar la detección
temprana downstream con y sin relabeling, usando el mismo encoder, MLP y
decisor.

Las dos preguntas propuestas son:

1. **RQ1 — Sensibilidad y utilidad:** ¿Cómo interactúan el encoder y los
   parámetros `J/K` del k-NN al determinar qué proporción de mensajes
   `suffer` se relabela, y cuándo esa supervisión débil mejora la detección
   temprana del sujeto frente a no relabelar?
2. **RQ2 — Robustez de la propagación:** ¿La propagación iterativa y las
   restricciones para evitar vecinos dominados por un mismo sujeto reducen
   la deriva de etiquetas y/o mejoran la detección temprana frente a una
   pasada de k-NN sin esas restricciones?

No se requiere gold label humana de mensaje. Por ello, no se medirá ni se
afirmará la exactitud real de las etiquetas de mensaje. La evaluación
principal es por sujeto, con las etiquetas gold de sujeto que ya proporciona
el corpus, y por momento de detección.

## 2. Cómo continuar si eres una persona o un LLM nuevo

Lee primero, en este orden:

1. Este plan: decisiones, estado y próximos pasos.
2. [`README.md`](./README.md): detalles técnicos, comandos disponibles y
   resultados que ya se ejecutaron.
3. [`tarea_individual_enunciado.md`](./tarea_individual_enunciado.md):
   requisitos de la asignatura.
4. Código existente: [`relabeling.py`](./relabeling.py),
   [`run_relabeling.py`](./run_relabeling.py) y
   [`summarize_relabeling.py`](./summarize_relabeling.py).

Reglas para quien continúe:

- No regeneres ni modifiques los CSV de partición compartidos.
- No copies ni escribas mensajes del corpus en informes o manifiestos.
- No uses dev/test para construir vecinos, crear pseudoetiquetas de train,
  ajustar PCA/normalizadores ni entrenar el encoder o el MLP. Dev sí se usa
  para seleccionar configuraciones, early stopping y umbrales según el
  protocolo; test no se usa para selección.
- No elijas una configuración mirando test. Test se consulta una vez, después
  de congelar el protocolo y los parámetros.
- No llames “verdaderas”, “correctas” o “precisas” a las etiquetas k-NN:
  son etiquetas débiles sin validación humana.
- Antes de modificar un archivo, revisa su contenido y los cambios locales:
  hay cambios no relacionados en el worktree.
- Mantén todos los nuevos scripts, notebooks, informes y resultados dentro
  de esta rama. No hace falta crear otro repositorio para avanzar.
- Cada corrida debe llevar configuración completa, semilla, IDs de modelo y
  ruta de salida únicos. Nunca sobrescribas un experimento previo.
- Si una decisión del protocolo no está acordada, marca la decisión como
  pendiente y no presentes la opción elegida como un hecho del grupo.

## 3. Estado actual verificado

### Hecho

- Los CSV comunes están en `supervised_splits/resultados/`.
- Una auditoría previa registró 19.805 mensajes de train, 5.452 de dev y
  2.828 de test. Los sujetos están separados entre particiones; hay
  timestamps repetidos, por lo que el orden temporal desempata con
  `id_message`.
- [`relabeling.py`](./relabeling.py) implementa k-NN exacto con coseno,
  exclusión automática del propio mensaje, cambios solo `suffer → control`,
  modo una pasada/iterativo, exclusión opcional del sujeto y límite opcional
  de vecinos por sujeto.
- La exploración inicial TF-IDF/SVD probó que el pipeline funciona, pero no
  es el encoder elegido para el baseline.
- Se alinearon seis embeddings Transformer preexistentes y se ejecutó
  `J=10, K=5`, una pasada, por trastorno. Sobre la intersección común hay
  18.998 mensajes (Ansiedad 3.701, Depresión 7.988, TCA 7.309); faltan 807
  mensajes train en las cachés antiguas debido al filtro previo
  `min_tokens=4`.
- Los seis encoders cambiaron porcentajes diferentes de mensajes iniciales
  `suffer`: entre 19,6–28,7 % en Ansiedad, 40,2–51,2 % en Depresión y
  48,7–71,9 % en TCA. Estos porcentajes son **cobertura/actividad del
  algoritmo**, no precisión.
- Los resultados agregados están en
  [`reports/encoder_comparison_j10_k5.csv`](./reports/encoder_comparison_j10_k5.csv);
  las cachés comunes, en
  [`reports/embeddings/transformer_common_train/`](./reports/embeddings/transformer_common_train/).
- El baseline BETO-STS + MLP ya se ejecutó en todos los mensajes de train y
  dev (incluidos los cortos) con cinco semillas. Compara etiquetas iniciales
  frente a una pasada k-NN con y sin vecinos del mismo sujeto.
- El resumen detallado está en
  [`BASELINE_RESULTADOS.md`](./BASELINE_RESULTADOS.md); el manifiesto confirma
  que test no se usó. Las métricas son de selección en dev y no evalúan la
  verdad de etiquetas de mensaje.

### Aún no hecho

- No se ha hecho comparación downstream con otros encoders ni una ablación
  con pesos de clase.
- No se ha evaluado test; la configuración debe cerrarse con el grupo antes
  de cualquier consulta final a test.
- No se han probado sistemáticamente `J/K`, exclusión del sujeto, límite de
  vecinos por sujeto ni relabeling iterativo.
- Los embeddings BETO-STS se regeneraron para todos los mensajes de train y
  dev, sin `min_tokens=4`; los embeddings de test quedan pendientes hasta
  congelar el protocolo.
- Las cinco semillas miden variación de inicialización, no incertidumbre
  sobre los sujetos de dev. Falta calcular intervalos bootstrap pareados por
  sujeto.
- La métrica macro-F1, la media acumulada y la selección de época/umbral se
  usaron como propuesta provisional; falta ratificación del grupo.

El estado tracked del repositorio también incluye cambios previos ajenos a
esta rama. No deben revertirse ni mezclarse con este trabajo.

## 4. Definición del problema y límites de interpretación

### Tarea

Se resuelven tres problemas binarios independientes: Ansiedad, Depresión y
TCA. Cada mensaje conserva su identidad como instancia. El sistema recibe
mensajes cronológicamente y puede actualizar la predicción del sujeto tras
cada nuevo mensaje. El sujeto completo debe pertenecer a una sola partición.

### Etiquetas disponibles y etiquetas débiles

- `label=1` / `suffer`: el sujeto está etiquetado como afectado por el
  trastorno.
- `label=0` / `control`: el sujeto está etiquetado como control.
- Estas son etiquetas gold de **sujeto**, no de mensaje.
- La inicialización que copia esa etiqueta a cada mensaje es una fuente de
  supervisión débil. El MLP entrenado con esa inicialización es el control
  experimental **sin k-NN**, no una afirmación de que cada mensaje tenga esa
  clase verdadera.
- El k-NN actual solo puede cambiar positivos iniciales a negativos. Los
  controles no se convierten en positivos. Esta asimetría debe figurar en la
  discusión: puede reducir falsos indicios en mensajes de sujetos afectados,
  pero no puede descubrir indicios en mensajes de sujetos control.
- Sin anotaciones humanas independientes no se pueden reportar precision,
  recall, F1 ni accuracy “reales” de mensaje. Sí se puede informar de
  cobertura, cambios, estabilidad, distribución y utilidad downstream.

### Salidas del proyecto

1. Código fuente reproducible y documentado.
2. Manifiestos, tablas y gráficos generados solo con datos/resultados
   necesarios; no exportar texto de mensajes.
3. Un póster DIN-A3 con método, RQs, experimentos, resultados y limitaciones.
4. Un archivo comprimido con código para la entrega.

## 5. Decisiones de protocolo que deben cerrarse antes del experimento final

Registrar la respuesta en la tabla. No bloquear tareas de preparación que no
dependan de la decisión.

| Decisión | Recomendación operativa | Estado |
|---|---|---|
| ¿Qué encoder implementa el baseline? | Comparar en dev BETO-STS, MiniLM y RoBERTuito; considerar RoBERTalex o E5 como cuarta alternativa. Elegir por detección downstream y coste, no por cobertura de pseudoetiquetas. Coordinar con el grupo para evitar duplicar las ramas de otros. | Pendiente |
| ¿Qué significa “mean pooling”? | Si se usa un Sentence Transformer que ya devuelve un vector por mensaje, usar ese vector; no volver a promediar. Si se usa el `last_hidden_state` de un encoder base, calcular media de tokens no padding con `attention_mask`. Especificar cuál antes de llamar baseline al modelo. | Pendiente |
| ¿Cómo se convierte la secuencia de scores en detección del sujeto? | Proponer como protocolo simple un score acumulado online (p. ej., media de `P(suffer)` de los mensajes vistos) y un umbral escogido en dev. Acordar la regla común con el grupo y no cambiarla entre controles/ablaciones. | Pendiente |
| ¿Cuál es la métrica primaria? | Macro-F1 a nivel sujeto en checkpoints/prefijos y una métrica de latencia temprana exigida por el curso o acordada por el grupo. La selección de umbral debe optimizar una sola métrica primaria declarada. | Pendiente |
| ¿Qué universo de mensajes usar? | Para análisis final, preferir regenerar embeddings para todas las filas válidas de train/dev/test. Si no es posible, definir de antemano el tratamiento de los 807 mensajes train sin caché y aplicar el mismo criterio a todas las configuraciones. | Pendiente |
| ¿Permitir otros mensajes del mismo sujeto entre los vecinos? | La configuración actual los permite excepto el propio mensaje. Mantenerla como control histórico y comparar con exclusión total del sujeto. No asumir qué variante es mejor. | Pendiente |
| ¿Cuál es la política de balance del MLP? | Fijar una sola política antes de comparar (pesos por clase o sin pesos). Si los pesos dependen de la distribución resultante de pseudoetiquetas, reportar explícitamente esa interacción y añadir una comparación con pesos congelados a partir de etiquetas iniciales. | Pendiente |
| ¿Cuántas semillas y qué incertidumbre? | Empezar con 5 semillas de inicialización del MLP. Para diferencias por sujeto, usar intervalos bootstrap pareados y agrupados por sujeto, no por mensaje. | Pendiente |

### Acuerdos y propuestas actuales (23-11-2026)

La tabla anterior se conserva como lista de decisiones que había que acordar.
Este registro refleja la conversación más reciente y prevalece cuando difiere
de su estado inicial. “Provisional” significa que es una propuesta para
presentar al grupo, no una decisión ya ratificada por el grupo.

#### Acordado por ahora

| Tema | Estado actual |
|---|---|
| Encoder principal | **BETO-STS, provisional**, hasta compararlo downstream con alternativas. Corresponde al modelo `hiiamsid/sentence_similarity_spanish_es`. La selección se justifica por la similitud semántica que necesita k-NN, no por su porcentaje de relabeling. |
| Unidad y representación | Un vector por mensaje; los mensajes siguen siendo instancias independientes. No promediar mensajes de un mismo sujeto. |
| Universo de mensajes | La intención es incluir todos los mensajes del split compartido. Las matrices antiguas de `vectorizations2/` son **cachés**: archivos guardados con los vectores ya calculados para no volver a ejecutar el encoder. Se generaron aplicando `min_tokens=4`, por eso faltan mensajes cortos; no es una propiedad del corpus ni del split. Para el análisis final hay que generar embeddings para las filas faltantes/todas las filas, o demostrar un procedimiento equivalente que incluya todos los mensajes. |
| Vecinos del mismo sujeto | Se analizará explícitamente la dependencia del relabeling respecto a otros mensajes del mismo sujeto. El propio mensaje consultado se excluye siempre; se comparará permitir frente a excluir los demás mensajes del sujeto. |
| Decisor temporal | La media acumulada de las probabilidades `P(suffer)` de los mensajes vistos se propone como baseline común. Otras reglas de acumulación pueden ser una alternativa experimental, manteniendo iguales los scores del MLP. |
| Métricas | Propuesta aceptada para el diseño: macro-F1 a nivel sujeto como métrica principal, junto con métricas de la clase `suffer` y latencia/early detection. El protocolo exacto de latencia y selección del umbral debe coordinarse con el grupo/enunciado. |
| Semillas | Cinco inicializaciones del MLP y reporte de media/desviación; cuando sea viable, comparación pareada e intervalos agrupados por sujeto. |

#### Propuestas provisionales para llevar al grupo

| Tema | Propuesta concreta | Motivo / condición |
|---|---|---|
| Desbalance del MLP | **Empezar sin pesos de clase en la corrida principal** y reportar el balance de mensajes iniciales y relabelados. Añadir una ablación con `BCEWithLogitsLoss(pos_weight=N_control/N_suffer)`, calculando el peso una sola vez con etiquetas iniciales de train y manteniéndolo fijo para Init y k-NN. | Da un control simple y evita que el cambio de distribución provocado por k-NN altere también los pesos. Si el baseline ignora `suffer`, la ablación informa si el problema se debe al desbalance. No calcular pesos con dev/test. |
| MLP baseline | MLP pequeño para clasificación binaria, sobre el vector fijo de mensaje: `Linear(d, 256) → ReLU → Dropout(0.3) → Linear(256, 1)`, con salida logit. Como punto inicial para validación: AdamW, `lr=1e-3`, `weight_decay=1e-4`, batch 128, máximo 50 épocas, cinco semillas. Afinar solo con train/dev y congelar los mismos valores para MLP-Init y MLP-kNN. | Es una cabeza ligera, suficientemente clara como baseline y separa el coste del encoder del clasificador. Las dimensiones e hiperparámetros son un punto de partida que debe verificarse en el entorno y acordarse con el grupo; no son resultados experimentales. |
| Early stopping y umbral | Seleccionar época/umbral únicamente con dev. Comparar los checkpoints de prefijo de la media acumulada y escoger el umbral según la métrica primaria acordada; documentar qué hacer si un sujeto nunca cruza el umbral. | El conjunto de desarrollo puede orientar selección; test queda congelado. Evitar fijar un número de mensajes sin validarlo en dev. |
| Alternativas de clasificador | Mantener el MLP como comparador común para aislar embeddings y relabeling. Si la aportación individual también debe comparar algoritmos, proponer con el grupo una **CNN 1D** o una **BiGRU/BiLSTM** sobre estados de tokens del mismo encoder, y comparar sistemas completos. | Una CNN/RNN necesita una secuencia de tokens; no tiene sentido aplicar convolución temporal a un solo vector de frase como si ese vector conservara el orden del texto. Estas alternativas requieren especificar pooling/tokenización y amplían el alcance; no sustituirían el experimento controlado MLP-Init vs MLP-kNN. |
| Alternativas de embeddings | Shortlist sugerida: BETO-STS como principal provisional; MiniLM como encoder compacto de frases; RoBERTuito por texto social; E5-small como encoder general. RoBERTalex queda como candidato opcional si hay presupuesto. | Elegir/recortar la lista con el grupo y comparar downstream con el mismo MLP. Los seis encoders ya comparados solo se evaluaron por cobertura del k-NN; esa tabla no permite proclamar ganador. |
| Alternativas de pooling | En el baseline, mantener un vector por mensaje. Si hace falta una ablación para la rama de pooling del equipo, comparar el vector de frase entregado por BETO-STS con una media enmascarada de estados token-level del mismo encoder. | Acordar qué produce realmente la implementación/caché de BETO-STS. La media enmascarada es una extracción de vector de mensaje, no una bolsa de mensajes del sujeto. |
| Cuatro alternativas de la asignatura | Propuesta: si se exige comparar al menos cuatro sistemas, hacer una comparación controlada de cuatro encoders con el mismo MLP y protocolo; además mostrar Init vs k-NN para cada encoder o, como mínimo, para la shortlist final. Confirmar con el equipo si las alternativas deben ser algoritmos o si pueden ser combinaciones representación+clasificador. | El enunciado habla de cuatro alternativas de sistema. La definición de “alternativa” debe confirmarse para no asumir que cuatro encoders con un MLP común satisfacen por sí solos el requisito de diversidad de clasificadores. |

#### Qué llevar como propuesta al grupo

> “Propongo usar provisionalmente BETO-STS, un vector por mensaje, y un MLP
> pequeño. El experimento principal compara el mismo MLP entrenado con la
> etiqueta inicial heredada y con la etiqueta relabelada por k-NN. Empezaría
> sin ponderación de clases y haría una ablación con un `pos_weight` fijado
> por las etiquetas iniciales. Para early detection usaría la media
> acumulada de `P(suffer)` y elegiría el umbral en dev. Compararía BETO-STS
> con MiniLM, RoBERTuito y E5 con cinco semillas. En k-NN excluiría siempre el
> propio mensaje y mediría permitir frente a excluir otros mensajes del
> mismo sujeto. ¿Estas decisiones sirven como protocolo común y cómo debemos
> definir las cuatro alternativas requeridas?”

Las decisiones ratificadas por el grupo deben añadirse aquí con fecha. Si el
grupo cambia una propuesta, conservar el historial breve y actualizar las
fases experimentales afectadas.

### Mean pooling: evitar una ambigüedad metodológica

“Mean pooling” puede referirse a cosas distintas:

1. **Pooling de tokens dentro de un mensaje:** obtener estados ocultos
   `h_1, ..., h_T` y promediar solo los tokens reales:
   `z = sum(mask_t * h_t) / sum(mask_t)`. Este produce un vector por mensaje.
2. **Pooling de mensajes de un sujeto:** promediar embeddings de varios
   mensajes. Esto convertiría al sujeto/historial en la instancia y no debe
   sustituir el enfoque principal de mensajes independientes.
3. **Salida de Sentence Transformer:** muchos modelos ya incorporan pooling
   en su arquitectura y entregan un vector de frase. Aplicar otro promedio a
   una matriz que ya contiene un único vector por mensaje no tiene sentido.

El acuerdo actual es producir **un vector por mensaje**. Para BETO-STS se
debe verificar si el artefacto usado ya devuelve un embedding de frase; si es
así, usarlo directamente. Si se extraen estados token-level, hacer media
enmascarada de tokens no padding. La agregación temporal solo ocurre después
de que el MLP predice cada mensaje, para emitir una decisión online de sujeto.

## 6. Pipeline experimental fijado

Para cada trastorno y cada partición:

1. **Ordenar mensajes** por `date` ascendente y desempatar por `id_message`.
2. **Vectorizar cada mensaje por separado** con el encoder congelado. Guardar
   modelo/revisión, tokenizer, dimensión, pooling, truncamiento,
   normalización, semilla, librerías y política de mensajes vacíos/cortos.
3. **Construir el banco k-NN solo con train** de ese trastorno. La similitud
   es coseno; el propio mensaje se excluye siempre.
4. **Generar etiquetas débiles solo para train**:
   - etiqueta inicial `y_i` heredada del sujeto;
   - para cada mensaje inicialmente `suffer`, encontrar `J` vecinos;
   - cambiar a `control` si al menos `K` vecinos tienen etiqueta actual
     `control`;
   - nunca cambiar un control inicial a suffer.
5. **Entrenar el MLP** para predecir la etiqueta inicial (control) o la
   pseudoetiqueta final (condición relabelada). Mantener constante el diseño
   del MLP, presupuesto de entrenamiento, pérdida y preprocesamiento.
6. **Inferencia de dev/test mensaje a mensaje** en orden temporal. No usar la
   etiqueta gold del sujeto, mensajes futuros, vecinos de dev/test ni sus
   datos para recalcular normalizaciones/selección.
7. **Aplicar un único decisor online acordado** y calcular métricas por
   sujeto, checkpoints/prefijos y latencia. Los umbrales se escogen en dev.
8. **Congelar todo y evaluar test una sola vez.**

### Qué son las cachés de embeddings y por qué faltan mensajes

Una **caché** aquí es una matriz guardada con una fila de números (embedding)
por mensaje. Se prepara una vez para no volver a cargar el Transformer y
calcular los vectores cada vez que se ejecuta k-NN o el MLP. No contiene una
etiqueta adicional ni es un formato especial del corpus. Las matrices
anteriores se generaron con el filtro `min_tokens=4`, por lo que omitieron
mensajes cortos antes de calcular sus embeddings. Los CSV de train/dev/test
no tienen ese filtro: la intención acordada es incluir todas sus filas y
generar los embeddings que faltan.

### Validez de las cachés de embeddings ya existentes

Las matrices de `vectorizations2` se generaron con encoders congelados y sin
ajuste a las etiquetas. La inferencia fija de un modelo preentrenado sobre
textos de dev/test no equivale por sí misma a entrenar con esos ejemplos; aun
así, para máxima claridad del protocolo final:

- construir/reconstruir las matrices de cada partición por separado o dejar
  claramente documentado que solo hubo inferencia congelada;
- no ajustar vocabulario, PCA, normalización estadística ni selección de
  características con datos fuera de train;
- ajustar cualquier transformación con train y aplicarla sin re-ajuste a
  dev/test;
- no usar test para relabeling, selección, validación de parámetros ni
  debugging semántico.

Las cachés existentes omiten mensajes cortos (`min_tokens=4`). La
intersección de 18.998 ya sirvió para comparar encoders de forma exploratoria,
pero no es el universo final acordado. Para el experimento final, codificar
todas las filas válidas de train/dev/test (incluidos mensajes cortos) y
verificar alineamiento exacto por IDs. No mezclar una condición entrenada/
evaluada en todos los mensajes con otra en 18.998 sin declarar que los
universos son diferentes.

### Control contra fuga de sujeto durante el k-NN

La condición histórica permite que otros mensajes del sujeto consultado
aparezcan entre sus vecinos. Esto puede hacer que la expansión aproveche
repeticiones o temas del mismo usuario. Por eso:

- nunca incluir el propio mensaje entre sus vecinos;
- mantener la política actual como comparación;
- ejecutar una ablación que excluya **todos** los mensajes del sujeto
  consultado;
- si se aplica un máximo de vecinos por sujeto, reportar cuántos sujetos
  distintos contribuyen efectivamente a cada voto;
- usar únicamente el banco de train. No construir vecinos contra mensajes de
  dev/test.

## 7. Research questions, hipótesis y evidencia

### RQ1 — Representación y selectividad del relabeling

**Pregunta:** ¿Cómo interactúan la representación de mensajes y los
parámetros `J`/`K` para determinar cobertura, distribución y estabilidad de
las pseudoetiquetas y su efecto en detección temprana frente a no relabelar?

**Motivación:** los sistemas previos no establecen que las etiquetas a nivel
de usuario sean etiquetas verdaderas de cada mensaje. El resultado inicial
muestra además que distintas geometrías de embeddings inducen porcentajes
de relabeling muy diferentes.

**Hipótesis comprobables, no hechos asumidos:**

- H1a: al subir `K/J`, la regla debería ser más conservadora y reducir la
  proporción de mensajes cambiados.
- H1b: los encoders producirán coberturas y conjuntos de mensajes modificados
  distintos, incluso con los mismos `J/K`.
- H1c: no existe garantía de que mayor cobertura mejore macro-F1, F1 de
  suffer o latencia; la utilidad debe determinarse en dev/test a nivel de
  sujeto.
- H1d: la relación encoder × `J/K` puede diferir entre trastornos; presentar
  resultados separados antes de cualquier promedio global.

**Diseño por etapas:**

1. Mantener la tabla exploratoria actual de seis encoders como evidencia de
   que la representación afecta al proceso.
2. Mantener como shortlist provisional BETO-STS, MiniLM, RoBERTuito y E5
   (RoBERTalex opcional). La lista debe cerrarse con el grupo y justificarse
   con antecedentes, coste y compatibilidad con el baseline.
3. Correr para cada encoder una configuración común `J=10, K=5`, una pasada,
   y una condición sin relabeling. Esto completa la comparación mínima de al
   menos cuatro alternativas del enunciado bajo un MLP fijo.
4. Para la sensibilidad de parámetros, ejecutar primero con BETO-STS (o el
   encoder que el grupo confirme como principal) una cuadrícula manejable:
   `J ∈ {5, 10, 20}` y `K/J ∈ {0.5, 0.8}` (redondear `K` hacia arriba y
   garantizar `1 ≤ K ≤ J`). Son seis configuraciones por trastorno.
5. Repetir con el segundo encoder elegido como contraste al menos `J=10` con
   los dos niveles de `K/J`; si hay tiempo, ejecutar la cuadrícula completa
   también para ese encoder.
6. Para cada corrida, reportar métricas de pseudoetiquetas y luego el
   rendimiento del MLP downstream. El MLP debe tener la misma arquitectura,
   configuración de entrenamiento, regla de early detection y presupuesto.
7. Ejecutar varias semillas de inicialización del MLP para las configuraciones
   finalistas; la búsqueda de vecinos es determinista si el orden y las
   reglas de desempate no cambian.

La cuadrícula es un punto de partida, no una obligación dogmática. Si el
coste es alto, reducir encoders/configuraciones en la etapa downstream, pero
conservar el barrido de cobertura de k-NN y documentar el recorte antes de
mirar test.

**Qué medir para la parte de relabeling:**

- número y proporción de mensajes `suffer` cambiados a `control`;
- proporción de positivos restantes y balance antes/después;
- sujetos positivos con algún mensaje cambiado y distribución de fracción
  cambiada por sujeto (mediana, rango intercuartílico, percentiles);
- proporción de mensajes cambiados por quintil/posición temporal;
- solapamiento entre conjuntos relabelados de encoders/configuraciones
  (por ejemplo, Jaccard);
- tiempo y memoria si se comparan configuraciones de distinto coste.

**Qué medir para la parte downstream:**

- macro-F1 por sujeto;
- F1, precision y recall de la clase `suffer` (informar explícitamente la
  clase minoritaria);
- balanced accuracy y ROC-AUC; PR-AUC cuando sea informativa para el
  desbalance;
- métrica de early detection definida por el grupo/enunciado, latencia en
  número de mensajes y, si el protocolo lo permite, tiempo natural;
- falsas alarmas en controles y rendimiento por checkpoints/prefijos.

No seleccionar encoder o `J/K` por cuántas pseudoetiquetas cambia. Seleccionar
por la métrica primaria acordada en dev, observando también estabilidad,
latencia y coste.

### RQ2 — Política de propagación y deriva de etiquetas

**Pregunta:** ¿La iteración y las restricciones de diversidad de sujetos
reducen la dependencia de mensajes repetidos de un mismo usuario y mejoran
la utilidad downstream del relabeling frente a una pasada simple?

**Motivación:** la expansión repetida puede reforzar decisiones anteriores.
Además, varios mensajes de un mismo sujeto podrían dominar los `J` vecinos y
hacer que la evidencia de un solo usuario parezca evidencia independiente.

**Hipótesis comprobables:**

- H2a: excluir el sujeto consultado reducirá los vecinos procedentes de
  mensajes del propio usuario; la cobertura puede aumentar o disminuir, así
  que no se predice su dirección.
- H2b: un límite de vecinos por sujeto aumentará diversidad de fuentes, pero
  puede reemplazar vecinos muy similares por vecinos menos próximos.
- H2c: iterar puede cambiar más etiquetas y propagar decisiones; converger
  no implica mayor utilidad.
- H2d: una política puede mejorar detección aunque relabele menos si evita
  cambios concentrados o deriva perjudicial.

**Diseño de ablaciones, sin hacer un factorial excesivo:**

| Código | Configuración |
|---|---|
| P0 | Referencia: una pasada; se excluye el propio mensaje; otros mensajes del sujeto permitidos; sin límite por sujeto. |
| P1 | Como P0, pero se excluyen todos los mensajes del sujeto consultado. |
| P2 | Como P0, máximo de 1 vecino por sujeto. |
| P3 | Como P0, máximo de 2 vecinos por sujeto (solo si hay suficientes candidatos). |
| P4 | Iterativo hasta convergencia, con vecinos fijos y actualización síncrona. |
| P5 (opcional) | Iterativo con límite de rondas previamente fijado para estudiar deriva limitada. |

Congelar encoder y `J/K` antes de correr estas variantes. Empezar con P0–P4
en el encoder principal y repetir al menos P0/P1/P4 con el encoder contraste.
No cambiar a la vez encoder, `J/K`, número de rondas y política por sujeto.

**Qué registrar por ronda:**

- mensajes nuevos cambiados y total acumulado;
- porcentaje de positivos restante y balance por clase;
- cantidad de sujetos positivos afectados;
- estabilidad del conjunto cambiado respecto a la ronda anterior;
- número de rondas hasta convergencia y condición de parada;
- si es posible, diversidad de sujetos entre vecinos (no guardar texto).

**Interpretación:** comparar tanto con el mismo `J/K` bruto como con la
cobertura resultante. El mismo `J/K` no garantiza la misma cobertura al
excluir sujetos o limitar vecinos; reportarlo explícitamente. Si se desea
separar el efecto de la política del número de cambios, añadir como análisis
secundario una comparación aproximada a cobertura similar seleccionada solo
con datos de train/dev; no usar test para emparejar coberturas. Nunca afirmar
que iteración “mejora etiquetas” solo porque cambia más o converge.

## 8. Diseño MLP y evaluación de early detection

### Control experimental esencial

Para medir la contribución del relabeling, cada pareja de sistemas debe
diferir únicamente en las etiquetas de entrenamiento:

| Sistema | Encoder / MLP | Etiqueta train |
|---|---|---|
| MLP-Init | Iguales a la variante con relabeling | Etiqueta de sujeto heredada por mensaje |
| MLP-kNN | Iguales a MLP-Init | Pseudoetiqueta tras k-NN |

Debe usarse el mismo conjunto de mensajes, embeddings, split, arquitectura,
semillas, optimizador, criterio de parada y regla online. Si se usan pesos por
clase, la política debe ser común y estar documentada. Guardar curvas de
entrenamiento/validación suficientes para detectar colapso a una clase, pero
no seleccionar hiperparámetros mirando test.

### Flujo de predicciones

1. El MLP produce un score/probabilidad `p_i = P(suffer | mensaje_i)` para
   cada mensaje de dev/test.
2. Para cada sujeto, actualizar el score de prefijo tras cada mensaje. Usar
   la regla de acumulación acordada por el grupo (media acumulada, evidencia
   acumulada u otra), idéntica para todas las ablaciones.
3. Comparar el score con un umbral escogido en dev según la métrica primaria.
4. Definir explícitamente cuándo se emite la decisión, qué pasa si no alcanza
   umbral antes del final y cómo se mide la latencia.
5. Congelar la regla y umbral antes de test.

No usar una regla en que mensajes futuros influyan en predicciones de
prefijos. Si se compara una decisión final con una decisión temprana,
presentarlas como métricas diferentes.

### Métricas y unidad estadística

- Unidad principal para rendimiento: **sujeto**, no mensaje. No tratar los
  mensajes del mismo usuario como observaciones independientes al calcular
  incertidumbre.
- Informar conteos por clase junto con métricas: el test puede tener pocos
  sujetos, por lo que una cifra aislada puede ser inestable.
- Reportar media y desviación estándar entre semillas para el MLP.
- Para diferencia MLP-kNN vs MLP-Init, calcular diferencias pareadas por
  sujeto y, si el tamaño lo permite, intervalos bootstrap agrupados por
  sujeto. Para varias RQs/configuraciones, describir resultados como
  exploratorios si no se controla formalmente la multiplicidad.
- Si cada trastorno tiene una tarea binaria independiente, mostrar primero
  resultados separados; explicar cualquier promedio global y no mezclar
  directamente sujetos/trastornos sin justificarlo.

### Pruebas de validez

- Verificar que ningún `subject_id` aparece en más de una partición por
  trastorno.
- Verificar orden temporal y desempates por `id_message`.
- Verificar que no se haya usado `subject_id`, `id_message`, `source_partition`
  ni la etiqueta gold del sujeto como feature del MLP.
- Verificar que el banco de vecinos contiene solo train del trastorno.
- Verificar que train y dev/test tienen el mismo tratamiento de mensajes
  ausentes/cortos.
- Verificar que no se computa el score de un prefijo con mensajes posteriores.
- Ejecutar al menos un test sintético donde se conozca el resultado esperado
  de una pasada y de iteración.

## 9. Plan paso a paso con entregables y criterios de salida

Los pasos están ordenados por dependencia. Una fase se considera cerrada
cuando su criterio de salida se cumple; no basta con que el script termine.

### Paso 0 — Congelar especificación y crear registro

**Acciones**

- Acordar las decisiones de la sección 5 con el equipo/profesora.
- Escribir encoder(s), pooling, MLP, regla online, métrica primaria y política
  de mensajes cortos en el manifiesto.
- Crear una tabla de experimentos con IDs únicos antes de ejecutarlos.

**Entregable:** protocolo de una página en el informe y tabla de
configuraciones.

**Salida:** todas las decisiones que cambian el significado de la tarea están
resueltas o explícitamente marcadas como limitación.

### Paso 1 — Congelar los datos y generar embeddings completos

**Acciones**

- Reejecutar `audit_splits.py` y guardar el resultado.
- Usar el split compartido sin alterar filas; la intención actual es incluir
  todos los mensajes.
- Generar embeddings por mensaje para train/dev/test, incluyendo los mensajes
  cortos omitidos por las cachés anteriores (`min_tokens=4`).
- Guardar cachés por partición con IDs alineados y manifiestos; los mensajes
  no se escriben en informes.
- Si algún mensaje no puede codificarse, registrar explícitamente IDs,
  cantidad y motivo, y aplicar una máscara idéntica a todas las condiciones.

**Entregable:** informe de alineamiento completo y cachés reproducibles.

**Salida:** todas las matrices se alinean 1:1 con IDs/split, las dimensiones
son constantes y los límites/fugas potenciales están documentados. Los
18.998 vectores de la prueba exploratoria no bastan para cerrar esta fase.

### Paso 2 — Completar comparación mínima de alternativas

**Acciones**

- Usar el MLP acordado y evaluar al menos cuatro configuraciones
  representación+clasificador coherentes requeridas por el enunciado. Para
  aislar la rama B, comparar primero BETO-STS, MiniLM, RoBERTuito y E5 con el
  mismo MLP. Confirmar con el grupo si cuatro encoders satisfacen las cuatro
  alternativas del enunciado o si hay que variar también el clasificador.
- Mantener MLP-Init y MLP-kNN emparejados para cada configuración finalista.
- Si hay que comparar clasificadores, proponer con el grupo una CNN 1D o
  BiGRU/BiLSTM sobre secuencias de estados token-level del mismo encoder.
  Presentarlos como sistemas completos distintos, porque ya no consumen el
  mismo vector de frase que el MLP. No aplicar CNN/RNN a una única fila
  vectorial de frase como si preservara la secuencia de tokens.
- La exploración previa de seis encoders solo midió relabeling, no
  downstream.
- Usar dev para descartar configuraciones por rendimiento, inestabilidad o
  coste. No consultar test.

**Entregable:** tabla dev de al menos cuatro sistemas, con métricas y coste.

**Salida:** shortlist de encoder principal y encoder contraste, justificada
sin usar cobertura como proxy de exactitud.

### Paso 3 — Implementar y validar el estudiante MLP

**Acciones**

- Separar código de vectorización, relabeling, entrenamiento, inferencia
  temporal y métricas.
- Añadir lector de pseudoetiquetas que haga join uno-a-uno por
  `(trastorno, subject_id, id_message)` y rechace filas faltantes/duplicadas.
- Empezar con la propuesta de MLP: `Linear(d, 256) → ReLU → Dropout(0.3) →
  Linear(256, 1)`, AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch 128, máximo
  50 épocas y cinco semillas. Tratar esos valores como iniciales; seleccionar
  con train/dev y mantenerlos iguales para MLP-Init y MLP-kNN.
- Empezar sin pesos de clase en la comparación principal. Como ablación,
  calcular `pos_weight=N_control/N_suffer` solo con las etiquetas iniciales
  de train y mantenerlo fijo para Init y k-NN.
- Seleccionar early stopping/umbral en dev y documentar exactamente el
  criterio antes de test.
- Guardar predicciones por mensaje/sujeto/prefijo sin texto y un manifiesto
  por seed/configuración.

**Entregable:** baseline MLP-Init y MLP-kNN ejecutables de extremo a extremo.

**Salida:** tests sintéticos y una corrida pequeña demuestran alineamiento,
ausencia de fuga temporal, métricas reproducibles y salidas privacy-safe.

### Paso 4 — Ejecutar control downstream del relabeling

**Acciones**

- Para el encoder principal y `J=10, K=5`, entrenar MLP-Init y MLP-kNN en
  train, seleccionar el umbral/regla en dev y evaluar el protocolo en dev.
- Repetir varias semillas; comparar bajo el mismo presupuesto de
  entrenamiento.
- Reportar cambio de balance y resultados por clase; evitar interpretar
  diferencias pequeñas sin intervalo/incertidumbre.

**Entregable:** primera respuesta provisional a “¿ayuda relabelar?” en dev.

**Salida:** una comparación controlada que atribuye la diferencia a las
etiquetas débiles y no al encoder, la arquitectura o el decisor.

### Paso 5 — Responder RQ1 en train/dev

**Acciones**

- Ejecutar la cuadrícula `J={5,10,20}` y dos razones `K/J` predefinidas.
- Ejecutar todas las configuraciones de cobertura; entrenar downstream al
  menos las condiciones previstas en la sección RQ1.
- Añadir el encoder contraste de forma escalonada.
- Trazar mapas de calor de cobertura y métricas downstream; representar
  también número absoluto de mensajes y sujetos afectados.
- Seleccionar la configuración en dev usando la métrica primaria, no el
  porcentaje relabelado.

**Entregable:** tabla/figura de `J/K × encoder × trastorno`, cobertura y
rendimiento.

**Salida:** parámetros finalistas congelados y explicación de trade-offs.

### Paso 6 — Responder RQ2 con ablaciones

**Acciones**

- Con encoder y `J/K` finalistas congelados, correr P0–P4.
- Comparar cobertura, composición por sujeto, cambios por ronda y utilidad
  downstream.
- Verificar convergencia y que la etiqueta control siga siendo inmutable.
- Usar cobertura equivalente solo como análisis secundario y nunca ajustarla
  con test.

**Entregable:** tabla de ablaciones y gráfico de cambios por ronda/política.

**Salida:** determinar qué política resulta más robusta/útil o concluir
honestamente que no hay mejora clara.

### Paso 7 — Congelar experimento final y medir test

**Acciones**

- Elegir en dev una configuración final por trastorno o justificar una
  configuración común.
- Congelar encoder, MLP, semilla(s), `J/K`, política de vecinos, umbral y
  regla online.
- Ejecutar test sin retocar la configuración.
- Guardar resultados, predicciones necesarias, manifiestos y hashes/versiones.

**Entregable:** tabla final test y comparación con el control.

**Salida:** test no se vuelve a usar para buscar nuevas variantes. Si aparece
un error de implementación que invalida el resultado, explicar el cambio y
tratar la re-evaluación como una nueva corrida corregida, no como selección.

### Paso 8 — Incertidumbre, coste y análisis de fallos

**Acciones**

- Calcular media/desviación estándar entre semillas y diferencias pareadas
  agrupadas por sujeto.
- Analizar sujetos falsos positivos/falsos negativos a nivel de ID/métrica
  sin volcar mensajes sensibles a informes.
- Medir tiempo de generación de embeddings, k-NN, entrenamiento e inferencia;
  reportar hardware.
- Documentar el impacto de exclusión de 807 mensajes si las cachés no se
  reconstruyeron.

**Entregable:** tabla de incertidumbre/coste y limitaciones.

### Paso 9 — Redactar póster y empaquetar software

**Estructura sugerida del póster**

1. Problema y motivación: sujeto ≠ etiqueta de cada mensaje.
2. RQs y contribución: relabeling conservador, parámetros y diversidad.
3. Corpus, splits y tratamiento de train/dev/test.
4. Diagrama breve de pipeline: encoder por mensaje → k-NN train → MLP →
   predicciones cronológicas → decisor online.
5. Experimentos y controles.
6. Resultados principales: cobertura de pseudoetiqueta junto a métricas
   downstream; nunca cobertura sola.
7. Análisis por trastorno y latencia.
8. Limitaciones: ausencia de gold de mensaje, asimetría de relabeling,
   generalización, coste y posible efecto de mensajes cortos.
9. Conclusión vinculada explícitamente a cada RQ.

**Empaquetado**

- Incluir código fuente y pruebas; no incluir caches pesadas salvo que la
  entrega lo exija.
- Incluir instrucciones de ejecución, versiones y manifiestos.
- Verificar que el ZIP/TGZ abre y reproduce una corrida representativa.
- No incluir credenciales, rutas locales privadas ni mensajes completos si no
  son necesarios.

## 10. Matriz de experimentos recomendada

`Init` significa etiquetas iniciales heredadas del sujeto. `KNN` significa
una pasada salvo que se indique. Los valores son plan inicial y pueden
ajustarse en el Paso 0, antes de leer test.

| ID | Encoder | Train labels | `J/K` | Vecinos | Pasadas | Propósito |
|---|---|---|---|---|---|---|
| C0 | Encoder principal | Init | — | — | — | Control de ausencia de relabeling. |
| C1 | Encoder principal | KNN | 10/5 | política P0 | 1 | Comparación principal inicial. |
| E1–E4 | BETO-STS, MiniLM, RoBERTuito, E5 (propuesta) | Init y KNN | 10/5 | P0 | 1 | Alternativas de representación bajo MLP fijo; confirmar requisito con el grupo. |
| A1 (opcional) | Mismo encoder, estados token-level | Init y KNN | final | P0 | 1 | Comparar MLP con CNN 1D si se exige alternativa de arquitectura. |
| A2 (opcional) | Mismo encoder, estados token-level | Init y KNN | final | P0 | 1 | Comparar MLP con BiGRU/BiLSTM si hay tiempo y recursos. |
| R1–R6 | Principal | KNN | J 5/10/20; K/J 0,5/0,8 | P0 | 1 | Sensibilidad de RQ1. |
| R7–R10 | Contraste | KNN | J 10; K/J 0,5/0,8; ampliar si hay recursos | P0 | 1 | Interacción encoder/parámetro. |
| P1 | Principal | KNN | finalistas | Excluir todo el sujeto | 1 | Diversidad de fuentes/robustez. |
| P2 | Principal | KNN | finalistas | Máx. 1 vecino/sujeto | 1 | Evitar dominancia de autores prolíficos. |
| P3 | Principal | KNN | finalistas | Máx. 2 vecinos/sujeto | 1 | Sensibilidad al tope de contribución. |
| P4 | Principal | KNN | finalistas | P0 | Iterativa | Deriva y efecto de propagación. |
| F1 | Configuración final | Init y KNN final | congelado | congelada | congelada | Evaluación final de test, una vez. |

En cada trastorno registrar si hay instancias suficientes para todos los
vecinos, sobre todo con exclusión de sujeto. Un error de “no hay suficientes
vecinos” debe tratarse como una configuración inválida, no resolverse
reduciendo `J` silenciosamente.

## 11. Convenciones de resultados, trazabilidad y archivos

### Nombre de corridas

Usar nombres legibles y únicos, por ejemplo:

```text
dev-{disorder}-{encoder}-single-j10-k5-include-subject-seed3
dev-{disorder}-{encoder}-iterative-j10-k5-exclude-subject-seed3
test-final-{disorder}-{encoder}-{protocol-hash}
```

Separar directorios por etapa (`screening`, `rq1`, `rq2`, `final`) si mejora
la lectura. El nombre no reemplaza el manifiesto.

### Manifiesto mínimo por corrida

- `experiment_id`, fecha, etapa/RQ, trastorno y split;
- rutas/hashes de train/dev/test y universo efectivo de mensajes;
- modelo y revisión del encoder, dimensión, pooling, truncamiento,
  normalización y tokenización;
- parámetros de k-NN (`J`, `K`, métrica, pasada, exclusiones, límite por
  sujeto, desempates);
- MLP, pérdida, optimizador, hiperparámetros, semilla, epochs, mejor época;
- regla online, métrica primaria, umbral y método de selección en dev;
- versión del código, dependencias, hardware, tiempo y estado de ejecución;
- advertencias (mensajes ausentes, convergencia, configuración inválida).

### Resultados mínimos por corrida

- `run_manifest.json`;
- resumen de cobertura/distribución de etiquetas;
- historial por ronda si es iterativo;
- métricas de desarrollo/test separadas;
- scores/predicciones por sujeto/prefijo si son necesarios para calcular
  latencia o incertidumbre, sin texto.

No sobrescribir directorios; el código actual usa creación exclusiva de la
ruta de corrida. Si cambia un parámetro, debe cambiar el ID.

## 12. Cómo interpretar escenarios posibles

| Observación | Interpretación prudente | Qué no concluir |
|---|---|---|
| Más `J` o menor `K` aumenta la cobertura. | La regla acepta más mensajes como control bajo la vecindad elegida. | Que más mensajes no expresen síntomas. |
| Dos encoders generan cobertura muy diferente. | La geometría semántica y las vecindades son sensibles a la representación. | Que el encoder con menor/mayor cobertura es más exacto. |
| KNN mejora macro-F1 y/o latencia en test frente a Init, de forma consistente. | Evidencia de utilidad downstream bajo este protocolo y split. | Que las etiquetas de mensaje son gold o clínicamente correctas. |
| KNN mejora en un trastorno y empeora en otro. | La utilidad depende del trastorno; puede requerir protocolo/encoder específico y cautela por tamaño de muestra. | Un ganador global sin mostrar resultados separados. |
| Cobertura alta pero rendimiento peor. | Posible sobre-relabeling, sesgo de clase, pérdida de ejemplos informativos o desajuste train/inferencia. | Que el algoritmo es inútil para toda configuración. |
| Iteración cambia casi todo en pocas rondas. | Posible amplificación/deriva; estudiar composición de vecinos y caída de positivos. | Que convergencia es sinónimo de corrección. |
| Excluir sujeto baja cobertura y mejora rendimiento. | La evidencia intra-sujeto podía dominar vecinos; mayor diversidad puede ayudar. | Que toda señal del mismo sujeto sea fuga en el protocolo, si no se define el uso. |
| No hay diferencia estadística/intervalos amplios. | Datos insuficientes para afirmar una mejora robusta; reportar incertidumbre y efecto estimado. | Que los métodos son equivalentes sin análisis de potencia. |
| Dev mejora, test no. | Posible sobreajuste a dev o variabilidad del test; priorizar resultado test congelado. | Volver a ajustar repetidamente hasta que test mejore. |
| Mejora en métricas finales, pero peor latencia. | Existe un trade-off entre calidad final y rapidez; reportar ambas. | “Mejor early detection” basándose solo en F1 final. |

## 13. Riesgos y mitigaciones

1. **Confundir sujeto con mensaje:** describir siempre las etiquetas iniciales
   como débiles; no evaluar F1 de mensaje sin gold.
2. **Usar cobertura como selección:** elegir parámetros por efecto
   downstream en dev y reportar cobertura como diagnóstico.
3. **Leakage de sujeto en vecinos:** comparar incluir/excluir el sujeto,
   excluir el propio mensaje siempre y no tocar dev/test.
4. **Fuga temporal en evaluación:** producir y acumular scores solo con el
   prefijo observado.
5. **Diferencias de universo por mensajes cortos:** re-embebir o usar máscara
   común en toda comparación; explicar el coste.
6. **Búsqueda de parámetros en test:** congelar decisiones antes de
   consultar test una vez.
7. **Desbalance/colapso del MLP:** reportar F1 de suffer, matriz de
   confusión, pesos de clase y score de todas las clases.
8. **Diferencias entre trastornos:** entrenar y relabelar por trastorno; no
   mezclar vecinos de tareas binarias distintas.
9. **Iteración costosa/cuadrática:** usar el cálculo exacto actual mientras
   sea manejable; si se reemplaza por ANN, validar vecinos/recall del índice
   y registrar que cambia la implementación.
10. **Ampliación excesiva de alcance:** sentimiento, LLM local, ruido
    artificial, fine-tuning o nuevos encoders son extensiones opcionales, no
    requisitos para responder las dos RQ principales.
11. **Conclusiones no reproducibles:** semillas, splits compartidos,
    artefactos, versiones y IDs deben quedar guardados en la rama.

## 14. Antecedentes para justificar el diseño

Usar estas fuentes como contexto, no como prueba de que el método actual ya
está validado:

- Peix et al., MentalRiskES 2023, compararon BETO, RoBERTa-base-bne y
  `roberta-suicide-es`; el encoder especializado fue el más útil en sus
  experimentos con representaciones/regresores. Su unidad de análisis era el
  historial de usuario, distinta de esta tarea de mensaje independiente.
  [arXiv:2306.16125](https://arxiv.org/abs/2306.16125).
- UNED-GELP, MentalRiskES 2024, empleó embeddings de mensaje de Universal
  Sentence Encoder y Annoy para una estrategia de expansión/relabeling.
  Motiva estudiar vecinos semánticos, pero no hace gold la etiqueta generada.
  [CEUR paper 2](https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper2.pdf).
- ELiRF-VRAIN, MentalRiskES 2024, exploró RoBERTa y Longformer de dominio
  mental para early detection con contextos largos. No es una comparación
  directa con este protocolo de embeddings de mensajes independientes.
  [CEUR paper 3](https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper3.pdf).
- UMUTeam, MentalRiskES 2024, comparó modelos españoles e incorporó features
  de sentimiento en su enfoque. Motiva sentimiento como futura ablación, no
  como indicador clínico.
  [CEUR paper 4](https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper4.pdf).
- La documentación bibliográfica y los modelos disponibles localmente se
  resumen también en la sección de encoders del
  [`README.md`](./README.md).

## 15. Lista de tareas viva

Marcar `[x]` solo cuando el artefacto/criterio exista en la rama. Actualizar
esta lista al final de cada bloque de trabajo para que otra persona pueda
retomar el proyecto.

### Preparación y métodos

- [x] Definir el relabeler unidireccional y probar reglas sintéticas.
- [x] Auditar los splits y los metadatos de mensajes.
- [x] Ejecutar una prueba exploratoria TF-IDF/SVD separada por trastorno.
- [x] Alinear seis encoders existentes al conjunto común de train.
- [x] Comparar cobertura de relabeling a `J=10, K=5` con seis encoders.
- [x] Elegir BETO-STS y un vector por mensaje como baseline provisional.
- [x] Proponer media acumulada de scores y macro-F1 por sujeto como protocolo.
- [x] Proponer MLP sin pesos como corrida principal, con ablación de peso de
  clase fijo calculado a partir de las etiquetas iniciales de train.
- [x] Generar embeddings BETO-STS de todos los mensajes de train/dev, incluidos
  los mensajes cortos, y guardar el manifiesto.
- [x] Implementar el MLP, la selección de época/umbral en dev y la evaluación
  online por sujeto.
- [x] Ejecutar MLP-Init frente a k-NN en dev con cinco semillas y ambas
  políticas de vecinos del sujeto.
- [x] Guardar pseudoetiquetas, historiales separados por política,
  predicciones, métricas y manifiesto sin exportar textos.
- [ ] Confirmar con el grupo la regla online, métrica y tratamiento de
  desbalance.
- [ ] Confirmar con el grupo si hacen falta alternativas de clasificador
  (CNN/BiGRU token-level) además de alternativas de embeddings.
- [ ] Comparar encoders downstream con el mismo protocolo y evaluar la
  ablación de pesos de clase.
- [ ] Congelar modelo, regla y métricas antes de generar embeddings y evaluar
  test.

### RQ1

- [x] Implementar el MLP de mensajes y pruebas de alineamiento/temporalidad.
- [x] Ejecutar MLP-Init frente a MLP-kNN con encoder principal en dev.
- [ ] Evaluar al menos cuatro sistemas comparables requeridos por el
  enunciado.
- [ ] Ejecutar la sensibilidad `J/K` en train/dev.
- [x] Repetir la primera comparación con cinco semillas; falta ampliar
  repeticiones/intervalos a los experimentos finalistas.
- [ ] Congelar configuración y responder RQ1 usando test una sola vez.

### RQ2

- [x] Ejecutar la comparación inicial permitiendo y excluyendo vecinos del
  mismo sujeto; se guardaron historiales separados por condición.
- [ ] Probar topes de vecinos por sujeto y modo iterativo.
- [ ] Comparar estabilidad, cobertura, downstream y latencia.
- [ ] Congelar protocolo y responder RQ2 en test una sola vez.

### Entrega

- [ ] Calcular incertidumbre agrupada/pareada por sujeto y coste.
- [ ] Generar gráficos y tablas compactos para el póster.
- [ ] Redactar discusión crítica y limitaciones sin claims de exactitud de
  mensaje.
- [ ] Preparar y probar el ZIP/TGZ del software.
- [ ] Revisar que póster y código cumplen el enunciado.

## 16. Comando de validación tras cambios de código

Desde la raíz del repositorio:

```bash
python -m pytest supervised_splits/rama_b_relabeling -q
python -m flake8 supervised_splits/rama_b_relabeling
git diff --check
```

Para una tarea exclusivamente documental no hace falta ejecutar tests; si se
edita código relacionado, sí se deben ejecutar las pruebas pertinentes.
