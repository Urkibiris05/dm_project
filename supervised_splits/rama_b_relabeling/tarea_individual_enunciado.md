# Tarea: Clasificación supervisada de texto

## Especificaciones

### Datos de entrada

- Se utilizarán los datos de la tarea previa.
- En el proyecto anterior cada equipo se familiarizó con un conjunto de datos
  y exploró, entre otras cuestiones, distintas vectorizaciones.
- En esta práctica, todos los equipos trabajan con el mismo conjunto de
  datos.
- El equipo debe acordar una partición común de **train**, **dev** y **test**
  para que los resultados se evalúen sobre los mismos conjuntos y sean
  comparables.

### Algoritmos de clasificación supervisada

- Se acordará un clasificador común (**baseline**) para que todos los
  integrantes puedan compararlo con su propuesta.
- Cada integrante explorará al menos un clasificador diferente al del resto
  de su equipo. Las propuestas deben acordarse para evitar solapamientos.

### Representación vectorial

- Cada integrante explorará al menos una representación distinta de las de
  sus compañeros, evitando solapamientos.
- Se puede enriquecer el espacio de atributos con características extraídas
  mediante *clustering*.
- La representación y el algoritmo de clasificación deben elegirse de forma
  coherente.
- Se valorará la discusión de antecedentes que justifique las elecciones y,
  especialmente, una mejora original que aborde debilidades de esos
  antecedentes.

### Research questions

Cada integrante planteará **dos preguntas de investigación originales**,
distintas de las de sus compañeros y de las de los antecedentes, y las
responderá mediante evidencia experimental. Algunos temas posibles son:

- familias de algoritmos o de vectorizaciones;
- estabilidad y robustez ante ruido, tanto en entrenamiento como en test;
- coste computacional según las parametrizaciones;
- competitividad y selección de parámetros;
- criterios de optimización, por ejemplo, la función objetivo del *fine
  tuning*;
- métricas de evaluación y mejoras estadísticamente significativas.

Las preguntas deben abordar debilidades identificadas en los antecedentes.

### Marco experimental

- Se busca un sistema (representación y clasificador) que obtenga la mejor
  F-score posible entre **al menos cuatro alternativas**.
- Si las clases están desbalanceadas, se debe considerar la F-score de la
  clase minoritaria.
- La experimentación debe ser comparable entre los integrantes (mismas
  particiones) y, a la vez, la contribución de cada uno debe ser original.
- Se valorará una experimentación exhaustiva, con métricas apropiadas
  (AUC, precisión/recall, kappa, F detallada por clase, F ponderada, etc.).
- Se recomienda realizar varios *runs* e informar la desviación estándar
  para medir estabilidad.
- Los resultados deben mostrarse de forma compacta, por ejemplo, mediante
  gráficos y tablas. Puede resultar útil
  [Seaborn](https://seaborn.pydata.org).
- Las decisiones deben justificarse mediante antecedentes y los resultados
  deben discutirse críticamente en relación con ellos.

### Formato

- Póster científico-técnico en papel, tamaño **DIN-A3**. Las sugerencias de
  formato están disponibles en eGela, en «Póster: especificaciones».
- Los resultados deben presentarse de forma compacta (tablas, gráficos,
  etc.) y con un estilo científico-técnico formal.

## Grupos de trabajo

La tarea es individual y se coordina con el equipo de trabajo del proyecto.

## Entregables

Cada estudiante entregará en eGela:

1. **Software:** código fuente documentado. Se permiten librerías externas.
   - Entregar un único archivo comprimido `.tgz` (`.tar.gz`) o `.zip`.
   - No se admiten archivos `.rar` ni otros formatos.
2. **Póster:** debe mostrar la experimentación exhaustiva, los resultados y
   la discusión.
   - Formato: PDF.

### Problemas de espacio

Si no es posible subir toda la entrega por falta de espacio, se debe subir a
eGela al menos el póster en PDF y el código fuente. En el cuadro de diálogo se
puede incluir un enlace público de descarga directa para el resto de las
aportaciones que se consideren necesarias.

## Presentación oral

La semana siguiente a la entrega se realizará una presentación en el
laboratorio. Los pósteres se expondrán organizados por equipo, como en una
presentación de pósteres de un congreso.
