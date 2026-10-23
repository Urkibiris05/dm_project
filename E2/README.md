# E2/E3: clustering jerárquico de mensajes y usuarios

## Objetivo

Este experimento estudia si la representación semántica produce grupos de
mensajes y si la distribución de esos grupos permite agrupar posteriormente
a los usuarios.

- **E2**: aplica DBSCAN a los mensajes individuales y conserva
  `id_message` y `subject_key` en cada asignación.
- **E3**: representa cada usuario mediante la distribución de sus mensajes
  entre los clústeres de E2 y aplica un segundo DBSCAN sobre esos perfiles.

La primera unidad de análisis es el mensaje. La segunda es el usuario, pero
su vector no se obtiene promediando embeddings: se construye a partir de los
clústeres de mensajes, incluyendo una categoría para el ruido.

## Metodología

Para la primera etapa se prueban:

- distancia coseno;
- `min_samples` igual a 5, 10, 20 y 40;
- valores de `eps` definidos por los percentiles 5, 10, 15, 20, 30, 40, 50,
  60, 70, 80 y 90 de la curva k-distance;
- `min_samples` de 5, 10, 20 y 40;
- percentiles de la curva k-distance.

Para la segunda etapa se repite el barrido de `min_samples` y percentiles
sobre los histogramas de clústeres por usuario.

Las etiquetas `label` y `disorder` se usan únicamente para evaluar las
particiones ya generadas mediante ARI y NMI. `label` es la evaluación
principal; `disorder` sirve para comprobar si los grupos separan el tipo de
trastorno en lugar del riesgo.

## Resultados

El nuevo barrido jerárquico está en:

- [e2_message_hierarchical_sweep.csv](./resultados/e2_message_hierarchical_sweep.csv):
  resumen de la primera etapa;
- [e3_user_from_message_clusters.csv](./resultados/e3_user_from_message_clusters.csv):
  resultados de la segunda etapa;
- [e3_user_cluster_assignments.csv](./resultados/e3_user_cluster_assignments.csv):
  asignaciones de usuarios.

En la repetición final, DBCV se calcula para todas las configuraciones con al
menos dos clusters, antes de aplicar los filtros de ruido y cobertura.

El ejecutor reproducible es
[experiment_message_hierarchical.py](./experiment_message_hierarchical.py).

Las vectorizaciones Word2Vec y Doc2Vec se generan con
[generate_word_doc_embeddings.py](./generate_word_doc_embeddings.py). Se
entrenan sin etiquetas sobre todos los mensajes y después se promedian por
usuario para hacer la comparación E3.

## Piloto ejecutado

El primer piloto usaba un umbral de ruido del 80 % y silhouette como criterio
principal. Ese protocolo era demasiado permisivo: aceptaba configuraciones
con 70--76 % de ruido. Esta versión lo sustituye por el protocolo estricto
anterior. Si ninguna configuración cumple ruido, cobertura, DBCV y estabilidad,
el resultado correcto es declarar que no se ha encontrado una estructura de
densidad suficientemente robusta, no elegir la que tenga más silhouette.

El resultado de esta repetición está documentado en
[RESULTADO_PROTOCOLO_ESTRICTO.md](./RESULTADO_PROTOCOLO_ESTRICTO.md).

Antes de interpretar los valores negativos del corpus se validó la métrica
sobre blobs, moons, círculos y ruido sintético. La validación está en
[dbcv_validation.csv](./resultados/dbcv_validation.csv) y se reproduce con
[validate_dbcv.py](./validate_dbcv.py). DBCV fue positivo en las particiones
correctas de los tres datasets estructurados y no produjo una puntuación
válida cuando DBSCAN encontraba un único cluster o todo era ruido.

## Primer piloto a nivel de mensaje

Después de validar DBCV se hizo un piloto sobre `message__all__minilm`, con
`min_samples=10` y percentiles P10--P70. A diferencia del espacio de usuario,
el espacio de mensajes sí produce varios clusters con ruido entre el 24,6 % y
el 77,9 %. Sin embargo, la fracción del cluster mayor llega al 65,4 % con
34,3 % de ruido y el ARI/NMI de `label` permanecen prácticamente en cero.

Este resultado apunta a que los embeddings de mensaje tienen estructura local,
pero esa estructura no está alineada con la etiqueta de riesgo. El resultado
provisional está en
[e2_message_minilm_pilot.csv](./resultados/e2_message_minilm_pilot.csv).
El barrido reproducible por encoder está en
[run_e2_messages.py](./run_e2_messages.py); se ejecuta un encoder cada vez
porque el coste de vecinos y DBSCAN crece con el número de mensajes.

Las conclusiones acumuladas de usuario y mensaje están en
[CONCLUSIONES_ACUMULADAS.md](./CONCLUSIONES_ACUMULADAS.md).

## Limitaciones actuales

- La implementación de DBCV es una versión reproducible basada en
  alcanzabilidad mutua y árboles de expansión mínimos; debe validarse contra
  `evaluation/` si se incorpora ese módulo común.
- La estabilidad se calcula sobre un shortlist de las cinco mejores
  configuraciones por DBCV de cada representación y espacio.
- No se debe escoger una configuración final por el máximo ARI/NMI.
- UMAP queda reservado para visualizar resultados, no para generar el
  clustering principal.
