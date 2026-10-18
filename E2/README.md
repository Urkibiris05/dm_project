# E2/E3: embeddings y representación del usuario

## Objetivo

Este experimento estudia si la representación semántica y la forma de
agregar los mensajes producen una geometría adecuada para DBSCAN.

- **E2**: compara MiniLM frente a E5.
- **E3**: compara `concat`, `mean`, `max` y `blocks` para cada modelo.

La unidad de análisis es el usuario, porque `label` está anotada a nivel de
usuario. El clustering se realiza en el espacio original de 384 dimensiones,
sin aplicar UMAP antes del clustering.

## Metodología

Para cada representación y espacio se prueban:

- distancia coseno;
- `min_samples` igual a 5, 10, 20 y 40;
- valores de `eps` definidos por los percentiles 5, 10, 15, 20, 30, 40, 50,
  60, 70, 80 y 90 de la curva k-distance;
- espacios `original`, `centered`, `pca32`, `pca64`, `pca128` y `pca256`;
- selección basada primero en DBCV, con ruido máximo del 40 %, fracción del
  cluster mayor inferior al 70 % y al menos dos clusters;
- estabilidad por 20 remuestras bootstrap con perturbación de `eps` de ±5 %.

Las etiquetas `label` y `disorder` se usan únicamente para evaluar las
particiones ya generadas mediante ARI y NMI. `label` es la evaluación
principal; `disorder` sirve para comprobar si los grupos separan el tipo de
trastorno en lugar del riesgo.

## Resultados

El barrido completo está en:

- [e2_e3_dbscan_sweep.csv](./resultados/e2_e3_dbscan_sweep.csv): todas las
  configuraciones;
- [e2_e3_internal_summary.csv](./resultados/e2_e3_internal_summary.csv):
  mejor configuración interna por representación y espacio tras comprobar
  estabilidad;
- [e2_e3_stability_shortlist.csv](./resultados/e2_e3_stability_shortlist.csv):
  configuraciones candidatas sometidas al análisis de estabilidad.

El ejecutor reproducible es
[run_e2_user.py](./run_e2_user.py).

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
