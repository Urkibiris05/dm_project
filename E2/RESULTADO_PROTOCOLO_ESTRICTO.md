# Resultado del protocolo estricto E2/E3

## Protocolo ejecutado

Se repitió el barrido sobre las ocho representaciones de usuario:

- MiniLM y E5;
- `concat`, `mean`, `max` y `blocks`.

Para cada representación se probaron seis espacios:

- `original`;
- `centered`;
- `pca32`;
- `pca64`;
- `pca128`;
- `pca256`.

En cada espacio se probaron cuatro valores de `min_samples` (`5`, `10`,
`20`, `40`) y once cuantiles de la curva k-distance. En total:

```text
8 representaciones × 6 espacios × 4 min_samples × 11 cuantiles
= 2.112 configuraciones
```

La selección no utiliza `label` ni `disorder`. Una configuración solo puede
pasar a DBCV si cumple:

- al menos 2 clusters;
- ruido `<= 40 %`;
- fracción del cluster mayor `< 70 %`.

Después se exige `DBCV > 0`. La estabilidad bootstrap con 20 remuestras y una
perturbación de `eps` de `±5 %` solo se aplica a las candidatas que superan
todos esos filtros.

## Resultado

| Fase | Configuraciones |
|---|---:|
| Barrido total | 2.112 |
| Con al menos 2 clusters, ruido `<=40 %` y cluster mayor `<70 %` | 96 |
| Con DBCV positivo | 0 |
| Candidatas sometidas a estabilidad | 0 |

Los valores máximos de DBCV por representación y espacio son negativos. Por
ejemplo, los mejores valores observados fueron aproximadamente:

- MiniLM `concat`, `centered`: `-0,193`;
- MiniLM `max`, `centered`: `-0,099`;
- E5 `concat`, `centered`: `-0,098`;
- E5 `max`, `centered`: `-0,088`.

Los espacios originales producen con frecuencia un único cluster o un
porcentaje de ruido demasiado alto. PCA y centrado mejoran la cobertura en
algunas configuraciones, pero no producen una separación de densidad positiva
según DBCV.

## Interpretación

Con este protocolo no se ha encontrado una estructura de densidad robusta en
las representaciones de usuario analizadas. No es correcto escoger como
"mejor" la configuración con mayor silhouette del piloto anterior, porque
aquellas configuraciones tenían entre aproximadamente 70 % y 76 % de ruido.

El archivo
[e2_e3_internal_summary.csv](./resultados/e2_e3_internal_summary.csv) queda
vacío salvo por la cabecera porque no existe ninguna configuración válida que
pueda pasar a la fase de estabilidad.

Los resultados completos están en:

- [e2_e3_dbscan_sweep.csv](./resultados/e2_e3_dbscan_sweep.csv);
- [e2_e3_stability_shortlist.csv](./resultados/e2_e3_stability_shortlist.csv).

Este resultado no es directamente comparable con la captura del experimento
de otro miembro si aquella utiliza embeddings por mensaje. Aquí la unidad es
el usuario, que es la unidad coherente con las etiquetas disponibles.
