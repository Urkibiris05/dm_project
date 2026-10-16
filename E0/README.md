# E0 — ¿Es correcta nuestra implementación de DBSCAN?

Experimento E0 del diseño experimental (§6 de `estado_del_arte_vectorizacion_MentalRiskES.md`):
nuestro DBSCAN frente a scikit-learn, y coseno frente a euclídea sobre vectores
normalizados (§5.2). **No usa el corpus**: solo datos de juguete de scikit-learn.

## Cómo ejecutarlo

Desde la raíz del repositorio, en WSL:

```bash
source .venv/bin/activate
python -m E0.run_e0            # completo, ≈ 15 min (sin otras cargas)
python -m E0.run_e0 --rapido   # rejillas reducidas, ≈ 2 min
flake8 E0
```

El pseudocódigo está en `pseudocodigo.tex` (LaTeX, `algorithm2e`, compilable
con `pdflatex`; no se ha compilado en esta máquina por no haber LaTeX).

## Datos

| Dataset | Forma | Notas |
|---|---|---|
| blobs, varied, aniso, moons, circles, no_structure | 1000 × 2 | `make_*`, estandarizados |
| iris, wine, digits | 150×4, 178×13, 1797×64 | `load_*`, estandarizados |
| emb_384d | 2000 × 384 | 5 blobs con norma 1 (simula embeddings); solo E0.2 |

Las etiquetas `y` solo colorean figuras; nunca se usan para elegir parámetros.
`eps` se toma como 9 cuantiles (0,1 … 0,9) del k-dist, para `min_samples` ∈ {3, 5, 10}.
Un cuantil puede caer exactamente sobre una distancia real: eso fuerza empates
en la frontera **a propósito** (prueba de robustez numérica).

## Qué se hace

- **E0.1** (`exp_vs_sklearn.py`): 4 variantes nuestras (`brute`, `blocked`,
  `optimized_blocked`, `optimized_blocked + deduplicate_queue`) × 5 métricas
  (euclídea, manhattan, chebyshev, minkowski p=3, coseno) × 9 datasets ×
  3 `min_samples` × 9 eps = 4860 comparaciones con `sklearn.cluster.DBSCAN`.
- **E0.2** (`exp_cos_vs_l2.py`): coseno con `eps_cos` frente a euclídea con
  `eps_l2 = sqrt(2·eps_cos)`, con vectores normalizados y, como control
  negativo, sin normalizar.
- **E0.3** (`exp_libraries.py`): nuestras variantes, scikit-learn (brute,
  kd_tree, ball_tree), `dbscan` (PyPI) y `mlpack` (kd-tree y brute): concordancia
  por pares, desglose de desacuerdos, determinismo (5 repeticiones) e
  invariancia al orden (20 permutaciones).
- `verify_ties.py`: repite cada discrepancia desplazando eps un 1e-7 relativo.

## Salidas

`resultados/` (CSV)

| Archivo | Contenido |
|---|---|
| `e01_vs_sklearn.csv` | las 4860 comparaciones (igualdad exacta, ARI, AMI, diferencias en núcleo/frontera/ruido, pares empatados con eps, tiempo) |
| `e01_resumen.csv` | % de particiones idénticas por variante y métrica |
| `e01_discrepancias.csv` | solo los casos con partición distinta |
| `e01_particiones_2d.csv` | etiquetas de los datasets 2D para dibujar |
| `e02_cos_vs_l2.csv`, `e02_resumen.csv` | coseno vs euclídea, normalizado y sin normalizar |
| `e02_identidad_distancias.csv` | 2·d_cos frente a d² en 2000 pares |
| `e03_ari_pares.csv`, `e03_ari_matriz.csv` | ARI entre implementaciones |
| `e03_desacuerdos.csv` | desacuerdos con sklearn y validez de cada partición |
| `e03_reproducibilidad.csv`, `…_resumen.csv` | determinismo e invariancia al orden |
| `e0_verificacion_empates.csv` | verificación de que las discrepancias son empates |
| `e00_kdist.csv` | curvas k-dist |

`figuras/` (PNG): `e00_datasets`, `e00_kdist`, `e01_coincidencia_heatmap`,
`e01_particiones_2d`, `e01_nclusters_vs_eps`, `e02_identidad_distancias`,
`e02_cos_vs_l2`, `e03_ari_matriz`, `e03_desacuerdos`, `e03_reproducibilidad`.

## Resultados

**E0.1.** Las 4 variantes dan la partición idéntica a scikit-learn en el 100 %
de los casos con coseno, manhattan y chebyshev, el 99,2 % con minkowski p=3 y el
97,1 % (`brute`, `blocked`) / 97,5 % (`optimized_blocked`, con o sin cola
deduplicada) con euclídea. Hay 34 discrepancias, **todas** en casos con pares a
distancia ≈ eps (`n_boundary_pairs > 0`), y **las 34 desaparecen** (68/68) al
desplazar eps 1e-7: son empates de coma flotante (cada implementación calcula la
distancia con un redondeo distinto y decide distinto si `d ≤ eps` cuando `d = eps`
al límite de la precisión), no errores del algoritmo. Los núcleos casi siempre
coinciden; las diferencias son pocos puntos (ARI mínimo 0,938).

**E0.2.** La identidad `d² = 2·d_cos` se cumple con error máximo 1,3e-15. Con
vectores normalizados el 97,8 % de los casos da la misma partición (`brute`,
`blocked` y también `sklearn_brute`; 97,4 % las optimizadas). Las 32
discrepancias son de nuevo empates en la frontera (64/64 coinciden al desplazar
eps). **Sin normalizar, ninguna** partición coincide (0 %), como debe ser: la
equivalencia solo vale con norma 1.

**E0.3.** ARI medio entre nuestras variantes y scikit-learn ≈ 0,999;
`dbscan` (PyPI) 0,978 y `mlpack` 0,988. Todas las implementaciones son
deterministas (misma entrada → mismas etiquetas). Ante permutaciones de los
datos, nuestras variantes se comportan **igual que scikit-learn** (ARI medio
0,9988, mínimo 0,9875; mismo ruido siempre): lo que cambia es la asignación de
las fronteras alcanzables desde dos clusters (indeterminación conocida de
DBSCAN, Ester et al. 1996; Schubert et al. 2017), no los núcleos ni el ruido.
`mlpack` es más sensible al orden (ARI medio 0,983) y solo conserva el mismo
conjunto de ruido en el 38 % de las permutaciones; `dbscan` (PyPI) no varía con
el orden pero difiere de sklearn en algunos núcleos y fronteras.
Nuestras variantes y `sklearn_brute` difieren como mucho en unos pocos
puntos de ruido/frontera por caso, otra vez en empates (ver
`e03_desacuerdos.png`); la comprobación de validez DBSCAN en `e03_desacuerdos.csv`
usa los vecindarios de scikit-learn, así que esos mismos empates pueden marcar
`noise_correct = False` en nuestras variantes.

## Conclusión

La implementación es correcta: coincide con scikit-learn salvo en pares
situados exactamente sobre el radio, donde la decisión `d ≤ eps` depende del
redondeo. Es un límite inherente a comparar floats, no un fallo. En los
experimentos con datos reales conviene **no elegir eps en un valor que coincida
con una distancia exacta** (p. ej. un cuantil del k-dist con interpolación) si se
quiere comparar bit a bit con otra librería; para eps en general, las
particiones son estables.

## Referencias

Ester, Kriegel, Sander, Xu (1996), KDD-96 · Schubert, Sander, Ester, Kriegel, Xu
(2017), ACM TODS 42(3).
