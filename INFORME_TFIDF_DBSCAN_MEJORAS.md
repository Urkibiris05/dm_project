# Informe de TF-IDF, DBSCAN y mejoras modulares

## 1. Alcance

Este documento recoge:

- las vectorizaciones TF-IDF generadas para MentalRiskES;
- las pruebas realizadas con `sklearn.cluster.DBSCAN`;
- las métricas obtenidas;
- las conclusiones del baseline;
- las mejoras modulares añadidas a la implementación propia de DBSCAN;
- las validaciones ejecutadas;
- las siguientes comparaciones recomendadas.

Los resultados de las vectorizaciones no se guardan dentro del repositorio,
sino en:

```text
/home/urkibiris/datasets/vectorizations_tfidf/
```

Los resultados CSV de las pruebas se encuentran en esa misma carpeta:

- `sklearn_dbscan_results.csv`: pruebas con radios estimados a partir de
  distancias al k-ésimo vecino;
- `sklearn_dbscan_eps_sweep.csv`: barrido exploratorio de `eps`.

## 2. Vectorizaciones disponibles

Se localizaron cinco vectorizaciones TF-IDF:

| Vectorización | Filas | Dimensiones | Unidad |
|---|---:|---:|---|
| TF-IDF por palabras | 15 327 | 128 | mensaje |
| TF-IDF por caracteres (`char_wb`) | 15 327 | 128 | mensaje |
| TF-IDF por caracteres ajustado solo con Train | 15 327 | 128 | mensaje |
| TF-IDF por palabras | 485 | 128 | usuario |
| TF-IDF por caracteres (`char_wb`) | 485 | 128 | usuario |

Todas las matrices tienen tipo `float32`. Las matrices de caracteres y las de
usuarios están normalizadas en L2. La matriz de palabras por mensaje contiene
3 vectores nulos.

El corpus cargado contiene:

- 15 327 mensajes;
- 485 usuarios;
- 5 328 mensajes de `Ansiedad`;
- 9 999 mensajes de `TCA`;
- 9 321 mensajes en `Test`;
- 5 641 mensajes en `Train`;
- 365 mensajes en `Trial`.

Las etiquetas binarias contienen:

- 8 407 ejemplos con `label=0`;
- 6 920 ejemplos con `label=1`.

En estas vectorizaciones no aparece `Depresion`.

## 3. Evaluación inicial con scikit-learn

Se utilizó `sklearn.cluster.DBSCAN` con:

- métrica euclídea;
- algoritmo de fuerza bruta;
- `min_samples=5` en el barrido principal;
- matrices ya normalizadas;
- evaluación mediante:
  - ARI;
  - NMI;
  - homogeneidad;
  - completitud;
  - número de clusters;
  - porcentaje de ruido.

También se probó `min_samples=10` para las vectorizaciones a nivel de
usuario.

### 3.1. Selección exploratoria de `eps`

Primero se estimaron radios utilizando la distancia al k-ésimo vecino. Los
valores obtenidos fueron aproximadamente:

- mensajes con caracteres: `0.84 - 0.94`;
- mensajes con palabras: `0.78 - 0.99`;
- usuarios con caracteres: `0.89 - 1.13`;
- usuarios con palabras: `1.01 - 1.19`.

Después se realizó un barrido más amplio:

- mensajes: `eps` desde `0.20` hasta `1.00`;
- usuarios: `eps` desde `0.20` hasta `1.30`;
- paso de `0.05`;
- `min_samples=5`.

Este barrido es exploratorio. No debe considerarse una evaluación final
independiente porque la selección del mejor `eps` utiliza las etiquetas para
comparar resultados.

## 4. Mejores resultados encontrados

El mejor ARI obtenido en el barrido fue:

| Vectorización | `eps` | Clusters | Ruido | ARI | NMI |
|---|---:|---:|---:|---:|---:|
| TF-IDF palabras, mensaje | 0.80 | 2 | 27.2 % | **0.0139** | 0.0067 |
| TF-IDF palabras, mensaje | 0.75 | 18 | 40.0 % | 0.0137 | 0.0088 |
| TF-IDF palabras, usuario | 0.80 | 3 | 87.8 % | 0.0065 | 0.0687 |
| TF-IDF caracteres, usuario | 0.65 | 3 | 89.1 % | 0.0062 | 0.0303 |

El resultado de TF-IDF por palabras y mensaje es el mejor de los probados,
pero el ARI sigue siendo muy bajo.

### 4.1. Resultados por partición del mejor baseline

Para TF-IDF por palabras a nivel de mensaje, `eps=0.80` y `min_samples=5`:

| Split | Filas | Clusters | Ruido | ARI | NMI |
|---|---:|---:|---:|---:|---:|
| Test | 9 321 | 2 | 28.8 % | 0.0119 | 0.0072 |
| Train | 5 641 | 2 | 25.0 % | 0.0160 | 0.0057 |
| Trial | 365 | 1 | 20.3 % | 0.0074 | 0.0018 |

Para TF-IDF por palabras a nivel de usuario, `eps=0.80`:

| Split | Filas | Clusters | Ruido | ARI | NMI |
|---|---:|---:|---:|---:|---:|
| Test | 300 | 3 | 85.3 % | 0.0101 | 0.1018 |
| Train | 175 | 3 | 92.6 % | -0.0093 | 0.0198 |
| Trial | 10 | 2 | 80.0 % | 0.0481 | 0.2461 |

El resultado de `Trial` no es representativo por su tamaño reducido.

## 5. Interpretación del baseline

El patrón dominante de DBSCAN sobre TF-IDF es:

1. un único cluster gigante cuando `eps` es suficientemente grande;
2. muchos grupos pequeños y una cantidad alta de ruido cuando `eps` se reduce.

No aparece una separación robusta de las etiquetas Gold.

La conclusión actual es:

> TF-IDF sirve como baseline débil o negativo, pero no parece una
> representación competitiva para DBSCAN en este corpus.

Esto no demuestra que DBSCAN esté mal implementado. Indica que las
representaciones TF-IDF reducidas mediante SVD no presentan regiones de
densidad claramente separadas según las etiquetas disponibles.

Los tiempos de scikit-learn fueron aproximadamente:

- mensajes: `0.7 - 1.0` segundos por ejecución;
- usuarios: `0.01 - 0.03` segundos por ejecución.

## 6. Mejoras modulares de nuestro DBSCAN

La implementación existente no se ha reemplazado. Las rutas anteriores siguen
siendo válidas:

```python
DBSCAN(algorithm="brute")
DBSCAN(algorithm="blocked")
```

Las mejoras se activan explícitamente.

### 6.1. Búsqueda por bloques optimizada

Archivo:

```text
dbscan_implementation/neighbors/optimized_blocked.py
```

Nueva opción:

```python
DBSCAN(algorithm="optimized_blocked")
```

Características:

- Euclídea:
  - compara distancias cuadradas con `eps**2`;
  - evita calcular la raíz cuadrada de todas las distancias.
- Coseno:
  - normaliza el conjunto una sola vez;
  - compara productos escalares con `1 - eps`.
- Otras distancias:
  - utiliza la ruta genérica existente.
- Se mantiene el cálculo por bloques.

También puede controlarse el presupuesto aproximado de la matriz temporal:

```python
from dbscan_implementation.neighbors import (
    OptimizedBlockedBruteForceSearch,
)

search = OptimizedBlockedBruteForceSearch(memory_limit_mb=128)
model = DBSCAN(eps=0.8, min_samples=5, algorithm=search)
labels = model.fit_predict(X)
```

El límite se refiere a la matriz temporal de distancias; no incluye toda la
memoria del proceso ni la lista final de vecindarios.

### 6.2. Cola de expansión deduplicada

Archivo:

```text
dbscan_implementation/dbscan.py
```

Nueva opción:

```python
DBSCAN(
    algorithm="optimized_blocked",
    deduplicate_queue=True,
)
```

La opción evita insertar varias veces un mismo punto en la cola de expansión
antes de procesarlo. Por defecto está desactivada para preservar el
comportamiento anterior.

### 6.3. Monitor temporal de memoria

Archivo:

```text
dbscan_implementation/profiling/memory_monitor.py
```

El monitor muestrea `VmRSS` desde `/proc/self/status` en Linux o WSL y guarda
una serie temporal:

```text
seconds,rss_mb
```

Además del pico, permite calcular:

- memoria media;
- percentil 95;
- memoria máxima;
- tiempo hasta el pico;
- duración;
- área bajo la curva en `MB·s`.

El muestreo solo se activa cuando se solicita una traza, por lo que las
mediciones normales no incluyen ese sobrecoste.

### 6.4. Integración con el benchmark

Archivos modificados:

- `dbscan_implementation/profiling/benchmark.py`;
- `dbscan_implementation/profiling/measure.py`;
- `dbscan_implementation/profiling/implementations.py`;
- `dbscan_implementation/profiling/plots.py`.

El benchmark incluye ahora:

```text
ours_optimized_blocked
```

Las trazas temporales se pueden guardar como un CSV por caso, permitiendo
estudiar cómo evoluciona la memoria durante la ejecución y no solo su máximo.

Las librerías externas `dbscan` y `mlpack` se tratan como dependencias
opcionales del benchmark. Si no están instaladas, las variantes externas
fallan de forma explícita, pero las implementaciones propias y scikit-learn
siguen pudiéndose medir.

## 7. Tests y validación

Se añadieron tests para:

- equivalencia de la búsqueda optimizada con fuerza bruta;
- Euclídea optimizada;
- coseno optimizado;
- Manhattan en la ruta genérica;
- presupuesto de memoria del bloque;
- expansión con cola deduplicada;
- recogida de muestras de memoria;
- escritura de trazas CSV.

Validación realizada:

```text
82 passed
```

También se comprobó:

- compilación con `compileall`;
- ausencia de errores de formato con `git diff --check`;
- generación de trazas RSS;
- equivalencia de vecindarios entre rutas optimizadas y referencia.

## 8. Estado de los resultados de rendimiento

Todavía no se ha realizado una comparación completa y definitiva entre:

```text
ours_blocked
ours_optimized_blocked
ours_optimized_blocked + deduplicate_queue
```

La infraestructura para hacerla ya está preparada. El benchmark pequeño
ejecutado durante la validación confirmó que la nueva implementación aparece
correctamente y que las trazas de memoria se generan.

Las variantes externas `dbscan` y `mlpack` no se pudieron ejecutar en el
entorno actual porque esos paquetes no están instalados.

## 9. Próximos experimentos recomendados

La siguiente comparación debe mantener constantes:

- conjunto de datos;
- métrica;
- `eps`;
- `min_samples`;
- número de hilos;
- número de repeticiones.

Se deben comparar, como mínimo:

1. fuerza bruta original;
2. bloques original;
3. bloques optimizados;
4. bloques optimizados con cola deduplicada.

Para cada ejecución conviene guardar:

- tiempo mediano;
- memoria RSS inicial;
- memoria RSS final;
- pico RSS;
- media RSS;
- percentil 95;
- tiempo hasta el pico;
- área `MB·s`;
- número de clusters;
- porcentaje de ruido;
- ARI frente a la implementación de referencia.

La interpretación esperada es:

- la versión optimizada debe conservar las mismas etiquetas;
- la comparación sin raíz cuadrada puede mejorar el tiempo en Euclídea;
- la ruta coseno puede reducir trabajo con embeddings normalizados;
- la cola deduplicada puede ayudar sobre clusters densos;
- el presupuesto de bloque debe reducir los picos temporales, aunque puede
  aumentar el tiempo si fuerza bloques demasiado pequeños.

## 10. Limitaciones actuales

- El monitor RSS es específico de Linux/WSL.
- El límite de `memory_limit_mb` controla la matriz temporal, no el consumo
  total del proceso.
- DBSCAN sigue teniendo coste temporal cuadrático en la búsqueda por bloques.
- La lista completa de vecindarios continúa conservándose antes de expandir
  clusters.
- El barrido exploratorio de `eps` no debe utilizarse como evaluación final
  independiente.
- El baseline TF-IDF todavía no se ha conectado con una evaluación completa
  de todas las configuraciones Transformer.
- No se han instalado todavía las dependencias opcionales `dbscan` y
  `mlpack` del benchmark.

