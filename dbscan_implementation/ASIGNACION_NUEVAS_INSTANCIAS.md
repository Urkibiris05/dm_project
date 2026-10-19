# Asignación de nuevas instancias en DBSCAN

## 1. El problema

DBSCAN es un algoritmo de *clustering*, no un clasificador tradicional.
Durante `fit`, agrupa los datos disponibles y decide qué puntos son:

- **Puntos núcleo**: tienen al menos `min_samples` vecinos dentro de
  `eps`.
- **Puntos frontera**: están cerca de un punto núcleo, pero no tienen
  suficiente densidad para ser núcleo.
- **Ruido**: no pertenecen a ningún cluster.

DBSCAN no tiene un `predict` estándar porque una nueva instancia podría,
en teoría, cambiar la estructura de densidad: podría convertirse en punto
núcleo o conectar dos clusters. Por eso, en esta implementación el método
se llama `assign` (asignar).

`assign` mantiene los clusters calculados durante `fit` y solo decide a qué
cluster pertenece cada instancia nueva.

## 2. Uso básico

```python
from dbscan_implementation import DBSCAN

X_train = [
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 7],
    [8, 8],
    [25, 80],
]

model = DBSCAN(eps=3, min_samples=2).fit(X_train)

X_new = [
    [2, 2],       # Cerca del primer cluster
    [8, 7],       # Cerca del segundo cluster
    [100, 100],   # No está cerca de ningún cluster
]

labels = model.assign(X_new)
print(labels)
```

El resultado es un array de etiquetas:

```text
[ 0,  1, -1]
```

Como en DBSCAN, la etiqueta `-1` significa **ruido**.

## 3. Qué información se guarda en el modelo

Después de ejecutar `fit`, el modelo conserva la información necesaria para
asignar nuevos puntos:

### `components_`

Contiene las coordenadas de todos los puntos núcleo:

```python
model.components_
```

No es necesario guardar todos los puntos originales. Para asignar una nueva
instancia basta con compararla con los puntos núcleo.

### `core_sample_labels_`

Contiene el cluster al que pertenece cada punto de `components_`.

Los dos arrays están alineados:

```text
components_[i]          -> coordenadas del punto núcleo i
core_sample_labels_[i]  -> cluster del punto núcleo i
```

Por ejemplo, si:

```python
model.components_[3] == [8, 8]
model.core_sample_labels_[3] == 1
```

significa que el punto núcleo `[8, 8]` pertenece al cluster `1`.

### `eps`

Es la distancia máxima permitida para considerar que una nueva instancia
está cerca de un punto núcleo.

### `metric` y `p`

Definen cómo se calcula la distancia. Se reutilizan exactamente los mismos
parámetros usados durante `fit`, por ejemplo:

- distancia euclídea;
- Manhattan;
- Chebyshev;
- Minkowski;
- coseno;
- una implementación propia de `Distance`.

### `n_features_in_`

Guarda el número de dimensiones de los datos de entrenamiento. Sirve para
comprobar que las nuevas instancias tienen el mismo formato.

## 4. Cómo funciona `assign`

Para cada instancia nueva se siguen estos pasos:

1. Se comprueba que el modelo se haya ajustado previamente con `fit`.
2. Se comprueba que la entrada sea una matriz de dos dimensiones.
3. Se comprueba que tenga el mismo número de características que los datos
   de entrenamiento.
4. Se calcula la distancia entre la nueva instancia y todos los puntos
   núcleo.
5. Se selecciona el punto núcleo más cercano.
6. Si su distancia es menor o igual que `eps`, se devuelve el cluster de
   ese núcleo.
7. Si la distancia es mayor que `eps`, se devuelve `-1`.

De forma simplificada:

```text
para cada punto nuevo:
    núcleo = núcleo más cercano
    si distancia(punto nuevo, núcleo) <= eps:
        asignar cluster del núcleo
    si no:
        asignar ruido (-1)
```

## 5. Importante: `assign` no vuelve a entrenar DBSCAN

La asignación es deliberadamente conservadora:

- una instancia nueva no se convierte en punto núcleo;
- una instancia nueva no conecta dos clusters;
- los clusters encontrados durante `fit` no cambian;
- los puntos nuevos no se añaden a `labels_`;
- el modelo no se modifica durante la asignación.

Esta decisión permite tratar el modelo ajustado como una estructura estable
para procesar datos que llegan posteriormente.

Si se quiere que los nuevos datos influyan en los clusters, hay que ejecutar
de nuevo:

```python
model.fit(X_completo)
```

donde `X_completo` contiene los datos originales y los nuevos.

## 6. Casos especiales y errores

### Modelo sin ajustar

Esto produce un error:

```python
DBSCAN().assign([[1, 2]])
```

Primero hay que ejecutar `fit`.

### Número de dimensiones incorrecto

Si el modelo se ajustó con datos de dos dimensiones, las nuevas instancias
también deben tener dos dimensiones:

```python
model.fit([[0, 0], [1, 1]])
model.assign([[2]])  # Error
```

### Modelo sin puntos núcleo

Si todos los puntos de entrenamiento fueron considerados ruido, no existe
ningún cluster al que asignar nuevas instancias. En ese caso `assign`
devuelve `-1` para todos los puntos nuevos.

### Entrada vacía

Una entrada con cero filas devuelve un array vacío de etiquetas, siempre que
mantenga la forma correcta `(0, n_dimensiones)`.

## 7. Relación con los tests

La implementación incluye pruebas para comprobar que:

- los puntos nuevos se asignan al cluster correcto;
- los puntos alejados se marcan como ruido;
- se elige el núcleo más cercano;
- se rechaza el uso de `assign` antes de `fit`;
- se rechaza un número de dimensiones incorrecto.

Estas pruebas están en
[`dbscan_implementation/tests/test_dbscan.py`](./tests/test_dbscan.py).

## 8. Resumen

La información mínima necesaria para asignar nuevas instancias es:

```text
puntos núcleo + etiqueta de cada punto núcleo + eps + definición de distancia
```

La implementación guarda esa información en `components_`,
`core_sample_labels_`, `eps`, `metric` y `p`. El método `assign` compara cada
nueva instancia con los puntos núcleo, utiliza el más cercano y devuelve su
cluster si está dentro del radio `eps`; en caso contrario, devuelve `-1`.
