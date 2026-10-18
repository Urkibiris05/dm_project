# Conclusiones acumuladas de E2/E3

## 1. Alcance

Se han estudiado dos unidades de análisis:

- **usuario**: 820 vectores, con cuatro formas de agregación;
- **mensaje**: 26.914 vectores, con tres encoders.

La evaluación principal usa `label`. `disorder` se conserva como diagnóstico
para detectar agrupamientos temáticos o por subcorpus. Las etiquetas no se
usan para elegir hiperparámetros.

## 2. Validación de DBCV

La implementación de DBCV se validó con blobs, moons, círculos y ruido
sintético. Dio valores positivos en particiones correctas de los datasets
estructurados y valores no válidos cuando había un único cluster o todo era
ruido. Por tanto, los valores negativos observados en el corpus no parecen
explicarse por un fallo básico de la métrica.

El detalle está en
[dbcv_validation.csv](./resultados/dbcv_validation.csv).

## 3. Representaciones de usuario

Se probaron MiniLM y E5 con `concat`, `mean`, `max` y `blocks`, en los
espacios original, centrado y PCA a 32, 64, 128 y 256 dimensiones. Se
evaluaron 2.112 configuraciones.

Con el protocolo estricto:

- ruido máximo del 40 %;
- cluster mayor inferior al 70 %;
- al menos dos clusters;
- DBCV positivo;
- estabilidad posterior;

el resultado fue:

| Fase | Configuraciones |
|---|---:|
| Barrido completo | 2.112 |
| Pasan ruido y cobertura | 96 |
| DBCV positivo | 0 |
| Evaluadas por estabilidad | 0 |

Conclusión: las representaciones agregadas por usuario no muestran una
estructura de densidad robusta para DBSCAN bajo estos criterios. El piloto
anterior sí encontraba clusters, pero con 70--76 % de ruido; no debe
considerarse una solución válida.

## 4. Piloto de mensajes

Se ejecutó el mismo barrido básico por encoder, con distancia coseno,
`min_samples` en `{5, 10, 20, 40}` y percentiles P10, P20, P30, P40, P50 y
P70. El piloto no calcula todavía DBCV porque una matriz completa de
distancias para 26.914 mensajes resulta demasiado costosa; primero se
descartan configuraciones claramente inviables.

### Configuraciones que cumplen ruido y cobertura

| Encoder | Configuraciones válidas básicas | Mejor ARI `label` dentro del filtro | Ruido |
|---|---:|---:|---:|
| MiniLM | 1 | 0,000 | 34,3 % |
| E5 | 3 | 0,001 | 39,8 % |
| BETO-STS | 4 | 0,019 | 39,0 % |

BETO-STS es el que presenta la mejor señal provisional para `label`, pero el
ARI sigue siendo muy bajo. Su mejor configuración dentro del filtro básico
es:

```text
min_samples=20
P30
3 clusters
38,98 % de ruido
ARI label=0,019
NMI label=0,019
```

El mayor ARI absoluto de BETO-STS (`ARI=0,037`) aparece con aproximadamente
66 % de ruido, por lo que no se considera una configuración válida.

Los resultados completos están en:

- [e2_message_minilm_pilot.csv](./resultados/e2_message_minilm_pilot.csv);
- [e2_message_e5_pilot.csv](./resultados/e2_message_e5_pilot.csv);
- [e2_message_beto-sts_pilot.csv](./resultados/e2_message_beto-sts_pilot.csv).

## 5. Interpretación actual

Los embeddings por mensaje tienen más estructura local que las
representaciones agregadas por usuario: producen varios clusters con ruido
moderado. Sin embargo, esa estructura no se alinea claramente con el riesgo.
El patrón es compatible con clusters que representan:

- tema de conversación;
- estilo lingüístico;
- grupo de Telegram;
- longitud o formato del mensaje;
- similitud semántica general.

La etiqueta de riesgo puede ser una propiedad gradual o estar concentrada en
una pequeña parte de los mensajes de cada usuario, algo que un promedio
global no conserva.

## 6. Siguiente fase

Antes de afirmar que un encoder es mejor, falta:

1. calcular DBCV de forma escalable sobre candidatas de mensajes;
2. repetir con una selección de `eps` alrededor del codo;
3. medir estabilidad;
4. agregar los clusters de mensajes a nivel de usuario;
5. comparar el cluster dominante, la distribución de mensajes y la cobertura
   por usuario;
6. comprobar si la señal de BETO-STS se mantiene sin seleccionar por ARI.

La conclusión actual es negativa para la hipótesis fuerte “DBSCAN separa
directamente `label`”, pero todavía no descarta una señal útil a nivel de
mensaje que pueda agregarse posteriormente por usuario.
