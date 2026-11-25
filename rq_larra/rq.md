# Research Questions — MentalRiskES: incertidumbre y detección temprana

## 0. Contexto general del proyecto

Estamos desarrollando un proyecto académico de *supervised learning* utilizando el dataset **MentalRiskES 2023**, que contiene historiales de mensajes de usuarios en español y etiquetas asignadas originalmente a nivel de usuario.

El objetivo general es investigar cómo mejorar la detección temprana de indicadores de riesgo de salud mental a partir de mensajes escritos por los usuarios.

### Baseline de referencia

El sistema de referencia debe mantener los siguientes componentes:

1. **Representación de mensajes:** utilizar embeddings precomputados en una práctica anterior. Cada mensaje tiene su propio vector.
2. **Relabeling de mensajes:** trasladar las etiquetas originales de usuario a mensajes y ampliar la clase negativa mediante un procedimiento iterativo basado en *nearest neighbours*:
   - Inicialmente, los mensajes heredan la etiqueta de su usuario.
   - Se identifican mensajes de usuarios negativos.
   - Se buscan mensajes semánticamente cercanos mediante k-NN.
   - Se amplía progresivamente la clase negativa siguiendo un criterio de mayoría de vecinos negativos.
   - Se repite el procedimiento hasta convergencia.
3. **Clasificador:** MLP binario que recibe el vector de cada mensaje y produce un logit, transformado mediante sigmoid en una probabilidad de riesgo.
4. **Agregación del historial:** utilizar *mean pooling* de las predicciones de los mensajes disponibles hasta el momento para obtener una puntuación acumulada por usuario.
5. **Decisión:** activar una alarma cuando la puntuación supera un umbral fijo.

Referencia metodológica para el relabeling:

- Fernández-Hernández et al. (2024), *UNED-GELP at MentalRiskES 2024: Transformer-Based Encoders and Similarity Techniques for Early Risk Prediction of Mental Disorders*.
- PDF: https://ceur-ws.org/Vol-3756/MentalRiskES2024_paper2.pdf

Es importante comprobar la implementación exacta del algoritmo de relabeling: número de vecinos, umbral de vecinos negativos, clases que pueden modificarse, exclusión de mensajes del mismo usuario y criterio de convergencia.

**Importante:** el relabeling genera pseudoetiquetas a nivel de mensaje. Estas no deben interpretarse como etiquetas clínicas verificadas. La evaluación final debe utilizar las etiquetas originales de usuario cuando corresponda.

---

# RQ1 — Confidence thresholds, uncertainty estimation and selective classification

## 1. Research question

**Can confidence-based and uncertainty-aware decision strategies improve the reliability of mental health risk predictions in Spanish social media messages?**

Queremos estudiar si es posible mejorar la fiabilidad de las predicciones del baseline utilizando umbrales de confianza, medidas de incertidumbre y mecanismos de abstención.

La idea principal es que el modelo no tenga que emitir obligatoriamente una clasificación para todos los mensajes. En algunos casos puede ser preferible reconocer que no dispone de suficiente información y abstenerse de predecir.

## 2. Motivación

El MLP produce una probabilidad de riesgo para cada mensaje:

\[
p(y=1\mid x)
\]

Por ejemplo:

- Mensaje A: probabilidad de riesgo = 0.95.
- Mensaje B: probabilidad de riesgo = 0.53.
- Mensaje C: probabilidad de riesgo = 0.08.

Un umbral convencional de 0.5 clasificaría A y B como positivos, y C como negativo.

Sin embargo, la predicción de B es mucho más ambigua que la de A.

La investigación debe estudiar si podemos identificar estos casos ambiguos y tratarlos de manera diferente.

Hay que distinguir tres conceptos:

- **Confidence:** grado de confianza que el modelo asigna a su predicción.
- **Uncertainty:** medida de la incertidumbre asociada a la predicción.
- **Decision threshold:** valor establecido para decidir si aceptamos una predicción, emitimos una alarma o nos abstenemos.

La confianza y la incertidumbre no son exactamente lo mismo que el umbral. El umbral es una regla de decisión aplicada a las puntuaciones del modelo.

## 3. Métodos que se deben implementar y comparar

### 3.1. Baseline: fixed probability threshold

Utilizar un umbral fijo sobre la probabilidad de riesgo.

\[
\hat y =
\begin{cases}
1 & p(y=1\mid x)\geq t\\
0 & p(y=1\mid x)<t
\end{cases}
\]

Empezar con \(t=0.5\).

Posteriormente, seleccionar otros valores de umbral utilizando exclusivamente validación:

\[
t\in\{0.3,0.4,0.5,0.6,0.7,0.8,0.9\}
\]

Esto permitirá estudiar cómo cambia el compromiso entre falsos positivos y falsos negativos.

### 3.2. Confidence threshold

Utilizar la confianza de la clase más probable:

\[
Confidence(x)=\max(p,1-p)
\]

En clasificación binaria:

- Si \(p=0.95\), la confianza es 0.95.
- Si \(p=0.53\), la confianza es 0.53.
- Si \(p=0.08\), la confianza es 0.92.

Si la confianza no alcanza un umbral \(c\), el modelo se abstiene.

\[
\hat y =
\begin{cases}
1 & p\geq 0.5 \land Confidence(x)\geq c\\
0 & p<0.5 \land Confidence(x)\geq c\\
Abstain & Confidence(x)<c
\end{cases}
\]

Probar diferentes valores de \(c\), seleccionados mediante validación.

Es importante distinguir este mecanismo de un umbral de alarma sobre la probabilidad positiva. Un umbral de alarma decide cuándo clasificar un caso como positivo; un umbral de confianza decide cuándo el modelo dispone de suficiente seguridad para emitir cualquier clasificación.

### 3.3. Entropy-based uncertainty

Calcular la entropía de la distribución de probabilidades:

\[
H(p)=-p\log(p)-(1-p)\log(1-p)
\]

Interpretación:

- Entropía baja: las probabilidades están concentradas en una clase.
- Entropía alta: las probabilidades están más repartidas y existe mayor ambigüedad.

En clasificación binaria, la entropía máxima se alcanza cuando \(p=0.5\).

Se puede normalizar:

\[
H_{norm}(p)=\frac{H(p)}{\log 2}
\]

De esta forma, la incertidumbre queda comprendida entre 0 y 1.

Definir una regla de abstención:

\[
\hat y =
\begin{cases}
Prediction(x) & H_{norm}(p)\leq u\\
Abstain & H_{norm}(p)>u
\end{cases}
\]

Probar diferentes valores de \(u\).

Comparar este enfoque con el umbral de confianza para comprobar si la entropía aporta información adicional.

**Nota:** en un problema binario con probabilidades perfectamente complementarias, la entropía es una transformación de la confianza máxima. Por ello, ambas medidas pueden producir ordenaciones equivalentes de los ejemplos. No asumir que son métodos independientes ni que uno tiene que superar al otro.

### 3.4. Probability calibration

Antes de interpretar una probabilidad como una medida fiable de confianza, comprobar si el modelo está calibrado.

Un clasificador está calibrado cuando, entre los ejemplos a los que asigna una probabilidad del 80 %, aproximadamente el 80 % pertenece realmente a la clase predicha.

Comparar, si el tamaño del conjunto de validación lo permite:

- Probabilidades originales del MLP.
- Temperature scaling.
- Platt scaling.
- Isotonic regression.

La calibración debe ajustarse con datos de validación o con un subconjunto independiente de calibración, nunca con test.

No asumir que calibrar mejora automáticamente el Macro-F1. La calibración busca mejorar la interpretación de las probabilidades, no necesariamente la capacidad de separar las clases.

### 3.5. Selective classification / abstention

Implementar una estrategia en la que el modelo pueda devolver tres resultados:

- Positive.
- Negative.
- Abstain.

La abstención se activa cuando la confianza es baja o la incertidumbre es elevada.

El objetivo es comprobar si podemos reducir los errores entre las predicciones aceptadas, aunque eso implique clasificar un porcentaje menor de mensajes.

Construir curvas de:

- Coverage: proporción de ejemplos en los que el modelo emite una predicción.
- Selective risk: tasa de error entre los ejemplos aceptados.

\[
Coverage=\frac{N_{accepted}}{N_{total}}
\]

\[
SelectiveRisk=\frac{N_{errors,accepted}}{N_{accepted}}
\]

Comparar los resultados para distintos niveles de coverage.

## 4. Hipótesis experimentales

- **H1:** los umbrales de confianza pueden reducir los errores de clasificación entre los mensajes aceptados, a costa de reducir el porcentaje de mensajes clasificados.
- **H2:** las medidas de incertidumbre basadas en entropía pueden identificar mensajes ambiguos y ofrecer una alternativa al umbral de confianza.
- **H3:** la calibración de probabilidades puede mejorar la utilidad de los umbrales de decisión, especialmente cuando se interpretan como estimaciones de riesgo.
- **H4:** la abstención puede mejorar la fiabilidad de las predicciones aceptadas, pero no necesariamente el rendimiento global si se considera también el porcentaje de mensajes sin clasificar.

## 5. Métricas de evaluación

Utilizar:

- Macro-F1.
- Precision de la clase positiva.
- Recall de la clase positiva.
- False Positive Rate (FPR).
- False Negative Rate (FNR).
- Expected Calibration Error (ECE).
- Brier Score.
- Coverage.
- Selective risk.

Las métricas de clasificación deben calcularse de forma coherente con el nivel de evaluación elegido. Si la tarea final es clasificar usuarios, no presentar una métrica de mensajes como si fuera una métrica de usuarios.

La métrica primaria debe definirse antes de ejecutar los experimentos.

## 6. Referencias bibliográficas para RQ1

Consultar los siguientes trabajos:

1. **Fisch, Jaakkola & Barzilay (2022). _Calibrated Selective Classification_. Transactions on Machine Learning Research.**
   - Tema: clasificación selectiva, abstención y calibración de incertidumbre.
   - Especialmente relevante para coverage y selective risk.
   - https://openreview.net/forum?id=zFhNBs8GaV
   - PDF: https://openreview.net/pdf/af329ecf778809c5d45b01f6da3400d570d872cb.pdf

2. **Silva Filho et al. (2023). _Classifier Calibration: A Survey on How to Assess and Improve Predicted Class Probabilities_. Machine Learning, 112, 3211–3260.**
   - Tema: calibración, reliability diagrams, ECE, Brier Score, Platt scaling, isotonic regression y temperature scaling.
   - DOI: https://doi.org/10.1007/s10994-023-06336-7
   - https://classifier-calibration.github.io/survey/

3. **Ulmer, Frellsen & Hardmeier (2022). _Exploring Predictive Uncertainty and Calibration in NLP: A Study on the Impact of Method & Data Scarcity_. Findings of EMNLP 2022, pp. 2707–2735.**
   - Tema: incertidumbre predictiva en NLP, calibración, ensembles y escasez de datos.
   - DOI: https://doi.org/10.18653/v1/2022.findings-emnlp.198
   - https://aclanthology.org/2022.findings-emnlp.198/

4. **Peluso et al. (2024). _Deep learning uncertainty quantification for clinical text classification_. Journal of Biomedical Informatics, 149, 104576.**
   - Tema: cuantificación de incertidumbre en clasificación de textos clínicos, predicciones poco fiables y revisión humana.
   - DOI: https://doi.org/10.1016/j.jbi.2023.104576
   - https://pubmed.ncbi.nlm.nih.gov/38101690/

---

# RQ2 — Temporal modelling and adaptive alarm triggering

## 1. Research question

**Can temporal aggregation and adaptive alarm-triggering strategies improve the trade-off between early detection and prediction reliability in Spanish social media mental health risk detection?**

Queremos estudiar cómo cambia la calidad de la detección cuando el modelo recibe progresivamente nuevos mensajes de un usuario.

El baseline utiliza una media acumulada de las probabilidades de los mensajes disponibles y activa una alarma cuando esta supera un umbral fijo.

La investigación debe estudiar si otras funciones de agregación temporal o modelos de decisión sencillos pueden activar alarmas de manera más fiable y temprana.

## 2. Motivación

La detección temprana es un problema secuencial.

No disponemos de todo el historial del usuario desde el principio. Los mensajes llegan progresivamente y el sistema debe decidir cuándo dispone de suficiente evidencia.

Por ejemplo, imaginemos que un usuario publica estos mensajes:

| Mensaje | Probabilidad de riesgo |
|---|---:|
| 1 | 0.20 |
| 2 | 0.72 |
| 3 | 0.35 |
| 4 | 0.81 |
| 5 | 0.87 |

Si utilizamos un umbral fijo de 0.70 sobre cada mensaje, podríamos activar una alarma por un único mensaje aislado.

Sin embargo, ese mensaje podría ser ambiguo o representar una situación puntual.

Otra posibilidad es acumular las predicciones y considerar la evolución del riesgo.

La cuestión es encontrar un equilibrio entre:

- Detectar el riesgo pronto.
- Evitar falsas alarmas.
- No esperar tantos mensajes que la detección deje de ser temprana.

## 3. Preparación de los datos temporales

Para cada usuario:

1. Ordenar sus mensajes cronológicamente.
2. Obtener la probabilidad de riesgo de cada mensaje mediante el clasificador.
3. Construir secuencias de prefijos del historial.
4. Calcular las características temporales disponibles en cada prefijo.
5. Evaluar cómo cambia la puntuación del usuario al incorporar nuevos mensajes.

Ejemplo:

\[
P_u=[p_1,p_2,\ldots,p_T]
\]

Donde \(p_t\) es la probabilidad de riesgo del mensaje recibido en el instante \(t\).

Para cada prefijo:

\[
P_u^{(t)}=[p_1,p_2,\ldots,p_t]
\]

La decisión debe utilizar exclusivamente información disponible hasta el instante \(t\). Nunca se pueden utilizar mensajes futuros para tomar una decisión anterior.

## 4. Estrategias temporales que se deben comparar

### 4.1. Baseline: cumulative mean pooling

Calcular la media acumulada de las probabilidades:

\[
S_t=\frac{1}{t}\sum_{i=1}^{t}p_i
\]

Activar la alarma cuando:

\[
S_t\geq \tau
\]

Donde \(\tau\) es un umbral fijo seleccionado mediante validación.

Esta estrategia es sencilla y constituye el punto de comparación principal.

Ventaja: acumula evidencia de varios mensajes y reduce la influencia de predicciones aisladas.

Limitación: todos los mensajes tienen el mismo peso, independientemente de cuándo se publicaron.

### 4.2. Moving window / recent mean

Calcular la media de las últimas \(k\) predicciones:

\[
S_t^{window}=
\frac{1}{k}
\sum_{i=t-k+1}^{t}p_i
\]

Utilizar únicamente los últimos mensajes disponibles.

Ejemplo: con \(k=3\), la puntuación se calcula a partir de los tres mensajes más recientes.

Esto permite que el sistema dé más importancia a la evolución reciente del usuario.

Probar varios tamaños de ventana, por ejemplo:

\[
k\in\{3,5,10\}
\]

Para los primeros prefijos, cuando todavía no existen \(k\) mensajes, utilizar todos los mensajes disponibles.

### 4.3. Exponential Weighted Moving Average (EWMA)

Utilizar una media móvil exponencial:

\[
S_t=\lambda p_t+(1-\lambda)S_{t-1}
\]

Donde:

- \(p_t\): probabilidad del mensaje actual.
- \(S_{t-1}\): puntuación acumulada anterior.
- \(\lambda\): factor de actualización entre 0 y 1.

Interpretación:

- Un valor alto de \(\lambda\) da más importancia a los mensajes recientes.
- Un valor bajo de \(\lambda\) conserva más información histórica.

Probar algunos valores de \(\lambda\), seleccionados en validación.

Comparar EWMA con la media acumulada para estudiar si dar más importancia a la información reciente mejora la detección.

### 4.4. Consecutive threshold / sustained confidence

Estudiar una regla que exija varias predicciones consecutivas por encima de un umbral.

Por ejemplo:

\[
p_t\geq \tau
\]

durante \(r\) mensajes consecutivos antes de activar la alarma.

Probar:

\[
r\in\{2,3,4\}
\]

Esta estrategia evita activar alarmas por predicciones positivas aisladas.

Comparar sus efectos sobre:

- False positives.
- False negatives.
- Número de mensajes necesarios para activar la alarma.
- Latencia de detección.

### 4.5. Temporal trend features

Calcular características sencillas que describan la evolución de las probabilidades.

Posibles características por prefijo:

- Media acumulada de probabilidades.
- Máximo de probabilidad observado.
- Media de las últimas \(k\) probabilidades.
- Desviación estándar de las probabilidades.
- Pendiente de una regresión lineal sobre las últimas probabilidades.
- Diferencia entre la media reciente y la media histórica.
- Número o proporción de predicciones que superan un umbral.
- Número de mensajes observados.

La pendiente puede ayudar a distinguir entre un riesgo estable y un riesgo que está aumentando progresivamente.

Por ejemplo:

- Usuario A: 0.75, 0.72, 0.78, 0.74.
- Usuario B: 0.20, 0.35, 0.55, 0.80.

Aunque ambos puedan alcanzar probabilidades similares, la evolución temporal es diferente.

## 5. Modelos sencillos para aprender la función de alarma

La segunda parte de esta RQ debe investigar si podemos aprender una función de decisión a partir de las características temporales.

En lugar de utilizar una regla fija, entrenar un modelo sencillo que combine la información acumulada.

### 5.1. Logistic Regression como alarma adaptativa

Construir un vector de características temporales para cada prefijo:

\[
z_t=[
mean_t,\ max_t,\ recentMean_t,\ slope_t,\ std_t,\ count_t
]
\]

Entrenar una regresión logística:

\[
P(y_u=1\mid z_t)=\sigma(w^Tz_t+b)
\]

Donde \(y_u\) representa la etiqueta original del usuario.

La regresión logística aprenderá qué combinaciones de características temporales indican mayor riesgo.

Después, aplicar un umbral sobre la probabilidad producida por el modelo para decidir cuándo activar la alarma.

Esta estrategia permite comparar una regla fija con una función de decisión aprendida.

### 5.2. Linear Regression / Ridge Regression

Como alternativa, estudiar un regresor lineal o Ridge que genere una puntuación continua de riesgo a partir de las características temporales.

La puntuación se puede utilizar como señal para una regla de alarma.

Sin embargo, para un objetivo binario de clasificación, la regresión logística es generalmente una opción más natural, ya que produce probabilidades entre 0 y 1.

La regresión lineal puede servir como experimento complementario, pero no debe asumirse que sus salidas son probabilidades válidas.

### 5.3. Comparación de estrategias

Comparar al menos:

1. Mean pooling + fixed threshold.
2. Moving window + fixed threshold.
3. EWMA + fixed threshold.
4. Consecutive threshold.
5. Logistic Regression sobre características temporales.
6. Opcionalmente, Ridge Regression sobre características temporales.

Mantener el clasificador que genera las probabilidades de los mensajes constante durante las comparaciones principales.

Esto permite estudiar el efecto de la estrategia temporal sin confundirlo con cambios en el encoder o en el clasificador de mensajes.

## 6. Entrenamiento de los modelos temporales

La etiqueta original de usuario puede utilizarse para supervisar la predicción de los prefijos, siempre que el objetivo sea determinar si el usuario pertenece a la clase positiva.

Cada prefijo de un usuario hereda la etiqueta de ese usuario durante el entrenamiento de la política temporal.

Hay que tener cuidado con la dependencia entre prefijos: los prefijos de un mismo usuario comparten muchos mensajes y no son observaciones independientes.

Recomendaciones:

- Dividir siempre train/dev/test por usuario, nunca por mensaje o prefijo.
- Generar los prefijos después de realizar la separación por usuario.
- No utilizar prefijos de test durante el entrenamiento de la función de alarma.
- Considerar pesos por usuario o muestreo de prefijos para evitar que usuarios con historiales muy largos dominen el entrenamiento.
- Seleccionar hiperparámetros y umbrales exclusivamente en validación.
- No utilizar mensajes posteriores al momento de decisión.

Una comparación importante es utilizar exactamente los mismos usuarios y secuencias en todas las estrategias.

## 7. Definición de la alarma y evaluación temporal

La salida final no debe ser solamente una clasificación, sino también el momento en que se produce la alarma.

Para cada usuario registrar:

- Si se activa una alarma.
- Número de mensajes observados hasta la alarma.
- Posición relativa de la alarma en el historial.
- Probabilidad o puntuación en el momento de la alarma.
- Si la alarma es correcta según la etiqueta original del usuario.

Definir explícitamente qué ocurre si un usuario nunca supera el umbral:

- No se activa ninguna alarma.
- Se mantiene como caso no detectado.
- Se registra la predicción final al terminar el historial, si la evaluación contempla esa decisión.

No tratar una predicción final como si fuera una alarma temprana.

## 8. Métricas de evaluación

Utilizar métricas de clasificación:

- Macro-F1.
- Precision.
- Recall.
- FPR.
- FNR.

Y métricas temporales:

- ERDE5.
- ERDE30.
- Número medio o mediano de mensajes hasta la alarma.
- Latencia de detección.
- Porcentaje de usuarios detectados.
- Porcentaje de usuarios sin alarma.

MentalRiskES 2023 contempla métricas de clasificación, regresión y latencia. Consultar su documentación oficial:

- Tasks: https://sites.google.com/view/mentalriskes/tasks
- Evaluation: https://sites.google.com/view/mentalriskes/evaluation

**Importante:** ERDE penaliza el retraso en la detección y no equivale simplemente al número de mensajes hasta la alarma. Utilizar la definición oficial de la métrica y sus parámetros.

La métrica primaria debe elegirse según la tarea concreta del dataset y el protocolo de evaluación. No optimizar únicamente Macro-F1 si el objetivo central es la detección temprana.

## 9. Hipótesis experimentales

- **H1:** las estrategias de agregación temporal pueden mejorar la estabilidad de las predicciones frente a la media acumulada simple.
- **H2:** EWMA o moving window pueden responder mejor a cambios recientes en las señales de riesgo.
- **H3:** exigir varias predicciones positivas consecutivas puede reducir falsas alarmas causadas por mensajes aislados.
- **H4:** una regresión logística sobre características temporales puede aprender una función de alarma más adecuada que un umbral fijo.
- **H5:** existe un compromiso entre rapidez de detección y fiabilidad: activar alarmas demasiado pronto puede incrementar los falsos positivos, mientras que esperar demasiados mensajes puede aumentar la latencia.

## 10. Referencias bibliográficas para RQ2

Consultar los siguientes trabajos:

1. **Schäfer & Leser (2020). _TEASER: Early and Accurate Time Series Classification_. Data Mining and Knowledge Discovery, 34, 1336–1362.**
   - Tema: clasificación temprana de secuencias, decisión sobre cuándo existe suficiente información y compromiso entre rapidez y precisión.
   - https://link.springer.com/article/10.1007/s10618-020-00690-z

2. **Parapar et al. (2020). _A Big Data Platform for Real Time Analysis of Signs of Depression in Social Media_. International Journal of Environmental Research and Public Health.**
   - Tema: detección temprana de depresión en redes sociales, procesamiento secuencial de mensajes y métrica ERDE.
   - https://pmc.ncbi.nlm.nih.gov/articles/PMC7370096/

3. **Parapar et al. (2024). _Overview of eRisk 2024: Early Risk Prediction on the Internet_.**
   - Tema: evaluación de sistemas de detección temprana y metodologías de las tareas eRisk.
   - PDF: https://clef2024.imag.fr/publications_lncs/637930_1_En_Online_Combine_p2.pdf

4. **MentalRiskES 2023 — Official Evaluation.**
   - Tema: métricas de clasificación, regresión y latencia utilizadas en la competición.
   - https://sites.google.com/view/mentalriskes/evaluation

5. **MentalRiskES 2023 — Official Tasks.**
   - Tema: definición de las tareas de detección binaria, regresión y clasificación multicategoría.
   - https://sites.google.com/view/mentalriskes/tasks

---

# 3. Protocolo experimental común para las dos RQs

## 3.1. Separación de datos

La separación debe hacerse por usuario:

- Train: usuarios utilizados para entrenar los modelos.
- Dev/validation: usuarios utilizados para seleccionar hiperparámetros, umbrales y estrategias.
- Test: usuarios reservados para la evaluación final.

Nunca repartir mensajes del mismo usuario entre train y test.

El procedimiento de relabeling k-NN debe respetar esta separación. No se deben utilizar mensajes ni información de usuarios de test para construir el índice de entrenamiento o modificar las etiquetas de entrenamiento.

Si se necesita un índice de vecinos para relabeling en validación, debe definirse un procedimiento que no utilice información de test y que sea coherente con el escenario de evaluación.

## 3.2. Separación de las contribuciones

Mantener separados estos componentes:

1. Encoder: convierte mensajes en vectores.
2. Relabeling: genera pseudoetiquetas a nivel de mensaje.
3. Message classifier: predice la probabilidad de riesgo de cada mensaje.
4. Temporal aggregation: combina las predicciones disponibles.
5. Decision policy: decide si se activa una alarma o se produce una abstención.

Cambiar un componente cada vez en las comparaciones principales.

## 3.3. Reproducibilidad

- Utilizar las mismas particiones de usuarios para todos los experimentos.
- Fijar semillas aleatorias.
- Reportar media y desviación estándar cuando se realicen varias ejecuciones.
- Guardar los hiperparámetros seleccionados en validación.
- No modificar umbrales después de consultar resultados de test.
- Documentar el número de mensajes y usuarios antes y después del relabeling.
- Documentar la proporción de mensajes que cambia de etiqueta.
- Mantener un registro de los mensajes excluidos por falta de embeddings o por filtros de longitud.

## 3.4. Precauciones metodológicas

1. Las etiquetas originales son etiquetas de usuario. Las etiquetas de mensaje generadas por k-NN son pseudoetiquetas y pueden contener ruido.
2. Un modelo puede estar muy seguro de una predicción incorrecta. La confianza no equivale a exactitud.
3. La calibración no corrige automáticamente el ruido de las etiquetas.
4. No utilizar información futura para generar una alarma en un prefijo anterior.
5. No considerar los prefijos de un mismo usuario como muestras independientes.
6. Las métricas a nivel de mensaje y a nivel de usuario deben distinguirse claramente.
7. La detección de riesgo en este proyecto es una tarea experimental de clasificación, no un diagnóstico clínico.

---

# 4. Resultados que esperamos obtener

El proyecto debería terminar con dos contribuciones experimentales diferenciadas:

### Contribución de RQ1

Una comparación entre estrategias de decisión basadas en confianza, incertidumbre y abstención.

La pregunta final es si podemos hacer que las predicciones aceptadas sean más fiables y qué cantidad de cobertura perdemos al hacerlo.

### Contribución de RQ2

Una comparación entre reglas de agregación temporal y funciones de alarma aprendidas.

La pregunta final es si podemos detectar usuarios positivos antes, manteniendo una tasa de error razonable.

### Comparación final

El resultado debería permitir responder:

- ¿Qué estrategia ofrece mejor calidad de clasificación?
- ¿Qué estrategia detecta antes a los usuarios positivos?
- ¿Qué estrategia genera menos falsas alarmas?
- ¿Qué estrategia necesita menos mensajes para tomar una decisión?
- ¿Qué mejora es consistente respecto al baseline?
- ¿Qué coste tiene cada mejora en términos de cobertura, complejidad y latencia?

**Objetivo final:** construir un sistema de detección temprana que no solo produzca buenas predicciones, sino que también sepa cuándo confiar en ellas y cuándo dispone de evidencia temporal suficiente para activar una alarma.