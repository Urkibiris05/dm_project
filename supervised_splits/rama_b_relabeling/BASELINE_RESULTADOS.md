# Resultados iniciales: BETO-STS + MLP + k-NN

Este informe resume una primera comparación de desarrollo para compartir con
el grupo. Es un resultado **preliminar**, no la evaluación final del trabajo.
La implementación y el protocolo técnico están descritos en
[`README.md`](./README.md); el estado de las tareas y próximos experimentos,
en [`PLAN_DE_TRABAJO.md`](./PLAN_DE_TRABAJO.md).

## Protocolo ejecutado

- Corpus: MentalRiskES2023, con los splits compartidos por sujeto. Se usaron
  19.805 mensajes de train y 5.452 de dev; se incluyeron mensajes cortos.
- Encoder congelado: `hiiamsid/sentence_similarity_spanish_es`
  (BETO-STS), revisión fijada en el manifiesto; un vector de 768 dimensiones
  por mensaje.
- Tres etiquetas de entrenamiento: etiqueta inicial heredada del sujeto;
  k-NN de una pasada con `J=10`, `K=5`, coseno, permitiendo otros mensajes del
  mismo sujeto; y el mismo k-NN excluyendo al sujeto consultado. Siempre se
  excluyó el propio mensaje. El k-NN solo cambia `suffer → control`.
- Student: `Linear(768,256) → ReLU → Dropout(0.3) → Linear(256,1)`, BCE sin
  pesos de clase, AdamW, cinco semillas (`11, 23, 37, 51, 79`).
- Detección: media acumulada de `P(suffer)` en orden cronológico. Época y
  umbral se seleccionaron en dev maximizando macro-F1 por sujeto; si no se
  cruza el umbral, se decide `control` al final.
- **Test no se usó.** Las métricas de esta tabla son las que guiaron la
  selección en dev y, por ello, no son una estimación final independiente.

## Detección por sujeto en dev

Macro-F1 y F1 de `suffer` se muestran como media ± desviación entre semillas.
La desviación refleja variación entre inicializaciones del MLP, no
incertidumbre estadística sobre la población de sujetos.

| Trastorno | Etiquetas | Macro-F1 | Δ macro-F1 vs Init | F1 `suffer` |
|---|---|---:|---:|---:|
| Ansiedad | Iniciales (MLP-Init) | 0,8253 ± 0,0035 | — | 0,8616 ± 0,0073 |
| Ansiedad | k-NN, mismo sujeto permitido | 0,8149 ± 0,0145 | −0,0105 | 0,8424 ± 0,0275 |
| Ansiedad | k-NN, excluir sujeto | **0,8295 ± 0,0035** | **+0,0041** | 0,8535 ± 0,0074 |
| Depresión | Iniciales (MLP-Init) | 0,7691 ± 0,0262 | — | 0,7957 ± 0,0311 |
| Depresión | k-NN, mismo sujeto permitido | 0,7813 ± 0,0165 | +0,0122 | 0,8092 ± 0,0192 |
| Depresión | k-NN, excluir sujeto | **0,7857 ± 0,0217** | **+0,0167** | 0,8059 ± 0,0255 |
| TCA | Iniciales (MLP-Init) | **0,9285 ± 0,0122** | — | **0,9207 ± 0,0122** |
| TCA | k-NN, mismo sujeto permitido | 0,9256 ± 0,0105 | −0,0029 | 0,9183 ± 0,0120 |
| TCA | k-NN, excluir sujeto | 0,9260 ± 0,0105 | −0,0025 | 0,9198 ± 0,0108 |

La latencia siguiente es la fracción media de mensajes que se había visto
cuando el sistema detectó correctamente a los sujetos `suffer`; menor suele
significar detección más temprana, pero se interpreta junto con la cobertura
de positivos y las métricas de clasificación.

| Trastorno | Etiquetas | Recall `suffer` / cobertura positiva | Fracción de mensajes hasta detección |
|---|---|---:|---:|
| Ansiedad | Iniciales (MLP-Init) | 0,821 | 0,086 |
| Ansiedad | k-NN, mismo sujeto permitido | 0,768 | 0,059 |
| Ansiedad | k-NN, excluir sujeto | 0,768 | 0,060 |
| Depresión | Iniciales (MLP-Init) | 0,876 | 0,139 |
| Depresión | k-NN, mismo sujeto permitido | 0,900 | 0,119 |
| Depresión | k-NN, excluir sujeto | 0,871 | 0,124 |
| TCA | Iniciales (MLP-Init) | 0,959 | 0,139 |
| TCA | k-NN, mismo sujeto permitido | 0,966 | 0,136 |
| TCA | k-NN, excluir sujeto | 0,993 | 0,131 |

## Actividad de relabeling en train

Los recuentos se refieren a mensajes inicialmente `suffer` cambiados a
`control`; no miden acierto de mensaje.

| Trastorno | Política de vecinos | Mensajes cambiados / iniciales `suffer` | Fracción |
|---|---|---:|---:|
| Ansiedad | Mismo sujeto permitido | 538 / 2.307 | 23,3 % |
| Ansiedad | Excluir sujeto | 596 / 2.307 | 25,8 % |
| Depresión | Mismo sujeto permitido | 1.634 / 3.951 | 41,4 % |
| Depresión | Excluir sujeto | 1.785 / 3.951 | 45,2 % |
| TCA | Mismo sujeto permitido | 1.395 / 2.866 | 48,7 % |
| TCA | Excluir sujeto | 1.475 / 2.866 | 51,5 % |

Excluir otros mensajes del mismo sujeto produjo más cambios en las tres
tareas. Es un efecto de la composición del vecindario en esta configuración,
no evidencia de que los mensajes adicionales relabelados sean más adecuados.

## Lectura provisional

1. **No hay una mejora uniforme.** Excluir el sujeto aumenta modestamente
   macro-F1 en Ansiedad (+0,004) y Depresión (+0,017), pero lo reduce
   ligeramente en TCA (−0,0025). Permitir los mensajes del mismo sujeto
   empeora Ansiedad y TCA y mejora Depresión.
2. **La latencia parece favorable, pero hay intercambios.** En Ansiedad, la
   menor fracción de mensajes hasta detección acompaña una menor cobertura de
   sujetos positivos; en TCA, la cobertura aumenta, pero macro-F1 no mejora.
   No se debe presentar la latencia aislada como resultado suficiente.
3. **La señal de Depresión es prometedora pero preliminar.** La mejora
   observada es descriptiva y se midió en dev, con 67 sujetos; necesita
   confirmación con protocolo congelado y test reservado.
4. **No se ha medido calidad de etiquetas de mensaje.** Las etiquetas
   iniciales y relabeladas son supervisión débil. Sin anotación independiente
   no se pueden afirmar precisión, recall o F1 reales a nivel de mensaje.

## Artefactos reproducibles

- [Resumen agregado completo](./reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/dev_summary.csv)
- [Resumen compacto](./reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/dev_summary_compact.csv)
- [Métricas por semilla](./reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/dev_seed_metrics.csv)
- [Cobertura de relabeling](./reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/relabeling_coverage.csv)
- [Manifiesto de la corrida](./reports/baseline/baseline-beto-sts-dev-j10-k5-5seeds/run_manifest.json)
- [Manifiesto de embeddings BETO-STS](./reports/embeddings/beto_sts_full_baseline/embedding_manifest.json), con revisión exacta y tamaños de train/dev.
- Pseudoetiquetas e historiales separados por política en las carpetas
  `Ansiedad/`, `Depresion/` y `TCA/` de la misma corrida.
- Predicciones por sujeto, checkpoints y escaladores por semilla bajo
  `dev/` dentro de cada condición.

La siguiente fase recomendada es confirmar el protocolo con el grupo,
calcular incertidumbre pareada agrupada por sujeto y comparar los encoders
alternativos en dev. Solo después de congelar las decisiones se deben generar
embeddings de test y evaluar test una vez.
