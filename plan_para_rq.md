## Plan: Pipeline RQ con relabeling y baseline

Objetivo: dejar un flujo reproducible para tus RQ usando los scripts de supervised_splits/rama_b_relabeling, ejecutable desde notebooks (local o Colab), leyendo particiones desde particiones, y guardando artefactos por RQ/run_id en rq_larra.

**Steps**
1. Fase 1: Alinear entradas y rutas (bloqueante)
2. Validar contrato de columnas entre los scripts y los CSV de particiones.
3. Ajustar el pipeline para que use directamente particiones como fuente canónica (sin duplicar a otras carpetas).
4. Definir run_id único por corrida para evitar sobrescrituras.
5. Fase 2: Notebook de relabeling (depende de fase 1)
6. En notebook, ejecutar auditoría de split con supervised_splits/rama_b_relabeling/audit_splits.py.
7. Generar cache de embeddings de relabeling con supervised_splits/rama_b_relabeling/make_tfidf_cache.py o embedding equivalente.
8. Ejecutar relabeling con supervised_splits/rama_b_relabeling/run_relabeling.py, que usa supervised_splits/rama_b_relabeling/relabeling.py.
9. Persistir weak labels, historial de rondas y manifiesto en subcarpeta por run_id dentro de rq_larra/csv.
10. Fase 3: Notebook baseline MLP (parcialmente en paralelo con fase 2 si embeddings ya listos)
11. Preparar embeddings train/dev para baseline con supervised_splits/rama_b_relabeling/make_beto_sts_embeddings.py.
12. Ejecutar baseline con supervised_splits/rama_b_relabeling/baseline_mlp.py sobre condiciones comparables (initial y variantes kNN).
13. Exportar métricas dev por semilla/resumen y predicciones por sujeto a rq_larra/csv/rq_baseline/run_id.
14. Fase 4: Persistencia y trazabilidad (depende de fases 2-3)
15. Estructurar resultados: rq_larra/csv/rq1/run_id, rq_larra/csv/rq2/run_id, rq_larra/figuras/rq1/run_id, rq_larra/figuras/rq2/run_id.
16. Crear y mantener rq_larra/MEMORIA_COPILOT.md con decisiones, rutas válidas, parámetros y checklist anti-leakage.
17. Crear índice global de corridas (rq_larra/csv/index_runs.csv) con run_id, rq, seed, embeddings, params de relabeling, notebook, fecha y hash/commit.
18. Fase 5: Base analítica para RQ1/RQ2 (depende de fases 2-4)
19. RQ1 en rq_larra/ipynb/rq1.ipynb: threshold fijo, confidence/entropy-abstention, calibración y cobertura-riesgo selectivo.
20. RQ2 en rq_larra/ipynb/rq2.ipynb: mean pooling, moving window, EWMA, consecutive threshold y alarma adaptativa (logistic regression temporal).
21. Mantener selección de hiperparámetros únicamente en dev; test congelado para evaluación final.
22. Fase 6: Verificación final
23. Ejecutar tests clave de rama_b_relabeling.
24. Correr una corrida smoke end-to-end (1 seed) y verificar que genera todos los artefactos esperados.
25. Confirmar en manifiestos que no hubo uso de test en selección y que toda corrida quedó trazada.

**Relevant files**
- rq_larra/rq.md
- supervised_splits/rama_b_relabeling/README.md
- supervised_splits/rama_b_relabeling/run_relabeling.py
- supervised_splits/rama_b_relabeling/relabeling.py
- supervised_splits/rama_b_relabeling/make_tfidf_cache.py
- supervised_splits/rama_b_relabeling/make_beto_sts_embeddings.py
- supervised_splits/rama_b_relabeling/baseline_mlp.py
- particiones/manifest.json
- rq_larra/ipynb/rq1.ipynb
- rq_larra/ipynb/rq2.ipynb

**Decisiones Capturadas**
1. Flujo principal en notebooks para facilitar Colab.
2. Particiones fuente desde particiones.
3. Memoria persistente en rq_larra/MEMORIA_COPILOT.md.
4. Organización de resultados por RQ/run_id.