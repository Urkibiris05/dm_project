"""Evaluación y selección de hiperparámetros para DBSCAN (común a E1, E2...).

Módulos: ``kdist`` (grafo k-dist y codo), ``dbcv`` (índice interno de
densidad), ``metrics`` (internas y externas), ``stability`` (bootstrap) y
``selection`` (regla de elección sin etiquetas).
"""

NOISE: int = -1
