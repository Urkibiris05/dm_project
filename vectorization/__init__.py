"""Pipeline de vectorización del corpus MentalRiskES.

Se usa desde la línea de comandos (desde la raíz del repositorio)::

    python -m vectorization tfidf --unit user --analyzer char_wb
    python -m vectorization transformer --model minilm --unit message
    python -m vectorization features --unit user --affect 0.5 --time 0.5

Cada ejecución crea una carpeta ``vectorizations/<nombre>/`` con la matriz
``X.npy``, los metadatos de cada fila y la configuración usada.
"""
