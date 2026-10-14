# Guía: generar las vectorizaciones con modelos en Google Colab (GPU)

## De qué va esto

El proyecto aplica DBSCAN a mensajes de Telegram (corpus MentalRiskES). Antes
hay que convertir los textos en vectores ("vectorizar"). Las vectorizaciones
con modelos de lenguaje (embeddings y pysentimiento) tardan mucho en CPU
(~40 min) y en una GPU gratuita de Colab tardan unos 10–15 min. Tu tarea es
ejecutarlas en Colab y devolver la carpeta de resultados. No hace falta
entender el código.

Necesitas una cuenta de Google. Los modelos se descargan solos desde Colab; no
tienes que instalar nada en tu ordenador.

## Qué te tienen que pasar

Una carpeta de Google Drive compartida contigo llamada `mineria/` con:

- `vectorization/` (el código),
- `run_vectorizations.sh` (el script que lanza todo),
- `DatosMentalRiskchallenge-20261005.zip` (el corpus, 6 MB).

Cópiala a tu propio Drive (clic derecho → *Hacer una copia*, o arrastrándola)
para que quede en `Mi unidad/mineria/`. Los pasos siguientes asumen esa ruta.

## Pasos

Crea un cuaderno nuevo en <https://colab.research.google.com> y ejecuta las
celdas en orden (una celda = un bloque de código de abajo).

### 0. Activar la GPU

Menú *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU →
Guardar*.

### 1. Comprobar la GPU y conectar Drive

```python
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
```

La primera línea debe mostrar una GPU (por ejemplo `Tesla T4`). Si dice que no
hay, vuelve al paso 0 o inténtalo más tarde: Colab gratuito no siempre tiene
GPU libre. Acepta los permisos de Drive cuando los pida.

### 2. Copiar el código y descomprimir el corpus

```bash
!mkdir -p /content/proj
!cp -r /content/drive/MyDrive/mineria/vectorization /content/drive/MyDrive/mineria/run_vectorizations.sh /content/proj/
!unzip -q /content/drive/MyDrive/mineria/DatosMentalRiskchallenge-20261005.zip -d /content/proj
%cd /content/proj
```

### 3. Instalar las dependencias que faltan

```bash
!pip install -q emoji pysentimiento
```

**No instales** `vectorization/requirements.txt`: cambiaría el torch con GPU de
Colab por uno de CPU. Comprueba que sigue habiendo GPU:

```python
import torch
print(torch.cuda.is_available())   # debe imprimir True
```

Si `pip` muestra errores de versiones incompatibles, reinicia
(*Entorno de ejecución → Reiniciar sesión*) y repite desde el paso 2.

### 4. Dejar solo las vectorizaciones con modelos

Las de TF-IDF no usan GPU y se hacen en local, así que se desactivan:

```bash
!sed -i '/^run_vectorization tfidf/s/^/# /' run_vectorizations.sh
```

### 5. Lanzar

```bash
!OUT_DIR=/content/drive/MyDrive/mineria/vectorizations \
 TIMES_FILE=/content/drive/MyDrive/mineria/vectorizations/times_gpu.txt \
 VECTORIZATION_ARGS="--device cuda --cache-dir /content/drive/MyDrive/mineria/cache" \
 bash run_vectorizations.sh
```

Es un solo comando (las `\` unen las líneas). Los resultados se escriben
directamente en tu Drive, así que si Colab se desconecta no se pierde lo ya
hecho.

- Tarda unos 10–15 minutos. Deja la pestaña abierta y el ordenador despierto.
- Cada línea imprime `>>> nombre-de-la-vectorización` cuando empieza. Verás
  barras de progreso mientras codifica.
- **Si se corta la sesión:** repite los pasos 1, 2, 3, 4 y 5. Lo que ya estaba
  calculado sale de la caché de Drive y es casi instantáneo.

### 6. Comprobar que ha ido bien

```bash
!cat /content/drive/MyDrive/mineria/vectorizations/times_gpu.txt
!ls /content/drive/MyDrive/mineria/vectorizations
```

`times_gpu.txt` tiene una línea por vectorización con la fecha, el nombre de la
carpeta, los segundos y `OK` o `FALLO`. Todas deberían ser `OK`.

## Qué devolver

Comparte con quien te pidió esto la carpeta de Drive
`mineria/vectorizations/` (puedes comprimirla en un zip). Contiene una carpeta
por vectorización y el archivo `times_gpu.txt`. No hace falta compartir
`mineria/cache/`.

Avisa si algún tamaño pasa de 95 MB: el script imprime un `AVISO` en ese caso.

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `nvidia-smi` no muestra GPU | Repite el paso 0 o reintenta más tarde. No continúes sin GPU: tardaría horas. |
| Error al instalar `pysentimiento` o incompatibilidades | Reinicia la sesión y repite desde el paso 2. Si sigue fallando, di cuál es el error y, mientras tanto, desactiva las líneas con `--affect` y la de `features`: `!sed -i '/--affect/s/^/# /; /^run_vectorization features/s/^/# /' run_vectorizations.sh` antes del paso 5. |
| Alguna línea sale como `FALLO` | Vuelve a lanzar el paso 5: se saltan los cálculos ya hechos. Si persiste, copia el mensaje de error que aparece justo después de su `>>>`. |
| `Out of memory` en GPU | Añade ` --batch-size 8` dentro de `VECTORIZATION_ARGS` y repite el paso 5. |
| Se acabó la cuota de GPU de Colab | Espera unas horas o pide a quien te lo encargó que lo lance en CPU. |

## Notas

- Los tiempos de `times_gpu.txt` son de GPU, no de CPU; no se mezclan con los
  de `vectorizations/times.txt`.
- No cambies los nombres de las carpetas de `vectorizations/`: indican con qué
  opciones se generó cada una.
