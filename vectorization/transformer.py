"""Vectorización densa con Sentence Transformers (solo CPU).

Los embeddings de los mensajes se guardan en una caché en disco por modelo
(``vectorization.cache``): las vectorizaciones por usuario ``mean``/``max``
reutilizan los embeddings de mensaje ya calculados.
"""

import argparse
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .cache import VectorCache
from .combine import l2_normalize
from .units import Units, aggregate_rows, make_blocks

MODEL_ALIASES: dict[str, str] = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5": "intfloat/multilingual-e5-small",
    "beto-sts": "hiiamsid/sentence_similarity_spanish_es",
    "robertuito": "pysentimiento/robertuito-base-uncased",
    "robertuito-base-uncased": "pysentimiento/robertuito-base-uncased",
    "beto-raw": "dccuchile/bert-base-spanish-wwm-uncased",
    "roberta-base-bne": "PlanTL-GOB-ES/roberta-base-bne",
    "robertalex": "PlanTL-GOB-ES/RoBERTalex",
    "modernbert": ("mrm8488/modernbert-embed-base-ft-sts-spanish-"
                   "matryoshka-768-64"),
}

# Palabras por bloque para que el bloque no se trunque (≈0,7 palabras por
# token). MiniLM corta en 128 tokens; los demás admiten 512.
BLOCK_WORDS: dict[str, int] = {
    "minilm": 90, "e5": 350, "beto-sts": 350,
    "robertuito": 350, "robertuito-base-uncased": 350, "beto-raw": 350,
    "roberta-base-bne": 350, "robertalex": 350, "modernbert": 350,
}
DEFAULT_BLOCK_WORDS: int = 200

POOLINGS: list[str] = ["concat", "mean", "max", "blocks"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Añade al subcomando ``transformer`` sus opciones propias."""
    parser.add_argument("--model", default="minilm",
                        help="alias (" + ", ".join(MODEL_ALIASES) + ") o "
                        "nombre de Hugging Face")
    parser.add_argument("--user-pooling", choices=POOLINGS, default="mean",
                        help="cómo se obtiene el vector de usuario: texto "
                        "concatenado, media/máximo de mensajes o media de "
                        "bloques de palabras (def. mean)")
    parser.add_argument("--block-words", type=int, default=None,
                        help="palabras por bloque (def. según el modelo)")
    parser.add_argument("--no-normalize", action="store_true",
                        help="no normalizar los vectores en L2")
    parser.add_argument("--truncate-dim", type=int, default=None,
                        help="dimensiones a conservar (modelos Matryoshka)")
    parser.add_argument("--pca", type=int, default=None, metavar="N",
                        help="reducir a N dimensiones con PCA")


def resolve_model(name: str) -> tuple[str, str]:
    """Devuelve ``(nombre_hf, prefijo)`` de un alias o nombre de modelo.

    Los modelos E5 necesitan el prefijo ``"query: "`` delante de cada texto.
    """
    hf_name = MODEL_ALIASES.get(name, name)
    prefix = "query: " if "multilingual-e5" in hf_name else ""
    return hf_name, prefix


def block_words_for(args: argparse.Namespace) -> int:
    """Devuelve las palabras por bloque: la opción o el valor del modelo."""
    if args.block_words is not None:
        return args.block_words
    return BLOCK_WORDS.get(args.model, DEFAULT_BLOCK_WORDS)


def resolve_device(device: str) -> str:
    """Convierte ``auto`` en ``cuda`` si hay GPU y en ``cpu`` si no.

    ``cpu`` y ``cuda`` se devuelven tal cual.
    """
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_encoder(
    hf_name: str,
    batch_size: int,
    truncate_dim: int | None,
    device: str = "cpu",
) -> Callable[[list[str]], np.ndarray]:
    """Crea una función ``textos -> matriz`` que carga el modelo al usarse.

    El modelo se carga en la primera llamada, así que si todos los textos
    están en la caché ni siquiera se importa ``sentence_transformers`` ni
    se mira la GPU. Los vectores no dependen del dispositivo (salvo
    diferencias numéricas mínimas), por eso la caché es común.
    """
    state: dict = {}

    def encode(texts: list[str]) -> np.ndarray:
        """Codifica una lista de textos con el modelo (en CPU)."""
        if hf_name == MODEL_ALIASES["roberta-base-bne"]:
            return encode_roberta_bne(texts, batch_size, device, state)
        if "model" not in state:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise ImportError(
                    "Falta sentence-transformers: pip install -r "
                    "vectorization/requirements.txt") from error
            options = {"device": resolve_device(device)}
            if truncate_dim is not None:
                options["truncate_dim"] = truncate_dim
            state["model"] = SentenceTransformer(hf_name, **options)
            if hf_name == MODEL_ALIASES["robertuito-base-uncased"]:
                transformer = state["model"][0]
                transformer.max_seq_length = 128
                transformer.tokenizer.model_max_length = 128
        return state["model"].encode(
            texts, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=False)

    return encode


def encode_roberta_bne(
    texts: list[str],
    batch_size: int,
    device: str,
    state: dict,
) -> np.ndarray:
    """Codifica RoBERTa-BNE, cuyo config antiguo no declara ``model_type``."""
    if "model" not in state:
        from transformers import RobertaModel, RobertaTokenizer

        target_device = resolve_device(device)
        state["tokenizer"] = RobertaTokenizer.from_pretrained(
            MODEL_ALIASES["roberta-base-bne"])
        state["model"] = RobertaModel.from_pretrained(
            MODEL_ALIASES["roberta-base-bne"]).to(target_device)
        state["model"].eval()

    import torch

    tokenizer = state["tokenizer"]
    model = state["model"]
    target_device = next(model.parameters()).device
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start:start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            batch = {key: value.to(target_device)
                     for key, value in batch.items()}
            output = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def cache_path(
    cache_dir: Path,
    hf_name: str,
    truncate_dim: int | None,
) -> Path:
    """Devuelve el archivo de caché de un modelo (y dimensión truncada)."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", hf_name)
    if truncate_dim is not None:
        slug += f"__dim{truncate_dim}"
    return cache_dir / "embeddings" / f"{slug}.npz"


def embed_texts(
    texts: list[str],
    prefix: str,
    cache: VectorCache,
    encode: Callable[[list[str]], np.ndarray],
) -> np.ndarray:
    """Devuelve los embeddings (sin normalizar) de una lista de textos.

    Se añade el prefijo del modelo antes de calcular o buscar en la caché.
    """
    prefixed = [prefix + text for text in texts]
    return cache.get_matrix(prefixed, encode)


def vectorize(
    df: pd.DataFrame,
    units: Units,
    fit_mask: np.ndarray,
    args: argparse.Namespace,
    cache_dir: Path,
) -> tuple[np.ndarray, object, dict]:
    """Calcula el embedding de cada unidad según la unidad y el pooling.

    - ``message``: un embedding por mensaje.
    - ``user`` + ``concat``: embedding del texto concatenado (el modelo lo
      trunca a su longitud máxima).
    - ``user`` + ``mean``/``max``: agrega los embeddings de sus mensajes.
    - ``user`` + ``blocks``: media de los embeddings de bloques de palabras.

    Con normalización (por defecto) los vectores se normalizan en L2 antes
    y después de agregar. Si se pide ``--pca``, se ajusta con ``fit_mask``.
    Devuelve ``(matriz, pca_o_None, info)``.
    """
    hf_name, prefix = resolve_model(args.model)
    cache = VectorCache(cache_path(cache_dir, hf_name, args.truncate_dim),
                        enabled=not args.no_cache)
    encode = make_encoder(hf_name, args.batch_size, args.truncate_dim,
                          args.device)
    normalize = not args.no_normalize

    def embed(texts: list[str]) -> np.ndarray:
        """Embeddings de ``texts``, normalizados si corresponde."""
        matrix = embed_texts(texts, prefix, cache, encode)
        return l2_normalize(matrix) if normalize else matrix

    if args.unit == "message":
        matrix = embed(units.texts)
    elif args.user_pooling == "concat":
        matrix = embed(units.texts)
    elif args.user_pooling in ("mean", "max"):
        matrix = aggregate_rows(embed(df["message"].tolist()), units.owner,
                                args.user_pooling)
    else:
        blocks, block_owner = make_blocks(units.texts, block_words_for(args))
        matrix = aggregate_rows(embed(blocks), block_owner, "mean")
    if normalize:
        matrix = l2_normalize(matrix)

    pca = None
    if args.pca is not None:
        matrix, pca = reduce_pca(matrix, fit_mask, args.pca, args.seed,
                                 normalize)
    info = {
        "model": hf_name,
        "device": args.device,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }
    if args.unit == "user" and args.user_pooling == "blocks":
        info["block_words"] = block_words_for(args)
    return matrix.astype(np.float32), pca, info


def reduce_pca(
    matrix: np.ndarray,
    fit_mask: np.ndarray,
    n_components: int,
    seed: int,
    normalize: bool,
) -> tuple[np.ndarray, PCA]:
    """Reduce la dimensión con PCA ajustada solo con ``fit_mask``.

    Si ``normalize`` es verdadero, renormaliza en L2 tras la proyección para
    que el coseno siga siendo la distancia natural.
    """
    n_components = min(n_components, matrix.shape[1],
                       int(fit_mask.sum()))
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(matrix[fit_mask])
    reduced = pca.transform(matrix)
    return (l2_normalize(reduced) if normalize else reduced), pca
