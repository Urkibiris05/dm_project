"""Nombre de la carpeta de salida a partir de los argumentos.

Formato: ``unidad__trastornos__método__bloques__preprocesado``. Solo se
añade lo que se aparta de los valores por defecto, por ejemplo::

    message__all__tfidf-char_wb-1-5-svd128
    user__Depresion__minilm-mean__affect0.2-time0.1
    message__all__e5__fit-train
"""

import argparse

from .corpus import DISORDERS, SPLITS
from .tfidf import resolve_ngram_range

TASK_SHORT: dict[str, str] = {
    "sentiment": "sent", "emotion": "emo", "hate_speech": "hate",
}


def method_name(args: argparse.Namespace) -> str:
    """Devuelve el segmento que describe el método de vectorización."""
    if args.command == "features":
        return "features"
    if args.command == "tfidf":
        low, high = resolve_ngram_range(args)
        parts = ["tfidf", args.analyzer, str(low), str(high),
                 f"svd{args.svd}"]
        if args.sublinear_tf:
            parts.append("sub")
        if args.min_df != 2:
            parts.append(f"mindf{args.min_df}")
        if args.max_df != 0.95:
            parts.append(f"maxdf{args.max_df:g}")
        if args.max_features is not None:
            parts.append(f"max{args.max_features}")
        return "-".join(parts)
    parts = [args.model.split("/")[-1]]
    if args.unit == "user":
        parts.append(args.user_pooling)
        if args.user_pooling == "blocks" and args.block_words is not None:
            parts.append(f"w{args.block_words}")
    if args.truncate_dim is not None:
        parts.append(f"dim{args.truncate_dim}")
    if args.pca is not None:
        parts.append(f"pca{args.pca}")
    if args.no_normalize:
        parts.append("nonorm")
    return "-".join(parts)


def blocks_name(args: argparse.Namespace) -> str | None:
    """Devuelve el segmento de bloques opcionales (afecto y hora), si hay."""
    parts = []
    if args.affect > 0:
        part = f"affect{args.affect:g}"
        if sorted(args.affect_tasks) != ["emotion", "sentiment"]:
            part += "-" + "+".join(TASK_SHORT[task]
                                   for task in sorted(args.affect_tasks))
        parts.append(part)
    if args.time > 0:
        parts.append(f"time{args.time:g}")
    return "-".join(parts) or None


def build_name(args: argparse.Namespace) -> str:
    """Construye el nombre determinista de la carpeta de una vectorización."""
    disorders = ("all" if sorted(args.disorders) == sorted(DISORDERS)
                 else "+".join(args.disorders))
    segments = [args.unit, disorders, method_name(args), blocks_name(args)]
    extras = []
    if sorted(args.splits) != sorted(SPLITS):
        extras.append("splits-" + "+".join(args.splits))
    if args.fit_on == "train":
        extras.append("fit-train")
    if args.min_tokens != 4:
        extras.append(f"mintok{args.min_tokens}")
    if args.count_emoji_tokens:
        extras.append("emojitok")
    if args.keep_duplicates:
        extras.append("dups")
    if args.emojis == "remove":
        extras.append("noemoji")
    segments.append("-".join(extras) or None)
    return "__".join(segment for segment in segments if segment)
