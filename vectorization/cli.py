"""Línea de comandos: ``python -m vectorization <método> [opciones]``.

Métodos: ``tfidf``, ``transformer`` y ``features`` (solo afecto y/o hora).
El flujo de ``run`` es: cargar corpus → preprocesar → construir unidades →
vectorizar → bloques opcionales (afecto, hora) → combinar → guardar.
"""

import argparse
import platform
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import affect, tfidf, time_features, transformer
from .combine import combine_blocks
from .corpus import DISORDERS, SPLITS, TRAIN_SPLITS, load_corpus
from .naming import build_name
from .output import prepare_directory, save_vectorization
from .preprocessing import (build_emoji_index, load_emoji_names,
                            needs_emoji_names, preprocess)
from .timing import PhaseTimer
from .units import aggregate_rows, build_units

PACKAGES: list[str] = [
    "numpy", "pandas", "scikit-learn", "sentence-transformers",
    "pysentimiento", "emoji",
]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Añade a un subcomando las opciones comunes a todos los métodos."""
    group = parser.add_argument_group("datos")
    group.add_argument("--corpus-root", default="MentalRiskEs2023/corpus")
    group.add_argument("--disorders", nargs="+", choices=DISORDERS,
                       default=DISORDERS,
                       help="trastornos a usar (def. todos)")
    group.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS,
                       help="particiones a usar (def. todas)")
    group.add_argument("--unit", choices=["message", "user"],
                       default="message",
                       help="un vector por mensaje o por usuario")
    group.add_argument("--fit-on", choices=["all", "train"], default="all",
                       help="textos con los que se ajusta el vectorizador "
                       "(vocabulario, SVD, PCA); nunca se usan etiquetas")

    group = parser.add_argument_group("preprocesado")
    group.add_argument("--min-tokens", type=int, default=4,
                       help="se descartan los mensajes con menos tokens "
                       "(def. 4: se quitan los de 3 o menos)")
    group.add_argument("--count-emoji-tokens", action="store_true",
                       help="contar también los nombres de emoji como tokens")
    group.add_argument("--keep-duplicates", action="store_true",
                       help="no eliminar mensajes repetidos de un usuario")
    group.add_argument("--emojis", choices=["keep", "remove"],
                       default="keep",
                       help="conservar o quitar los nombres de emoji")

    group = parser.add_argument_group("bloques opcionales")
    affect.add_arguments(group)
    time_features.add_arguments(group)

    group = parser.add_argument_group("salida")
    group.add_argument("--output-dir", default="vectorizations")
    group.add_argument("--cache-dir", default=".cache")
    group.add_argument("--no-cache", action="store_true",
                       help="recalcular embeddings ignorando la caché")
    group.add_argument("--name", default=None,
                       help="nombre de carpeta (def. generado)")
    group.add_argument("--overwrite", action="store_true",
                       help="reemplazar la carpeta si ya existe")
    group.add_argument("--print-name", action="store_true",
                       help="solo imprime el nombre de la carpeta y sale")
    group.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    """Crea el analizador de argumentos con sus tres subcomandos."""
    parser = argparse.ArgumentParser(
        prog="python -m vectorization",
        description="Vectoriza el corpus MentalRiskES.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser(
        "tfidf", help="TF-IDF (palabras o caracteres) + SVD")
    tfidf.add_arguments(sub)
    add_common_arguments(sub)

    sub = subparsers.add_parser(
        "transformer", help="embeddings de Sentence Transformers")
    transformer.add_arguments(sub)
    add_common_arguments(sub)

    sub = subparsers.add_parser(
        "features", help="solo bloques de afecto y/o hora")
    add_common_arguments(sub)
    return parser


def validate_args(parser: argparse.ArgumentParser,
                  args: argparse.Namespace) -> None:
    """Comprueba combinaciones de opciones incoherentes y aborta si las hay."""
    total = args.affect + args.time
    if args.affect < 0 or args.time < 0:
        parser.error("los pesos --affect y --time no pueden ser negativos")
    if args.command == "features":
        if total <= 0:
            parser.error("features necesita --affect y/o --time mayores "
                         "que 0")
    elif total >= 1:
        parser.error("--affect + --time debe ser menor que 1 (el resto "
                     "es el peso del bloque semántico)")
    if args.fit_on == "train" and not set(args.splits) & set(TRAIN_SPLITS):
        parser.error("--fit-on train necesita Trial o Train en --splits")
    if args.command == "tfidf" and args.svd < 1:
        parser.error("--svd debe ser al menos 1")
    if args.min_tokens < 0:
        parser.error("--min-tokens no puede ser negativo")


def make_fit_mask(split_column: pd.Series, fit_on: str) -> np.ndarray:
    """Devuelve qué unidades se usan para ajustar el vectorizador.

    ``all``: todas. ``train``: las de las particiones Trial y Train.
    """
    if fit_on == "all":
        return np.ones(len(split_column), dtype=bool)
    return split_column.isin(TRAIN_SPLITS).to_numpy()


def library_versions() -> dict[str, str]:
    """Devuelve las versiones de las librerías relevantes instaladas."""
    versions = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def run(args: argparse.Namespace) -> Path:
    """Ejecuta la vectorización completa y devuelve la carpeta creada."""
    timer = PhaseTimer()
    out_dir = Path(args.output_dir) / (args.name or build_name(args))
    prepare_directory(out_dir, args.overwrite)
    cache_dir = Path(args.cache_dir)
    config: dict = {"command": args.command, "args": vars(args).copy()}

    with timer.phase("load"):
        df = load_corpus(args.corpus_root, args.disorders, args.splits)

    with timer.phase("preprocess"):
        index = {}
        if needs_emoji_names(args.count_emoji_tokens, args.emojis):
            index = build_emoji_index(load_emoji_names())
        df, config["preprocessing"] = preprocess(
            df, args.min_tokens, args.count_emoji_tokens,
            args.keep_duplicates, args.emojis, index)

    with timer.phase("units"):
        units = build_units(df, args.unit)
        fit_mask = make_fit_mask(units.metadata["split"], args.fit_on)
        if not fit_mask.any():
            raise ValueError("--fit-on train: no hay unidades de Trial/Train")

    semantic = None
    vectorizer = None
    with timer.phase("vectorize"):
        if args.command == "tfidf":
            semantic, vectorizer, config["vectorizer"] = tfidf.vectorize(
                units, fit_mask, args)
        elif args.command == "transformer":
            semantic, vectorizer, config["vectorizer"] = (
                transformer.vectorize(df, units, fit_mask, args, cache_dir))

    blocks: dict[str, np.ndarray] = {}
    affect_block = time_block = None
    if args.affect > 0:
        with timer.phase("affect"):
            messages, config["affect"] = affect.message_affect(
                df["message"].tolist(), args.affect_tasks, cache_dir,
                not args.no_cache)
            affect_block = (messages if args.unit == "message" else
                            aggregate_rows(messages, units.owner, "mean"))
            config["affect"]["columns"] = affect.affect_columns(
                args.affect_tasks)
            blocks["affect"] = affect_block
    if args.time > 0:
        with timer.phase("time"):
            time_block, columns = time_features.unit_time_matrix(
                df["date"], units.owner, args.unit)
            config["time_columns"] = columns
            blocks["time"] = time_block

    with timer.phase("combine"):
        if semantic is not None:
            blocks["semantic"] = semantic
        matrix = combine_blocks(semantic, affect_block, time_block,
                                args.affect, args.time)

    with timer.phase("save"):
        config["timings_seconds"] = timer.seconds
        config["libraries"] = library_versions()
        save_vectorization(out_dir, matrix, units.metadata, config,
                           vectorizer, blocks)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada: interpreta los argumentos y lanza ``run``.

    Con ``--print-name`` solo imprime el nombre de la carpeta. Devuelve el
    código de salida del proceso (0 si todo fue bien).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    if args.print_name:
        print(args.name or build_name(args))
        return 0
    try:
        out_dir = run(args)
    except (FileExistsError, ValueError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Guardado en {out_dir}")
    return 0
