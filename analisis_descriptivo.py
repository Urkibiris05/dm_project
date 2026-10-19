"""Análisis descriptivo reproducible del corpus MentalRiskES.

Genera tablas CSV, gráficos PNG y un informe JSON/Markdown en
``resultados_analisis_descriptivo``. No modifica el corpus de entrada.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


EMOTICON_PATTERN = re.compile(
    r"^([x:;=8]-?[)dpo3v\]\[)(/\|\\]|xd|<3|t_t|\^\^)$",
    flags=re.IGNORECASE,
)
EMOJI_KEYWORDS = ("cara", "corazón", "mano", "dedo", "sonrisa", "ojos", "pulgar")


def read_labels(split_dir: Path) -> pd.DataFrame:
    """Lee las etiquetas de sujeto disponibles para una partición."""
    gold_dir = split_dir / "Gold"
    label_path = gold_dir / "gold_a.txt"
    if not label_path.exists():
        return pd.DataFrame(columns=["subject_id", "class_label", "class_name"])

    labels = pd.read_csv(label_path).rename(
        columns={"Subject": "subject_id", "label": "class_label"}
    )
    labels = labels[["subject_id", "class_label"]]
    target_path = gold_dir / "gold_c.txt"
    if target_path.exists():
        targets = pd.read_csv(target_path).rename(
            columns={"Subject": "subject_id", "target": "class_name"}
        )
        labels = labels.merge(
            targets[["subject_id", "class_name"]], on="subject_id", how="left"
        )
    else:
        labels["class_name"] = pd.NA
    return labels


def load_corpus(corpus_root: Path) -> pd.DataFrame:
    """Carga todos los JSON del corpus, conservando su partición y trastorno."""
    rows: list[dict[str, Any]] = []
    for file_path in sorted(corpus_root.glob("**/*.json")):
        relative_parts = file_path.relative_to(corpus_root).parts
        if len(relative_parts) < 3:
            continue
        disorder, split = relative_parts[-3], relative_parts[-2]
        subject_id = file_path.stem
        try:
            content = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"No se pudo leer {file_path}: {exc}") from exc

        messages = content if isinstance(content, list) else [content]
        for message in messages:
            row: dict[str, Any] = {
                "subject_id": subject_id,
                "trastorno": disorder,
                "dataset_type": split,
            }
            if isinstance(message, dict):
                row.update(message)
            else:
                row["message"] = message
            rows.append(row)

    if not rows:
        raise ValueError(f"No se encontraron archivos JSON en {corpus_root}")
    df = pd.DataFrame(rows)
    if "message" not in df:
        df["message"] = pd.NA
    df["trastorno"] = df["trastorno"].replace(
        to_replace=r".*Depres.*", value="Depresion", regex=True
    )
    labels = pd.concat(
        [
            read_labels(corpus_root / disorder / split).assign(
                trastorno=disorder, dataset_type=split
            )
            for disorder in sorted(
                {
                    path.relative_to(corpus_root).parts[-3]
                    for path in corpus_root.glob("**/subject*.json")
                }
            )
            for split in sorted(
                {
                    path.relative_to(corpus_root).parts[-2]
                    for path in corpus_root.glob(
                        f"{disorder}/**/subject*.json"
                    )
                }
            )
        ],
        ignore_index=True,
    )
    labels["trastorno"] = labels["trastorno"].replace(
        to_replace=r".*Depres.*", value="Depresion", regex=True
    )
    labels["class_name"] = labels["class_name"].fillna(
        labels["class_label"].map({0: "control", 1: "suffer"})
    )
    df = df.merge(
        labels,
        on=["subject_id", "trastorno", "dataset_type"],
        how="left",
    )
    df["message"] = df["message"].astype("string")
    df["word_count"] = df["message"].fillna("").str.split().str.len()
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def is_emoticon_or_emoji(text: object) -> bool:
    value = str(text).strip().lower()
    if EMOTICON_PATTERN.match(value):
        return True
    return value.startswith(EMOJI_KEYWORDS)


def save_table(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    df.to_csv(output_dir / f"{name}.csv", index=True, encoding="utf-8")


def make_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    """Guarda las visualizaciones del análisis sin requerir un entorno gráfico."""
    counts = pd.crosstab(df["dataset_type"], df["trastorno"])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    editorial_cmap = LinearSegmentedColormap.from_list(
        "editorial_beige_burgundy",
        ["#f7efe3", "#e1b98f", "#a85d4a", "#632b35"],
    )
    image = ax.imshow(
        counts.to_numpy(),
        cmap=editorial_cmap,
        aspect="auto",
    )
    ax.set_xticks(range(len(counts.columns)), counts.columns)
    ax.set_yticks(range(len(counts.index)), counts.index)
    ax.set_title("Mensajes por partición y trastorno")
    ax.set_xlabel("Trastorno")
    ax.set_ylabel("Partición")
    for row in range(len(counts.index)):
        for column in range(len(counts.columns)):
            value = counts.iloc[row, column]
            if value:
                ax.text(
                    column,
                    row,
                    f"{value:,}".replace(",", r"\,"),
                    ha="center",
                    va="center",
                    color="white"
                    if value > counts.to_numpy().max() * 0.45
                    else "#4a2525",
                    fontsize=9,
                )
    fig.colorbar(image, ax=ax, label="Número de mensajes", pad=0.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "distribucion_particion_trastorno.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["word_count"], bins=100, color="#2878b5", edgecolor="white")
    ax.set_xlim(0, max(1, df["word_count"].quantile(0.95)))
    ax.set_title("Distribución del número de palabras por mensaje (hasta P95)")
    ax.set_xlabel("Número de palabras")
    ax.set_ylabel("Frecuencia")
    fig.tight_layout()
    fig.savefig(plots_dir / "distribucion_longitud_mensajes.png", dpi=160)
    plt.close(fig)

    boxplot_data = []
    boxplot_labels = []
    for (split, disorder), group in df.groupby(
        ["dataset_type", "trastorno"], sort=True
    ):
        boxplot_data.append(group["word_count"].to_numpy())
        boxplot_labels.append(f"{disorder}\n{split}")
    upper_limit = df["word_count"].quantile(0.99)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(
        boxplot_data,
        tick_labels=boxplot_labels,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.35},
    )
    ax.set_ylim(0, upper_limit)
    ax.set_title("Longitud de los mensajes por trastorno y partición")
    ax.set_xlabel("Trastorno y partición")
    ax.set_ylabel("Número de palabras")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "boxplot_longitud_por_grupo.png", dpi=160)
    plt.close(fig)

    top_words = df.attrs["top_words"]
    words = pd.Series(dict(top_words)).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    positions = range(len(words))
    ax.hlines(positions, 0, words.to_numpy(), color="#9ecae1", linewidth=2)
    ax.plot(words.to_numpy(), positions, "o", color="#2171b5", markersize=7)
    ax.set_yticks(list(positions), words.index)
    ax.set_title("15 palabras más frecuentes")
    ax.set_xlabel("Frecuencia")
    ax.set_ylabel("Palabra")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "palabras_mas_frecuentes.png", dpi=160)
    plt.close(fig)

    class_counts = pd.crosstab(
        [df["trastorno"], df["dataset_type"]], df["class_name"]
    )
    if not class_counts.empty and class_counts.shape[1] > 1:
        proportions = class_counts.div(class_counts.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(11, 5.5))
        left = pd.Series(0.0, index=proportions.index)
        colors = plt.cm.Set2(
            [index / max(1, len(proportions.columns) - 1)
             for index in range(len(proportions.columns))]
        )
        for color, class_name in zip(colors, proportions.columns):
            values = proportions[class_name].fillna(0)
            ax.barh(
                range(len(proportions)),
                values,
                left=left,
                color=color,
                label=class_name,
                height=0.72,
            )
            left += values
        labels = [f"{disorder} — {split}" for disorder, split in proportions.index]
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Porcentaje de mensajes")
        ax.set_title("Composición porcentual de las clases")
        ax.legend(title="Clase", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "distribucion_clases.png", dpi=160)
        plt.close(fig)


def run_analysis(corpus_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tablas"
    plots_dir = output_dir / "graficos"
    tables_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)

    df = load_corpus(corpus_root)
    df["message_missing"] = df["message"].isna() | df["message"].eq("")

    distribution = pd.crosstab(
        df["dataset_type"], df["trastorno"], margins=True, margins_name="Total"
    )
    subject_df = df.drop_duplicates(
        subset=["subject_id", "trastorno", "dataset_type"]
    )
    subject_class_distribution = (
        subject_df.groupby(["trastorno", "dataset_type", "class_name"], dropna=False)
        .size()
        .rename("sujetos")
        .reset_index()
    )
    subject_class_distribution["porcentaje"] = (
        subject_class_distribution["sujetos"]
        / subject_class_distribution.groupby(["trastorno", "dataset_type"])[
            "sujetos"
        ].transform("sum")
        * 100
    ).round(3)
    message_class_distribution = (
        df.groupby(["trastorno", "dataset_type", "class_name"], dropna=False)
        .size()
        .rename("mensajes")
        .reset_index()
    )
    message_class_distribution["porcentaje"] = (
        message_class_distribution["mensajes"]
        / message_class_distribution.groupby(["trastorno", "dataset_type"])[
            "mensajes"
        ].transform("sum")
        * 100
    ).round(3)
    group_summary = (
        df.groupby(["dataset_type", "trastorno"])
        .agg(
            mensajes=("message", "size"),
            sujetos=("subject_id", "nunique"),
            palabras_media=("word_count", "mean"),
            palabras_mediana=("word_count", "median"),
            palabras_desv_std=("word_count", "std"),
            palabras_min=("word_count", "min"),
            palabras_max=("word_count", "max"),
        )
        .round(3)
    )
    length_summary = df["word_count"].describe().to_frame("word_count").round(3)
    exact_lengths = df["word_count"].value_counts().sort_index().rename("mensajes")

    short = df[df["word_count"] <= 3].copy()
    short["is_emoticon"] = short["message"].map(is_emoticon_or_emoji)
    short_summary = pd.DataFrame(
        {
            "valor": [
                len(short),
                int(short["is_emoticon"].sum()),
                round(float(short["is_emoticon"].mean() * 100), 3)
                if len(short)
                else 0.0,
            ]
        },
        index=["mensajes_hasta_3_palabras", "emoticonos_o_emojis", "porcentaje_emoticonos"],
    )

    tokens = re.findall(
        r"\b\w+\b", " ".join(df["message"].fillna("").tolist()).lower()
    )
    frequencies = Counter(tokens)
    top_words = frequencies.most_common(15)
    vocabulary = pd.DataFrame(
        {
            "metrica": [
                "tokens_totales",
                "vocabulario_unico",
                "hapax_legomena",
                "porcentaje_hapax_sobre_vocabulario",
            ],
            "valor": [
                len(tokens),
                len(frequencies),
                sum(freq == 1 for freq in frequencies.values()),
                round(
                    100 * sum(freq == 1 for freq in frequencies.values())
                    / len(frequencies),
                    3,
                )
                if frequencies
                else 0.0,
            ],
        }
    ).set_index("metrica")
    top_words_table = pd.DataFrame(top_words, columns=["palabra", "frecuencia"]).set_index(
        "palabra"
    )

    data_quality = pd.DataFrame(
        {
            "metrica": [
                "filas_totales",
                "mensajes_nulos_o_vacios",
                "mensajes_duplicados_exactos",
                "archivos_subject",
            ],
            "valor": [
                len(df),
                int(df["message_missing"].sum()),
                int(df.duplicated(subset=["message"]).sum()),
                int(df[["trastorno", "dataset_type", "subject_id"]]
                    .drop_duplicates()
                    .shape[0]),
            ],
        }
    ).set_index("metrica")

    subject_summary = (
        df.groupby(["dataset_type", "trastorno", "subject_id"])
        .agg(mensajes=("message", "size"), palabras_mediana=("word_count", "median"))
        .reset_index()
    )

    save_table(distribution, tables_dir, "distribucion_particion_trastorno")
    save_table(
        subject_class_distribution,
        tables_dir,
        "distribucion_clases_por_sujeto",
    )
    save_table(
        message_class_distribution,
        tables_dir,
        "distribucion_clases_por_mensaje",
    )
    save_table(group_summary, tables_dir, "resumen_por_grupo")
    save_table(length_summary, tables_dir, "estadisticas_longitud")
    save_table(exact_lengths, tables_dir, "frecuencia_longitudes")
    save_table(short_summary, tables_dir, "mensajes_cortos_resumen")
    save_table(
        short[short["is_emoticon"]]["message"].value_counts().head(10).rename("frecuencia"),
        tables_dir,
        "emoticonos_mas_frecuentes",
    )
    save_table(
        short[~short["is_emoticon"]]["message"].value_counts().head(10).rename("frecuencia"),
        tables_dir,
        "mensajes_cortos_no_emoticonos",
    )
    save_table(vocabulary, tables_dir, "vocabulario_resumen")
    save_table(top_words_table, tables_dir, "palabras_mas_frecuentes")
    save_table(data_quality, tables_dir, "calidad_datos")
    save_table(subject_summary, tables_dir, "resumen_por_sujeto")

    df.attrs["top_words"] = top_words
    make_plots(df, plots_dir)
    df.drop(columns=["message_missing"]).to_csv(
        tables_dir / "corpus_con_metricas.csv", index=False, encoding="utf-8"
    )

    report = {
        "corpus": str(corpus_root),
        "filas": len(df),
        "trastornos": df["trastorno"].value_counts().to_dict(),
        "particiones": df["dataset_type"].value_counts().to_dict(),
        "clases_por_sujeto": subject_class_distribution.to_dict(orient="records"),
        "clases_por_mensaje": message_class_distribution.to_dict(orient="records"),
        "calidad_datos": data_quality["valor"].to_dict(),
        "longitud_palabras": length_summary["word_count"].to_dict(),
        "vocabulario": vocabulary["valor"].to_dict(),
        "mensajes_cortos": short_summary["valor"].to_dict(),
        "palabras_mas_frecuentes": dict(top_words),
    }
    (output_dir / "resumen.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    markdown = [
        "# Análisis descriptivo del corpus",
        "",
        f"- Mensajes analizados: **{len(df):,}**",
        f"- Sujetos/archivos: **{df[['trastorno', 'dataset_type', 'subject_id']].drop_duplicates().shape[0]:,}**",
        f"- Vocabulario: **{len(frequencies):,}** palabras únicas",
        f"- Mensajes nulos o vacíos: **{int(df['message_missing'].sum()):,}**",
        "",
        "La distribución de clases se encuentra en "
        "`tablas/distribucion_clases_por_sujeto.csv` y "
        "`tablas/distribucion_clases_por_mensaje.csv`. En Depresión se usan "
        "las categorías de `gold_c.txt`; en Ansiedad y TCA se usa la clase "
        "binaria de `gold_a.txt`.",
        "",
        "Las tablas están en `tablas/` y los gráficos en `graficos/`. "
        "El archivo `resumen.json` contiene las métricas principales.",
    ]
    (output_dir / "informe.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Análisis completado. Resultados: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("MentalRiskEs2023/corpus"),
        help="Ruta a la carpeta raíz del corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resultados_analisis_descriptivo"),
        help="Carpeta de salida para tablas, gráficos e informe.",
    )
    args = parser.parse_args()
    run_analysis(args.corpus, args.output)


if __name__ == "__main__":
    main()
