"""Gráficas de E0. Leen los CSV de resultados/ y escriben PNG en figuras/."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from .datasets import load_datasets  # noqa: E402

# Paleta categórica de referencia (skill dataviz, modo claro) y tintas.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
NOISE_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
BAD = "#e34948"

TWO_D = ["blobs", "varied", "aniso", "moons", "circles", "no_structure"]
VARIANT_MARKERS = {
    "ours_brute": "o", "ours_blocked": "s",
    "ours_optimized_blocked": "^", "ours_optimized_dedup": "D",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
})


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / name)


def _save(fig: Figure, name: str) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _cluster_color(label: int) -> str:
    return NOISE_COLOR if label < 0 else SERIES[label % len(SERIES)]


def _annotated_heatmap(
    ax: Axes, matrix: pd.DataFrame, vmin: float, vmax: float, fmt: str,
) -> None:
    """Heatmap de un solo tono (azul) con el valor escrito en cada celda."""
    image = ax.imshow(matrix.values, cmap="Blues", vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=40,
                  ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    middle = (vmin + vmax) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    fontsize=8, color="white" if value > middle else INK)
    ax.figure.colorbar(image, ax=ax, fraction=0.04, pad=0.03)


def plot_datasets() -> None:
    datasets = load_datasets()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, name in zip(axes.ravel(), TWO_D):
        dataset = datasets[name]
        colors = [SERIES[int(v) % len(SERIES)] for v in dataset.y]
        ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=colors, s=6)
        ax.set_title(f"{name} (n={len(dataset.X)})")
    others = ", ".join(
        f"{n}: {datasets[n].X.shape[0]}x{datasets[n].X.shape[1]}"
        for n in ("iris", "wine", "digits", "emb_384d")
    )
    fig.suptitle("Datasets de juguete 2D (color = etiqueta, solo para "
                 "visualizar)")
    fig.text(0.5, -0.01, f"Además: {others}", ha="center", color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e00_datasets.png")


def plot_kdist() -> None:
    kdist = _read("e00_kdist.csv")
    table = _read("e01_vs_sklearn.csv")
    names = list(kdist["dataset"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    for ax, name in zip(axes.ravel(), names):
        curve = kdist[kdist["dataset"] == name]
        ax.plot(curve["rank"], curve["kdist"], color=SERIES[0], lw=2)
        grid = table[
            (table["dataset"] == name) & (table["metric"] == "euclidean")
            & (table["min_samples"] == 5)
            & (table["variant"] == "ours_brute")
        ]
        for eps in grid["eps"]:
            ax.axhline(eps, color=SERIES[1], lw=0.7, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("puntos ordenados")
        ax.set_ylabel("distancia al 5.º vecino")
    fig.suptitle("Gráfico k-dist (min_samples = 5) y valores de eps "
                 "probados (líneas naranjas)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e00_kdist.png")


def plot_e01_heatmap() -> None:
    table = _read("e01_vs_sklearn.csv")
    pct = table.assign(eq=table["exact_equal"] * 100).pivot_table(
        index="variant", columns="metric", values="eq", aggfunc="mean"
    )
    counts = table.groupby(["variant", "metric"]).size().iloc[0]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _annotated_heatmap(ax, pct, 0, 100, ".1f")
    ax.set_title("Particiones idénticas a scikit-learn (%), "
                 f"{counts} casos por celda")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e01_coincidencia_heatmap.png")


def plot_e01_partitions() -> None:
    data = _read("e01_particiones_2d.csv")
    columns = ["sklearn", "ours_brute", "ours_optimized_blocked"]
    titles = ["scikit-learn", "Nuestro (brute)",
              "Nuestro (optimized_blocked)", "Diferencias"]
    fig, axes = plt.subplots(len(TWO_D), 4, figsize=(12, 3 * len(TWO_D)))
    for row, name in enumerate(TWO_D):
        part = data[data["dataset"] == name]
        for col, column in enumerate(columns):
            ax = axes[row, col]
            ax.grid(False)
            colors = [_cluster_color(v) for v in part[column]]
            sizes = np.where(part["is_core"], 12, 4)
            ax.scatter(part["x"], part["y"], c=colors, s=sizes)
            if row == 0:
                ax.set_title(titles[col])
            if col == 0:
                ax.set_ylabel(name)
        ax = axes[row, 3]
        ax.grid(False)
        ax.scatter(part["x"], part["y"], c=NOISE_COLOR, s=3)
        different = (part["ours_brute"] != part["sklearn"]) | (
            part["ours_optimized_blocked"] != part["sklearn"])
        ax.scatter(part["x"][different], part["y"][different], c=BAD,
                   marker="x", s=30)
        ax.text(0.03, 0.95, f"{int(different.sum())} diferencias",
                transform=ax.transAxes, va="top")
        if row == 0:
            ax.set_title(titles[3])
    fig.suptitle("Euclídea, min_samples = 5, eps = cuantil 0.7 del k-dist "
                 "(núcleo grande, frontera pequeño, ruido gris)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e01_particiones_2d.png")


def plot_e01_nclusters() -> None:
    table = _read("e01_vs_sklearn.csv")
    table = table[(table["metric"] == "euclidean")
                  & (table["min_samples"] == 5)]
    fig, axes = plt.subplots(2, len(TWO_D), figsize=(16, 5.5), sharex="col")
    for col, name in enumerate(TWO_D):
        part = table[table["dataset"] == name]
        base = part[part["variant"] == "ours_brute"]
        axes[0, col].plot(base["eps"], base["n_clusters_sklearn"],
                          color=INK, lw=2, label="scikit-learn")
        for k, (variant, marker) in enumerate(VARIANT_MARKERS.items()):
            sub = part[part["variant"] == variant]
            axes[0, col].scatter(sub["eps"], sub["n_clusters_ours"],
                                 marker=marker, s=22, color=SERIES[k],
                                 label=variant, zorder=3)
            axes[1, col].scatter(sub["eps"], sub["noise_rate"],
                                 marker=marker, s=22, color=SERIES[k])
        axes[0, col].set_title(name)
        axes[1, col].set_xlabel("eps")
    axes[0, 0].set_ylabel("n.º de clusters")
    axes[1, 0].set_ylabel("tasa de ruido")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Clusters y ruido frente a eps: nuestras variantes "
                 "(marcadores) sobre scikit-learn (línea)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e01_nclusters_vs_eps.png")


def plot_e02_identity() -> None:
    data = _read("e02_identidad_distancias.csv")
    names = list(data["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(9, 4))
    for ax, name in zip(np.atleast_1d(axes), names):
        part = data[data["dataset"] == name]
        top = max(part["two_d_cos"].max(), part["d_l2_squared"].max())
        ax.plot([0, top], [0, top], color=SERIES[1], lw=1.5, label="y = x")
        ax.scatter(part["two_d_cos"], part["d_l2_squared"], s=6,
                   color=SERIES[0])
        error = part["abs_error"].max()
        ax.set_title(f"{name} (error máx. {error:.1e})")
        ax.set_xlabel("2 · d_cos")
        ax.set_ylabel("d_euclídea²")
        ax.legend(frameon=False)
    fig.suptitle("Vectores de norma 1: d² = 2 · d_cos")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e02_identidad_distancias.png")


def plot_e02_bars() -> None:
    summary = _read("e02_resumen.csv")
    names = list(summary["implementation"].unique())
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (flag, label) in enumerate(
        [(True, "normalizado"), (False, "sin normalizar")]
    ):
        values = []
        for name in names:
            row = summary[(summary["implementation"] == name)
                          & (summary["normalized"] == flag)]
            values.append(
                row["pct_exact_equal"].iloc[0] if len(row) else np.nan
            )
        bars = ax.bar(x + (k - 0.5) * 0.38, values, 0.36,
                      color=SERIES[0] if flag else SERIES[1], label=label)
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 1,
                        f"{value:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("% particiones idénticas coseno vs euclídea")
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title("Coseno vs euclídea con eps_l2 = sqrt(2 · eps_cos)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e02_cos_vs_l2.png")


def plot_e03_matrix() -> None:
    matrix = pd.read_csv(RESULTS_DIR / "e03_ari_matriz.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    low = min(0.99, float(np.nanmin(matrix.values)))
    _annotated_heatmap(ax, matrix, low, 1.0, ".3f")
    ax.set_title("ARI medio entre implementaciones (euclídea)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e03_ari_matriz.png")


def plot_e03_disagreements() -> None:
    data = _read("e03_desacuerdos.csv")
    group = data.groupby("implementation")[
        ["n_diff_core", "n_diff_border", "n_diff_noise",
         "n_ambiguous_border"]].mean()
    x = np.arange(len(group))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(group))
    for column, color, label in [
        ("n_diff_core", SERIES[0], "núcleo"),
        ("n_diff_border", SERIES[1], "frontera"),
        ("n_diff_noise", NOISE_COLOR, "ruido"),
    ]:
        ax.bar(x, group[column], 0.6, bottom=bottom, color=color,
               label=label, edgecolor=SURFACE, linewidth=1)
        bottom += group[column].values
    ax.plot(x, group["n_ambiguous_border"], color=INK, marker="_",
            ms=18, ls="none", label="fronteras ambiguas (media)")
    ax.set_xticks(x, group.index, rotation=25, ha="right")
    ax.set_ylabel("puntos distintos frente a sklearn_brute (media)")
    ax.set_title("Dónde difieren las particiones")
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e03_desacuerdos.png")


def plot_e03_reproducibility() -> None:
    data = _read("e03_reproducibilidad.csv")
    perms = data[data["kind"] == "permutation"]
    summary = _read("e03_reproducibilidad_resumen.csv")
    names = list(summary["implementation"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot([perms[perms["implementation"] == n]["ari"] for n in names],
               tick_labels=names, patch_artist=True,
               boxprops={"facecolor": SERIES[0], "alpha": 0.6},
               medianprops={"color": INK})
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    n_deterministic = int(summary["determinista"].sum())
    ax.set_ylabel("ARI frente a la ejecución sin permutar")
    ax.set_title("Invariancia al orden de los datos. Deterministas: "
                 f"{n_deterministic}/{len(names)}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "e03_reproducibilidad.png")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    for function in (
        plot_datasets, plot_kdist, plot_e01_heatmap, plot_e01_partitions,
        plot_e01_nclusters, plot_e02_identity, plot_e02_bars,
        plot_e03_matrix, plot_e03_disagreements, plot_e03_reproducibility,
    ):
        function()
        print(f"  figura: {function.__name__}")
