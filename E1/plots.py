"""Gráficas de E1. Leen los CSV de resultados/ y escriben PNG en figuras/."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from evaluation import selection  # noqa: E402

from .config import (ANALYZERS, DISORDERS, FIGURES_DIR,  # noqa: E402
                     RESULTS_DIR, UNITS)
from .exp_kdist import NO_SVD  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
ANALYZER_COLOR = {"word": SERIES[0], "char_wb": SERIES[1]}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
BAD = "#e34948"

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
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _reference_svd(frame: pd.DataFrame) -> int:
    """SVD que se dibuja por defecto: 128 si existe, si no el mayor."""
    dims = sorted(set(frame["svd"]) - {NO_SVD})
    return 128 if 128 in dims else dims[-1]


def plot_kdist() -> None:
    curves = _read("e11_kdist.csv")
    svd = _reference_svd(curves)
    fig, axes = plt.subplots(3, 2, figsize=(9, 9), sharey="row")
    for i, disorder in enumerate(DISORDERS):
        for j, unit in enumerate(UNITS):
            ax = axes[i, j]
            min_samples = 10 if unit == "message" else 5
            for analyzer in ANALYZERS:
                for dim, style in ((svd, "-"), (NO_SVD, ":")):
                    part = curves[
                        (curves["disorder"] == disorder)
                        & (curves["unit"] == unit)
                        & (curves["analyzer"] == analyzer)
                        & (curves["svd"] == dim)
                        & (curves["min_samples"] == min_samples)]
                    if part.empty:
                        continue
                    label = f"{analyzer}, " + (
                        f"SVD {dim}" if dim else "sin SVD")
                    ax.plot(part["quantile"], part["kdist"], style,
                            color=ANALYZER_COLOR[analyzer], label=label)
            ax.set_title(f"{disorder}, {unit} (minPts={min_samples})")
            ax.set_xlabel("fracción de puntos (ordenados)")
            ax.set_ylabel("distancia coseno al k-ésimo vecino")
    axes[0, 0].legend(fontsize=7)
    _save(fig, "e11_kdist.png")


def plot_contrast() -> None:
    diag = _read("e11_diagnostico.csv").drop_duplicates(SPACE)
    fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=True)
    dims = sorted(diag["svd"].unique())
    positions = {dim: k for k, dim in enumerate(dims)}
    for i, disorder in enumerate(DISORDERS):
        for j, unit in enumerate(UNITS):
            ax = axes[i, j]
            for analyzer in ANALYZERS:
                part = diag[(diag["disorder"] == disorder)
                            & (diag["unit"] == unit)
                            & (diag["analyzer"] == analyzer)]
                part = part.sort_values("svd")
                ax.plot([positions[d] for d in part["svd"]],
                        part["relative_contrast"], "o-",
                        color=ANALYZER_COLOR[analyzer], label=analyzer)
            ax.set_title(f"{disorder}, {unit}")
            ax.set_ylabel("contraste relativo")
            ax.set_xticks(range(len(dims)), [
                "sin SVD" if d == NO_SVD else str(d) for d in dims])
    axes[0, 0].legend(fontsize=7)
    _save(fig, "e11_contraste.png")


SPACE = ["disorder", "unit", "analyzer", "svd"]


def _plot_vs_eps(column: str, ylabel: str, name: str) -> None:
    grid = _read("e12_grid.csv")
    svd = _reference_svd(grid)
    grid = grid[(grid["svd"] == svd) & (grid["eps_source"] == "quantile")]
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), sharex=False)
    for i, disorder in enumerate(DISORDERS):
        for j, (unit, analyzer) in enumerate(
                (u, a) for u in UNITS for a in ANALYZERS):
            ax = axes[i, j]
            part = grid[(grid["disorder"] == disorder)
                        & (grid["unit"] == unit)
                        & (grid["analyzer"] == analyzer)]
            for k, (min_samples, rows) in enumerate(
                    part.groupby("min_samples")):
                rows = rows.sort_values("eps")
                ax.plot(rows["eps"], rows[column], "o-", ms=3,
                        color=SERIES[k % len(SERIES)],
                        label=f"minPts={min_samples}")
            ax.set_title(f"{disorder}, {unit}, {analyzer}", fontsize=8)
            ax.set_xlabel("eps")
            ax.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"SVD {svd}", color=MUTED)
    _save(fig, name)


def plot_validity_map() -> None:
    robustness = _read("e13_robustez_espacio.csv")
    robustness["row"] = (robustness["disorder"] + " "
                         + robustness["unit"] + " " + robustness["analyzer"])
    table = robustness.pivot(index="row", columns="svd", values="pct_valid")
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(table.values, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if value > 0.5 else INK)
    ax.set_xlabel("componentes de la SVD")
    ax.set_title("Fracción de configuraciones válidas")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    _save(fig, "e12_mapa_validez.png")


def plot_stability() -> None:
    grid = _read("e13_estabilidad.csv").dropna(subset=["stab_p10"])
    fig, ax = plt.subplots(figsize=(6, 5))
    for analyzer in ANALYZERS:
        part = grid[grid["analyzer"] == analyzer]
        ax.scatter(part["dbcv"], part["stab_p10"], s=14,
                   color=ANALYZER_COLOR[analyzer], label=analyzer)
    ax.axhline(selection.MIN_STAB_P10, color=BAD, lw=0.8, ls="--")
    ax.set_xlabel("DBCV")
    ax.set_ylabel("estabilidad (percentil 10 del ARI bootstrap)")
    ax.set_title("Lista corta: DBCV frente a estabilidad")
    ax.legend()
    _save(fig, "e13_estabilidad.png")


def plot_summary() -> None:
    chosen = _read("e13_seleccion.csv")
    external = _read("e14_externas.csv")
    external = external[external["target_type"] == "label"]
    chosen = chosen.merge(
        external[SPACE + ["ami_all"]], on=SPACE, how="left")
    keys = ["disorder", "unit", "analyzer"]
    mean = chosen.groupby(keys)[
        ["dbcv", "noise_rate", "largest_cluster_frac", "ami_all"]
    ].mean().reset_index()
    groups = [(d, u) for d in DISORDERS for u in UNITS]
    panels = [("dbcv", "DBCV"), ("noise_rate", "tasa de ruido"),
              ("largest_cluster_frac", "fracción del cluster mayor"),
              ("ami_all", "AMI con la etiqueta (solo evaluación)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    x = np.arange(len(groups))
    for ax, (column, title) in zip(axes.ravel(), panels):
        for k, analyzer in enumerate(ANALYZERS):
            values = [
                mean[(mean["disorder"] == d) & (mean["unit"] == u)
                     & (mean["analyzer"] == analyzer)][column].mean()
                for d, u in groups]
            ax.bar(x + (k - 0.5) * 0.38, values, 0.38,
                   color=ANALYZER_COLOR[analyzer], label=analyzer)
        ax.set_xticks(x, [f"{d}\n{u}" for d, u in groups], fontsize=7)
        ax.set_title(title)
    axes[0, 0].legend()
    fig.suptitle(
        "Mejor configuración por espacio (incluye las sin_estructura, no "
        "válidas), media sobre las dimensiones de SVD", color=MUTED)
    _save(fig, "e14_resumen.png")


def main() -> None:
    plot_kdist()
    plot_contrast()
    _plot_vs_eps("noise_rate", "tasa de ruido", "e12_ruido_vs_eps.png")
    _plot_vs_eps("dbcv", "DBCV", "e12_dbcv_vs_eps.png")
    plot_validity_map()
    plot_stability()
    plot_summary()
