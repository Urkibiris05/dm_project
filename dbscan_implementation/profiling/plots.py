"""Dibuja las gráficas de tiempo y memoria a partir de ``results.csv``.

Uso (desde la raíz del repositorio)::

    python -m dbscan_implementation.profiling.plots
"""

import csv

import matplotlib

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .benchmark import RESULTS_DIR, RESULTS_FILE, SCENARIOS  # noqa: E402
from .implementations import IMPLEMENTATIONS  # noqa: E402

# Un color y un marcador fijos por implementación, en el mismo orden que
# ``IMPLEMENTATIONS``. El marcador permite distinguirlas sin ver el color.
COLORS: list[str] = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
]
MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*"]

SURFACE: str = "#fcfcfb"
TEXT: str = "#0b0b0b"
TEXT_SECONDARY: str = "#52514e"
GRID: str = "#e1e0d9"
AXIS: str = "#c3c2b7"

SCENARIO_TITLES: dict[str, str] = {
    "euclidea_2d": "Euclídea, 2 dimensiones",
    "euclidea_32d": "Euclídea, 32 dimensiones",
    "coseno_32d": "Coseno, 32 dimensiones",
}

# Por debajo de esto la medida de memoria es ruido del sistema.
MIN_MEMORY_MB: float = 0.1


def load_results() -> list[dict]:
    with open(RESULTS_FILE, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["n_samples"] = int(row["n_samples"])
        row["seconds"] = float(row["seconds"])
        row["memory_mb"] = max(float(row["memory_mb"]), MIN_MEMORY_MB)
    return rows


def format_number(value: float, _position: int = 0) -> str:
    """Escribe 0.001, 0.5, 20 o 2 000 sin notación científica."""
    if value >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:g}"


def style_axes(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, which="major", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axes.spines[side].set_color(AXIS)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, which="both")
    axes.tick_params(which="minor", length=0)


def plot_measure(
    rows: list[dict], column: str, title: str, y_label: str, file_name: str
) -> None:
    """Una figura con un panel por escenario para la columna indicada."""
    figure, panels = plt.subplots(
        1, len(SCENARIOS), figsize=(13, 5), sharey=True
    )
    figure.patch.set_facecolor(SURFACE)

    for panel, scenario in zip(panels, SCENARIOS):
        style_axes(panel)
        sizes: set[int] = set()
        for index, implementation in enumerate(IMPLEMENTATIONS):
            points = sorted(
                (row["n_samples"], row[column])
                for row in rows
                if row["scenario"] == scenario.name
                and row["implementation"] == implementation.name
            )
            if not points:
                continue
            x_values = [point[0] for point in points]
            y_values = [point[1] for point in points]
            sizes.update(x_values)
            panel.plot(
                x_values, y_values,
                color=COLORS[index], marker=MARKERS[index],
                linewidth=2, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1,
                label=implementation.label,
            )

        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xticks(sorted(sizes))
        panel.xaxis.set_major_formatter(FuncFormatter(format_number))
        panel.yaxis.set_major_formatter(FuncFormatter(format_number))
        panel.set_title(
            SCENARIO_TITLES[scenario.name], color=TEXT, fontsize=11, loc="left"
        )
        panel.set_xlabel("Número de puntos", color=TEXT_SECONDARY, fontsize=9)

    panels[0].set_ylabel(y_label, color=TEXT_SECONDARY, fontsize=9)

    # Una sola leyenda para toda la figura, con todas las implementaciones.
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0.045, 0.93),
        ncol=4, frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
    )
    figure.suptitle(
        title, color=TEXT, fontsize=14, x=0.05, y=0.98, ha="left"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))

    output = RESULTS_DIR / file_name
    figure.savefig(output, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"Gráfica guardada en {output}")


def main() -> None:
    rows = load_results()
    plot_measure(
        rows, "seconds",
        "Tiempo de DBSCAN según el número de puntos (ejes logarítmicos)",
        "Segundos (mediana)", "tiempo.png",
    )
    plot_measure(
        rows, "memory_mb",
        "Memoria extra en el pico según el número de puntos "
        "(ejes logarítmicos)",
        "MB", "memoria.png",
    )


if __name__ == "__main__":
    main()
