"""Rasgos de la hora del mensaje.

La hora y el día de la semana son cíclicos (las 23:59 están junto a las
00:00), así que se codifican con seno y coseno. Para el usuario se promedian
los rasgos de sus mensajes: la media de seno/coseno es la media circular y
la media de los indicadores de franja es la proporción de mensajes en ella.
"""

import argparse

import numpy as np
import pandas as pd

from .units import aggregate_rows

TIME_COLUMNS: list[str] = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "early_morning", "morning", "afternoon", "night", "weekend",
]
# A nivel de mensaje solo se usan las cuatro primeras (cíclicas).
MESSAGE_COLUMNS: int = 4


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Añade la opción del bloque de hora (común a los subcomandos)."""
    parser.add_argument("--time", type=float, default=0.0, metavar="W",
                        help="peso del bloque de hora (0 = no se usa)")


def message_time_matrix(dates: pd.Series) -> np.ndarray:
    """Calcula los rasgos de hora de cada mensaje (columnas ``TIME_COLUMNS``).

    Incluye seno/coseno de la hora (con minutos) y del día de la semana, los
    indicadores de franja (madrugada 0-6, mañana 6-12, tarde 12-18, noche
    18-24) y de fin de semana.
    """
    hour = (dates.dt.hour + dates.dt.minute / 60).to_numpy()
    weekday = dates.dt.dayofweek.to_numpy()
    hour_angle = 2 * np.pi * hour / 24
    weekday_angle = 2 * np.pi * weekday / 7
    return np.column_stack([
        np.sin(hour_angle), np.cos(hour_angle),
        np.sin(weekday_angle), np.cos(weekday_angle),
        hour < 6,
        (hour >= 6) & (hour < 12),
        (hour >= 12) & (hour < 18),
        hour >= 18,
        weekday >= 5,
    ]).astype(np.float32)


def unit_time_matrix(
    dates: pd.Series,
    owner: np.ndarray,
    unit: str,
) -> tuple[np.ndarray, list[str]]:
    """Calcula los rasgos de hora de cada unidad.

    Mensaje: las cuatro columnas cíclicas. Usuario: la media de las nueve
    columnas de sus mensajes. Devuelve ``(matriz, nombres_de_columna)``.
    """
    matrix = message_time_matrix(dates)
    if unit == "message":
        return matrix[:, :MESSAGE_COLUMNS], TIME_COLUMNS[:MESSAGE_COLUMNS]
    return aggregate_rows(matrix, owner, "mean"), TIME_COLUMNS
