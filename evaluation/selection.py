"""Regla de selección de (eps, minPts) sin usar etiquetas (aportación P1).

Pasos: (1) configuraciones válidas, (2) lista corta por DBCV, (3) las que son
estables, (4) mejor DBCV entre las estables. Este módulo no lee nunca
columnas de etiquetas Gold.
"""

import pandas as pd

MAX_NOISE: float = 0.40
MAX_LARGEST: float = 0.70
MIN_CLUSTERS: int = 2
MIN_DBCV: float = 0.0
MIN_STAB_P10: float = 0.60
MIN_EPS_ARI: float = 0.80
SHORTLIST: int = 5
TIE_DBCV: float = 0.02


def is_valid(row: pd.Series) -> bool:
    """Una configuración es válida si no es el caso degenerado típico.

    Exige al menos 2 clusters, ruido y cluster mayor acotados y DBCV > 0.
    """
    return bool(
        row["n_clusters"] >= MIN_CLUSTERS
        and row["noise_rate"] <= MAX_NOISE
        and row["largest_cluster_frac"] <= MAX_LARGEST
        and row["dbcv"] > MIN_DBCV
    )


def add_validity(grid: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de la rejilla con la columna ``is_valid``."""
    grid = grid.copy()
    grid["is_valid"] = grid.apply(is_valid, axis=1)
    return grid


def shortlist(grid: pd.DataFrame, group_cols: list[str]) -> pd.Index:
    """Índices de las ``SHORTLIST`` mejores válidas por DBCV en cada grupo.

    Son las configuraciones a las que hay que calcular la estabilidad.
    """
    valid = add_validity(grid)
    valid = valid[valid["is_valid"]]
    top = (valid.sort_values("dbcv", ascending=False)
           .groupby(group_cols, sort=False).head(SHORTLIST))
    return top.index


def _pick(candidates: pd.DataFrame) -> pd.Series:
    """Mejor DBCV; entre empates (< 0.02) menos ruido y luego más minPts."""
    best = candidates["dbcv"].max()
    close = candidates[candidates["dbcv"] >= best - TIE_DBCV]
    close = close.sort_values(
        ["noise_rate", "min_samples"], ascending=[True, False])
    return close.iloc[0]


def select_one(group: pd.DataFrame) -> pd.Series:
    """Elige una configuración de un grupo y le añade ``status``.

    ``ok``: válida y estable. ``inestable``: la mejor válida, pero sin
    estabilidad suficiente. ``sin_estructura``: ninguna válida (resultado
    legítimo: el espacio no tiene estructura de densidad aprovechable).
    """
    group = add_validity(group)
    valid = group[group["is_valid"]]
    if valid.empty:
        scored = group.dropna(subset=["dbcv"])
        chosen = (scored if not scored.empty else group).sort_values(
            "dbcv", ascending=False).iloc[0].copy()
        chosen["status"] = "sin_estructura"
        return chosen
    if "stab_p10" in valid:
        checked = valid.dropna(subset=["stab_p10", "eps_ari"])
        stable = checked[(checked["stab_p10"] >= MIN_STAB_P10)
                         & (checked["eps_ari"] >= MIN_EPS_ARI)]
        if not stable.empty:
            chosen = _pick(stable).copy()
            chosen["status"] = "ok"
            return chosen
        if not checked.empty:
            valid = checked
    chosen = _pick(valid).copy()
    chosen["status"] = "inestable"
    return chosen


def select(grid: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aplica ``select_one`` a cada grupo (por ejemplo, cada espacio)."""
    rows = [select_one(group) for _, group in grid.groupby(group_cols)]
    return pd.DataFrame(rows).reset_index(drop=True)
