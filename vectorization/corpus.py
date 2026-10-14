"""Carga del corpus MentalRiskES en un DataFrame de mensajes."""

import json
from pathlib import Path

import pandas as pd

DISORDERS: list[str] = ["Ansiedad", "Depresion", "TCA"]
SPLITS: list[str] = ["Trial", "Train", "Test"]
# Particiones con las que se ajusta el vectorizador si --fit-on train.
TRAIN_SPLITS: list[str] = ["Trial", "Train"]

COLUMNS: list[str] = [
    "subject_key", "disorder", "split", "subject", "id_message", "message",
    "date", "label", "target",
]


def read_gold(split_dir: Path) -> pd.DataFrame:
    """Lee las etiquetas de una partición (carpeta ``Gold``).

    Devuelve una tabla con ``subject``, ``label`` (de ``gold_a.txt``, la
    etiqueta binaria) y ``target`` (de ``gold_c.txt``, que solo existe en
    Depresión). Si no hay ``Gold`` devuelve una tabla vacía. Las etiquetas
    solo sirven para evaluar: ningún vectorizador las usa.
    """
    path_a = split_dir / "Gold" / "gold_a.txt"
    if not path_a.exists():
        return pd.DataFrame(columns=["subject", "label", "target"])
    gold = pd.read_csv(path_a).rename(columns={"Subject": "subject"})
    gold = gold[["subject", "label"]]
    path_c = split_dir / "Gold" / "gold_c.txt"
    if path_c.exists():
        targets = pd.read_csv(path_c).rename(columns={"Subject": "subject"})
        gold = gold.merge(targets[["subject", "target"]], on="subject",
                          how="left")
    else:
        gold["target"] = None
    return gold


def subject_number(path: Path) -> int:
    """Devuelve N de ``subjectN.json`` para ordenar numéricamente."""
    return int(path.stem[len("subject"):])


def load_corpus(
    root: str | Path,
    disorders: list[str],
    splits: list[str],
) -> pd.DataFrame:
    """Carga los mensajes de los trastornos y particiones indicados.

    Devuelve un DataFrame con una fila por mensaje y las columnas de
    ``COLUMNS``. Los mensajes de cada usuario quedan contiguos y ordenados
    por fecha (otros módulos dependen de ello). Los nombres ``subjectN`` se
    repiten entre trastornos, así que la clave única del usuario es
    ``subject_key = "<Trastorno>/<Partición>/subjectN"``.
    """
    rows: list[dict] = []
    for disorder in disorders:
        for split in splits:
            split_dir = Path(root) / disorder / split
            if not split_dir.is_dir():
                continue
            gold = read_gold(split_dir).set_index("subject")
            paths = sorted(split_dir.glob("subject*.json"), key=subject_number)
            for path in paths:
                messages = json.loads(path.read_text(encoding="utf-8"))
                messages.sort(key=lambda message: message["date"])
                subject = path.stem
                label = target = None
                if subject in gold.index:
                    label = gold.at[subject, "label"]
                    target = gold.at[subject, "target"]
                for message in messages:
                    rows.append({
                        "subject_key": f"{disorder}/{split}/{subject}",
                        "disorder": disorder,
                        "split": split,
                        "subject": subject,
                        "id_message": message["id_message"],
                        "message": message["message"],
                        "date": message["date"],
                        "label": label,
                        "target": target,
                    })
    if not rows:
        raise ValueError(
            f"No se encontraron mensajes en {root} para {disorders} "
            f"y {splits}")
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df
