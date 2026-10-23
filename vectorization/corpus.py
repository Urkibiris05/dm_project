"""Carga del corpus MentalRiskES en un DataFrame de mensajes."""

import json
import re
import unicodedata
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
    """Devuelve N de un archivo ``subjectN.json`` para ordenarlo."""
    match = re.search(r"subject[_-]?(\d+)$", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _canonical_name(value: str) -> str:
    """Normaliza nombres de carpetas para comparar acentos y mayúsculas."""
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized
                   if not unicodedata.combining(char)).casefold()


def _find_directory(parent: Path, name: str) -> Path | None:
    """Encuentra una carpeta aunque varíe el uso de acentos o mayúsculas."""
    wanted = _canonical_name(name)
    for candidate in parent.iterdir():
        if candidate.is_dir() and _canonical_name(candidate.name) == wanted:
            return candidate
    return None


def _subject_files(split_dir: Path) -> list[Path]:
    """Devuelve JSON de sujetos, incluidos nombres como ``subject_1``."""
    return sorted(
        (path for path in split_dir.rglob("*.json")
         if re.match(r"subject[_-]?\d+$", path.stem, re.IGNORECASE)),
        key=lambda path: (subject_number(path), str(path)),
    )


def _messages_from_json(path: Path) -> list[dict]:
    """Lee una lista de mensajes o un objeto que contiene esa lista."""
    content = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(content, list):
        messages = content
    elif isinstance(content, dict):
        messages = content.get("messages", content.get("data", [content]))
    else:
        raise ValueError(f"Formato JSON no válido en {path}")
    if not isinstance(messages, list) or not all(
            isinstance(message, dict) for message in messages):
        raise ValueError(f"Los mensajes de {path} deben ser objetos JSON")
    return messages


def _message_value(message: dict, *names: str) -> object:
    """Obtiene un campo usando las variantes presentes en el corpus."""
    for name in names:
        if name in message:
            return message[name]
    return None


def _subject_key(value: str) -> str:
    """Normaliza ``subject1``, ``subject_1`` y ``subject-1``."""
    return re.sub(r"[_-]", "", value).casefold()


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
            disorder_dir = _find_directory(Path(root), disorder)
            split_dir = (_find_directory(disorder_dir, split)
                         if disorder_dir is not None else None)
            if split_dir is None:
                continue
            gold = read_gold(split_dir).set_index("subject")
            gold_by_subject = {
                _subject_key(str(subject)): subject
                for subject in gold.index
            }
            paths = _subject_files(split_dir)
            for path in paths:
                messages = _messages_from_json(path)
                messages.sort(key=lambda message: str(
                    _message_value(message, "date", "datetime", "timestamp")
                    or ""))
                subject = path.stem
                label = target = None
                gold_subject = gold_by_subject.get(_subject_key(subject))
                if gold_subject is not None:
                    label = gold.at[gold_subject, "label"]
                    target = gold.at[gold_subject, "target"]
                for index, message in enumerate(messages):
                    text = _message_value(message, "message", "text",
                                          "content")
                    date = _message_value(message, "date", "datetime",
                                          "timestamp")
                    if text is None or date is None:
                        raise ValueError(
                            f"Mensaje sin texto o fecha en {path}: "
                            f"campos disponibles={list(message)}")
                    rows.append({
                        "subject_key": f"{disorder}/{split}/{subject}",
                        "disorder": disorder,
                        "split": split,
                        "subject": subject,
                        "id_message": _message_value(
                            message, "id_message", "id", "message_id"
                        ) or index,
                        "message": text,
                        "date": date,
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
