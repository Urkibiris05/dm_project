"""Corpus sintético pequeño para los tests (sin descargar modelos)."""

import json
from pathlib import Path

import pytest

MESSAGES: dict[str, dict[str, list[tuple[str, str]]]] = {
    "Depresion": {
        "subject1": [
            ("2021-01-02 23:30:00", "hoy me siento muy cansado y triste"),
            ("2021-01-02 23:40:00", "jaja cara llorando de risa"),
            ("2021-01-03 09:00:00", "no puedo dormir por las noches"),
            ("2021-01-03 09:05:00", "no puedo dormir por las noches"),
            ("2021-01-04 15:00:00", "ok"),
        ],
        "subject2": [
            ("2021-01-02 10:00:00", "me encanta salir a correr por la tarde"),
            ("2021-01-02 11:00:00", "hoy comimos pasta con toda la familia"),
            ("2021-01-05 22:00:00", "mañana empiezo el curso de cocina"),
        ],
    },
    "TCA": {
        "subject1": [
            ("2021-02-01 01:00:00", "no quiero comer nada de nada hoy"),
            ("2021-02-01 02:00:00", "siento mucha culpa después de cenar"),
        ],
    },
}


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    """Crea un corpus con dos trastornos que comparten el nombre subject1."""
    root = tmp_path / "corpus"
    for disorder, subjects in MESSAGES.items():
        split_dir = root / disorder / "Train"
        (split_dir / "Gold").mkdir(parents=True)
        gold = ["Subject,label"]
        for number, (subject, messages) in enumerate(subjects.items()):
            data = [{"id_message": number * 100 + i, "date": date,
                     "message": text}
                    for i, (date, text) in enumerate(messages)]
            (split_dir / f"{subject}.json").write_text(
                json.dumps(data), encoding="utf-8")
            gold.append(f"{subject},{number % 2}")
        (split_dir / "Gold" / "gold_a.txt").write_text(
            "\n".join(gold) + "\n", encoding="utf-8")
    return root
