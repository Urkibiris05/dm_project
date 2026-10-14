"""Capa de preprocesado que se aplica antes de cualquier vectorización.

Orden de los pasos (cada uno es una función pequeña e independiente):

1. filtro de mensajes cortos (``min_tokens``);
2. eliminación de duplicados (usuario, texto);
3. tratamiento de emojis (``keep`` o ``remove``).

En el corpus los emojis ya vienen convertidos a texto en español (por
ejemplo ``cara sonriendo ligeramente``). Para reconocerlos se usa la lista de
nombres del paquete ``emoji``. Solo se usan nombres de **dos o más
palabras**: los de una palabra (``casa``, ``sol``, ``fuego``) borrarían
palabras normales del español.
"""

import re

import pandas as pd

WORD_PATTERN = re.compile(r"\w+")

# Índice de nombres de emoji: primera palabra -> nombres que empiezan por
# ella (como tuplas de palabras), de más largo a más corto.
EmojiIndex = dict[str, list[tuple[str, ...]]]


def load_emoji_names() -> list[str]:
    """Devuelve los nombres de emoji en español con dos o más palabras.

    Los obtiene de ``emoji.demojize(..., language="es")`` para cada emoji,
    en minúsculas y con espacios en vez de guiones bajos. Falla con un
    mensaje claro si el paquete ``emoji`` no está instalado.
    """
    try:
        import emoji
    except ImportError as error:
        raise ImportError(
            "Falta el paquete 'emoji': pip install -r "
            "vectorization/requirements.txt") from error
    names = set()
    for character in emoji.EMOJI_DATA:
        # Los nombres en español se cargan al pedirlos con demojize.
        name = emoji.demojize(character, language="es")
        if name == character:
            continue
        name = name.strip(":").replace("_", " ").lower()
        words = WORD_PATTERN.findall(name)
        if len(words) >= 2:
            names.add(" ".join(words))
    return sorted(names)


def build_emoji_index(names: list[str]) -> EmojiIndex:
    """Organiza los nombres de emoji por su primera palabra.

    Así, para cada palabra de un texto solo hay que comparar unos pocos
    candidatos. Los más largos van primero para que ``mano saludando tono
    de piel medio`` se reconozca entera y no solo ``mano saludando``.
    """
    index: EmojiIndex = {}
    for name in names:
        words = tuple(name.split())
        index.setdefault(words[0], []).append(words)
    for candidates in index.values():
        candidates.sort(key=len, reverse=True)
    return index


def strip_emoji_names(text: str, index: EmojiIndex) -> tuple[str, int]:
    """Quita del texto los nombres de emoji y cuenta cuántos había.

    Devuelve ``(texto_limpio, n_emojis)``. El resto del texto (puntuación
    incluida) se conserva y se colapsan los espacios sobrantes.
    """
    if not index:
        return text, 0
    matches = list(WORD_PATTERN.finditer(text))
    words = [match.group().lower() for match in matches]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(words):
        length = 0
        for candidate in index.get(words[i], []):
            end = i + len(candidate)
            if tuple(words[i:end]) == candidate:
                length = len(candidate)
                break
        if length:
            spans.append((matches[i].start(), matches[i + length - 1].end()))
            i += length
        else:
            i += 1
    if not spans:
        return text, 0
    pieces: list[str] = []
    position = 0
    for start, end in spans:
        pieces.append(text[position:start])
        position = end
    pieces.append(text[position:])
    return " ".join("".join(pieces).split()), len(spans)


def count_tokens(text: str) -> int:
    """Cuenta los tokens (secuencias de letras/dígitos) de un texto."""
    return len(WORD_PATTERN.findall(text))


def needs_emoji_names(count_emoji_tokens: bool, emojis: str) -> bool:
    """Indica si hace falta la lista de nombres de emoji.

    Hace falta para contar tokens sin emojis (comportamiento por defecto) y
    para quitarlos con ``--emojis remove``.
    """
    return (not count_emoji_tokens) or emojis == "remove"


def preprocess(
    df: pd.DataFrame,
    min_tokens: int,
    count_emoji_tokens: bool,
    keep_duplicates: bool,
    emojis: str,
    emoji_index: EmojiIndex,
) -> tuple[pd.DataFrame, dict]:
    """Aplica el preprocesado y devuelve ``(df, informe)``.

    El informe cuenta cuántos mensajes quitó cada paso para guardarlo en
    ``config.json``. Se eliminan los mensajes con menos de ``min_tokens``
    tokens (por defecto 4, es decir, los de 3 o menos). Los tokens se
    cuentan sin los nombres de emoji salvo que ``count_emoji_tokens``. Los
    usuarios que se quedan sin mensajes desaparecen.
    """
    report: dict = {
        "messages_in": len(df),
        "subjects_in": int(df["subject_key"].nunique()),
    }
    cleaned = []
    removed = []
    for text in df["message"]:
        clean_text, n_emojis = strip_emoji_names(text, emoji_index)
        cleaned.append(clean_text)
        removed.append(n_emojis)
    df = df.assign(_clean=cleaned, _emojis=removed)

    # Paso 1: mensajes cortos.
    source = df["message"] if count_emoji_tokens else df["_clean"]
    keep = source.map(count_tokens) >= min_tokens
    report["dropped_short"] = int((~keep).sum())
    df = df[keep]

    # Paso 2: duplicados exactos del mismo usuario.
    if not keep_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["subject_key", "message"])
        report["dropped_duplicates"] = before - len(df)
    else:
        report["dropped_duplicates"] = 0

    # Paso 3: emojis.
    report["emoji_names_found"] = int(df["_emojis"].sum())
    report["dropped_empty"] = 0
    if emojis == "remove":
        df = df.assign(message=df["_clean"])
        not_empty = df["message"].map(count_tokens) > 0
        report["dropped_empty"] = int((~not_empty).sum())
        df = df[not_empty]

    df = df.drop(columns=["_clean", "_emojis"]).reset_index(drop=True)
    report["messages_out"] = len(df)
    report["subjects_out"] = int(df["subject_key"].nunique())
    return df, report
