"""Text processing utilities: normalization, chunking, cleaning."""

from __future__ import annotations

import re
from typing import Iterator


RUSSIAN_ABBREVIATIONS: dict[str, str] = {
    "т.е.": "то есть",
    "т.к.": "так как",
    "и т.д.": "и так далее",
    "и т.п.": "и тому подобное",
    "т.н.": "так называемый",
    "др.": "другие",
    "пр.": "прочие",
    "им.": "имени",
    "г.": "год",
    "гг.": "годы",
    "вв.": "века",
    "руб.": "рублей",
    "коп.": "копеек",
    "млн.": "миллионов",
    "млрд.": "миллиардов",
    "тыс.": "тысяч",
    "см.": "смотри",
    "ср.": "средний",
    "напр.": "например",
}


def expand_abbreviations(text: str) -> str:
    """Expand common Russian abbreviations for better TTS pronunciation."""
    result = text
    for abbr, full in RUSSIAN_ABBREVIATIONS.items():
        result = result.replace(abbr, full)
    return result


def normalize_numbers(text: str, language: str = "ru") -> str:
    """Normalize numbers for TTS.

    Russian TTS Studio3 has limited Russian number normalization, so we
    pre-process common cases. For production use a dedicated tool
    like 'num2words' (pip install num2words[ru]).
    """
    try:
        from num2words import num2words

        def replace_number(match: re.Match) -> str:
            num_str = match.group(0)
            try:
                if "." in num_str or "," in num_str:
                    num_str_clean = num_str.replace(",", ".")
                    parts = num_str_clean.split(".")
                    int_part = int(parts[0])
                    dec_part = parts[1] if len(parts) > 1 else ""
                    int_words = num2words(int_part, lang="ru")
                    if dec_part:
                        dec_words = " ".join(
                            num2words(int(d), lang="ru") for d in dec_part
                        )
                        return f"{int_words} целых {dec_words}"
                    return int_words
                return num2words(int(num_str), lang="ru")
            except (ValueError, Exception):
                return num_str

        return re.sub(r"\b\d+[.,]?\d*\b", replace_number, text)
    except ImportError:
        return text


def split_into_sentences(text: str, language: str = "ru") -> list[str]:
    """Split text into sentences for safer chunk-by-chunk synthesis."""
    if language == "ru":
        sentence_end = re.compile(r"(?<=[.!?…])\s+")
    else:
        sentence_end = re.compile(r"(?<=[.!?])\s+")

    sentences = sentence_end.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_for_tts(
    text: str,
    max_chars: int = 200,
    max_sentences: int = 4,
) -> list[str]:
    """Split text into chunks safe for TTS inference.

    Russian TTS Studio3 tends to degrade on chunks >200 chars or >4 sentences.
    """
    text = expand_abbreviations(text.strip())
    sentences = split_into_sentences(text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)

        if sent_len > max_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            parts = re.split(r"(?<=[,;:])\s+", sentence)
            for part in parts:
                if len(part) > max_chars:
                    words = part.split()
                    sub_chunk: list[str] = []
                    sub_len = 0
                    for word in words:
                        if sub_len + len(word) + 1 > max_chars and sub_chunk:
                            chunks.append(" ".join(sub_chunk))
                            sub_chunk = [word]
                            sub_len = len(word)
                        else:
                            sub_chunk.append(word)
                            sub_len += len(word) + 1
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                else:
                    chunks.append(part)
            continue

        if (
            current_len + sent_len + 1 > max_chars
            or len(current) >= max_sentences
        ) and current:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sent_len
        else:
            current.append(sentence)
            current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def fix_yo_letter(text: str) -> str:
    """Ensure 'ё' is used where required (TTS models sometimes drop it).

    Preserves the original case of the first letter.
    """
    def _replace_e_to_yo(match: re.Match) -> str:
        word = match.group(0)
        return word[:-1] + "ё"

    result = text
    result = re.sub(r"\bВсе\b", _replace_e_to_yo, result)
    result = re.sub(r"\bвсе\b", _replace_e_to_yo, result)
    result = re.sub(r"\b(?:Еще|Ещё)\b", _replace_e_to_yo, result)
    result = re.sub(r"\b(?:еще|ещё)\b", _replace_e_to_yo, result)
    result = re.sub(r"\b(?:Елка|Ёлка)\b", _replace_e_to_yo, result)
    result = re.sub(r"\b(?:елка|ёлка)\b", _replace_e_to_yo, result)
    return result
