"""Text normalization for TTS: dictionaries + auto-rules.

Inspired by LTV's normalization system. Prepares a pronunciation-friendly
copy of the source text before TTS without modifying the original.

Two layers:
1. **Dictionary replacements** — user-editable SQLite-backed word/phrase
   lists. Sorted longest-match-first so "и т.д." beats "т.д.".
2. **Auto-rules** — pattern-based transformations for numbers, ordinals,
   dates, currencies, percentages, measurements, Roman numerals. Each
   rule has a global switch and individual toggles.

The original source text is NEVER modified. ``normalize_for_tts`` returns
a *copy* ready for the TTS engine; the pipeline keeps the original for
display, editing, and Whisper comparison.

Usage::

    from russian_tts_studio.text.normalization import (
        NormalizationConfig, normalize_for_tts, get_dictionary,
    )

    cfg = NormalizationConfig()
    normalized = normalize_for_tts("Купи 200 г муки и т.д.", cfg)
    # → "Купи двести граммов муки и так далее"
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Default dictionaries (built-in starter data)
# ---------------------------------------------------------------------------

# Abbreviation expansion — the core use case the user described:
# "и т.д." → "и так далее", "т.е." → "то есть", etc.
DEFAULT_ABBREVIATIONS: dict[str, str] = {
    "и т.д.": "и так далее",
    "и т.п.": "и тому подобное",
    "т.е.": "то есть",
    "т.к.": "так как",
    "т.н.": "так называемый",
    "т.д.": "так далее",
    "т.п.": "и тому подобное",
    "напр.": "например",
    "ср.": "средний",
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
    "стр.": "страница",
    "рис.": "рисунок",
    "табл.": "таблица",
    "ул.": "улица",
    "д.": "дом",
    "корп.": "корпус",
    "кв.": "квартира",
    "г.": "город",
    "обл.": "область",
    "р-н.": "район",
    "п.": "посёлок",
    "с.": "село",
    "пгт.": "посёлок городского типа",
    "с/с": "сельсовет",
    "РФ": "Российская Федерация",
    "СССР": "Союз Советских Социалистических Республик",
    "США": "Соединённые Штаты Америки",
    "ЕС": "Европейский Союз",
    "ООН": "Организация Объединённых Наций",
    "НATO": "Северо-Атлантический альянс",
    "ФСБ": "Федеральная служба безопасности",
    "МВД": "Министерство внутренних дел",
    "МЧС": "Министерство по делам гражданской обороны",
    "ЦБ": "Центральный банк",
    "НДС": "налог на добавленную стоимость",
    "НДФЛ": "налог на доходы физических лиц",
    "ДТП": "дорожно-транспортное происшествие",
    "СМС": "короткое текстовое сообщение",
    "ГСМ": "горюче-смазочные материалы",
    "ЖКХ": "жилищно-коммунальное хозяйство",
    "ДНР": "Донецкая Народная Республика",
    "ЛНР": "Луганская Народная Республика",
    "КРЫМ": "Крым",
}

# Address dictionary — expands address abbreviations.
DEFAULT_ADDRESSES: dict[str, str] = {
    "ул.": "улица",
    "ул ": "улица ",
    "д.": "дом",
    "д ": "дом ",
    "корп.": "корпус",
    "корп ": "корпус ",
    "кв.": "квартира",
    "кв ": "квартира ",
    "г.": "город",
    "г ": "город ",
    "обл.": "область",
    "обл ": "область ",
    "р-н.": "район",
    "р-н ": "район ",
    "п.": "посёлок",
    "п ": "посёлок ",
    "с.": "село",
    "с ": "село ",
    "пгт.": "посёлок городского типа",
    "с/с": "сельсовет",
    "просп.": "проспект",
    "просп ": "проспект ",
    "ш.": "шоссе",
    "ш ": "шоссе ",
    "пер.": "переулок",
    "пер ": "переулок ",
    "наб.": "набережная",
    "наб ": "набережная ",
    "пл.": "площадь",
    "пл ": "площадь ",
    "б-р.": "бульвар",
    "б-р ": "бульвар ",
}


# ---------------------------------------------------------------------------
# SQLite schema for user dictionaries
# ---------------------------------------------------------------------------

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS dictionaries (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
);

CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dict_id TEXT NOT NULL REFERENCES dictionaries(id) ON DELETE CASCADE,
    source TEXT NOT NULL,
    replacement TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    sort_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE(dict_id, source)
);

CREATE INDEX IF NOT EXISTS idx_entries_dict ON entries(dict_id);
"""


@dataclass
class NormalizationConfig:
    """Configuration for text normalization.

    Controls which auto-rules are active and which dictionaries to
    apply. Each auto-rule can be individually toggled.
    """

    # Master switch — when False, normalization is a no-op.
    enabled: bool = True

    # Dictionary switches
    use_abbreviations: bool = True
    use_addresses: bool = True
    use_custom_dicts: bool = True

    # Auto-rule switches
    normalize_numbers: bool = True
    normalize_ordinals: bool = True
    normalize_dates: bool = True
    normalize_currencies: bool = True
    normalize_percentages: bool = True
    normalize_measurements: bool = True
    normalize_roman: bool = True

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "use_abbreviations": self.use_abbreviations,
            "use_addresses": self.use_addresses,
            "use_custom_dicts": self.use_custom_dicts,
            "normalize_numbers": self.normalize_numbers,
            "normalize_ordinals": self.normalize_ordinals,
            "normalize_dates": self.normalize_dates,
            "normalize_currencies": self.normalize_currencies,
            "normalize_percentages": self.normalize_percentages,
            "normalize_measurements": self.normalize_measurements,
            "normalize_roman": self.normalize_roman,
        }


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_path(output_dir: str | Path = "output") -> Path:
    return Path(output_dir) / "normalization.db"


def get_db(output_dir: str | Path = "output") -> sqlite3.Connection:
    db_path = get_db_path(output_dir)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    thread_id = threading.current_thread().ident
    key = (str(db_path), thread_id)
    conn = _thread_conns.get(key)
    if conn is not None:
        return conn
    with _DB_LOCK:
        conn = sqlite3.connect(str(db_path), timeout=10, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        _thread_conns[key] = conn
        return conn


_thread_conns: dict[tuple[str, int], sqlite3.Connection] = {}


def _ensure_builtin_dicts(db: sqlite3.Connection) -> None:
    """Create built-in dictionaries if they don't exist."""
    for dict_id, name, desc, entries in [
        ("abbreviations", "Аббревиатуры", "Расширение русских аббревиатур для TTS", DEFAULT_ABBREVIATIONS),
        ("addresses", "Адресные сокращения", "Расширение адресных сокращений", DEFAULT_ADDRESSES),
    ]:
        exists = db.execute(
            "SELECT 1 FROM dictionaries WHERE id = ?", (dict_id,)
        ).fetchone()
        if not exists:
            db.execute(
                "INSERT INTO dictionaries (id, name, description) VALUES (?, ?, ?)",
                (dict_id, name, desc),
            )
            for i, (src, repl) in enumerate(entries.items()):
                db.execute(
                    "INSERT INTO entries (dict_id, source, replacement, sort_order) "
                    "VALUES (?, ?, ?, ?)",
                    (dict_id, src, repl, i),
                )
            db.commit()
            logger.info("Created built-in dictionary: %s (%d entries)", dict_id, len(entries))


# ---------------------------------------------------------------------------
# Dictionary CRUD
# ---------------------------------------------------------------------------

def list_dictionaries(output_dir: str | Path = "output") -> list[dict]:
    db = get_db(output_dir)
    _ensure_builtin_dicts(db)
    rows = db.execute("SELECT * FROM dictionaries ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_dictionary(dict_id: str, output_dir: str | Path = "output") -> Optional[dict]:
    db = get_db(output_dir)
    _ensure_builtin_dicts(db)
    row = db.execute("SELECT * FROM dictionaries WHERE id = ?", (dict_id,)).fetchone()
    if row is None:
        return None
    entries = db.execute(
        "SELECT * FROM entries WHERE dict_id = ? ORDER BY sort_order",
        (dict_id,),
    ).fetchall()
    return {
        **dict(row),
        "entries": [dict(e) for e in entries],
    }


def create_dictionary(
    dict_id: str,
    name: str,
    description: str = "",
    output_dir: str | Path = "output",
) -> dict:
    db = get_db(output_dir)
    db.execute(
        "INSERT INTO dictionaries (id, name, description) VALUES (?, ?, ?)",
        (dict_id, name, description),
    )
    db.commit()
    return {"id": dict_id, "name": name, "description": description, "enabled": 1}


def delete_dictionary(dict_id: str, output_dir: str | Path = "output") -> bool:
    db = get_db(output_dir)
    cur = db.execute("DELETE FROM dictionaries WHERE id = ?", (dict_id,))
    db.commit()
    return cur.rowcount > 0


def add_entry(
    dict_id: str,
    source: str,
    replacement: str,
    output_dir: str | Path = "output",
) -> int:
    db = get_db(output_dir)
    cur = db.execute(
        "INSERT INTO entries (dict_id, source, replacement) VALUES (?, ?, ?)",
        (dict_id, source, replacement),
    )
    db.commit()
    return cur.lastrowid


def update_entry(
    entry_id: int,
    source: str | None = None,
    replacement: str | None = None,
    enabled: bool | None = None,
    output_dir: str | Path = "output",
) -> None:
    db = get_db(output_dir)
    sets, vals = [], []
    if source is not None:
        sets.append("source = ?")
        vals.append(source)
    if replacement is not None:
        sets.append("replacement = ?")
        vals.append(replacement)
    if enabled is not None:
        sets.append("enabled = ?")
        vals.append(1 if enabled else 0)
    if not sets:
        return
    vals.append(entry_id)
    db.execute(f"UPDATE entries SET {', '.join(sets)} WHERE id = ?", vals)
    db.commit()


def delete_entry(entry_id: int, output_dir: str | Path = "output") -> bool:
    db = get_db(output_dir)
    cur = db.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
    db.commit()
    return cur.rowcount > 0


def export_dictionary_json(dict_id: str, output_dir: str | Path = "output") -> str:
    """Export a dictionary as JSON string."""
    d = get_dictionary(dict_id, output_dir)
    if d is None:
        return "{}"
    return json.dumps(d, ensure_ascii=False, indent=2)


def import_dictionary_json(
    dict_id: str,
    json_str: str,
    output_dir: str | Path = "output",
) -> int:
    """Import entries from JSON. Returns number of entries imported."""
    data = json.loads(json_str)
    entries = data.get("entries", [])
    db = get_db(output_dir)
    count = 0
    for e in entries:
        try:
            db.execute(
                "INSERT OR REPLACE INTO entries (dict_id, source, replacement, enabled) "
                "VALUES (?, ?, ?, ?)",
                (dict_id, e["source"], e["replacement"], e.get("enabled", 1)),
            )
            count += 1
        except Exception as exc:
            logger.warning("Import entry failed: %s", exc)
    db.commit()
    return count


# ---------------------------------------------------------------------------
# Normalization engine
# ---------------------------------------------------------------------------

# Roman numeral regex (I, II, III, IV, V, VI, VII, VIII, IX, X, ...)
_ROMAN_RE = re.compile(
    r"\b(M{0,3})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b",
    re.IGNORECASE,
)

# Ordinal patterns: "1-й", "2-го", "3-е", "1-ая"
_ORDINAL_RE = re.compile(
    r"(\d+)-(й|го|го|е|ая|ого|ому|ой|ый|ий|ых|ым)\b"
)

# Date patterns: "5 мая 1990 года", "01.01.2025"
_DATE_RE = re.compile(
    r"(\d{1,2})[.\s](\d{1,2})[.\s](\d{4})\s*(года|год|г\.)?"
)

# Currency: "100 руб.", "50$", "10 евро"
_CURRENCY_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(руб\.|₽|\$|долл\.|евро|€|yen|¥|фунт|£)"
)

# Percentage: "50%", "0.5%"
_PERCENT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")

# Measurement: "200 г", "5 кг", "3 м", "10 см", "2 л", "100 мл"
_MEASUREMENT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(г|кг|м|мм|см|дм|км|л|мл|кл|тонн?|шт\.?)\b"
)


def _roman_to_int(s: str) -> Optional[int]:
    """Convert a Roman numeral string to int."""
    vals = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
    s = s.upper()
    if not s or not all(c in vals for c in s):
        return None
    result = 0
    for i in range(len(s)):
        if i + 1 < len(s) and vals[s[i]] < vals[s[i + 1]]:
            result -= vals[s[i]]
        else:
            result += vals[s[i]]
    return result


def _apply_auto_rules(text: str, config: NormalizationConfig) -> str:
    """Apply pattern-based auto-rules to the text."""
    # Numbers → words
    if config.normalize_numbers:
        try:
            from num2words import num2words

            def _num_replace(m: re.Match) -> str:
                num_str = m.group(0).replace(",", ".")
                try:
                    if "." in num_str:
                        parts = num_str.split(".")
                        int_part = int(parts[0])
                        dec_part = parts[1]
                        int_words = num2words(int_part, lang="ru")
                        dec_words = " ".join(num2words(int(d), lang="ru") for d in dec_part)
                        return f"{int_words} целых {dec_words}"
                    return num2words(int(num_str), lang="ru")
                except (ValueError, Exception):
                    return num_str

            text = re.sub(r"\b\d+[.,]?\d*\b", _num_replace, text)
        except ImportError:
            pass

    # Ordinals: "1-й" → "первый"
    if config.normalize_ordinals:
        try:
            from num2words import num2words

            _ORD_MAP = {
                "й": ("m", "ordinal"),
                "го": ("m", "ordinal"),
                "е": ("n", "ordinal"),
                "ая": ("f", "ordinal"),
                "ого": ("m", "ordinal"),
                "ому": ("m", "ordinal"),
                "ой": ("m", "ordinal"),
                "ый": ("m", "ordinal"),
                "ий": ("m", "ordinal"),
                "ых": ("m", "ordinal"),
                "ым": ("m", "ordinal"),
            }

            def _ord_replace(m: re.Match) -> str:
                num = int(m.group(1))
                suffix = m.group(2)
                try:
                    return num2words(num, lang="ru", to="ordinal")
                except Exception:
                    return m.group(0)

            text = _ORDINAL_RE.sub(_ord_replace, text)
        except ImportError:
            pass

    # Currency
    if config.normalize_currencies:
        _CUR_MAP = {
            "руб.": "рублей", "₽": "рублей",
            "$": "долларов", "долл.": "долларов",
            "евро": "евро", "€": "евро",
            "фунт": "фунтов", "£": "фунтов",
            "yen": "иен", "¥": "иен",
        }

        def _cur_replace(m: re.Match) -> str:
            num_str = m.group(1).replace(",", ".")
            currency = m.group(2)
            try:
                from num2words import num2words

                num = int(float(num_str))
                word = num2words(num, lang="ru")
            except Exception:
                word = num_str
            cur_word = _CUR_MAP.get(currency, currency)
            return f"{word} {cur_word}"

        text = _CURRENCY_RE.sub(_cur_replace, text)

    # Percentages
    if config.normalize_percentages:
        def _pct_replace(m: re.Match) -> str:
            num_str = m.group(1).replace(",", ".")
            try:
                from num2words import num2words

                num = int(float(num_str))
                word = num2words(num, lang="ru")
            except Exception:
                word = num_str
            return f"{word} процентов"

        text = _PERCENT_RE.sub(_pct_replace, text)

    # Measurements
    if config.normalize_measurements:
        _MEAS_MAP = {
            "г": "граммов", "кг": "килограммов",
            "м": "метров", "мм": "миллиметров",
            "см": "сантиметров", "дм": "дециметров",
            "км": "километров",
            "л": "литров", "мл": "миллилитров", "кл": "килолитров",
            "тонн": "тонн", "тонна": "тонн",
            "шт.": "штук", "шт": "штук",
        }

        def _meas_replace(m: re.Match) -> str:
            num_str = m.group(1).replace(",", ".")
            unit = m.group(2)
            try:
                from num2words import num2words

                num = int(float(num_str))
                word = num2words(num, lang="ru")
            except Exception:
                word = num_str
            unit_word = _MEAS_MAP.get(unit, unit)
            return f"{word} {unit_word}"

        text = _MEASUREMENT_RE.sub(_meas_replace, text)

    # Roman numerals
    if config.normalize_roman:
        try:
            from num2words import num2words

            def _roman_replace(m: re.Match) -> str:
                val = _roman_to_int(m.group(0))
                if val is not None and 1 <= val <= 3999:
                    try:
                        return num2words(val, lang="ru")
                    except Exception:
                        return m.group(0)
                return m.group(0)

            text = _ROMAN_RE.sub(_roman_replace, text)
        except ImportError:
            pass

    return text


def _apply_dictionaries(
    text: str,
    config: NormalizationConfig,
    output_dir: str | Path = "output",
) -> str:
    """Apply dictionary replacements, longest-match-first."""
    db = get_db(output_dir)
    _ensure_builtin_dicts(db)

    # Collect all enabled entries from all enabled dictionaries.
    entries: list[tuple[str, str]] = []

    if config.use_abbreviations:
        rows = db.execute(
            "SELECT e.source, e.replacement FROM entries e "
            "JOIN dictionaries d ON e.dict_id = d.id "
            "WHERE e.enabled = 1 AND d.enabled = 1 AND d.id = 'abbreviations'"
        ).fetchall()
        entries.extend((r["source"], r["replacement"]) for r in rows)

    if config.use_addresses:
        rows = db.execute(
            "SELECT e.source, e.replacement FROM entries e "
            "JOIN dictionaries d ON e.dict_id = d.id "
            "WHERE e.enabled = 1 AND d.enabled = 1 AND d.id = 'addresses'"
        ).fetchall()
        entries.extend((r["source"], r["replacement"]) for r in rows)

    if config.use_custom_dicts:
        rows = db.execute(
            "SELECT e.source, e.replacement FROM entries e "
            "JOIN dictionaries d ON e.dict_id = d.id "
            "WHERE e.enabled = 1 AND d.enabled = 1 "
            "AND d.id NOT IN ('abbreviations', 'addresses')"
        ).fetchall()
        entries.extend((r["source"], r["replacement"]) for r in rows)

    # Sort longest source first so "и т.д." beats "т.д."
    entries.sort(key=lambda x: len(x[0]), reverse=True)

    for source, replacement in entries:
        text = text.replace(source, replacement)

    return text


def normalize_for_tts(
    text: str,
    config: NormalizationConfig | None = None,
    output_dir: str | Path = "output",
) -> str:
    """Normalize text for TTS pronunciation.

    Applies dictionaries (longest-match-first) then auto-rules.
    Returns a *copy* — the original text is never modified.
    """
    cfg = config or NormalizationConfig()
    if not cfg.enabled:
        return text

    result = text
    result = _apply_dictionaries(result, cfg, output_dir)
    result = _apply_auto_rules(result, cfg)
    return result


def get_normalization_diff(
    text: str,
    config: NormalizationConfig | None = None,
    output_dir: str | Path = "output",
) -> dict:
    """Show a before/after diff of normalization.

    Returns ``{original, normalized, changes: [{original, normalized}]}``
    where each change is a sentence-level diff.
    """
    from ..utils.text_utils import split_into_sentences

    cfg = config or NormalizationConfig()
    original_sentences = split_into_sentences(text)
    normalized = normalize_for_tts(text, cfg, output_dir)
    normalized_sentences = split_into_sentences(normalized)

    changes: list[dict] = []
    for i in range(max(len(original_sentences), len(normalized_sentences))):
        orig = original_sentences[i] if i < len(original_sentences) else ""
        norm = normalized_sentences[i] if i < len(normalized_sentences) else ""
        if orig != norm:
            changes.append({"original": orig, "normalized": norm})

    return {
        "original": text,
        "normalized": normalized,
        "changes": changes,
        "total_changes": len(changes),
    }