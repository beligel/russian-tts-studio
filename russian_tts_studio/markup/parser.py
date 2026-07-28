"""Parser for RTTS Markup ``{{...}}`` commands.

Syntax (LTV-inspired, case-insensitive command names):

    {{pause 700}}              # ms
    {{pause 700ms}}
    {{pause 0.7s}}
    {{pause.short}}            # 300 ms preset
    {{pause.medium}}           # 700 ms preset
    {{pause.long}}             # 1200 ms preset
    {{pause random 500 1200}}  # uniform sample per segment

    {{speed 0.9}}
    {{speed.slow}}             # 0.85
    {{speed.normal}}           # 1.0
    {{speed.fast}}             # 1.15

    {{volume -3db}}
    {{volume 0.8}}
    {{volume 80%}}
    {{volume.db -3}}
    {{volume.normalize -16}}
    {{volume.lufs -16}}
    {{volume.normal}}          # reset

    {{chapter "Урок 1"}}
    {{chapter 'Урок 1'}}

    {{alias "GPT" "gee pee tee"}}

    {{reset}}
    {{reset.voice}}
    {{reset.audio}}

    {{laugh}}                 # inline sound event
    {{cough}}
    {{sigh}}
    {{bgm start}}             # span sound event (start)
    ... text inside span ...
    {{bgm end}}               # span sound event (end)
    {{hum start}} ... {{hum end}}
    {{laugh soft}}            # inline event with modifier (engine-specific)

    {{stress "за́мок"}}        # word already containing stress mark
    {{stress "замок" "а"}}     # plain word + hint vowel (auto-detect)
    {{accent "за́мок"}}        # alias for {{stress ...}}

Unknown commands → :class:`Unknown` with a warning; generation continues.
Malformed commands (missing quote, bad number) → :class:`Unknown` too.

The parser is intentionally tolerant: a typo in markup should never
abort a 10 000-character audiobook render. All problems surface as
warnings collected on :class:`ParsedDocument`.
"""

from __future__ import annotations

import re
from typing import Optional

from .commands import (
    COMBINING_ACUTE,
    PAUSE_PRESETS,
    SOUND_EVENT_PRESETS,
    SPEED_PRESETS,
    Alias,
    Chapter,
    Command,
    Pause,
    Reset,
    SoundEvent,
    Speed,
    Stress,
    Unknown,
    Volume,
)


# Main extractor: finds every ``{{...}}`` block (non-greedy, single line).
# We don't allow newlines inside a command — LTV markup is one line per
# command. ``{{preset ...}}`` with multi-line JSON is a Phase-6 concern.
_MARKUP_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)

# Smart quotes / unicode dashes → ASCII, so ``{{speed “0.9”}}`` and
# ``{{volume –3db}}`` still parse. Applied inside each command body.
_SMART_QUOTE_MAP = {
    "\u2018": "'",  # left single
    "\u2019": "'",  # right single
    "\u201c": '"',  # left double
    "\u201d": '"',  # right double
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2212": "-",  # minus sign
}


def _normalize_smart_chars(s: str) -> str:
    return "".join(_SMART_QUOTE_MAP.get(c, c) for c in s)


def _strip_quotes(s: str) -> str:
    """Strip surrounding matched quotes from a value."""
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1]
    return s


def _parse_number(s: str) -> Optional[float]:
    """Parse a number, tolerating ``-3db``-style suffixes (caller strips)."""
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pause(body: str) -> Pause:
    """Parse the body after ``pause`` (case-insensitive, may start with ``.``)."""
    body = body.strip()
    # Subcommand presets: pause.short / pause.medium / pause.long
    if body.startswith("."):
        key = body[1:].strip().lower()
        if key in PAUSE_PRESETS:
            return Pause(ms=PAUSE_PRESETS[key])
        return Unknown(raw=body, warning=f"pause.{key} is not a preset")

    # ``random A B`` → uniform sample
    m = re.match(r"random\s+(\d+)\s+(\d+)", body, re.IGNORECASE)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            lo, hi = hi, lo
        return Pause(min_ms=lo, max_ms=hi, random=True)

    # ``700`` / ``700ms`` / ``0.7s``
    m = re.match(r"(\d+(?:\.\d+)?)\s*(ms|s)?$", body, re.IGNORECASE)
    if m:
        num = float(m.group(1))
        unit = (m.group(2) or "ms").lower()
        ms = int(num * 1000) if unit == "s" else int(num)
        return Pause(ms=ms)

    return Unknown(raw=body, warning=f"unparseable pause: {body!r}")


def _parse_speed(body: str) -> Command:
    body = body.strip()
    if body.startswith("."):
        key = body[1:].strip().lower()
        if key in SPEED_PRESETS:
            return Speed(value=SPEED_PRESETS[key])
        return Unknown(raw=body, warning=f"speed.{key} is not a preset")
    num = _parse_number(body)
    if num is not None:
        return Speed(value=num)
    return Unknown(raw=body, warning=f"unparseable speed: {body!r}")


def _parse_volume(body: str) -> Command:
    body = body.strip().lower()

    # Subcommand resets / presets
    if body in ("normal", ".normal", "reset", ".reset"):
        return Volume()  # all-None → reset
    if body.startswith("."):
        body = body[1:].strip()

    # Strip surrounding quotes (smart-quote-normalised by caller) so
    # ``{{volume "-3db"}}`` parses the same as ``{{volume -3db}}``.
    body = _strip_quotes(body)

    # ``volume.normalize -16`` / ``volume.lufs -16``
    m = re.match(r"(?:normalize|lufs)\s+(-?\d+(?:\.\d+)?)", body)
    if m:
        return Volume(normalize_lufs=float(m.group(1)))

    # ``volume.db -3``
    m = re.match(r"db\s+(-?\d+(?:\.\d+)?)", body)
    if m:
        return Volume(gain_db=float(m.group(1)))

    # ``-3db`` (direct gain with suffix)
    m = re.match(r"(-?\d+(?:\.\d+)?)\s*db$", body)
    if m:
        return Volume(gain_db=float(m.group(1)))

    # ``80%``
    m = re.match(r"(\d+(?:\.\d+)?)\s*%$", body)
    if m:
        return Volume(multiplier=float(m.group(1)) / 100.0)

    # ``0.8`` (bare multiplier)
    num = _parse_number(body)
    if num is not None:
        return Volume(multiplier=num)

    return Unknown(raw=body, warning=f"unparseable volume: {body!r}")


def _parse_quoted_args(body: str, expected: int) -> Optional[list[str]]:
    """Extract ``expected`` quoted strings from ``body``.

    Supports both ``"..."`` and ``'...'``. Returns ``None`` on mismatch.
    """
    args: list[str] = []
    pattern = r'["\']([^"\']*)["\']'
    for m in re.finditer(pattern, body):
        args.append(m.group(1))
        if len(args) == expected:
            return args
    return None if len(args) < expected else args


def _parse_chapter(body: str) -> Command:
    args = _parse_quoted_args(body, 1)
    if args is None:
        # Tolerate unquoted single-word chapter titles.
        title = body.strip()
        if title:
            return Chapter(title=title)
        return Unknown(raw=body, warning="chapter: missing title")
    return Chapter(title=args[0])


def _parse_alias(body: str) -> Command:
    args = _parse_quoted_args(body, 2)
    if args is None:
        return Unknown(raw=body, warning="alias: needs two quoted args")
    target, replacement = args[0], args[1]
    if not target:
        return Unknown(raw=body, warning="alias: empty target")
    return Alias(target=target, replacement=replacement)


def _parse_reset(body: str) -> Command:
    body = body.strip().lower()
    if not body:
        return Reset(scope="all")
    if body.startswith("."):
        body = body[1:]
    if body in ("all", "voice", "audio"):
        return Reset(scope=body)
    return Unknown(raw=body, warning=f"reset: unknown scope {body!r}")


def _parse_sound_event(body: str, event_name: str) -> Command:
    """Parse the body after a sound-event command name.

    For inline events: ``{{laugh}}``, ``{{laugh soft}}``, ``{{laugh x2}}``.
    For span events:   ``{{bgm start}}``, ``{{bgm end}}``, ``{{bgm}}`` (defaults to start).

    The modifier (anything after the event name / start-end keyword)
    is captured in ``SoundEvent.modifier`` for engines that understand
    intensity/repetition hints. Engines that don't, ignore it.
    """
    body = body.strip()
    preset = SOUND_EVENT_PRESETS.get(event_name)
    if preset is None:
        # Shouldn't happen — the dispatch table only routes known
        # event names here — but handle gracefully.
        return Unknown(raw=event_name, warning=f"unknown sound event: {event_name!r}")

    category = preset["category"]
    token = preset["token"]
    label = preset["label"]

    if category == "inline":
        # Inline events take an optional free-form modifier.
        modifier = body
        return SoundEvent(
            name=event_name, token=token, category="inline",
            label=label, span_marker=None, modifier=modifier,
        )

    # Span event: expect optional "start" / "end" keyword + optional modifier.
    span_marker = "start"  # default
    modifier = ""
    if body:
        lower = body.lower()
        # Match leading start/end keyword (possibly after stripping a dot).
        first = lower.split(None, 1)[0] if lower.split(None, 1) else ""
        first = first.lstrip(".")
        if first in ("start", "begin", "on"):
            span_marker = "start"
            rest = lower[len(first):].strip()
            modifier = rest
        elif first in ("end", "stop", "off"):
            span_marker = "end"
            rest = lower[len(first):].strip()
            modifier = rest
        else:
            # No start/end keyword — treat the whole body as a modifier
            # and default to "start" (so ``{{bgm}}`` opens a span).
            modifier = body
    return SoundEvent(
        name=event_name, token=token, category="span",
        label=label, span_marker=span_marker, modifier=modifier,
    )


# Vowel lookup table for stress auto-detection. Lowercase only;
# the matcher lowercases the word before searching.
_VOWEL_SET = set("аеёиоуыэюяaeiouy")


def _apply_combining_acute(word: str, hint_vowel: str) -> tuple[str, str]:
    """Insert U+0301 after the first occurrence of ``hint_vowel`` in ``word``.

    Returns ``(stressed_word, warning)``. ``warning`` is "" on success
    or a human-readable note when the hint vowel wasn't found and we
    fell back to the first vowel in the word.

    Matching is case-insensitive. ``ё``/``е`` are treated as distinct
    (Russian ``ё`` is already stressed by convention — we still honour
    an explicit ``е`` hint without conflating it with ``ё``).
    """
    if not word:
        return word, "stress: empty word"
    hint = hint_vowel.lower()
    if hint and hint not in _VOWEL_SET:
        return word, f"stress: {hint_vowel!r} is not a vowel"
    lowered = word.lower()
    # Find the hint vowel position.
    idx = -1
    if hint:
        idx = lowered.find(hint)
    if idx < 0:
        # Fallback: first vowel in the word.
        for i, ch in enumerate(lowered):
            if ch in _VOWEL_SET:
                idx = i
                break
        if idx < 0:
            return word, f"stress: no vowel in {word!r}"
        warning = (
            f"stress: vowel {hint_vowel!r} not found in {word!r}, "
            f"used first vowel {lowered[idx]!r}"
        )
    else:
        warning = ""
    # Insert the combining acute AFTER the matched vowel.
    return word[: idx + 1] + COMBINING_ACUTE + word[idx + 1:], warning


def _parse_stress(body: str) -> Command:
    """Parse ``{{stress "за́мок"}}`` or ``{{stress "замок" "а"}}``.

    Two forms:
    1. One quoted arg — the word already containing a stress mark.
       We use it verbatim as the ``stressed`` form and strip any
       precomposed acute (U+0301 stays; precomposed forms like ``́``
       are normalised to base + combining for consistency).
    2. Two quoted args — plain word + hint vowel. Auto-detect the
       vowel position and insert U+0301.
    """
    args = _parse_quoted_args(body, 2)
    if args is None:
        # Try single-arg form.
        args = _parse_quoted_args(body, 1)
        if args is None:
            return Unknown(raw=body, warning="stress: needs one or two quoted args")
        word = args[0].strip()
        if not word:
            return Unknown(raw=body, warning="stress: empty word")
        # Single-arg form: the user supplied the stressed form directly.
        # Normalise: if the word has no combining acute and no precomposed
        # acute, leave as-is (engine decides). We don't invent stress.
        return Stress(target=_strip_stress(word), stressed=word, hint_vowel="")

    word, hint = args[0].strip(), args[1].strip()
    if not word:
        return Unknown(raw=body, warning="stress: empty word")
    if not hint:
        # Empty hint → same as single-arg form.
        return Stress(target=word, stressed=word, hint_vowel="")
    stressed, warning = _apply_combining_acute(word, hint)
    return Stress(target=word, stressed=stressed, hint_vowel=hint, warning=warning)


def _strip_stress(word: str) -> str:
    """Remove combining acute (U+0301) from a word — used to derive
    the plain ``target`` form when the user supplies only the stressed
    form in the single-arg ``{{stress "за́мок"}}`` variant."""
    return word.replace(COMBINING_ACUTE, "")


# Dispatch table: command name (lowercase, no dot) → parser function.
# Subcommands (``pause.short``, ``reset.voice``) are handled inside each
# parser by inspecting the body for a leading ``.``.
#
# Sound events are registered dynamically below from SOUND_EVENT_PRESETS
# so adding a new event only needs an entry there — no parser changes.
_PARSERS: dict[str, callable] = {
    "pause": _parse_pause,
    "speed": _parse_speed,
    "volume": _parse_volume,
    "chapter": _parse_chapter,
    "alias": _parse_alias,
    "reset": _parse_reset,
    "stress": _parse_stress,
    "accent": _parse_stress,  # alias for {{stress ...}}
}

# Register every named sound event as a top-level command. We use a
# closure to bind the event name (so the parser knows which preset
# to look up). ``{{laugh}}``, ``{{cough}}``, ``{{bgm start}}`` etc.
for _ev_name in SOUND_EVENT_PRESETS:
    # Capture the name by default arg to avoid late-binding gotcha.
    def _make_parser(name=_ev_name):
        def _parse(body: str) -> Command:
            return _parse_sound_event(body, name)
        return _parse
    _PARSERS[_ev_name] = _make_parser()


def parse_command(raw_body: str) -> Command:
    """Parse a single ``{{...}}`` body (without braces) into a Command.

    Returns :class:`Unknown` for anything unrecognised or malformed —
    never raises.
    """
    body = _normalize_smart_chars(raw_body).strip()
    if not body:
        return Unknown(raw=raw_body, warning="empty command")

    # First token (up to whitespace or dot) is the command name.
    m = re.match(r"([a-zA-Z]+)(?=[\s.]|$)", body)
    if not m:
        return Unknown(raw=raw_body, warning=f"unrecognised command: {body!r}")
    name = m.group(1).lower()
    rest = body[m.end():]

    parser = _PARSERS.get(name)
    if parser is None:
        return Unknown(raw=raw_body, warning=f"unknown command: {name!r}")
    return parser(rest)


def find_markup(text: str) -> list[tuple[int, int, str]]:
    """Return ``[(start, end, body), ...]`` for every ``{{...}}`` in ``text``.

    ``end`` is the index just past the closing ``}}``. ``body`` is the
    inner text (already smart-char-normalised). Used by the document
    builder to split text and commands.
    """
    out: list[tuple[int, int, str]] = []
    for m in _MARKUP_RE.finditer(text):
        body = _normalize_smart_chars(m.group(1))
        out.append((m.start(), m.end(), body))
    return out