"""Markup commands dataclasses for RTTS Markup (LTV-inspired).

Each command modifies narration state for the following text segment.
Commands are extracted by the parser from ``{{...}}`` blocks and turned
into the dataclasses below. Unknown commands become :class:`Unknown`
with a warning string — they never stop generation.

Design notes:
- Voice/language commands are intentionally absent. RTTS uses a single
  engine (VoxCPM2) with voice cloning by reference audio file, not by
  voice name. ``{{voice "..."}}`` may be added later as an alias to
  saved references from ``/api/references``.
- ``{{cmd}}`` / ``{{preset}}`` (engine-specific TTS params) are absent
  because VoxCPM2 SDK 2.0.3 dropped ``instruct`` (see
  ``voxcpm_synth.py:236-241``). Re-add when the SDK restores the param.
- ``{{play}}`` / ``{{stop}}`` belong to the future Audio Mix phase and
  are not part of Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


class MarkupCommandKind(str, Enum):
    """Discriminator for the command type."""

    PAUSE = "pause"
    SPEED = "speed"
    VOLUME = "volume"
    CHAPTER = "chapter"
    ALIAS = "alias"
    RESET = "reset"
    SOUND_EVENT = "sound_event"
    STRESS = "stress"
    UNKNOWN = "unknown"


# Preset pause durations in ms (mirror LTV defaults).
PAUSE_PRESETS: dict[str, int] = {
    "short": 300,
    "medium": 700,
    "long": 1200,
}

# Preset speed multipliers.
SPEED_PRESETS: dict[str, float] = {
    "slow": 0.85,
    "normal": 1.0,
    "fast": 1.15,
}

# Named sound events. Mirrors Higgs Audio's inline tag scheme
# ([laugh] → <SE>[Laughter]</SE>) but adapted to RTTS markup. The
# ``token`` field is what a capable engine receives in-text; for
# engines that don't support sound events the pipeline strips the
# token before synthesis (the engine just reads the surrounding
# sentence without the non-speech event).
#
# Two categories, matching Higgs Audio's distinction:
#   - ``inline``: replaces a moment in the text (laugh, cough, sigh).
#     Emitted as a single inline token at the position of the command.
#   - ``span``: wraps a following text region (background music,
#     humming, singing). Uses ``start`` / ``end`` markers.
SOUND_EVENT_PRESETS: dict[str, dict] = {
    "laugh":    {"token": "[laugh]",    "category": "inline", "label": "Laughter"},
    "chuckle":  {"token": "[chuckle]", "category": "inline", "label": "Laughter"},
    "giggle":   {"token": "[giggle]",  "category": "inline", "label": "Laughter"},
    "cough":    {"token": "[cough]",   "category": "inline", "label": "Cough"},
    "sigh":     {"token": "[sigh]",    "category": "inline", "label": "Sigh"},
    "gasp":     {"token": "[gasp]",    "category": "inline", "label": "Gasp"},
    "cry":      {"token": "[cry]",     "category": "inline", "label": "Crying"},
    "sniffle":  {"token": "[sniffle]", "category": "inline", "label": "Sniffle"},
    "sneeze":   {"token": "[sneeze]",  "category": "inline", "label": "Sneeze"},
    "yawn":     {"token": "[yawn]",    "category": "inline", "label": "Yawn"},
    "applause": {"token": "[applause]","category": "inline", "label": "Applause"},
    "cheer":    {"token": "[cheer]",   "category": "inline", "label": "Cheering"},
    # Span events wrap a region. {{bgm start}} ... {{bgm end}}.
    "bgm":      {"token": "[music]",    "category": "span",  "label": "Music"},
    "music":    {"token": "[music]",    "category": "span",  "label": "Music"},
    "hum":      {"token": "[humming]",  "category": "span",  "label": "Humming"},
    "humming":  {"token": "[humming]",  "category": "span",  "label": "Humming"},
    "sing":     {"token": "[singing]",  "category": "span",  "label": "Singing"},
}

# Combining acute accent (U+0301) — the Unicode-standard way to mark
# word stress in Cyrillic text. Most Russian-capable TTS backends
# (eSpeak/Silero, Higgs Audio, PHONEMIZER-based systems) treat this
# diacritic as a stress cue. VoxCPM2 ignores it — its TSLM uses
# autoprosoody — so the pipeline strips it before calling VoxCPM2
# but leaves it for engines that honour it.
COMBINING_ACUTE = "\u0301"

# Russian vowels (lowercase + uppercase). Used by the stress
# auto-detection path to find where to place the combining accent.
_RU_VOWELS = "аеёиоуыэюяaeiouy" + "АЕЁИОУЫЭЮЯAEIOUY"


@dataclass
class Pause:
    """``{{pause 700ms}}`` / ``{{pause.short}}`` / ``{{pause random 500 1200}}``.

    ``random`` mode: when ``min_ms`` and ``max_ms`` are both set, the
    parser records the range; the pipeline samples a concrete value at
    synthesis time (per segment) so two runs of the same text differ.
    """

    ms: int = 0
    min_ms: int = 0
    max_ms: int = 0
    random: bool = False

    def is_enabled(self) -> bool:
        if self.random:
            return self.max_ms > self.min_ms
        return self.ms > 0

    def sample_ms(self) -> int:
        """Resolve to a concrete pause duration.

        Non-random pauses return ``ms`` unchanged. Random pauses sample
        uniformly in ``[min_ms, max_ms]``. The pipeline calls this once
        per segment so the same markup yields different timings across
        runs — matching LTV's "naturalistic" pause behaviour.
        """
        if not self.random:
            return self.ms
        import random as _r

        return _r.randint(self.min_ms, self.max_ms)


@dataclass
class Speed:
    """``{{speed 0.9}}`` / ``{{speed.slow}}``.

    Linear multiplier. ``1.0`` = normal. Stored as-is; the pipeline
    passes it to VoxCPM2 ``generate(speed=...)`` or post-processes via
    FFmpeg ``atempo`` for engines without native speed control.
    """

    value: float = 1.0

    def is_enabled(self) -> bool:
        return 0.1 <= self.value <= 4.0 and abs(self.value - 1.0) > 1e-6


@dataclass
class Volume:
    """``{{volume -3db}}`` / ``{{volume 80%}}`` / ``{{volume.normalize -16}}``.

    Three modes:
    - ``gain_db``: direct gain in dB (``-3db``, ``volume.db -3``)
    - ``multiplier``: linear multiplier (``0.8``, ``80%``)
    - ``normalize_lufs``: normalize segment to target LUFS (``-16``)

    Only one mode is active per command; the parser sets exactly one.
    """

    gain_db: Optional[float] = None
    multiplier: Optional[float] = None
    normalize_lufs: Optional[float] = None

    def is_enabled(self) -> bool:
        return any(v is not None for v in (self.gain_db, self.multiplier, self.normalize_lufs))

    def is_reset(self) -> bool:
        """``{{volume.normal}}`` → all None → reset to default."""
        return not self.is_enabled()


@dataclass
class Chapter:
    """``{{chapter "Урок 1"}}`` — names the following narration group."""

    title: str = ""


@dataclass
class Alias:
    """``{{alias "GPT" "gee pee tee"}}`` — replace later text before TTS.

    The pipeline applies all accumulated aliases to the segment text
    before passing it to the engine. Aliases are scoped to the whole
    document (not reset by ``{{reset}}``) — matching LTV semantics.
    """

    target: str = ""
    replacement: str = ""


@dataclass
class Reset:
    """``{{reset}}`` / ``{{reset.voice}}`` / ``{{reset.audio}}``.

    ``reset`` (no subcommand) resets everything. Subcommand-scoped
    resets only clear the named state. Voice reset is a no-op in RTTS
    (single engine) but kept for parser compatibility.
    """

    scope: str = "all"  # "all" | "voice" | "audio"

    def resets_speed(self) -> bool:
        return self.scope in ("all", "audio")

    def resets_volume(self) -> bool:
        return self.scope in ("all", "audio")

    def resets_pause(self) -> bool:
        # Pause is per-segment in LTV (not persistent state), so reset
        # never clears it. Kept for completeness.
        return False


@dataclass
class SoundEvent:
    """``{{laugh}}`` / ``{{cough}}`` / ``{{bgm start}}`` / ``{{bgm end}}``.

    Inline events (laugh, cough, sigh, ...) replace a moment in the
    narration with a non-speech sound. Span events (bgm, hum, sing)
    wrap a following region: ``{{bgm start}}`` ... text ... ``{{bgm end}}``.

    The ``token`` field (e.g. ``[laugh]``, ``[music]``) is what a
    capable engine receives in-text, mirroring Higgs Audio's scheme.
    Engines without sound-event support (VoxCPM2, basic Silero) have
    the token stripped by the pipeline before synthesis — the engine
    just reads the surrounding text.
    """

    name: str = ""
    token: str = ""
    category: str = "inline"  # "inline" | "span"
    label: str = ""
    # For span events: ``start`` or ``end``. Inline events leave this
    # as None. The document builder pairs start/end markers and
    # attaches the span info to every segment inside the region.
    span_marker: Optional[str] = None  # None | "start" | "end"
    # Free-form intensity / repetition hint, e.g. ``{{laugh soft}}``,
    # ``{{cough x2}}``. Engines that don't understand it ignore it.
    modifier: str = ""

    def is_inline(self) -> bool:
        return self.category == "inline"

    def is_span_start(self) -> bool:
        return self.category == "span" and self.span_marker == "start"

    def is_span_end(self) -> bool:
        return self.category == "span" and self.span_marker == "end"


@dataclass
class Stress:
    """``{{stress "за́мок"}}`` or ``{{stress "замок" "а"}}``.

    Marks word stress in the text. Two forms:

    1. ``{{stress "за́мок"}}`` — the argument already contains the
       stress mark (either precomposed ``́`` or combining ``\u0301``).
       The pipeline replaces the *plain* form of the word in the
       following text with this stressed form.

    2. ``{{stress "замок" "а"}}`` — plain word + the vowel that bears
       the stress. The parser finds the first occurrence of that vowel
       in the word and inserts ``\u0301`` after it, producing
       ``замо́к``. If the named vowel doesn't appear in the word, the
       parser falls back to the first vowel in the word and emits a
       warning via the document.

    The stressed form is applied to the segment text at synthesis
    time (like an alias) so the source stays editable. Engines that
    don't honour stress marks (VoxCPM2) have the combining accent
    stripped by the pipeline before the call; engines that do (Silero
    via eSpeak, Higgs Audio) receive the marked text as-is.
    """

    target: str = ""           # plain word, e.g. "замок"
    stressed: str = ""         # stressed form, e.g. "замо́к"
    # When the user supplies {{stress "замок" "а"}}, we auto-detect
    # the vowel and store the resolved stressed form in ``stressed``.
    # ``hint_vowel`` records the hint for diagnostics/UI.
    hint_vowel: str = ""
    warning: str = ""          # set when auto-detect fell back


@dataclass
class Unknown:
    """Fallback for unrecognised commands — recorded as a warning."""

    raw: str = ""
    warning: str = ""


# Union of all command dataclasses. Used by the parser's type narrowing.
Command = Union[
    Pause, Speed, Volume, Chapter, Alias, Reset,
    SoundEvent, Stress, Unknown,
]


# --- Active narration state (mutable, segment-to-segment) -------------------


@dataclass
class NarrationState:
    """Mutable state carried from one segment to the next.

    Reset by ``{{reset}}`` (or scoped variants). Pause is *not* stored
    here because a pause command applies only to the boundary right
    after it (it doesn't persist across segments) — it's recorded on
    the preceding segment's ``pause_after`` field instead.

    Sound events: inline events (laugh, cough) are emitted inline at
    their position and don't persist. Span events (bgm, humming) set
    ``active_span`` from ``start`` to ``end`` — every segment inside
    the span carries the span marker in its state.

    Stress marks accumulate like aliases and apply to all following
    text. They are NOT cleared by ``{{reset}}`` (same LTV semantics as
    aliases) — a stress override is a pronunciation rule, not a
    transient narration parameter.
    """

    speed: float = 1.0
    volume: Volume = field(default_factory=Volume)
    chapter: Optional[str] = None
    aliases: list[Alias] = field(default_factory=list)
    # Accumulated stress marks — applied at synthesis time like aliases.
    stresses: list[Stress] = field(default_factory=list)
    # Active span sound event (bgm, humming, singing) or None. Set by
    # ``{{bgm start}}`` and cleared by ``{{bgm end}}``. While active,
    # each segment's snapshot records it so the pipeline knows to
    # emit the wrapping tokens (and strip them for engines that
    # don't support them).
    active_span: Optional[SoundEvent] = None

    def apply(self, cmd: Command) -> None:
        """Mutate state from a command. ``Pause`` is a no-op here (it's
        handled by the document builder, not the state)."""
        if isinstance(cmd, Speed):
            self.speed = cmd.value
        elif isinstance(cmd, Volume):
            if cmd.is_reset():
                self.volume = Volume()
            else:
                self.volume = cmd
        elif isinstance(cmd, Chapter):
            self.chapter = cmd.title
        elif isinstance(cmd, Alias):
            if cmd.target:
                self.aliases.append(cmd)
        elif isinstance(cmd, Stress):
            if cmd.target and cmd.stressed:
                self.stresses.append(cmd)
        elif isinstance(cmd, SoundEvent):
            if cmd.is_span_start():
                self.active_span = cmd
            elif cmd.is_span_end():
                # Only clear if the span name matches. Mismatched end
                # markers are tolerated (cleared anyway) to avoid
                # permanent state pollution.
                self.active_span = None
        elif isinstance(cmd, Reset):
            if cmd.resets_speed():
                self.speed = 1.0
            if cmd.resets_volume():
                self.volume = Volume()

    def snapshot(self) -> "NarrationState":
        """Copy current state (for attaching to a segment)."""
        return NarrationState(
            speed=self.speed,
            volume=Volume(
                gain_db=self.volume.gain_db,
                multiplier=self.volume.multiplier,
                normalize_lufs=self.volume.normalize_lufs,
            ),
            chapter=self.chapter,
            aliases=list(self.aliases),
            stresses=list(self.stresses),
            active_span=self.active_span,
        )