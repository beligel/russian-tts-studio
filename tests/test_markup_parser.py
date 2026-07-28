"""Tests for the markup parser and document builder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from russian_tts_studio.markup import (  # noqa: E402
    Alias,
    Chapter,
    NarrationSegment,
    ParsedDocument,
    Pause,
    Reset,
    SoundEvent,
    Speed,
    Stress,
    Unknown,
    Volume,
    parse,
    parse_command,
)


class TestParserCommands:
    """Unit tests for ``parse_command`` — single ``{{...}}`` body."""

    def test_pause_ms(self):
        cmd = parse_command("pause 700ms")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 700
        assert not cmd.random

    def test_pause_bare_number_defaults_ms(self):
        cmd = parse_command("pause 700")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 700

    def test_pause_seconds(self):
        cmd = parse_command("pause 0.7s")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 700

    def test_pause_short_preset(self):
        cmd = parse_command("pause.short")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 300

    def test_pause_medium_preset(self):
        cmd = parse_command("pause.medium")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 700

    def test_pause_long_preset(self):
        cmd = parse_command("pause.long")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 1200

    def test_pause_random(self):
        cmd = parse_command("pause random 500 1200")
        assert isinstance(cmd, Pause)
        assert cmd.random
        assert cmd.min_ms == 500
        assert cmd.max_ms == 1200

    def test_pause_random_swapped_bounds(self):
        cmd = parse_command("pause random 1200 500")
        assert isinstance(cmd, Pause)
        assert cmd.min_ms == 500
        assert cmd.max_ms == 1200

    def test_speed_numeric(self):
        cmd = parse_command("speed 0.9")
        assert isinstance(cmd, Speed)
        assert cmd.value == pytest.approx(0.9)

    def test_speed_slow_preset(self):
        cmd = parse_command("speed.slow")
        assert isinstance(cmd, Speed)
        assert cmd.value == pytest.approx(0.85)

    def test_speed_fast_preset(self):
        cmd = parse_command("speed.fast")
        assert isinstance(cmd, Speed)
        assert cmd.value == pytest.approx(1.15)

    def test_volume_db_suffix(self):
        cmd = parse_command("volume -3db")
        assert isinstance(cmd, Volume)
        assert cmd.gain_db == pytest.approx(-3.0)
        assert cmd.multiplier is None

    def test_volume_db_subcommand(self):
        cmd = parse_command("volume.db -3")
        assert isinstance(cmd, Volume)
        assert cmd.gain_db == pytest.approx(-3.0)

    def test_volume_percent(self):
        cmd = parse_command("volume 80%")
        assert isinstance(cmd, Volume)
        assert cmd.multiplier == pytest.approx(0.8)

    def test_volume_multiplier(self):
        cmd = parse_command("volume 0.8")
        assert isinstance(cmd, Volume)
        assert cmd.multiplier == pytest.approx(0.8)

    def test_volume_normalize_lufs(self):
        cmd = parse_command("volume.normalize -16")
        assert isinstance(cmd, Volume)
        assert cmd.normalize_lufs == pytest.approx(-16.0)

    def test_volume_lufs_alias(self):
        cmd = parse_command("volume.lufs -16")
        assert isinstance(cmd, Volume)
        assert cmd.normalize_lufs == pytest.approx(-16.0)

    def test_volume_normal_reset(self):
        cmd = parse_command("volume.normal")
        assert isinstance(cmd, Volume)
        assert cmd.is_reset()

    def test_chapter_quoted(self):
        cmd = parse_command('chapter "Урок 1"')
        assert isinstance(cmd, Chapter)
        assert cmd.title == "Урок 1"

    def test_chapter_single_quotes(self):
        cmd = parse_command("chapter 'Урок 1'")
        assert isinstance(cmd, Chapter)
        assert cmd.title == "Урок 1"

    def test_alias_two_args(self):
        cmd = parse_command('alias "GPT" "gee pee tee"')
        assert isinstance(cmd, Alias)
        assert cmd.target == "GPT"
        assert cmd.replacement == "gee pee tee"

    def test_reset_bare(self):
        cmd = parse_command("reset")
        assert isinstance(cmd, Reset)
        assert cmd.scope == "all"

    def test_reset_voice(self):
        cmd = parse_command("reset.voice")
        assert isinstance(cmd, Reset)
        assert cmd.scope == "voice"

    def test_reset_audio(self):
        cmd = parse_command("reset.audio")
        assert isinstance(cmd, Reset)
        assert cmd.scope == "audio"

    def test_unknown_command(self):
        cmd = parse_command("frobnicate 42")
        assert isinstance(cmd, Unknown)
        assert "frobnicate" in cmd.warning

    def test_empty_command(self):
        cmd = parse_command("")
        assert isinstance(cmd, Unknown)

    def test_malformed_pause(self):
        cmd = parse_command("pause hello")
        assert isinstance(cmd, Unknown)

    def test_smart_quotes_normalized(self):
        # Smart double quotes around value
        cmd = parse_command("volume \u201c-3db\u201d")
        assert isinstance(cmd, Volume)
        assert cmd.gain_db == pytest.approx(-3.0)

    def test_en_dash_in_volume(self):
        cmd = parse_command("volume \u20133db")  # en dash
        assert isinstance(cmd, Volume)
        assert cmd.gain_db == pytest.approx(-3.0)

    def test_case_insensitive_command(self):
        cmd = parse_command("PAUSE 700")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 700

    def test_case_insensitive_voice_match(self):
        cmd = parse_command("Pause.MEDIUM")
        assert isinstance(cmd, Pause)
        assert cmd.ms == 700


class TestParsedDocument:
    """Tests for ``parse()`` — full document segmentation."""

    def test_no_markup_returns_single_segment(self):
        doc = parse("Просто текст без разметки.")
        assert len(doc.segments) == 1
        assert doc.segments[0].text == "Просто текст без разметки."
        assert doc.segments[0].state.speed == 1.0
        assert not doc.has_markup

    def test_chapter_splits_and_labels(self):
        doc = parse('{{chapter "Урок 1"}} Привет. Пока.')
        assert len(doc.segments) == 1
        assert doc.segments[0].state.chapter == "Урок 1"
        assert doc.segments[0].text == "Привет. Пока."
        assert len(doc.chapters) == 1
        assert doc.chapters[0].title == "Урок 1"

    def test_pause_attaches_to_preceding_segment(self):
        doc = parse("Привет. {{pause 700ms}} Пока.")
        assert len(doc.segments) == 2
        assert doc.segments[0].text == "Привет."
        assert doc.segments[0].pause_after is not None
        assert doc.segments[0].pause_after.ms == 700
        assert doc.segments[1].text == "Пока."
        assert doc.segments[1].pause_after is None

    def test_speed_starts_new_segment(self):
        doc = parse("Обычно. {{speed 0.9}} Медленно.")
        assert len(doc.segments) == 2
        assert doc.segments[0].state.speed == 1.0
        assert doc.segments[1].state.speed == pytest.approx(0.9)
        assert doc.segments[1].text == "Медленно."

    def test_volume_starts_new_segment(self):
        doc = parse("Громко. {{volume -3db}} Тихо.")
        assert len(doc.segments) == 2
        assert doc.segments[1].state.volume.gain_db == pytest.approx(-3.0)

    def test_reset_clears_state(self):
        doc = parse("{{speed 0.9}} Медленно. {{reset}} Снова обычно.")
        assert len(doc.segments) == 2
        assert doc.segments[0].state.speed == pytest.approx(0.9)
        assert doc.segments[1].state.speed == 1.0

    def test_reset_audio_keeps_chapter(self):
        doc = parse('{{chapter "Урок 1"}} {{speed 0.9}} {{reset.audio}} Текст.')
        # chapter is not cleared by reset.audio
        assert doc.segments[-1].state.chapter == "Урок 1"
        # speed IS cleared by reset.audio
        assert doc.segments[-1].state.speed == 1.0

    def test_alias_accumulates(self):
        doc = parse('{{alias "GPT" "gee pee tee"}} GPT — это хорошо.')
        assert len(doc.segments) == 1
        assert len(doc.segments[0].state.aliases) == 1
        assert doc.segments[0].state.aliases[0].target == "GPT"

    def test_alias_applies_to_following_segments(self):
        doc = parse('{{alias "GPT" "gee pee tee"}} Первый. {{speed 0.9}} GPT второй.')
        # alias carries into the second segment
        assert any(a.target == "GPT" for a in doc.segments[-1].state.aliases)

    def test_unknown_command_warning_no_split(self):
        doc = parse("Текст {{frobnicate 42}} продолжается.")
        assert len(doc.warnings) == 1
        assert "frobnicate" in doc.warnings[0]
        # Unknown command doesn't split — text is one segment
        assert len(doc.segments) == 1
        assert "продолжается" in doc.segments[0].text

    def test_multiple_chapters(self):
        doc = parse(
            '{{chapter "Глава 1"}} Текст один. '
            '{{chapter "Глава 2"}} Текст два.'
        )
        assert len(doc.chapters) == 2
        assert doc.chapters[0].title == "Глава 1"
        assert doc.chapters[1].title == "Глава 2"
        assert doc.segments[0].state.chapter == "Глава 1"
        assert doc.segments[1].state.chapter == "Глава 2"

    def test_leading_pause_before_text(self):
        doc = parse("{{pause 500ms}} Сначала пауза, потом текст.")
        # Leading pause attaches to an empty segment, then text follows
        assert len(doc.segments) >= 1
        # The pause is on the first (empty) segment
        assert doc.segments[0].pause_after is not None
        assert doc.segments[0].pause_after.ms == 500

    def test_trailing_pause(self):
        doc = parse("Текст. {{pause 300ms}}")
        assert doc.segments[0].text == "Текст."
        assert doc.segments[0].pause_after is not None
        assert doc.segments[0].pause_after.ms == 300

    def test_empty_input(self):
        doc = parse("")
        assert len(doc.segments) == 1
        assert doc.segments[0].text == ""

    def test_only_commands(self):
        doc = parse("{{speed 0.9}}{{volume -3db}}")
        # No text → at least one segment so callers don't IndexError
        assert len(doc.segments) >= 1

    def test_pause_random_resolves_to_range(self):
        doc = parse("Текст. {{pause random 500 1200}} Ещё.")
        pause = doc.segments[0].pause_after
        assert pause is not None
        assert pause.random
        # sample_ms should be within range
        for _ in range(20):
            ms = pause.sample_ms()
            assert 500 <= ms <= 1200

    def test_pause_non_random_sample_is_fixed(self):
        pause = Pause(ms=700)
        assert pause.sample_ms() == 700

    def test_full_example_from_roadmap(self):
        doc = parse(
            '{{chapter "Урок 1"}} Привет. '
            "{{pause 700ms}} Пока. "
            "{{speed 0.9}} Медленно."
        )
        assert len(doc.segments) == 3
        assert doc.segments[0].state.chapter == "Урок 1"
        assert doc.segments[0].text == "Привет."
        assert doc.segments[0].pause_after is not None
        assert doc.segments[0].pause_after.ms == 700
        assert doc.segments[2].state.speed == pytest.approx(0.9)
        assert doc.segments[2].text == "Медленно."

    def test_russian_text_preserved(self):
        doc = parse("Привет, мир! Это тест на русском языке.")
        assert "Привет, мир!" in doc.segments[0].text
        assert "русском" in doc.segments[0].text

    def test_source_field_preserved(self):
        src = '{{chapter "Урок 1"}} Привет. {{pause 700ms}}'
        doc = parse(src)
        assert doc.source == src

    def test_has_markup_true_with_commands(self):
        doc = parse("{{speed 0.9}} Текст.")
        assert doc.has_markup

    def test_has_markup_false_without_commands(self):
        doc = parse("Просто текст.")
        assert not doc.has_markup


class TestNarrationState:
    """Tests for the mutable state machine."""

    def test_apply_speed(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Speed(value=0.9))
        assert state.speed == pytest.approx(0.9)

    def test_apply_volume_reset(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Volume(gain_db=-3.0))
        assert state.volume.gain_db == pytest.approx(-3.0)
        state.apply(Volume())  # reset
        assert state.volume.is_reset()

    def test_apply_chapter(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Chapter(title="Урок 1"))
        assert state.chapter == "Урок 1"

    def test_apply_alias_accumulates(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Alias(target="GPT", replacement="gee pee tee"))
        state.apply(Alias(target="API", replacement="ay pee eye"))
        assert len(state.aliases) == 2

    def test_reset_all_clears_speed_and_volume(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Speed(value=0.9))
        state.apply(Volume(gain_db=-3.0))
        state.apply(Reset(scope="all"))
        assert state.speed == 1.0
        assert state.volume.is_reset()

    def test_snapshot_is_independent_copy(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Speed(value=0.9))
        snap = state.snapshot()
        state.apply(Speed(value=1.5))
        assert snap.speed == pytest.approx(0.9)
        assert state.speed == pytest.approx(1.5)


class TestSoundEvents:
    """Tests for ``{{laugh}}``, ``{{cough}}``, ``{{bgm start}}`` etc."""

    def test_laugh_inline(self):
        cmd = parse_command("laugh")
        assert isinstance(cmd, SoundEvent)
        assert cmd.token == "[laugh]"
        assert cmd.is_inline()
        assert not cmd.is_span_start()
        assert not cmd.is_span_end()

    def test_cough_with_modifier(self):
        cmd = parse_command("cough soft")
        assert isinstance(cmd, SoundEvent)
        assert cmd.token == "[cough]"
        assert cmd.modifier == "soft"

    def test_laugh_case_insensitive(self):
        cmd = parse_command("LAUGH")
        assert isinstance(cmd, SoundEvent)
        assert cmd.token == "[laugh]"

    def test_bgm_start(self):
        cmd = parse_command("bgm start")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_start()
        assert not cmd.is_span_end()
        assert cmd.token == "[music]"

    def test_bgm_end(self):
        cmd = parse_command("bgm end")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_end()
        assert not cmd.is_span_start()

    def test_bgm_default_is_start(self):
        # No start/end keyword → defaults to start so {{bgm}} opens a span.
        cmd = parse_command("bgm")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_start()

    def test_hum_start_begin_alias(self):
        cmd = parse_command("hum begin")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_start()

    def test_hum_end_stop_alias(self):
        cmd = parse_command("hum stop")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_end()

    def test_humming_span(self):
        cmd = parse_command("humming start")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_start()
        assert cmd.token == "[humming]"

    def test_sing_span(self):
        cmd = parse_command("sing end")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_span_end()

    def test_applause_inline(self):
        cmd = parse_command("applause")
        assert isinstance(cmd, SoundEvent)
        assert cmd.is_inline()
        assert cmd.token == "[applause]"

    def test_inline_event_inserts_token_into_text(self):
        doc = parse("Привет. {{laugh}} Это смешно.")
        # Inline event doesn't split — token is part of the segment text.
        assert len(doc.segments) == 1
        assert "[laugh]" in doc.segments[0].text
        assert "Привет." in doc.segments[0].text
        assert "Это смешно." in doc.segments[0].text

    def test_span_start_sets_active_span(self):
        doc = parse("{{bgm start}} Музыка играет. {{bgm end}} Тишина.")
        # First segment (inside span) carries active_span
        inside = [s for s in doc.segments if s.text == "Музыка играет."]
        assert len(inside) == 1
        assert inside[0].state.active_span is not None
        assert inside[0].state.active_span.name == "bgm"
        # Last segment (after end) has no active span
        after = [s for s in doc.segments if s.text == "Тишина."]
        assert len(after) == 1
        assert after[0].state.active_span is None

    def test_span_end_without_start_clears_anyway(self):
        # Tolerate a stray {{bgm end}} without a matching start —
        # active_span stays None, no crash.
        doc = parse("Текст. {{bgm end}} Ещё.")
        assert all(s.state.active_span is None for s in doc.segments)

    def test_nested_spans_replace_not_nest(self):
        # {{bgm start}} ... {{hum start}} — second start replaces the
        # first (we don't support nesting; last one wins). This matches
        # the simple state machine in NarrationState.apply.
        doc = parse("{{bgm start}} А. {{hum start}} Б. {{hum end}} В.")
        seg_b = [s for s in doc.segments if s.text == "Б."]
        assert len(seg_b) == 1
        # The last active span is "hum" (it replaced bgm)
        assert seg_b[0].state.active_span.name == "hum"


class TestStress:
    """Tests for ``{{stress "за́мок"}}`` and ``{{stress "замок" "а"}}``."""

    def test_stress_single_arg_precomposed(self):
        # User supplied the stressed form directly.
        cmd = parse_command('stress "за́мок"')
        assert isinstance(cmd, Stress)
        assert cmd.target == "замок"
        assert cmd.stressed == "за́мок"
        assert cmd.hint_vowel == ""

    def test_stress_single_arg_combining_acute(self):
        # Word with combining acute (U+0301) — should be preserved.
        cmd = parse_command('stress "замо\u0301к"')
        assert isinstance(cmd, Stress)
        # target is the stripped form
        assert cmd.target == "замок"
        # stressed keeps the combining acute
        assert "\u0301" in cmd.stressed

    def test_stress_two_args_vowel_o(self):
        cmd = parse_command('stress "замок" "о"')
        assert isinstance(cmd, Stress)
        assert cmd.target == "замок"
        assert "\u0301" in cmd.stressed
        # The acute should be after "о"
        assert "о\u0301" in cmd.stressed
        assert cmd.hint_vowel == "о"

    def test_stress_two_args_vowel_a(self):
        cmd = parse_command('stress "замок" "а"')
        assert isinstance(cmd, Stress)
        assert "а\u0301" in cmd.stressed

    def test_stress_hint_vowel_not_in_word_falls_back(self):
        # Hint vowel "я" not in "замок" → falls back to first vowel "а"
        cmd = parse_command('stress "замок" "я"')
        assert isinstance(cmd, Stress)
        assert cmd.warning  # warning recorded
        assert "а\u0301" in cmd.stressed  # fell back to first vowel

    def test_stress_no_vowels_in_word(self):
        cmd = parse_command('stress "кфтс" "а"')
        assert isinstance(cmd, Stress)
        # No vowel found → word returned unchanged, warning set
        assert "\u0301" not in cmd.stressed
        assert cmd.warning

    def test_stress_case_insensitive_vowel_match(self):
        # Uppercase hint vowel should still match lowercase in word.
        cmd = parse_command('stress "замок" "О"')
        assert isinstance(cmd, Stress)
        assert "о\u0301" in cmd.stressed

    def test_accent_alias_works(self):
        # {{accent ...}} is an alias for {{stress ...}}
        cmd = parse_command('accent "за́мок"')
        assert isinstance(cmd, Stress)

    def test_stress_empty_word_is_unknown(self):
        cmd = parse_command('stress ""')
        assert isinstance(cmd, Unknown)

    def test_stress_missing_args_is_unknown(self):
        cmd = parse_command("stress")
        assert isinstance(cmd, Unknown)

    def test_stress_accumulates_in_state(self):
        from russian_tts_studio.markup.commands import NarrationState

        state = NarrationState()
        state.apply(Stress(target="замок", stressed="за́мок"))
        state.apply(Stress(target="рука", stressed="рука́"))
        assert len(state.stresses) == 2
        snap = state.snapshot()
        # Snapshot copies — original stays intact if state mutates.
        state.apply(Stress(target="дома", stressed="дома́"))
        assert len(snap.stresses) == 2
        assert len(state.stresses) == 3

    def test_stress_applies_to_following_segments(self):
        doc = parse('{{stress "за́мок"}} Замок большой. {{speed 0.9}} Замок стоит.')
        # Stress carries into both segments (accumulates, not reset).
        for seg in doc.segments:
            assert any(s.target == "замок" for s in seg.state.stresses)

    def test_stress_warning_surfaces_in_document_warnings(self):
        # Hint vowel not found → warning should appear in doc.warnings.
        doc = parse('{{stress "замок" "я"}} Замок.')
        assert any("not found" in w for w in doc.warnings)

    def test_stress_not_reset_by_reset_command(self):
        # Stress marks survive {{reset}} — they're pronunciation rules.
        doc = parse('{{stress "за́мок"}} Замок. {{reset}} Замок снова.')
        last_seg = doc.segments[-1]
        assert any(s.target == "замок" for s in last_seg.state.stresses)


class TestPipelineMarkupStripping:
    """Tests for the pipeline's _apply_stresses and _strip_unsupported_markup.

    ``_apply_aliases`` and ``_apply_stresses`` are static methods and
    are called directly. ``_strip_unsupported_markup`` is an instance
    method that queries the active engine's ``supports_sound_events()``
    and ``supports_stress_marks()`` capabilities, so these tests build
    a lightweight ``TTSPipeline`` with a stub synthesizer.
    """

    def _make_pipeline(self, supports_sound=False, supports_stress=False):
        """Build a TTSPipeline with a stub synth that advertises the
        given capabilities. Doesn't call ``initialize()`` (which would
        try to load real models)."""
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline

        pipe = TTSPipeline.__new__(TTSPipeline)
        pipe.engine = "voxcpm"
        pipe.config = None
        pipe.silero = None
        pipe.transcriber = None
        pipe.similarity_calc = None
        pipe._initialized = False

        class _StubSynth:
            @staticmethod
            def supports_sound_events():
                return supports_sound

            @staticmethod
            def supports_stress_marks():
                return supports_stress

        pipe.synth = _StubSynth()
        return pipe

    def test_apply_stresses_replaces_plain_word(self):
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline
        from russian_tts_studio.markup import Stress

        stresses = [Stress(target="замок", stressed="за́мок")]
        out = TTSPipeline._apply_stresses("Замок большой", stresses)
        # Note: case-sensitive replace — "Замок" != "замок". The user
        # controls the form they put in the markup. We test the
        # lowercase form here.
        out_lower = TTSPipeline._apply_stresses("замок большой", stresses)
        assert "за́мок" in out_lower

    def test_strip_removes_sound_tokens_for_voxcpm(self):
        # Default (no support) → strip tokens (VoxCPM2 path).
        pipe = self._make_pipeline(supports_sound=False, supports_stress=False)
        text = "Привет. [laugh] Это смешно. [music] игра"
        out = pipe._strip_unsupported_markup(text)
        assert "[laugh]" not in out
        assert "[music]" not in out
        assert "Привет." in out
        assert "Это смешно." in out
        assert "игра" in out

    def test_strip_preserves_sound_tokens_for_higgs(self):
        # Engine supports sound events → tokens preserved.
        pipe = self._make_pipeline(supports_sound=True, supports_stress=True)
        text = "Привет. [laugh] Это смешно. [music] игра"
        out = pipe._strip_unsupported_markup(text)
        assert "[laugh]" in out
        assert "[music]" in out

    def test_strip_removes_combining_acute_for_voxcpm(self):
        pipe = self._make_pipeline(supports_sound=False, supports_stress=False)
        text = "за\u0301мок большо\u0301й"
        out = pipe._strip_unsupported_markup(text)
        assert "\u0301" not in out
        assert "замок" in out

    def test_strip_preserves_combining_acute_for_higgs(self):
        pipe = self._make_pipeline(supports_sound=True, supports_stress=True)
        text = "за\u0301мок большо\u0301й"
        out = pipe._strip_unsupported_markup(text)
        assert "\u0301" in out  # preserved

    def test_strip_collapses_multiple_spaces(self):
        pipe = self._make_pipeline(supports_sound=False, supports_stress=False)
        # After stripping [laugh], "Привет.  Это" should collapse to "Привет. Это"
        text = "Привет. [laugh] Это"
        out = pipe._strip_unsupported_markup(text)
        assert "  " not in out

    def test_strip_handles_empty_text(self):
        pipe = self._make_pipeline()
        assert pipe._strip_unsupported_markup("") == ""

    def test_strip_stress_only_when_engine_lacks_support(self):
        # Engine supports sound events but NOT stress marks → strip only U+0301.
        pipe = self._make_pipeline(supports_sound=True, supports_stress=False)
        text = "за\u0301мок [laugh] смешно"
        out = pipe._strip_unsupported_markup(text)
        assert "\u0301" not in out  # stress stripped
        assert "[laugh]" in out  # sound event preserved

    def test_apply_then_strip_pipeline_voxcpm(self):
        # Full sequence: stress applied, then stripped for VoxCPM2.
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline
        from russian_tts_studio.markup import Stress

        pipe = self._make_pipeline(supports_sound=False, supports_stress=False)
        stresses = [Stress(target="замок", stressed="замо\u0301к")]
        text = "замок большой"
        after_apply = TTSPipeline._apply_stresses(text, stresses)
        assert "\u0301" in after_apply
        after_strip = pipe._strip_unsupported_markup(after_apply)
        # After strip, the acute is gone (VoxCPM2 path).
        assert "\u0301" not in after_strip
        assert "замок" in after_strip

    def test_apply_then_strip_pipeline_higgs(self):
        # Higgs supports both → stress preserved through the whole path.
        from russian_tts_studio.pipeline.tts_pipeline import TTSPipeline
        from russian_tts_studio.markup import Stress

        pipe = self._make_pipeline(supports_sound=True, supports_stress=True)
        stresses = [Stress(target="замок", stressed="замо\u0301к")]
        text = "замок большой"
        after_apply = TTSPipeline._apply_stresses(text, stresses)
        after_strip = pipe._strip_unsupported_markup(after_apply)
        # Higgs honours U+0301 — preserved.
        assert "\u0301" in after_strip