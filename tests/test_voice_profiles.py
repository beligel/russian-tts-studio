"""Tests for voice profiles (text-described voices via YAML).

Covers the ``models/voice_profiles.py`` module and the
``/api/voice-profiles`` endpoints. Tests use an isolated temp directory
for the profiles YAML so they don't touch the user's real
``output/reference/profiles.yaml``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_profiles_path(tmp_path):
    """Isolated profiles.yaml path for each test."""
    return tmp_path / "profiles.yaml"


class TestProfileLoading:
    """Loading + starter-file creation."""

    def test_load_creates_starter_file(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import load_profiles

        assert not temp_profiles_path.exists()
        reg = load_profiles(temp_profiles_path)
        # Starter file created on first access
        assert temp_profiles_path.exists()
        # Starter includes the 8 built-in profiles
        assert len(reg.profiles) == 8

    def test_starter_includes_russian_voices(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import load_profiles

        reg = load_profiles(temp_profiles_path)
        # RTTS ships Russian-language profiles
        assert "male_ru_calm" in reg.profiles
        assert "female_ru_warm" in reg.profiles

    def test_starter_includes_english_voices(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import load_profiles

        reg = load_profiles(temp_profiles_path)
        # From upstream Higgs Audio profile.yaml
        assert "male_en" in reg.profiles
        assert "male_en_british" in reg.profiles
        assert "female_en_british" in reg.profiles

    def test_cache_hit_no_reload(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import load_profiles

        reg1 = load_profiles(temp_profiles_path)
        reg2 = load_profiles(temp_profiles_path)
        # Same object — cache hit
        assert reg1 is reg2

    def test_force_reload(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import load_profiles

        reg1 = load_profiles(temp_profiles_path)
        reg2 = load_profiles(temp_profiles_path, force=True)
        # Different object — forced reload
        assert reg1 is not reg2
        # But same content
        assert reg1.list_names() == reg2.list_names()

    def test_reload_on_file_change(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import load_profiles, add_profile

        reg1 = load_profiles(temp_profiles_path)
        # add_profile writes to disk + reloads
        add_profile("new_voice", "description", temp_profiles_path)
        reg2 = load_profiles(temp_profiles_path)
        # Cache invalidated by mtime change → new object
        assert reg1 is not reg2
        assert "new_voice" in reg2.profiles


class TestProfileLookup:
    """get_profile, list_profiles, is_profile_reference, resolve_reference."""

    def test_get_profile_returns_description(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import get_profile, load_profiles

        load_profiles(temp_profiles_path)
        p = get_profile("male_en", temp_profiles_path)
        assert p is not None
        assert p.name == "male_en"
        assert "Male" in p.description
        assert "American" in p.description

    def test_get_profile_unknown_returns_none(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import get_profile, load_profiles

        load_profiles(temp_profiles_path)
        assert get_profile("nonexistent_profile", temp_profiles_path) is None

    def test_list_profiles_sorted(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import list_profiles, load_profiles

        load_profiles(temp_profiles_path)
        names = [p.name for p in list_profiles(temp_profiles_path)]
        assert names == sorted(names)

    def test_is_profile_reference(self):
        from russian_tts_studio.models.voice_profiles import is_profile_reference

        assert is_profile_reference("profile:foo") is True
        assert is_profile_reference("profile:male_en") is True
        assert is_profile_reference("/path/to.wav") is False
        assert is_profile_reference("speaker.wav") is False
        assert is_profile_reference("") is False
        assert is_profile_reference(None) is False

    def test_resolve_reference_returns_description(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import resolve_reference, load_profiles

        load_profiles(temp_profiles_path)
        desc = resolve_reference("profile:male_en", temp_profiles_path)
        assert desc is not None
        assert "Male" in desc

    def test_resolve_reference_unknown_returns_none(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import resolve_reference, load_profiles

        load_profiles(temp_profiles_path)
        assert resolve_reference("profile:nonexistent", temp_profiles_path) is None

    def test_resolve_reference_non_profile_returns_none(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import resolve_reference

        # File paths are not profile references
        assert resolve_reference("/path/to.wav", temp_profiles_path) is None


class TestProfileMutation:
    """add_profile, delete_profile — persistence."""

    def test_add_new_profile(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import add_profile, get_profile, load_profiles

        load_profiles(temp_profiles_path)
        add_profile("custom_voice", "A custom test voice", temp_profiles_path)
        p = get_profile("custom_voice", temp_profiles_path)
        assert p is not None
        assert p.description == "A custom test voice"

    def test_add_overwrites_existing(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import add_profile, get_profile, load_profiles

        load_profiles(temp_profiles_path)
        original = get_profile("male_en", temp_profiles_path).description
        add_profile("male_en", "OVERWRITTEN", temp_profiles_path)
        new = get_profile("male_en", temp_profiles_path).description
        assert new == "OVERWRITTEN"
        assert new != original

    def test_delete_profile(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import delete_profile, get_profile, load_profiles

        load_profiles(temp_profiles_path)
        assert get_profile("male_en", temp_profiles_path) is not None
        result = delete_profile("male_en", temp_profiles_path)
        assert result is True
        assert get_profile("male_en", temp_profiles_path) is None

    def test_delete_nonexistent_returns_false(self, temp_profiles_path):
        from russian_tts_studio.models.voice_profiles import delete_profile, load_profiles

        load_profiles(temp_profiles_path)
        result = delete_profile("nonexistent", temp_profiles_path)
        assert result is False


class TestHiggsProfileIntegration:
    """HiggsAudioSynthesizer._resolve_profiles + _parse_reference_list."""

    def test_resolve_profiles_extracts_descriptions(self, temp_profiles_path, monkeypatch):
        from russian_tts_studio.models.voice_profiles import load_profiles
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        # Load starter profiles into the temp path.
        load_profiles(temp_profiles_path)
        # Monkeypatch the DEFAULT_PROFILES_PATH so resolve_reference
        # reads from our temp file.
        import russian_tts_studio.models.voice_profiles as vp_mod
        monkeypatch.setattr(vp_mod, "DEFAULT_PROFILES_PATH", temp_profiles_path)

        synth = HiggsAudioSynthesizer()
        # Single profile reference
        descs = synth._resolve_profiles("profile:male_en")
        assert len(descs) == 1
        assert "Male" in descs[0]

        # Mixed: profile + file path → only profile resolves
        descs = synth._resolve_profiles("profile:male_en,/tmp/foo.wav")
        assert len(descs) == 1  # only the profile

        # Multi-speaker: two profiles
        descs = synth._resolve_profiles("profile:male_en,profile:female_en_british")
        assert len(descs) == 2

        # Unknown profile → skipped (warning logged)
        descs = synth._resolve_profiles("profile:nonexistent")
        assert descs == []

        # No reference → empty
        assert synth._resolve_profiles(None) == []
        assert synth._resolve_profiles("") == []

    def test_parse_reference_list_skips_profiles(self, temp_profiles_path, monkeypatch):
        from russian_tts_studio.models.voice_profiles import load_profiles
        from russian_tts_studio.models.higgs_synth import HiggsAudioSynthesizer

        load_profiles(temp_profiles_path)
        import russian_tts_studio.models.voice_profiles as vp_mod
        monkeypatch.setattr(vp_mod, "DEFAULT_PROFILES_PATH", temp_profiles_path)

        synth = HiggsAudioSynthesizer()
        # profile: refs are NOT file paths → excluded from the path list
        paths = synth._parse_reference_list("profile:male_en")
        assert paths == []

        # Mixed → only the file path survives
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav = f.name
        try:
            paths = synth._parse_reference_list(f"profile:male_en,{tmp_wav}")
            assert len(paths) == 1
            assert paths[0] == Path(tmp_wav)
        finally:
            os.unlink(tmp_wav)


class TestVoiceProfilesAPI:
    """``/api/voice-profiles`` endpoints."""

    def test_list_profiles_endpoint(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from web.app import app

        # Isolate the profiles file so we don't touch the real one.
        test_path = tmp_path / "profiles.yaml"
        import russian_tts_studio.models.voice_profiles as vp_mod
        monkeypatch.setattr(vp_mod, "DEFAULT_PROFILES_PATH", test_path)

        client = TestClient(app)
        resp = client.get("/api/voice-profiles")
        assert resp.status_code == 200
        body = resp.json()
        assert "profiles" in body
        assert len(body["profiles"]) >= 8
        assert any(p["name"] == "male_en" for p in body["profiles"])

    def test_add_profile_endpoint(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from web.app import app

        test_path = tmp_path / "profiles.yaml"
        import russian_tts_studio.models.voice_profiles as vp_mod
        monkeypatch.setattr(vp_mod, "DEFAULT_PROFILES_PATH", test_path)

        client = TestClient(app)
        resp = client.post(
            "/api/voice-profiles",
            data={"name": "api_test_voice", "description": "API test description"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "api_test_voice"

        # Verify it shows up in the list
        resp = client.get("/api/voice-profiles")
        names = [p["name"] for p in resp.json()["profiles"]]
        assert "api_test_voice" in names

    def test_delete_profile_endpoint(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from web.app import app

        test_path = tmp_path / "profiles.yaml"
        import russian_tts_studio.models.voice_profiles as vp_mod
        monkeypatch.setattr(vp_mod, "DEFAULT_PROFILES_PATH", test_path)

        client = TestClient(app)
        # Add then delete
        client.post(
            "/api/voice-profiles",
            data={"name": "to_delete", "description": "temporary"},
        )
        resp = client.delete("/api/voice-profiles/to_delete")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == "to_delete"

        # Deleting again → 404
        resp = client.delete("/api/voice-profiles/to_delete")
        assert resp.status_code == 404

    def test_add_profile_validation(self, monkeypatch, tmp_path):
        from fastapi.testclient import TestClient
        from web.app import app

        test_path = tmp_path / "profiles.yaml"
        import russian_tts_studio.models.voice_profiles as vp_mod
        monkeypatch.setattr(vp_mod, "DEFAULT_PROFILES_PATH", test_path)

        client = TestClient(app)
        # Empty name → FastAPI returns 422 (required field validation)
        # before our handler's explicit 400 check. Both are "rejected".
        resp = client.post(
            "/api/voice-profiles",
            data={"name": "", "description": "desc"},
        )
        assert resp.status_code in (400, 422)
        # Empty description → same
        resp = client.post(
            "/api/voice-profiles",
            data={"name": "valid_name", "description": ""},
        )
        assert resp.status_code in (400, 422)