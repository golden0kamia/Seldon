"""Identification des voix et outil « voix » (sans modèle Resemblyzer)."""

from __future__ import annotations

import stat
import time

import numpy as np
import pytest

from seldon.ear.speakers import DIM, Resemblyzer, Voices, read_samples, safe_name
from tests.conftest import FakeApp, context


def vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def ident(tmp_path):
    return Resemblyzer(voices_dir=str(tmp_path / "voices"))


@pytest.mark.parametrize("name, safe", [("Bastien", "Bastien"), ("Jean Paul", "Jean_Paul"),
                                        ("../../etc/passwd", "etc_passwd"), ("Ｂａｓｔｉｅｎ", "Bastien")])
def test_safe_name(name, safe):
    assert safe_name(name) == safe


@pytest.mark.parametrize("name", ["", "..", "/", "x" * 65])
def test_safe_name_rejects(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_voiceprints_are_private_and_never_unpickled(ident, tmp_path):
    ident.add_samples("Bastien", vec(0)[None, :])
    path = ident.voices_dir / "Bastien.npy"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    evil = ident.voices_dir / "Pirate.npy"
    np.save(evil, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        read_samples(evil)


async def test_identify_known_and_unknown(ident, monkeypatch):
    ident.add_samples("Bastien", vec(0)[None, :])
    from seldon.base import Utterance

    monkeypatch.setattr(ident, "embed", lambda audio: vec(0) if audio[0] > 0 else vec(1))
    known = Utterance("salut", audio=np.ones(16000, dtype=np.float32))
    await ident.identify(known)
    assert known.speaker == "Bastien"
    stranger = Utterance("salut", audio=-np.ones(16000, dtype=np.float32))
    await ident.identify(stranger)
    assert stranger.speaker is None and stranger.unknown_voice and len(ident.unknown) == 1


@pytest.mark.parametrize("claimed", ["Bastien", "bastien", "BASTIEN"])
async def test_unknown_voice_cannot_take_over_a_known_one(ident, claimed):
    ident.add_samples("Bastien", vec(0)[None, :])
    ident.unknown = [(time.time(), vec(1))]
    app = FakeApp(answer=False)
    app.speakers = ident
    result = await Voices().run({"operation": "enregistrer", "nom": claimed}, context(app))
    assert result.error and app.questions
    assert np.allclose(read_samples(ident.voices_dir / "Bastien.npy"), vec(0))


async def test_new_name_needs_no_confirmation(ident):
    ident.unknown = [(time.time(), vec(1))]
    app = FakeApp()
    app.speakers = ident
    ctx = context(app, unknown_voice=True)
    result = await Voices().run({"operation": "enregistrer", "nom": "Alice"}, ctx)
    assert not result.error and not app.questions
    assert "Alice" in ident.known() and ctx.utterance.speaker == "Alice"
    assert (await Voices().run({"operation": "enregistrer", "nom": "../"}, ctx)).error


async def test_forget_needs_confirmation(ident):
    ident.add_samples("Bastien", vec(0)[None, :])
    app = FakeApp(answer=False)
    app.speakers = ident
    assert (await Voices().run({"operation": "oublier", "nom": "bastien"}, context(app))).error
    assert "Bastien" in ident.known()
    app.answer = True
    await Voices().run({"operation": "oublier", "nom": "bastien"}, context(app))
    assert "Bastien" not in ident.known()
