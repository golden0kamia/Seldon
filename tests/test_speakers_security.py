"""Voix : noms hostiles, usurpation d'une voix connue, fichiers abîmés, droits, mémoire bornée."""

from __future__ import annotations

import stat

import numpy as np
import pytest

from seldon.core.types import ActionContext, Utterance
from seldon.speakers import resemblyzer_id
from seldon.speakers.resemblyzer_id import ResemblyzerID, _safe_name
from seldon.tools.voices import Voices
from tests.test_stt_speakers import FakeEncoder, voice


@pytest.fixture
def ident(tmp_path, monkeypatch):
    resemblyzer = pytest.importorskip("resemblyzer")
    monkeypatch.setattr(resemblyzer, "preprocess_wav", lambda wav, source_sr: wav)
    i = ResemblyzerID(voices_dir=str(tmp_path / "voices"), threshold=0.8)
    i.encoder = FakeEncoder()
    return i


@pytest.mark.parametrize("name,safe", [
    ("Paul", "Paul"), ("../../etc/passwd", "etc_passwd"), ("Jean Pierre", "Jean_Pierre"),
    ("Ｂａｓｔｉｅｎ", "Bastien"), (".bashrc", "bashrc"), ("Zoé", "Zoé"), ("a‮b", "a_b"),
])
def test_safe_name(name, safe):
    assert _safe_name(name) == safe


@pytest.mark.parametrize("name", ["", "   ", ".", "..", "../", "/", "x" * 65, "\x00"])
def test_safe_name_rejects(name):
    with pytest.raises(ValueError):
        _safe_name(name)


async def test_names_are_matched_without_case(ident, tmp_path):
    await ident.enroll("Bastien", [voice(0.10)])
    assert ident.canonical_name("bastien") == ident.canonical_name("ＢＡＳＴＩＥＮ") == "Bastien"
    await ident.enroll("bastien", [voice(0.10)])   # même personne : pas de second fichier
    assert [p.name for p in (tmp_path / "voices").iterdir()] == ["Bastien.npy"]
    assert np.load(tmp_path / "voices" / "Bastien.npy").shape == (2, 256)
    assert ident.forget("BASTIEN") and ident.known_speakers() == []
    assert ident.forget("..") is False


def ctx_for(ident, utterance, answer):
    asked = []

    async def confirm(q):
        asked.append(q)
        return answer
    return ActionContext(utterance=utterance, services={"speakers": ident}, confirm=confirm), asked


@pytest.mark.parametrize("claimed", ["Bastien", "bastien", " BASTIEN. "])
async def test_unknown_voice_cannot_take_over_a_known_one(ident, tmp_path, claimed):
    await ident.enroll("Bastien", [voice(0.10)])
    before = np.load(tmp_path / "voices" / "Bastien.npy").copy()
    u = await ident.identify(Utterance(text=f"je suis {claimed}", audio=voice(0.30)))
    assert u.unknown_voice
    ctx, asked = ctx_for(ident, u, answer=False)
    r = await Voices().run({"operation": "enregistrer", "nom": claimed}, ctx)
    assert r.is_error and "Refusé" in r.text and len(asked) == 1 and "Bastien" in asked[0]
    assert u.speaker is None   # pas les droits de Bastien, même pour ce tour
    assert np.array_equal(np.load(tmp_path / "voices" / "Bastien.npy"), before)
    assert (await ident.identify(Utterance(text="x", audio=voice(0.30)))).speaker is None


async def test_confirmed_addition_to_known_voice(ident, tmp_path):
    await ident.enroll("Bastien", [voice(0.10)])
    u = await ident.identify(Utterance(text="c'est moi, bastien", audio=voice(0.30)))
    ctx, asked = ctx_for(ident, u, answer=True)
    r = await Voices().run({"operation": "enregistrer", "nom": "bastien"}, ctx)
    assert not r.is_error and asked and u.speaker == "Bastien"   # le nom tel qu'il est enregistré
    assert np.load(tmp_path / "voices" / "Bastien.npy").shape == (2, 256)


async def test_new_name_needs_no_confirmation_and_invalid_names_are_refused(ident, tmp_path):
    u = await ident.identify(Utterance(text="je suis Jean Pierre", audio=voice(0.30)))
    ctx, asked = ctx_for(ident, u, answer=False)
    assert (await Voices().run({"operation": "enregistrer", "nom": "../.."}, ctx)).is_error
    r = await Voices().run({"operation": "enregistrer", "nom": "Jean Pierre"}, ctx)
    assert not r.is_error and asked == [] and u.speaker == "Jean_Pierre"
    assert (await ident.identify(Utterance(text="re", audio=voice(0.30)))).speaker == "Jean_Pierre"   # même nom
    assert [p.name for p in (tmp_path / "voices").iterdir()] == ["Jean_Pierre.npy"]


async def test_voiceprints_are_private_and_never_unpickled(ident, tmp_path, caplog):
    await ident.enroll("Alice", [voice(0.10)])
    folder = tmp_path / "voices"
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / "Alice.npy").stat().st_mode) == 0o600
    assert [p.name for p in folder.iterdir()] == ["Alice.npy"]   # aucun fichier temporaire
    np.save(folder / "Objet.npy", np.array([{"pickle": 1}], dtype=object), allow_pickle=True)
    (folder / "Tronque.npy").write_bytes((folder / "Alice.npy").read_bytes()[:90])
    np.save(folder / "Petit.npy", np.zeros((2, 128), np.float32))
    np.save(folder / "NaN.npy", np.full((1, 256), np.nan, np.float32))
    (folder / "Vide.npy").write_bytes(b"")
    ident.load_voiceprints()   # les fichiers invalides n'empêchent pas le démarrage
    assert list(ident.voiceprints) == ["Alice"]
    assert caplog.text.count("Empreinte vocale ignorée") == 5
    # une empreinte abîmée n'est jamais écrasée
    with pytest.raises(ValueError):
        await ident.enroll("Petit", [voice(0.20)])
    assert np.load(folder / "Petit.npy").shape == (2, 128)


async def test_adaptation_failure_does_not_break_identification(ident, tmp_path):
    await ident.enroll("Alice", [voice(0.10)])
    (tmp_path / "voices" / "Alice.npy").write_bytes(b"abime")
    u = await ident.identify(Utterance(text="...", audio=voice(0.10)))
    assert u.speaker == "Alice"


async def test_unknown_voices_buffer_is_bounded(ident, monkeypatch):
    monkeypatch.setattr(resemblyzer_id, "MAX_UNKNOWN", 5)
    for i in range(20):
        await ident.identify(Utterance(text="...", audio=voice(0.30 + i / 100)))
    assert len(ident.unknown) == 5 and ident.unknown[-1].embedding.argmax() == round((0.30 + 19 / 100) * 100)
