"""Tests STT / identification sans micro ni téléchargement de modèle."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from seldon.core import registry
from seldon.core.types import Utterance
from seldon.speakers.resemblyzer_id import ResemblyzerID
from seldon.stt.whisper import FRAME, SAMPLE_RATE, EnergyVAD, Segmenter, WhisperSTT, strip_wake_word


def frames(seconds: float, amplitude: float) -> list[np.ndarray]:
    n = int(seconds * SAMPLE_RATE / FRAME)
    rng = np.random.default_rng(0)
    return [(rng.standard_normal(FRAME) * amplitude).astype(np.float32) for _ in range(n)]


def run_vad(chunks: list[np.ndarray], segmenter: Segmenter, vad=EnergyVAD(-40)) -> list[np.ndarray]:
    out = [segmenter.push(f, vad(f)) for f in chunks]
    return [s for s in out if s is not None]


# -- VAD / segmentation ------------------------------------------------------

def test_segmenter_splits_on_silence():
    seg = Segmenter(silence_ms=500, min_speech_ms=200, pad_ms=100)
    audio = frames(1, 0.001) + frames(1.5, 0.3) + frames(1, 0.001) + frames(0.8, 0.3) + frames(1, 0.001)
    segments = run_vad(audio, seg)
    assert len(segments) == 2
    # parole + marge avant (100 ms) + marge après (100 ms), à une trame près
    for seconds, s in zip((1.5, 0.8), segments):
        assert abs(len(s) / SAMPLE_RATE - (seconds + 0.2)) < 2 * FRAME / SAMPLE_RATE
    assert not seg.active


def test_segmenter_ignores_short_bursts():
    seg = Segmenter(silence_ms=300, min_speech_ms=300)
    assert run_vad(frames(0.5, 0.001) + frames(0.1, 0.5) + frames(1, 0.001), seg) == []


def test_segmenter_max_duration_and_flush():
    seg = Segmenter(silence_ms=500, max_speech_s=1.0, pad_ms=0)
    segments = run_vad(frames(2.5, 0.3), seg)
    assert len(segments) == 2 and all(len(s) <= SAMPLE_RATE + FRAME for s in segments)
    assert seg.active and seg.flush() is not None and not seg.active


def test_segmenter_preserves_samples():
    seg = Segmenter(silence_ms=200, min_speech_ms=0, pad_ms=0)
    speech = frames(0.5, 0.3)
    [segment] = run_vad(frames(0.3, 0.0) + speech + frames(0.5, 0.0), seg)
    np.testing.assert_array_equal(segment, np.concatenate(speech))


def test_energy_vad():
    vad = EnergyVAD(-40)
    assert vad(np.zeros(FRAME, np.float32)) == 0.0
    assert vad(np.full(FRAME, 0.1, np.float32)) == 1.0


# -- WhisperSTT (modèle et micro simulés) ----------------------------------------

def test_wake_words():
    assert strip_wake_word("Seldon, allume la lumière.", ["seldon"]) == "allume la lumière."
    assert strip_wake_word("seldon", ["Seldon"]) == ""
    assert strip_wake_word("Allume la lumière", ["Seldon"]) is None
    assert strip_wake_word("Seldonia est loin", ["Seldon"]) is None


def test_accept_filters():
    stt = WhisperSTT(wake_words=["Seldon"])
    assert stt.accept("") is None
    assert stt.accept("Sous-titres réalisés par la communauté d'Amara.org") is None
    assert stt.accept("Seldon") == ("Seldon", True)      # simple appel
    assert stt.accept("Bonjour") == ("Bonjour", False)   # gardé : le pipeline décide
    assert stt.accept("Seldon, quelle heure est-il ?") == ("quelle heure est-il ?", True)
    assert WhisperSTT().accept("Bonjour") == ("Bonjour", None)


def test_transcribe_drops_no_speech():
    stt = WhisperSTT(language=None)
    segs = [SimpleNamespace(text=" Bonjour", no_speech_prob=0.1), SimpleNamespace(text=" euh", no_speech_prob=0.9)]
    stt.model = SimpleNamespace(transcribe=lambda audio, **kw: (iter(segs), SimpleNamespace(language="fr")))
    assert stt.transcribe(np.zeros(SAMPLE_RATE, np.float32)) == ("Bonjour", "fr")


async def test_vad_thread_and_listen_with_pause():
    stt = WhisperSTT(vad="energy", energy_threshold_db=-40, silence_ms=300, min_speech_ms=100)
    stt.vad = EnergyVAD(-40)
    stt.transcribe = lambda audio: (f"{len(audio)}", "fr")
    stt._segments = asyncio.Queue()
    stt._worker = threading.Thread(target=stt._vad_loop, args=(asyncio.get_running_loop(),), daemon=True)
    stt._worker.start()

    async def feed(chunks):  # simule le callback micro (qui jette tout pendant la pause)
        for c in chunks:
            if not stt._paused.is_set():
                stt._frames.put(c)
        while not stt._frames.empty():  # laisse le thread VAD suivre, comme en temps réel
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

    await feed(frames(0.5, 0.3) + frames(0.5, 0.0))         # phrase 1
    stt.pause()
    await feed(frames(0.5, 0.3) + frames(0.5, 0.0))         # le TTS parle : ignoré
    stt.resume()
    await feed(frames(0.2, 0.3))                           # début de phrase...
    stt.pause(); stt.resume()                          # ...coupé : segment en cours jeté
    await feed(frames(1.5, 0.3) + frames(0.5, 0.0))         # phrase 2

    got = []
    async def collect():
        async for u in stt.listen():
            got.append(u)
            if len(got) == 2:
                return
    await asyncio.wait_for(collect(), 5)
    await stt.stop()
    assert [u.language for u in got] == ["fr", "fr"]
    assert all(isinstance(u.audio, np.ndarray) and u.text == str(len(u.audio)) for u in got)
    # ni la phrase « TTS » (~0,9 s) ni le début coupé (+0,2 s) ne doivent apparaître
    assert [round(len(u.audio) / SAMPLE_RATE, 1) for u in got] == [0.7, 1.7]


def test_registry_resolves_plugins():
    assert registry.resolve("stt", "whisper") is WhisperSTT
    assert registry.resolve("speakers", "resemblyzer") is ResemblyzerID


# -- Identification du locuteur (encodeur simulé) -------------------------------

class FakeEncoder:
    """Embedding = direction déterminée par la « fréquence » du signal."""

    def embed_utterance(self, wav):
        v = np.zeros(256, np.float32)
        v[int(round(wav[0] * 100)) % 256] = 1.0
        v[255] = 0.3
        return v


def voice(code: float, seconds: float = 1.0) -> np.ndarray:
    return np.full(int(seconds * SAMPLE_RATE), code, np.float32)


@pytest.fixture
def ident(tmp_path, monkeypatch):
    resemblyzer = pytest.importorskip("resemblyzer")
    monkeypatch.setattr(resemblyzer, "preprocess_wav", lambda wav, source_sr: wav)
    i = ResemblyzerID(voices_dir=str(tmp_path / "voices"), threshold=0.8)
    i.encoder = FakeEncoder()
    return i


async def test_enroll_persists_and_identifies(ident, tmp_path):
    await ident.enroll("Alice", [voice(0.10), voice(0.10)])
    await ident.enroll("Bob", [voice(0.20)])
    stored = np.load(tmp_path / "voices" / "Alice.npy")
    assert stored.shape == (2, 256)

    fresh = ResemblyzerID(voices_dir=str(tmp_path / "voices"), threshold=0.8)
    fresh.encoder = ident.encoder
    await fresh.start()
    assert set(fresh.voiceprints) == {"Alice", "Bob"}

    u = await fresh.identify(Utterance(text="salut", audio=voice(0.20)))
    assert u.speaker == "Bob" and u.speaker_confidence > 0.99
    u = await fresh.identify(Utterance(text="salut", audio=voice(0.50)))
    assert u.speaker is None and u.speaker_confidence < 0.8


async def test_enroll_appends_samples(ident, tmp_path):
    await ident.enroll("Alice", [voice(0.10)])
    await ident.enroll("Alice", [voice(0.10)])
    assert np.load(tmp_path / "voices" / "Alice.npy").shape == (2, 256)


async def test_identify_skips_short_or_missing_audio(ident):
    await ident.enroll("Alice", [voice(0.10)])
    u = Utterance(text="x", speaker="clavier")
    assert (await ident.identify(u)).speaker == "clavier"
    u = Utterance(text="x", audio=voice(0.10, seconds=0.1))
    assert (await ident.identify(u)).speaker is None


def test_match_threshold():
    i = ResemblyzerID(threshold=0.9)
    assert i.match(np.ones(4)) == (None, 0.0)
    i.voiceprints = {"a": np.array([1.0, 0, 0, 0]), "b": np.array([0, 1.0, 0, 0])}
    assert i.match(np.array([0.1, 1.0, 0, 0]))[0] == "b"
    name, score = i.match(np.array([1.0, 1.0, 0, 0]))
    assert name is None and 0.70 < score < 0.71


# -- Voix inconnues : enregistrement à la volée ----------------------------------

async def test_identify_flags_unknown_voice_without_voiceprints(ident):
    u = await ident.identify(Utterance(text="bonjour", audio=voice(0.30)))
    assert u.speaker is None and u.unknown_voice
    assert len(ident.unknown) == 1
    u = await ident.identify(Utterance(text="x", speaker="clavier"))
    assert not u.unknown_voice


async def test_enroll_recent_keeps_only_last_unknown_voice(ident, tmp_path):
    await ident.enroll("Alice", [voice(0.10)])
    for code in (0.30, 0.40, 0.30, 0.10):   # Paul, un autre inconnu, Paul, puis Alice (reconnue)
        await ident.identify(Utterance(text="...", audio=voice(code)))
    await ident.identify(Utterance(text="je m'appelle Paul", audio=voice(0.30)))
    assert await ident.enroll_recent("Paul") == 3
    assert np.load(tmp_path / "voices" / "Paul.npy").shape == (3, 256)
    assert len(ident.unknown) == 1   # l'autre inconnu reste enregistrable
    u = await ident.identify(Utterance(text="re", audio=voice(0.30)))
    assert u.speaker == "Paul" and not u.unknown_voice


async def test_enroll_recent_forgets_old_voices(ident):
    ident.unknown_memory_s = 60
    await ident.identify(Utterance(text="...", audio=voice(0.30), timestamp=time.time() - 120))
    assert await ident.enroll_recent("Paul") == 0


async def test_forget_voice(ident, tmp_path):
    await ident.enroll("Alice", [voice(0.10)])
    assert ident.known_speakers() == ["Alice"]
    assert ident.forget("Alice") and not (tmp_path / "voices" / "Alice.npy").exists()
    assert not ident.forget("Alice") and ident.known_speakers() == []


async def test_voices_action(ident):
    from seldon.actions.voices import Voices
    from seldon.brain.prompts import speaker_label
    from seldon.core.types import ActionContext

    action = Voices()
    u = await ident.identify(Utterance(text="je suis Paul", audio=voice(0.30)))
    assert speaker_label(u) == "voix inconnue"
    asked = []

    async def confirm(q):
        asked.append(q)
        return True

    ctx = ActionContext(utterance=u, services={"speakers": ident}, confirm=confirm)
    r = await action.run({"operation": "enregistrer", "nom": " Paul "}, ctx)
    assert not r.is_error and "Paul" in r.text and u.speaker == "Paul" and speaker_label(u) == "Paul"
    assert (await action.run({"operation": "enregistrer", "nom": "Paul"}, ctx)).is_error   # plus rien à enregistrer
    assert "Paul" in (await action.run({"operation": "lister"}, ctx)).text
    assert not (await action.run({"operation": "oublier", "nom": "Paul"}, ctx)).is_error and asked
    assert (await action.run({"operation": "enregistrer"}, ctx)).is_error
    assert (await action.run({"operation": "lister"}, ActionContext())).is_error
    assert speaker_label(Utterance(text="x")) == "inconnu"


# -- Voix qui s'affinent avec l'usage ---------------------------------------------

async def test_recognized_voice_is_refined(ident, tmp_path):
    ident.max_samples = 3
    await ident.enroll("Alice", [voice(0.10)])
    await ident.enroll("Bob", [voice(0.20)])
    now = time.time()
    for dt in (0, 10, 70, 140, 210):   # une phrase apprise au plus par minute
        await ident.identify(Utterance(text="...", audio=voice(0.10), timestamp=now + dt))
    assert np.load(tmp_path / "voices" / "Alice.npy").shape == (3, 256)   # 1 + 4 apprises, 3 gardées
    assert np.load(tmp_path / "voices" / "Bob.npy").shape == (1, 256)


async def test_uncertain_or_disabled_refinement(ident, tmp_path):
    await ident.enroll("Alice", [voice(0.10)])
    ident.adapt_threshold = 1.01    # score (au plus 1) insuffisant
    await ident.identify(Utterance(text="...", audio=voice(0.10)))
    ident.adapt_threshold = None
    await ident.identify(Utterance(text="...", audio=voice(0.10)))
    assert np.load(tmp_path / "voices" / "Alice.npy").shape == (1, 256)


def test_refinement_needs_clear_margin():
    i = ResemblyzerID(adapt_threshold=0.8)
    assert i._should_adapt([("a", 0.95), ("b", 0.70)], 0)
    assert not i._should_adapt([("a", 0.95), ("b", 0.90)], 0)   # deux voix trop proches


# -- Coupure de parole --------------------------------------------------------------

def test_contains_phrase():
    from seldon.stt.whisper import contains_phrase

    assert contains_phrase("Attends, Seldon !", ["attends", "seldon", "stop"]) == ["attends", "seldon"]
    assert contains_phrase("Tais toi.", ["tais-toi"]) == ["tais-toi"]
    assert contains_phrase("Le stoppeur", ["stop"]) == []


async def test_whisper_interruption_ignores_echo_and_requeues_requests():
    stt = WhisperSTT(wake_words=["Seldon"])
    stt._segments, stt._barge = asyncio.Queue(), asyncio.Queue()
    heard = iter(["Il faut s'arrêter au stop.", "euh", "Attends, plutôt demain", "Stop !"])
    stt.transcribe = lambda audio: (next(heard), "fr")
    for _ in range(4):
        stt._barge.put_nowait(np.zeros(10, np.float32))

    spoken = "Tourne à droite au stop."
    task = asyncio.create_task(stt.interruption(spoken))
    await asyncio.sleep(0)   # la file est vidée au début : on la remplit ensuite
    for _ in range(4):
        stt._barge.put_nowait(np.zeros(10, np.float32))
    u = await asyncio.wait_for(task, 2)
    assert u.text == "Attends, plutôt demain" and u.addressed and u.notes
    assert stt._segments.get_nowait() is u           # demande à traiter ensuite

    stt.transcribe = lambda audio: ("Seldon, stop.", "fr")
    task = asyncio.create_task(stt.interruption("Bonjour"))
    await asyncio.sleep(0)
    stt._barge.put_nowait(np.zeros(10, np.float32))
    u = await asyncio.wait_for(task, 2)
    assert u.text == "stop." and stt._segments.empty()   # simple « stop » : rien d'autre à traiter


async def test_vad_thread_routes_loud_speech_to_barge_in_while_paused():
    stt = WhisperSTT(vad="energy", energy_threshold_db=-50, barge_in_db=-20)
    stt.vad = EnergyVAD(-50)
    stt._segments, stt._barge = asyncio.Queue(), asyncio.Queue()
    stt._worker = threading.Thread(target=stt._vad_loop, args=(asyncio.get_running_loop(),), daemon=True)
    stt._worker.start()

    async def feed(chunks):
        for c in chunks:
            stt._frames.put(c)
        while not stt._frames.empty():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

    stt.pause()
    await feed(frames(0.6, 0.03) + frames(0.5, 0.0))   # écho (≈ -30 dBFS) : sous le seuil
    assert stt._barge.empty()
    await feed(frames(0.6, 0.3) + frames(0.5, 0.0))    # voix forte (≈ -10 dBFS)
    assert stt._barge.qsize() == 1 and stt._segments.empty()
    stt.resume()
    await feed(frames(0.6, 0.03) + frames(1.0, 0.0))   # écoute normale : pas de seuil
    await stt.stop()
    assert stt._segments.qsize() == 2   # le segment + le None de fin
