"""Tests des moteurs TTS sans carte son : sounddevice, Piper et pyttsx3 sont simulés."""

from __future__ import annotations

import asyncio
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest

from seldon.core.registry import create
from seldon.tts.piper import PiperTTS
from seldon.tts.pyttsx3_tts import Pyttsx3TTS, find_voice


class FakeSD(types.ModuleType):
    """Remplace sounddevice : ``play`` enregistre, ``wait`` bloque jusqu'à ``release``/``stop``."""

    def __init__(self, block: bool = False) -> None:
        super().__init__("sounddevice")
        self.played: list[tuple[np.ndarray, int]] = []
        self.done = threading.Event()
        if not block:
            self.done.set()
        self.started = threading.Event()

    def play(self, data, samplerate, device=None):
        self.played.append((data, samplerate))
        self.started.set()

    def wait(self):
        if self.played:  # comme sounddevice : ne bloque que si un son est en cours
            self.done.wait(5)

    def stop(self):
        self.done.set()


@pytest.fixture
def fake_sd(monkeypatch):
    sd = FakeSD()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd


def chunk(value: float) -> SimpleNamespace:
    return SimpleNamespace(audio_float_array=np.full(10, value, dtype=np.float32), sample_rate=22050)


class FakeVoice:
    def __init__(self, n: int = 3) -> None:
        self.n, self.calls = n, []

    def synthesize(self, text, syn_config=None):
        self.calls.append((text, syn_config))
        for i in range(self.n):
            yield chunk(float(i))


def make_piper(voice: FakeVoice, **opts) -> PiperTTS:
    tts = PiperTTS(**opts)
    tts._voice = voice
    return tts


def test_registry_builds_plugins():
    assert isinstance(create("tts", "piper", voice="fr_FR-upmc-medium"), PiperTTS)
    assert isinstance(create("tts", "pyttsx3", rate=150), Pyttsx3TTS)


async def test_piper_plays_every_sentence(fake_sd):
    voice = FakeVoice(3)
    tts = make_piper(voice, speaker_id=1, speed=1.25)
    await tts.speak("Bonjour. Comment ça va ? Très bien.")
    assert [float(d[0]) for d, _ in fake_sd.played] == [0.0, 1.0, 2.0]
    assert all(sr == 22050 for _, sr in fake_sd.played)
    cfg = voice.calls[0][1]
    assert cfg.speaker_id == 1 and cfg.length_scale == pytest.approx(0.8)


async def test_piper_empty_text_is_noop(fake_sd):
    voice = FakeVoice()
    await make_piper(voice).speak("   ")
    assert voice.calls == [] and fake_sd.played == []


async def test_piper_stop_interrupts(monkeypatch):
    sd = FakeSD(block=True)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    tts = make_piper(FakeVoice(5))
    task = asyncio.create_task(tts.speak("Un long texte."))
    await asyncio.to_thread(sd.started.wait, 5)
    await tts.stop()
    await asyncio.wait_for(task, 5)
    assert len(sd.played) == 1  # les phrases suivantes ne sont pas jouées
    # un nouvel appel fonctionne après l'interruption
    sd.done.set()
    await tts.speak("Encore.")
    assert len(sd.played) == 6


async def test_piper_downloads_missing_voice(tmp_path, monkeypatch):
    import piper
    import piper.download_voices as dv

    downloaded = []

    def fake_download(name, directory):
        downloaded.append((name, directory))
        (directory / f"{name}.onnx").write_bytes(b"x")
        (directory / f"{name}.onnx.json").write_text("{}")

    monkeypatch.setattr(dv, "download_voice", fake_download)
    monkeypatch.setattr(piper.PiperVoice, "load", staticmethod(lambda path: ("voix", path)))
    tts = PiperTTS(voices_dir=str(tmp_path / "voices"))
    await tts.start()
    assert downloaded == [("fr_FR-siwis-medium", tmp_path / "voices")]
    assert tts._voice == ("voix", tmp_path / "voices" / "fr_FR-siwis-medium.onnx")
    # déjà présente : pas de second téléchargement
    await PiperTTS(voices_dir=str(tmp_path / "voices")).start()
    assert len(downloaded) == 1


async def test_piper_missing_voice_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        await PiperTTS(voices_dir=str(tmp_path), download=False).start()


VOICES = [
    SimpleNamespace(id="gmw/af", name="Afrikaans", languages=["af"]),
    SimpleNamespace(id="roa/fr-be", name="French (Belgium)", languages=["fr-be"]),
    SimpleNamespace(id="roa/fr", name="French (France)", languages=["fr-fr"]),
    SimpleNamespace(id="gmw/en", name="English", languages=[b"\x05en-gb"]),
]


@pytest.mark.parametrize(
    "wanted, expected",
    [("fr", "roa/fr"), ("fr-be", "roa/fr-be"), ("roa/fr", "roa/fr"), ("en", "gmw/en"),
     ("english", "gmw/en"), ("klingon", None)],
)
def test_find_voice(wanted, expected):
    v = find_voice(VOICES, wanted)
    assert (v.id if v else None) == expected


class FakeEngine:
    def __init__(self) -> None:
        self.props, self.said, self.stopped = {}, [], False

    def getProperty(self, name):
        return VOICES if name == "voices" else self.props.get(name)

    def setProperty(self, name, value):
        self.props[name] = value

    def say(self, text):
        self.said.append(text)

    def runAndWait(self):
        pass

    def stop(self):
        self.stopped = True


async def test_pyttsx3_speak(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setitem(sys.modules, "pyttsx3", SimpleNamespace(init=lambda: engine))
    tts = Pyttsx3TTS(voice="fr", rate=150, volume=0.5)
    await tts.speak("Bonjour")
    assert engine.said == ["Bonjour"]
    assert engine.props == {"voice": "roa/fr", "rate": 150, "volume": 0.5}
    await tts.stop()
    assert engine.stopped


async def test_piper_cancelled_speak_stops_playback(monkeypatch):
    sd = FakeSD(block=True)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    tts = make_piper(FakeVoice(5))
    task = asyncio.create_task(tts.speak("Un long texte."))
    await asyncio.to_thread(sd.started.wait, 5)
    task.cancel()   # ex. tour annulé à l'arrêt : le thread de lecture ne doit pas continuer seul
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sd.done.is_set() and tts._stop.is_set()
    await asyncio.sleep(0.1)
    assert len(sd.played) == 1
