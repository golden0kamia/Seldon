"""Comportements « humains » : accueil après une absence, attention, coupure de parole."""

from __future__ import annotations

import asyncio
import time

from seldon.brain.prompts import user_message
from seldon.core.memory import ConversationMemory, _duration
from seldon.core.pipeline import Assistant
from seldon.core.types import ActionContext, Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.base import STTEngine
from seldon.tools.speak import Speak
from seldon.tts.base import TTSEngine
from tests.test_memory import EchoBrain, RecordingTTS


class ListSTT(STTEngine):
    def __init__(self, utterances):
        self.utterances = utterances

    async def listen(self):
        for u in self.utterances:
            yield u


def assistant(tmp_path, utterances, follow_up_s=20, memory=True):
    brain = EchoBrain()
    mem = ConversationMemory(brain, "echo", dir=str(tmp_path)) if memory else None
    seen = []
    original = brain.handle

    async def handle(utterance, actions, execute):
        seen.append(utterance)
        return await original(utterance, actions, execute)

    brain.handle = handle
    tts = RecordingTTS()
    a = Assistant(ListSTT(utterances), NoSpeakerId(), brain, tts, [Speak()], memory=mem, follow_up_s=follow_up_s)
    return a, seen, tts, mem


# -- accueil ------------------------------------------------------------------------

def test_duration():
    assert _duration(3 * 3600) == "3 heures"
    assert _duration(86400 * 2.2) == "2 jours"
    assert _duration(86400 * 21) == "3 semaines"
    assert _duration(86400 * 95) == "3 mois"


async def test_absence_note(tmp_path):
    a, seen, _, mem = assistant(tmp_path, [Utterance("bonjour", speaker="Paul"), Utterance("ça va ?", speaker="Paul")])
    await a.run()
    assert "première conversation avec Paul" in seen[0].notes[0]
    assert seen[1].notes == []   # vient de parler
    later = time.time() + 3 * 86400
    note = mem.absence_note("Paul", later)
    assert "3 jours" in note and "« bonjour »" in note
    # reconstruit depuis le disque
    fresh = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    assert fresh.absence_note("Paul", later) == note
    assert fresh.absence_note("Paul") is None and fresh.absence_note(None) is None
    fresh.greet_after_s = None
    assert fresh.absence_note("Paul", later) is None


def test_user_message_includes_notes():
    msg = user_message(Utterance("salut", speaker="Paul", notes=["une note"]))
    assert msg.splitlines()[1:] == ["[note : une note]", "salut"] and "locuteur : Paul" in msg


# -- attention (mots d'éveil) -----------------------------------------------------------

async def test_follow_up_window_and_overheard(tmp_path):
    now = time.time()
    a, seen, tts, _ = assistant(tmp_path, [
        Utterance("on mange quoi ?", speaker="Paul", addressed=False, timestamp=now),   # ignoré, retenu
        Utterance("des pâtes", speaker="Bastien", addressed=False, timestamp=now),
        Utterance("quelle heure est-il ?", speaker="Bastien", addressed=True),
        Utterance("et demain ?", speaker="Bastien", addressed=False),     # dans la foulée
    ], memory=False)
    await a.run()
    assert [u.text for u in seen] == ["quelle heure est-il ?", "et demain ?"]
    assert "Paul : « on mange quoi ? » ; Bastien : « des pâtes »" in seen[0].notes[0]
    assert "sans mot d'éveil" in seen[1].notes[0] and len(seen[1].notes) == 1
    assert len(tts.said) == 2


async def test_without_follow_up_unaddressed_is_ignored(tmp_path):
    a, seen, _, _ = assistant(tmp_path, [
        Utterance("Seldon", addressed=True), Utterance("et demain ?", addressed=False),
    ], follow_up_s=0, memory=False)
    await a.run()
    assert [u.text for u in seen] == ["Seldon"]


# -- coupure de parole ------------------------------------------------------------------

class SlowTTS(TTSEngine):
    def __init__(self):
        self.stopped = asyncio.Event()

    async def speak(self, text):
        await asyncio.wait_for(self.stopped.wait(), 2)

    async def stop(self):
        self.stopped.set()


class InterruptingSTT(STTEngine):
    def __init__(self, phrase=None):
        self.phrase, self.paused = phrase, []

    async def listen(self):
        yield

    def pause(self):
        self.paused.append(True)

    def resume(self):
        self.paused.append(False)

    async def interruption(self, spoken):
        if self.phrase is None:
            return await super().interruption(spoken)
        await asyncio.sleep(0.01)
        return Utterance(self.phrase)


async def test_speak_interrupted():
    tts, stt = SlowTTS(), InterruptingSTT("stop")
    r = await Speak().run({"text": "une longue phrase"}, ActionContext(services={"tts": tts, "stt": stt}))
    assert not r.is_error and "Interrompu" in r.text and "« stop »" in r.text
    assert tts.stopped.is_set() and stt.paused == [True, False]


async def test_speak_not_interrupted():
    tts, stt = RecordingTTS(), InterruptingSTT()
    r = await Speak().run({"text": "bonjour"}, ActionContext(services={"tts": tts, "stt": stt}))
    assert r.text == "Prononcé." and tts.said == ["bonjour"] and stt.paused == [True, False]
