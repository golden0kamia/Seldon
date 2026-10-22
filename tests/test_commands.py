from typing import AsyncIterator

import pytest

from seldon.actions.speak import Speak
from seldon.brain.base import Brain
from seldon.core.commands import match, normalize
from seldon.core.pipeline import Assistant
from seldon.core.types import Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.base import STTEngine
from seldon.tts.base import TTSEngine


class ScriptedSTT(STTEngine):
    def __init__(self, lines):
        self.lines = lines

    async def listen(self) -> AsyncIterator[Utterance]:
        for line in self.lines:
            yield Utterance(line)


class EchoBrain(Brain):
    def __init__(self):
        self.seen, self.resets = [], 0

    async def handle(self, utterance, actions, execute):
        self.seen.append(utterance.text)
        return f"écho {utterance.text}"

    def reset(self):
        self.resets += 1


class RecordingTTS(TTSEngine):
    def __init__(self):
        self.said = []

    async def speak(self, text):
        self.said.append(text)


def assistant(lines, **kw):
    brain, tts = EchoBrain(), RecordingTTS()
    return Assistant(ScriptedSTT(lines), NoSpeakerId(), brain, tts, [Speak()], **kw), brain, tts


@pytest.mark.parametrize("text,name", [
    ("exit", "exit"), ("/exit", "exit"), ("  Quitter. ", "exit"), ("QUIT!", "exit"),
    ("/reset", "nouvelle"), ("Oublie tout", "nouvelle"), ("aide", "help"), ("/actions", "actions"),
    ("/muet", "muet"), ("Coupe le micro !", "muet"),
])
def test_match(text, name):
    assert match(text)[0].aliases[0] == name


@pytest.mark.parametrize("text", ["comment quitter vim ?", "exit le dossier", "aide-moi à coder", ""])
def test_no_match_in_normal_sentences(text):
    assert match(text) is None


def test_normalize():
    assert normalize("/Stop   Seldon !") == "stop seldon"


async def test_exit_stops_loop_before_next_utterance():
    a, brain, tts = assistant(["bonjour", "exit", "jamais traité"])
    await a.run()
    assert brain.seen == ["bonjour"]
    assert tts.said == ["écho bonjour", "Au revoir."]


async def test_reset_and_actions_skip_brain():
    a, brain, tts = assistant(["/reset", "actions"])
    await a.run()
    assert brain.seen == [] and brain.resets == 1
    assert tts.said == ["Conversation effacée.", "Actions : speak."]


async def test_commands_can_be_disabled():
    a, brain, _ = assistant(["exit", "suite"], enable_commands=False)
    await a.run()
    assert brain.seen == ["exit", "suite"]


class MutableSTT(ScriptedSTT):
    muted = False

    def mute(self, on):
        self.muted = on


async def test_mute_command_skips_brain():
    brain, tts, stt = EchoBrain(), RecordingTTS(), MutableSTT(["Coupe ton micro."])
    await Assistant(stt, NoSpeakerId(), brain, tts, [Speak()]).run()
    assert stt.muted and brain.seen == [] and "Micro coupé" in tts.said[0]


class ControlBrain(EchoBrain):
    """Appelle l'action controle_seldon avec l'opération dite (« fais quitter »), puis répond."""

    async def handle(self, utterance, actions, execute):
        self.seen.append(utterance.text)
        result = await execute("controle_seldon", {"operation": utterance.text.removeprefix("fais ")})
        return result.text


async def test_control_action_mutes_and_quits():
    from seldon.actions.control import Control

    brain, tts, stt = ControlBrain(), RecordingTTS(), MutableSTT(["fais couper_micro", "fais quitter", "jamais traité"])
    await Assistant(stt, NoSpeakerId(), brain, tts, [Speak(), Control()]).run()
    assert stt.muted
    assert brain.seen == ["fais couper_micro", "fais quitter"]
    assert "après ta réponse" in tts.said[-1]
