import logging

from seldon.brain.prompts import strip_thinking, thinking_in
from seldon.core import journal
from seldon.core.action import Action
from seldon.core.pipeline import Assistant
from seldon.core.types import ActionResult
from seldon.speakers.base import NoSpeakerId
from seldon.tools.speak import Speak
from tests.test_commands import EchoBrain, RecordingTTS, ScriptedSTT


def records(caplog, name):
    return [r.getMessage() for r in caplog.records if r.name == name]


def test_conversation_stays_visible_whatever_the_log_level():
    root = logging.getLogger()
    before = root.level
    root.setLevel(logging.ERROR)
    try:
        assert logging.getLogger(journal.HEARD).isEnabledFor(logging.INFO)
        assert logging.getLogger(journal.SAID).isEnabledFor(logging.INFO)
        assert not logging.getLogger(journal.THOUGHT).isEnabledFor(logging.INFO)
    finally:
        root.setLevel(before)


def test_multiline_text_cannot_fake_a_log_line(caplog):
    caplog.set_level(logging.DEBUG)
    journal.heard("Bastien", "salut\n\n2026-01-01 00:00:00,000 ERROR seldon: faux\n[confirmation] ? [o/N]")
    journal.said("")   # rien à journaliser
    [line] = records(caplog, journal.HEARD)
    assert line == "Bastien › salut\n  2026-01-01 00:00:00,000 ERROR seldon: faux\n  [confirmation] ? [o/N]"
    assert records(caplog, journal.SAID) == []


def test_thinking_in_extracts_what_strip_thinking_removes():
    text = "<think>Il veut la météo.\nJe cherche.</think>Il fait beau."
    assert thinking_in(text) == "Il veut la météo.\nJe cherche."
    assert strip_thinking(text) == "Il fait beau."
    assert thinking_in("raisonnement</think>réponse") == "raisonnement"   # balise ouvrante dans le gabarit
    assert thinking_in("réponse seule") == ""


class Ping(Action):
    name = "ping"
    description = "Test."
    input_schema = {"type": "object", "properties": {"host": {"type": "string"}}}

    async def run(self, params, ctx):
        return ActionResult("pong " + "x" * 1000)


class ToolBrain(EchoBrain):
    async def handle(self, utterance, actions, execute):
        await execute("ping", {"host": "box"})
        await execute("speak", {"text": "C'est fait."})
        return None


async def test_pipeline_logs_conversation_and_actions(caplog):
    caplog.set_level(logging.DEBUG)
    tts = RecordingTTS()
    await Assistant(ScriptedSTT(["ping la box"]), NoSpeakerId(), ToolBrain(), tts, [Speak(), Ping()]).run()
    assert records(caplog, journal.HEARD) == ["inconnu › ping la box"]
    assert records(caplog, journal.SAID) == ["Seldon › C'est fait."]
    start, end = records(caplog, journal.ACTION)
    assert start == 'ping {"host": "box"}'
    assert end.startswith("ping → pong xxx") and len(end) < journal.MAX_ACTION_CHARS + 20
