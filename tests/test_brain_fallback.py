import json

import httpx
import pytest

from seldon.brain.base import Brain
from seldon.brain.fallback import FallbackBrain
from seldon.brain.openai import OpenAIBrain
from seldon.config import Config
from seldon.core.errors import SeldonError
from seldon.core.pipeline import Assistant
from seldon.core.types import Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.text import TextInput
from seldon.tools.speak import Speak
from seldon.tts.base import TTSEngine


class RecordingTTS(TTSEngine):
    def __init__(self):
        self.said = []

    async def speak(self, text):
        self.said.append(text)


class FakeBrain(Brain):
    def __init__(self, say="", fail_start=False, fail=False, fail_after_action=False):
        self.say, self.fail_start, self.fail, self.fail_after_action = say, fail_start, fail, fail_after_action
        self.messages = []
        self.starts = self.calls = 0

    async def start(self):
        self.starts += 1
        if self.fail_start:
            raise SeldonError("serveur injoignable")

    async def handle(self, utterance, actions, execute):
        self.calls += 1
        if self.fail:
            raise ConnectionError("hors ligne")
        await execute("speak", {"text": self.say})
        if self.fail_after_action:
            raise ConnectionError("coupé en route")
        self.messages.append({"role": "user", "content": utterance.text})
        self.messages.append({"role": "assistant", "content": self.say})
        return None


def assistant(brain):
    tts = RecordingTTS()
    return Assistant(TextInput(), NoSpeakerId(), brain, tts, [Speak()]), tts


async def test_primary_answers_and_backup_gets_the_exchange():
    primary, backup = FakeBrain("Bonjour"), FakeBrain("Secours")
    brain = FallbackBrain(primary, backup)
    a, tts = assistant(brain)
    await brain.start()
    await a.process(Utterance("salut", speaker="Bastien"))
    assert tts.said == ["Bonjour"] and backup.calls == 0
    assert backup.messages[-1] == {"role": "assistant", "content": "Bonjour"}
    assert "salut" in backup.messages[-2]["content"]


async def test_backup_takes_over_when_primary_fails_then_primary_is_retried_later():
    primary, backup = FakeBrain("Bonjour", fail=True), FakeBrain("Secours")
    brain = FallbackBrain(primary, backup, retry_primary_s=0)
    a, tts = assistant(brain)
    await brain.start()
    await a.process(Utterance("salut"))
    assert tts.said == ["Secours"] and primary.calls == 1
    assert primary.messages[-1]["content"] == "Secours"   # le principal garde le fil
    primary.fail = False
    await a.process(Utterance("encore"))
    assert tts.said[-1] == "Bonjour"


async def test_primary_down_at_start_is_skipped_until_retry():
    primary, backup = FakeBrain("Bonjour", fail_start=True), FakeBrain("Secours")
    brain = FallbackBrain(primary, backup)
    a, tts = assistant(brain)
    await brain.start()
    await a.process(Utterance("salut"))
    await a.process(Utterance("encore"))
    assert tts.said == ["Secours", "Secours"] and primary.calls == 0


async def test_no_replay_after_an_action_and_both_down_raises():
    primary, backup = FakeBrain("Bonjour", fail_after_action=True), FakeBrain("Secours")
    brain = FallbackBrain(primary, backup)
    await brain.start()
    said = []

    async def execute(name, params):
        said.append(params["text"])

    with pytest.raises(ConnectionError):
        await brain.handle(Utterance("salut"), [Speak()], execute)
    assert said == ["Bonjour"] and backup.calls == 0

    with pytest.raises(SeldonError, match="Aucun cerveau"):
        await FallbackBrain(FakeBrain(fail_start=True), FakeBrain(fail_start=True)).start()


def test_state_round_trip_and_plain_transcript():
    primary, backup = FakeBrain(), FakeBrain()
    brain = FallbackBrain(primary, backup)
    primary.messages = [{"role": "user", "content": "a"}]
    backup.messages = [{"role": "user", "content": "b"}]
    state = json.loads(json.dumps(brain.get_state()))
    brain.reset()
    brain.set_state(state)
    assert primary.messages[0]["content"] == "a" and backup.messages[0]["content"] == "b"
    brain.set_state({"messages": [{"role": "user", "content": "t"}]})
    assert primary.messages == backup.messages == [{"role": "user", "content": "t"}]


def test_config_backup_section_and_pipeline():
    cfg = Config.parse("brain:\n  engine: llamacpp\n  backup:\n    engine: openai\n    options: {model: gpt-x}\n")
    assert cfg.brain.engine == "llamacpp" and cfg.brain_backup.options == {"model": "gpt-x"}
    assert Config.parse("brain: claude").brain_backup is None
    a = Assistant.from_config(Config.parse(
        "stt: text\ntts: console\nmemory: {enabled: false}\n"
        "brain:\n  engine: llamacpp\n  backup: {engine: openai, options: {api_key: k}}\n"))
    assert isinstance(a.brain, FallbackBrain) and isinstance(a.brain.backup, OpenAIBrain)
    with pytest.raises(SeldonError, match="inattendue"):
        Config.parse("brain:\n  engine: claude\n  secours: openai\n")


async def test_openai_request_body(monkeypatch):
    bodies = []

    def handler(request):
        assert request.headers["authorization"] == "Bearer sk-test"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "gpt-x"}]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "Salut"}}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(*a, transport=httpx.MockTransport(handler), **k))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    brain = OpenAIBrain(model="gpt-x", stream=False)
    await brain.start()
    a, tts = assistant(brain)
    await a.process(Utterance("salut"))
    assert tts.said == ["Salut"]
    body = bodies[0]
    assert body["model"] == "gpt-x" and body["reasoning_effort"] == "low" and "max_completion_tokens" in body
    assert "max_tokens" not in body and "chat_template_kwargs" not in body


async def test_openai_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SeldonError, match="OPENAI_API_KEY"):
        await OpenAIBrain().start()
