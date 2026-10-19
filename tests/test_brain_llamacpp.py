import json

import httpx
import pytest

from seldon.actions.base import Action
from seldon.actions.speak import Speak
from seldon.brain.llamacpp import LlamaCppBrain
from seldon.core.pipeline import Assistant
from seldon.core.types import ActionResult, Image, Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.text import TextInput
from seldon.tts.base import TTSEngine

MODELS = {"data": [
    {"id": "a", "status": {"value": "unloaded"}},
    {"id": "b", "status": {"value": "loaded"}},
]}


def call(id_, name, args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def msg(content="", *calls):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": list(calls) or None}}]}


class RecordingTTS(TTSEngine):
    def __init__(self):
        self.said = []

    async def speak(self, text):
        self.said.append(text)


class Cam(Action):
    name, description = "take_photo", "Prend une photo."
    input_schema = {"type": "object", "properties": {}}

    async def run(self, params, ctx):
        return ActionResult("Photo prise.", images=[Image(b"\xff\xd8")])


async def make(replies, actions=(), **opts):
    bodies = []

    def handler(request: httpx.Request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=MODELS)
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))

    brain = LlamaCppBrain(**opts)
    brain._transport = httpx.MockTransport(handler)  # pris en compte par start() (voir fixture)
    await brain.start()
    tts = RecordingTTS()
    return Assistant(TextInput(), NoSpeakerId(), brain, tts, [Speak(), *actions]), brain, tts, bodies


@pytest.fixture(autouse=True)
def mock_transport(monkeypatch):
    """Les clients httpx créés par start() utilisent le transport factice du cerveau."""
    orig = LlamaCppBrain.start

    async def start(self):
        transport = getattr(self, "_transport", httpx.MockTransport(lambda r: httpx.Response(200, json=MODELS)))
        real = httpx.AsyncClient

        def client(*args, **kwargs):
            return real(*args, transport=transport, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", client)
        try:
            await orig(self)
        finally:
            monkeypatch.setattr(httpx, "AsyncClient", real)

    monkeypatch.setattr(LlamaCppBrain, "start", start)


async def test_picks_loaded_model_and_speak_ends_turn():
    a, brain, tts, bodies = await make([msg("", call("c1", "speak", {"text": "Salut"}))])
    assert brain.model == "b"
    await a.process(Utterance("salut", speaker="Bastien"))
    assert tts.said == ["Salut"]
    assert len(bodies) == 1
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert bodies[0]["tools"][0]["function"]["name"] == "speak"
    assert [m["role"] for m in brain.messages] == ["user", "assistant", "tool"]
    assert brain.messages[2]["tool_call_id"] == "c1"


async def test_images_with_vision_follow_in_user_message():
    a, brain, tts, bodies = await make(
        [msg("", call("c1", "take_photo", {})), msg("Je vois un chat.")], [Cam()], vision=True
    )
    await a.process(Utterance("que vois-tu ?"))
    sent = bodies[1]["messages"]
    assert sent[-2]["role"] == "tool" and sent[-1]["content"][1]["type"] == "image_url"
    assert tts.said == ["Je vois un chat."]


async def test_unknown_model_raises():
    brain = LlamaCppBrain(model="zzz")
    with pytest.raises(RuntimeError, match="absent"):
        await brain.start()
