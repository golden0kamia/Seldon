from types import SimpleNamespace as NS

import pytest

from seldon.actions.base import Action
from seldon.actions.speak import Speak
from seldon.brain.ollama import OllamaBrain
from seldon.core.pipeline import Assistant
from seldon.core.types import ActionResult, Image, Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.text import TextInput
from seldon.tts.base import TTSEngine


def call(name, arguments):
    return NS(function=NS(name=name, arguments=arguments),
              model_dump=lambda: {"function": {"name": name, "arguments": arguments}})


def resp(content="", *calls):
    return NS(message=NS(content=content, tool_calls=list(calls) or None))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, **kwargs):
        # copie superficielle : l'historique est muté après coup
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def fake_brain(responses, **opts):
    brain = OllamaBrain(**opts)
    brain.client = FakeClient(responses)
    return brain


class RecordingTTS(TTSEngine):
    def __init__(self):
        self.said = []

    async def speak(self, text):
        self.said.append(text)


def assistant(brain, actions=()):
    tts = RecordingTTS()
    return Assistant(TextInput(), NoSpeakerId(), brain, tts, [Speak(), *actions]), tts


class Run(Action):
    name, description = "run", "action quelconque"
    input_schema = {"type": "object", "properties": {}}

    async def run(self, params, ctx):
        return ActionResult("ok")


class Cam(Action):
    name = "take_photo"
    description = "Prend une photo."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(self, params, ctx):
        return ActionResult(text="Photo prise.", images=[Image(b"\xff\xd8jpeg")])


async def test_tool_loop_and_speak():
    brain = fake_brain([resp("", call("speak", {"text": "Bonjour Bastien"}))])
    a, tts = assistant(brain)
    await a.process(Utterance("salut", speaker="Bastien"))
    assert tts.said == ["Bonjour Bastien"]
    first = brain.client.calls[0]
    assert first["messages"][0]["role"] == "system"
    assert "locuteur : Bastien" in first["messages"][1]["content"]
    assert first["tools"][0] == {
        "type": "function",
        "function": {"name": "speak", "description": Speak.description, "parameters": Speak.input_schema},
    }
    # speak termine le tour : pas de second appel au modèle
    assert [m["role"] for m in brain.messages] == ["user", "assistant", "tool"]
    assert len(brain.client.calls) == 1
    assert brain.messages[2]["tool_name"] == "speak"


async def test_plain_text_answer_is_spoken():
    a, tts = assistant(fake_brain([resp("Il est midi.")]))
    await a.process(Utterance("quelle heure ?", speaker="Bastien"))
    assert tts.said == ["Il est midi."]


async def test_string_arguments_and_unknown_tool():
    brain = fake_brain([
        resp("", call("speak", '{"text": "ok"}'), call("fly", {})),
        resp(""),
    ])
    a, tts = assistant(brain)
    await a.process(Utterance("vole", speaker="Bastien"))
    assert tts.said == ["ok"]
    unknown = brain.messages[3]
    assert unknown["role"] == "tool" and unknown["content"].startswith("ERREUR") and "speak" in unknown["content"]


@pytest.mark.parametrize("vision", [True, False])
async def test_images(vision):
    brain = fake_brain([resp("", call("take_photo", {})), resp("")], vision=vision)
    a, _ = assistant(brain, [Cam()])
    await a.process(Utterance("photo", speaker="Bastien"))
    msg = brain.messages[2]
    if vision:
        assert msg["images"] and "images" not in brain.messages[0]
    else:
        assert "images" not in msg and "ne peut pas les voir" in msg["content"]


async def test_error_rolls_back_history():
    class Boom(FakeClient):
        async def chat(self, **kwargs):
            raise RuntimeError("serveur tombé")

    brain = OllamaBrain()
    brain.client = Boom([])
    a, _ = assistant(brain)
    await a.process(Utterance("salut", speaker="Bastien"))
    assert brain.messages == []


async def test_max_steps():
    brain = fake_brain([resp("", call("run", {}))] * 2, max_steps=2)
    a, tts = assistant(brain, [Run()])
    await a.process(Utterance("boucle", speaker="Bastien"))
    assert len(brain.client.calls) == 2 and brain.messages[-1]["role"] == "tool"
    assert tts.said == ["Désolé, je n'ai pas réussi à terminer cette demande."]


async def test_bad_arguments_and_thinking():
    brain = fake_brain([resp("", call("speak", "pas du json")), resp("<think>poli</think>Bonjour !")])
    a, tts = assistant(brain)
    await a.process(Utterance("salut", speaker="Bastien"))
    assert brain.messages[2]["content"].startswith("ERREUR : Arguments illisibles")
    assert tts.said == ["Bonjour !"] and brain.messages[-1]["content"] == "Bonjour !"


async def test_only_last_image_kept():
    brain = fake_brain([resp("", call("take_photo", {})), resp("")] * 2, vision=True)
    a, _ = assistant(brain, [Cam()])
    await a.process(Utterance("photo", speaker="Bastien"))
    await a.process(Utterance("encore", speaker="Bastien"))
    with_images = [m for m in brain.messages if m.get("images")]
    assert len(with_images) == 1 and with_images[0] is [m for m in brain.messages if m["role"] == "tool"][-1]


async def test_warmup_loads_model_with_tools():
    brain = fake_brain([resp("")])
    await brain.warmup([Speak()])
    sent = brain.client.calls[0]
    assert sent["options"]["num_predict"] == 1 and sent["tools"][0]["function"]["name"] == "speak"
