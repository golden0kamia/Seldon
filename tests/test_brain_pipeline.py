from types import SimpleNamespace as NS

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from seldon.actions.speak import Speak
from seldon.brain.claude import ClaudeBrain
from seldon.core.pipeline import Assistant
from seldon.core.types import Image, Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.text import TextInput
from seldon.tts.base import TTSEngine


def text(t):
    return TextBlock(type="text", text=t)


def tool(id_, name, **inp):
    return ToolUseBlock(type="tool_use", id=id_, name=name, input=inp)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        # copie superficielle : l'historique est muté après coup
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


def fake_brain(responses, **opts):
    brain = ClaudeBrain(fallbacks=False, **opts)
    brain.client = NS(messages=FakeMessages(responses))
    return brain


def resp(stop, *blocks):
    return NS(stop_reason=stop, stop_details=None, content=list(blocks))


class RecordingTTS(TTSEngine):
    def __init__(self):
        self.said = []

    async def speak(self, text):
        self.said.append(text)


def assistant(brain, actions=()):
    tts = RecordingTTS()
    return Assistant(TextInput(), NoSpeakerId(), brain, tts, [Speak(), *actions]), tts


async def test_tool_loop_and_speak():
    brain = fake_brain([
        resp("tool_use", text("…"), tool("t1", "speak", text="Bonjour Bastien")),
    ])
    a, tts = assistant(brain)
    await a.process(Utterance("salut", speaker="Bastien"))
    assert tts.said == ["Bonjour Bastien"]
    first = brain.client.messages.calls[0]
    assert "locuteur : Bastien" in first["messages"][0]["content"][-1]["text"]
    assert first["output_config"] == {"effort": "low"}
    # speak termine le tour ; son résultat part avec le message suivant
    assert len(brain.client.messages.calls) == 1
    assert [m["role"] for m in brain.messages] == ["user", "assistant"]
    assert brain._pending[0]["tool_use_id"] == "t1"


async def test_final_text_spoken_when_no_speak_call():
    a, tts = assistant(fake_brain([resp("end_turn", text("Il est midi."))]))
    await a.process(Utterance("quelle heure ?"))
    assert tts.said == ["Il est midi."]


async def test_unknown_action_and_images_in_tool_result():
    from seldon.actions.base import Action
    from seldon.core.types import ActionResult

    class Cam(Action):
        name, description = "capture_image", "photo"
        input_schema = {"type": "object", "properties": {}}

        async def run(self, params, ctx):
            return ActionResult("photo prise", images=[Image(b"\xff\xd8jpeg")])

    brain = fake_brain([
        resp("tool_use", tool("a", "capture_image"), tool("b", "nope")),
        resp("end_turn", text("Je vois un chat.")),
    ])
    a, tts = assistant(brain, [Cam()])
    await a.process(Utterance("que vois-tu ?"))
    results = brain.messages[2]["content"]
    assert results[0]["content"][0]["type"] == "image"
    assert results[1]["is_error"] is True
    assert tts.said == ["Je vois un chat."]



def slow_action():
    from seldon.actions.base import Action
    from seldon.core.types import ActionResult

    class Slow(Action):
        name, description = "slow", "tâche longue"
        input_schema = {"type": "object", "properties": {}}
        acknowledgement = "Compris, je m'en occupe."

        async def run(self, params, ctx):
            return ActionResult("fait")

    return Slow()


async def test_action_acknowledged_once_before_answer():
    brain = fake_brain([
        resp("tool_use", tool("a", "slow"), tool("b", "slow")),
        resp("tool_use", tool("c", "speak", text="C'est fait.")),
    ])
    a, tts = assistant(brain, [slow_action()])
    await a.process(Utterance("fais-le"))
    assert tts.said == ["Compris, je m'en occupe.", "C'est fait."]


async def test_no_acknowledgement_if_already_spoken_or_disabled():
    brain = fake_brain([
        # le cerveau a déjà prévenu : pas d'accusé de réception en plus
        resp("tool_use", tool("a", "speak", text="Je lance ça."), tool("b", "slow")),
        resp("end_turn", text("Terminé.")),
    ])
    a, tts = assistant(brain, [slow_action()])
    await a.process(Utterance("fais-le"))
    assert "Compris, je m'en occupe." not in tts.said

    a, tts = assistant(fake_brain([resp("tool_use", tool("a", "slow")), resp("end_turn", text("Fait."))]),
                       [slow_action()])
    a.acknowledge = False
    await a.process(Utterance("fais-le"))
    assert tts.said == ["Fait."]

async def test_step_limit_carries_pending_results():
    brain = fake_brain(
        [resp("tool_use", tool("x", "speak", text="un")), resp("end_turn")], max_steps=1
    )
    a, _ = assistant(brain)
    await a.process(Utterance("boucle"))
    assert brain.messages[-1]["role"] == "assistant"
    await a.process(Utterance("suite"))
    sent = brain.client.messages.calls[1]["messages"][-1]["content"]
    assert sent[0]["type"] == "tool_result" and sent[-1]["type"] == "text"


async def test_api_error_rolls_back_history():
    class Boom(FakeMessages):
        async def create(self, **kwargs):
            raise RuntimeError("réseau")

    brain = ClaudeBrain(fallbacks=False)
    brain.client = NS(messages=Boom([]))
    a, tts = assistant(brain)
    await a.process(Utterance("hello"))
    assert brain.messages == []
    assert tts.said == ["Désolé, j'ai rencontré une erreur."]


def test_registry_and_config(tmp_path):
    from seldon.config import Config
    from seldon.core import registry
    from seldon.core.errors import SeldonError

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("stt: text\nbrain: {engine: claude, options: {effort: medium}}\nactions: [speak]\n")
    cfg = Config.load(cfg_file)
    a = Assistant.from_config(cfg)
    assert a.brain.effort == "medium" and list(a.actions) == ["speak"]
    with pytest.raises(SeldonError, match="moteurs disponibles"):
        registry.resolve("tts", "inexistant")
