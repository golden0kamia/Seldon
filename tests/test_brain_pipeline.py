from types import SimpleNamespace as NS

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from seldon.brain.claude import ClaudeBrain
from seldon.core.pipeline import MUTE_SIGNALS, Assistant
from seldon.core.types import Image, Utterance
from seldon.speakers.base import NoSpeakerId
from seldon.stt.text import TextInput
from seldon.tools.speak import Speak
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
    from seldon.core.action import Action
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
    from seldon.core.action import Action
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


@pytest.mark.skipif(not MUTE_SIGNALS, reason="signaux POSIX")
async def test_mute_signals_reach_stt_while_running():
    import asyncio
    import os
    import signal

    class MutableInput(TextInput):
        def __init__(self):
            self.muted = []

        def mute(self, on):
            self.muted.append(on)

        async def listen(self):
            for sig in (signal.SIGUSR1, signal.SIGUSR2):
                os.kill(os.getpid(), sig)
                await asyncio.sleep(0.05)
            return
            yield

    stt = MutableInput()
    await Assistant(stt, NoSpeakerId(), fake_brain([]), RecordingTTS(), [Speak()]).run()
    assert stt.muted == [True, False]
    assert signal.getsignal(signal.SIGUSR1) is signal.SIG_DFL   # gestionnaires retirés à l'arrêt


def failing_action():
    from seldon.core.action import Action

    class Boom(Action):
        name, description = "boom", "échoue"
        input_schema = {"type": "object", "properties": {}}

        async def run(self, params, ctx):
            raise ConnectionRefusedError("refusé")

    return Boom()


async def test_failed_action_announced_if_brain_stays_silent():
    brain = fake_brain([resp("tool_use", tool("a", "boom")), resp("end_turn")])
    a, tts = assistant(brain, [failing_action()])
    await a.process(Utterance("fais boum"))
    assert tts.said == ["Désolé, l'action boom a échoué."]


async def test_brain_error_is_spoken():
    brain = fake_brain([])
    brain.client.messages.create = None   # TypeError à l'appel
    a, tts = assistant(brain)
    await a.process(Utterance("salut"))
    assert tts.said == ["Désolé, j'ai rencontré une erreur."]


async def test_pipeline_error_is_spoken():
    class OneLine(TextInput):
        async def listen(self):
            yield Utterance("salut")

    tts = RecordingTTS()
    a = Assistant(OneLine(), NoSpeakerId(), fake_brain([]), tts, [Speak()])

    async def nothing():
        pass

    async def broken(_):
        raise ConnectionRefusedError()

    a.start, a.process = nothing, broken
    await a.run()
    assert tts.said == ["Désolé, je n'arrive pas à joindre un service dont j'ai besoin."]


# -- robustesse du pipeline --------------------------------------------------------------

async def test_truncated_tool_use_is_not_kept_in_history():
    brain = fake_brain([resp("max_tokens", text("Je"), tool("t1", "speak", text="coupé")),
                        resp("end_turn", text("Suite."))])
    a, tts = assistant(brain)
    await a.process(Utterance("parle"))
    assert tts.said == ["Je"]
    assert all(b["type"] != "tool_use" for m in brain.messages if m["role"] == "assistant" for b in m["content"])
    await a.process(Utterance("encore"))   # requête suivante valide : pas de tool_use sans résultat
    assert tts.said[-1] == "Suite."


async def test_memory_failure_does_not_mask_the_answer(tmp_path):
    from seldon.core.memory import ConversationMemory

    class BrokenMemory(ConversationMemory):
        def end_turn(self):
            raise OSError(28, "No space left on device")

    brain = fake_brain([resp("tool_use", tool("t1", "speak", text="Voilà."))])
    tts = RecordingTTS()
    a = Assistant(TextInput(), NoSpeakerId(), brain, tts, [Speak()],
                  memory=BrokenMemory(brain, "claude", dir=str(tmp_path)))
    errors = []
    a.events.on("error", lambda error, stage: errors.append(stage))
    await a.process(Utterance("salut"))
    assert tts.said == ["Voilà."] and errors == ["memory"]


async def test_failing_acknowledgement_does_not_break_the_action():
    from seldon.core.action import Action
    from seldon.core.types import ActionResult

    class Odd(Action):
        name, description = "odd", "accusé de réception défaillant"
        input_schema = {"type": "object", "properties": {}}

        def acknowledge(self, params):
            raise KeyError("oups")

        async def run(self, params, ctx):
            return ActionResult("fait")

    brain = fake_brain([resp("tool_use", tool("a", "odd")), resp("end_turn", text("Fait."))])
    a, tts = assistant(brain, [Odd()])
    await a.process(Utterance("vas-y"))
    assert tts.said == ["Fait."]


async def test_command_reply_pauses_listening():
    class PausingInput(TextInput):
        def __init__(self):
            self.calls = []

        def pause(self):
            self.calls.append("pause")

        def resume(self):
            self.calls.append("resume")

    stt, tts = PausingInput(), RecordingTTS()
    a = Assistant(stt, NoSpeakerId(), fake_brain([]), tts, [Speak()])
    await a.process(Utterance("/actions"))
    assert tts.said and stt.calls == ["pause", "resume"]


@pytest.mark.skipif(not MUTE_SIGNALS, reason="signaux POSIX")
async def test_sigterm_stops_cleanly_and_previous_handlers_are_restored():
    import asyncio
    import os
    import signal

    stopped = []

    class Waiting(TextInput):
        async def listen(self):
            os.kill(os.getpid(), signal.SIGTERM)
            await asyncio.Event().wait()
            yield

        async def stop(self):
            stopped.append(True)

    signal.signal(signal.SIGUSR1, signal.SIG_IGN)   # comme __main__ avant le démarrage
    try:
        await asyncio.wait_for(Assistant(Waiting(), NoSpeakerId(), fake_brain([]), RecordingTTS(), [Speak()]).run(), 2)
        assert stopped == [True]
        assert signal.getsignal(signal.SIGUSR1) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    finally:
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)


async def test_brain_is_warmed_up_in_background():
    import asyncio

    warmed = asyncio.Event()

    class Waiting(TextInput):
        async def listen(self):
            await asyncio.wait_for(warmed.wait(), 1)
            return
            yield

    brain = fake_brain([])

    async def warmup(actions):
        assert [x.name for x in actions] == ["speak"]
        warmed.set()

    brain.warmup = warmup
    await Assistant(Waiting(), NoSpeakerId(), brain, RecordingTTS(), [Speak()]).run()
    assert warmed.is_set()


def test_memory_options_are_checked():
    from seldon.config import Config
    from seldon.core.errors import SeldonError

    for memory, match in [("{compress_every_s: '30 min'}", "nombre"), ("{dirr: x}", "dir"),
                          ("{enabled: 'non'}", "true ou false")]:
        cfg = Config.parse(f"stt: text\nbrain: claude\nmemory: {memory}\nactions: [speak]\n")
        with pytest.raises(SeldonError, match=match):
            Assistant.from_config(cfg)


def test_pending_keyboard_read_does_not_block_shutdown(monkeypatch):
    import asyncio
    import threading
    import time

    from seldon.core.pipeline import console_confirm

    release = threading.Event()
    monkeypatch.setattr("builtins.input", lambda *a: (release.wait(5), "o")[1])

    async def main():
        task = asyncio.create_task(console_confirm("Lancer ?"))
        await asyncio.sleep(0.05)
        task.cancel()   # arrêt de Seldon pendant la question

    started = time.perf_counter()
    asyncio.run(main())
    assert time.perf_counter() - started < 1   # avec asyncio.to_thread(input), l'arrêt attendait la réponse
    release.set()


async def test_console_confirm_answers(monkeypatch):
    from seldon.core.pipeline import console_confirm

    answers = iter(["Oui", "non"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert await console_confirm("A ?") is True and await console_confirm("B ?") is False

    def eof(*a):
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert await console_confirm("C ?") is False
