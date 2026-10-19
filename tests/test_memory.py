import json
from typing import AsyncIterator

from seldon.actions.conversations import Conversations
from seldon.actions.speak import Speak
from seldon.brain.base import Brain
from seldon.core.memory import ConversationMemory, transcript_messages
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
            yield Utterance(line, speaker="Bastien")


class EchoBrain(Brain):
    """Répond avec speak et garde un historique au format messages."""

    def __init__(self, script=None):
        self.messages = []
        self.script = script or {}   # texte -> liste d'appels (nom, params) avant de répondre

    async def handle(self, utterance, actions, execute):
        self.messages.append({"role": "user", "content": utterance.text})
        for name, params in self.script.get(utterance.text, []):
            await execute(name, params)
        await execute("speak", {"text": f"écho {utterance.text}"})
        self.messages.append({"role": "assistant", "content": f"écho {utterance.text}"})
        return None


class RecordingTTS(TTSEngine):
    def __init__(self):
        self.said = []

    async def speak(self, text):
        self.said.append(text)


def build(tmp_path, lines, brain=None, kind="echo", **mem):
    brain = brain or EchoBrain()
    memory = ConversationMemory(brain, kind, dir=str(tmp_path), **mem)
    tts = RecordingTTS()
    a = Assistant(ScriptedSTT(lines), NoSpeakerId(), brain, tts, [Speak(), Conversations()], memory=memory)
    return a, brain, memory, tts


async def test_conversation_saved_with_title_transcript_and_state(tmp_path):
    a, brain, memory, _ = build(tmp_path, ["parlons du jardin et des tomates", "quand planter ?"])
    await a.run()
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["title"] == "parlons du jardin et des tomates"
    assert data["speakers"] == ["Bastien"]
    assert [m["role"] for m in data["transcript"]] == ["user", "assistant"] * 2
    assert data["state"]["messages"][-1]["content"] == "écho quand planter ?"


async def test_new_and_resume_restore_brain_history(tmp_path):
    a, brain, memory, tts = build(tmp_path, [
        "parlons du jardin", "/nouvelle", "et la voiture ?", "/conversations jardin", "/reprendre 1",
    ])
    await a.run()
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert memory.current.title == "parlons du jardin"
    assert [m["content"] for m in brain.messages] == ["parlons du jardin", "écho parlons du jardin"]
    assert tts.said[-1].startswith("On reprend « parlons du jardin »")


async def test_idle_starts_new_conversation(tmp_path):
    a, brain, memory, _ = build(tmp_path, [], new_after_idle_s=0)
    await a.process(Utterance("premier sujet"))
    first = memory.current.id
    memory.current.updated -= 10
    await a.process(Utterance("second sujet"))
    assert memory.current.id != first and brain.messages[0]["content"] == "second sujet"


async def test_search_ranks_by_topic(tmp_path):
    a, brain, memory, _ = build(tmp_path, [])
    for text in ["recette de la tarte aux pommes", "/nouvelle", "réparer le vélo", "/nouvelle",
                 "les pommes du jardin"]:
        await a.process(Utterance(text))
    titles = [c.title for c in memory.search("pommes")]
    assert set(titles) == {"recette de la tarte aux pommes", "les pommes du jardin"}
    assert memory.search("vélo")[0].title == "réparer le vélo"
    assert memory.search("avion") == []


async def test_brain_tool_resume_is_applied_after_turn(tmp_path):
    a, brain, memory, _ = build(tmp_path, [])
    await a.process(Utterance("parlons du jardin"))
    jardin = memory.current.id
    await a.process(Utterance("/nouvelle"))
    brain.script["reprends le jardin"] = [("conversations", {"operation": "reprendre", "id": jardin})]
    await a.process(Utterance("reprends le jardin"))
    assert memory.current.id == jardin
    assert brain.messages[0]["content"] == "parlons du jardin"
    # la demande de reprise rejoint la conversation reprise ; l'autre, vide, disparaît
    assert [c.title for c in memory.all()] == ["parlons du jardin"]
    assert memory.current.transcript[-2]["text"] == "reprends le jardin"


async def test_tool_search_excludes_current(tmp_path):
    a, brain, memory, _ = build(tmp_path, [])
    await a.process(Utterance("parlons du jardin"))
    tool = Conversations()
    from seldon.core.types import ActionContext
    r = await tool.run({"operation": "chercher", "query": "jardin"}, ActionContext(services={"memory": memory}))
    assert r.text == "Aucune conversation correspondante."


async def test_other_brain_resumes_from_transcript(tmp_path):
    a, brain, memory, _ = build(tmp_path, ["parlons du jardin"], kind="claude")
    await a.run()
    conv_id = memory.current.id
    other = ConversationMemory(EchoBrain(), "llamacpp", dir=str(tmp_path))
    other.resume(conv_id)
    assert other.brain.messages == [
        {"role": "user", "content": "[locuteur : Bastien]\nparlons du jardin"},
        {"role": "assistant", "content": "écho parlons du jardin"},
    ]


def test_transcript_messages_merges_and_drops_trailing_user():
    msgs = transcript_messages([
        {"role": "assistant", "text": "orphelin"},
        {"role": "user", "text": "a"}, {"role": "user", "text": "b"},
        {"role": "assistant", "text": "c"}, {"role": "user", "text": "d"},
    ])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"].endswith("b")


async def test_resume_request_turn_moves_to_target(tmp_path):
    a, brain, memory, _ = build(tmp_path, [])
    await a.process(Utterance("parlons du jardin"))
    jardin = memory.current.id
    await a.process(Utterance("/nouvelle"))
    await a.process(Utterance("17 fois 23"))
    calcul = memory.current.id
    brain.script["et pour arroser le jardin ?"] = [("conversations", {"operation": "reprendre", "id": jardin})]
    await a.process(Utterance("et pour arroser le jardin ?"))
    target, old = memory.load(jardin), memory.load(calcul)
    assert [m["text"] for m in target.transcript] == [
        "parlons du jardin", "écho parlons du jardin", "et pour arroser le jardin ?", "écho et pour arroser le jardin ?"]
    assert [m["text"] for m in old.transcript] == ["17 fois 23", "écho 17 fois 23"]
    assert [m["content"] for m in old.state["messages"]] == ["17 fois 23", "écho 17 fois 23"]
    assert brain.messages[-1]["content"] == "écho et pour arroser le jardin ?"
