"""Mémoire des conversations et outil « conversations » (dont la sécurité des identifiants)."""

from __future__ import annotations

import json
import stat

import pytest

from seldon.brain.brain import Brain
from seldon.brain.memory import Conversations, Memory, load_all, valid_id
from tests.conftest import FakeApp, ScriptedLLM, context


@pytest.fixture
def memory(tmp_path):
    return Memory(Brain(ScriptedLLM()), dir=str(tmp_path / "conv"), new_after_idle_s=600)


def talk(memory: Memory, text: str, reply: str, speaker: str = "Bastien", at: float = 1000.0) -> None:
    memory.before_turn(at)
    memory.brain.history += [{"role": "user", "text": text, "speaker": speaker, "time": at},
                             {"role": "assistant", "text": reply}]
    memory.save()
    memory.current.updated = at


def test_save_is_private_and_readable(memory):
    talk(memory, "parle-moi des volcans", "Les volcans…")
    path = memory.dir / f"{memory.current.id}.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    data = json.loads(path.read_text())
    assert data["title"] == "parle-moi des volcans" and data["speakers"] == ["Bastien"]
    assert [c.id for c in load_all(memory.dir)] == [memory.current.id]


def test_new_conversation_after_silence(memory):
    talk(memory, "bonjour", "salut", at=1000)
    first = memory.current.id
    memory.before_turn(1000 + 601)
    assert memory.current.id != first and memory.brain.history == []


@pytest.mark.parametrize("conv_id", ["../x", "a/b", "..", "", "x" * 200, "a.json", None, 3])
def test_hostile_ids_never_leave_the_folder(memory, conv_id):
    assert not valid_id(conv_id)
    assert memory.load(conv_id if isinstance(conv_id, str) else "") is None


def test_symlink_out_of_folder_is_refused(memory, tmp_path):
    outside = tmp_path / "secret.json"
    outside.write_text(json.dumps({"messages": [{"role": "user", "text": "secret"}]}))
    memory.dir.mkdir(parents=True)
    (memory.dir / "lien.json").symlink_to(outside)
    assert memory.load("lien") is None


def test_unreadable_files_are_skipped(memory):
    memory.dir.mkdir(parents=True)
    (memory.dir / "abime.json").write_text("{pas du json")
    (memory.dir / "liste.json").write_text("[]")
    assert load_all(memory.dir) == []


async def test_tool_search_resume_and_delete(memory):
    app = FakeApp(answer=True)
    app.memory = memory
    tool = Conversations()
    talk(memory, "on parle des volcans", "D'accord.")
    old = memory.current.id
    memory.new()
    talk(memory, "et la météo ?", "Il pleut.", at=5000)

    found = await tool.run({"operation": "chercher", "query": "volcans"}, context(app))
    assert old in found.text
    resumed = await tool.run({"operation": "reprendre", "id": old}, context(app))
    assert "volcans" in resumed.text and memory.current.id == old
    assert memory.brain.history[0]["text"] == "on parle des volcans"

    assert (await tool.run({"operation": "reprendre", "id": "../../etc/passwd"}, context(app))).error
    await tool.run({"operation": "supprimer", "id": old}, context(app))
    assert app.questions and memory.load(old) is None


async def test_delete_refused_keeps_the_file(memory):
    app = FakeApp(answer=False)
    app.memory = memory
    talk(memory, "garde-moi", "Oui.")
    conv_id = memory.current.id
    result = await Conversations().run({"operation": "supprimer", "id": conv_id}, context(app))
    assert "refusé" in result.text and memory.load(conv_id) is not None
