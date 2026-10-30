"""Mémoire des conversations : identifiants hostiles, fichiers abîmés, droits, cache, compression."""

from __future__ import annotations

import asyncio
import json
import stat
import time

import pytest

from seldon.core import memory as memory_mod
from seldon.core.memory import ConversationMemory
from seldon.core.types import ActionContext, Utterance
from seldon.tools.conversations import Conversations
from tests.test_memory import EchoBrain, answers, build


def conv_data(conv_id="c1", title="Le jardin", text="parlons du jardin", **extra):
    return dict(id=conv_id, created=1_760_000_000, updated=1_760_000_060, brain="echo", title=title,
                speakers=["Paul"], transcript=[{"role": "user", "text": text, "speaker": "Paul", "ts": 1_760_000_000}],
                state={"messages": [{"role": "user", "content": text}]}, **extra)


def write(path, data):
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


HOSTILE_IDS = ["../secret", "../../secret", "/tmp/secret", "secret.json", "a/../secret", "..", ".", "", "x" * 101, "c1\x00"]


@pytest.mark.parametrize("conv_id", HOSTILE_IDS)
def test_hostile_ids_never_leave_the_folder(tmp_path, conv_id):
    folder = tmp_path / "convs"
    folder.mkdir()
    secret = write(tmp_path / "secret.json", conv_data("secret"))
    before = secret.read_text()
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(folder))
    assert memory.load(conv_id) is None
    assert memory.resume(conv_id) is None
    assert memory.rename(conv_id, "piraté") is False
    assert memory.forget(conv_id, "jardin") == 0
    assert memory.delete(conv_id) is False
    memory.request_delete(conv_id)
    assert secret.read_text() == before and list(folder.iterdir()) == []


async def test_tool_rejects_traversal_ids(tmp_path):
    folder = tmp_path / "convs"
    folder.mkdir()
    secret = write(tmp_path / "secret.json", conv_data("secret"))
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(folder))
    confirm, asked = answers(True, True, True)
    ctx = ActionContext(services={"memory": memory}, confirm=confirm)
    tool = Conversations()
    for op in ("reprendre", "supprimer", "renommer", "oublier"):
        r = await tool.run({"operation": op, "id": "../secret", "titre": "x", "texte": "jardin"}, ctx)
        assert r.is_error and "introuvable" in r.text
    assert secret.exists() and asked == [] and memory._pending == []


def test_symlink_out_of_folder_is_refused(tmp_path):
    folder = tmp_path / "convs"
    folder.mkdir()
    secret = write(tmp_path / "secret.json", conv_data("secret"))
    (folder / "lien.json").symlink_to(secret)
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(folder))
    assert memory.load("lien") is None and memory.rename("lien", "x") is False
    assert memory.search("jardin") == [] and memory.all() == []   # ni dans la recherche (donc pour le cerveau)
    assert json.loads(secret.read_text())["title"] == "Le jardin"


def test_legacy_ids_still_work(tmp_path):
    write(tmp_path / "ancien_nom-2.json", conv_data("autre-chose"))
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    conv = memory.load("ancien_nom-2")
    assert conv is not None and conv.id == "ancien_nom-2"   # l'identifiant est le nom du fichier
    assert memory.rename("ancien_nom-2", "Potager")
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["ancien_nom-2.json"]


@pytest.mark.parametrize("content", [
    "{pas du json", "[1, 2]", "null", '{"inconnu": 1}',
    json.dumps({**conv_data(), "transcript": [{"role": "user"}]}),                 # message sans texte
    json.dumps({**conv_data(), "transcript": "texte"}),
    json.dumps({**conv_data(), "updated": "hier"}),
    json.dumps(conv_data()).replace("1760000060", "NaN"),
    b"\xff\xfe".decode("latin-1"),
])
def test_unreadable_files_are_skipped(tmp_path, content, caplog):
    write(tmp_path / "abime.json", content)
    write(tmp_path / "bon.json", conv_data("bon"))
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    assert memory.load("abime") is None
    assert [c.id for c in memory.all()] == ["bon"]
    assert [c.id for c in memory_mod.load_conversations(tmp_path)] == ["bon"]
    assert "illisible" in caplog.text


def test_lenient_with_extra_keys_and_missing_ts(tmp_path):
    data = conv_data(futur="champ ajouté plus tard")
    del data["transcript"][0]["ts"]
    write(tmp_path / "c1.json", data)
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    conv = memory.load("c1")
    assert conv.title == "Le jardin" and conv.transcript[0]["ts"] == data["updated"]
    assert memory.absence_note("Paul", now=data["updated"] + 7 * 86400) is not None


def test_invalid_brain_state_is_rebuilt_from_transcript(tmp_path):
    data = conv_data() | {"state": {"messages": "pas une liste"}}
    data["transcript"].append({"role": "assistant", "text": "volontiers", "speaker": None, "ts": 1_760_000_001})
    write(tmp_path / "c1.json", data)
    brain = EchoBrain()
    memory = ConversationMemory(brain, "echo", dir=str(tmp_path))
    assert memory.resume("c1") is not None
    assert brain.messages == [{"role": "user", "content": "[locuteur : Paul]\nparlons du jardin"},
                              {"role": "assistant", "content": "volontiers"}]


async def test_files_are_private_and_written_atomically(tmp_path, monkeypatch):
    folder = tmp_path / "convs"
    a, brain, memory, _ = build(folder, ["bonjour"])
    await a.run()
    [path] = folder.glob("*.json")
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in folder.iterdir()] == [path.name]   # aucun fichier temporaire laissé
    # écriture interrompue : l'ancien contenu reste intact, pas de fichier temporaire
    before = path.read_text()
    memory.current.title = "nouveau"

    def boom(*_):
        raise OSError("disque plein")
    from seldon.core import storage
    monkeypatch.setattr(storage.os, "replace", boom)
    with pytest.raises(OSError):
        memory.save()
    assert path.read_text() == before and [p.name for p in folder.iterdir()] == [path.name]


async def test_cache_skips_unchanged_files_and_sees_external_changes(tmp_path, monkeypatch):
    for i in range(5):
        write(tmp_path / f"c{i}.json", conv_data(f"c{i}", title=f"sujet {i}", text=f"les tomates numéro {i}"))
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    reads = []
    real = memory_mod._read
    monkeypatch.setattr(memory_mod, "_read", lambda path, *a, **k: reads.append(path.name) or real(path, *a, **k))
    memory._index.clear()
    assert len(memory.search("tomates")) == 5 and len(reads) == 5
    memory.search("tomates")
    memory.last_seen("Paul")
    assert len(reads) == 5   # rien de relu
    # modifié par l'interface (éditeur) : relu, même dans la même seconde
    write(tmp_path / "c2.json", conv_data("c2", title="la voiture", text="vidange"))
    (tmp_path / "c4.json").unlink()
    assert {c.title for c in memory.search("voiture")} == {"la voiture"}
    assert reads[5:] == ["c2.json"] and len(memory.all()) == 4
    # nos propres écritures invalident le cache
    assert memory.rename("c1", "les poireaux")
    assert [c.id for c in memory.search("poireaux")] == ["c1"]


async def test_compress_summarizes_outside_the_lock(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    lock = asyncio.Lock()

    class SlowSummary(EchoBrain):
        async def summarize(self, instructions, text):
            started.set()
            await release.wait()
            return "résumé"

    a, brain, memory, _ = build(tmp_path, [], brain=SlowSummary(), compress_max_chars=50, compress_keep=2)
    for i in range(3):
        await a.process(Utterance(f"message numéro {i} sur les tomates"))
    job = asyncio.create_task(memory.compress_all(lock))
    await started.wait()
    assert not lock.locked()   # l'utilisateur peut parler pendant la rédaction du résumé
    await a.process(Utterance("une phrase pendant le résumé"))
    release.set()
    assert await job == 1
    texts = [m["text"] for m in memory.current.transcript]
    assert texts[0] == "résumé" and texts[-2:] == ["une phrase pendant le résumé", "écho une phrase pendant le résumé"]


async def test_compress_skipped_if_conversation_changed_meanwhile(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()

    class SlowSummary(EchoBrain):
        async def summarize(self, instructions, text):
            started.set()
            await release.wait()
            return "résumé"

    a, brain, memory, _ = build(tmp_path, [], brain=SlowSummary(), compress_max_chars=50, compress_keep=2)
    for i in range(3):
        await a.process(Utterance(f"message numéro {i} sur les tomates"))
    job = asyncio.create_task(memory.compress(memory.current.id))
    await started.wait()
    memory._forget_now(memory.current.id, "numéro 0")   # oubli pendant le résumé
    release.set()
    assert await job is False
    assert all(m["role"] != "summary" for m in memory.current.transcript)


async def test_resuming_the_current_conversation_keeps_it(tmp_path):
    a, brain, memory, _ = build(tmp_path, [])
    await a.process(Utterance("parlons du jardin"))
    current = memory.current.id
    brain.script["reprends"] = [("conversations", {"operation": "reprendre", "id": current})]
    await a.process(Utterance("reprends"))
    assert memory.current.id == current and (tmp_path / f"{current}.json").exists()
    assert [m["text"] for m in memory.load(current).transcript if m["role"] == "user"] == ["parlons du jardin", "reprends"]
    # même demandée directement (sans l'outil), la reprise de la conversation en cours ne perd rien
    memory.begin_turn("seule phrase", None)
    memory._pending.append(("switch", current))
    memory.end_turn()
    assert (tmp_path / f"{current}.json").exists()


async def test_last_seen_uses_the_latest_message(tmp_path):
    now = time.time()
    older = conv_data("ancienne", title="ancienne") | {"updated": now - 10}
    older["transcript"][0]["ts"] = now - 3600
    newer = conv_data("recente", title="récente") | {"updated": now - 100}
    newer["transcript"][0]["ts"] = now - 200   # message plus récent dans la conversation modifiée plus tôt
    write(tmp_path / "ancienne.json", older)
    write(tmp_path / "recente.json", newer)
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    assert memory.last_seen("Paul")[2] == "recente"


def test_new_id_never_reuses_an_existing_file(tmp_path, monkeypatch):
    memory = ConversationMemory(EchoBrain(), "echo", dir=str(tmp_path))
    first = memory._new()
    write(tmp_path / f"{first.id}.json", conv_data(first.id))

    class Fixed:
        def __init__(self, hex):
            self.hex = hex
    values = iter([first.id[-4:] + "0" * 28, "beef" + "0" * 28])
    monkeypatch.setattr(memory_mod.uuid, "uuid4", lambda: Fixed(next(values)))
    monkeypatch.setattr(memory_mod.time, "time", lambda: first.created)
    assert memory._new().id.endswith("-beef")


# --- commandes ---------------------------------------------------------------------

async def test_number_without_listing_designates_nothing(tmp_path):
    a, brain, memory, tts = build(tmp_path, [])
    await a.process(Utterance("parlons du jardin"))
    await a.process(Utterance("/nouvelle"))
    a.confirm, asked = answers(True)
    await a.process(Utterance("/supprimer 3"))      # pas de liste affichée : surtout pas « la plus récente »
    await a.process(Utterance("/reprendre 1"))
    await a.process(Utterance("/supprimer le"))     # aucun mot significatif
    assert asked == [] and len(list(tmp_path.glob("*.json"))) == 1
    assert tts.said[-3:] == ["Rien n'a été supprimé.", "Je n'ai pas trouvé cette conversation. Tape /conversations pour la liste.",
                             "Rien n'a été supprimé."]


async def test_listing_is_per_memory_and_delete_confirm_describes(tmp_path):
    a, brain, memory, tts = build(tmp_path / "a", [])
    b, _, other, _ = build(tmp_path / "b", [])
    await a.process(Utterance("parlons du jardin"))
    await a.process(Utterance("/nouvelle"))
    await a.process(Utterance("/conversations"))
    assert len(memory.listing) == 1 and other.listing == []
    b.confirm, asked_b = answers(True)
    await b.process(Utterance("/supprimer 1"))      # la liste affichée par l'autre assistant ne compte pas
    assert asked_b == []
    a.confirm, asked = answers(False)
    await a.process(Utterance("/supprimer 1"))
    assert asked[0].startswith("Supprimer la conversation ") and "« parlons du jardin »" in asked[0] and "2 messages" in asked[0]
