"""Tests des skills et de la répartition des actions en tools / mcp / skills."""

import pytest

from seldon.config import Config
from seldon.core.errors import SeldonError
from seldon.core.types import ActionContext, Utterance
from seldon.skills.base import Skills, available, load, split_front_matter

SKILL = """---
name: courses
description: Prépare la liste de courses quand on la demande.
---
Lis le modèle liste.md puis demande ce qui manque.
"""


def make(tmp_path, name="courses", text=SKILL, files=None):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "SKILL.md").write_text(text, encoding="utf-8")
    for rel, content in (files or {}).items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_text(content, encoding="utf-8")
    return folder


def ctx():
    return ActionContext(Utterance("bilan"))


def test_front_matter_and_loading(tmp_path):
    meta, body = split_front_matter(SKILL)
    assert meta["name"] == "courses" and body.startswith("Lis le modèle")
    assert split_front_matter("Pas d'en-tête") == ({}, "Pas d'en-tête")
    make(tmp_path)
    make(tmp_path, "vide", "Pas d'en-tête\n")
    (tmp_path / "pas-un-skill").mkdir()
    assert available(tmp_path) == ["courses", "vide"]
    skill = load("courses", tmp_path)
    assert skill.description.startswith("Prépare") and skill.path == (tmp_path / "courses").resolve()
    with pytest.raises(SeldonError, match="description manquante"):
        load("vide", tmp_path)
    with pytest.raises(SeldonError, match="introuvable"):
        load("absent", tmp_path)
    assert load(str(tmp_path / "courses")).name == "courses"   # chemin plutôt que nom


async def test_skill_tool_gives_instructions_and_files(tmp_path):
    make(tmp_path, files={"liste.md": "- pain", "modeles/semaine.md": "lundi"})
    (tmp_path / "secret.txt").write_text("hors du skill")
    tool = Skills(["courses"], tmp_path)
    definition = tool.tool_definition()
    assert definition["name"] == "skill" and "- courses : Prépare la liste" in definition["description"]
    assert definition["input_schema"]["properties"]["name"]["enum"] == ["courses"]
    result = await tool.run({"name": "courses"}, ctx())
    assert not result.is_error and result.text.startswith("Lis le modèle") and "---" not in result.text
    assert "liste.md" in result.text and "modeles/semaine.md" in result.text
    assert (await tool.run({"name": "courses", "file": "liste.md"}, ctx())).text == "- pain"
    escape = await tool.run({"name": "courses", "file": "../secret.txt"}, ctx())
    assert escape.is_error and "hors du skill" not in escape.text
    assert (await tool.run({"name": "autre"}, ctx())).is_error
    # consignes relues à chaque appel
    (tmp_path / "courses" / "SKILL.md").write_text(SKILL.replace("Lis le modèle", "Nouvelle consigne"))
    assert (await tool.run({"name": "courses"}, ctx())).text.startswith("Nouvelle consigne")


def test_duplicate_skill_names_are_refused(tmp_path):
    make(tmp_path)
    make(tmp_path, "courses-bis")   # même « name: courses »
    with pytest.raises(SeldonError, match="Deux skills"):
        Skills(["courses", "courses-bis"], tmp_path)


def test_config_sections_and_old_actions_list():
    cfg = Config.parse("tools: [speak, terminal]\nmcp: [{engine: home_assistant}]\nskills: [courses]\n")
    assert [t.engine for t in cfg.tools] == ["speak", "terminal"]
    assert [m.engine for m in cfg.mcp] == ["home_assistant"] and cfg.skills == ["courses"]
    old = Config.parse("actions: [speak, home_assistant, {engine: camera}]\n")
    assert [t.engine for t in old.tools] == ["speak", "camera"] and [m.engine for m in old.mcp] == ["home_assistant"]
    assert [t.engine for t in Config.parse("stt: text\n").tools] == ["speak"]   # rien d'indiqué : speak
    assert Config.parse("tools: []\n").tools == []
    with pytest.raises(SeldonError, match="skills"):
        Config.parse("skills: [{engine: x}]\n")


async def test_assistant_offers_tools_and_skills(tmp_path, monkeypatch):
    from seldon.core.commands import _actions
    from seldon.core.pipeline import Assistant

    (tmp_path / "skills").mkdir()
    make(tmp_path / "skills")
    monkeypatch.chdir(tmp_path)   # skills/ relatif au dossier de lancement, comme data/
    a = Assistant.from_config(Config.parse("stt: text\nbrain: claude\nmemory: {enabled: false}\n"
                                           "tools: [speak]\nskills: [courses]\n"))
    assert list(a.actions) == ["speak", "skill"]
    assert await _actions(a, "") == "Tools : speak. Skills : courses."
