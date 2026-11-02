"""Outil « skill » : consignes lues dans skills/<nom>/SKILL.md."""

from __future__ import annotations

import pytest

from seldon.errors import SeldonError
from seldon.skills import Skills, parse
from tests.conftest import FakeApp, context


def write_skill(folder, name: str, description: str, body: str) -> None:
    (folder / name).mkdir(parents=True)
    (folder / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")


async def test_description_lists_skills_and_run_returns_body(tmp_path):
    write_skill(tmp_path, "bilan", "Fait le point du matin.", "1. Lis l'agenda.")
    tool = Skills(["bilan"], tmp_path)
    assert "- bilan : Fait le point du matin." in tool.description
    assert tool.parameters["properties"]["name"]["enum"] == ["bilan"]
    result = await tool.run({"name": "bilan"}, context(FakeApp()))
    assert result.text == "1. Lis l'agenda."
    assert (await tool.run({"name": "../secret"}, context(FakeApp()))).error


def test_missing_skill_is_a_clear_error(tmp_path):
    with pytest.raises(SeldonError, match="absent"):
        Skills(["absent"], tmp_path)


def test_parse_without_header():
    assert parse("juste du texte") == ({}, "juste du texte")


def test_project_skills_load():
    Skills(["bilan-du-matin"])
