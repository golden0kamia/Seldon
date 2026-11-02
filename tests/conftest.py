"""Outils communs aux tests : un faux Seldon et un faux LLM, sans micro ni modèle."""

from __future__ import annotations

from typing import Any

import pytest

from seldon.base import LLM, Context, Tool, ToolCall, Utterance


class FakeApp:
    """Ce que les outils voient de Seldon (``ctx.app``)."""

    def __init__(self, answer: bool = False) -> None:
        self.answer = answer        # réponse aux confirmations
        self.questions: list[str] = []
        self.said: list[str] = []
        self.memory: Any = None
        self.speakers: Any = None

    async def confirm(self, question: str) -> bool:
        self.questions.append(question)
        return self.answer

    async def say(self, text: str) -> None:
        self.said.append(text)

    async def announce(self, text: str) -> None:
        self.said.append(text)


class ScriptedLLM(LLM):
    """LLM qui appelle les outils d'un script : ``[("speak", {"text": "…"}), …]``, ou lève ``fail``."""

    def __init__(self, script: list[tuple[str, dict]] | None = None, reply: str | None = None,
                 fail: Exception | None = None) -> None:
        self.script = script or []
        self.reply = reply
        self.fail = fail
        self.seen: list[list[dict[str, str]]] = []

    async def run(self, system: str, messages: list[dict[str, str]], tools: list[Tool], call: ToolCall) -> str | None:
        self.seen.append(messages)
        if self.fail:
            raise self.fail
        for name, args in self.script:
            await call(name, args)
        return self.reply


@pytest.fixture
def app() -> FakeApp:
    return FakeApp()


def context(app: Any, text: str = "", speaker: str | None = None, **kw: Any) -> Context:
    return Context(app, Utterance(text=text, speaker=speaker, **kw))
