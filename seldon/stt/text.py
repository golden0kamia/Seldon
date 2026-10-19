"""STT factice : lit le clavier. Utile pour développer sans micro.

Syntaxe ``nom: message`` pour simuler un locuteur identifié.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from seldon.core.registry import register
from seldon.core.types import Utterance
from seldon.stt.base import STTEngine


@register("stt", "text")
class TextInput(STTEngine):
    def __init__(self, default_speaker: str | None = None, prompt: str = "> ") -> None:
        self.default_speaker = default_speaker
        self.prompt = prompt

    async def listen(self) -> AsyncIterator[Utterance]:
        while True:
            try:
                line = await asyncio.to_thread(input, self.prompt)
            except EOFError:
                return
            line = line.strip()
            if not line:
                continue
            speaker = self.default_speaker
            head, sep, tail = line.partition(":")
            if sep and head and " " not in head.strip():
                speaker, line = head.strip(), tail.strip()
            yield Utterance(text=line, speaker=speaker, speaker_confidence=1.0 if speaker else 0.0)
