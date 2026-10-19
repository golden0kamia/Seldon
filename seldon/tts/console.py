from __future__ import annotations

from seldon.core.registry import register
from seldon.tts.base import TTSEngine


@register("tts", "console")
class ConsoleTTS(TTSEngine):
    """TTS factice : affiche la réponse dans le terminal."""

    def __init__(self, prefix: str = "Seldon: ") -> None:
        self.prefix = prefix

    async def speak(self, text: str) -> None:
        print(f"{self.prefix}{text}", flush=True)
