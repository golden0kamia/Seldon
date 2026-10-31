from __future__ import annotations

from seldon.core.registry import register
from seldon.tts.base import TTSEngine


@register("tts", "console")
class ConsoleTTS(TTSEngine):
    """TTS factice (mode texte) : la réponse figure déjà dans le journal (« Seldon › … », seldon.core.journal)."""

    async def speak(self, text: str) -> None:
        pass
