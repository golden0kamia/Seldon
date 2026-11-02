"""TTS muet (mode ``--text``) : la réponse est déjà affichée par le journal (« Seldon › … »)."""

from __future__ import annotations

from seldon.base import TTS


class Console(TTS):
    async def speak(self, text: str) -> None:
        pass
