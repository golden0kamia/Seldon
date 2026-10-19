from __future__ import annotations

from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Synthèse vocale. ``speak`` ne rend la main qu'une fois l'audio joué."""

    async def start(self) -> None:
        """Charge la voix."""

    async def stop(self) -> None:
        """Interrompt la lecture en cours et libère les ressources."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        ...
