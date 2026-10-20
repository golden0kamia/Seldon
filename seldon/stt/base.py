from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator

from seldon.core.types import Utterance


class STTEngine(ABC):
    """Source de paroles transcrites.

    ``listen()`` produit une ``Utterance`` par prise de parole terminée
    (détection de fin de phrase incluse). Le champ ``audio`` doit contenir
    le segment audio (float32, 16 kHz, mono) pour que l'identification du
    locuteur puisse s'en servir ; ``speaker`` peut rester None.
    """

    async def start(self) -> None:
        """Ouvre le micro / charge le modèle."""

    async def stop(self) -> None:
        """Libère les ressources."""

    @abstractmethod
    def listen(self) -> AsyncIterator[Utterance]:
        ...

    def pause(self) -> None:
        """Coupe l'écoute (ex. pendant que le TTS parle, pour éviter l'auto-écoute)."""

    def resume(self) -> None:
        """Reprend l'écoute."""

    async def interruption(self, spoken: str) -> Utterance:
        """Pendant que Seldon dit ``spoken`` (écoute en pause), attend qu'on lui coupe la parole.

        Renvoie la phrase qui l'a interrompu. Par défaut, n'arrive jamais.
        """
        await asyncio.Event().wait()
        raise AssertionError("inaccessible")
