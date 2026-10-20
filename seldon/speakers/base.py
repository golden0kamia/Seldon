from __future__ import annotations

from abc import ABC, abstractmethod

from seldon.core.registry import register
from seldon.core.types import Utterance


class SpeakerIdentifier(ABC):
    """Attribue un nom de locuteur à une ``Utterance`` à partir de son audio."""

    async def start(self) -> None:
        """Charge le modèle et les empreintes vocales enregistrées."""

    @abstractmethod
    async def identify(self, utterance: Utterance) -> Utterance:
        """Renseigne ``speaker`` / ``speaker_confidence`` et renvoie l'utterance."""

    async def enroll(self, name: str, audio_samples: list) -> None:
        """Enregistre une nouvelle voix (liste de segments float32 16 kHz)."""
        raise NotImplementedError

    async def enroll_recent(self, name: str) -> int:
        """Enregistre sous ``name`` la dernière voix inconnue entendue.

        Renvoie le nombre de segments utilisés (0 si aucune voix inconnue récente).
        """
        raise NotImplementedError

    def known_speakers(self) -> list[str]:
        """Noms des voix enregistrées."""
        return []

    def forget(self, name: str) -> bool:
        """Supprime la voix ``name`` ; faux si elle n'existe pas."""
        raise NotImplementedError


@register("speakers", "none")
class NoSpeakerId(SpeakerIdentifier):
    """Ne fait rien : garde le locuteur fourni par le STT (ou inconnu)."""

    async def identify(self, utterance: Utterance) -> Utterance:
        return utterance
