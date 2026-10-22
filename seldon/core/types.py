"""Types de données partagés entre tous les étages du pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Utterance:
    """Une prise de parole transcrite, éventuellement attribuée à un locuteur."""

    text: str
    speaker: str | None = None          # nom du locuteur identifié, None si inconnu
    speaker_confidence: float = 0.0     # score de similarité 0..1
    unknown_voice: bool = False         # voix analysée mais absente de la base de locuteurs
    language: str | None = None
    timestamp: float = field(default_factory=time.time)
    audio: Any = None                   # numpy array float32 16 kHz mono (optionnel)
    addressed: bool | None = None       # mot d'éveil prononcé ; None si aucun n'est configuré
    typed: bool = False                 # tapé au clavier : les commandes locales exigent le « / »
    notes: list[str] = field(default_factory=list)   # contexte ajouté par Seldon pour le cerveau


@dataclass
class Image:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass
class ActionResult:
    """Résultat renvoyé au cerveau après l'exécution d'une action."""

    text: str = ""
    images: list[Image] = field(default_factory=list)
    is_error: bool = False

    @classmethod
    def error(cls, message: str) -> "ActionResult":
        return cls(text=message, is_error=True)


ConfirmFn = Callable[[str], Awaitable[bool]]


async def _deny(_: str) -> bool:
    return False


@dataclass
class ActionContext:
    """Contexte passé à chaque action : qui parle, services partagés, confirmation."""

    utterance: Utterance | None = None
    services: dict[str, Any] = field(default_factory=dict)   # ex. {"tts": TTSEngine}
    confirm: ConfirmFn = _deny                                # demande une validation humaine
