"""Les contrats de Seldon : à lire en premier pour comprendre le reste.

Le trajet d'une phrase (voir ``seldon/pipeline.py``) :

    micro ──STT──> Utterance ──Speakers──> Utterance + locuteur
          ──Brain (LLM + outils)──> outil « speak » ──TTS──> haut-parleur

Chaque étage est une classe ci-dessous ; une implémentation en hérite et se
déclare dans ``seldon/registry.py``. Les outils du cerveau (``Tool``) viennent
de quatre sources, toutes rassemblées en une seule liste pour le LLM :

- les tools (``seldon/tools/``), activés sous ``tools:`` dans ``config.yaml`` ;
- les plugins de Seldon, qui apportent leur outil s'ils sont activés : la
  mémoire (``conversations``) et l'identification des voix (``voix``) ;
- les serveurs MCP (``seldon/mcp.py``) : un outil par serveur ;
- les skills (``skills/<nom>/SKILL.md``) : l'outil ``skill`` lit leurs consignes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

if TYPE_CHECKING:
    from seldon.pipeline import Seldon


# --- Données -----------------------------------------------------------------

@dataclass
class Utterance:
    """Une phrase entendue (ou tapée), et qui l'a dite."""

    text: str
    speaker: str | None = None      # nom du locuteur, None s'il n'est pas reconnu
    unknown_voice: bool = False     # voix analysée mais absente de la base des voix
    audio: Any = None               # numpy float32, 16 kHz, mono (pour l'identification)
    time: float = field(default_factory=time.time)


@dataclass
class Image:
    data: bytes
    media_type: str = "image/jpeg"


@dataclass
class Result:
    """Ce qu'un outil renvoie au cerveau."""

    text: str = ""
    images: list[Image] = field(default_factory=list)
    error: bool = False

    @classmethod
    def fail(cls, message: str) -> Result:
        return cls(text=message, error=True)


@dataclass
class Context:
    """Passé à chaque outil : la phrase en cours et Seldon lui-même (``app``)."""

    app: Seldon
    utterance: Utterance

    @property
    def speaker(self) -> str | None:
        return self.utterance.speaker

    async def confirm(self, question: str) -> bool:
        """Demande à l'utilisateur de valider une action sensible (option ``confirm``)."""
        return await self.app.confirm(question)


# --- Outils du cerveau ---------------------------------------------------------

class Tool:
    """Un outil que le LLM peut appeler.

    Exemple complet :

        class Meteo(Tool):
            name = "meteo"
            description = "Donne la météo d'une ville."
            parameters = {"type": "object", "properties": {"ville": {"type": "string"}},
                          "required": ["ville"]}

            async def run(self, args, ctx):
                return Result("Il fait beau à " + args["ville"])

    puis une ligne dans ``seldon/registry.py`` et une entrée sous ``tools:``.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    ends_turn: bool = False     # True (speak) : le tour s'arrête après cet outil

    async def start(self, app: Seldon) -> None:
        """Au démarrage de Seldon. ``app`` permet de parler plus tard (``app.announce``)."""

    async def stop(self) -> None:
        """À l'arrêt de Seldon."""

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        raise NotImplementedError

    def schema(self) -> dict[str, Any]:
        """Définition envoyée au LLM."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


ToolCall = Callable[[str, dict[str, Any]], Awaitable[Result]]   # (nom, arguments) -> résultat


# --- Étages du pipeline ----------------------------------------------------------

class STT(ABC):
    """Reconnaissance vocale : produit une ``Utterance`` par phrase terminée."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    @abstractmethod
    def listen(self) -> AsyncIterator[Utterance]: ...

    def pause(self) -> None:
        """Pendant que Seldon parle : il ne doit pas s'entendre lui-même."""

    def resume(self) -> None: ...


class Speakers(ABC):
    """Identification du locuteur par sa voix."""

    async def start(self) -> None: ...

    @abstractmethod
    async def identify(self, utterance: Utterance) -> None:
        """Renseigne ``speaker`` et ``unknown_voice``."""

    def tools(self) -> list[Tool]:
        return []


class LLM(ABC):
    """Un modèle de langage qui sait appeler des outils.

    ``messages`` : ``[{"role": "user" | "assistant", "content": str}]``, la phrase
    en cours en dernier. Le LLM appelle ``call(nom, arguments)`` autant que
    nécessaire et renvoie son texte final (ou None s'il a déjà répondu par ``speak``).
    """

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    @abstractmethod
    async def run(self, system: str, messages: list[dict[str, str]], tools: list[Tool],
                  call: ToolCall) -> str | None: ...


class TTS(ABC):
    """Synthèse vocale : ``speak`` rend la main quand la phrase a été dite."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    @abstractmethod
    async def speak(self, text: str) -> None: ...


def turn_over(tools: list[Tool], done: list[tuple[str, Result]]) -> bool:
    """Vrai si les outils appelés à cette étape terminent le tour (ex. ``speak`` réussi)."""
    by_name = {t.name: t for t in tools}
    return bool(done) and all(n in by_name and by_name[n].ends_turn and not r.error for n, r in done)
