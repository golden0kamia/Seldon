"""Contrat des actions que le cerveau peut déclencher.

Une action = un outil exposé au modèle. Pour en ajouter une :

    @register("actions", "meteo")
    class Weather(Action):
        name = "get_weather"
        description = "Donne la météo d'une ville."
        input_schema = {"type": "object", "properties": {"city": {"type": "string"}},
                        "required": ["city"], "additionalProperties": False}

        async def run(self, params, ctx):
            return ActionResult(text="...")

puis l'activer dans ``config.yaml`` sous ``actions:``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from seldon.core.types import ActionContext, ActionResult


class Action(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[dict[str, Any]]
    # True : si le modèle n'a appelé que des actions de ce type (ex. speak), le
    # tour s'arrête là sans le relancer (évite répétitions et latence inutile).
    ends_turn: ClassVar[bool] = False
    # Phrase dite aussitôt l'action demandée (ex. « Compris, je m'en occupe. »), pour
    # que l'utilisateur sache qu'on l'a compris pendant qu'elle s'exécute. None : rien.
    # Une liste : une formulation au hasard, jamais deux fois de suite la même.
    acknowledgement: ClassVar[str | Sequence[str] | None] = None

    async def start(self) -> None:
        """Initialisation (ouvrir un périphérique, etc.)."""

    async def stop(self) -> None:
        """Libération des ressources."""

    @abstractmethod
    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        ...

    def acknowledge(self, params: dict[str, Any]) -> str | None:
        """Phrase d'accusé de réception pour ces paramètres (défaut : ``acknowledgement``)."""
        return self.vary(self.acknowledgement)

    def vary(self, phrases: str | Sequence[str] | None) -> str | None:
        """Une des ``phrases`` au hasard, différente de la dernière dite par cette action."""
        if phrases is None or isinstance(phrases, str):
            return phrases
        last = getattr(self, "_last_ack", None)
        choices = [p for p in phrases if p != last] or list(phrases)
        if not choices:
            return None
        self._last_ack: str | None = random.choice(choices)
        return self._last_ack

    def tool_definition(self) -> dict[str, Any]:
        """Définition au format outil (nom, description, JSON Schema)."""
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}
