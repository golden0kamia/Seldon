from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from seldon.core.action import Action
from seldon.core.types import ActionResult, Utterance

# Appelé par le cerveau pour exécuter une action ; géré par le pipeline
# (événements, contexte, gestion d'erreurs).
ActionExecutor = Callable[[str, dict], Awaitable[ActionResult]]


def ends_turn(actions: list[Action], done: list[tuple[str, ActionResult]]) -> bool:
    """Vrai si toutes les actions exécutées terminent le tour et ont réussi."""
    by_name = {a.name: a for a in actions}
    return bool(done) and all(
        name in by_name and by_name[name].ends_turn and not result.is_error for name, result in done
    )


class Brain(ABC):
    """Décide des actions à effectuer pour une prise de parole."""

    async def start(self) -> None:
        ...

    async def warmup(self, actions: list[Action]) -> None:
        """Prépare le modèle en arrière-plan au démarrage (chargement, cache du prompt système et
        des outils) pour que la première phrase ne l'attende pas. Optionnel."""

    @abstractmethod
    async def handle(
        self,
        utterance: Utterance,
        actions: list[Action],
        execute: ActionExecutor,
    ) -> str | None:
        """Traite une prise de parole.

        Appelle ``execute(nom, params)`` autant de fois que nécessaire et
        renvoie éventuellement un texte final (prononcé par le pipeline si
        aucune action ``speak`` n'a été utilisée).
        """

    messages: list[dict[str, Any]]

    def reset(self) -> None:
        """Oublie l'historique de conversation."""
        self.messages = []

    def get_state(self) -> dict[str, Any]:
        """État de la conversation, sérialisable en JSON (pour la mémoire)."""
        return {"messages": list(self.messages)}

    def append_exchange(self, user: str, assistant: str) -> None:
        """Ajoute un échange en texte simple à l'historique (ex. reprise de conversation)."""
        self.messages.append({"role": "user", "content": user})
        self.messages.append({"role": "assistant", "content": assistant})

    async def summarize(self, instructions: str, text: str) -> str | None:
        """Texte produit par le modèle pour ``instructions`` appliquées à ``text``, sans outils
        ni historique (ex. résumé d'une conversation). None : non pris en charge."""
        return None

    def set_state(self, state: dict[str, Any]) -> None:
        """Restaure un état produit par ``get_state`` (même type de cerveau)."""
        self.reset()
        self.messages = list(state.get("messages", []))
