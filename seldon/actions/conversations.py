"""Action « conversations » : permet au cerveau de retrouver et reprendre une conversation passée."""

from __future__ import annotations

from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult


@register("actions", "conversations")
class Conversations(Action):
    name = "conversations"
    description = (
        "Mémoire des conversations passées avec l'utilisateur. "
        "operation='chercher' : liste les conversations dont le sujet correspond à `query` "
        "(ou les plus récentes si `query` est vide), avec leur identifiant. "
        "operation='reprendre' : reprend la conversation `id` (son historique complet sera "
        "rechargé après ta réponse) et renvoie ses derniers échanges, pour que tu puisses "
        "rappeler brièvement à l'utilisateur où vous en étiez. "
        "operation='nouvelle' : démarre une nouvelle conversation après ta réponse. "
        "Utilise-la quand l'utilisateur veut revenir sur un sujet dont vous avez parlé dans "
        "une autre conversation ; inutile si le sujet est déjà dans la conversation actuelle."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["chercher", "reprendre", "nouvelle"]},
            "query": {"type": "string", "description": "Mots-clés du sujet recherché (pour 'chercher')."},
            "id": {"type": "string", "description": "Identifiant de la conversation (pour 'reprendre')."},
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    def __init__(self, limit: int = 8, tail: int = 10) -> None:
        self.limit = limit
        self.tail = tail

    def acknowledge(self, params: dict[str, Any]) -> str | None:
        return "Je cherche dans nos conversations." if params.get("operation") == "chercher" else None

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        memory = ctx.services.get("memory")
        if memory is None:
            return ActionResult.error("La mémoire des conversations est désactivée.")
        op = params.get("operation")

        if op == "chercher":
            current = memory.current.id if memory.current else None
            found = [c for c in memory.search(params.get("query", ""), self.limit + 1) if c.id != current]
            if not found:
                return ActionResult("Aucune conversation correspondante.")
            return ActionResult("\n".join(f"- id={c.id} : {c.describe()}" for c in found[: self.limit]))

        if op == "reprendre":
            conv = memory.load(str(params.get("id", "")))
            if conv is None:
                return ActionResult.error("Conversation introuvable : utilise d'abord operation='chercher'.")
            memory.request_resume(conv.id)
            return ActionResult(
                f"Conversation « {conv.title} » reprise (historique rechargé après ta réponse). "
                f"Derniers échanges :\n{conv.tail(self.tail)}"
            )

        if op == "nouvelle":
            memory.request_new()
            return ActionResult("Une nouvelle conversation commencera après ta réponse.")

        return ActionResult.error(f"Opération inconnue : {op}")
