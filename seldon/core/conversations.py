"""Outil « conversations », fourni par la mémoire (``ConversationMemory.actions``) : permet
au cerveau de retrouver, reprendre, modifier ou supprimer une conversation passée. Chaque
modification ou suppression demande une confirmation (``ctx.confirm``) : elle n'est faite
que si l'utilisateur l'accepte."""

from __future__ import annotations

from typing import Any

from seldon.core.action import Action
from seldon.core.types import ActionContext, ActionResult

MAX_TITLE = 200   # caractères gardés d'un titre proposé par le cerveau


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
        "operation='renommer' : change le titre de la conversation `id` en `titre`. "
        "operation='oublier' : retire de la conversation `id` les messages qui contiennent `texte`. "
        "operation='supprimer' : supprime définitivement la conversation `id`. "
        "Pour ces trois opérations, `id` vide = conversation actuelle ; l'utilisateur doit confirmer "
        "chacune, ne les fais que s'il le demande. "
        "Utilise-la quand l'utilisateur veut revenir sur un sujet dont vous avez parlé dans "
        "une autre conversation ; inutile si le sujet est déjà dans la conversation actuelle."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string",
                          "enum": ["chercher", "reprendre", "nouvelle", "renommer", "oublier", "supprimer"]},
            "query": {"type": "string", "description": "Mots-clés du sujet recherché (pour 'chercher')."},
            "id": {"type": "string", "description": "Identifiant de la conversation (vide = actuelle, sauf pour 'reprendre')."},
            "titre": {"type": "string", "description": "Nouveau titre (pour 'renommer')."},
            "texte": {"type": "string", "description": "Passage des messages à retirer (pour 'oublier')."},
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    def __init__(self, limit: int = 8, tail: int = 10) -> None:
        self.limit = limit
        self.tail = tail

    def acknowledge(self, params: dict[str, Any]) -> str | None:
        if params.get("operation") != "chercher":
            return None
        query = " ".join(str(params.get("query") or "").split())
        if query and len(query) <= 40:
            return self.vary((f"Je cherche ce qu'on s'est dit à propos de {query}.",
                              f"Je retrouve nos échanges sur {query}."))
        return self.vary(("Je cherche dans nos conversations.", "Je fouille dans nos échanges.",
                          "Je regarde ce qu'on s'est dit."))

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        memory = ctx.services.get("memory")
        if memory is None:
            return ActionResult.error("La mémoire des conversations est désactivée.")
        op = params.get("operation")

        if op == "chercher":
            current = memory.current.id if memory.current else None
            found = [c for c in memory.search(str(params.get("query") or ""), self.limit + 1) if c.id != current]
            if not found:
                return ActionResult("Aucune conversation correspondante.")
            return ActionResult("\n".join(f"- id={c.id} : {c.describe()}" for c in found[: self.limit]))

        if op == "reprendre":
            conv_id = str(params.get("id") or "")
            if memory.is_current(conv_id):
                return ActionResult("C'est déjà la conversation en cours : rien à reprendre.")
            conv = memory.load(conv_id)   # identifiant invalide (chemin…) : None
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

        if op in ("renommer", "oublier", "supprimer"):
            return await self._edit(op, params, memory, ctx)

        return ActionResult.error(f"Opération inconnue : {op}")

    async def _edit(self, op: str, params: dict[str, Any], memory, ctx: ActionContext) -> ActionResult:
        """Renomme, allège ou supprime une conversation, après confirmation de l'utilisateur."""
        conv_id = str(params.get("id") or "") or (memory.current.id if memory.current else "")
        conv = memory.current if memory.is_current(conv_id) else memory.load(conv_id)
        if conv is None:
            return ActionResult.error("Conversation introuvable : utilise d'abord operation='chercher'.")
        name = f"« {conv.title or 'sans titre'} »"

        if op == "renommer":
            title = " ".join(str(params.get("titre") or "").split())[:MAX_TITLE]
            if not title:
                return ActionResult.error("Précise le nouveau titre (paramètre `titre`).")
            if not await ctx.confirm(f"Renommer la conversation {name} en « {title} » ?"):
                return ActionResult("L'utilisateur a refusé : rien n'a été modifié.")
            memory.rename(conv.id, title)
            return ActionResult(f"Conversation renommée « {title} ».")

        if op == "oublier":
            text = str(params.get("texte") or "").strip()
            found = conv.matching(text) if text else []
            if not found:
                return ActionResult.error(f"Aucun message de {name} ne contient « {text} ».")
            if not await ctx.confirm(f"Retirer {len(found)} message(s) contenant « {text} » de la conversation {name} ?"):
                return ActionResult("L'utilisateur a refusé : rien n'a été modifié.")
            memory.forget(conv.id, text)
            return ActionResult(f"{len(found)} message(s) retiré(s) de {name}.")

        if not await ctx.confirm(f"Supprimer définitivement la conversation {name} ({len(conv.transcript)} messages) ?"):
            return ActionResult("L'utilisateur a refusé : rien n'a été supprimé.")
        current = memory.is_current(conv.id)
        memory.request_delete(conv.id)
        return ActionResult(f"Conversation {name} supprimée" + (" après ta réponse." if current else "."))
