"""Action « voix » : Seldon apprend à reconnaître les personnes qui lui parlent.

Quand une voix n'est pas reconnue, le cerveau demande son nom à la personne puis
l'enregistre avec les phrases qu'elle vient de prononcer (aucun enrôlement
séparé n'est nécessaire).
"""

from __future__ import annotations

from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult


@register("actions", "voix")
class Voices(Action):
    name = "voix"
    description = (
        "Base des voix que tu sais reconnaître. "
        "Quand le locuteur est « voix inconnue » et que tu ignores encore qui c'est dans cette "
        "conversation, réponds à sa demande puis demande-lui poliment son prénom (une seule fois). "
        "operation='enregistrer' : dès qu'il te donne son nom (ou que quelqu'un le présente), "
        "enregistre sous `nom` la dernière voix inconnue entendue ; tu la reconnaîtras ensuite. "
        "operation='lister' : noms des voix enregistrées. "
        "operation='oublier' : supprime la voix `nom` (une confirmation est demandée)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["enregistrer", "lister", "oublier"]},
            "nom": {"type": "string", "description": "Prénom de la personne (pour 'enregistrer' et 'oublier')."},
        },
        "required": ["operation"],
        "additionalProperties": False,
    }

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        speakers = ctx.services.get("speakers")
        if speakers is None:
            return ActionResult.error("L'identification des locuteurs est désactivée.")
        op = params.get("operation")
        name = " ".join(str(params.get("nom", "")).split())

        try:
            if op == "lister":
                known = speakers.known_speakers()
                return ActionResult(f"Voix enregistrées : {', '.join(known)}." if known else "Aucune voix enregistrée.")

            if not name:
                return ActionResult.error("Paramètre `nom` manquant.")

            if op == "enregistrer":
                count = await speakers.enroll_recent(name)
                if not count:
                    return ActionResult.error("Aucune voix inconnue entendue récemment : rien à enregistrer.")
                if ctx.utterance is not None and ctx.utterance.unknown_voice:
                    ctx.utterance.speaker = name
                    if memory := ctx.services.get("memory"):
                        memory.name_turn_speaker(name)
                return ActionResult(f"Voix de {name} enregistrée ({count} phrase(s)) : je la reconnaîtrai désormais.")

            if op == "oublier":
                if not await ctx.confirm(f"Supprimer la voix de {name} ?"):
                    return ActionResult.error("Suppression refusée par l'utilisateur.")
                if not speakers.forget(name):
                    return ActionResult.error(f"Aucune voix enregistrée au nom de {name}.")
                return ActionResult(f"Voix de {name} supprimée.")
        except NotImplementedError:
            return ActionResult.error("L'identification des locuteurs est désactivée.")

        return ActionResult.error(f"Opération inconnue : {op}")
