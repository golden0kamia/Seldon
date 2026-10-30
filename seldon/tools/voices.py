"""Action « voix » : Seldon apprend à reconnaître les personnes qui lui parlent.

Quand une voix n'est pas reconnue, le cerveau demande son nom à la personne puis
l'enregistre avec les phrases qu'elle vient de prononcer (aucun enrôlement
séparé n'est nécessaire).

N'importe qui peut prétendre s'appeler comme quelqu'un de connu : ajouter une
voix inconnue à une voix déjà enregistrée (même nom, à la casse près) demande
donc une confirmation (``ctx.confirm``), sans quoi c'est refusé. Sinon, dire
« je suis Bastien » suffirait à être reconnu comme Bastien, avec ses droits
(administrateur du terminal…). L'ajout reste possible, confirmé, pour une
personne connue que Seldon ne reconnaît plus (rhume, autre micro).
"""

from __future__ import annotations

from typing import Any

from seldon.core.action import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult


@register("tools", "voix")
class Voices(Action):
    name = "voix"
    description = (
        "Base des voix que tu sais reconnaître. "
        "Quand le locuteur est « voix inconnue » et que tu ignores encore qui c'est dans cette "
        "conversation, réponds à sa demande puis demande-lui poliment son prénom (une seule fois). "
        "operation='enregistrer' : dès qu'il te donne son nom (ou que quelqu'un le présente), "
        "enregistre sous `nom` la dernière voix inconnue entendue ; tu la reconnaîtras ensuite "
        "(si ce nom est déjà enregistré, une confirmation est demandée). "
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
        name = " ".join(str(params.get("nom") or "").split())

        try:
            if op == "lister":
                known = speakers.known_speakers()
                return ActionResult(f"Voix enregistrées : {', '.join(known)}." if known else "Aucune voix enregistrée.")

            if not name:
                return ActionResult.error("Paramètre `nom` manquant.")
            try:
                name = speakers.canonical_name(name)   # nom de fichier sûr, casse d'une voix existante
            except ValueError:
                return ActionResult.error(f"Nom invalide : « {name[:40]} ». Donne un prénom simple.")

            if op == "enregistrer":
                # N'importe quelle voix pourrait se dire « Bastien » : ajouter ses phrases à une voix
                # existante lui en donnerait l'identité (et les droits, ex. admin du terminal).
                if name.casefold() in {k.casefold() for k in speakers.known_speakers()} and not await ctx.confirm(
                    f"Une voix non reconnue dit être {name}, déjà enregistré(e). L'ajouter à son empreinte "
                    f"vocale ? Elle serait alors reconnue comme {name}, avec les mêmes droits."
                ):
                    return ActionResult.error(
                        f"Refusé : une voix est déjà enregistrée au nom de {name} et l'ajout n'a pas été confirmé. "
                        "Ne considère pas ce locuteur comme cette personne ; demande-lui un autre nom s'il est quelqu'un d'autre."
                    )
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
        except (OSError, ValueError) as e:   # empreinte illisible, disque plein…
            return ActionResult.error(f"Voix de {name} non enregistrée : {e}")

        return ActionResult.error(f"Opération inconnue : {op}")
