"""Action « controle » : permet au cerveau d'agir sur l'état de Seldon lui-même, comme les
commandes locales ``/exit`` et ``/muet`` mais en langage naturel (« coupe ton micro »,
« tu peux t'éteindre »).

Le micro coupé ne se rétablit pas à la voix (plus rien n'est entendu) : seulement par le
bouton Micro de la page Journal. L'arrêt a lieu à la fin du tour, après la réponse.
"""

from __future__ import annotations

from typing import Any

from seldon.core.action import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult


@register("tools", "controle")
class Control(Action):
    name = "controle_seldon"
    description = (
        "Agit sur ton propre fonctionnement. "
        "operation='couper_micro' : coupe ton micro ; tu n'entends plus rien jusqu'à ce que "
        "l'utilisateur le rétablisse avec le bouton Micro de la page Journal (dis-le-lui). "
        "operation='quitter' : arrête Seldon après ta réponse (dis au revoir). "
        "Seulement si l'utilisateur te le demande clairement, jamais pour un simple « stop » "
        "ou « tais-toi » qui veut juste que tu arrêtes de parler."
    )
    input_schema = {
        "type": "object",
        "properties": {"operation": {"type": "string", "enum": ["couper_micro", "quitter"]}},
        "required": ["operation"],
        "additionalProperties": False,
    }

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        op = params.get("operation")
        if op == "couper_micro":
            stt = ctx.services.get("stt")
            if stt is None:
                return ActionResult.error("Aucune reconnaissance vocale : pas de micro à couper.")
            stt.mute(True)
            return ActionResult("Micro coupé : rétabli seulement par le bouton Micro de la page Journal.")
        if op == "quitter":
            stop = ctx.services.get("stop")
            if stop is None:
                return ActionResult.error("Arrêt impossible depuis cette action.")
            stop()
            return ActionResult("Seldon s'arrêtera après ta réponse.")
        return ActionResult.error(f"Opération inconnue : {op}")
