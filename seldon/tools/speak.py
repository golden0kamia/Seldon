from __future__ import annotations

import asyncio
import logging
from typing import Any

from seldon.core.action import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)


@register("tools", "speak")
class Speak(Action):
    name = "speak"
    ends_turn = True
    description = (
        "Dit une phrase à voix haute à l'utilisateur via la synthèse vocale. "
        "C'est le seul moyen de lui répondre : utilise-le pour toute réponse, "
        "confirmation ou question. Phrases courtes et naturelles, sans markdown."
    )
    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Texte à prononcer."}},
        "required": ["text"],
        "additionalProperties": False,
    }

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip():
            return ActionResult.error("Paramètre `text` manquant ou vide : rien n'a été dit.")
        tts = ctx.services.get("tts")
        if tts is None:
            return ActionResult.error("Aucun moteur TTS configuré.")
        stt = ctx.services.get("stt")
        if stt is None:
            await tts.speak(text)
            return ActionResult(text="Prononcé.")

        stt.pause()  # évite que l'assistant s'entende lui-même...
        speaking = asyncio.create_task(tts.speak(text))
        listening = asyncio.create_task(stt.interruption(text))   # ...sauf pour se faire couper
        try:
            await asyncio.wait({speaking, listening}, return_when=asyncio.FIRST_COMPLETED)
            heard = listening.done() and not listening.cancelled()
            if heard and listening.exception() is not None:
                log.warning("Écoute des interruptions échouée : %s", listening.exception())
            elif heard and not speaking.done():
                await tts.stop()
                await speaking
                return ActionResult(
                    text=f"Interrompu par l'utilisateur (« {listening.result().text} ») : "
                    "la fin de ta phrase n'a pas été entendue."
                )
            await speaking
        finally:
            try:
                listening.cancel()
                if not speaking.done():   # Speak annulé (arrêt de Seldon…) : il se tait
                    speaking.cancel()
                    try:
                        await tts.stop()
                    except Exception as e:
                        log.warning("Arrêt de la synthèse vocale impossible : %s", e)
                # attend la fin des deux tâches (et récupère leurs exceptions) avant de réécouter
                await asyncio.gather(speaking, listening, return_exceptions=True)
            finally:
                stt.resume()
        return ActionResult(text="Prononcé.")
