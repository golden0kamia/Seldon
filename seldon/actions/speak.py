from __future__ import annotations

import asyncio
import logging
from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)


@register("actions", "speak")
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
        tts = ctx.services.get("tts")
        if tts is None:
            return ActionResult.error("Aucun moteur TTS configuré.")
        stt = ctx.services.get("stt")
        if stt is None:
            await tts.speak(params["text"])
            return ActionResult(text="Prononcé.")

        stt.pause()  # évite que l'assistant s'entende lui-même...
        speaking = asyncio.create_task(tts.speak(params["text"]))
        listening = asyncio.create_task(stt.interruption(params["text"]))   # ...sauf pour se faire couper
        try:
            await asyncio.wait({speaking, listening}, return_when=asyncio.FIRST_COMPLETED)
            if listening.done() and listening.exception() is not None:
                log.warning("Écoute des interruptions échouée : %s", listening.exception())
            elif listening.done() and not speaking.done():
                await tts.stop()
                await speaking
                return ActionResult(
                    text=f"Interrompu par l'utilisateur (« {listening.result().text} ») : "
                    "la fin de ta phrase n'a pas été entendue."
                )
            await speaking
        finally:
            listening.cancel()
            stt.resume()
        return ActionResult(text="Prononcé.")
