from __future__ import annotations

from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult


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
        if stt is not None:
            stt.pause()  # évite que l'assistant s'entende lui-même
        try:
            await tts.speak(params["text"])
        finally:
            if stt is not None:
                stt.resume()
        return ActionResult(text="Prononcé.")
