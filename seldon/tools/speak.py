"""Outil ``speak`` : la seule façon pour le cerveau de parler à l'utilisateur."""

from __future__ import annotations

from typing import Any

from seldon.base import Context, Result, Tool


class Speak(Tool):
    name = "speak"
    description = (
        "Dit une phrase à voix haute à l'utilisateur. C'est le seul moyen de lui répondre : "
        "utilise-le pour toute réponse, confirmation ou question. Phrases courtes et naturelles, sans markdown."
    )
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Texte à prononcer."}},
        "required": ["text"],
    }
    ends_turn = True

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        text = args["text"].strip()
        if not text:
            return Result.fail("Texte vide : rien n'a été dit.")
        await ctx.app.say(text)
        return Result("Prononcé.")
