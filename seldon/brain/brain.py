"""Le cerveau : l'historique de la conversation, confié à un LLM à chaque phrase.

L'historique est une liste simple, commune à tous les LLM et sauvegardée telle
quelle par la mémoire :

    [{"role": "user", "text": "…", "speaker": "Bastien", "time": …},
     {"role": "assistant", "text": "ce que Seldon a dit"}]

Seules les phrases restent dans l'historique ; les appels d'outils d'un tour
sont oubliés à la fin de ce tour.

Si le LLM principal échoue avant d'avoir appelé un outil (serveur arrêté,
quota…), la même phrase est confiée au LLM de secours (``brain.backup``).
"""

from __future__ import annotations

import logging
from typing import Any

from seldon.base import LLM, Result, Tool, ToolCall, Utterance
from seldon.brain.prompt import SYSTEM_PROMPT, to_messages
from seldon.errors import describe

log = logging.getLogger(__name__)


class Brain:
    def __init__(self, llm: LLM, backup: LLM | None = None, max_history: int = 40) -> None:
        self.llms = [llm] + ([backup] if backup else [])
        self.max_history = max_history   # messages envoyés au LLM (les plus récents)
        self.history: list[dict[str, Any]] = []

    async def start(self) -> None:
        errors = []
        for llm in self.llms:
            try:
                await llm.start()
            except Exception as e:
                log.warning("LLM %s indisponible : %s", type(llm).__name__, describe(e))
                errors.append(e)
        if len(errors) == len(self.llms):   # aucun LLM : Seldon ne peut pas répondre
            raise errors[0]

    async def stop(self) -> None:
        for llm in self.llms:
            await llm.stop()

    async def answer(self, utterance: Utterance, tools: list[Tool], call: ToolCall) -> str | None:
        """Fait traiter la phrase par le LLM ; renvoie son texte final s'il n'a pas utilisé ``speak``."""
        message = {"role": "user", "text": utterance.text, "speaker": utterance.speaker,
                   "unknown_voice": utterance.unknown_voice, "time": utterance.time}
        messages = to_messages([*self.history[-self.max_history:], message])
        said: list[str] = []
        used: list[str] = []

        async def tracked(name: str, args: dict[str, Any]) -> Result:
            used.append(name)
            result = await call(name, args)
            if name == "speak" and not result.error:
                said.append(args["text"])
            return result

        reply = None
        for n, llm in enumerate(self.llms):
            try:
                reply = await llm.run(SYSTEM_PROMPT, messages, tools, tracked)
                break
            except Exception as e:
                if used or n == len(self.llms) - 1:   # rejouer referait des actions déjà faites
                    raise
                log.warning("LLM %s en échec (%s) : LLM de secours", type(llm).__name__, describe(e))

        self.history.append(message)
        self.history.append({"role": "assistant", "text": " ".join([*said, reply or ""]).strip() or "…"})
        return reply
