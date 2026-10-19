"""Cerveau basé sur Claude : chaque action est exposée comme un outil.

Boucle : message utilisateur -> Claude choisit des outils -> on les exécute ->
résultats renvoyés -> ... jusqu'à ce que Claude ait terminé.

L'historique est conservé entre les tours (append-only : les blocs renvoyés
par l'API, y compris les blocs de réflexion, sont réinjectés tels quels) et
réinitialisé après une période d'inactivité ou au-delà d'une taille maximale.
"""

from __future__ import annotations

import base64
import logging
import time
from datetime import datetime
from typing import Any

import anthropic

from seldon.actions.base import Action
from seldon.brain.base import ActionExecutor, Brain, ends_turn
from seldon.brain.prompts import SYSTEM_PROMPT
from seldon.core.registry import register
from seldon.core.types import ActionResult, Utterance

log = logging.getLogger(__name__)


def _result_content(result: ActionResult) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img.media_type,
                "data": base64.standard_b64encode(img.data).decode("ascii"),
            },
        }
        for img in result.images
    ]
    content.append({"type": "text", "text": result.text or ("Erreur." if result.is_error else "OK.")})
    return content


@register("brain", "claude")
class ClaudeBrain(Brain):
    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: str = "low",               # low | medium | high | xhigh | max
        max_tokens: int = 16000,
        max_steps: int = 10,               # appels API max par prise de parole
        max_history_messages: int = 200,
        fallbacks: bool = True,            # relance côté serveur en cas de refus
        extra_instructions: str = "",
        timeout_s: float = 120,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.max_history_messages = max_history_messages
        self.fallbacks = fallbacks
        self.system = SYSTEM_PROMPT + (f"\n{extra_instructions.strip()}\n" if extra_instructions else "")
        self.timeout_s = timeout_s
        self.client: anthropic.AsyncAnthropic | None = None
        self.messages: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []   # tool_result non encore envoyés

    async def start(self) -> None:
        self.client = anthropic.AsyncAnthropic(timeout=self.timeout_s)

    def reset(self) -> None:
        self.messages.clear()
        self._pending.clear()

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "pending": list(self._pending)}

    def append_exchange(self, user: str, assistant: str) -> None:
        pending, self._pending = self._pending, []
        self.messages.append({"role": "user", "content": [*pending, {"type": "text", "text": user}]})
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant}]})

    def set_state(self, state: dict[str, Any]) -> None:
        super().set_state(state)
        self._pending = list(state.get("pending", []))

    async def _call(self, tools: list[dict[str, Any]]):
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system,
            tools=tools,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
        )
        if self.fallbacks:
            return await self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
            )
        return await self.client.messages.create(**kwargs)

    async def handle(self, utterance: Utterance, actions: list[Action], execute: ActionExecutor) -> str | None:
        assert self.client is not None, "start() n'a pas été appelé"
        if len(self.messages) > self.max_history_messages:
            self.reset()

        speaker = utterance.speaker or "inconnu"
        header = f"[{datetime.now():%Y-%m-%d %H:%M} | locuteur : {speaker}]"
        pending, self._pending = self._pending, []
        content = [*pending, {"type": "text", "text": f"{header}\n{utterance.text}"}]

        snapshot = len(self.messages)
        self.messages.append({"role": "user", "content": content})
        tools = [a.tool_definition() for a in actions]

        try:
            for _ in range(self.max_steps):
                response = await self._call(tools)
                # Blocs convertis en dicts JSON : l'historique peut être sauvegardé tel quel.
                content = [b.model_dump(mode="json", exclude_none=True) for b in response.content]
                self.messages.append({"role": "assistant", "content": content})

                if response.stop_reason == "refusal":
                    log.warning("Refus du modèle : %s", response.stop_details)
                    return "Désolé, je ne peux pas faire ça."

                text = "".join(b.text for b in response.content if b.type == "text").strip()
                calls = [b for b in response.content if b.type == "tool_use"]
                if response.stop_reason != "tool_use" or not calls:
                    return text or None

                results, done = [], []
                for call in calls:
                    result = await execute(call.name, dict(call.input))
                    done.append((call.name, result))
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": _result_content(result),
                        "is_error": result.is_error,
                    })
                if ends_turn(actions, done):
                    # Réponse prononcée : fin du tour. Les résultats partiront avec
                    # le prochain message utilisateur.
                    self._pending = results
                    return None
                self.messages.append({"role": "user", "content": results})
            # Limite atteinte : les derniers résultats d'outils partiront avec
            # le prochain message utilisateur (pas deux tours "user" d'affilée).
            log.warning("Nombre maximal d'étapes atteint (%d)", self.max_steps)
            self._pending = self.messages.pop()["content"]
            return None
        except Exception:
            # Revient à l'état cohérent précédent (pas de tool_use sans résultat).
            del self.messages[snapshot:]
            self._pending = pending
            raise
