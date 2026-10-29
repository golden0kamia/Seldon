"""Cerveau principal + cerveau de secours.

Configuré par ``brain.backup`` dans config.yaml :

    brain:
      engine: claude
      options: {model: claude-opus-5-5}
      backup:
        engine: openai
        options: {model: gpt-5.6-luna}

Le cerveau principal traite chaque prise de parole. S'il ne démarre pas ou
échoue (serveur injoignable, clé refusée, quota…) avant d'avoir lancé la
moindre action, la même phrase est confiée au cerveau de secours ; le principal
est réessayé après ``retry_primary_s``. Une erreur survenue après une action
n'est pas rejouée (l'action serait exécutée deux fois).

Chaque cerveau garde son propre historique (formats différents) ; après chaque
tour, l'échange est recopié en texte simple dans l'autre pour qu'il puisse
reprendre la conversation à tout moment.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from seldon.actions.base import Action
from seldon.brain.base import ActionExecutor, Brain
from seldon.brain.prompts import user_message
from seldon.core.errors import SeldonError, describe
from seldon.core.types import ActionResult, Utterance

log = logging.getLogger(__name__)


class FallbackBrain(Brain):
    def __init__(self, primary: Brain, backup: Brain, retry_primary_s: float = 300) -> None:
        self.primary = primary
        self.backup = backup
        self.retry_primary_s = retry_primary_s
        self._started: set[int] = set()        # id() des cerveaux démarrés
        self._primary_down_until = 0.0         # avant cette date : le secours passe en premier

    @staticmethod
    def _label(brain: Brain) -> str:
        return type(brain).__name__

    @property
    def messages(self) -> list[dict[str, Any]]:   # type: ignore[override]
        return self.active.messages

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        self.primary.messages = list(value)
        self.backup.messages = list(value)

    @property
    def active(self) -> Brain:
        """Cerveau qui sera essayé en premier."""
        return self._order()[0]

    def _order(self) -> list[Brain]:
        if time.monotonic() < self._primary_down_until:
            return [self.backup, self.primary]
        return [self.primary, self.backup]

    def _mark_down(self, brain: Brain, error: BaseException) -> None:
        if brain is self.primary:
            self._primary_down_until = time.monotonic() + self.retry_primary_s
            log.warning("Cerveau principal indisponible (%s) : cerveau de secours utilisé", describe(error))
        else:
            log.warning("Cerveau de secours indisponible : %s", describe(error))

    async def _ensure_started(self, brain: Brain) -> None:
        if id(brain) not in self._started:
            await brain.start()
            self._started.add(id(brain))

    async def start(self) -> None:
        errors: list[BaseException] = []
        for brain in (self.primary, self.backup):
            try:
                await self._ensure_started(brain)
            except Exception as e:
                self._mark_down(brain, e)
                errors.append(e)
        if len(errors) == 2:
            first = errors[0]
            raise SeldonError(
                f"Aucun cerveau disponible. Principal : {describe(first)} — secours : {describe(errors[1])}",
                first.hint if isinstance(first, SeldonError) else None,
            ) from first

    async def stop(self) -> None:
        for brain in (self.primary, self.backup):
            stop = getattr(brain, "stop", None)
            if stop is not None and id(brain) in self._started:
                try:
                    await stop()
                except Exception as e:
                    log.debug("Arrêt du cerveau %s : %s", self._label(brain), describe(e))

    async def warmup(self, actions: list[Action]) -> None:
        for brain in (self.primary, self.backup):
            if id(brain) in self._started:
                try:
                    await brain.warmup(actions)
                except Exception as e:
                    log.debug("Préparation du cerveau %s impossible : %s", self._label(brain), describe(e))

    async def handle(self, utterance: Utterance, actions: list[Action], execute: ActionExecutor) -> str | None:
        order = self._order()
        for n, brain in enumerate(order):
            last = n == len(order) - 1
            said: list[str] = []
            used: list[str] = []

            async def tracked(name: str, params: dict) -> ActionResult:
                used.append(name)
                if name == "speak" and isinstance(params.get("text"), str):
                    said.append(params["text"])
                return await execute(name, params)

            try:
                await self._ensure_started(brain)
                reply = await brain.handle(utterance, actions, tracked)
            except Exception as e:
                if used or last:   # action déjà lancée : la rejouer la ferait deux fois
                    raise
                self._mark_down(brain, e)
                continue
            if brain is self.primary:
                self._primary_down_until = 0.0
            other = self.backup if brain is self.primary else self.primary
            text = " ".join([*said, reply or ""]).strip() or f"(fait : {', '.join(dict.fromkeys(used)) or 'rien'})"
            other.append_exchange(user_message(utterance), text)
            return reply
        return None

    async def summarize(self, instructions: str, text: str) -> str | None:
        for brain in self._order():
            if id(brain) not in self._started:
                continue
            try:
                result = await brain.summarize(instructions, text)
            except Exception as e:
                log.debug("Résumé par %s impossible : %s", self._label(brain), describe(e))
                continue
            if result:
                return result
        return None

    def reset(self) -> None:
        self.primary.reset()
        self.backup.reset()

    def append_exchange(self, user: str, assistant: str) -> None:
        self.primary.append_exchange(user, assistant)
        self.backup.append_exchange(user, assistant)

    def get_state(self) -> dict[str, Any]:
        # « messages » en tête : la mémoire vérifie qu'il est restaurable (voir core.memory).
        return {"messages": list(self.active.messages),
                "primary": self.primary.get_state(), "backup": self.backup.get_state()}

    def set_state(self, state: dict[str, Any]) -> None:
        if isinstance(state.get("primary"), dict) and isinstance(state.get("backup"), dict):
            self.primary.set_state(state["primary"])
            self.backup.set_state(state["backup"])
        else:   # état texte (reconstruit depuis la transcription) : commun aux deux
            self.primary.set_state(state)
            self.backup.set_state(state)
