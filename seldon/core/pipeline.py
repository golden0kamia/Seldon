"""Orchestration : STT -> identification du locuteur -> cerveau -> actions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from seldon.actions.base import Action
from seldon.brain.base import Brain
from seldon.config import Config
from seldon.core import commands, registry
from seldon.core.events import EventBus
from seldon.core.memory import ConversationMemory
from seldon.core.types import ActionContext, ActionResult, ConfirmFn, Utterance
from seldon.speakers.base import SpeakerIdentifier
from seldon.stt.base import STTEngine
from seldon.tts.base import TTSEngine

log = logging.getLogger(__name__)


async def console_confirm(question: str) -> bool:
    answer = await asyncio.to_thread(input, f"\n[confirmation] {question} [o/N] ")
    return answer.strip().lower() in {"o", "oui", "y", "yes"}


async def always_confirm(_: str) -> bool:
    return True


async def never_confirm(_: str) -> bool:
    return False


CONFIRMERS: dict[str, ConfirmFn] = {
    "console": console_confirm,
    "always": always_confirm,
    "never": never_confirm,
}


class Assistant:
    def __init__(
        self,
        stt: STTEngine,
        speakers: SpeakerIdentifier,
        brain: Brain,
        tts: TTSEngine,
        actions: list[Action],
        confirm: ConfirmFn = console_confirm,
        events: EventBus | None = None,
        enable_commands: bool = True,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.stt = stt
        self.speakers = speakers
        self.brain = brain
        self.tts = tts
        self.actions = {a.name: a for a in actions}
        self.confirm = confirm
        self.events = events or EventBus()
        self.enable_commands = enable_commands
        self._stop_requested = False
        self.memory = memory
        # Services accessibles aux actions via ctx.services ; on peut en ajouter.
        self.services: dict[str, Any] = {"tts": tts, "stt": stt, "events": self.events, "memory": memory}

    @classmethod
    def from_config(cls, cfg: Config) -> "Assistant":
        def build(category: str, stage) -> Any:
            return registry.create(category, stage.engine, **stage.options)

        brain = build("brain", cfg.brain)
        memory_opts = dict(cfg.memory)
        memory = (
            ConversationMemory(brain, cfg.brain.engine, **memory_opts)
            if memory_opts.pop("enabled", True) else None
        )
        return cls(
            stt=build("stt", cfg.stt),
            speakers=build("speakers", cfg.speakers),
            brain=brain,
            tts=build("tts", cfg.tts),
            actions=[build("actions", a) for a in cfg.actions],
            confirm=CONFIRMERS[cfg.confirm],
            enable_commands=cfg.commands,
            memory=memory,
        )

    def _components(self) -> list[Any]:
        return [self.stt, self.speakers, self.brain, self.tts, *self.actions.values()]

    async def start(self) -> None:
        for component in self._components():
            await component.start()

    async def stop(self) -> None:
        if self.memory:
            try:
                self.memory.save()
            except Exception:
                log.exception("Sauvegarde de la conversation échouée")
        for component in reversed(self._components()):
            stop = getattr(component, "stop", None)
            if stop is None:
                continue
            try:
                await stop()
            except Exception:
                log.exception("Erreur à l'arrêt de %s", component)

    async def run(self) -> None:
        await self.start()
        try:
            async for utterance in self.stt.listen():
                await self.process(utterance)
                if self._stop_requested:
                    break
        finally:
            await self.stop()

    def request_stop(self) -> None:
        """Demande l'arrêt après la prise de parole en cours."""
        self._stop_requested = True

    async def process(self, utterance: Utterance) -> None:
        if self.enable_commands and (found := commands.match(utterance.text)):
            cmd, args = found
            await self.events.emit("command", name=cmd.aliases[0], args=args)
            reply = await cmd.handler(self, args)
            if reply:
                await self.tts.speak(reply)
            return
        try:
            utterance = await self.speakers.identify(utterance)
        except Exception as e:
            log.exception("Identification du locuteur échouée")
            await self.events.emit("error", error=e, stage="speakers")
        log.info("[%s] %s", utterance.speaker or "inconnu", utterance.text)
        await self.events.emit("utterance", utterance=utterance)
        if self.memory:
            self.memory.begin_turn(utterance.text, utterance.speaker)

        ctx = ActionContext(utterance=utterance, services=self.services, confirm=self.confirm)
        spoke = False

        async def execute(name: str, params: dict) -> ActionResult:
            nonlocal spoke
            action = self.actions.get(name)
            if action is None:
                return ActionResult.error(f"Action inconnue : {name}")
            await self.events.emit("action.start", name=name, params=params)
            try:
                result = await action.run(params, ctx)
            except Exception as e:
                log.exception("Action %s échouée", name)
                result = ActionResult.error(f"{type(e).__name__}: {e}")
            if name == "speak" and not result.is_error:
                spoke = True
                if self.memory:
                    self.memory.record("assistant", params.get("text", ""))
            await self.events.emit("action.end", name=name, result=result)
            return result

        try:
            try:
                final = await self.brain.handle(utterance, list(self.actions.values()), execute)
            except Exception as e:
                log.exception("Le cerveau a échoué")
                await self.events.emit("error", error=e, stage="brain")
                final = "Désolé, j'ai rencontré une erreur."
            if final:
                await self.events.emit("response", text=final)
                if not spoke and "speak" in self.actions:
                    await execute("speak", {"text": final})
                elif not spoke:
                    await self.tts.speak(final)
                    if self.memory:
                        self.memory.record("assistant", final)
        finally:
            if self.memory:
                self.memory.end_turn()
