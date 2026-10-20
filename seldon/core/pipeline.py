"""Orchestration : STT -> identification du locuteur -> cerveau -> actions.

Attention : si l'STT utilise des mots d'éveil, Seldon répond aux phrases qui en
commencent par un, puis reste attentif ``follow_up_s`` secondes après chaque
échange (on peut lui répondre sans le rappeler). Ce qui est dit hors de ces
moments n'est pas traité, mais les dernières phrases (avec leur locuteur) sont
transmises au cerveau la fois suivante, pour qu'il sache de quoi on parlait.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from seldon.actions.base import Action
from seldon.brain.base import Brain
from seldon.config import Config
from seldon.core import commands, registry
from seldon.core.errors import SeldonError, describe, log_error, spoken
from seldon.core.events import EventBus
from seldon.core.memory import ConversationMemory
from seldon.core.types import ActionContext, ActionResult, ConfirmFn, Utterance
from seldon.speakers.base import SpeakerIdentifier
from seldon.stt.base import STTEngine
from seldon.tts.base import TTSEngine

log = logging.getLogger(__name__)

OVERHEARD_S = 120     # ancienneté maximale des phrases entendues transmises au cerveau
OVERHEARD_MAX = 6


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
        follow_up_s: float = 20,
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
        self.follow_up_s = follow_up_s
        self._attentive_until = 0.0
        self._overheard: deque[Utterance] = deque(maxlen=OVERHEARD_MAX)
        # Services accessibles aux actions via ctx.services ; on peut en ajouter.
        self.services: dict[str, Any] = {
            "tts": tts, "stt": stt, "speakers": speakers, "events": self.events, "memory": memory,
        }

    @classmethod
    def from_config(cls, cfg: Config) -> "Assistant":
        def build(category: str, stage) -> Any:
            try:
                return registry.create(category, stage.engine, **stage.options)
            except SeldonError:
                raise
            except Exception as e:
                raise SeldonError(f"Création du moteur {category} « {stage.engine} » impossible : {describe(e)}") from e

        brain = build("brain", cfg.brain)
        memory_opts = dict(cfg.memory)
        try:
            memory = (
                ConversationMemory(brain, cfg.brain.engine, **memory_opts)
                if memory_opts.pop("enabled", True) else None
            )
        except TypeError as e:
            raise SeldonError(f"Section « memory » invalide : {e}") from e
        return cls(
            stt=build("stt", cfg.stt),
            speakers=build("speakers", cfg.speakers),
            brain=brain,
            tts=build("tts", cfg.tts),
            actions=[build("actions", a) for a in cfg.actions],
            confirm=CONFIRMERS[cfg.confirm],
            enable_commands=cfg.commands,
            memory=memory,
            follow_up_s=cfg.follow_up_s,
        )

    def _components(self) -> list[tuple[str, Any]]:
        return [
            ("la reconnaissance vocale (stt)", self.stt),
            ("l'identification des voix (speakers)", self.speakers),
            ("le cerveau (brain)", self.brain),
            ("la synthèse vocale (tts)", self.tts),
            *((f"l'action « {name} »", a) for name, a in self.actions.items()),
        ]

    async def start(self) -> None:
        for label, component in self._components():
            try:
                await component.start()
            except SeldonError as e:
                raise SeldonError(f"Impossible de démarrer {label} : {e.message}", e.hint) from e
            except Exception as e:
                raise SeldonError(f"Impossible de démarrer {label} : {describe(e)}") from e

    async def stop(self) -> None:
        if self.memory:
            try:
                self.memory.save()
            except Exception as e:
                log_error(log, "Sauvegarde de la conversation échouée", e)
        for label, component in reversed(self._components()):
            stop = getattr(component, "stop", None)
            if stop is None:
                continue
            try:
                await stop()
            except Exception as e:
                log_error(log, f"Erreur pendant l'arrêt ({label})", e)

    async def run(self) -> None:
        try:
            await self.start()
            async for utterance in self.stt.listen():
                try:
                    await self.process(utterance)
                except Exception as e:   # une phrase ratée n'arrête pas Seldon
                    log_error(log, f"Traitement de « {utterance.text[:60]} » échoué", e)
                    await self.events.emit("error", error=e, stage="pipeline")
                if self._stop_requested:
                    break
        finally:
            await self.stop()

    def request_stop(self) -> None:
        """Demande l'arrêt après la prise de parole en cours."""
        self._stop_requested = True

    def _attend(self) -> None:
        """Reste attentif quelques secondes (sans mot d'éveil) après un échange."""
        self._attentive_until = time.time() + self.follow_up_s

    async def _identify(self, utterance: Utterance) -> Utterance:
        try:
            return await self.speakers.identify(utterance)
        except Exception as e:
            log_error(log, "Identification du locuteur échouée", e)
            await self.events.emit("error", error=e, stage="speakers")
            return utterance

    def _context_notes(self, utterance: Utterance) -> None:
        """Ajoute à ``utterance`` le contexte utile au cerveau (phrases entendues, absence)."""
        if utterance.addressed is False:
            utterance.notes.append(
                "dit sans mot d'éveil, juste après votre échange : réponds seulement si cela t'est adressé."
            )
        recent = [u for u in self._overheard if utterance.timestamp - u.timestamp <= OVERHEARD_S]
        self._overheard.clear()
        if recent:
            said = " ; ".join(f"{u.speaker or 'inconnu'} : « {u.text} »" for u in recent)
            utterance.notes.append(f"entendu juste avant, sans t'être adressé : {said}")
        if self.memory and (note := self.memory.absence_note(utterance.speaker, utterance.timestamp)):
            utterance.notes.append(note)

    async def process(self, utterance: Utterance) -> None:
        if utterance.addressed is False and time.time() > self._attentive_until:
            utterance = await self._identify(utterance)
            log.debug("[%s] (non adressé) %s", utterance.speaker or "inconnu", utterance.text)
            self._overheard.append(utterance)
            await self.events.emit("overheard", utterance=utterance)
            return
        if utterance.addressed:
            self._attend()
        if self.enable_commands and (found := commands.match(utterance.text)):
            cmd, args = found
            await self.events.emit("command", name=cmd.aliases[0], args=args)
            reply = await cmd.handler(self, args)
            if reply:
                await self.tts.speak(reply)
                self._attend()
            return
        utterance = await self._identify(utterance)
        self._context_notes(utterance)
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
                log_error(log, f"Action {name} échouée", e)
                result = ActionResult.error(describe(e))
            if name == "speak" and not result.is_error:
                spoke = True
                self._attend()
                if self.memory:
                    self.memory.record("assistant", params.get("text", ""))
            await self.events.emit("action.end", name=name, result=result)
            return result

        try:
            try:
                final = await self.brain.handle(utterance, list(self.actions.values()), execute)
            except Exception as e:
                log_error(log, "Le cerveau n'a pas pu répondre", e)
                await self.events.emit("error", error=e, stage="brain")
                final = spoken(e)
            if final:
                await self.events.emit("response", text=final)
                if not spoke and "speak" in self.actions:
                    await execute("speak", {"text": final})
                elif not spoke:
                    await self.tts.speak(final)
                    self._attend()
                    if self.memory:
                        self.memory.record("assistant", final)
        finally:
            if self.memory:
                self.memory.end_turn()
