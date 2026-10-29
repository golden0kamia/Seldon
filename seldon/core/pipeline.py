"""Orchestration : STT -> identification du locuteur -> cerveau -> actions.

Attention : si l'STT utilise des mots d'éveil, Seldon répond aux phrases qui en
commencent par un, puis reste attentif ``follow_up_s`` secondes après chaque
échange (on peut lui répondre sans le rappeler). Ce qui est dit hors de ces
moments n'est pas traité, mais les dernières phrases (avec leur locuteur) sont
transmises au cerveau la fois suivante, pour qu'il sache de quoi on parlait.

Avec la mémoire, une tâche de fond vérifie toutes les ``compress_every_s``
secondes la taille des conversations et résume les plus volumineuses ; elle ne
travaille jamais pendant le traitement d'une prise de parole (verrou ``_busy``).

Les signaux ``MUTE_SIGNALS`` (SIGUSR1 / SIGUSR2, envoyés par le bouton micro de
la page Journal) coupent et rétablissent le micro (``STTEngine.mute``).

Une erreur pendant un tour est annoncée à voix haute (``errors.spoken``) : cerveau
injoignable, commande en échec, ou action échouée sans que le cerveau en parle.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import deque
from typing import Any

from seldon.actions.base import Action
from seldon.brain.base import Brain
from seldon.brain.fallback import FallbackBrain
from seldon.brain.prompts import neutralize
from seldon.config import Config
from seldon.core import commands, registry, tags
from seldon.core.errors import SeldonError, describe, log_error, spoken
from seldon.core.events import EventBus
from seldon.core.memory import ConversationMemory
from seldon.core.types import ActionContext, ActionResult, ConfirmFn, Utterance
from seldon.speakers.base import SpeakerIdentifier
from seldon.stt.base import STTEngine
from seldon.stt.text import read_line
from seldon.tts.base import TTSEngine

log = logging.getLogger(__name__)

OVERHEARD_S = 120     # ancienneté maximale des phrases entendues transmises au cerveau
OVERHEARD_MAX = 6


# signal -> micro coupé ? (POSIX seulement)
MUTE_SIGNALS = {signal.SIGUSR1: True, signal.SIGUSR2: False} if hasattr(signal, "SIGUSR1") else {}
# SIGTERM (kill, systemd) : arrêt propre, comme Ctrl+C (conversation sauvegardée).
STOP_SIGNALS = [signal.SIGTERM] if hasattr(signal, "SIGTERM") else []

CONFIRM_TAG = tags.tag(tags.CONFIRM, "")   # forme sans jeton ; voir seldon.core.tags


async def console_confirm(question: str) -> bool:
    # Question sur une ligne complète : un lecteur ligne à ligne (l'interface) la voit aussitôt ; avec
    # le jeton de l'interface, une sortie qui imite la balise ne déclenche pas de fausse question.
    print(f"\n{tags.tag(tags.CONFIRM)} {question} [o/N]", flush=True)
    try:
        answer = await read_line()   # thread démon : une question sans réponse ne bloque pas l'arrêt
    except EOFError:   # pas d'entrée standard : refus
        return False
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
        acknowledge: bool = True,
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
        self._terminating = False
        self._task: asyncio.Task | None = None   # tâche de run(), annulée par SIGTERM
        self.memory = memory
        self.follow_up_s = follow_up_s
        self.acknowledge = acknowledge
        self._attentive_until = 0.0
        self._overheard: deque[Utterance] = deque(maxlen=OVERHEARD_MAX)
        self._busy = asyncio.Lock()   # pris pendant un tour, et par la compression des conversations
        # Services accessibles aux actions via ctx.services ; on peut en ajouter.
        self.services: dict[str, Any] = {
            "tts": tts, "stt": stt, "speakers": speakers, "events": self.events, "memory": memory,
            "busy": self._busy,   # verrou d'un échange : une annonce en arrière-plan attend la fin du tour
            "stop": self.request_stop,   # arrêt de Seldon à la fin du tour (action « controle »)
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
        brain_kind = cfg.brain.engine
        if cfg.brain_backup is not None:
            brain = FallbackBrain(brain, build("brain", cfg.brain_backup))
            brain_kind += f"+{cfg.brain_backup.engine}"   # état de conversation propre à ce couple
        memory_opts = dict(cfg.memory)
        enabled = memory_opts.pop("enabled", True)
        if not isinstance(enabled, bool):
            raise SeldonError(f"Section « memory » : enabled doit valoir true ou false, pas {enabled!r}")
        try:
            memory = None
            if enabled:
                registry.check_options(ConversationMemory, "la section", "memory",
                                       {"brain": brain, "brain_kind": brain_kind, **memory_opts})
                memory = ConversationMemory(brain, brain_kind, **memory_opts)
        except (TypeError, ValueError) as e:
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
            acknowledge=cfg.acknowledge,
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
        for action in self.actions.values():   # services (tts, busy…) pour les actions en arrière-plan
            if callable(bind := getattr(action, "bind_services", None)):
                bind(self.services)
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

    async def _compress_periodically(self) -> None:
        """Résume régulièrement le début des conversations trop volumineuses."""
        while True:
            try:
                if n := await self.memory.compress_all(self._busy):
                    log.info("%d conversation(s) compressée(s)", n)
            except Exception as e:
                log_error(log, "Compression des conversations échouée", e)
            await asyncio.sleep(self.memory.compress_every_s)

    async def _warm_up(self) -> None:
        """Prépare le cerveau en arrière-plan (cache du prompt) : la première phrase est plus rapide."""
        started = time.perf_counter()
        try:
            await self.brain.warmup(list(self.actions.values()))
        except Exception as e:
            log.debug("Préparation du cerveau impossible : %s", describe(e))
        else:
            log.debug("Cerveau prêt en %.1f s", time.perf_counter() - started)

    def _terminate(self) -> None:
        """SIGTERM : arrêt propre (comme Ctrl+C), une seule fois."""
        if self._terminating or self._task is None:
            return
        log.info("Arrêt demandé (SIGTERM)")
        self._terminating = self._stop_requested = True
        self._task.cancel()

    def _add_signal_handlers(self, loop: asyncio.AbstractEventLoop) -> dict[int, Any]:
        """Installe les gestionnaires ; renvoie les précédents (rétablis par ``_remove_signal_handlers``)."""
        handlers = {**{sig: (self.stt.mute, on) for sig, on in MUTE_SIGNALS.items()},
                    **{sig: (self._terminate,) for sig in STOP_SIGNALS}}
        previous: dict[int, Any] = {}
        for sig, (callback, *args) in handlers.items():
            try:
                before = signal.getsignal(sig)
                loop.add_signal_handler(sig, callback, *args)
            except (ValueError, RuntimeError, NotImplementedError) as e:   # hors du thread principal…
                log.debug("Signal %s non géré : %s", sig, e)
                continue
            previous[sig] = before
        return previous

    @staticmethod
    def _remove_signal_handlers(loop: asyncio.AbstractEventLoop, previous: dict[int, Any]) -> None:
        for sig, before in previous.items():
            loop.remove_signal_handler(sig)
            if before is not None:   # ex. SIG_IGN posé par __main__ : un clic tardif ne tue pas Seldon
                signal.signal(sig, before)

    async def run(self) -> None:
        background: list[asyncio.Task] = []
        loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        previous = self._add_signal_handlers(loop)
        try:
            await self.start()
            background.append(asyncio.create_task(self._warm_up()))
            if self.memory and self.memory.compress_every_s:
                background.append(asyncio.create_task(self._compress_periodically()))
            async for utterance in self.stt.listen():
                try:
                    await self.process(utterance)
                except Exception as e:   # une phrase ratée n'arrête pas Seldon
                    log_error(log, f"Traitement de « {utterance.text[:60]} » échoué", e)
                    await self.events.emit("error", error=e, stage="pipeline")
                    await self._say_error(spoken(e, "pipeline"))
                if self._stop_requested:
                    break
        except asyncio.CancelledError:
            if not self._terminating:
                raise
            self._task.uncancel()   # SIGTERM : arrêt normal, pas une annulation
        finally:
            for task in background:
                task.cancel()
            # attendues avant stop() : la compression ne touche plus la mémoire pendant la sauvegarde
            await asyncio.gather(*background, return_exceptions=True)
            try:
                await self.stop()
            finally:
                self._remove_signal_handlers(loop, previous)
                self._task = None

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
        try:
            if self.memory and (note := self.memory.absence_note(utterance.speaker, utterance.timestamp)):
                utterance.notes.append(note)
        except Exception as e:   # note facultative : le tour continue sans elle
            log_error(log, "Dernière visite du locuteur introuvable", e)

    async def _say(self, text: str) -> None:
        """Dit ``text`` sans s'entendre soi-même (écoute en pause), puis reste attentif."""
        self.stt.pause()
        try:
            await self.tts.speak(text)
        finally:
            self.stt.resume()
        self._attend()

    async def _say_ack(self, text: str) -> None:
        """Accusé de réception, dit pendant que l'action s'exécute (hors mémoire)."""
        await self.events.emit("acknowledge", text=text)
        try:
            await self._say(text)
        except Exception as e:
            log_error(log, "Accusé de réception non prononcé", e)

    async def _say_error(self, text: str) -> None:
        """Prévient oralement d'une erreur ; ne lève jamais (la synthèse peut être en cause)."""
        try:
            await self._say(text)
        except Exception as e:
            log_error(log, "Message d'erreur non prononcé", e)

    async def process(self, utterance: Utterance) -> None:
        async with self._busy:
            await self._process(utterance)

    async def _process(self, utterance: Utterance) -> None:
        # Seul Seldon écrit des crochets dans les messages du cerveau (en-tête, [note : …]) :
        # une phrase (ou un texte tapé) ne peut pas s'en faire passer pour une.
        utterance.text = neutralize(utterance.text)
        if utterance.addressed is False and time.time() > self._attentive_until:
            utterance = await self._identify(utterance)
            log.debug("[%s] (non adressé) %s", utterance.speaker or "inconnu", utterance.text)
            self._overheard.append(utterance)
            await self.events.emit("overheard", utterance=utterance)
            return
        if utterance.addressed:
            self._attend()
        if self.enable_commands and (found := commands.match(utterance.text, utterance.typed)):
            cmd, args = found
            await self.events.emit("command", name=cmd.aliases[0], args=args)
            reply = await cmd.handler(self, args)
            if reply:
                await self._say(reply)
            return
        utterance = await self._identify(utterance)
        self._context_notes(utterance)
        log.info("[%s] %s", utterance.speaker or "inconnu", utterance.text)
        await self.events.emit("utterance", utterance=utterance)
        memory = self.memory
        if memory:
            try:
                memory.begin_turn(utterance.text, utterance.speaker)
                if note := memory.related_note(utterance.text):
                    utterance.notes.append(note)
            except Exception as e:   # la mémoire en panne ne doit pas empêcher de répondre
                log_error(log, "Mémoire des conversations indisponible pour ce tour", e)
                await self.events.emit("error", error=e, stage="memory")
                memory = None

        ctx = ActionContext(utterance=utterance, services=self.services, confirm=self.confirm)
        spoke = False
        failed: list[tuple[str, str]] = []   # actions en échec (nom, message)
        ack: asyncio.Task | None = None

        async def wait_ack() -> None:
            # La réponse ne doit pas chevaucher l'accusé de réception.
            if ack is not None:
                await ack

        async def execute(name: str, params: dict) -> ActionResult:
            nonlocal spoke, ack
            action = self.actions.get(name)
            if action is None:
                return ActionResult.error(f"Action inconnue : {name}")
            if problem := invalid_params(action, params):
                return ActionResult.error(f"{problem} (outil {name}).")
            if name == "speak":
                await wait_ack()
            elif self.acknowledge and not spoke and ack is None and (text := self._acknowledgement(action, params)):
                # Une fois par tour, sauf si le cerveau a déjà parlé : dit en parallèle de l'action.
                ack = asyncio.create_task(self._say_ack(text))
            await self.events.emit("action.start", name=name, params=params)
            try:
                result = await action.run(params, ctx)
            except Exception as e:
                log_error(log, f"Action {name} échouée", e)
                result = ActionResult.error(describe(e))
            if result.is_error and name != "speak":
                failed.append((name, result.text))
            if name == "speak" and not result.is_error:
                spoke = True
                self._attend()
                if memory:
                    memory.record("assistant", params.get("text", ""))
            await self.events.emit("action.end", name=name, result=result)
            return result

        try:
            try:
                final = await self.brain.handle(utterance, list(self.actions.values()), execute)
            except Exception as e:
                log_error(log, "Le cerveau n'a pas pu répondre", e)
                await self.events.emit("error", error=e, stage="brain")
                final = spoken(e)
            await wait_ack()
            if not final and not spoke and failed:
                # Le cerveau n'a rien dit après un échec : on prévient quand même.
                final = f"Désolé, l'action {failed[-1][0]} a échoué."
            if final:
                await self.events.emit("response", text=final)
                if not spoke and "speak" in self.actions:
                    await execute("speak", {"text": final})
                elif not spoke:
                    await self._say(final)
                    if memory:
                        memory.record("assistant", final)
            elif not spoke:
                log.info("Pas de réponse du cerveau à cette phrase")
        finally:
            if ack is not None and not ack.done():
                ack.cancel()
            if memory:
                try:
                    memory.end_turn()
                except Exception as e:   # ne masque pas l'erreur du tour, ni une réponse déjà donnée
                    log_error(log, "Sauvegarde de la conversation échouée", e)
                    await self.events.emit("error", error=e, stage="memory")

    @staticmethod
    def _acknowledgement(action: Action, params: dict) -> str | None:
        try:
            return action.acknowledge(params)
        except Exception as e:   # un accusé de réception raté n'empêche pas l'action
            log_error(log, f"Accusé de réception de {action.name} impossible", e)
            return None


_JSON_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list}


def invalid_params(action: Action, params: dict) -> str | None:
    """Problème évident des paramètres d'un appel d'outil (manquant, mauvais type), sinon None.

    Le modèle reçoit ce message et peut corriger son appel au lieu de faire échouer l'action.
    """
    schema = action.input_schema or {}
    if missing := [k for k in schema.get("required", []) if k not in params]:
        return f"Paramètre(s) obligatoire(s) manquant(s) : {', '.join(missing)}"
    for key, value in params.items():
        expected = (schema.get("properties") or {}).get(key, {}).get("type")
        if expected in _JSON_TYPES and not isinstance(value, _JSON_TYPES[expected]):
            return f"Paramètre « {key} » : type {expected} attendu, pas {type(value).__name__}"
    return None   # nombres : les actions convertissent elles-mêmes (« 30 » accepté)
