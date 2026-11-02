"""La boucle de Seldon : écouter -> reconnaître la voix -> réfléchir -> répondre.

``Seldon`` construit les plugins depuis config.yaml, puis pour chaque phrase :

1. le STT la transcrit (``listen``) ;
2. l'identification des voix dit qui parle (``identify``) ;
3. le cerveau (``Brain.answer``) appelle les outils dont il a besoin, et répond
   avec l'outil ``speak`` ;
4. la mémoire sauvegarde la conversation.

Les outils reçoivent ``Seldon`` dans leur contexte (``ctx.app``) : ``say`` pour
parler, ``confirm`` pour faire valider une action, ``memory``, ``speakers``…
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any

from seldon import journal, registry, tags
from seldon.base import STT, TTS, Context, Result, Speakers, Tool, Utterance
from seldon.brain.brain import Brain
from seldon.brain.memory import Memory
from seldon.brain.prompt import neutralize
from seldon.config import Config
from seldon.ear.keyboard import read_line
from seldon.errors import describe, log_error
from seldon.skills import Skills

log = logging.getLogger(__name__)

_JSON_TYPES = {"string": str, "boolean": bool, "object": dict, "array": list}


class Seldon:
    def __init__(self, cfg: Config) -> None:
        self.stt: STT = registry.create("stt", cfg.stt.engine, cfg.stt.options)
        self.speakers: Speakers | None = (
            registry.create("speakers", cfg.speakers.engine, cfg.speakers.options) if cfg.speakers else None)
        self.tts: TTS = registry.create("tts", cfg.tts.engine, cfg.tts.options)
        self.brain = Brain(
            registry.create("brain", cfg.brain.engine, cfg.brain.options),
            registry.create("brain", cfg.backup.engine, cfg.backup.options) if cfg.backup else None,
        )
        self.memory = Memory(self.brain, **cfg.memory) if cfg.memory is not None else None
        self.confirm_mode = cfg.confirm
        # Tous les outils du cerveau, en une seule liste (voir seldon/base.py).
        tools: list[Tool] = [
            *(registry.create("tools", t.engine, t.options) for t in cfg.tools),
            *(self.memory.tools() if self.memory else []),
            *(self.speakers.tools() if self.speakers else []),
            *(registry.create("mcp", m.engine, m.options) for m in cfg.mcp),
            *([Skills(cfg.skills)] if cfg.skills else []),
        ]
        self.tools = {t.name: t for t in tools}
        self.busy = asyncio.Lock()   # pris pendant le traitement d'une phrase

    # --- services offerts aux outils ---------------------------------------------

    async def say(self, text: str) -> None:
        """Dit ``text`` sans s'entendre soi-même."""
        journal.said(text)
        self.stt.pause()
        try:
            await self.tts.speak(text)
        finally:
            self.stt.resume()

    async def announce(self, text: str) -> None:
        """Parle de sa propre initiative (notification…), sans couper une réponse en cours."""
        async with self.busy:
            await self.say(text)

    async def confirm(self, question: str) -> bool:
        """Validation d'une action sensible, selon l'option ``confirm`` de config.yaml."""
        if self.confirm_mode != "console":
            return self.confirm_mode == "always"
        # La balise permet à l'interface d'afficher la question dans une fenêtre.
        print(f"\n{tags.tag(tags.CONFIRM)} {question} [o/N]", flush=True)
        try:
            answer = await read_line()
        except EOFError:
            return False
        return answer.strip().lower() in {"o", "oui", "y", "yes"}

    # --- cycle de vie --------------------------------------------------------------

    def _parts(self) -> list[Any]:
        return [p for p in (self.stt, self.speakers, self.brain, self.tts) if p is not None]

    async def run(self) -> None:
        task = asyncio.current_task()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, task.cancel)   # arrêt demandé par l'interface
        try:
            for part in self._parts():
                await part.start()
            for tool in self.tools.values():
                await tool.start(self)
            log.info("Seldon est prêt. Outils : %s", ", ".join(self.tools))
            async for utterance in self.stt.listen():
                try:
                    await self.handle(utterance)
                except Exception as e:   # une phrase ratée n'arrête pas Seldon
                    log_error(log, "Phrase non traitée", e)
                    await self.say("Désolé, j'ai rencontré une erreur.")
        except asyncio.CancelledError:
            pass
        finally:
            loop.remove_signal_handler(signal.SIGTERM)
            for part in [*self.tools.values(), *reversed(self._parts())]:
                try:
                    await part.stop()
                except Exception as e:
                    log_error(log, f"Arrêt de {type(part).__name__}", e)

    # --- une phrase -----------------------------------------------------------------

    async def handle(self, utterance: Utterance) -> None:
        async with self.busy:
            # Seldon seul écrit des crochets dans les messages du cerveau (en-tête) : une
            # phrase entendue ou tapée ne peut pas s'en faire passer pour un.
            utterance.text = neutralize(utterance.text)
            if self.speakers and utterance.speaker is None:
                try:
                    await self.speakers.identify(utterance)
                except Exception as e:
                    log_error(log, "Identification de la voix", e)
            journal.heard(utterance.speaker, utterance.text)
            if self.memory:
                self.memory.before_turn(utterance.time)

            spoke = False

            async def call(name: str, args: dict[str, Any]) -> Result:
                nonlocal spoke
                result = await self.call_tool(name, args, Context(self, utterance))
                spoke = spoke or (name == "speak" and not result.error)
                return result

            try:
                reply = await self.brain.answer(utterance, list(self.tools.values()), call)
            except Exception as e:
                log_error(log, "Le cerveau n'a pas pu répondre", e)
                reply = "Désolé, je n'arrive pas à joindre mon modèle de langage."
            if reply and not spoke:   # le modèle a répondu en texte au lieu d'appeler speak
                await self.say(reply)
            if self.memory:
                try:
                    self.memory.save()
                except OSError as e:
                    log_error(log, "Sauvegarde de la conversation", e)

    async def call_tool(self, name: str, args: dict[str, Any], ctx: Context) -> Result:
        """Exécute un outil demandé par le cerveau ; une erreur devient un résultat lisible."""
        tool = self.tools.get(name)
        if tool is None:
            return Result.fail(f"Outil inconnu « {name} ». Outils disponibles : {', '.join(self.tools)}.")
        if problem := invalid_args(tool, args):
            return Result.fail(f"{problem} (outil {name}).")
        if name != "speak":
            journal.action(name, args)
        try:
            result = await tool.run(args, ctx)
        except Exception as e:
            log_error(log, f"Outil {name}", e)
            result = Result.fail(describe(e))
        if name != "speak":
            journal.action_result(name, result.error, result.text)
        return result


def invalid_args(tool: Tool, args: dict[str, Any]) -> str | None:
    """Erreur évidente dans les arguments (manquant, mauvais type), que le modèle pourra corriger."""
    schema = tool.parameters or {}
    if missing := [k for k in schema.get("required", []) if k not in args]:
        return f"Paramètre(s) obligatoire(s) manquant(s) : {', '.join(missing)}"
    for key, value in args.items():
        expected = (schema.get("properties") or {}).get(key, {}).get("type")
        if expected in _JSON_TYPES and not isinstance(value, _JSON_TYPES[expected]):
            return f"Paramètre « {key} » : {expected} attendu, pas {type(value).__name__}"
    return None
