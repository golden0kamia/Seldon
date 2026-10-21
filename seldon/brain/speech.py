"""Parole anticipée : dire la réponse phrase par phrase pendant que le modèle l'écrit.

Un modèle local génère quelques tokens par seconde : attendre la fin de l'appel
``speak`` avant de parler coûte plusieurs secondes sur une réponse de deux ou
trois phrases. Le cerveau lit la réponse en flux continu ; dès qu'une phrase de
l'argument ``text`` est complète, ``SpeechStream`` la confie à l'action
``speak`` (via le pipeline : pause du micro, coupure de parole, mémoire…), puis
la suivante, pendant que la génération continue.

Si l'utilisateur coupe la parole (ou si la synthèse échoue), la suite n'est pas
dite et ``stopped`` devient vrai : le cerveau interrompt alors la génération.
"""

from __future__ import annotations

import asyncio
import json
import re

from seldon.brain.base import ActionExecutor
from seldon.core.types import ActionResult

INTERRUPTED = "Interrompu"   # début du résultat de l'action speak quand on lui coupe la parole

# Abréviations suivies d'un point qui ne terminent pas une phrase.
_ABBREVIATIONS = {"m", "mm", "mme", "mmes", "mlle", "dr", "pr", "me", "st", "ste", "cf", "ex", "env", "av", "bd", "vol",
                  "p", "n", "no", "art", "chap", "fig", "réf", "tél", "j-c"}
_SENTENCE_END = re.compile(r"[.!?…]+(?:\s*[»\")\]])*(?=\s)|\n")   # « C'est fini. » : guillemet compris


def partial_json_string(raw: str, key: str = "text") -> tuple[str, bool] | None:
    """Valeur (éventuellement incomplète) de la chaîne ``key`` dans un objet JSON en cours d'écriture.

    Renvoie (texte décodé jusqu'ici, chaîne terminée ?) ou None si la clé n'est pas encore là.
    """
    m = re.search(rf'"{re.escape(key)}"\s*:\s*"', raw)
    if m is None:
        return None
    start = j = m.end()
    while j < len(raw):
        ch = raw[j]
        if ch == "\\":
            step = 6 if raw[j + 1: j + 2] == "u" else 2
            if j + step > len(raw):
                break   # séquence d'échappement incomplète : on s'arrête avant
            j += step
        elif ch == '"':
            return json.loads(raw[start - 1: j + 1], strict=False), True
        else:
            j += 1
    try:
        return json.loads(f'"{raw[start:j]}"', strict=False), False
    except ValueError:
        return None


def sentence_end(text: str, start: int = 0) -> int:
    """Fin de la dernière phrase complète de ``text`` après ``start`` (``start`` si aucune).

    Une phrase est complète quand sa ponctuation finale est suivie d'un espace : le
    modèle a donc déjà écrit la suite (« 3.5 » ou « M. Dupont » ne coupent pas).
    """
    end = start
    for m in _SENTENCE_END.finditer(text, start):
        word = re.search(r"([\w-]+)\W*$", text[max(0, m.start() - 40):m.start()])
        if m.group().startswith(".") and word and (word.group(1).lower() in _ABBREVIATIONS or len(word.group(1)) == 1):
            continue
        end = m.end()
    return end


class SpeechStream:
    """Dit au fil de l'eau le texte d'un appel ``speak`` en cours de génération."""

    def __init__(self, execute: ActionExecutor, name: str = "speak") -> None:
        self.execute = execute
        self.name = name
        self.sent = 0                      # caractères du texte déjà confiés à la synthèse
        self.spoken: list[str] = []        # phrases effectivement passées à l'action
        self.results: list[ActionResult] = []
        self.stopped = False               # coupure de parole ou erreur : on n'en dit pas plus
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    @property
    def started(self) -> bool:
        return self._task is not None

    @property
    def spoken_text(self) -> str:
        return " ".join(self.spoken)

    def feed(self, raw_arguments: str) -> None:
        """Arguments JSON de l'appel tels que reçus jusqu'ici : dit les phrases complètes."""
        if self.stopped or (found := partial_json_string(raw_arguments)) is None:
            return
        text, complete = found
        self._send(text, len(text) if complete else sentence_end(text, self.sent))

    def _send(self, text: str, end: int) -> None:
        if end <= self.sent or self.stopped:
            return
        chunk, self.sent = text[self.sent:end].strip(), end
        if not chunk:
            return
        if self._task is None:
            self._task = asyncio.create_task(self._speak())
        self._queue.put_nowait(chunk)

    async def _speak(self) -> None:
        while (text := await self._queue.get()) is not None:
            if self.stopped:
                continue
            result = await self.execute(self.name, {"text": text})
            self.spoken.append(text)
            self.results.append(result)
            if result.is_error or result.text.startswith(INTERRUPTED):
                self.stopped = True

    async def finish(self, text: str | None) -> ActionResult | None:
        """Dit la fin de ``text`` (texte complet de l'appel) et attend la dernière phrase.

        Renvoie le résultat à transmettre au modèle, ou None si rien n'a été dit en flux
        (l'appel s'exécute alors normalement).
        """
        if isinstance(text, str):
            self._send(text, len(text))
        if self._task is None:
            return None
        await self.close()
        return next((r for r in self.results if r.is_error or r.text.startswith(INTERRUPTED)),
                    self.results[0] if self.results else ActionResult.error("Rien n'a été prononcé."))

    async def close(self) -> None:
        """Attend la fin de la phrase en cours (les suivantes sont abandonnées si ``stopped``)."""
        if self._task is None:
            return
        if not self._task.done():
            self._queue.put_nowait(None)
        try:
            await asyncio.shield(self._task)
        except asyncio.CancelledError:
            self._task.cancel()
            raise

    def abort(self) -> None:
        """N'en dit pas plus que la phrase en cours (erreur pendant la génération)."""
        self.stopped = True

    def cancel(self) -> None:
        """Arrêt immédiat (tour annulé)."""
        self.stopped = True
        if self._task is not None:
            self._task.cancel()
