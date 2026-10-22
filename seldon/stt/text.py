"""STT factice : lit le clavier. Utile pour développer sans micro.

Syntaxe ``nom: message`` pour simuler un locuteur identifié.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import AsyncIterator

from seldon.core.registry import register
from seldon.core.types import Utterance
from seldon.stt.base import STTEngine

_stdin = threading.Lock()   # une seule lecture du clavier à la fois


async def read_line(prompt: str = "") -> str:
    """``input(prompt)`` sans bloquer la boucle asyncio ; lève ``EOFError`` en fin d'entrée.

    La lecture se fait dans un thread démon et non via ``asyncio.to_thread`` : une
    lecture en attente (personne au clavier) n'empêche pas Seldon de s'arrêter.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(error: BaseException | None, line: str | None) -> None:
        if future.done():   # lecteur annulé (arrêt) : la ligne est perdue
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(line)

    def read() -> None:
        with _stdin:
            try:
                line, error = input(prompt), None
            except (EOFError, OSError, ValueError) as e:   # ValueError : stdin fermé
                line, error = None, e if isinstance(e, EOFError) else EOFError(str(e))
            try:
                loop.call_soon_threadsafe(deliver, error, line)
            except RuntimeError:   # boucle déjà fermée
                pass

    threading.Thread(target=read, name="seldon-stdin", daemon=True).start()
    return await future


@register("stt", "text")
class TextInput(STTEngine):
    def __init__(self, default_speaker: str | None = None, prompt: str = "> ") -> None:
        self.default_speaker = default_speaker
        self.prompt = prompt

    async def listen(self) -> AsyncIterator[Utterance]:
        while True:
            try:
                # hors terminal (interface), l'invite sans retour à la ligne se collerait au log suivant
                line = await read_line(self.prompt if sys.stdin.isatty() else "")
            except EOFError:
                return
            line = line.strip()
            if not line:
                continue
            speaker = self.default_speaker
            head, sep, tail = line.partition(":")
            if sep and head and " " not in head.strip():
                speaker, line = head.strip(), tail.strip()
            yield Utterance(text=line, speaker=speaker, speaker_confidence=1.0 if speaker else 0.0, typed=True)
