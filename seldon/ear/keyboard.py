"""STT de test : lit le clavier au lieu du micro (``python -m seldon --text``).

``Bastien: bonjour`` simule une phrase de Bastien.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from typing import AsyncIterator

from seldon.base import STT, Utterance

_stdin = threading.Lock()   # une seule lecture du clavier à la fois (phrase ou confirmation)
_pending = bytearray()      # octets lus après la dernière ligne rendue


def _read_raw_line(prompt: str) -> str | None:
    """Une ligne de l'entrée standard, ``None`` en fin d'entrée.

    ``os.read`` sur le descripteur plutôt que ``input()`` : un thread démon bloqué
    dans ``input()`` garde le verrou du tampon de ``sys.stdin``, et l'arrêt de
    l'interpréteur échoue alors (« could not acquire lock … at interpreter shutdown »).
    """
    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    fd = sys.stdin.fileno()
    while b"\n" not in _pending:
        chunk = os.read(fd, 4096)
        if not chunk:   # fin d'entrée : dernière ligne sans retour à la ligne, puis plus rien
            if not _pending:
                return None
            line = bytes(_pending)
            _pending.clear()
            return line.decode("utf-8", "replace").rstrip("\r")
        _pending.extend(chunk)
    end = _pending.index(b"\n")
    line = bytes(_pending[:end])
    del _pending[:end + 1]
    return line.decode("utf-8", "replace").rstrip("\r")


async def read_line(prompt: str = "") -> str:
    """``input()`` sans bloquer asyncio ; lève ``EOFError`` en fin d'entrée.

    Thread démon plutôt que ``asyncio.to_thread`` : une lecture en attente
    n'empêche pas Seldon de s'arrêter.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(line: str | None) -> None:
        if not future.done():
            future.set_result(line) if line is not None else future.set_exception(EOFError())

    def read() -> None:
        with _stdin:
            try:
                line = _read_raw_line(prompt)
            except (OSError, ValueError):
                line = None
            try:
                loop.call_soon_threadsafe(deliver, line)
            except RuntimeError:   # Seldon déjà arrêté
                pass

    threading.Thread(target=read, daemon=True).start()
    return await future


class Keyboard(STT):
    def __init__(self, default_speaker: str | None = None) -> None:   # locuteur sans « Nom: »
        self.default_speaker = default_speaker

    async def listen(self) -> AsyncIterator[Utterance]:
        while True:
            try:
                line = (await read_line("> " if sys.stdin.isatty() else "")).strip()
            except EOFError:
                return
            if not line:
                continue
            speaker = self.default_speaker
            head, sep, rest = line.partition(":")
            if sep and head.strip() and " " not in head.strip():
                speaker, line = head.strip(), rest.strip()
            yield Utterance(text=line, speaker=speaker)
