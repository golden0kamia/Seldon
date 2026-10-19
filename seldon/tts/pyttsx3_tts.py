"""Synthèse vocale légère via pyttsx3 (espeak-ng sous Linux) — solution de repli."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from seldon.core.registry import register
from seldon.tts.base import TTSEngine

log = logging.getLogger(__name__)


def find_voice(voices: list[Any], wanted: str) -> Any | None:
    """Trouve une voix par identifiant, langue (``fr``, ``fr-fr``…) ou nom."""
    w = wanted.lower()

    def langs(v: Any) -> list[str]:
        return [(l.decode(errors="ignore") if isinstance(l, bytes) else str(l)).lower().lstrip("\x05")
                for l in (v.languages or [])]

    tests = (
        lambda v: v.id.lower() == w or v.id.lower().endswith("/" + w),
        lambda v: w in langs(v),
        lambda v: any(l.split("-")[0].split("_")[0] == w for l in langs(v)),
        lambda v: w in (v.name or "").lower(),
    )
    for test in tests:
        for v in voices:
            if test(v):
                return v
    return None


@register("tts", "pyttsx3")
class Pyttsx3TTS(TTSEngine):
    def __init__(self, voice: str = "fr", rate: int | None = 170, volume: float | None = None) -> None:
        """
        :param voice: identifiant, langue ou nom de voix (``fr``, ``roa/fr``, ``French``…).
        :param rate: débit en mots par minute (défaut pyttsx3 : 200).
        :param volume: volume entre 0.0 et 1.0.
        """
        self.voice, self.rate, self.volume = voice, rate, volume
        self._engine: Any = None
        # pyttsx3 n'est pas thread-safe : tout passe par un unique thread dédié.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyttsx3")
        self._lock = asyncio.Lock()

    def _init(self) -> None:
        import pyttsx3

        engine = pyttsx3.init()
        if self.voice:
            v = find_voice(engine.getProperty("voices"), self.voice)
            if v is None:
                log.warning("Voix pyttsx3 '%s' introuvable, voix par défaut utilisée", self.voice)
            else:
                engine.setProperty("voice", v.id)
        if self.rate is not None:
            engine.setProperty("rate", self.rate)
        if self.volume is not None:
            engine.setProperty("volume", self.volume)
        self._engine = engine

    def _say(self, text: str) -> None:
        self._engine.say(text)
        self._engine.runAndWait()

    async def _run(self, fn, *args) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def start(self) -> None:
        if self._engine is None:
            await self._run(self._init)

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        await self.start()
        async with self._lock:
            await self._run(self._say, text)

    async def stop(self) -> None:
        # Appelé hors du thread dédié (qui est bloqué dans runAndWait) : interrompt la lecture.
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                log.debug("Arrêt pyttsx3 en échec", exc_info=True)
