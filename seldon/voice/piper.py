"""Synthèse vocale locale avec Piper.

La voix (``<voix>.onnx`` + ``.onnx.json``) est téléchargée dans ``voices_dir``
si elle manque. Piper découpe le texte en phrases : la suivante est calculée
pendant que la précédente est jouée.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from seldon.base import TTS

log = logging.getLogger(__name__)


class Piper(TTS):
    def __init__(
        self,
        voice: str = "fr_FR-siwis-medium",   # nom de voix Piper ou chemin d'un .onnx
        voices_dir: str = "data/voices_tts",
        speaker_id: int | None = None,       # voix à plusieurs locuteurs
        speed: float = 1.0,                  # 1.25 = 25 % plus rapide
        device: int | str | None = None,     # haut-parleur (index ou nom), défaut du système sinon
    ) -> None:
        self.voice_name = voice
        self.voices_dir = Path(voices_dir)
        self.speaker_id = speaker_id
        self.speed = speed
        self.device = device
        self.voice: Any = None

    async def start(self) -> None:
        self.voice = await asyncio.to_thread(self._load)

    def _load(self) -> Any:
        from piper import PiperVoice

        model = Path(self.voice_name) if self.voice_name.endswith(".onnx") else self.voices_dir / f"{self.voice_name}.onnx"
        if not model.exists():
            from piper.download_voices import download_voice

            log.info("Téléchargement de la voix Piper %s…", self.voice_name)
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            download_voice(self.voice_name, self.voices_dir)
        return PiperVoice.load(model)

    def _speak(self, text: str) -> None:
        import sounddevice as sd
        from piper import SynthesisConfig

        config = SynthesisConfig(speaker_id=self.speaker_id, length_scale=1 / self.speed)
        for chunk in self.voice.synthesize(text, syn_config=config):
            sd.wait()   # fin de la phrase précédente
            sd.play(chunk.audio_float_array, chunk.sample_rate, device=self.device)
        sd.wait()

    async def speak(self, text: str) -> None:
        try:
            await asyncio.to_thread(self._speak, text)
        except asyncio.CancelledError:   # arrêt de Seldon : on se tait
            await self.stop()
            raise

    async def stop(self) -> None:
        import sounddevice as sd

        sd.stop()
