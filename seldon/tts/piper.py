"""Synthèse vocale neuronale locale avec Piper (https://github.com/OHF-Voice/piper1-gpl).

La voix (``<voix>.onnx`` + ``<voix>.onnx.json``) est téléchargée automatiquement
depuis le dépôt Hugging Face ``rhasspy/piper-voices`` si elle est absente.
Piper découpe le texte en phrases : la phrase N+1 est synthétisée pendant que
la phrase N est jouée, pour commencer à parler plus tôt.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

from seldon.core.registry import register
from seldon.tts.base import TTSEngine

log = logging.getLogger(__name__)


@register("tts", "piper")
class PiperTTS(TTSEngine):
    def __init__(
        self,
        voice: str = "fr_FR-siwis-medium",
        voices_dir: str = "data/voices_tts",
        speaker_id: int | None = None,
        length_scale: float | None = None,
        speed: float | None = None,
        volume: float = 1.0,
        device: int | str | None = None,
        download: bool = True,
    ) -> None:
        """
        :param voice: nom de voix Piper (``fr_FR-siwis-medium``…) ou chemin vers un ``.onnx``.
        :param voices_dir: dossier où chercher / télécharger les voix.
        :param speaker_id: locuteur pour les voix multi-locuteurs.
        :param length_scale: durée des phonèmes (>1 = plus lent). Prioritaire sur ``speed``.
        :param speed: vitesse relative (1.0 = normale, 1.2 = 20 % plus rapide).
        :param volume: gain appliqué au signal (1.0 = inchangé).
        :param device: périphérique de sortie sounddevice (index ou nom), défaut système sinon.
        :param download: télécharge la voix si elle est absente.
        """
        self.voice_name = voice
        self.voices_dir = Path(voices_dir)
        self.speaker_id = speaker_id
        self.length_scale = length_scale if length_scale is not None else (1 / speed if speed else None)
        self.volume = volume
        self.device = device
        self.download = download
        self._voice: Any = None
        self._stop = threading.Event()
        self._lock = asyncio.Lock()

    def model_path(self) -> Path:
        if self.voice_name.endswith(".onnx"):
            return Path(self.voice_name)
        return self.voices_dir / f"{self.voice_name}.onnx"

    def _load(self) -> Any:
        from piper import PiperVoice

        model = self.model_path()
        if not (model.exists() and Path(f"{model}.json").exists()):
            if not self.download or self.voice_name.endswith(".onnx"):
                raise FileNotFoundError(f"Voix Piper introuvable : {model}")
            from piper.download_voices import download_voice

            log.info("Téléchargement de la voix Piper %s dans %s…", self.voice_name, self.voices_dir)
            self.voices_dir.mkdir(parents=True, exist_ok=True)
            download_voice(self.voice_name, self.voices_dir)
        log.info("Chargement de la voix Piper %s", model)
        return PiperVoice.load(model)

    async def start(self) -> None:
        if self._voice is None:
            self._voice = await asyncio.to_thread(self._load)

    def _syn_config(self) -> Any:
        from piper import SynthesisConfig

        return SynthesisConfig(
            speaker_id=self.speaker_id, length_scale=self.length_scale, volume=self.volume
        )

    def synthesize(self, text: str):
        """Itère sur les ``AudioChunk`` (une phrase chacun) — synthèse paresseuse."""
        return self._voice.synthesize(text, syn_config=self._syn_config())

    def synthesize_wav(self, text: str, path: str | Path) -> None:
        """Écrit la synthèse de ``text`` dans un fichier WAV (sans carte son)."""
        import wave

        with wave.open(str(path), "wb") as wav:
            self._voice.synthesize_wav(text, wav, syn_config=self._syn_config())

    def _speak_blocking(self, text: str) -> None:
        import sounddevice as sd

        # sd.play est non bloquant : la phrase suivante est synthétisée
        # (itération du générateur) pendant que la courante est jouée.
        for chunk in self.synthesize(text):
            sd.wait()
            if self._stop.is_set():
                return
            sd.play(chunk.audio_float_array, chunk.sample_rate, device=self.device)
        sd.wait()

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        await self.start()
        async with self._lock:
            self._stop.clear()
            try:
                await asyncio.to_thread(self._speak_blocking, text)
            except asyncio.CancelledError:
                # Le thread continuerait à parler (et chevaucherait la phrase suivante) : on le coupe.
                self._interrupt()
                raise

    def _interrupt(self) -> None:
        self._stop.set()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:  # PortAudio absent : rien à interrompre
            pass

    async def stop(self) -> None:
        self._interrupt()
