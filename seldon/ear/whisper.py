"""Écoute du micro et transcription avec faster-whisper.

Le micro (16 kHz, mono) est lu par trames de 32 ms. Chaque trame passe par le
détecteur de voix Silero (livré avec faster-whisper) ; une phrase se termine
après ``silence_ms`` de silence, puis Whisper la transcrit.

Pendant que Seldon parle (``pause``), le son du micro est jeté : il ne s'entend
pas lui-même.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from typing import Any, AsyncIterator

import numpy as np

from seldon.base import STT, Utterance

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512   # 32 ms : taille de trame imposée par Silero

# Phrases que Whisper « entend » dans le silence ou le bruit.
HALLUCINATIONS = {"sous-titres réalisés par la communauté d'amara.org", "merci d'avoir regardé cette vidéo"}


class Silero:
    """Probabilité de parole d'une trame (modèle ONNX de faster-whisper)."""

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self.session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(64, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame])[None, :].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-64:]
        return float(np.asarray(out).reshape(-1)[0])


class Segmenter:
    """Regroupe les trames en phrases : ``push`` renvoie la phrase finie, sinon None."""

    def __init__(self, silence_ms: int = 700, min_speech_ms: int = 300, max_speech_s: float = 20,
                 threshold: float = 0.5) -> None:
        def frames(ms: float) -> int:
            return max(1, round(ms * SAMPLE_RATE / 1000 / FRAME))

        self.silence_frames = frames(silence_ms)
        self.min_speech_frames = frames(min_speech_ms)
        self.max_frames = frames(max_speech_s * 1000)
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.before: deque[np.ndarray] = deque(maxlen=6)   # ~200 ms gardés avant la parole (début des mots)
        self.frames: list[np.ndarray] = []
        self.speech = 0
        self.silence = 0

    def push(self, frame: np.ndarray, prob: float) -> np.ndarray | None:
        speaking = prob >= self.threshold
        if not self.frames:
            if not speaking:
                self.before.append(frame)
                return None
            self.frames = list(self.before)
        self.frames.append(frame)
        self.speech += speaking
        self.silence = 0 if speaking else self.silence + 1
        if self.silence < self.silence_frames and len(self.frames) < self.max_frames:
            return None
        frames, speech = self.frames, self.speech
        self.reset()
        return np.concatenate(frames) if speech >= self.min_speech_frames else None


class Whisper(STT):
    def __init__(
        self,
        model: str = "small",           # tiny / base / small / medium / large-v3
        language: str | None = "fr",    # None : détection automatique
        device: str = "cpu",            # cuda si GPU NVIDIA
        compute_type: str = "int8",     # int8 (CPU), float16 (GPU)
        silence_ms: int = 700,          # silence qui termine une phrase
        input_device: int | str | None = None,   # micro (index ou nom, voir python -m sounddevice)
    ) -> None:
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.input_device = input_device
        self.segmenter = Segmenter(silence_ms)
        self.model: Any = None
        self.vad: Silero | None = None
        self.stream: Any = None
        self.paused = False
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue()

    async def start(self) -> None:
        from faster_whisper import WhisperModel

        threads = max(4, min(8, (os.cpu_count() or 8) // 2))
        self.model = await asyncio.to_thread(WhisperModel, self.model_name, device=self.device,
                                             compute_type=self.compute_type, cpu_threads=threads)
        self.vad = Silero()
        self._open_micro(asyncio.get_running_loop())
        log.info("Whisper « %s » prêt : j'écoute", self.model_name)

    def _open_micro(self, loop: asyncio.AbstractEventLoop) -> None:
        import sounddevice as sd

        def on_audio(data, _frames, _time, _status) -> None:   # thread audio : trame par trame
            if self.paused:
                return
            frame = data[:, 0].copy()
            if (sentence := self.segmenter.push(frame, self.vad(frame))) is not None:
                loop.call_soon_threadsafe(self.queue.put_nowait, sentence)

        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
                                     device=self.input_device, callback=on_audio)
        self.stream.start()

    async def stop(self) -> None:
        if self.stream is not None:
            self.stream.close()

    def pause(self) -> None:
        self.paused = True
        self.segmenter.reset()

    def resume(self) -> None:
        self.segmenter.reset()
        self.vad.reset()
        while not self.queue.empty():   # ce qui a été capté juste avant de parler
            self.queue.get_nowait()
        self.paused = False

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(audio, language=self.language, beam_size=5,
                                            condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()

    async def listen(self) -> AsyncIterator[Utterance]:
        while True:
            audio = await self.queue.get()
            text = await asyncio.to_thread(self.transcribe, audio)
            if text and text.lower().strip(" .!") not in HALLUCINATIONS:
                yield Utterance(text=text, audio=audio)
