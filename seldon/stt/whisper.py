"""STT micro : capture sounddevice + détection d'activité vocale + faster-whisper.

Un thread de capture découpe le flux micro (16 kHz mono float32) en trames
de 32 ms, les passe au VAD (Silero, fourni avec faster-whisper, ou un simple
seuil d'énergie) et pousse chaque prise de parole terminée dans une file
asyncio. ``listen()`` transcrit ensuite chaque segment dans un thread.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
from collections import deque
from typing import Any, AsyncIterator, Callable

import numpy as np

from seldon.core.registry import register
from seldon.core.types import Utterance
from seldon.stt.base import STTEngine

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME = 512  # 32 ms : taille de trame imposée par Silero VAD

# Hallucinations classiques de Whisper sur du bruit ou du silence.
DEFAULT_IGNORE = [
    "sous-titres réalisés par la communauté d'amara.org",
    "sous-titrage st' 501",
    "merci d'avoir regardé cette vidéo",
]


class Segmenter:
    """Machine à états qui regroupe des trames en prises de parole.

    ``push(trame, proba)`` renvoie le segment complet (np.ndarray) quand une
    pause d'au moins ``silence_ms`` suit de la parole, sinon None.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        silence_ms: int = 700,
        min_speech_ms: int = 300,
        max_speech_s: float = 20.0,
        pad_ms: int = 200,
        frame: int = FRAME,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        def ms(n: float) -> int:
            return round(n * sample_rate / 1000 / frame)

        self.threshold = threshold
        self.silence_frames = max(1, ms(silence_ms))
        self.min_speech_frames = ms(min_speech_ms)
        self.max_frames = max(1, ms(max_speech_s * 1000))
        self.pad_frames = ms(pad_ms)
        self.reset()

    def reset(self) -> None:
        self.preroll: deque[np.ndarray] = deque(maxlen=self.pad_frames)
        self.frames: list[np.ndarray] = []
        self.speech = 0    # trames de parole dans le segment courant
        self.silence = 0   # trames de silence consécutives

    @property
    def active(self) -> bool:
        return bool(self.frames)

    def push(self, frame: np.ndarray, prob: float) -> np.ndarray | None:
        is_speech = prob >= self.threshold
        if not self.frames:
            if is_speech:
                self.frames = [*self.preroll, frame]
                self.speech, self.silence = 1, 0
            else:
                self.preroll.append(frame)
            return None
        self.frames.append(frame)
        if is_speech:
            self.speech += 1
            self.silence = 0
        else:
            self.silence += 1
        if self.silence >= self.silence_frames or len(self.frames) >= self.max_frames:
            return self.flush()
        return None

    def flush(self) -> np.ndarray | None:
        """Termine le segment courant (garde ``pad_ms`` de silence final)."""
        frames, speech = self.frames, self.speech
        trailing = max(0, self.silence - self.pad_frames)
        self.reset()
        if speech < self.min_speech_frames:
            return None
        return np.concatenate(frames[: len(frames) - trailing])


class EnergyVAD:
    """VAD minimaliste : probabilité dérivée du niveau RMS (en dBFS)."""

    def __init__(self, threshold_db: float = -45.0) -> None:
        self.threshold_db = threshold_db

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame**2)) + 1e-10)
        return 1.0 if 20 * np.log10(rms) >= self.threshold_db else 0.0

    def reset(self) -> None:
        pass


class SileroVAD:
    """Silero VAD (modèle ONNX livré avec faster-whisper), en flux trame par trame."""

    CONTEXT = 64

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self.session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame])[None, :].astype(np.float32)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = frame[-self.CONTEXT :]
        return float(np.asarray(out).reshape(-1)[0])


def _normalize(text: str) -> str:
    return re.sub(r"[^\w' -]+", "", text.lower()).strip()


def strip_wake_word(text: str, wake_words: list[str]) -> str | None:
    """Renvoie le texte sans le mot d'éveil, ou None s'il n'y en a pas."""
    norm = _normalize(text)
    for word in wake_words:
        w = _normalize(word)
        if norm == w or norm.startswith(w + " "):
            # Retire le mot d'éveil (et la ponctuation qui le suit) du texte original.
            m = re.match(rf"\W*{re.escape(word)}\W*", text, re.IGNORECASE)
            return text[m.end():].strip() if m else text
    return None


@register("stt", "whisper")
class WhisperSTT(STTEngine):
    def __init__(
        self,
        model: str = "small",
        language: str | None = "fr",       # None = détection automatique
        device: str = "cpu",               # "cpu", "cuda" ou "auto"
        compute_type: str = "int8",        # "int8" (CPU), "float16" (GPU), ...
        beam_size: int = 5,
        initial_prompt: str | None = None,
        input_device: int | str | None = None,   # micro sounddevice (index ou nom)
        vad: str = "silero",               # "silero" ou "energy"
        vad_threshold: float = 0.5,        # proba Silero (ignoré pour "energy")
        energy_threshold_db: float = -45.0,
        silence_ms: int = 700,             # pause qui termine une phrase
        min_speech_ms: int = 300,          # en dessous : ignoré (clics, toux)
        max_speech_s: float = 20.0,
        pad_ms: int = 200,                 # marge audio avant/après la parole
        no_speech_threshold: float = 0.6,  # rejette les segments jugés « sans parole »
        wake_words: list[str] | None = None,   # ex. ["Seldon"] : n'accepte que ce qui commence ainsi
        ignore: list[str] | None = None,   # phrases à jeter (hallucinations)
    ) -> None:
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self.input_device = input_device
        self.vad_kind = vad
        self.vad_threshold = vad_threshold if vad == "silero" else 0.5
        self.energy_threshold_db = energy_threshold_db
        self.segmenter = Segmenter(self.vad_threshold, silence_ms, min_speech_ms, max_speech_s, pad_ms)
        self.no_speech_threshold = no_speech_threshold
        self.wake_words = wake_words or []
        self.ignore = {_normalize(s) for s in (DEFAULT_IGNORE if ignore is None else ignore)}

        self.model: Any = None
        self.vad: Callable[[np.ndarray], float] | Any = None
        self._stream: Any = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._segments: asyncio.Queue[np.ndarray | None] | None = None
        self._paused = threading.Event()
        self._stopping = threading.Event()
        self._epoch = 0   # incrémenté à chaque pause : le thread VAD jette le segment en cours
        self._worker: threading.Thread | None = None

    # -- cycle de vie -------------------------------------------------------

    async def start(self) -> None:
        self.model, self.vad = await asyncio.to_thread(self._load)
        self._segments = asyncio.Queue()
        self._stopping.clear()
        loop = asyncio.get_running_loop()
        self._worker = threading.Thread(target=self._vad_loop, args=(loop,), name="seldon-vad", daemon=True)
        self._worker.start()
        self._open_stream()
        log.info("Whisper '%s' prêt (%s/%s), écoute du micro", self.model_name, self.device, self.compute_type)

    def _load(self) -> tuple[Any, Any]:
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        vad = SileroVAD() if self.vad_kind == "silero" else EnergyVAD(self.energy_threshold_db)
        return model, vad

    def _open_stream(self) -> None:
        import sounddevice as sd

        def callback(indata, _frames, _time, status) -> None:
            if status:
                log.debug("sounddevice : %s", status)
            if not self._paused.is_set():
                self._frames.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME,
            device=self.input_device, callback=callback,
        )
        self._stream.start()

    async def stop(self) -> None:
        self._stopping.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._worker is not None:
            await asyncio.to_thread(self._worker.join, 2)
            self._worker = None
        if self._segments is not None:
            self._segments.put_nowait(None)

    def pause(self) -> None:
        self._paused.set()
        self._epoch += 1

    def resume(self) -> None:
        # Vide ce qui a pu être capturé juste avant la pause.
        self._epoch += 1
        with self._frames.mutex:
            self._frames.queue.clear()
        self._paused.clear()

    # -- thread VAD -----------------------------------------------------------

    def _vad_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        epoch, pending = self._epoch, np.zeros(0, dtype=np.float32)
        while not self._stopping.is_set():
            try:
                block = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            if epoch != self._epoch or self._paused.is_set():
                epoch, pending = self._epoch, np.zeros(0, dtype=np.float32)
                self.segmenter.reset()
                self.vad.reset()
                if self._paused.is_set():
                    continue
            pending = np.concatenate([pending, block])
            while len(pending) >= FRAME:
                frame, pending = pending[:FRAME], pending[FRAME:]
                segment = self.segmenter.push(frame, self.vad(frame))
                if segment is not None:
                    loop.call_soon_threadsafe(self._segments.put_nowait, segment)

    # -- transcription --------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> tuple[str, str | None]:
        """Transcrit un segment (bloquant). Renvoie (texte, langue)."""
        segments, info = self.model.transcribe(
            audio, language=self.language, beam_size=self.beam_size,
            initial_prompt=self.initial_prompt, condition_on_previous_text=False, vad_filter=False,
        )
        parts = [s.text for s in segments if s.no_speech_prob < self.no_speech_threshold]
        return " ".join(p.strip() for p in parts).strip(), info.language

    def accept(self, text: str) -> str | None:
        """Filtre le texte transcrit (hallucinations, mot d'éveil). None = à ignorer."""
        if not text or _normalize(text) in self.ignore:
            return None
        if self.wake_words:
            text = strip_wake_word(text, self.wake_words)
        return text or None

    async def listen(self) -> AsyncIterator[Utterance]:
        if self._segments is None:
            await self.start()
        while (audio := await self._segments.get()) is not None:
            if self._paused.is_set():
                continue
            text, language = await asyncio.to_thread(self.transcribe, audio)
            text = self.accept(text)
            if text is None:
                log.debug("Segment ignoré (%.1f s)", len(audio) / SAMPLE_RATE)
                continue
            yield Utterance(text=text, language=language, audio=audio)
