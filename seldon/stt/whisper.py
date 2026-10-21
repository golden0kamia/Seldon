"""STT micro : capture sounddevice + détection d'activité vocale + faster-whisper.

Un thread de capture découpe le flux micro (16 kHz mono float32) en trames
de 32 ms, les passe au VAD (Silero, fourni avec faster-whisper, ou un simple
seuil d'énergie) et pousse chaque prise de parole terminée dans une file
asyncio. ``listen()`` transcrit ensuite chaque segment dans un thread.

Avec des mots d'éveil, chaque phrase est marquée ``addressed`` (mot d'éveil
prononcé ou non) ; c'est le pipeline qui décide d'y répondre.

Coupure de parole (``barge_in``) : pendant que Seldon parle, l'écoute est en
pause mais le VAD continue sur les sons assez forts (``barge_in_db``). Une
phrase contenant un mot d'interruption (« stop », « attends »… ou un mot
d'éveil) absent de ce que dit Seldon — pour ne pas réagir à son propre écho —
le fait taire ; si elle dit autre chose (« attends, plutôt demain »), elle est
ensuite traitée comme une prise de parole normale.

Micro coupé (``mute``, bouton de la page Journal) : tout le son capté est jeté,
y compris pendant que Seldon parle, jusqu'à ce qu'il soit rétabli.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import re
import threading
from collections import deque
from typing import Any, AsyncIterator, Callable

import numpy as np

from seldon.core.errors import log_error
from seldon.core import tags
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

MIC_TAG = tags.tag(tags.MIC, "")   # forme sans jeton ; voir seldon.core.tags

DEFAULT_INTERRUPT = ["stop", "arrête", "arrêtez", "attends", "attendez", "tais-toi", "chut", "silence", "stoppe"]


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


def _level_db(frame: np.ndarray) -> float:
    return 20 * float(np.log10(np.sqrt(np.mean(frame**2)) + 1e-10))


def contains_phrase(text: str, phrases: list[str]) -> list[str]:
    """Phrases (mots ou groupes de mots) présentes dans ``text``, à la casse et ponctuation près."""
    norm = f" {' '.join(_normalize(text).replace('-', ' ').split())} "
    return [p for p in phrases if f" {' '.join(_normalize(p).replace('-', ' ').split())} " in norm]


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
        cpu_threads: int | None = None,    # threads CPU de Whisper ; None = auto (moitié des cœurs logiques, 4 à 8)
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
        wake_words: list[str] | None = None,   # ex. ["Seldon"] : marque les phrases qui commencent ainsi
        ignore: list[str] | None = None,   # phrases à jeter (hallucinations)
        barge_in: bool = True,             # écoute pendant que Seldon parle, pour pouvoir l'interrompre
        barge_in_db: float = -35.0,        # niveau minimal (dBFS) d'une interruption (au-dessus de l'écho)
        interrupt_words: list[str] | None = None,   # défaut : stop, attends, tais-toi… (+ mots d'éveil)
    ) -> None:
        self.model_name = model
        self.language = language
        self.device = device
        self.compute_type = compute_type
        # faster-whisper prend 4 threads par défaut ; ~8 est le plus rapide sur un CPU de bureau
        # (au-delà, les cœurs logiques n'aident plus).
        self.cpu_threads = cpu_threads if cpu_threads is not None else max(4, min(8, (os.cpu_count() or 8) // 2))
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
        self.barge_in = barge_in
        self.barge_in_db = barge_in_db
        self.interrupt_words = [*(DEFAULT_INTERRUPT if interrupt_words is None else interrupt_words), *self.wake_words]
        self.barge_segmenter = Segmenter(self.vad_threshold, silence_ms=300, min_speech_ms=200, max_speech_s=5, pad_ms=100)

        self.model: Any = None
        self.vad: Callable[[np.ndarray], float] | Any = None
        self._stream: Any = None
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._segments: asyncio.Queue[np.ndarray | Utterance | None] | None = None
        self._barge: asyncio.Queue[np.ndarray] | None = None   # segments entendus pendant que Seldon parle
        self._paused = threading.Event()
        self._pauses = 0   # pauses en cours (ex. accusé de réception + annonce de fond) : reprise à 0
        self._muted = threading.Event()
        self._stopping = threading.Event()
        self._epoch = 0   # incrémenté à chaque pause : le thread VAD jette le segment en cours
        self._worker: threading.Thread | None = None

    # -- cycle de vie -------------------------------------------------------

    async def start(self) -> None:
        self.model, self.vad = await asyncio.to_thread(self._load)
        self._segments = asyncio.Queue()
        self._barge = asyncio.Queue()
        self._stopping.clear()
        loop = asyncio.get_running_loop()
        self._worker = threading.Thread(target=self._vad_loop, args=(loop,), name="seldon-vad", daemon=True)
        self._worker.start()
        self._open_stream()
        log.info("Whisper '%s' prêt (%s/%s), écoute du micro", self.model_name, self.device, self.compute_type)

    def _load(self) -> tuple[Any, Any]:
        from faster_whisper import WhisperModel

        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type,
                             cpu_threads=self.cpu_threads)
        vad = SileroVAD() if self.vad_kind == "silero" else EnergyVAD(self.energy_threshold_db)
        return model, vad

    def _open_stream(self) -> None:
        import sounddevice as sd

        def callback(indata, _frames, _time, status) -> None:
            if status:
                log.debug("sounddevice : %s", status)
            if self._muted.is_set():
                return
            if self.barge_in or not self._paused.is_set():
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
        self._pauses += 1
        self._paused.set()
        self._epoch += 1

    def resume(self) -> None:
        self._pauses = max(0, self._pauses - 1)
        if self._pauses:   # Seldon parle encore (deux prises de parole se chevauchent)
            return
        # Vide ce qui a pu être capturé juste avant la pause.
        self._epoch += 1
        with self._frames.mutex:
            self._frames.queue.clear()
        self._paused.clear()

    def mute(self, on: bool) -> None:
        if on == self._muted.is_set():
            return
        if on:
            self._muted.set()
        else:
            with self._frames.mutex:
                self._frames.queue.clear()
            self._muted.clear()
        self._epoch += 1   # le thread VAD jette le segment en cours
        log.info("%s %s", tags.tag(tags.MIC), "coupé" if on else "rétabli")

    # -- thread VAD -----------------------------------------------------------

    def _vad_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        epoch, pending = self._epoch, np.zeros(0, dtype=np.float32)
        failing = False   # une erreur n'est journalisée qu'une fois, jusqu'au retour à la normale
        while not self._stopping.is_set():
            try:
                block = self._frames.get(timeout=0.1)
            except queue.Empty:
                continue
            paused = self._paused.is_set()
            if epoch != self._epoch:
                epoch, pending = self._epoch, np.zeros(0, dtype=np.float32)
                self.segmenter.reset()
                self.barge_segmenter.reset()
                self.vad.reset()
            if paused and not (self.barge_in and self._barge is not None):
                continue
            pending = np.concatenate([pending, block]) if len(pending) else block
            try:
                while len(pending) >= FRAME:
                    frame, pending = pending[:FRAME], pending[FRAME:]
                    prob = self.vad(frame)
                    if not paused:
                        segment = self.segmenter.push(frame, prob)
                        if segment is not None:
                            loop.call_soon_threadsafe(self._segments.put_nowait, segment)
                        continue
                    # Seldon parle : seuls les sons plus forts que son écho comptent.
                    segment = self.barge_segmenter.push(frame, prob if _level_db(frame) >= self.barge_in_db else 0.0)
                    if segment is not None:
                        loop.call_soon_threadsafe(self._barge.put_nowait, segment)
                failing = False
            except RuntimeError as e:
                if loop.is_closed():   # arrêt de Seldon
                    return
                failing = self._vad_error(e, failing)
            except Exception as e:   # sans cela, le thread mourrait et Seldon deviendrait sourd
                failing = self._vad_error(e, failing)

    def _vad_error(self, exc: Exception, failing: bool) -> bool:
        if not failing:
            log_error(log, "Détection de la parole (VAD) en échec", exc)
        self.segmenter.reset()
        self.barge_segmenter.reset()
        self.vad.reset()
        return True

    # -- transcription --------------------------------------------------------

    def transcribe(self, audio: np.ndarray) -> tuple[str, str | None]:
        """Transcrit un segment (bloquant). Renvoie (texte, langue)."""
        segments, info = self.model.transcribe(
            audio, language=self.language, beam_size=self.beam_size,
            initial_prompt=self.initial_prompt, condition_on_previous_text=False, vad_filter=False,
        )
        parts = [s.text for s in segments if s.no_speech_prob < self.no_speech_threshold]
        return " ".join(p.strip() for p in parts).strip(), info.language

    def accept(self, text: str) -> tuple[str, bool | None] | None:
        """Filtre le texte transcrit (hallucinations) et repère le mot d'éveil.

        Renvoie (texte sans le mot d'éveil, mot d'éveil prononcé ?) — None pour
        ce dernier si aucun mot d'éveil n'est configuré —, ou None si à ignorer.
        """
        if not text or _normalize(text) in self.ignore:
            return None
        if not self.wake_words:
            return text, None
        stripped = strip_wake_word(text, self.wake_words)
        if stripped is None:
            return text, False
        return stripped or text, True   # « Seldon » seul : un appel, auquel il répond

    async def listen(self) -> AsyncIterator[Utterance]:
        if self._segments is None:
            await self.start()
        while (audio := await self._segments.get()) is not None:
            if isinstance(audio, Utterance):   # phrase qui a interrompu Seldon
                yield audio
                continue
            if self._paused.is_set() or self._muted.is_set():
                continue
            try:
                text, language = await asyncio.to_thread(self.transcribe, audio)
            except Exception as e:   # une phrase perdue, pas l'écoute
                log_error(log, "Transcription échouée", e)
                continue
            accepted = self.accept(text)
            if accepted is None:
                log.debug("Segment ignoré (%.1f s)", len(audio) / SAMPLE_RATE)
                continue
            text, addressed = accepted
            yield Utterance(text=text, language=language, audio=audio, addressed=addressed)

    async def interruption(self, spoken: str) -> Utterance:
        if not self.barge_in or self._barge is None:
            return await super().interruption(spoken)
        while not self._barge.empty():   # sons d'avant cette prise de parole
            self._barge.get_nowait()
        while True:
            audio = await self._barge.get()
            try:
                text, language = await asyncio.to_thread(self.transcribe, audio)
            except Exception as e:
                log_error(log, "Transcription d'une interruption échouée", e)
                continue
            if not (accepted := self.accept(text)):
                continue
            # un mot que Seldon prononce lui-même est probablement son écho
            words = [w for w in contains_phrase(text, self.interrupt_words) if not contains_phrase(spoken, [w])]
            if not words:
                log.debug("Entendu pendant la parole, ignoré : %s", text)
                continue
            log.info("Interrompu : %s", text)
            utterance = Utterance(
                text=accepted[0], language=language, audio=audio, addressed=True,
                notes=["cette phrase t'a interrompu pendant que tu parlais"],
            )
            rest = _normalize(text).replace("-", " ").split()
            interrupt = {t for w in self.interrupt_words for t in _normalize(w).replace("-", " ").split()}
            if any(len(t) > 2 and t not in interrupt for t in rest) and self._segments is not None:
                self._segments.put_nowait(utterance)   # elle demande autre chose : à traiter ensuite
            return utterance
