"""Identification du locuteur par empreinte vocale (Resemblyzer).

Chaque personne enregistrée a un fichier ``<voices_dir>/<nom>.npy`` contenant
ses embeddings d'enrôlement (un par ligne, 256 dimensions). L'empreinte est
leur moyenne normalisée ; on compare l'audio d'une ``Utterance`` à chaque
empreinte par similarité cosinus.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from seldon.core.registry import register
from seldon.core.types import Utterance
from seldon.speakers.base import SpeakerIdentifier

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    if not safe:
        raise ValueError(f"Nom de locuteur invalide : {name!r}")
    return safe


@register("speakers", "resemblyzer")
class ResemblyzerID(SpeakerIdentifier):
    def __init__(
        self,
        voices_dir: str = "data/voices",
        threshold: float = 0.75,     # similarité cosinus minimale pour reconnaître quelqu'un
        device: str = "cpu",         # "cpu" ou "cuda"
        min_seconds: float = 0.5,    # audio plus court : pas d'identification
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self.threshold = threshold
        self.device = device
        self.min_seconds = min_seconds
        self.encoder: Any = None
        self.voiceprints: dict[str, np.ndarray] = {}

    async def start(self) -> None:
        if self.encoder is None:
            self.encoder = await asyncio.to_thread(self._load_encoder)
        self.load_voiceprints()

    def _load_encoder(self) -> Any:
        from resemblyzer import VoiceEncoder

        return VoiceEncoder(self.device, verbose=False)

    def load_voiceprints(self) -> None:
        self.voiceprints = {
            path.stem: _unit(np.atleast_2d(np.load(path)).mean(axis=0))
            for path in sorted(self.voices_dir.glob("*.npy"))
        }
        log.info("%d empreinte(s) vocale(s) chargée(s) : %s", len(self.voiceprints), ", ".join(self.voiceprints) or "-")

    def embed(self, audio: np.ndarray) -> np.ndarray:
        """Embedding normalisé d'un segment float32 16 kHz (bloquant)."""
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(np.asarray(audio, dtype=np.float32), source_sr=SAMPLE_RATE)
        if len(wav) < SAMPLE_RATE * 0.2:   # le prétraitement retire les silences
            wav = np.asarray(audio, dtype=np.float32)
        return _unit(self.encoder.embed_utterance(wav))

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Renvoie (nom, score) de la meilleure empreinte, nom=None sous le seuil."""
        if not self.voiceprints:
            return None, 0.0
        names = list(self.voiceprints)
        scores = np.stack([self.voiceprints[n] for n in names]) @ _unit(embedding)
        best = int(np.argmax(scores))
        score = float(np.clip(scores[best], 0.0, 1.0))
        return (names[best] if score >= self.threshold else None), score

    async def identify(self, utterance: Utterance) -> Utterance:
        audio = utterance.audio
        if audio is None or len(audio) < self.min_seconds * SAMPLE_RATE or not self.voiceprints:
            return utterance
        embedding = await asyncio.to_thread(self.embed, audio)
        utterance.speaker, utterance.speaker_confidence = self.match(embedding)
        log.debug("Locuteur : %s (%.2f)", utterance.speaker or "inconnu", utterance.speaker_confidence)
        return utterance

    async def enroll(self, name: str, audio_samples: list) -> None:
        """Ajoute les échantillons aux embeddings existants de ``name`` et sauvegarde."""
        if not audio_samples:
            raise ValueError("Aucun échantillon audio fourni")
        if self.encoder is None:
            await self.start()
        new = await asyncio.to_thread(lambda: np.stack([self.embed(a) for a in audio_samples]))
        path = self.voices_dir / f"{_safe_name(name)}.npy"
        if path.exists():
            new = np.concatenate([np.atleast_2d(np.load(path)), new])
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, new.astype(np.float32))
        self.voiceprints[path.stem] = _unit(new.mean(axis=0))
        log.info("Voix de %s enregistrée (%d échantillon(s))", path.stem, len(new))
