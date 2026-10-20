"""Identification du locuteur par empreinte vocale (Resemblyzer).

Chaque personne enregistrée a un fichier ``<voices_dir>/<nom>.npy`` contenant
ses embeddings d'enrôlement (un par ligne, 256 dimensions). L'empreinte est
leur moyenne normalisée ; on compare l'audio d'une ``Utterance`` à chaque
empreinte par similarité cosinus.

Les segments des voix inconnues des dernières minutes sont gardés en mémoire :
``enroll_recent(nom)`` enregistre ceux qui ressemblent à la dernière voix
inconnue entendue (c'est ainsi que Seldon apprend la voix de quelqu'un qui se
présente).

Les voix s'affinent avec l'usage : une phrase reconnue avec certitude (score
≥ ``adapt_threshold`` et nettement devant la 2e voix) est ajoutée à l'empreinte
de la personne, au plus une fois par ``adapt_interval_s`` ; seuls les
``max_samples`` échantillons les plus récents sont gardés, si bien que
l'empreinte suit les changements de micro, de pièce ou de voix (rhume…).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seldon.core.registry import register
from seldon.core.types import Utterance
from seldon.speakers.base import SpeakerIdentifier

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
ADAPT_MARGIN = 0.1   # écart minimal avec la 2e voix pour apprendre d'une phrase


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", name.strip()).strip("._")
    if not safe:
        raise ValueError(f"Nom de locuteur invalide : {name!r}")
    return safe


@dataclass
class _Heard:
    """Segment d'une voix inconnue, gardé pour un éventuel enregistrement."""

    embedding: np.ndarray
    ts: float


@register("speakers", "resemblyzer")
class ResemblyzerID(SpeakerIdentifier):
    def __init__(
        self,
        voices_dir: str = "data/voices",
        threshold: float = 0.75,     # similarité cosinus minimale pour reconnaître quelqu'un
        device: str = "cpu",         # "cpu" ou "cuda"
        min_seconds: float = 0.5,    # audio plus court : pas d'identification
        unknown_memory_s: float = 600,   # durée pendant laquelle une voix inconnue peut être enregistrée
        adapt_threshold: float | None = 0.85,   # score pour affiner une voix reconnue ; None = jamais
        adapt_interval_s: float = 60,   # au plus un échantillon appris par personne et par intervalle
        max_samples: int = 40,          # échantillons gardés par voix (les plus récents)
    ) -> None:
        self.voices_dir = Path(voices_dir)
        self.threshold = threshold
        self.device = device
        self.min_seconds = min_seconds
        self.unknown_memory_s = unknown_memory_s
        self.unknown: list[_Heard] = []   # voix inconnues récentes, la plus récente en dernier
        self.adapt_threshold = adapt_threshold
        self.adapt_interval_s = adapt_interval_s
        self.max_samples = max_samples
        self._adapted: dict[str, float] = {}   # nom -> instant du dernier échantillon appris
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

    def scores(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Similarité avec chaque empreinte, la meilleure d'abord."""
        names = list(self.voiceprints)
        if not names:
            return []
        values = np.stack([self.voiceprints[n] for n in names]) @ _unit(embedding)
        return sorted(((n, float(np.clip(v, 0.0, 1.0))) for n, v in zip(names, values)), key=lambda x: -x[1])

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Renvoie (nom, score) de la meilleure empreinte, nom=None sous le seuil."""
        ranked = self.scores(embedding)
        if not ranked:
            return None, 0.0
        name, score = ranked[0]
        return (name if score >= self.threshold else None), score

    async def identify(self, utterance: Utterance) -> Utterance:
        audio = utterance.audio
        if audio is None or len(audio) < self.min_seconds * SAMPLE_RATE:
            return utterance
        embedding = await asyncio.to_thread(self.embed, audio)
        ranked = self.scores(embedding)
        utterance.speaker, utterance.speaker_confidence = self.match(embedding)
        utterance.unknown_voice = utterance.speaker is None
        if utterance.unknown_voice:
            self._remember_unknown(_Heard(embedding, utterance.timestamp))
        elif self._should_adapt(ranked, utterance.timestamp):
            self._adapted[utterance.speaker] = utterance.timestamp
            self._add_embeddings(utterance.speaker, embedding[None, :])
        log.debug("Locuteur : %s (%.2f)", utterance.speaker or "inconnu", utterance.speaker_confidence)
        return utterance

    def _should_adapt(self, ranked: list[tuple[str, float]], now: float) -> bool:
        if self.adapt_threshold is None or not ranked:
            return False
        name, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        return (
            score >= self.adapt_threshold
            and score - runner_up >= ADAPT_MARGIN
            and now - self._adapted.get(name, float("-inf")) >= self.adapt_interval_s
        )

    def _remember_unknown(self, heard: _Heard) -> None:
        self.unknown = [h for h in self.unknown if heard.ts - h.ts <= self.unknown_memory_s] + [heard]

    async def enroll(self, name: str, audio_samples: list) -> None:
        """Ajoute les échantillons aux embeddings existants de ``name`` et sauvegarde."""
        if not audio_samples:
            raise ValueError("Aucun échantillon audio fourni")
        if self.encoder is None:
            await self.start()
        self._add_embeddings(name, await asyncio.to_thread(lambda: np.stack([self.embed(a) for a in audio_samples])))
        log.info("Voix de %s enregistrée (%d échantillon(s))", name, len(audio_samples))

    async def enroll_recent(self, name: str) -> int:
        """Enregistre sous ``name`` les segments inconnus récents proches du dernier entendu."""
        now = time.time()
        self.unknown = [h for h in self.unknown if now - h.ts <= self.unknown_memory_s]
        if not self.unknown:
            return 0
        last = self.unknown[-1].embedding
        same = [float(h.embedding @ last) >= self.threshold for h in self.unknown]
        self._add_embeddings(name, np.stack([h.embedding for h, s in zip(self.unknown, same) if s]))
        self.unknown = [h for h, s in zip(self.unknown, same) if not s]
        log.info("Voix de %s enregistrée à partir de %d phrase(s) entendue(s)", name, sum(same))
        return sum(same)

    def known_speakers(self) -> list[str]:
        return list(self.voiceprints)

    def forget(self, name: str) -> bool:
        path = self.voices_dir / f"{_safe_name(name)}.npy"
        self.voiceprints.pop(path.stem, None)
        if not path.is_file():
            return False
        path.unlink()
        log.info("Voix de %s supprimée", path.stem)
        return True

    def _add_embeddings(self, name: str, new: np.ndarray) -> None:
        path = self.voices_dir / f"{_safe_name(name)}.npy"
        if path.exists():
            new = np.concatenate([np.atleast_2d(np.load(path)), new])
        new = new[-self.max_samples:]
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, new.astype(np.float32))
        self.voiceprints[path.stem] = _unit(new.mean(axis=0))
        log.debug("Voix de %s enregistrée (%d échantillon(s))", path.stem, len(new))
