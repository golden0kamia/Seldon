"""Qui parle ? Identification par empreinte vocale (Resemblyzer), et l'outil ``voix``.

Chaque personne a un fichier ``<voices_dir>/<nom>.npy`` : ses échantillons de
voix (un vecteur de 256 nombres par phrase). Son empreinte est leur moyenne ;
une phrase est attribuée à l'empreinte la plus proche si la ressemblance
dépasse ``threshold``.

Les phrases des voix inconnues des dernières minutes sont gardées : quand la
personne dit son nom, l'outil ``voix`` les enregistre sous ce nom.

Sécurité : les noms viennent du cerveau, ils sont réduits à un nom de fichier
sûr ; ajouter une voix à un nom déjà connu demande une confirmation (sinon,
dire « je suis Bastien » suffirait à obtenir les droits de Bastien). Les
fichiers ``.npy`` sont lus sans pickle et écrits en privé.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from seldon.base import Context, Result, Speakers, Tool, Utterance
from seldon.storage import write_private

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
DIM = 256   # taille d'un échantillon Resemblyzer


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


def safe_name(name: str) -> str:
    """Nom utilisable comme nom de fichier ; ValueError s'il est vide ou trop long."""
    text = unicodedata.normalize("NFKC", str(name)).strip()
    safe = re.sub(r"[^\w.-]+", "_", text).strip("._")
    if not safe or len(safe) > 64:
        raise ValueError(f"nom de locuteur invalide : {str(name)[:40]!r}")
    return safe


def read_samples(path: Path) -> np.ndarray:
    """Échantillons d'un fichier ``.npy`` (jamais de pickle) ; ValueError s'il est invalide."""
    samples = np.atleast_2d(np.load(path, allow_pickle=False))
    if samples.ndim != 2 or samples.shape[1] != DIM or not np.isfinite(samples).all():
        raise ValueError(f"empreinte invalide {samples.shape}")
    return samples.astype(np.float32)


class Resemblyzer(Speakers):
    def __init__(
        self,
        voices_dir: str = "data/voices",
        threshold: float = 0.75,        # ressemblance minimale pour reconnaître quelqu'un
        unknown_memory_s: float = 600,  # une voix inconnue reste enregistrable 10 min
        max_samples: int = 40,          # échantillons gardés par voix (les plus récents)
    ) -> None:
        self.voices_dir = Path(voices_dir).expanduser()
        self.threshold = threshold
        self.unknown_memory_s = unknown_memory_s
        self.max_samples = max_samples
        self.encoder: Any = None
        self.voiceprints: dict[str, np.ndarray] = {}         # nom -> empreinte
        self.unknown: list[tuple[float, np.ndarray]] = []    # (heure, échantillon) des voix inconnues

    async def start(self) -> None:
        from resemblyzer import VoiceEncoder

        self.encoder = await asyncio.to_thread(VoiceEncoder, "cpu", verbose=False)
        for path in sorted(self.voices_dir.glob("*.npy")):
            try:
                self.voiceprints[path.stem] = _unit(read_samples(path).mean(axis=0))
            except (OSError, ValueError) as e:
                log.warning("Empreinte ignorée : %s (%s)", path, e)
        log.info("Voix connues : %s", ", ".join(self.voiceprints) or "aucune")

    def tools(self) -> list[Tool]:
        return [Voices()]

    def embed(self, audio: np.ndarray) -> np.ndarray:
        from resemblyzer import preprocess_wav

        wav = preprocess_wav(np.asarray(audio, dtype=np.float32), source_sr=SAMPLE_RATE)
        return _unit(self.encoder.embed_utterance(wav if len(wav) > SAMPLE_RATE * 0.2 else audio))

    async def identify(self, utterance: Utterance) -> None:
        if utterance.audio is None or len(utterance.audio) < SAMPLE_RATE * 0.5:
            return   # trop court pour reconnaître une voix
        sample = await asyncio.to_thread(self.embed, utterance.audio)
        scores = {name: float(vp @ sample) for name, vp in self.voiceprints.items()}
        best = max(scores, key=scores.get, default=None)
        if best is not None and scores[best] >= self.threshold:
            utterance.speaker = best
        else:
            utterance.unknown_voice = True
            now = time.time()
            self.unknown = [(t, s) for t, s in self.unknown if now - t < self.unknown_memory_s][-99:] + [(now, sample)]
        log.debug("Locuteur : %s (%s)", utterance.speaker or "inconnu", scores)

    def known(self) -> list[str]:
        return sorted({p.stem for p in self.voices_dir.glob("*.npy")} | set(self.voiceprints), key=str.casefold)

    def canonical(self, name: str) -> str:
        """Nom de fichier sûr, avec la casse d'une voix déjà connue (« bastien » -> « Bastien »)."""
        safe = safe_name(name)
        return next((k for k in self.known() if k.casefold() == safe.casefold()), safe)

    def add_samples(self, name: str, samples: np.ndarray) -> None:
        path = self.voices_dir / f"{name}.npy"
        if path.exists():
            samples = np.concatenate([read_samples(path), samples])
        samples = np.asarray(samples[-self.max_samples:], dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, samples, allow_pickle=False)
        write_private(path, buf.getvalue())
        self.voiceprints[name] = _unit(samples.mean(axis=0))

    def enroll_recent(self, name: str) -> int:
        """Enregistre sous ``name`` les phrases inconnues récentes qui ressemblent à la dernière."""
        now = time.time()
        recent = [(t, s) for t, s in self.unknown if now - t < self.unknown_memory_s]
        if not recent:
            return 0
        last = recent[-1][1]
        same = [s for _, s in recent if float(s @ last) >= self.threshold]
        self.add_samples(name, np.stack(same))
        self.unknown = [(t, s) for t, s in recent if float(s @ last) < self.threshold]
        return len(same)

    def forget(self, name: str) -> bool:
        self.voiceprints.pop(name, None)
        path = self.voices_dir / f"{name}.npy"
        if not path.is_file():
            return False
        path.unlink()
        return True


class Voices(Tool):
    name = "voix"
    description = (
        "Les voix que tu sais reconnaître. Quand le locuteur est « voix inconnue », réponds à sa demande "
        "puis demande-lui son prénom (une seule fois). "
        "operation='enregistrer' : dès qu'il donne son nom, enregistre sous `nom` la dernière voix inconnue entendue. "
        "operation='lister' : voix enregistrées. "
        "operation='oublier' : supprime la voix `nom` (l'utilisateur doit confirmer)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["enregistrer", "lister", "oublier"]},
            "nom": {"type": "string", "description": "Prénom de la personne."},
        },
        "required": ["operation"],
    }

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        speakers: Resemblyzer = ctx.app.speakers
        op = args.get("operation")
        if op == "lister":
            return Result(f"Voix enregistrées : {', '.join(speakers.known()) or 'aucune'}.")
        try:
            name = speakers.canonical(args.get("nom") or "")
        except ValueError:
            return Result.fail("Donne un prénom simple dans `nom`.")

        if op == "enregistrer":
            if name in speakers.known() and not await ctx.confirm(
                    f"Une voix non reconnue dit être {name}, déjà enregistré(e). L'ajouter à sa voix ? "
                    f"Elle serait reconnue comme {name}, avec les mêmes droits."):
                return Result.fail(f"Refusé : ne considère pas ce locuteur comme {name}.")
            count = speakers.enroll_recent(name)
            if not count:
                return Result.fail("Aucune voix inconnue entendue récemment.")
            ctx.utterance.speaker = name
            return Result(f"Voix de {name} enregistrée ({count} phrase(s)).")

        if op == "oublier":
            if not await ctx.confirm(f"Supprimer la voix de {name} ?"):
                return Result.fail("Suppression refusée par l'utilisateur.")
            return Result(f"Voix de {name} supprimée.") if speakers.forget(name) else Result.fail(f"Aucune voix « {name} ».")
        return Result.fail(f"Opération inconnue : {op}")
