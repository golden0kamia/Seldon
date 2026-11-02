"""Enregistrer une voix au micro : ``python -m seldon.cli.enroll Bastien [--samples 3] [--seconds 5]``.

Utilise les options de la section ``speakers`` de config.yaml (dossier des voix…).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np

from seldon.config import Config
from seldon.ear.speakers import Resemblyzer

SAMPLE_RATE = 16000
PHRASES = [
    "Bonjour Seldon, je m'appelle {name} et voici ma voix.",
    "Quel temps fera-t-il demain matin à la maison ?",
    "Peux-tu allumer la lumière du salon, s'il te plaît ?",
    "J'aime écouter de la musique en préparant le dîner.",
]


async def enroll(name: str, samples: int, seconds: float, config: str) -> None:
    import sounddevice as sd

    cfg = Config.load(config) if Path(config).exists() else None
    speakers = Resemblyzer(**(cfg.speakers.options if cfg and cfg.speakers else {}))
    await speakers.start()
    name = speakers.canonical(name)
    recordings = []
    for i in range(samples):
        input(f"\n[{i + 1}/{samples}] Entrée, puis lisez pendant {seconds:g} s :\n  « {PHRASES[i % len(PHRASES)].format(name=name)} »")
        audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        recordings.append(audio[:, 0])
    speakers.add_samples(name, np.stack([speakers.embed(a) for a in recordings]))
    print(f"\nVoix de {name} enregistrée.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seldon.cli.enroll", description="Enregistre la voix d'une personne")
    parser.add_argument("name")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    asyncio.run(enroll(args.name, args.samples, args.seconds, args.config))


if __name__ == "__main__":
    main()
