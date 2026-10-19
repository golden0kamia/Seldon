"""Enrôlement d'une voix : ``python -m seldon.tools.enroll NOM [--seconds N] [--samples K]``.

Enregistre K échantillons de N secondes au micro puis appelle ``enroll()`` du
moteur d'identification configuré (section ``speakers`` de config.yaml si elle
existe, sinon Resemblyzer avec ses options par défaut).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import numpy as np

from seldon.config import Config, StageConfig
from seldon.core import registry

SAMPLE_RATE = 16000

PHRASES = [
    "Bonjour Seldon, je m'appelle {name} et voici ma voix.",
    "Le vif renard brun saute par-dessus le chien paresseux.",
    "Quel temps fera-t-il demain matin à la maison ?",
    "Peux-tu allumer la lumière du salon, s'il te plaît ?",
    "J'aime écouter de la musique en préparant le dîner.",
]


def record(seconds: float, device: int | str | None = None) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    return audio[:, 0]


def speaker_stage(config: str, engine: str | None, voices_dir: str | None) -> StageConfig:
    stage = StageConfig("resemblyzer")
    if Path(config).exists():
        configured = Config.load(config).speakers
        if configured.engine != "none":
            stage = configured
    if engine:
        stage = StageConfig(engine, stage.options if engine == stage.engine else {})
    if voices_dir:
        stage.options["voices_dir"] = voices_dir
    return stage


async def run(args: argparse.Namespace) -> None:
    stage = speaker_stage(args.config, args.engine, args.voices_dir)
    identifier = registry.create("speakers", stage.engine, **stage.options)
    await identifier.start()

    samples = []
    for i in range(args.samples):
        phrase = PHRASES[i % len(PHRASES)].format(name=args.name)
        input(f"\n[{i + 1}/{args.samples}] Appuyez sur Entrée puis lisez à voix haute pendant {args.seconds:g} s :\n  « {phrase} »")
        print("  Enregistrement...")
        audio = await asyncio.to_thread(record, args.seconds, args.input_device)
        level = float(np.sqrt(np.mean(audio**2)))
        if level < 1e-3:
            print("  Attention : niveau très faible, le micro est-il bien branché ?")
        samples.append(audio)

    await identifier.enroll(args.name, samples)
    print(f"\nVoix de {args.name} enregistrée ({len(samples)} échantillon(s)).")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seldon.tools.enroll", description="Enregistre la voix d'une personne")
    parser.add_argument("name", help="nom du locuteur (tel que Seldon l'appellera)")
    parser.add_argument("--seconds", type=float, default=5.0, help="durée de chaque échantillon")
    parser.add_argument("--samples", type=int, default=3, help="nombre d'échantillons")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--engine", help="moteur d'identification (défaut : celui de la config, sinon resemblyzer)")
    parser.add_argument("--voices-dir", help="dossier des empreintes vocales")
    parser.add_argument("--input-device", help="micro sounddevice (index ou nom)")
    args = parser.parse_args()
    if args.input_device and args.input_device.isdigit():
        args.input_device = int(args.input_device)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nAnnulé.")


if __name__ == "__main__":
    main()
