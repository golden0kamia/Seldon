"""Entraînement du mot d'éveil : ``python -m seldon.cli.wakeword [MOT] [--samples K]``.

Seldon repère le mot d'éveil dans le texte transcrit par Whisper ; or Whisper
écrit souvent un nom propre de travers (« Celdon », « Sel don », « Seldom »),
et pas de la même façon pour toutes les voix. Cet outil fait prononcer le mot
d'éveil K fois (seul ou en début de phrase), transcrit chaque prise avec le
modèle Whisper de config.yaml et retient les transcriptions qui diffèrent.
Il fait ensuite lire quelques phrases sans le mot d'éveil : une variante qui
apparaît aussi en tête de l'une d'elles déclencherait Seldon à tort et est
écartée. Les variantes gardées sont ajoutées au fichier ``wake_variants`` de
la section ``stt`` (``data/wake_words.json`` par défaut), lu au démarrage.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from seldon.cli.enroll import record
from seldon.config import Config
from seldon.core.errors import log_error
from seldon.core.storage import write_private
from seldon.stt.whisper import WhisperSTT, words

# Suite de phrase après le mot d'éveil ("" : le mot seul, comme pour appeler Seldon).
CALLS = ["", "quelle heure est-il ?", "", "allume la lumière du salon", "", "mets un peu de musique",
         "", "comment vas-tu ?", "", "quel temps fera-t-il demain ?"]

# Phrases sans mot d'éveil, dont certaines commencent par un son proche de « Seldon ».
OTHERS = [
    "Selon la météo, il va pleuvoir demain.",
    "C'est l'heure de passer à table.",
    "Celle-ci me plaît davantage.",
    "Quelle heure est-il ?",
]

# Mots trop courants pour servir de variante : ils déclencheraient Seldon sans cesse.
COMMON = {"c'est", "ces", "ses", "ce", "se", "sel", "selon", "celle", "celles", "seul", "seule", "son", "sont",
          "le", "la", "les", "de", "des", "du", "et", "est", "je", "tu", "il", "elle", "on", "un", "une", "oui", "non"}


def stt_options(config: str) -> dict:
    """Options de l'étage ``stt`` de la config, si c'est Whisper."""
    if not Path(config).exists():
        return {}
    stage = Config.load(config).stt
    return dict(stage.options) if stage.engine == "whisper" else {}


def candidate(heard: list[str], rest: list[str], size: int) -> list[str] | None:
    """Ce que Whisper a écrit à la place du mot d'éveil (``size`` mots attendus), ou None si illisible."""
    if not rest:
        found = heard
    else:   # le mot d'éveil s'arrête là où commence la suite connue de la phrase
        start = next((i for i in range(1, len(heard)) if heard[i:i + 2] == rest[:2]), None)
        found = heard[:start] if start is not None else heard[:size]
    return found if 0 < len(found) <= size + 2 else None


def useful(variant: str) -> bool:
    w = words(variant)
    return len(variant.replace(" ", "")) > 2 and not all(t in COMMON for t in w)


async def take(stt: WhisperSTT, prompt: str, args: argparse.Namespace) -> str:
    input(f"{prompt}\n  (Entrée puis parlez, {args.seconds:g} s)")
    print("  Enregistrement...")
    audio = await asyncio.to_thread(record, args.seconds, args.input_device)
    if float(np.sqrt(np.mean(audio**2))) < 1e-3:
        print("  Attention : niveau très faible, le micro est-il bien branché ?")
    text, _ = await asyncio.to_thread(stt.transcribe, audio)
    print(f"  Entendu : « {text} »")
    return text


async def run(args: argparse.Namespace) -> None:
    options = stt_options(args.config)
    word = args.word or next(iter(options.get("wake_words") or []), None)
    if not word:
        sys.exit("Aucun mot d'éveil : passez-le en argument ou renseignez stt.options.wake_words dans la config.")
    path = Path(args.file or options.get("wake_variants") or "data/wake_words.json")
    learned = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if args.reset:
        learned.pop(word, None)
    if args.list or args.reset:
        print(f"Variantes de « {word} » : {', '.join(learned.get(word, [])) or 'aucune'}")
        if args.reset:
            write_private(path, json.dumps(learned, ensure_ascii=False, indent=2).encode())
        return

    known = set(inspect.signature(WhisperSTT).parameters)
    stt = WhisperSTT(**{k: v for k, v in options.items() if k in known})
    if args.input_device is None:
        args.input_device = stt.input_device
    print(f"Chargement de Whisper « {stt.model_name} »...")
    stt.model = (await asyncio.to_thread(stt._load))[0]

    target = words(word)
    counts: Counter[str] = Counter()
    recognized = 0
    print(f"\n1. Prononcez « {word} » comme vous le feriez pour appeler Seldon.")
    for i in range(args.samples):
        rest = CALLS[i % len(CALLS)]
        text = await take(stt, f"\n[{i + 1}/{args.samples}] Dites : « {word}{', ' + rest if rest else ''} »", args)
        found = candidate(words(text), words(rest), len(target))
        if found == target:
            recognized += 1
        elif found:
            counts[" ".join(found)] += 1
        else:
            print("  Transcription inattendue, prise ignorée.")

    variants = {v for v in counts if useful(v)} | set(learned.get(word, []))
    print("\n2. Lisez maintenant ces phrases, sans le mot d'éveil.")
    for i, phrase in enumerate(OTHERS[: args.others]):
        heard = words(await take(stt, f"\n[{i + 1}/{min(args.others, len(OTHERS))}] Dites : « {phrase} »", args))
        for v in [v for v in variants if heard[: len(words(v))] == words(v)]:
            print(f"  « {v} » apparaît aussi ici : variante écartée (elle réveillerait Seldon à tort).")
            variants.discard(v)

    print(f"\n« {word} » bien transcrit : {recognized}/{args.samples}.")
    if recognized < args.samples / 2:
        print(f"Astuce : initial_prompt: \"{word}\" (section stt) aide Whisper à l'écrire correctement.")
    old = set(learned.get(word, []))
    if variants == old:
        print("Aucune nouvelle variante à retenir.")
        return
    if new := sorted(variants - old):
        print("Variantes apprises : " + ", ".join(f"« {v} » ({counts[v]}×)" for v in new))
    if dropped := sorted(old - variants):
        print("Variantes retirées : " + ", ".join(f"« {v} »" for v in dropped))
    if input("Les enregistrer ? [O/n] ").strip().lower() not in ("", "o", "oui", "y", "yes"):
        print("Rien n'a été enregistré.")
        return
    learned[word] = sorted(variants)
    write_private(path, json.dumps(learned, ensure_ascii=False, indent=2).encode())
    print(f"Enregistré dans {path} ; redémarrez Seldon pour en tenir compte.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seldon.cli.wakeword",
                                     description="Apprend comment Whisper transcrit le mot d'éveil pour votre voix")
    parser.add_argument("word", nargs="?", help="mot d'éveil (défaut : le premier de stt.options.wake_words)")
    parser.add_argument("--samples", type=int, default=8, help="nombre de prises du mot d'éveil")
    parser.add_argument("--others", type=int, default=len(OTHERS), help="phrases de contrôle sans mot d'éveil")
    parser.add_argument("--seconds", type=float, default=3.0, help="durée de chaque prise")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--file", help="fichier des variantes (défaut : stt.options.wake_variants)")
    parser.add_argument("--input-device", help="micro sounddevice (index ou nom)")
    parser.add_argument("--list", action="store_true", help="affiche les variantes apprises")
    parser.add_argument("--reset", action="store_true", help="oublie les variantes apprises de ce mot")
    args = parser.parse_args()
    if args.samples < 1 or args.others < 0 or not 0 < args.seconds <= 30:
        parser.error("--samples doit valoir au moins 1, --others être positif et --seconds compris entre 0 et 30")
    if args.input_device and args.input_device.isdigit():
        args.input_device = int(args.input_device)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nAnnulé.")
    except Exception as e:
        log_error(logging.getLogger("seldon.wakeword"), "Entraînement du mot d'éveil impossible", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
