"""Point d'entrée : ``python -m seldon [-c config.yaml] [--text]``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from seldon.config import Config, StageConfig
from seldon.core import tags
from seldon.core.errors import log_error
from seldon.core.pipeline import MUTE_SIGNALS, Assistant

log = logging.getLogger("seldon")


def main() -> None:
    parser = argparse.ArgumentParser(prog="seldon", description="Assistant vocal polyvalent")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--text", action="store_true", help="Clavier + console au lieu du micro et du haut-parleur")
    args = parser.parse_args()
    tags.take_token()   # jeton de l'interface, retiré de l'environnement : aucun sous-processus ne le voit
    for sig in MUTE_SIGNALS:   # un clic sur le bouton micro avant que Seldon soit prêt ne le tue pas
        signal.signal(sig, signal.SIG_IGN)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = Config.load(args.config)
        if args.text:
            cfg.stt = StageConfig("text")
            cfg.speakers = StageConfig("none")
            cfg.tts = StageConfig("console")
        logging.getLogger().setLevel(cfg.log_level)
        if cfg.log_level != "DEBUG":   # httpx journalise chaque requête (URL comprise) au niveau INFO
            for name in ("httpx", "httpcore"):
                logging.getLogger(name).setLevel(logging.WARNING)
        asyncio.run(Assistant.from_config(cfg).run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_error(log, "Seldon n'a pas pu continuer", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
