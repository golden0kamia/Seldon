"""Point d'entrée : ``python -m seldon [-c config.yaml] [--text]``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from seldon.config import Config, StageConfig
from seldon.core.errors import log_error
from seldon.core.pipeline import Assistant

log = logging.getLogger("seldon")


def main() -> None:
    parser = argparse.ArgumentParser(prog="seldon", description="Assistant vocal polyvalent")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--text", action="store_true", help="Clavier + console au lieu du micro et du haut-parleur")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = Config.load(args.config)
        if args.text:
            cfg.stt = StageConfig("text")
            cfg.speakers = StageConfig("none")
            cfg.tts = StageConfig("console")
        logging.getLogger().setLevel(cfg.log_level)
        asyncio.run(Assistant.from_config(cfg).run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_error(log, "Seldon n'a pas pu continuer", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
