"""Point d'entrée : ``python -m seldon [-c config.yaml] [--text]``."""

from __future__ import annotations

import argparse
import asyncio
import logging

from seldon.config import Config, StageConfig
from seldon.core.pipeline import Assistant


def main() -> None:
    parser = argparse.ArgumentParser(prog="seldon", description="Assistant vocal polyvalent")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--text", action="store_true", help="Clavier + console au lieu du micro et du haut-parleur")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.text:
        cfg.stt = StageConfig("text")
        cfg.speakers = StageConfig("none")
        cfg.tts = StageConfig("console")

    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(Assistant.from_config(cfg).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
