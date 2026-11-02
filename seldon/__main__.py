"""Point d'entrée : ``python -m seldon [-c config.yaml] [--text]``."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from seldon import tags
from seldon.config import Config, Plugin, load_secrets
from seldon.errors import log_error
from seldon.pipeline import Seldon

log = logging.getLogger("seldon")


def main() -> None:
    parser = argparse.ArgumentParser(prog="seldon", description="Assistant vocal")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("--text", action="store_true", help="clavier et console au lieu du micro et du haut-parleur")
    args = parser.parse_args()
    tags.take_token()   # jeton de l'interface, retiré de l'environnement : aucun sous-processus ne le voit
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        load_secrets(Path(args.config).parent / ".env")   # identifiants, hors de config.yaml
        cfg = Config.load(args.config)
        if args.text:
            cfg.stt, cfg.speakers, cfg.tts = Plugin("keyboard"), None, Plugin("console")
        logging.getLogger().setLevel(cfg.log_level)
        for name in ("httpx", "httpcore"):   # chaque requête (URL comprise) sinon
            logging.getLogger(name).setLevel(logging.WARNING)
        asyncio.run(Seldon(cfg).run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log_error(log, "Seldon n'a pas pu démarrer", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
