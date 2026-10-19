"""Interface graphique : ``python -m seldon.gui [-c config.yaml] [dossier]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="seldon-gui", description="Paramètres, fichiers et éditeur de Seldon")
    parser.add_argument("-c", "--config", default="config.yaml", help="fichier de configuration à éditer")
    parser.add_argument("root", nargs="?", help="dossier à afficher dans l'arborescence")
    args = parser.parse_args()

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from seldon.gui import theme
    from seldon.gui.window import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Seldon")
    app.setOrganizationName("Seldon")
    theme.apply(app, str(QSettings("Seldon", "seldon-gui").value("theme", "auto")))
    window = MainWindow(Path(args.config), Path(args.root) if args.root else None)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
