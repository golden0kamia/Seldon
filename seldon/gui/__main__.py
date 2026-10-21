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
    app.aboutToQuit.connect(lambda: window.runner.stop(wait=True))   # jamais de Seldon orphelin (micro, GPU)
    _handle_signals(app, window)
    sys.exit(app.exec())


def _handle_signals(app, window) -> None:
    """Ctrl+C au terminal ferme la fenêtre (en proposant d'enregistrer) ; SIGTERM / SIGHUP arrêtent Seldon et quittent."""
    import signal

    from PySide6.QtCore import QTimer

    signal.signal(signal.SIGINT, lambda *_: window.close())
    for name in ("SIGTERM", "SIGHUP"):
        if (sig := getattr(signal, name, None)) is not None:
            signal.signal(sig, lambda *_: (window.runner.stop(wait=True), app.exit(0)))
    tick = QTimer(app)   # Python ne traite ses signaux qu'en exécutant du code : on lui en donne l'occasion
    tick.timeout.connect(lambda: None)
    tick.start(250)


if __name__ == "__main__":
    main()
