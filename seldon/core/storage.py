"""Écriture des données personnelles (conversations, empreintes vocales, photos).

Les fichiers sont privés (0o600, dossiers créés en 0o700 ; un dossier existant
garde ses droits) et écrits de façon atomique : fichier temporaire au nom unique
dans le même dossier, ``fsync`` puis remplacement. Un lecteur (l'interface, un
autre processus) ne voit jamais un fichier à moitié écrit, et deux écritures
simultanées ne se marchent pas dessus.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

DIR_MODE = 0o700


def private_dir(path: Path) -> Path:
    """Crée ``path`` (et ses parents) s'il n'existe pas, accessible au seul utilisateur."""
    path.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
    return path


def write_private(path: Path, data: bytes) -> None:
    """Remplace atomiquement le contenu de ``path`` par ``data`` (fichier en 0o600)."""
    private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")   # créé en 0o600
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
