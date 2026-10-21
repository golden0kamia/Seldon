"""Écriture atomique : un fichier enregistré est complet ou inchangé, jamais tronqué."""

from __future__ import annotations

import contextlib
import os
import stat
import tempfile
from pathlib import Path


def write_atomic(path: str | Path, data: bytes) -> None:
    """Écrit ``data`` dans un fichier temporaire voisin, puis le substitue à ``path``.

    Une coupure ou un disque plein en cours d'écriture laisse l'ancien fichier intact.
    Les droits du fichier existant sont conservés ; un lien symbolique est suivi (sa
    cible est remplacée, le lien reste). Dossier non modifiable : écriture directe.
    """
    path = Path(os.path.realpath(path))
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    except OSError:
        path.write_bytes(data)
        return
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
