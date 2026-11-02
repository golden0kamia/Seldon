"""Outil « camera » : prend une photo avec une webcam (OpenCV) et la montre au cerveau.

Exemple ``config.yaml`` :

    tools:
      - engine: camera
        options:
          device: 0                 # index ou chemin (/dev/video2)
          save_dir: data/captures
          devices:                  # facultatif : plusieurs caméras nommées
            bureau: 0
            porte: {device: /dev/video2, width: 1920, height: 1080}

Les photos enregistrées (``save_dir``) sont privées (0o600) et ne s'écrasent jamais.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any

from seldon.base import Context, Image, Result, Tool
from seldon.storage import write_private

log = logging.getLogger(__name__)

WARMUP_FRAMES = 5     # premières images souvent sombres (exposition auto) ou périmées
JPEG_QUALITY = 85


def _spec(defaults: dict[str, Any], spec: Any) -> dict[str, Any]:
    """Réglages complets d'une caméra ({device, width, height}) ; ValueError si invalides."""
    out = {**defaults, **(spec if isinstance(spec, dict) else {"device": spec})}
    if isinstance(out["device"], str) and out["device"].strip().isdecimal():
        out["device"] = int(out["device"])   # « 0 » : index, pas un nom de fichier
    for key in ("width", "height"):
        if out.get(key) is not None:
            out[key] = int(out[key])
            if out[key] <= 0:
                raise ValueError(f"Caméra : {key} doit être positif")
    return out


class Camera(Tool):
    name = "capture_image"
    description = (
        "Prend une photo avec une caméra branchée à l'ordinateur et te renvoie l'image. "
        "Utilise-la quand l'utilisateur te demande de regarder, de voir ou de décrire "
        "quelque chose devant la caméra, ou de lire un document qu'il te montre."
    )

    def __init__(
        self,
        device: int | str = 0,              # index ou chemin (/dev/video2)
        width: int | None = None,           # résolution demandée (null = celle de la caméra)
        height: int | None = None,
        save_dir: str | None = None,        # dossier où garder les photos (null = pas gardées)
        devices: dict[str, Any] | None = None,  # caméras nommées : {bureau: 0, porte: {device: …}}
        max_dimension: int = 1568,          # plus grand côté de l'image envoyée au cerveau (px)
        timeout: float = 20.0,              # capture plus longue (caméra bloquée) : abandonnée
    ) -> None:
        defaults = {"device": device, "width": width, "height": height}
        self.devices = {str(n): _spec(defaults, s) for n, s in (devices or {"default": device}).items()}
        self.default = next(iter(self.devices))
        self.save_dir = Path(save_dir).expanduser() if save_dir else None
        self.max_dimension = int(max_dimension)
        self.timeout = timeout
        self._lock = threading.Lock()   # une capture à la fois (et un nom de photo à la fois)
        self.parameters = {"type": "object", "properties": {}, "additionalProperties": False}
        if len(self.devices) > 1:
            self.parameters["properties"]["camera"] = {
                "type": "string", "enum": list(self.devices),
                "description": f"Caméra à utiliser (défaut : {self.default}).",
            }

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        name = args.get("camera") or self.default
        if not isinstance(name, str) or name not in self.devices:
            return Result.fail(f"Caméra inconnue : {str(name)[:40]}. Disponibles : {', '.join(self.devices)}")
        try:
            jpeg, (w, h) = await asyncio.wait_for(asyncio.to_thread(self._capture, name), self.timeout)
        except RuntimeError as e:
            return Result.fail(str(e))
        except TimeoutError:
            return Result.fail(f"La caméra « {name} » ne répond pas (plus de {self.timeout:g} s).")

        now = time.localtime()
        text = f"Image capturée ({name}) le {time.strftime('%Y-%m-%d %H:%M:%S', now)}, {w}x{h} px."
        if self.save_dir:
            stem = re.sub(r"[^\w-]+", "_", name) + time.strftime("_%Y%m%d_%H%M%S", now)
            try:
                path = await asyncio.to_thread(self._save, stem, jpeg)
                text += f" Enregistrée dans {path}."
            except OSError as e:   # l'image reste utilisable même si l'enregistrement échoue
                log.warning("Photo non enregistrée dans %s : %s", self.save_dir, e)
                text += " (Enregistrement impossible.)"
        return Result(text, images=[Image(jpeg)])

    # --- partie bloquante (exécutée dans un thread) ---

    def _capture(self, name: str) -> tuple[bytes, tuple[int, int]]:
        import cv2

        spec = self.devices[name]
        if not self._lock.acquire(timeout=self.timeout):
            raise RuntimeError("La caméra est occupée par une autre capture.")
        cap = None
        try:
            cap = cv2.VideoCapture(spec["device"])
            if not cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la caméra « {name} » ({spec['device']}) : "
                                   "aucune caméra détectée ou périphérique occupé.")
            if spec.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec["width"])
            if spec.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec["height"])
            for _ in range(WARMUP_FRAMES):
                cap.grab()
            ok, frame = cap.read()
        finally:
            if cap is not None:
                cap.release()
            self._lock.release()
        if not ok or frame is None:
            raise RuntimeError(f"La caméra « {name} » n'a renvoyé aucune image.")

        h, w = frame.shape[:2]
        scale = self.max_dimension / max(h, w)
        if scale < 1:
            w, h = round(w * scale), round(h * scale)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("Échec de l'encodage JPEG.")
        return buf.tobytes(), (w, h)

    def _save(self, stem: str, data: bytes) -> Path:
        """Enregistre la photo (privée) sous un nom libre : ``stem.jpg``, ``stem_2.jpg``…"""
        with self._lock:
            n = 1
            while (path := self.save_dir / (f"{stem}.jpg" if n == 1 else f"{stem}_{n}.jpg")).exists():
                n += 1
            write_private(path, data)
        return path
