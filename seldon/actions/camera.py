"""Action « camera » : capture une image depuis une webcam (OpenCV) pour le modèle.

Exemple ``config.yaml`` :

    actions:
      - engine: camera
        options:
          device: 0                 # index ou chemin (/dev/video2)
          width: 1280
          height: 720
          save_dir: ~/Images/seldon
          devices:                  # facultatif : plusieurs caméras nommées
            bureau: 0
            porte: {device: /dev/video2, width: 1920, height: 1080}

Les photos enregistrées (``save_dir``) sont privées (0o600) et ne s'écrasent jamais.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.storage import private_dir
from seldon.core.types import ActionContext, ActionResult, Image

log = logging.getLogger(__name__)


def _spec(defaults: dict[str, Any], spec: Any) -> dict[str, Any]:
    """Réglages complets d'une caméra ; ValueError si la configuration est invalide."""
    out = {**defaults, **(spec if isinstance(spec, dict) else {"device": spec})}
    if isinstance(out["device"], str) and out["device"].strip().isdecimal():
        out["device"] = int(out["device"])   # « 0 » : index, pas un nom de fichier
    for key in ("width", "height"):
        if out.get(key) is not None:
            out[key] = int(out[key])
            if out[key] <= 0:
                raise ValueError(f"Caméra : {key} doit être positif")
    return out


@register("actions", "camera")
class Camera(Action):
    name = "capture_image"
    acknowledgement = "Je regarde."
    description = (
        "Prend une photo avec une caméra branchée à l'ordinateur et te renvoie l'image. "
        "Utilise-la quand l'utilisateur te demande de regarder, de voir ou de décrire "
        "quelque chose devant la caméra, ou de lire un document qu'il te montre."
    )

    def __init__(
        self,
        device: int | str = 0,
        width: int | None = None,
        height: int | None = None,
        jpeg_quality: int = 85,
        max_dimension: int = 1568,
        warmup_frames: int = 5,
        save_dir: str | None = None,
        devices: dict[str, Any] | None = None,
        keep_open: bool = False,
        timeout: float = 20.0,   # capture plus longue (caméra bloquée) : abandonnée
    ) -> None:
        defaults = {"device": device, "width": width, "height": height}
        named = devices or {"default": device}
        # nom -> réglages complets ({device, width, height})
        self.devices: dict[str, dict[str, Any]] = {str(name): _spec(defaults, spec) for name, spec in named.items()}
        self.default = next(iter(self.devices))
        self.jpeg_quality = min(100, max(1, int(jpeg_quality)))
        self.max_dimension = int(max_dimension)
        self.warmup_frames = max(0, int(warmup_frames))
        self.save_dir = Path(save_dir).expanduser() if save_dir else None
        self.keep_open = keep_open
        self.timeout = timeout
        self._captures: dict[str, Any] = {}   # caméras ouvertes si keep_open
        self._lock = threading.Lock()
        self._closed = False   # après stop() : plus aucune caméra gardée ouverte

    @property
    def input_schema(self) -> dict[str, Any]:  # type: ignore[override]
        props: dict[str, Any] = {}
        if len(self.devices) > 1:
            props["camera"] = {
                "type": "string",
                "enum": list(self.devices),
                "description": f"Caméra à utiliser (défaut : {self.default}).",
            }
        return {"type": "object", "properties": props, "additionalProperties": False}

    async def start(self) -> None:
        self._closed = False

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        name = params.get("camera") or self.default
        if not isinstance(name, str) or name not in self.devices:
            return ActionResult.error(f"Caméra inconnue : {str(name)[:40]}. Disponibles : {', '.join(self.devices)}")
        try:
            jpeg, (w, h) = await asyncio.wait_for(asyncio.to_thread(self._capture, name), self.timeout)
        except RuntimeError as e:
            return ActionResult.error(str(e))
        except asyncio.TimeoutError:
            return ActionResult.error(f"La caméra « {name} » ne répond pas (plus de {self.timeout:g} s).")

        now = time.localtime()
        text = f"Image capturée ({name}) le {time.strftime('%Y-%m-%d %H:%M:%S', now)}, {w}x{h} px."
        if self.save_dir:
            stem = re.sub(r"[^\w-]+", "_", name) + time.strftime("_%Y%m%d_%H%M%S", now)
            try:
                path = await asyncio.to_thread(self._save, self.save_dir, stem, jpeg)
                text += f" Enregistrée dans {path}."
            except OSError as e:   # l'image reste utilisable même si l'enregistrement échoue
                log.warning("Photo non enregistrée dans %s : %s", self.save_dir, e)
                text += " (Enregistrement impossible.)"
        return ActionResult(text=text, images=[Image(data=jpeg, media_type="image/jpeg")])

    # --- partie bloquante (exécutée dans un thread) ---

    def _open(self, name: str) -> Any:
        import cv2

        spec = self.devices[name]
        cap = cv2.VideoCapture(spec["device"])
        try:
            if not cap.isOpened():
                raise RuntimeError(f"Impossible d'ouvrir la caméra « {name} » ({spec['device']}) : "
                                   "aucune caméra détectée ou périphérique occupé.")
            if spec.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec["width"])
            if spec.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec["height"])
        except BaseException:
            cap.release()
            raise
        return cap

    def _capture(self, name: str) -> tuple[bytes, tuple[int, int]]:
        import cv2

        if not self._lock.acquire(timeout=self.timeout):
            raise RuntimeError("La caméra est occupée par une autre capture.")
        try:
            cap = self._captures.pop(name, None) or self._open(name)
            ok, frame = False, None
            try:
                # les premières images sont souvent sombres (exposition auto) ou périmées (tampon)
                for _ in range(self.warmup_frames):
                    cap.grab()
                ok, frame = cap.read()
            finally:
                if self.keep_open and ok and not self._closed:
                    self._captures[name] = cap
                else:
                    cap.release()
        finally:
            self._lock.release()
        if not ok or frame is None:
            raise RuntimeError(f"La caméra « {name} » n'a renvoyé aucune image.")

        h, w = frame.shape[:2]
        scale = self.max_dimension / max(h, w)
        if scale < 1:
            w, h = round(w * scale), round(h * scale)
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("Échec de l'encodage JPEG.")
        return buf.tobytes(), (w, h)

    @staticmethod
    def _save(folder: Path, stem: str, data: bytes) -> Path:
        """Enregistre la photo (0o600) sous un nom libre : ``stem.jpg``, ``stem_2.jpg``…"""
        private_dir(folder)
        for n in itertools.count(1):
            path = folder / (f"{stem}.jpg" if n == 1 else f"{stem}_{n}.jpg")
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                continue
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            return path
        raise AssertionError("inaccessible")

    def _release_all(self) -> None:
        locked = self._lock.acquire(timeout=self.timeout)   # attend la fin d'une capture en cours
        try:
            self._closed = True
            while self._captures:
                self._captures.popitem()[1].release()
        finally:
            if locked:
                self._lock.release()

    async def stop(self) -> None:
        await asyncio.to_thread(self._release_all)   # sans bloquer la boucle pendant une capture
