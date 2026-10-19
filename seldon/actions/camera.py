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
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from seldon.actions.base import Action
from seldon.core.registry import register
from seldon.core.types import ActionContext, ActionResult, Image


@register("actions", "camera")
class Camera(Action):
    name = "capture_image"
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
    ) -> None:
        defaults = {"device": device, "width": width, "height": height}
        named = devices or {"default": device}
        # nom -> réglages complets ({device, width, height})
        self.devices: dict[str, dict[str, Any]] = {
            name: {**defaults, **(spec if isinstance(spec, dict) else {"device": spec})}
            for name, spec in named.items()
        }
        self.default = next(iter(self.devices))
        self.jpeg_quality = jpeg_quality
        self.max_dimension = max_dimension
        self.warmup_frames = warmup_frames
        self.save_dir = Path(save_dir).expanduser() if save_dir else None
        self.keep_open = keep_open
        self._captures: dict[str, Any] = {}   # caméras ouvertes si keep_open
        self._lock = threading.Lock()

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

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        name = params.get("camera") or self.default
        if name not in self.devices:
            return ActionResult.error(f"Caméra inconnue : {name}. Disponibles : {', '.join(self.devices)}")
        try:
            jpeg, (w, h) = await asyncio.to_thread(self._capture, name)
        except RuntimeError as e:
            return ActionResult.error(str(e))

        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"Image capturée ({name}) le {stamp}, {w}x{h} px."
        if self.save_dir:
            path = self.save_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            await asyncio.to_thread(self._save, path, jpeg)
            text += f" Enregistrée dans {path}."
        return ActionResult(text=text, images=[Image(data=jpeg, media_type="image/jpeg")])

    # --- partie bloquante (exécutée dans un thread) ---

    def _open(self, name: str) -> Any:
        import cv2

        spec = self.devices[name]
        cap = cv2.VideoCapture(spec["device"])
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Impossible d'ouvrir la caméra « {name} » ({spec['device']}) : "
                               "aucune caméra détectée ou périphérique occupé.")
        if spec.get("width"):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec["width"])
        if spec.get("height"):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec["height"])
        return cap

    def _capture(self, name: str) -> tuple[bytes, tuple[int, int]]:
        import cv2

        with self._lock:
            cap = self._captures.pop(name, None) or self._open(name)
            ok, frame = False, None
            try:
                # les premières images sont souvent sombres (exposition auto) ou périmées (tampon)
                for _ in range(self.warmup_frames):
                    cap.grab()
                ok, frame = cap.read()
            finally:
                if self.keep_open and ok:
                    self._captures[name] = cap
                else:
                    cap.release()
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
    def _save(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    async def stop(self) -> None:
        with self._lock:
            for cap in self._captures.values():
                cap.release()
            self._captures.clear()
