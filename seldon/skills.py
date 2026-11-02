"""Skills : des consignes en français, sans code, que le cerveau lit quand il en a besoin.

Un skill est un dossier ``skills/<nom>/`` contenant un ``SKILL.md`` :

    ---
    name: bilan-du-matin
    description: Fait le point du matin quand on le demande.
    ---
    1. Lis l'agenda du jour avec l'outil nextcloud…

Seuls le nom et la description sont envoyés au cerveau (dans la description de
l'outil ``skill``) ; il appelle ``skill`` pour lire les consignes complètes
quand une demande y correspond. Le fichier est relu à chaque appel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from seldon.base import Context, Result, Tool
from seldon.errors import SeldonError

SKILLS_DIR = Path("skills")


def parse(text: str) -> tuple[dict[str, Any], str]:
    """En-tête YAML (entre deux lignes ``---``) et corps d'un ``SKILL.md``."""
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        meta = yaml.safe_load(head) or {}
        if isinstance(meta, dict):
            return meta, body.strip()
    return {}, text.strip()


class Skills(Tool):
    name = "skill"

    def __init__(self, names: list[str], folder: Path = SKILLS_DIR) -> None:
        self.files: dict[str, Path] = {}
        catalog = []
        for name in names:
            file = folder / name / "SKILL.md"
            try:
                meta, _ = parse(file.read_text(encoding="utf-8"))
            except (OSError, ValueError, yaml.YAMLError) as e:
                raise SeldonError(f"Skill « {name} » illisible : {file} ({e})",
                                  f"créez {file} ou retirez « {name} » de la section skills") from None
            self.files[name] = file
            catalog.append(f"- {name} : {' '.join(str(meta.get('description', '')).split())}")
        self.description = (
            "Lit les consignes d'un skill : la marche à suivre pour un type de demande. Quand une demande "
            "correspond à un skill ci-dessous, appelle d'abord cet outil, puis suis ses consignes.\n"
            "Skills disponibles :\n" + "\n".join(catalog))
        self.parameters = {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": list(self.files)}},
            "required": ["name"],
        }

    async def run(self, args: dict[str, Any], ctx: Context) -> Result:
        file = self.files.get(args["name"])
        if file is None:
            return Result.fail(f"Skill inconnu. Disponibles : {', '.join(self.files)}.")
        _, body = parse(file.read_text(encoding="utf-8"))
        return Result(body)
