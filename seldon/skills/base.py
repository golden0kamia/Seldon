"""Skills : des consignes, sans code, que le cerveau charge quand une tâche s'y prête.

Un skill est un dossier ``skills/<nom>/`` contenant un ``SKILL.md`` :

    ---
    name: bilan-du-matin
    description: Fait le point du matin (agenda, tâches, maison) quand on le demande.
    ---
    1. Lis l'agenda du jour avec l'outil nextcloud…

Seuls le nom et la description sont envoyés au cerveau à chaque échange (dans la
description de l'outil ``skill``) ; il appelle ``skill`` pour lire les consignes
quand une demande y correspond, puis les suit avec les tools et serveurs MCP
activés. Les autres fichiers du dossier (modèles, listes, exemples…) se lisent
de même avec le paramètre ``file``. Les fichiers sont relus à chaque appel : une
consigne modifiée s'applique sans redémarrer (nom et description : au démarrage).

Exemple ``config.yaml`` :

    skills:
      - bilan-du-matin          # dossier skills/bilan-du-matin
      - ~/mes-skills/recettes   # ou chemin d'un dossier
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from seldon.core.action import Action
from seldon.core.errors import SeldonError
from seldon.core.types import ActionContext, ActionResult

log = logging.getLogger(__name__)

SKILLS_DIR = Path("skills")   # dossier des skills désignés par leur seul nom
SKILL_FILE = "SKILL.md"
MAX_FILE_CHARS = 100_000      # au-delà, le fichier lu est tronqué
MAX_LISTED = 50               # fichiers annexes listés au chargement d'un skill


@dataclass
class Skill:
    name: str
    description: str
    path: Path   # dossier du skill (résolu)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """En-tête YAML entre deux lignes ``---`` et corps d'un ``SKILL.md``."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = next((i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"), None)
    if end is None:
        return {}, text
    meta = yaml.safe_load("\n".join(lines[1:end])) or {}
    if not isinstance(meta, dict):
        raise ValueError("l'en-tête doit contenir des clés: valeurs (name, description)")
    return meta, "\n".join(lines[end + 1:]).strip()


def skill_dir(entry: str, base: Path = SKILLS_DIR) -> Path:
    """Dossier d'une entrée de ``skills:`` : un nom (sous ``base``) ou un chemin."""
    path = Path(entry).expanduser()
    return path if "/" in entry or path.is_absolute() else base / entry


def load(entry: str, base: Path = SKILLS_DIR) -> Skill:
    folder = skill_dir(entry, base)
    file = folder / SKILL_FILE
    try:
        meta, _body = split_front_matter(file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SeldonError(f"Skill « {entry} » : fichier {file} introuvable",
                          f"créez {file} ou retirez « {entry} » de la section skills") from None
    except (OSError, UnicodeDecodeError, ValueError, yaml.YAMLError) as e:
        raise SeldonError(f"Skill « {entry} » : {file} illisible ({e})") from None
    description = " ".join(str(meta.get("description") or "").split())
    if not description:
        raise SeldonError(f"Skill « {entry} » : description manquante dans {file}",
                          "l'en-tête doit contenir « description: » (quand utiliser ce skill)")
    name = str(meta.get("name") or folder.name).strip()
    return Skill(name, description, folder.resolve())


def available(base: Path = SKILLS_DIR) -> list[str]:
    """Noms des dossiers de ``base`` qui contiennent un ``SKILL.md``."""
    try:
        return sorted(p.parent.name for p in base.glob(f"*/{SKILL_FILE}") if p.is_file())
    except OSError:
        return []


def _read(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= MAX_FILE_CHARS else text[:MAX_FILE_CHARS] + "\n[… fichier tronqué]"


class Skills(Action):
    """Outil ``skill`` : donne au cerveau les consignes des skills activés."""

    name = "skill"

    def __init__(self, entries: list[str], base: str | Path = SKILLS_DIR) -> None:
        self.skills: dict[str, Skill] = {}
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                raise SeldonError(f"Section « skills » : nom de skill attendu, pas {entry!r}")
            skill = load(entry.strip(), Path(base))
            if skill.name in self.skills:
                raise SeldonError(f"Deux skills s'appellent « {skill.name} »",
                                  "changez « name: » dans l'un des SKILL.md")
            self.skills[skill.name] = skill
        catalog = "\n".join(f"- {s.name} : {s.description}" for s in self.skills.values())
        self.description = (
            "Charge les consignes d'un skill : la marche à suivre pour un type de demande. "
            "Quand une demande correspond à un skill ci-dessous, appelle d'abord cet outil, "
            "puis suis les consignes avec tes autres outils. Le paramètre file lit un autre "
            "fichier du skill quand les consignes y renvoient.\nSkills disponibles :\n" + catalog
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": list(self.skills), "description": "Nom du skill."},
                "file": {"type": "string", "description": "Fichier du skill à lire (chemin relatif), "
                                                          "seulement si les consignes y renvoient."},
            },
            "required": ["name"],
            "additionalProperties": False,
        }

    async def run(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        skill = self.skills.get(str(params.get("name") or ""))
        if skill is None:
            return ActionResult.error(f"Skill inconnu. Skills disponibles : {', '.join(self.skills)}.")
        file = params.get("file")
        try:
            if file:
                target = (skill.path / str(file)).resolve()
                if not target.is_relative_to(skill.path) or not target.is_file():
                    return ActionResult.error(f"Fichier « {file} » absent du skill {skill.name}.")
                return ActionResult(text=_read(target))
            _meta, body = split_front_matter(_read(skill.path / SKILL_FILE))
        except (OSError, ValueError, yaml.YAMLError) as e:
            log.warning("Skill %s illisible : %s", skill.name, e)
            return ActionResult.error(f"Skill {skill.name} illisible.")
        others = sorted(str(p.relative_to(skill.path)) for p in skill.path.rglob("*")
                        if p.is_file() and p.name != SKILL_FILE and not p.name.startswith("."))
        if others:
            body += "\n\nFichiers du skill (paramètre file) : " + ", ".join(others[:MAX_LISTED])
        return ActionResult(text=body)
