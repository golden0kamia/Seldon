"""Options des plugins, lues dans leur code source (AST) sans les importer.

Importer un plugin peut charger torch, whisper ou opencv : l'interface prend la
liste des plugins dans ``seldon.registry.PLUGINS`` (« module:Classe »), ouvre le
fichier du module et lit les paramètres de l'``__init__`` de la classe (valeur
par défaut, annotation, commentaire de fin de ligne). Une classe sans
``__init__`` hérite de celui d'une classe parente du même module.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seldon.registry import PLUGINS

PACKAGE = Path(__file__).resolve().parent.parent   # le paquet ``seldon``
MISSING = object()


@dataclass
class Param:
    name: str
    default: Any = MISSING      # valeur Python (littéral) ou MISSING si absente / non littérale
    default_src: str = ""       # texte source de la valeur par défaut
    annotation: str = ""
    comment: str = ""

    def hint(self) -> str:
        parts = [self.annotation, f"défaut : {self.default_src}" if self.default_src else "", self.comment]
        return " — ".join(p for p in parts if p)


@dataclass
class PluginInfo:
    category: str
    name: str
    path: Path | None           # fichier du module ; None : plugin hors du paquet seldon
    doc: str = ""
    params: dict[str, Param] = field(default_factory=dict)


def module_file(module: str, root: Path = PACKAGE) -> Path | None:
    """Fichier d'un module ``seldon.x.y`` (``x/y.py`` ou ``x/y/__init__.py`` sous ``root``)."""
    head, _, rest = module.partition(".")
    if head != "seldon" or not rest:
        return None
    base = root.joinpath(*rest.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return base.with_suffix(".py")   # pas encore écrit : le bouton ✎ sera grisé


def _line_comment(lines: list[str], lineno: int) -> str:
    line = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
    # pas de gestion fine des « # » dans les chaînes : suffisant pour des signatures
    return line.split("#", 1)[1].strip() if "#" in line else ""


def _params(init: ast.FunctionDef, lines: list[str]) -> dict[str, Param]:
    args = init.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    pairs = list(zip(positional, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
    out: dict[str, Param] = {}
    for arg, default in pairs:
        if arg.arg in ("self", "cls"):
            continue
        p = Param(arg.arg, annotation=ast.unparse(arg.annotation) if arg.annotation else "",
                  comment=_line_comment(lines, arg.lineno))
        if default is not None:
            p.default_src = ast.unparse(default)
            try:
                p.default = ast.literal_eval(default)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):   # non littéral, {[]}…
                pass
        out[arg.arg] = p
    return out


def _parse(path: Path) -> tuple[list[str], ast.Module] | None:
    """Lignes et arbre d'un fichier ; ``None`` s'il est absent, illisible ou invalide."""
    try:
        source = path.read_text(encoding="utf-8")
        return source.splitlines(), ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError, MemoryError, RecursionError):
        return None   # ValueError : octet nul dans la source


def class_info(path: Path, class_name: str) -> tuple[str, dict[str, Param]]:
    """Description (1er paragraphe de la docstring) et paramètres de ``__init__`` d'une classe.

    Sans ``__init__`` ni docstring propres, ceux d'une classe parente définie dans le même fichier.
    """
    if (parsed := _parse(path)) is None:
        return "", {}
    lines, tree = parsed
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    doc, params, seen = "", None, set()
    node = classes.get(class_name)
    while node is not None and node.name not in seen:
        seen.add(node.name)
        doc = doc or (ast.get_docstring(node) or "").strip()
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if params is None and init is not None:
            params = _params(init, lines)
        if doc and params is not None:
            break
        node = next((classes[b.id] for b in node.bases if isinstance(b, ast.Name) and b.id in classes), None)
    doc = doc or (ast.get_docstring(tree) or "").strip()
    return doc.split("\n\n")[0].replace("\n", " ").replace("``", ""), params or {}


def discover(root: Path = PACKAGE, table: dict[str, dict[str, str]] | None = None) -> dict[str, dict[str, PluginInfo]]:
    """Plugins du registre : ``{catégorie: {nom: PluginInfo}}`` (rien n'est importé)."""
    out: dict[str, dict[str, PluginInfo]] = {}
    for category, names in (PLUGINS if table is None else table).items():
        for name, target in names.items():
            module, _, cls = target.partition(":")
            path = module_file(module, root)
            doc, params = class_info(path, cls) if path else ("", {})
            out.setdefault(category, {})[name] = PluginInfo(category, name, path, doc, params)
    return out


_IDENT = r"[A-Za-z_][\w.\-]*"


def choices_from_comment(comment: str) -> list[str]:
    """Valeurs proposées par un commentaire : « a | b | c », « a / b / c » ou « "a" ou "b" »."""
    quoted = re.findall(r'"([^"\s]+)"', comment)
    if len(quoted) >= 2:
        return quoted
    for sep in (r"\|", "/"):
        m = re.search(rf"{_IDENT}(?:\s*{sep}\s*{_IDENT})+", comment)
        if m:
            return [s.strip() for s in re.split(sep, m.group(0))]
    return []
