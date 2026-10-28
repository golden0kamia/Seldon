"""Page Structure : carte des modules de Seldon, de leurs imports et des appels entre fonctions.

Comme ``plugins``, tout vient d'une analyse statique (AST) du code source, sans rien
importer. ``analyze`` relève pour chaque module ses imports internes et ses définitions
(classes, fonctions, méthodes), puis résout les appels qu'il sait suivre : noms locaux,
noms importés (même ré-exportés par un ``__init__``), ``self.méthode()`` et ``super()``
en remontant l'héritage. Les appels sur des objets dont le type n'est connu qu'à
l'exécution (``self.brain.respond()``) ne sont pas suivis.

La carte range les modules par paquet, en colonnes : à gauche ceux qui utilisent les
autres (interface, point d'entrée), à droite les fondations. Un clic sur un module
éclaire ce qu'il importe (vert) et ce qui l'importe (ambre) ; le panneau de droite
détaille ses définitions et leurs appels, un double-clic ouvre le code à la bonne ligne.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame, QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSplitter, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from seldon.gui import theme

PACKAGE_DIR = Path(__file__).resolve().parent.parent   # le paquet ``seldon`` lui-même
KINDS = {"classe": "◆", "fonction": "ƒ", "méthode": "·"}


# --- analyse ----------------------------------------------------------------
@dataclass
class Symbol:
    qualname: str        # ex. seldon.core.pipeline.Pipeline.run
    module: str
    name: str            # ex. Pipeline.run
    kind: str            # classe | fonction | méthode
    line: int
    doc: str = ""
    plugin: str = ""     # « catégorie/nom » pour une classe ``@register``
    bases: list[str] = field(default_factory=list)     # classes internes dont elle hérite
    derived: set[str] = field(default_factory=set)
    calls: set[str] = field(default_factory=set)
    called_by: set[str] = field(default_factory=set)


@dataclass
class Module:
    name: str            # ex. seldon.core.pipeline
    path: Path
    package: str         # colonne de la carte : core, brain… ou seldon pour le niveau supérieur
    doc: str = ""
    lines: int = 0
    imports: set[str] = field(default_factory=set)
    imported_by: set[str] = field(default_factory=set)
    symbols: list[str] = field(default_factory=list)
    error: str = ""      # source illisible : module affiché, mais vide

    @property
    def short(self) -> str:
        return self.name.rsplit(".", 1)[-1] if self.path.name != "__init__.py" else "__init__"


@dataclass
class Structure:
    root: Path
    modules: dict[str, Module] = field(default_factory=dict)
    symbols: dict[str, Symbol] = field(default_factory=dict)

    def packages(self) -> list[str]:
        """Paquets de gauche (utilisateurs) à droite (fondations) : solde des liens sortants et entrants."""
        out: dict[str, set[str]] = {}
        into: dict[str, set[str]] = {}
        for m in self.modules.values():
            out.setdefault(m.package, set())
            into.setdefault(m.package, set())
            for t in m.imports:
                p = self.modules[t].package
                if p != m.package:
                    out[m.package].add(p)
                    into.setdefault(p, set()).add(m.package)
        return sorted(out, key=lambda p: (len(into[p]) - len(out[p]), p))

    def search(self, text: str) -> set[str]:
        """Modules dont le nom, ou celui d'une définition, contient ``text`` (sans casse)."""
        t = text.strip().lower()
        if not t:
            return set()
        return {m.name for m in self.modules.values()
                if t in m.name.lower() or any(t in self.symbols[s].name.lower() for s in m.symbols)}


def _first_line(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node) or ""
    except TypeError:
        return ""
    return doc.strip().split("\n", 1)[0]


def _dotted(node: ast.expr) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _register(cls: ast.ClassDef) -> str:
    for deco in cls.decorator_list:
        if isinstance(deco, ast.Call) and len(deco.args) == 2 and _dotted(deco.func) in ("register", "registry.register"):
            a, b = deco.args
            if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                return f"{a.value}/{b.value}"
    return ""


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def analyze(root: Path = PACKAGE_DIR) -> Structure:
    """Analyse tous les ``.py`` du paquet ``root`` (dossier contenant ``__init__.py``)."""
    root = root.resolve()
    top = root.name
    st = Structure(root)
    trees: dict[str, ast.Module] = {}
    aliases: dict[str, dict[str, str]] = {}   # module -> nom local -> nom qualifié importé

    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or any(p.startswith(".") for p in path.relative_to(root).parts):
            continue
        name = _module_name(root, path)
        parts = name.split(".")
        package = parts[1] if len(parts) > 2 or (len(parts) == 2 and path.name == "__init__.py") else top
        mod = st.modules[name] = Module(name, path, package)
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, str(path))
        except (OSError, SyntaxError, ValueError) as e:
            mod.error = str(e)
            continue
        mod.lines = source.count("\n") + 1
        mod.doc = _first_line(tree)
        trees[name] = tree
        aliases[name] = _aliases(tree, name, path.name == "__init__.py")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _add(st, mod, node.name, "fonction", node)
            elif isinstance(node, ast.ClassDef):
                sym = _add(st, mod, node.name, "classe", node)
                sym.plugin = _register(node)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        _add(st, mod, f"{node.name}.{sub.name}", "méthode", sub)

    def canonical(full: str, depth: int = 0) -> str | None:
        """Nom qualifié d'une définition, en suivant les ré-exportations (``from .base import Brain``)."""
        if full in st.symbols:
            return full
        parts = full.split(".")
        for i in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:i])
            if prefix in st.modules:
                target = aliases.get(prefix, {}).get(parts[i])
                if target and depth < 8:
                    return canonical(".".join([target, *parts[i + 1:]]), depth + 1)
                return None
        return None

    def owner(full: str) -> str | None:
        """Module interne désigné par un nom importé (le module lui-même ou celui qui définit le nom)."""
        if (sym := canonical(full)) is not None:
            return st.symbols[sym].module
        parts = full.split(".")
        for i in range(len(parts), 0, -1):
            if (prefix := ".".join(parts[:i])) in st.modules:
                return prefix
        return None

    def method(cls: str, attr: str, seen: frozenset = frozenset()) -> str | None:
        if f"{cls}.{attr}" in st.symbols:
            return f"{cls}.{attr}"
        for base in st.symbols[cls].bases:
            if base not in seen and (found := method(base, attr, seen | {cls})):
                return found
        return None

    def resolve(func: ast.expr, mod: str, cls: str | None) -> str | None:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call) \
                and isinstance(func.value.func, ast.Name) and func.value.func.id == "super" and cls:
            return next((m for b in st.symbols[cls].bases if (m := method(b, func.attr))), None)
        dotted = _dotted(func)
        if dotted is None:
            return None
        head, *rest = dotted.split(".")
        if head in ("self", "cls") and cls:
            return method(cls, rest[0]) if len(rest) == 1 else None
        if f"{mod}.{head}" in st.symbols:
            base = f"{mod}.{head}"
        elif head in aliases[mod]:
            base = aliases[mod][head]
        else:
            return None
        found = canonical(".".join([base, *rest]))
        if found is None and len(rest) >= 1:   # Classe.méthode héritée
            cls_name = canonical(".".join([base, *rest[:-1]]))
            if cls_name and st.symbols[cls_name].kind == "classe":
                found = method(cls_name, rest[-1])
        return found

    for name, tree in trees.items():
        mod = st.modules[name]
        for target in aliases[name].values():
            if (o := owner(target)) and o != name:
                mod.imports.add(o)
                st.modules[o].imported_by.add(name)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls = f"{name}.{node.name}"
                for b in node.bases:
                    dotted = _dotted(b)
                    if dotted and (found := resolve(b, name, None)) and found != cls \
                            and st.symbols[found].kind == "classe":
                        st.symbols[cls].bases.append(found)
                        st.symbols[found].derived.add(cls)

    for name, tree in trees.items():
        for node in tree.body:
            funcs: list[tuple[str, ast.AST, str | None]] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append((f"{name}.{node.name}", node, None))
            elif isinstance(node, ast.ClassDef):
                cls = f"{name}.{node.name}"
                funcs += [(f"{cls}.{sub.name}", sub, cls) for sub in node.body
                          if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))]
            for qual, fn, cls in funcs:
                for call in ast.walk(fn):
                    if isinstance(call, ast.Call) and (target := resolve(call.func, name, cls)) and target != qual:
                        st.symbols[qual].calls.add(target)
                        st.symbols[target].called_by.add(qual)
    return st


def _aliases(tree: ast.Module, name: str, is_package: bool) -> dict[str, str]:
    """Noms locaux introduits par les imports du module (y compris ceux faits dans une fonction)."""
    package = name if is_package else name.rpartition(".")[0]
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[:len(base) - node.level + 1] if node.level > 1 else base
                source = ".".join(base + ([node.module] if node.module else []))
            else:
                source = node.module or ""
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = f"{source}.{a.name}"
    return out


def _add(st: Structure, mod: Module, name: str, kind: str, node: ast.AST) -> Symbol:
    sym = Symbol(f"{mod.name}.{name}", mod.name, name, kind, node.lineno, _first_line(node))
    st.symbols[sym.qualname] = sym
    mod.symbols.append(sym.qualname)
    return sym


# --- carte ------------------------------------------------------------------
BOX_W, BOX_H, ROW, COL_GAP, HEAD_H = 170, 26, 32, 80, 30
MAX_SCALE = 1.4   # « Ajuster » agrandit la carte jusque-là


class ModuleBox(QGraphicsItem):
    """Un module : nom, nombre de définitions ; l'état (``sel``, ``out``, ``in``, ``dim``) règle sa couleur."""

    def __init__(self, module: Module) -> None:
        super().__init__()
        self.module = module
        self.state = ""
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        tip = [module.name, module.doc, f"{module.lines} lignes · {len(module.symbols)} définitions"]
        self.setToolTip("\n".join(t for t in tip if t) if not module.error else f"{module.name}\n{module.error}")

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(0, 0, BOX_W, BOX_H)

    def paint(self, p: QPainter, _option, _widget=None) -> None:
        colors = {"sel": theme.ACCENT_HI, "out": theme.ACCENT_HI, "in": theme.AMBER, "match": theme.CRT_FG}
        edge = QColor(colors.get(self.state, theme.CRT_DIM))
        text = QColor(theme.CRT_FG)
        fill = QColor(edge)
        fill.setAlpha(70 if self.state == "sel" else 28 if self.state in ("out", "in", "match") else 10)
        if self.state == "dim":
            edge.setAlpha(70)
            text.setAlpha(90)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(edge, 2 if self.state in ("sel", "match") else 1))
        p.setBrush(fill)
        radius = 0 if theme.is_neon() else 6
        p.drawRoundedRect(self.boundingRect().adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)
        p.setFont(theme.mono_font(9.5))
        p.setPen(text)
        p.drawText(QRectF(8, 0, BOX_W - 50, BOX_H), Qt.AlignmentFlag.AlignVCenter, self.module.short)
        dim = QColor(theme.CRT_DIM)
        if self.state == "dim":
            dim.setAlpha(90)
        p.setPen(dim)
        p.setFont(theme.mono_font(8))
        p.drawText(QRectF(0, 0, BOX_W - 8, BOX_H), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   "!" if self.module.error else str(len(self.module.symbols)))

    def side(self, right: bool) -> QPointF:
        return self.pos() + QPointF(BOX_W if right else 0, BOX_H / 2)


class Edge(QGraphicsPathItem):
    def __init__(self, src: ModuleBox, dst: ModuleBox) -> None:
        super().__init__()
        self.src, self.dst = src, dst
        self.setZValue(-1)
        a, b = src.pos(), dst.pos()
        path = QPainterPath()
        if abs(a.x() - b.x()) < 1:   # même colonne : boucle par la droite
            p, q = src.side(True), dst.side(True)
            bend = 26 + min(abs(a.y() - b.y()) / 6, 60)
            path.moveTo(p)
            path.cubicTo(p + QPointF(bend, 0), q + QPointF(bend, 0), q)
        else:
            forward = b.x() > a.x()
            p, q = src.side(forward), dst.side(not forward)
            dx = (q.x() - p.x()) / 2
            path.moveTo(p)
            path.cubicTo(QPointF(p.x() + dx, p.y()), QPointF(q.x() - dx, q.y()), q)
        self.setPath(path)

    def set_state(self, state: str) -> None:
        color = QColor({"out": theme.ACCENT_HI, "in": theme.AMBER}.get(state, theme.CRT_DIM))
        color.setAlpha(230 if state in ("out", "in") else 18 if state == "dim" else 60)
        self.setPen(QPen(color, 1.8 if state in ("out", "in") else 1))
        self.setZValue(1 if state in ("out", "in") else -1)


class MapView(QGraphicsView):
    module_clicked = Signal(str)
    module_activated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet("background: transparent;")
        self.setAccessibleName("Carte des modules")
        self.setToolTip("")
        self.boxes: dict[str, ModuleBox] = {}
        self.edges: list[Edge] = []
        self.heads: list[QGraphicsSimpleTextItem] = []
        self.zoomed = False

    def build(self, st: Structure) -> None:
        scene = self.scene()
        scene.clear()
        self.boxes, self.edges, self.heads = {}, [], []
        groups = [sorted((m for m in st.modules.values() if m.package == package),
                         key=lambda m: (m.path.name != "__init__.py", m.short)) for package in st.packages()]
        col, y = 0, 0.0
        capacity = self._capacity([HEAD_H + len(g) * ROW for g in groups])
        for mods in groups:
            height = HEAD_H + len(mods) * ROW
            if y and y + height > capacity:   # colonne pleine : on passe à la suivante
                col, y = col + 1, 0.0
            x = col * (BOX_W + COL_GAP)
            head = scene.addSimpleText(mods[0].package.upper())
            head.setFont(theme.label_font(9))
            head.setPos(x, y + 4)
            self.heads.append(head)
            for row, m in enumerate(mods):
                box = ModuleBox(m)
                box.setPos(x, y + HEAD_H + row * ROW)
                scene.addItem(box)
                self.boxes[m.name] = box
            y += height + ROW / 2
        for m in st.modules.values():
            for t in sorted(m.imports):
                edge = Edge(self.boxes[m.name], self.boxes[t])
                scene.addItem(edge)
                self.edges.append(edge)
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-24, -16, 90, 24))
        self.theme_changed()

    def _capacity(self, heights: list[float]) -> float:
        """Hauteur de colonne (les petits paquets s'empilent) qui donne la carte la plus lisible dans la vue."""
        view = self.viewport().size()
        vw, vh = (view.width(), view.height()) if view.width() > 200 else (1000, 650)

        def scale(cap: float) -> float:
            cols, y, tallest = 1, 0.0, 0.0
            for h in heights:
                if y and y + h > cap:
                    cols, y = cols + 1, 0.0
                y += h + ROW / 2
                tallest = max(tallest, y)
            return min(vw / (cols * (BOX_W + COL_GAP)), vh / (tallest + HEAD_H))

        steps = range(int(max(heights)), int(sum(heights) + ROW * len(heights)) + ROW, ROW)
        return max(steps, key=scale)

    def highlight(self, selected: str | None, matches: set[str]) -> None:
        sel = self.boxes.get(selected) if selected else None
        mod = sel.module if sel else None
        for name, box in self.boxes.items():
            if mod and name == mod.name:
                box.state = "sel"
            elif mod and name in mod.imports:
                box.state = "out"
            elif mod and name in mod.imported_by:
                box.state = "in"
            elif name in matches:
                box.state = "match"
            else:
                box.state = "dim" if (mod or matches) else ""
            box.update()
        for e in self.edges:
            if mod and e.src.module is mod:
                e.set_state("out")
            elif mod and e.dst.module is mod:
                e.set_state("in")
            else:
                e.set_state("dim" if (mod or matches) else "")
        if sel:
            self.ensureVisible(sel, 40, 40)

    def theme_changed(self) -> None:
        for head in self.heads:
            head.setBrush(QBrush(QColor(theme.ACCENT_HI)))
        for e in self.edges:
            e.set_state(e.src.state if e.src.state in ("out", "in") else "")
        self.scene().update()

    def fit(self) -> None:
        self.zoomed = False
        self.resetTransform()
        r = self.sceneRect()
        if r.width() and self.viewport().width() > 50:
            scale = min(MAX_SCALE, self.viewport().width() / r.width(), self.viewport().height() / r.height())
            self.scale(max(scale, 0.5), max(scale, 0.5))
        self.centerOn(r.center())

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self.zoomed:
            self.fit()

    def _box_at(self, event) -> ModuleBox | None:
        item = self.itemAt(event.position().toPoint())
        return item if isinstance(item, ModuleBox) else None

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and (box := self._box_at(event)):
            self.module_clicked.emit(box.module.name)
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if (box := self._box_at(event)):
            self.module_activated.emit(box.module.name)
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:   # Ctrl+molette : zoom
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            if 0.3 < self.transform().m11() * factor < 3:
                self.zoomed = True   # plus de réajustement automatique jusqu'au prochain « Ajuster »
                self.scale(factor, factor)
            return
        super().wheelEvent(event)


# --- page -------------------------------------------------------------------
PATH_ROLE = Qt.ItemDataRole.UserRole       # (chemin, ligne) à ouvrir
MODULE_ROLE = Qt.ItemDataRole.UserRole + 1  # module à afficher


class StructurePanel(QWidget):
    """Carte des modules à gauche, détail du module choisi à droite."""

    open_requested = Signal(Path, int)

    def __init__(self, root: Path = PACKAGE_DIR, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = root
        self.structure: Structure | None = None
        self.selected: str | None = None

        self.summary = QLabel(objectName="path")
        self.search = QLineEdit(placeholderText="Chercher un module, une classe, une fonction…", clearButtonEnabled=True)
        self.search.setAccessibleName("Recherche dans la structure")
        self.search.textChanged.connect(lambda _t: self._refresh_highlight())
        self.search.returnPressed.connect(self._jump_to_match)
        fit_btn = QPushButton("Ajuster", toolTip="Faire tenir la carte dans la fenêtre (Ctrl+molette : zoom)")
        reload_btn = QPushButton("Actualiser", toolTip="Relire le code source")
        top = QHBoxLayout()
        top.setSpacing(theme.SPACE)
        top.addWidget(theme.caption("structure"))
        top.addWidget(self.summary, 1)
        top.addWidget(self.search, 1)
        top.addWidget(fit_btn)
        top.addWidget(reload_btn)

        self.map = MapView()
        self.map.module_clicked.connect(self.select)
        self.map.module_activated.connect(self._open_module)
        fit_btn.clicked.connect(self.map.fit)
        reload_btn.clicked.connect(self.reload)
        screen = theme.NeuCard("screen", radius=20, padding=16)
        screen.add(self.map, 1)
        self.legend = QLabel(wordWrap=True)
        self.legend.setFont(theme.mono_font(9))
        screen.add(self.legend)

        self.title = QLabel(objectName="path")
        self.title.setFont(theme.mono_font(12))
        self.doc = QLabel(wordWrap=True, textFormat=Qt.TextFormat.PlainText)
        self.tree = QTreeWidget(headerHidden=True)
        self.tree.setAccessibleName("Détail du module")
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemActivated.connect(self._item_activated)
        detail = theme.NeuCard("raised")
        detail.add(self.title)
        detail.add(self.doc)
        detail.add(self.tree, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(screen)
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([1000, 440])
        self.splitter = splitter

        head = QVBoxLayout()
        head.setContentsMargins(theme.GUTTER, theme.SPACE, theme.GUTTER, 0)
        head.addLayout(top)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, theme.SPACE)
        lay.setSpacing(0)
        lay.addLayout(head)
        lay.addWidget(splitter, 1)
        self.theme_changed()
        self._show_detail()

    def ensure_loaded(self) -> None:
        """Analyse paresseuse : au premier affichage de la page."""
        if self.structure is None:
            self.reload()

    def reload(self) -> None:
        self.structure = analyze(self.root)
        st = self.structure
        links = sum(len(s.calls) for s in st.symbols.values())
        imports = sum(len(m.imports) for m in st.modules.values())
        self.summary.setText(f"{len(st.modules)} modules · {len(st.symbols)} définitions · "
                             f"{imports} imports · {links} appels suivis")
        self.map.build(st)
        if self.selected not in st.modules:
            self.selected = None
        self._refresh_highlight()
        self._show_detail()
        self.map.fit()

    def select(self, name: str) -> None:
        if self.structure and name in self.structure.modules:
            self.selected = name
            self._refresh_highlight()
            self._show_detail()

    # --- détail -------------------------------------------------------------
    def _show_detail(self) -> None:
        self.tree.clear()
        st = self.structure
        mod = st.modules.get(self.selected) if st and self.selected else None
        if mod is None:
            self.title.setText("Aucun module choisi")
            self.doc.setText("Cliquez sur un module de la carte pour voir ce qu'il importe, ce qui l'utilise, "
                             "ses classes et fonctions, et qui les appelle.")
            return
        self.title.setText(mod.name)
        self.doc.setText(mod.error or mod.doc or "(sans description)")
        query = self.search.text().strip().lower()

        def section(text: str, names: list[str]) -> None:
            item = QTreeWidgetItem([f"{text} ({len(names)})"])
            item.setFont(0, theme.label_font())
            for n in names:
                child = QTreeWidgetItem(item, [n])
                child.setFont(0, theme.mono_font(10))
                child.setData(0, MODULE_ROLE, n)
                child.setToolTip(0, st.modules[n].doc)
            self.tree.addTopLevelItem(item)
            item.setExpanded(bool(names) and len(names) <= 12)

        section("Importe", sorted(mod.imports))
        section("Importé par", sorted(mod.imported_by))
        defs = QTreeWidgetItem([f"Définitions ({len(mod.symbols)})"])
        defs.setFont(0, theme.label_font())
        self.tree.addTopLevelItem(defs)
        first_match = None
        for q in mod.symbols:
            sym = st.symbols[q]
            label = f"{KINDS[sym.kind]} {sym.name}" + ("()" if sym.kind != "classe" else "")
            if sym.plugin:
                label += f"   [plugin {sym.plugin}]"
            item = QTreeWidgetItem(defs, [label])
            item.setFont(0, theme.mono_font(10))
            item.setData(0, PATH_ROLE, (str(mod.path), sym.line))
            item.setToolTip(0, f"{sym.doc}\nligne {sym.line} · double-clic : ouvrir" if sym.doc
                            else f"ligne {sym.line} · double-clic : ouvrir")
            self._links(item, "hérite de", sym.bases, mod.name)
            self._links(item, "hérité par", sorted(sym.derived), mod.name)
            self._links(item, "→ appelle", sorted(sym.calls), mod.name)
            self._links(item, "← appelé par", sorted(sym.called_by), mod.name)
            if query and first_match is None and query in sym.name.lower():
                first_match = item
        defs.setExpanded(True)
        if first_match is not None:
            self.tree.setCurrentItem(first_match)
            self.tree.scrollToItem(first_match)

    def _links(self, parent: QTreeWidgetItem, verb: str, targets: list[str], here: str) -> None:
        st = self.structure
        for t in targets:
            sym = st.symbols[t]
            where = "" if sym.module == here else f"   ({sym.module.removeprefix(st.root.name + '.')})"
            child = QTreeWidgetItem(parent, [f"{verb} {sym.name}{where}"])
            child.setData(0, PATH_ROLE, (str(st.modules[sym.module].path), sym.line))
            child.setData(0, MODULE_ROLE, sym.module)
            child.setToolTip(0, sym.doc)

    def _item_clicked(self, item: QTreeWidgetItem) -> None:
        if item.data(0, PATH_ROLE) is None and (name := item.data(0, MODULE_ROLE)):
            self.select(name)   # lien vers un module : on s'y rend

    def _item_activated(self, item: QTreeWidgetItem) -> None:
        if (loc := item.data(0, PATH_ROLE)) is not None:
            self.open_requested.emit(Path(loc[0]), int(loc[1]))
        elif (name := item.data(0, MODULE_ROLE)):
            self.select(name)

    def _open_module(self, name: str) -> None:
        self.select(name)
        self.open_requested.emit(self.structure.modules[name].path, 1)

    # --- recherche ----------------------------------------------------------
    def _matches(self) -> set[str]:
        return self.structure.search(self.search.text()) if self.structure else set()

    def _refresh_highlight(self) -> None:
        if self.structure is not None:
            self.map.highlight(self.selected, self._matches())

    def _jump_to_match(self) -> None:
        if matches := sorted(self._matches()):
            self.select(matches[0] if self.selected not in matches else
                        matches[(matches.index(self.selected) + 1) % len(matches)])

    def theme_changed(self) -> None:
        self.legend.setText(f'<span style="color:{theme.ACCENT_HI}">━ importe</span> &nbsp; '
                            f'<span style="color:{theme.AMBER}">━ importé par</span> &nbsp; '
                            f'<span style="color:{theme.CRT_DIM}">chiffre : définitions · glisser : déplacer · '
                            "Ctrl+molette : zoom · double-clic : ouvrir</span>")
        self.map.theme_changed()
