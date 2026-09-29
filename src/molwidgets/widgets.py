"""anywidget classes. Molecules are drawn in the browser with RDKit.js; RDKit (Python)
does the chemistry (standardization, fingerprints, bit environments)."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import anywidget
import numpy as np
import traitlets
from rdkit import Chem
from rdkit.Chem import rdDepictor

from .chem import BitCensus, bit_census, env_atoms_bonds, morgan_bits

_STATIC = Path(__file__).parent / "static"
# CoordGen draws rings and macrocycles more cleanly; RDKit.js uses it too (rdkit_loader.js).
rdDepictor.SetPreferCoordGen(True)
# Censuses are expensive (one pass over the reference set); share them between widget
# instances so re-creating an explorer in a reactive cell is cheap.
_CENSUS_CACHE: dict[tuple[int, int, int], BitCensus] = {}


def census_for(reference: list[str], radius: int = 2, n_bits: int = 2048) -> BitCensus:
    """Cached BitCensus shared with every MorganExplorer built on the same reference set."""
    key = (hash(tuple(reference)), radius, n_bits)
    if key not in _CENSUS_CACHE:
        _CENSUS_CACHE[key] = bit_census(list(reference), radius, n_bits)
    return _CENSUS_CACHE[key]


def _bundle(name: str) -> str:
    # anywidget ESM is loaded from a blob URL, so relative imports do not work:
    # inline the shared RDKit.js loader in front of each widget module.
    loader = re.sub(r"^export ", "", (_STATIC / "rdkit_loader.js").read_text(), flags=re.MULTILINE)
    return loader + "\n" + (_STATIC / name).read_text()


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records(data: Any) -> list[dict]:
    """Accept a polars/pandas DataFrame or a list of dicts."""
    if hasattr(data, "to_dicts"):  # polars
        rows = data.to_dicts()
    elif hasattr(data, "to_dict"):  # pandas
        rows = data.to_dict(orient="records")
    else:
        rows = list(data)
    return [{k: _clean(v) for k, v in r.items()} for r in rows]


class MolGrid(anywidget.AnyWidget):
    """Paged molecule grid with text + SMARTS filtering, sorting, colouring and selection.

    `selection` holds the ids of selected molecules and syncs both ways, so a marimo
    cell that reads `grid.selection` (via `mo.ui.anywidget`) re-runs on every click.
    `selection_mode` is "multiple", "single" or "pair" (keeps the last two picks).
    """

    _esm = _bundle("molgrid.js")

    data = traitlets.List(traitlets.Dict()).tag(sync=True)
    id_col = traitlets.Unicode("id").tag(sync=True)
    smiles_col = traitlets.Unicode("smiles").tag(sync=True)
    subset = traitlets.List(traitlets.Unicode()).tag(sync=True)
    color_by = traitlets.Unicode("").tag(sync=True)
    color_range = traitlets.List(allow_none=True, default_value=None).tag(sync=True)
    sort_by = traitlets.Unicode("").tag(sync=True)
    smarts = traitlets.Unicode("").tag(sync=True)
    highlights = traitlets.Dict().tag(sync=True)
    selection = traitlets.List(traitlets.Unicode()).tag(sync=True)
    selection_mode = traitlets.Unicode("multiple").tag(sync=True)
    page_size = traitlets.Int(24).tag(sync=True)
    cell_size = traitlets.Int(170).tag(sync=True)

    def __init__(self, data: Any = (), **kwargs):
        rows = _records(data)
        if rows and "subset" not in kwargs:
            skip = {kwargs.get("id_col", "id"), kwargs.get("smiles_col", "smiles")}
            kwargs["subset"] = [k for k in rows[0] if k not in skip][:3]
        super().__init__(data=rows, **kwargs)

    def set_data(self, data: Any) -> None:
        self.data = _records(data)


class _Computing(anywidget.AnyWidget):
    """A widget whose browser side waits for Python to recompute after some changes.

    `rev` goes up once Python has handled a change to any name passed to `_signal_done`; the
    browser shows a "computing" indicator from its own change until then (see busyIndicator).
    """

    rev = traitlets.Int(0).tag(sync=True)

    def _signal_done(self, names: list[str]) -> None:
        # registered after the widget's own observers, so it runs once they have finished
        self.observe(self._bump_rev, names=names)

    def _bump_rev(self, _change=None) -> None:
        self.rev += 1


class MorganExplorer(_Computing):
    """Explain a Morgan fingerprint bit by bit, for one molecule or a pair.

    Pass `reference` SMILES (and optionally activities `y`) to get dataset context:
    how many molecules set each bit, how many *different* substructures collide in it,
    and the mean activity with the bit on vs off. `selected_bit` syncs back to Python.
    `stereo_labels=True` annotates stereocentres (R/S) and double bonds (E/Z) in the drawings.
    """

    _esm = _bundle("morgan.js")

    molecules = traitlets.List(traitlets.Dict()).tag(sync=False)
    radius = traitlets.Int(2).tag(sync=True)
    n_bits = traitlets.Int(2048).tag(sync=True)
    payload = traitlets.List(traitlets.Dict()).tag(sync=True)
    bit_stats = traitlets.Dict().tag(sync=True)
    selected_bit = traitlets.Int(-1).tag(sync=True)
    bit_examples = traitlets.List(traitlets.Dict()).tag(sync=True)
    y_label = traitlets.Unicode("y").tag(sync=True)
    pair_note = traitlets.Unicode("").tag(sync=True)
    stereo_labels = traitlets.Bool(False).tag(sync=True)  # draw R/S and E/Z labels
    # Optional model attribution per molecule: {bit: contribution}, e.g. LightGBM TreeSHAP.
    # Shown only while the explorer's radius / n_bits match the model's fingerprint.
    contributions = traitlets.List(traitlets.Dict()).tag(sync=True)
    contrib_label = traitlets.Unicode("SHAP").tag(sync=True)
    contrib_radius = traitlets.Int(2).tag(sync=True)
    contrib_n_bits = traitlets.Int(2048).tag(sync=True)

    def __init__(
        self,
        molecules: list[dict] | None = None,
        reference: list[str] | None = None,
        y: Any = None,
        **kwargs,
    ):
        self._reference = list(reference) if reference is not None else None
        self._y = None if y is None else np.asarray(y, dtype=float)
        super().__init__(**kwargs)
        self.observe(self._refresh, names=["molecules", "radius", "n_bits"])
        self.observe(self._refresh_examples, names=["selected_bit"])
        self._signal_done(["radius", "n_bits", "selected_bit"])
        self.molecules = molecules or []
        self._refresh()

    def census(self) -> BitCensus | None:
        if self._reference is None:
            return None
        return census_for(self._reference, self.radius, self.n_bits)

    def _refresh(self, _change=None) -> None:
        self.payload = [
            {
                "id": str(m.get("id", i)),
                "smiles": m["smiles"],
                "label": m.get("label", ""),
                "bits": morgan_bits(m["smiles"], self.radius, self.n_bits),
            }
            for i, m in enumerate(self.molecules[:2])
        ]
        census = self.census()
        stats = census.stats(self._y) if census is not None else {}
        on = {b["bit"] for m in self.payload for b in m["bits"]}
        self.bit_stats = {str(k): v for k, v in stats.items() if k in on}
        self._refresh_examples()

    def _refresh_examples(self, _change=None) -> None:
        census = self.census()
        bit = self.selected_bit
        if census is None or self._reference is None or bit < 0:
            self.bit_examples = []
            return
        rows = []
        for ex in census.examples.get(bit, []):
            parent = self._reference[ex["mol_index"]]
            atoms, bonds = env_atoms_bonds(Chem.MolFromSmiles(parent), ex["center"], ex["radius"])
            rows.append({**ex, "parent_smiles": parent, "atoms": atoms, "bonds": bonds})
        self.bit_examples = rows


# the ECFPMovie palette, so the same letters look the same in the movie and the stepper
_PALETTE = [
    "#8b5cf6",
    "#3b82f6",
    "#f97316",
    "#10b981",
    "#ec4899",
    "#6366f1",
    "#f59e0b",
    "#22c55e",
    "#e11d48",
    "#a855f7",
]
_DROPPED = "#94a3b8"

ECFP_EXAMPLES = [
    ["N-methylacetamide", "CC(=O)NC"],
    ["paracetamol", "CC(=O)Nc1ccc(O)cc1"],
    ["aspirin", "CC(=O)Oc1ccccc1C(=O)O"],
    ["caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"],
    ["ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"],
]


def _label_colours(layers: list[list[dict]]) -> dict[str, str]:
    """Kept identifiers get the movie's colours in order of appearance; dropped ones are grey."""
    import colorsys

    colours: dict[str, str] = {}
    for layer in layers:
        for row in layer:
            if row["status"] == "new" and row["label"] not in colours:
                n = len(colours)
                if n < len(_PALETTE):
                    colours[row["label"]] = _PALETTE[n]
                else:
                    rgb = colorsys.hls_to_rgb((n * 0.381966 + 0.1) % 1.0, 0.55, 0.7)
                    colours[row["label"]] = "#" + "".join(f"{round(255 * c):02x}" for c in rgb)
    for layer in layers:
        for row in layer:
            colours.setdefault(row["label"], _DROPPED)
    return colours


def _pastel(hex_colour: str, amount: float = 0.55) -> tuple[float, float, float]:
    r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _env_text(mol: Chem.Mol, row: dict) -> str:
    atom = mol.GetAtomWithIdx(row["atom"])
    if not row["bonds"]:
        h = atom.GetTotalNumHs()
        text = atom.GetSymbol() + (f"H{h if h > 1 else ''}" if h else "")
        return text + (" (ring)" if atom.IsInRing() else "")
    return Chem.MolFragmentToSmiles(
        mol, atomsToUse=row["atoms"], bondsToUse=row["bonds"], rootedAtAtom=row["atom"]
    )


class ECFPStepper(_Computing):
    """Step through the ECFP/Morgan algorithm for any molecule, following the ECFPMovie story.

    Guided mode walks atom by atom through every radius: the six invariants hashed at radius 0,
    the atom's own identifier plus its sorted (bond, neighbour) pairs at larger radii, dropped
    duplicate environments, the growing set of features, and finally folding into ``n_bits``
    bits and the collisions that folding causes. It can play itself (like the movie) or be
    stepped by hand; explore mode jumps between radii and lets you click atoms. Identifiers and
    letters are RDKit's own values, labelled as in the movie. Type a SMILES into the widget or
    set ``smiles``.
    """

    _esm = _bundle("stepper.js")

    smiles = traitlets.Unicode("CC(=O)Nc1ccc(O)cc1").tag(sync=True)
    max_radius = traitlets.Int(2).tag(sync=True)
    n_bits = traitlets.Int(64).tag(sync=True)
    examples = traitlets.List().tag(sync=True)
    guided = traitlets.Bool(True).tag(sync=True)
    radius = traitlets.Int(0).tag(sync=True)
    atom = traitlets.Int(0).tag(sync=True)
    # environments to light up instead of `atom` (e.g. the two sides of a collision): [[atom, radius]]
    envs = traitlets.List().tag(sync=True)
    steps = traitlets.List().tag(sync=True)
    colours = traitlets.Dict().tag(sync=True)
    svg = traitlets.Unicode("").tag(sync=True)
    error = traitlets.Unicode("").tag(sync=True)
    max_atoms = 60

    def __init__(self, smiles: str | None = None, max_radius: int = 2, **kwargs: Any) -> None:
        if smiles is not None:
            kwargs["smiles"] = smiles
        kwargs.setdefault("examples", ECFP_EXAMPLES)
        super().__init__(max_radius=max_radius, **kwargs)
        self.observe(self._build, names=["smiles", "max_radius"])
        self.observe(self._render, names=["radius", "atom", "envs", "guided"])
        self._signal_done(["smiles", "max_radius", "radius", "atom", "envs", "guided"])
        self._build()

    def _build(self, _change=None) -> None:
        from .ecfp import ecfp_story

        mol = Chem.MolFromSmiles(self.smiles) if self.smiles.strip() else None
        if mol is None:
            self.error = f"Cannot parse SMILES: {self.smiles}"
            return
        if mol.GetNumAtoms() > self.max_atoms:
            self.error = (
                f"{mol.GetNumAtoms()} heavy atoms: too many to step through (max {self.max_atoms})."
            )
            return
        self.error = ""
        story = ecfp_story(self.smiles, self.max_radius)
        self._mol = mol
        for layer in story["layers"]:
            for row in layer:
                row["env"] = _env_text(mol, row)
        with self.hold_sync():
            self.colours = _label_colours(story["layers"])
            self.steps = story["layers"]
            self.radius, self.atom, self.envs = 0, 0, []
        self._render()

    def _render(self, _change=None) -> None:
        from rdkit.Chem.Draw import rdMolDraw2D

        if not self.steps:
            return
        mol = Chem.Mol(self._mol)
        r = min(self.radius, len(self.steps) - 1)
        layer = self.steps[r]
        walking = self.guided and not self.envs and self.atom >= 0
        for s in layer:
            if walking and s["atom"] > self.atom:
                # not reached yet: no letter at radius 0, last round's letter afterwards
                if not r:
                    continue
                s = self.steps[r - 1][s["atom"]]
            mol.GetAtomWithIdx(s["atom"]).SetProp("atomNote", s["label"])
        colours = {s["atom"]: _pastel(self.colours[s["label"]]) for s in layer}
        atoms = list(colours)
        bond_colours: dict[int, tuple[float, float, float]] = {}
        radii = {a: 0.32 for a in atoms}
        fade = (0.93, 0.93, 0.93)
        if self.envs:
            colours = {a: fade for a in atoms}
            for a, rr in self.envs:
                row = self.steps[rr][a]
                c = _pastel(self.colours[row["label"]], 0.35)
                colours.update({x: c for x in row["atoms"]})
                bond_colours.update({b: c for b in row["bonds"]})
                radii[a] = 0.42
        elif self.atom >= 0:
            sel = layer[self.atom]
            c = _pastel(self.colours[sel["label"]], 0.35)
            colours = {a: (c if a in set(sel["atoms"]) else fade) for a in atoms}
            colours[self.atom] = _pastel(self.colours[sel["label"]], 0.05)
            bond_colours = {b: c for b in sel["bonds"]}
            radii = {a: (0.42 if a == self.atom else 0.32) for a in atoms}
        drawer = rdMolDraw2D.MolDraw2DSVG(460, 340)
        opts = drawer.drawOptions()
        opts.annotationFontScale = 0.85
        # Small teaching molecules should fill the panel; large ones keep a sane bond length.
        opts.fixedBondLength = (
            70 if mol.GetNumAtoms() <= 8 else 45 if mol.GetNumAtoms() <= 16 else 34
        )
        opts.padding = 0.08
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=atoms,
            highlightAtomColors=colours,
            highlightBonds=list(bond_colours),
            highlightBondColors=bond_colours,
            highlightAtomRadii=radii,
        )
        # Invisible click targets at the atom positions.
        hits = "".join(
            f'<circle class="es-hit" data-atom="{i}" cx="{p.x:.1f}" cy="{p.y:.1f}" r="13" '
            'fill="transparent" style="cursor:pointer"/>'
            for i in range(mol.GetNumAtoms())
            for p in [drawer.GetDrawCoords(i)]
        )
        drawer.FinishDrawing()
        self.svg = drawer.GetDrawingText().replace("</svg>", hits + "</svg>")


class ECFPMovie(anywidget.AnyWidget):
    """The "Inside ECFP4" explainer: an ~80 s scripted animation of how N-methylacetamide becomes
    a folded bit vector, and what Tanimoto similarity makes of it. Identifiers are real RDKit
    values; the page is self-contained HTML shown in an iframe."""

    _esm = (_STATIC / "movie.js").read_text()
    page = traitlets.Unicode("").tag(sync=True)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(page=(_STATIC / "ecfp_movie.html").read_text(), **kwargs)


def _env_svg(mol: Chem.Mol, center: int, radius: int, size: tuple[int, int] = (160, 130)) -> str:
    """RDKit's own bit depiction (Draw.DrawMorganEnv) as an SVG string."""
    from rdkit.Chem import Draw

    svg = Draw.DrawMorganEnv(mol, center, radius, molSize=size, useSVG=True)
    return svg if isinstance(svg, str) else svg.data


class MorganBitTiles(_Computing):
    """A molecule next to its Morgan bits, each drawn like RDKit's DrawMorganBit.

    The right-hand list has one row per distinct identifier with its ``Draw.DrawMorganEnv``
    picture (blue: centre atom, yellow: aromatic, grey: ring atoms, light grey: neighbours
    outside the environment) and the bit it folds onto; hovering or clicking a row highlights
    that environment in the full molecule on the left. Rows that fold onto the same bit are
    marked in red. With a reference set, each row shows how many other substructures share its
    bit, and clicking it draws them.
    """

    _esm = _bundle("tiles.js")

    radius = traitlets.Int(2).tag(sync=True)
    n_bits = traitlets.Int(2048).tag(sync=True)
    label = traitlets.Unicode("").tag(sync=True)
    smiles = traitlets.Unicode("").tag(sync=True)
    tiles = traitlets.List().tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)  # bit
    gallery = traitlets.List().tag(sync=True)

    def __init__(
        self,
        smiles: str,
        reference: list[str] | None = None,
        ids: list[str] | None = None,
        max_gallery: int = 16,
        **kwargs,
    ):
        self._smiles = smiles
        self._mol = Chem.MolFromSmiles(smiles)
        self._reference = list(reference) if reference is not None else None
        self._ids = [str(i) for i in ids] if ids is not None else None
        self._max_gallery = max_gallery
        super().__init__(smiles=smiles, **kwargs)  # same atom order as self._mol
        self.observe(self._refresh, names=["radius", "n_bits"])
        self.observe(self._refresh_gallery, names=["selected"])
        self._signal_done(["radius", "n_bits", "selected"])
        self._refresh()

    def _census(self) -> BitCensus | None:
        if self._reference is None:
            return None
        return census_for(self._reference, self.radius, self.n_bits)

    def _refresh(self, _change=None) -> None:
        from .chem import molecule_bit_tiles

        census = self._census()
        tiles = []
        for t in molecule_bit_tiles(self._smiles, self.radius, self.n_bits):
            row = {**t, "svg": _env_svg(self._mol, t["center"], t["radius"], (120, 90))}
            if census is not None:
                row["n_envs"] = int(census.n_envs[t["bit"]])
                row["n_on"] = int(census.on[:, t["bit"]].sum())
            tiles.append(row)
        self.tiles = tiles
        # start with the first bit selected so the molecule highlight and gallery are visible
        self.selected = int(tiles[0]["bit"]) if tiles else -1
        self._refresh_gallery()

    def _refresh_gallery(self, _change=None) -> None:
        census = self._census()
        if census is None or self.selected < 0 or self._reference is None:
            self.gallery = []
            return
        mine = {t["uid"] for t in self.tiles if t["bit"] == self.selected}
        rows = []
        for ex in census.examples.get(self.selected, [])[: self._max_gallery]:
            parent = Chem.MolFromSmiles(self._reference[ex["mol_index"]])
            rows.append(
                {
                    "svg": _env_svg(parent, ex["center"], ex["radius"]),
                    "env": ex["smiles"],
                    "radius": ex["radius"],
                    "count": ex["count"],
                    "id": self._ids[ex["mol_index"]] if self._ids else str(ex["mol_index"]),
                    "mine": ex["uid"] in mine,
                }
            )
        self.gallery = rows


class BitAtlas(_Computing):
    """Every folded bit of a reference set, one row per bit.

    Each row lists the distinct substructures (Morgan environments) from the reference set that
    fold onto that bit, drawn with ``Draw.DrawMorganEnv``, next to how many molecules set the bit.
    Rows are paged and can be sorted by bit index, by number of substructures or by number of
    molecules. Pictures are rendered in Python for the visible page only and cached.
    """

    _esm = _bundle("atlas.js")

    radius = traitlets.Int(2).tag(sync=True)
    n_bits = traitlets.Int(2048).tag(sync=True)
    sort = traitlets.Unicode("bit").tag(sync=True)  # "bit" | "envs" | "mols"
    page = traitlets.Int(0).tag(sync=True)
    page_size = traitlets.Int(20).tag(sync=True)
    per_row = traitlets.Int(10).tag(sync=True)
    focus = traitlets.Int(-1).tag(sync=True)  # bit to highlight after a jump (set in JS)
    order = traitlets.List(traitlets.Int()).tag(sync=True)  # bit shown at each position
    rows = traitlets.List().tag(sync=True)
    summary = traitlets.Dict().tag(sync=True)

    def __init__(self, reference: list[str], ids: list[str] | None = None, **kwargs):
        self._reference = list(reference)
        self._ids = [str(i) for i in ids] if ids is not None else None
        self._mols: dict[int, Chem.Mol] = {}
        self._svgs: dict[tuple[int, int, int], str] = {}
        super().__init__(**kwargs)
        self.observe(self._reset, names=["radius", "n_bits", "sort"])
        self.observe(self._refresh, names=["page", "page_size", "per_row"])
        self._signal_done(["radius", "n_bits", "sort", "page", "page_size", "per_row"])
        self._reset()

    def _census(self) -> BitCensus:
        return census_for(self._reference, self.radius, self.n_bits)

    def _order(self) -> np.ndarray:
        census = self._census()
        bits = np.arange(self.n_bits)
        if self.sort == "envs":
            return bits[np.lexsort((bits, -census.n_envs))]
        if self.sort == "mols":
            return bits[np.lexsort((bits, -census.on.sum(0)))]
        return bits

    def _svg(self, mol_index: int, center: int, radius: int) -> str:
        key = (mol_index, center, radius)
        if key not in self._svgs:
            if mol_index not in self._mols:
                self._mols[mol_index] = Chem.MolFromSmiles(self._reference[mol_index])
            self._svgs[key] = _env_svg(self._mols[mol_index], center, radius, (110, 90))
        return self._svgs[key]

    def _reset(self, _change=None) -> None:
        census = self._census()
        n_envs = census.n_envs
        self.summary = {
            "n_mols": census.n_mols,
            "n_envs": int(n_envs.sum()),
            "used_bits": int((census.on.sum(0) > 0).sum()),
            "max_envs": int(n_envs.max()) if len(n_envs) else 0,
            "max_mols": int(census.on.sum(0).max()) if census.n_mols else 0,
        }
        self.order = [int(b) for b in self._order()]
        self.focus = -1
        if self.page != 0:
            self.page = 0  # triggers _refresh
        else:
            self._refresh()

    def _refresh(self, _change=None) -> None:
        census = self._census()
        n_on = census.on.sum(0)
        order = self.order or [int(b) for b in self._order()]
        start = self.page * self.page_size
        rows = []
        for bit in order[start : start + self.page_size]:
            examples = census.examples.get(int(bit), [])
            rows.append(
                {
                    "bit": int(bit),
                    "n_envs": len(examples),
                    "n_mols": int(n_on[bit]),
                    "envs": [
                        {
                            "svg": self._svg(ex["mol_index"], ex["center"], ex["radius"]),
                            "env": ex["smiles"],
                            "radius": ex["radius"],
                            "count": ex["count"],
                            "id": self._ids[ex["mol_index"]] if self._ids else str(ex["mol_index"]),
                        }
                        for ex in examples[: self.per_row]
                    ],
                }
            )
        self.rows = rows


class BitImportance(_Computing):
    """A fingerprint model's bits ranked by importance, and what is behind the selected bit.

    ``importance`` maps a name to a per-bit array (length ``n_bits``), e.g. LightGBM gain and
    mean |SHAP|; the table ranks every bit by the chosen one (or by how many molecules set it, or
    how many substructures share it), in either direction, so bits the model never uses can be
    found too. ``effect`` is an optional signed per-bit value (e.g. mean SHAP in the molecules that
    set the bit) shown as the direction. Below the table, the selected bit's substructures in
    ``reference`` and the molecules that set it, with the responsible atoms highlighted; clicking a
    substructure keeps only the molecules that contain it. Pass ``y`` to label (and order) them.
    """

    _esm = _bundle("importance.js")

    radius = traitlets.Int(2).tag(sync=True)
    n_bits = traitlets.Int(2048).tag(sync=True)
    score_names = traitlets.List(traitlets.Unicode()).tag(sync=True)
    sort = traitlets.Unicode("").tag(sync=True)  # an importance name, "mols" or "envs"
    descending = traitlets.Bool(True).tag(sync=True)
    page = traitlets.Int(0).tag(sync=True)
    page_size = traitlets.Int(30).tag(sync=True)
    order = traitlets.List(traitlets.Int()).tag(sync=True)  # every bit, in rank order
    max_score = traitlets.Dict().tag(sync=True)  # largest share per importance, to scale the bars
    effect_label = traitlets.Unicode("").tag(sync=True)
    y_label = traitlets.Unicode("").tag(sync=True)
    rows = traitlets.List().tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)
    detail = traitlets.Dict().tag(sync=True)
    mol_filter = traitlets.Int(-1).tag(sync=True)  # unfolded identifier of one substructure, or -1
    mol_page = traitlets.Int(0).tag(sync=True)
    mol_page_size = traitlets.Int(24).tag(sync=True)
    mols = traitlets.Dict().tag(sync=True)  # {"total": n, "items": [...]} for the selected bit

    def __init__(
        self,
        reference: list[str],
        importance: dict[str, Any],
        effect: Any = None,
        effect_label: str = "effect",
        radius: int = 2,
        ids: list[str] | None = None,
        y: Any = None,
        y_label: str = "y",
        **kwargs,
    ):
        self._reference = list(reference)
        self._ids = [str(i) for i in ids] if ids is not None else None
        self._y = None if y is None else np.asarray(y, dtype=float)
        self._imp = {k: np.asarray(v, dtype=float) for k, v in importance.items()}
        self._share = {k: v / v.sum() if v.sum() else v for k, v in self._imp.items()}
        self._effect = None if effect is None else np.asarray(effect, dtype=float)
        self._mols: dict[int, Chem.Mol] = {}
        self._svgs: dict[tuple[int, int, int, int], str] = {}
        self._hits: dict[int, list[dict]] = {}  # molecule -> its substructures in the selected bit
        self._uid_index: dict[int, int] = {}  # substructure -> its position in the selected bit
        self._quiet = False
        kwargs.setdefault("sort", next(iter(self._imp)))
        super().__init__(
            radius=radius,
            n_bits=len(next(iter(self._imp.values()))),
            score_names=list(self._imp),
            max_score={k: float(v.max()) for k, v in self._share.items()},
            effect_label=effect_label if effect is not None else "",
            y_label=y_label if y is not None else "",
            **kwargs,
        )
        self.observe(self._rank, names=["sort", "descending"])
        self.observe(self._refresh, names=["page", "page_size"])
        self.observe(self._select, names=["selected"])
        self.observe(self._mol_view_changed, names=["mol_filter", "mol_page"])
        self._signal_done(
            ["sort", "descending", "page", "page_size", "selected", "mol_filter", "mol_page"]
        )
        self._rank()

    def _census(self) -> BitCensus:
        return census_for(self._reference, self.radius, self.n_bits)

    def _mol(self, i: int) -> Chem.Mol:
        if i not in self._mols:
            self._mols[i] = Chem.MolFromSmiles(self._reference[i])
        return self._mols[i]

    def _svg(self, ex: dict, size: tuple[int, int]) -> str:
        key = (ex["mol_index"], ex["center"], ex["radius"], size[0])
        if key not in self._svgs:
            self._svgs[key] = _env_svg(self._mol(ex["mol_index"]), ex["center"], ex["radius"], size)
        return self._svgs[key]

    def _rank(self, _change=None) -> None:
        census = self._census()
        n_on = census.on.sum(0)
        key = {"mols": n_on, "envs": census.n_envs}.get(self.sort)
        if key is None:
            key = self._imp[self.sort]
        bits = np.arange(self.n_bits)
        # ties (e.g. every bit the model never splits on) go to the bits most molecules set
        order = np.lexsort((bits, -n_on, -key if self.descending else key))
        self.order = [int(b) for b in order]
        if self.page != 0:
            self.page = 0  # triggers _refresh
        else:
            self._refresh()

    def _refresh(self, _change=None) -> None:
        census = self._census()
        n_on = census.on.sum(0)
        start = self.page * self.page_size
        rows = []
        for rank, bit in enumerate(self.order[start : start + self.page_size], start=start + 1):
            examples = census.examples.get(int(bit), [])
            main = examples[0] if examples else None
            rows.append(
                {
                    "rank": rank,
                    "bit": int(bit),
                    "scores": {k: float(v[bit]) for k, v in self._share.items()},
                    "effect": None if self._effect is None else float(self._effect[bit]),
                    "n_envs": len(examples),
                    "n_mols": int(n_on[bit]),
                    # the most common substructure, and the share of the bit's molecules it covers
                    "main_svg": self._svg(main, (96, 72)) if main else "",
                    "main_env": main["smiles"] if main else "",
                    "main_share": main["count"] / max(int(n_on[bit]), 1) if main else 0.0,
                }
            )
        self.rows = rows
        if self.selected < 0 and rows:
            self.selected = rows[0]["bit"]  # opens the top bit

    def _select(self, _change=None) -> None:
        self._hits = {}
        census = self._census()
        bit = self.selected
        examples = census.examples.get(bit, []) if bit >= 0 else []
        self._uid_index = {ex["uid"]: k for k, ex in enumerate(examples)}
        self.detail = (
            {
                "bit": bit,
                "n_mols": int(census.on[:, bit].sum()),
                "envs": [
                    {
                        "uid": ex["uid"],
                        "svg": self._svg(ex, (130, 100)),
                        "env": ex["smiles"],
                        "radius": ex["radius"],
                        "count": ex["count"],
                        "id": self._ids[ex["mol_index"]] if self._ids else str(ex["mol_index"]),
                    }
                    for ex in examples
                ],
            }
            if bit >= 0
            else {}
        )
        self._quiet = True
        self.mol_filter, self.mol_page = -1, 0
        self._quiet = False
        self._refresh_mols()

    def _mol_view_changed(self, change) -> None:
        if self._quiet:
            return
        if change["name"] == "mol_filter" and self.mol_page:
            self._quiet = True
            self.mol_page = 0
            self._quiet = False
        self._refresh_mols()

    def _hits_of(self, i: int) -> list[dict]:
        """The substructures of molecule i that set the selected bit, with the atoms of each."""
        from .chem import molecule_bit_tiles

        if i not in self._hits:
            self._hits[i] = [
                {
                    "uid": t["uid"],
                    "k": self._uid_index.get(t["uid"], -1),
                    "atoms": sorted({a for w in t["where"] for a in w["atoms"]}),
                    "bonds": sorted({b for w in t["where"] for b in w["bonds"]}),
                }
                for t in molecule_bit_tiles(self._reference[i], self.radius, self.n_bits)
                if t["bit"] == self.selected
            ]
        return self._hits[i]

    def _refresh_mols(self) -> None:
        bit = self.selected
        if bit < 0:
            self.mols = {}
            return
        idx = np.flatnonzero(self._census().on[:, bit])
        if self._y is not None:  # most active first
            idx = idx[np.argsort(-np.nan_to_num(self._y[idx], nan=-np.inf), kind="stable")]
        if self.mol_filter >= 0:
            idx = [
                i for i in idx if any(h["uid"] == self.mol_filter for h in self._hits_of(int(i)))
            ]
        start = self.mol_page * self.mol_page_size
        self.mols = {
            "total": len(idx),
            "items": [
                {
                    "id": self._ids[i] if self._ids else str(i),
                    "smiles": self._reference[i],
                    "y": None if self._y is None else float(self._y[i]),
                    "hits": self._hits_of(int(i)),
                }
                for i in (int(i) for i in idx[start : start + self.mol_page_size])
            ],
        }


# Descriptors MolPair can compare, by name; pass your own {name: fn(mol) -> float} as well.
def _descriptor_table() -> dict[str, Any]:
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    return {
        "MW": Descriptors.MolWt,  # ty: ignore[unresolved-attribute]
        "cLogP": Crippen.MolLogP,  # ty: ignore[unresolved-attribute]
        "TPSA": rdMolDescriptors.CalcTPSA,
        "HBD": Lipinski.NumHDonors,  # ty: ignore[unresolved-attribute]
        "HBA": Lipinski.NumHAcceptors,  # ty: ignore[unresolved-attribute]
        "RotB": rdMolDescriptors.CalcNumRotatableBonds,
        "heavy atoms": lambda m: m.GetNumHeavyAtoms(),
        "rings": rdMolDescriptors.CalcNumRings,
        "arom. rings": rdMolDescriptors.CalcNumAromaticRings,
        "Fsp3": rdMolDescriptors.CalcFractionCSP3,
    }


PAIR_DESCRIPTORS = ["MW", "cLogP", "TPSA", "HBD", "HBA", "RotB", "heavy atoms", "rings"]
_ONLY = (1.0, 0.72, 0.3)  # atoms in one molecule only (orange)
_CHANGED = (0.84, 0.6, 1.0)  # matched atoms whose element / charge / stereo differ (violet)
_CORE = (0.62, 0.8, 1.0)  # the common substructure (light blue)


class MolPair(_Computing):
    """Two molecules side by side, drawn in the same orientation, with what differs marked.

    The maximum common substructure (any element may match any other; bonds and rings must
    agree) maps B's atoms onto A's, and B is laid out on A's coordinates. Atoms found in only
    one molecule are orange; matched atoms whose element, charge or R/S label differ are violet
    (bonds whose E/Z differs too). ``highlight="core"`` shades the common part instead.
    Hovering an atom rings its partner in the other molecule.

    ``a`` and ``b`` are dicts (a DataFrame row works) or SMILES strings. ``value_cols`` are shown
    under each drawing; ``delta_cols`` get B − A between them, and ``fold_cols`` also the
    fold-change ``10**|Δ|`` (for log units such as pEC50). ``descriptors`` are RDKit
    descriptors compared in a table (names from ``PAIR_DESCRIPTORS`` or ``{name: fn(mol)}``).
    The centre also shows the Morgan Tanimoto at ``radius`` / ``n_bits``.
    """

    _esm = _bundle("molpair.js")

    molecules = traitlets.List(traitlets.Dict()).tag(sync=True)
    highlight = traitlets.Unicode("diff").tag(sync=True)  # "diff" | "core" | "none"
    stereo_labels = traitlets.Bool(False).tag(sync=True)
    radius = traitlets.Int(2).tag(sync=True)
    n_bits = traitlets.Int(2048).tag(sync=True)
    note = traitlets.Unicode("").tag(sync=True)
    # computed in Python
    view = traitlets.Dict().tag(sync=True)
    mapping = traitlets.List().tag(sync=True)  # [[atom in A, atom in B], ...]
    mcs_smarts = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        a: Any,
        b: Any,
        id_col: str = "id",
        smiles_col: str = "smiles",
        value_cols: list[str] | None = None,
        delta_cols: list[str] | None = None,
        fold_cols: list[str] | None = None,
        formats: dict[str, str] | None = None,
        descriptors: list[str] | dict[str, Any] | None = None,
        mcs_timeout: float = 2.0,
        size: tuple[int, int] = (300, 220),
        **kwargs: Any,
    ) -> None:
        self._id_col, self._smiles_col = id_col, smiles_col
        self._value_cols = list(value_cols or [])
        self._delta_cols = list(self._value_cols if delta_cols is None else delta_cols)
        self._fold_cols = list(fold_cols or [])
        self._formats = formats or {}
        table = _descriptor_table()
        if descriptors is None:
            descriptors = PAIR_DESCRIPTORS
        self._desc_fns = (
            dict(descriptors)
            if isinstance(descriptors, dict)
            else {name: table[name] for name in descriptors}
        )
        self._timeout = mcs_timeout
        self._size = size
        super().__init__(molecules=[self._as_dict(a, "A"), self._as_dict(b, "B")], **kwargs)
        self.observe(self._align, names=["molecules"])
        self.observe(self._render, names=["highlight", "stereo_labels"])
        self.observe(self._summarise, names=["radius", "n_bits"])
        self._signal_done(["molecules", "highlight", "stereo_labels", "radius", "n_bits"])
        self._align()

    def _as_dict(self, m: Any, fallback_id: str) -> dict:
        if isinstance(m, str):
            m = {self._smiles_col: m}
        m = {k: _clean(v) for k, v in dict(m).items()}
        return {
            "id": str(m.get(self._id_col, fallback_id)),
            "smiles": m[self._smiles_col],
            "values": {c: m.get(c) for c in self._value_cols},
        }

    def set_pair(self, a: Any, b: Any) -> None:
        """Show another pair in the same widget."""
        self.molecules = [self._as_dict(a, "A"), self._as_dict(b, "B")]

    def _align(self, _change=None) -> None:
        from .chem import pair_alignment

        a, b = self.molecules
        self._al = pair_alignment(a["smiles"], b["smiles"], self._timeout)
        with self.hold_sync():
            self.mapping = [list(p) for p in self._al["mapping"]]
            self.mcs_smarts = self._al["mcs_smarts"]
            self._summarise()
            self._render()

    def _summarise(self, _change=None) -> None:
        from .chem import fingerprint_matrix, tanimoto_matrix

        a, b = self.molecules
        fps = fingerprint_matrix([a["smiles"], b["smiles"]], self.radius, self.n_bits)
        al = self._al
        deltas = []
        for c in self._delta_cols:
            va, vb = a["values"].get(c), b["values"].get(c)
            if va is None or vb is None:
                continue
            d = vb - va
            deltas.append(
                {"name": c, "delta": d, "fold": 10 ** abs(d) if c in self._fold_cols else None}
            )
        mols = [al["mol_a"], al["mol_b"]]
        desc = []
        for name, fn in self._desc_fns.items():
            va, vb = (float(fn(m)) for m in mols)
            desc.append({"name": name, "a": va, "b": vb})
        self._summary = {
            "tanimoto": float(tanimoto_matrix(fps)[0, 1]),
            "fp": f"ECFP{2 * self.radius}"
            if self.n_bits == 2048
            else f"Morgan r{self.radius}/{self.n_bits}",
            "deltas": deltas,
            "descriptors": desc,
            "core": len(al["mapping"]),
            "n_a": al["mol_a"].GetNumAtoms(),
            "n_b": al["mol_b"].GetNumAtoms(),
            "only": [len(al["only_a"]), len(al["only_b"])],
            "changed": len(al["changed_a"]),
            "changed_bonds": len(al["changed_bonds_a"]),
            "timed_out": al["timed_out"],
        }
        if self.view:
            self.view = {**self.view, "summary": self._summary}

    def _bond_length(self) -> float:
        """One scale for both drawings: the largest bond length at which both still fit."""
        w, h = self._size
        fits = []
        for m in (self._al["mol_a"], self._al["mol_b"]):
            pos = m.GetConformer().GetPositions()
            span_x, span_y = np.ptp(pos[:, 0]) + 1.5, np.ptp(pos[:, 1]) + 1.5  # room for labels
            fits.append(min(w * 0.88 / span_x, h * 0.88 / span_y) * 1.5)  # coords: bonds ~1.5 Å
        return float(min(min(fits), 40.0))

    def _render(self, _change=None) -> None:
        from rdkit.Chem.Draw import rdMolDraw2D

        al = self._al
        bond_len = self._bond_length()
        panels = []
        for side in ("a", "b"):
            mol = Chem.Mol(al[f"mol_{side}"])
            atoms: dict[int, tuple] = {}
            bonds: dict[int, tuple] = {}
            if self.highlight == "diff":
                atoms.update({i: _ONLY for i in al[f"only_{side}"]})
                bonds.update({i: _ONLY for i in al[f"only_bonds_{side}"]})
                atoms.update({i: _CHANGED for i in al[f"changed_{side}"]})
                bonds.update({i: _CHANGED for i in al[f"changed_bonds_{side}"]})
            elif self.highlight == "core":
                mine = [p[0 if side == "a" else 1] for p in al["mapping"]]
                atoms.update({i: _CORE for i in mine})
                bonds.update(
                    {
                        bd.GetIdx(): _CORE
                        for bd in mol.GetBonds()
                        if bd.GetIdx() not in al[f"only_bonds_{side}"]
                        and bd.GetBeginAtomIdx() in mine
                        and bd.GetEndAtomIdx() in mine
                    }
                )
            drawer = rdMolDraw2D.MolDraw2DSVG(*self._size)
            opts = drawer.drawOptions()
            opts.clearBackground = False
            opts.fixedBondLength = bond_len
            opts.addStereoAnnotation = self.stereo_labels
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer,
                mol,
                highlightAtoms=list(atoms),
                highlightAtomColors=atoms,
                highlightBonds=list(bonds),
                highlightBondColors=bonds,
            )
            coords = [
                [round(p.x, 1), round(p.y, 1)]
                for i in range(mol.GetNumAtoms())
                for p in [drawer.GetDrawCoords(i)]
            ]
            drawer.FinishDrawing()
            symbols = [at.GetSymbol() for at in mol.GetAtoms()]
            panels.append({"svg": drawer.GetDrawingText(), "coords": coords, "symbols": symbols})
        self.view = {"panels": panels, "summary": self._summary, "formats": self._formats}


# Colour key for Draw.DrawMorganEnv's default colours (same as MORGAN_ENV_KEY in rdkit_loader.js).
_ENV_KEY_HTML = (
    '<i style="background:rgb(153,153,230);margin-left:0"></i>centre atom'
    '<i style="background:rgb(230,230,51)"></i>aromatic atom'
    '<i style="background:rgb(204,204,204)"></i>aliphatic ring atom'
    '<i style="background:rgb(230,230,230);border:1px dashed #9ca3af"></i>* where it attaches'
)


def bit_gallery(
    bit: int,
    reference: list[str],
    ids: list[str] | None = None,
    radius: int = 2,
    n_bits: int = 2048,
    max_items: int = 12,
) -> str:
    """HTML grid of every distinct environment that folds onto ``bit`` in a reference set.

    Uses the same RDKit ``DrawMorganEnv`` pictures as :class:`MorganBitTiles`, so a bit looks the
    same wherever it appears in a notebook. Wrap the result in ``mo.Html``.
    """
    import html

    examples = census_for(reference, radius, n_bits).examples.get(bit, [])
    cards = []
    for ex in examples[:max_items]:
        mol = Chem.MolFromSmiles(reference[ex["mol_index"]])
        label = ids[ex["mol_index"]] if ids else str(ex["mol_index"])
        cards.append(
            '<div class="mwg-card"><div class="mwg-head">'
            f"<span>r{ex['radius']}</span><span>in {ex['count']} mols</span></div>"
            f"{_env_svg(mol, ex['center'], ex['radius'], (150, 120))}"
            f'<div class="mwg-env">{html.escape(ex["smiles"])}</div>'
            f'<div class="mwg-id">e.g. {html.escape(str(label))}</div></div>'
        )
    more = len(examples) - len(cards)
    return (
        "<style>.mwg-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px}"
        ".mwg-card{background:#fff;color:#1f2328;border:1px solid #d0d7de;border-radius:8px;padding:4px 6px}"
        ".mwg-card svg{width:100%;height:auto;display:block}"
        ".mwg-head{display:flex;justify-content:space-between;font-size:11px;color:#6b7280}"
        ".mwg-env{font:11px ui-monospace,monospace;color:#6b7280;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}"
        ".mwg-id{font-size:10.5px;color:#6b7280}.mwg-title{margin:0 0 6px;font-weight:600}"
        ".mwg-key{font-size:11.5px;color:#6b7280;margin:0 0 6px}"
        ".mwg-key i{display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin:0 4px 0 10px}</style>"
        f'<div class="mwg-title">Bit {bit}: {len(examples)} different environment'
        f"{'s' if len(examples) != 1 else ''} in the reference set"
        f"{f' (first {len(cards)} shown)' if more > 0 else ''}</div>"
        f'<div class="mwg-key">{_ENV_KEY_HTML}</div>'
        f'<div class="mwg-grid">{"".join(cards)}</div>'
    )
