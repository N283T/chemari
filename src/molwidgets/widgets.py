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

from .chem import BitCensus, bit_census, env_atoms_bonds, morgan_bits

_STATIC = Path(__file__).parent / "static"
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


class MorganExplorer(anywidget.AnyWidget):
    """Explain a Morgan fingerprint bit by bit, for one molecule or a pair.

    Pass `reference` SMILES (and optionally activities `y`) to get dataset context:
    how many molecules set each bit, how many *different* substructures collide in it,
    and the mean activity with the bit on vs off. `selected_bit` syncs back to Python.
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


def _id_colour(identifier: int) -> tuple[float, float, float]:
    """A stable pastel colour per identifier, so equal identifiers look equal."""
    import colorsys

    hue = (identifier * 0.618033988749895) % 1.0
    return colorsys.hls_to_rgb(hue, 0.72, 0.75)


class ECFPStepper(anywidget.AnyWidget):
    """Step through the ECFP/Morgan algorithm for one molecule.

    Pick an iteration to see every atom's identifier (equal colours = equal identifiers),
    click an atom to see what was hashed to build its identifier and whether the environment
    was kept, and fold the collected identifiers into a bit vector to watch collisions appear.
    Identifiers come from :func:`molwidgets.ecfp.ecfp_trace` (same features as RDKit, different
    hash function).
    """

    _esm = _bundle("stepper.js")

    radius = traitlets.Int(1).tag(sync=True)
    atom = traitlets.Int(-1).tag(sync=True)
    n_bits = traitlets.Int(64).tag(sync=True)
    steps = traitlets.List().tag(sync=True)
    svg = traitlets.Unicode("").tag(sync=True)
    invariant_names = traitlets.List(traitlets.Unicode()).tag(sync=True)

    def __init__(self, smiles: str, max_radius: int = 3, **kwargs):
        from .ecfp import INVARIANT_NAMES, ecfp_trace

        self._mol = Chem.MolFromSmiles(smiles)
        trace = ecfp_trace(smiles, max_radius)
        steps = [[s.__dict__ for s in layer] for layer in trace.steps]
        super().__init__(steps=steps, invariant_names=list(INVARIANT_NAMES), **kwargs)
        self.observe(self._render, names=["radius", "atom"])
        self._render()

    def _render(self, _change=None) -> None:
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = Chem.Mol(self._mol)
        layer = self.steps[min(self.radius, len(self.steps) - 1)]
        for s in layer:
            mol.GetAtomWithIdx(s["atom"]).SetProp("atomNote", f"{s['identifier']:08x}"[:3])
        colours = {s["atom"]: _id_colour(s["identifier"]) for s in layer}
        atoms = list(colours)
        bond_colours: dict[int, tuple[float, float, float]] = {}
        radii = {a: 0.32 for a in atoms}
        if self.atom >= 0:
            sel = layer[self.atom]
            focus = set(sel["atoms"])
            c = _id_colour(sel["identifier"])
            fade = (0.93, 0.93, 0.93)
            colours = {a: (c if a in focus else fade) for a in atoms}
            colours[self.atom] = (
                max(0.0, c[0] - 0.25),
                max(0.0, c[1] - 0.25),
                max(0.0, c[2] - 0.25),
            )
            bond_colours = {b: c for b in sel["bonds"]}
            radii = {a: (0.42 if a == self.atom else 0.32) for a in atoms}
        drawer = rdMolDraw2D.MolDraw2DSVG(460, 340)
        opts = drawer.drawOptions()
        opts.annotationFontScale = 0.6
        opts.fixedBondLength = 34
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
