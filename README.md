# Do you really know your ECFP4?

A [marimo](https://marimo.io) notebook for the molab Notebook Competition #3 (OpenADMET × marimo).
Part 1 shows what ECFP4 computes; part 2 uses the OpenADMET PXR induction data, a dataset where
ECFP4 models do poorly, to show where and why it fails. Custom [anywidget](https://anywidget.dev)
components are used throughout.

- `notebooks/ecfp_pxr.py`: **the notebook**. An 80 s ECFP4 video and a step-by-step version for
  any molecule, ECFP4 on real molecules and a
  dataset-wide bit atlas, how the PXR test set was built, where ECFP4 breaks (similarity principle,
  activity cliffs, identical fingerprints, whole-molecule properties), LightGBM importance and
  TreeSHAP on bits, and a model lab
- `notebooks/ecfp_pxr_ja.py`: the same notebook in Japanese
- `src/molwidgets/` — the widget package
  - `MolGrid`: paged, sortable molecule grid with text and SMARTS filtering, colour scale, and
    two-way selection (`single`, `multiple` or `pair` mode)
  - `ECFPMovie`: an ~80 s animated explainer of ECFP4 (radius 0 → 2, duplicates, folding,
    collisions, Tanimoto) for N-methylacetamide, built on real RDKit identifiers; a self-contained
    HTML page (`static/ecfp_movie.html`) shown in an iframe
  - `ECFPStepper`: the `ECFPMovie` story for any molecule (type a SMILES or pick an example). A
    guided tour goes atom by atom through every radius, showing the hashed invariants or the
    sorted (bond, neighbour) pairs, dropped duplicate environments and the growing feature set,
    then folds the features into a bit vector and lists the collisions; it can play itself or be
    stepped by hand, and an explore mode lets you click atoms. Identifiers are RDKit's own values,
    lettered as in the movie
  - `MorganBitTiles`: a molecule's fingerprint drawn bit by bit like RDKit's `DrawMorganBits`,
    with in-molecule collisions framed and, per bit, every other substructure in a dataset that
    folds onto it
  - `BitAtlas`: every folded bit of a dataset, one row per bit, with the distinct substructures
    that fold onto it; sortable by bit index, number of substructures or number of molecules
  - `BitImportance`: a fingerprint model's most important bits (e.g. LightGBM gain or mean |SHAP|),
    ranked, with each bit's direction, its most common substructure and how many distinct
    substructures share it; clicking a bit lists every substructure that folds onto it
  - `MorganExplorer`: bit-by-bit view of a Morgan fingerprint for one molecule or a pair, with the
    atom environments behind each bit, dataset statistics per bit, and a gallery of the different
    substructures that collide in a selected bit; optionally per-bit model contributions
    (e.g. LightGBM TreeSHAP, pinned to the right of the table) and a per-atom attribution map;
    `stereo_labels=True` adds R/S and E/Z labels to the drawings
- `src/molwidgets/ecfp.py` — a readable re-implementation of ECFP, and `ecfp_story`, which collects
  RDKit's identifiers for every atom and radius for `ECFPStepper`

Molecules are drawn in the browser with RDKit.js; standardization, fingerprints, bit environments
and collision statistics are computed with RDKit in Python. When a click needs Python to recompute (a new
page, radius or bit), the widget dims and shows "computing…" until the answer arrives.

## Run locally

```bash
uv sync
uv run marimo edit notebooks/ecfp_pxr.py
```

The notebook downloads the data from Hugging Face on first run
([openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test),
CC-BY-4.0). If `data/` contains the CSV files it uses them instead.

## Use the widgets elsewhere

```python
import marimo as mo
from molwidgets import MolGrid, MorganExplorer

grid = mo.ui.anywidget(MolGrid(df, id_col="id", smiles_col="smiles", color_by="pEC50"))
grid.value["selection"]  # ids of the selected molecules

explorer = mo.ui.anywidget(
    MorganExplorer(
        [{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "CCN"}],
        reference=df["smiles"].to_list(),
        y=df["pEC50"].to_numpy(),
        y_label="pEC50",
    )
)
```

## Development

```bash
uv run pytest
uv run ruff format . && uv run ruff check --fix .
uv run marimo check notebooks/*.py
```
