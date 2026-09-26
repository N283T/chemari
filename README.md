# Similar, but not the same — Morgan fingerprints on PXR

A [marimo](https://marimo.io) notebook for the molab Notebook Competition #3 (OpenADMET × marimo).
It asks why Morgan-fingerprint models struggle on the OpenADMET PXR induction data, and answers
it with two custom [anywidget](https://anywidget.dev) components.

- `notebooks/pxr_fingerprints.py` — the notebook
- `notebooks/pxr_fingerprints_ja.py` — Japanese edition (same analysis, Japanese prose)
- `notebooks/ecfp_inside.py` — *Inside ECFP4*, a hands-on tutorial on the Morgan/ECFP algorithm,
  folding and collisions, blind spots, and LightGBM importance / TreeSHAP maps on fingerprint bits
- `notebooks/ecfp_inside_ja.py` — Japanese edition of the tutorial
- `scripts/translate_*_ja.py` — regenerate the Japanese editions after editing the English notebooks
- `src/molwidgets/` — the widget package
  - `MolGrid`: paged, sortable molecule grid with text and SMARTS filtering, colour scale, and
    two-way selection (`single`, `multiple` or `pair` mode)
  - `ECFPStepper`: step through the ECFP algorithm iteration by iteration, with per-atom
    identifiers, the hashed "recipe" for each atom, duplicate removal, and the growing feature set
    shown both as identifiers and as a folded bit vector
  - `MorganBitTiles`: a molecule's fingerprint drawn bit by bit like RDKit's `DrawMorganBits`,
    with in-molecule collisions framed and, per bit, every other substructure in a dataset that
    folds onto it
  - `MorganExplorer`: bit-by-bit view of a Morgan fingerprint for one molecule or a pair, with the
    atom environments behind each bit, dataset statistics per bit, and a gallery of the different
    substructures that collide in a selected bit; optionally per-bit model contributions
    (e.g. LightGBM TreeSHAP) and a per-atom attribution map
- `src/molwidgets/ecfp.py` — a readable re-implementation of ECFP used by the tutorial

Molecules are drawn in the browser with RDKit.js; standardization, fingerprints, bit environments
and collision statistics are computed with RDKit in Python.

## Run locally

```bash
uv sync
uv run marimo edit notebooks/pxr_fingerprints.py
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

# after editing an English notebook, regenerate its Japanese edition
uv run python scripts/translate_pxr_fingerprints_ja.py
uv run python scripts/translate_ecfp_inside_ja.py
uv run ruff format notebooks
```
