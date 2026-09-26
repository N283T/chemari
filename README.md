# Similar, but not the same — Morgan fingerprints on PXR

A [marimo](https://marimo.io) notebook for the molab Notebook Competition #3 (OpenADMET × marimo).
It asks why Morgan-fingerprint models struggle on the OpenADMET PXR induction data, and answers
it with two custom [anywidget](https://anywidget.dev) components.

- `notebooks/pxr_fingerprints.py` — the notebook
- `src/molwidgets/` — the widget package
  - `MolGrid`: paged, sortable molecule grid with text and SMARTS filtering, colour scale, and
    two-way selection (`single`, `multiple` or `pair` mode)
  - `MorganExplorer`: bit-by-bit view of a Morgan fingerprint for one molecule or a pair, with the
    atom environments behind each bit, dataset statistics per bit, and a gallery of the different
    substructures that collide in a selected bit

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
uv run marimo check notebooks/pxr_fingerprints.py
```
