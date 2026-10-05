# CheMari

Chemistry widgets for [marimo](https://marimo.io) notebooks: molecule grids, pairs and scatter
plots, and views inside Morgan fingerprints (ECFP). Built on [anywidget](https://anywidget.dev)
and RDKit.

> [!NOTE]
> CheMari is planned to grow into a cheminformatics package for marimo. For now it holds the
> widgets written for the notebooks in `examples/`.

## Install

```bash
pip install "chemari @ git+https://github.com/N283T/chemari"
```

## Usage

```python
import marimo as mo
from chemari import MolGrid, MolPair, MorganExplorer

grid = mo.ui.anywidget(MolGrid(df, id_col="id", smiles_col="smiles", color_by="pEC50"))
grid.value["selection"]  # ids of the selected molecules

pair = mo.ui.anywidget(MolPair(row_a, row_b, value_cols=["pEC50"]))  # dicts or SMILES

explorer = mo.ui.anywidget(
    MorganExplorer(
        [{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "CCN"}],
        reference=df["smiles"].to_list(),
        y=df["pEC50"].to_numpy(),
        y_label="pEC50",
    )
)
```

Molecules are drawn in the browser with RDKit.js. Fingerprints, bit environments and collision
statistics are computed with RDKit in Python.

## Widgets

Every widget with the code that makes it: [n283t.github.io/chemari](https://n283t.github.io/chemari/)
(a static page; the notebook behind it is `docs/widgets.py`).

| Widget | What it shows |
| --- | --- |
| `MolGrid` | A paged, sortable grid of molecules with text, SMARTS and similarity search, and selection |
| `MolPair` | Two compounds side by side: Tanimoto, properties, the common substructure and other similarity measures |
| `MolScatter` | A scatter plot of molecules (e.g. measured vs predicted) with the hovered or pinned molecule beside it |
| `ECFPMovie` | An 80-second animation of how ECFP4 is built |
| `ECFPStepper` | The same steps for any molecule, atom by atom and radius by radius |
| `MorganBitTiles` | The bits of one molecule, with collisions inside the molecule and across a dataset |
| `MorganExplorer` | The bits of one molecule or a pair, with the atoms behind each bit, dataset statistics and per-bit model contributions |
| `BitAtlas` | Every bit of a dataset with the substructures folded onto it |
| `BitImportance` | The bits of a fingerprint model ranked by importance, with their substructures and molecules |

## Examples

Notebooks in `examples/`, written for the molab Notebook Competition #3 (OpenADMET × marimo).

- **Do you really know your ECFP4?** — `ecfp_openadmet.py` (`ecfp_openadmet_ja.py` in Japanese)
  [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/N283T/chemari/blob/main/examples/ecfp_openadmet.py/server)

  ECFP4 across three OpenADMET datasets (PXR, ASAP-Polaris antiviral, ExpansionRx; 16 endpoints):
  how ECFP4 is built and two of its pitfalls, what it looks like inside each dataset, and a
  comparison of the 16 endpoints.
- **ECFP4 on the PXR data** — `ecfp_pxr.py` (`ecfp_pxr_ja.py` in Japanese)

  The earlier notebook on the PXR dataset alone.

```bash
uv sync
uv run marimo edit examples/ecfp_openadmet.py
```

The notebooks read the data from Hugging Face on first run (PXR, CC-BY-4.0; ASAP-Polaris, MIT;
ExpansionRx, CC-BY-4.0), or from `data/` when the CSV files are there. A notebook installed from
a release reads the precomputed tables of that release's tag.

## Repository

- `src/chemari/` — the package: the widgets (`widgets.py`, `static/`) and the chemistry behind
  them (`chem.py`, `ecfp.py`, `similarity.py`)
- `src/chemari/examples/openadmet.py` — the code behind `ecfp_openadmet`: datasets, features,
  the LightGBM baseline, nearest neighbours and TreeSHAP
- `docs/widgets.py` — the notebook of the documentation page, exported by `.github/workflows/docs.yml`
- `dev/precompute.py` — writes the tables in `results/precomputed/`
- `results/` — the precomputed tables and the analyses in `dev/analysis/`

## Development

```bash
uv run pytest
uv run ruff format . && uv run ruff check --fix .
uv run marimo check examples/*.py
```
