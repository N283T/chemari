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
- `notebooks/ecfp_openadmet_ja.py`: **work in progress** (Japanese first). ECFP4 across three
  OpenADMET datasets (PXR, ASAP-Polaris antiviral, ExpansionRx; 16 endpoints): ECFP4's internals
  and two pitfalls (different molecules with the same fingerprint; near-identical molecules with a
  low Tanimoto; the same pair by other fingerprints, MCES and properties), then a dataset /
  endpoint picker that drives train–test distance, the similarity principle, cliffs, identical fingerprints, bit collisions, models on bit / count / descriptor
  features and TreeSHAP. Reads `results/precomputed/` (DuckDB SQL cells) and computes what is
  missing
- `src/molwidgets/` — the widget package. Every whole-molecule drawing has a copy icon in its
  top-right corner that puts the molecule's SMILES on the clipboard; `copy_smiles=False` hides
  them
  - `MolGrid`: paged, sortable molecule grid with text and SMARTS filtering, colour scale, and
    two-way selection (`single`, `multiple` or `pair` mode)
  - `MolPair`: two compounds side by side with the Tanimoto similarity (ECFP4; the bar is coloured
    and labelled by band, low / medium / high / very high by default, set with `similarity_bands`)
    and a table of
    properties (plus any values you pass, such as a measured pEC50). `properties` picks the rows
    from 11 built-in descriptors or your own functions; the default is the rule of five. Each row shows A and B as a
    dumbbell on the property's typical range and B − A as a bar; `value_ranges` sets the axis of
    your own values, and `show_formula` / `show_smiles` turn off the captions. Two switches,
    both off by default, highlight the common substructure and redraw B in A's orientation. Below
    the drawings, the table can switch to the difference: the common substructure with R1, R2, …
    where the two differ, and per site A's piece → B's piece, with a copy button for the common
    part's SMILES. `mcs={"atoms": "any"}` (options of `find_mcs`) keeps a ring CH → N
    matched and shows it as a changed element. A third view, "similarity", compares the pair by
    ECFP4 (bit and count), ECFP6, FCFP4, atom pair, topological torsion, RDKit path and MACCS
    fingerprints (Tanimoto, Dice or cosine), MCES (RDKit's RASCAL) and whole-molecule properties,
    and says whether the Murcko scaffolds match (`molwidgets.similarity`); a note under it says
    that each method has its own scale
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
    folds onto it; `mode="count"` shows each bit's count, and a chirality switch uses
    `includeChirality=True`
  - `BitAtlas`: every folded bit of a dataset, one row per bit, with the distinct substructures
    that fold onto it; sortable by bit index, number of substructures or number of molecules
  - `BitImportance`: every bit of a fingerprint model ranked by its importance (e.g. LightGBM gain
    or mean |SHAP|), in either direction so unused bits can be browsed too, with each bit's
    direction, its most common substructure and how many substructures and molecules it has;
    below, the selected bit's substructures and the molecules that set it, with the responsible
    atoms highlighted (click a substructure to keep only its molecules)
  - `MorganExplorer`: bit-by-bit view of a Morgan fingerprint for one molecule or a pair, with the
    atom environments behind each bit, dataset statistics per bit, and a gallery of the different
    substructures that collide in a selected bit; for a pair, the table can show only the shared
    bits, only the differing ones, or those of one molecule; optionally per-bit model contributions
    (e.g. LightGBM TreeSHAP, pinned to the right of the table) and a per-atom attribution map;
    `stereo_labels=True` adds R/S and E/Z labels to the drawings. A bit / count switch
    (`mode`) and a chirality switch (`chirality`) change the fingerprint; for a pair, the
    summary gives the Tanimoto of the current setting (Σmin / Σmax for counts) and the table
    lists each molecule's count and the bits whose counts differ
- `src/molwidgets/bench.py` — the datasets and endpoints of `ecfp_openadmet_ja.py` (loading,
  standardisation, log10(x + 1) for ratio-scale endpoints), features (ECFP4 bit / count, RDKit
  descriptors), the LightGBM baseline, nearest neighbours and TreeSHAP per endpoint;
  `dev/precompute.py` writes them to `results/precomputed/`
- `src/molwidgets/ecfp.py` — a readable re-implementation of ECFP, and `ecfp_story`, which collects
  RDKit's identifiers for every atom and radius for `ECFPStepper`

- `results/` — reference results computed outside the notebook (LightGBM on CheMeleon
  fingerprints, compared with the notebook's baselines), the precomputed tables of
  `ecfp_openadmet_ja.py`, and the analyses in `dev/analysis/` (nearest neighbours, pair
  differences with TreeSHAP, the same checks across datasets)

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
from molwidgets import MolGrid, MolPair, MorganExplorer

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

## Development

```bash
uv run pytest
uv run ruff format . && uv run ruff check --fix .
uv run marimo check notebooks/*.py
```
