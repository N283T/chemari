# CheMari

Chemistry widgets for [marimo](https://marimo.io) notebooks: molecule grids, pairs, scatter plots
and views inside Morgan fingerprints (ECFP), built on [anywidget](https://anywidget.dev) and RDKit.

```bash
pip install "chemari @ git+https://github.com/N283T/chemari"
```

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

Molecules are drawn in the browser with RDKit.js; standardization, fingerprints, bit environments
and collision statistics are computed with RDKit in Python. When a click needs Python to recompute (a new
page, radius or bit), the widget dims and shows "computing…" until the answer arrives.

## Examples

Notebooks in `examples/`, written for the molab Notebook Competition #3 (OpenADMET × marimo).

- `examples/ecfp_openadmet.py` (`ecfp_openadmet_ja.py` in Japanese): **Do you really know your
  ECFP4?** ECFP4 across three OpenADMET datasets (PXR, ASAP-Polaris antiviral, ExpansionRx; 16
  endpoints): ECFP4's internals
  and two pitfalls (different molecules with the same fingerprint; near-identical molecules with a
  low Tanimoto; the same pair by other fingerprints, MCES and properties), then a dataset /
  endpoint picker that drives train–test distance, the similarity principle, cliffs, identical fingerprints, bit collisions, models on bit / count / descriptor
  features and TreeSHAP, and a last part that compares the 16 endpoints (substructures per bit,
  fingerprint length against test scores, neighbours, features, activity cliffs) and points
  back to the widgets. Reads `results/precomputed/` (DuckDB SQL cells) and computes what is
  missing
- `examples/ecfp_pxr.py` (`ecfp_pxr_ja.py` in Japanese): the earlier notebook on the PXR data
  alone: an 80 s ECFP4 video and a step-by-step version for any molecule, a dataset-wide bit
  atlas, how the PXR test set was built, where ECFP4 breaks, LightGBM importance and TreeSHAP on
  bits, and a model lab

```bash
uv sync
uv run marimo edit examples/ecfp_openadmet.py
```

The notebooks read the data from Hugging Face on first run (PXR, CC-BY-4.0; ASAP-Polaris, MIT;
ExpansionRx, CC-BY-4.0), or from `data/` when the CSV files are there.

## Widgets

Every whole-molecule drawing has a copy icon in its top-right corner that puts the molecule's
SMILES on the clipboard; `copy_smiles=False` hides them.

- `MolGrid`: paged, sortable molecule grid with text and SMARTS filtering, colour scale, and
  two-way selection (`single`, `multiple` or `pair` mode); `group_by` adds buttons to show one group of a
  column (e.g. train / test) and colours each card's edge by it. One search box with a mode: text filter,
  substructure (SMARTS), or similarity, which ranks the molecules by ECFP4 Tanimoto to a SMILES
  typed there (or to the first selected molecule)
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
  and says whether the Murcko scaffolds match (`chemari.similarity`); a note above it says
  that each method has its own scale. Given one compound only, it shows that compound with its
  values and properties (e.g. under a grid: one selected molecule, then a pair)
- `MolScatter`: a scatter plot of molecules (e.g. measured vs predicted) with a card of the
  same height beside it: hovering a point shows the molecule, clicking pins it (click it again,
  an empty spot or the card's "Pinned ×" chip to unpin). Under the drawing, `axis_fields` such
  as measured and predicted are labelled dots on one number line with their difference in a
  chip, followed by `fields`; a partner molecule given per row (e.g. its nearest neighbour) is
  listed by id; hovering the id pops up its drawing and clicking it copies its SMILES.
  `color_by` colours the points by a column (a viridis ramp for numbers, one colour per value
  otherwise, set with `color_map`), `mark_by` draws those flagged by a boolean one as
  triangles, `diagonal` draws y = x; `selected` syncs both ways
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
  that fold onto it and its purity (the share of its molecules that contain its most
  common substructure); sortable by bit index, number of substructures, number of molecules
  or purity, either way round. A panel on the right has the dataset's totals and a histogram
  of substructures per bit; clicking a bar lists only those bits
- `BitImportance`: every bit of a fingerprint model ranked by its importance (e.g. LightGBM gain
  or mean |SHAP|), in either direction so unused bits can be browsed too, with each bit's
  direction, its most common substructure and how many substructures and molecules it has;
  below, the selected bit's substructures and the molecules that set it, with the responsible
  atoms highlighted (click a substructure to keep only its molecules). `layout="side"` puts the
  detail beside the table instead (2 : 1, both `height` pixels tall, without the substructure
  column)
- `MorganExplorer`: bit-by-bit view of a Morgan fingerprint for one molecule or a pair, with the
  atom environments behind each bit, dataset statistics per bit, and a gallery of the different
  substructures that collide in a selected bit; for a pair, the table can show only the shared
  bits, only the differing ones, or those of one molecule; optionally per-bit model contributions
  (e.g. LightGBM TreeSHAP, pinned to the right of the table) and a per-atom attribution map;
  `stereo_labels=True` adds R/S and E/Z labels to the drawings. A bit / count switch
  (`mode`) and a chirality switch (`chirality`) change the fingerprint; for a pair, the
  summary gives the Tanimoto of the current setting (Σmin / Σmax for counts) and the table
  lists each molecule's count and the bits whose counts differ

## Layout

- `src/chemari/` — the package. `chem.py` (fingerprints, bit census), `ecfp.py` (a readable
  re-implementation of ECFP, and `ecfp_story`, which collects RDKit's identifiers for every atom
  and radius for `ECFPStepper`), `similarity.py` (the methods of MolPair's similarity view),
  `widgets.py` and `static/` (the widgets)
- `src/chemari/examples/openadmet.py` — the code behind `ecfp_openadmet`: datasets and endpoints
  (loading, standardisation, log10(x + 1) for ratio-scale endpoints), features (ECFP4 bit / count,
  RDKit descriptors), the LightGBM baseline, nearest neighbours, TreeSHAP, gain and mean |SHAP|
  per bit, and the bit model refitted at 1024–8192 bits. `dev/precompute.py` writes these to
  `results/precomputed/` (`--only-gain` / `--only-bitlen` rebuild one table from the stored
  molecules). A notebook installed from a release reads the tables of that release's tag
- `results/` — the precomputed tables, reference results computed outside the notebooks
  (LightGBM on CheMeleon fingerprints) and the analyses in `dev/analysis/`

## Development

```bash
uv run pytest
uv run ruff format . && uv run ruff check --fix .
uv run marimo check examples/*.py
```
