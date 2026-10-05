# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "chemari @ git+https://github.com/N283T/chemari",
#     "polars>=1.30",
#     "numpy>=2",
#     "rdkit>=2025.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="CheMari widgets")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CheMari widgets

    Chemistry widgets for [marimo](https://marimo.io) notebooks. Every widget of
    [CheMari](https://github.com/N283T/chemari) is shown below with the code that makes it.

    /// attention | This page is a snapshot

    The widgets on this page run without Python. What a widget does in the browser still works
    here: drawing, hovering, the paging, sorting and filtering of MolGrid, pinning a point of
    MolScatter, the row filters of MorganExplorer and MorganBitTiles, the movie.

    What a widget asks Python to compute does not work: the steps of ECFPStepper, another page
    or order of BitAtlas and BitImportance, another radius or fingerprint length, selecting a
    bit, aligning a pair or comparing it by other measures, a similarity search. Such a click
    shows "needs Python" and changes nothing.

    To try everything, open the notebook on molab:
    [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/github/github.com/N283T/chemari/blob/main/docs/widgets.py/server)
    ///

    ```bash
    pip install "chemari @ git+https://github.com/N283T/chemari"
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data for the examples

    The examples use the PXR pEC50 data of the OpenADMET PXR challenge (CC-BY-4.0), read from
    the tables of this repository: compounds with their measured value, the test predictions of
    a LightGBM model on ECFP4 bits, and that model's gain per bit.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from chemari import (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MolScatter,
        MorganBitTiles,
        MorganExplorer,
    )
    from chemari.examples import openadmet as bench

    tables = bench.open_tables(Path("results/precomputed"))
    mols = (
        pl.read_parquet(tables["molecules"])
        .filter(pl.col("task") == "pxr/pEC50")
        .select("id", "smiles", pl.col("y").round(2).alias("pEC50"), "split")
    )
    train = mols.filter(pl.col("split") == "train")
    test = mols.filter(pl.col("split") == "test")
    return (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MolScatter,
        MorganBitTiles,
        MorganExplorer,
        mo,
        mols,
        pl,
        tables,
        test,
        train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MolGrid

    A paged grid of molecules. It sorts, filters by text or SMARTS, ranks by similarity to a
    SMILES, and keeps a selection that Python can read (`grid.value["selection"]`).
    `selection_mode` is `"single"`, `"multiple"` or `"pair"`.
    """)
    return


@app.cell
def _(MolGrid, mo, mols):
    grid = mo.ui.anywidget(
        MolGrid(
            mols.sort("pEC50", descending=True),
            id_col="id",
            smiles_col="smiles",
            color_by="pEC50",
            group_by="split",
            page_size=12,
            selection_mode="pair",
        )
    )
    grid
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MolPair

    Two compounds side by side: the Tanimoto similarity (ECFP4), a table of properties and of
    any values you pass, the common substructure and what differs, and the pair by other
    fingerprints and similarity measures. Each compound is a row (a dict) or a SMILES.
    """)
    return


@app.cell
def _(MolPair, mo, train):
    _a, _b = train.sort("pEC50", descending=True).head(2).iter_rows(named=True)
    mo.ui.anywidget(MolPair(_a, _b, value_cols=["pEC50"], show_common=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MolScatter

    A scatter plot in which every point is a molecule. Hovering a point shows the molecule in
    the card on the right, clicking pins it. `color_by` colours the points by a column and
    `diagonal` draws y = x.
    """)
    return


@app.cell
def _(MolScatter, mo, pl, tables, test):
    _pred = (
        pl.read_parquet(tables["predictions"])
        .filter(
            (pl.col("task") == "pxr/pEC50")
            & (pl.col("features") == "ECFP4 bit")
            & (pl.col("split") == "test")
        )
        .select("id", pl.col("pred").round(2).alias("predicted"))
    )
    mo.ui.anywidget(
        MolScatter(
            test.join(_pred, on="id").rename({"pEC50": "measured"}),
            x="measured",
            y="predicted",
            x_label="measured pEC50",
            y_label="predicted pEC50 (LightGBM on ECFP4 bits)",
            diagonal=True,
            same_axes=True,
            axis_fields=["measured", "predicted"],
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ECFPMovie

    An 80-second animation of how ECFP4 is built: radius 0 to 2, duplicates, folding, collisions
    and Tanimoto, for N-methylacetamide.
    """)
    return


@app.cell
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ECFPStepper

    The steps of the movie for any molecule: atom by atom through every radius, then the folding
    into bits and the collisions. It plays by itself or is stepped by hand.
    """)
    return


@app.cell
def _(ECFPStepper, mo):
    mo.ui.anywidget(ECFPStepper("CC(=O)Oc1ccccc1C(=O)O"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MorganBitTiles

    The fingerprint of one molecule, one row per substructure. Substructures of the molecule
    that share a bit are marked. With a `reference` set, each row also counts the other
    substructures of that set on the same bit, and selecting a row lists them.
    """)
    return


@app.cell
def _(MorganBitTiles, mo, test, train):
    mo.ui.anywidget(MorganBitTiles(test["smiles"][0], reference=train["smiles"].to_list()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MorganExplorer

    The bits of one molecule or of a pair, with the atoms behind each bit. For a pair, the table
    shows all bits, the shared ones, the ones that differ, or those of one molecule. With a
    `reference` set and its values `y`, each bit also has the number of molecules that set it,
    the number of substructures on it, and the difference in mean `y` between molecules with
    and without it.
    """)
    return


@app.cell
def _(MorganExplorer, mo, test, train):
    _pair = [
        {"id": r["id"], "smiles": r["smiles"], "label": f"pEC50 {r['pEC50']}"}
        for r in test.head(2).iter_rows(named=True)
    ]
    mo.ui.anywidget(
        MorganExplorer(
            _pair,
            reference=train["smiles"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BitAtlas

    Every bit of a set of molecules, one row per bit, with the substructures folded onto it and
    its purity (the share of its molecules that carry its most common substructure). The panel
    on the right has the totals and a histogram of substructures per bit.
    """)
    return


@app.cell
def _(BitAtlas, mo, train):
    mo.ui.anywidget(BitAtlas(train["smiles"].to_list(), ids=train["id"].to_list()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## BitImportance

    The bits of a fingerprint model ranked by importance. `importance` is a dict of one array
    per measure, each with one value per bit. Selecting a bit shows its substructures and the
    molecules that set it.
    """)
    return


@app.cell
def _(BitImportance, mo, pl, tables, train):
    _gain = pl.read_parquet(tables["gain"]).filter(pl.col("task") == "pxr/pEC50").sort("bit")
    mo.ui.anywidget(
        BitImportance(
            train["smiles"].to_list(),
            importance={
                "gain": _gain["gain"].to_numpy(),
                "mean |SHAP|": _gain["shap_abs"].to_numpy(),
            },
            ids=train["id"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    return


if __name__ == "__main__":
    app.run()
