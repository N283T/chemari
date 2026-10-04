import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    from chemari import MolGrid, MorganExplorer, standardize_smiles

    df = (
        pl.read_csv("data/pxr-challenge_TRAIN.csv")
        .select(pl.col("Molecule Name").alias("id"), pl.col("SMILES").alias("smiles"), "pEC50")
        .head(200)
    )
    df = df.with_columns(pl.col("smiles").map_elements(standardize_smiles, return_dtype=pl.Utf8))
    return MolGrid, MorganExplorer, df, mo


@app.cell
def _(MolGrid, df, mo):
    grid = mo.ui.anywidget(MolGrid(df, color_by="pEC50", selection_mode="pair", page_size=12))
    grid
    return (grid,)


@app.cell
def _(grid):
    grid.value
    return


@app.cell
def _(MorganExplorer, df, grid, mo):
    sel = grid.value.get("selection", [])
    lookup = dict(zip(df["id"], df["smiles"]))
    mols = [{"id": i, "smiles": lookup[i]} for i in sel] or [
        {"id": df["id"][0], "smiles": df["smiles"][0]}
    ]
    ex = mo.ui.anywidget(
        MorganExplorer(
            mols, reference=df["smiles"].to_list(), y=df["pEC50"].to_numpy(), y_label="pEC50"
        )
    )
    ex
    return (ex,)


@app.cell
def _(ex):
    ex.value.get("selected_bit"), ex.value.get("radius"), len(ex.value.get("bit_examples", []))
    return


if __name__ == "__main__":
    app.run()
