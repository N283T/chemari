import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # MolScatter smoke test

    Measured vs predicted (LightGBM on ECFP4 bits) for the test set of one task, coloured by
    Tanimoto to the nearest train compound; activity cliffs are triangles. Hover a point, click
    to pin it.
    """)
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import polars as pl

    from molwidgets import MolScatter, bench

    return MolScatter, Path, bench, np, pl


@app.cell
def _(Path, bench, pl):
    _where = bench.open_tables(Path(__file__).parent.parent / "results" / "precomputed")
    molecules = pl.read_parquet(_where["molecules"])
    neighbours = pl.read_parquet(_where["neighbours"])
    predictions = pl.read_parquet(_where["predictions"])
    return molecules, neighbours, predictions


@app.cell
def _(bench, mo):
    task_pick = mo.ui.dropdown({t.key: t.key for t in bench.TASKS}, value="pxr/pEC50", label="task")
    height = mo.ui.slider(320, 640, step=20, value=520, label="height", show_value=True)
    mo.hstack([task_pick, height], justify="start")
    return height, task_pick


@app.cell
def _(bench, molecules, neighbours, np, pl, predictions, task_pick):
    task = bench.TASK[task_pick.value]
    _smi = dict(
        zip(*molecules.filter(pl.col("task") == task.key).select("id", "smiles").get_columns())
    )
    _nn = neighbours.filter(pl.col("task") == task.key).with_columns(
        (pl.col("y_test") - pl.col("y_nn")).abs().alias("dy")
    )
    # the notebook's cliff rule: Tanimoto to the NN >= 0.6 and |Δ| >= mean |Δ| of random pairs
    _random = float(
        np.abs(
            np.random.default_rng(0).permutation(_nn["y_nn"].to_numpy()) - _nn["y_test"].to_numpy()
        ).mean()
    )
    _cliffs = _nn.filter((pl.col("tanimoto") >= 0.6) & (pl.col("dy") >= _random))["test_id"]
    rows = (
        predictions.filter(
            (pl.col("task") == task.key)
            & (pl.col("split") == "test")
            & (pl.col("features") == "ECFP4 bit")
        )
        .select("id", "y", "pred")
        .join(_nn.select(pl.col("test_id").alias("id"), "nn_id", "tanimoto"), on="id")
        .with_columns(
            pl.col("id").replace_strict(_smi).alias("smiles"),
            pl.col("nn_id").replace_strict(_smi).alias("nn_smiles"),
            pl.col("id").is_in(_cliffs.implode()).alias("cliff"),
            pl.when(pl.col("tanimoto") >= 0.6)
            .then(pl.lit("≥ 0.6"))
            .otherwise(pl.lit("< 0.6"))
            .alias("NN"),
        )
        .sort("tanimoto")
        .select(
            "id",
            "smiles",
            pl.col("y").alias("measured"),
            pl.col("pred").alias("predicted"),
            pl.col("tanimoto").alias("Tanimoto to the NN"),
            "cliff",
            "NN",
            "nn_id",
            "nn_smiles",
        )
    )
    return rows, task


@app.cell
def _(MolScatter, height, mo, rows, task):
    scatter = mo.ui.anywidget(
        MolScatter(
            rows,
            x="measured",
            y="predicted",
            x_label=f"measured {task.label}",
            y_label="predicted (ECFP4 bit)",
            color_by="NN",
            color_label="Tanimoto to the NN",
            color_map={"≥ 0.6": "#1c7ed6", "< 0.6": "#b8c2cc"},
            mark_by="cliff",
            mark_label="activity cliff",
            diagonal=True,
            same_axes=True,
            card_title="test compound",
            info_title="prediction (ECFP4 bit)",
            height=height.value,
            axis_fields=["measured", "predicted"],
            partner_id_col="nn_id",
            partner_smiles_col="nn_smiles",
            partner_label="NN (train)",
            partner_fields=["Tanimoto to the NN"],
        )
    )
    scatter
    return (scatter,)


@app.cell
def _(mo, scatter):
    mo.md(f"`selected` = `{scatter.value.get('selected', '')!r}`")
    return


if __name__ == "__main__":
    app.run()
