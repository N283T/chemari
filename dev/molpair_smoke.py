import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # MolPair smoke test

    Pick a pair (PXR, drugs, unrelated, stereo) or type two SMILES.
    """)
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import polars as pl

    from molwidgets import MolPair

    return MolPair, Path, pl


@app.cell
def _(Path, pl):
    _data = Path(__file__).parent.parent / "data"
    _csvs = [
        _data / f
        for f in ["pxr-challenge_TRAIN.csv", "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv"]
        if (_data / f).exists()
    ]
    EXAMPLES = {}
    if _csvs:
        _rows = {
            r["Molecule Name"]: r
            for f in _csvs
            for r in pl.read_csv(f).select("Molecule Name", "SMILES", "pEC50").iter_rows(named=True)
        }

        def _pxr(i):
            r = _rows[i]
            return {"id": i, "smiles": r["SMILES"], "pEC50": r["pEC50"]}

        EXAMPLES["PXR · one methyl, ~50-fold"] = (_pxr("OADMET-0001944"), _pxr("OADMET-0002007"))
        EXAMPLES["PXR · CH → N and Br moved"] = (_pxr("OADMET-0002810"), _pxr("OADMET-0006254"))
    EXAMPLES |= {
        "paracetamol → phenacetin": (
            {"id": "paracetamol", "smiles": "CC(=O)Nc1ccc(O)cc1"},
            {"id": "phenacetin", "smiles": "CCOc1ccc(NC(C)=O)cc1"},
        ),
        "sildenafil → vardenafil": (
            {
                "id": "sildenafil",
                "smiles": "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C",
            },
            {
                "id": "vardenafil",
                "smiles": "CCCC1=NC(=C2N1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)CC)OCC)C",
            },
        ),
        "ibuprofen vs atorvastatin (unrelated)": (
            {"id": "ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
            {
                "id": "atorvastatin",
                "smiles": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
            },
        ),
        "phenethylamine → pyridine (ring CH → N)": (
            {"id": "phenethylamine", "smiles": "NCCc1ccccc1"},
            {"id": "pyridyl-ethylamine", "smiles": "NCCc1ccncc1"},
        ),
        "(R)- vs (S)-thalidomide": (
            {"id": "(R)-thalidomide", "smiles": "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"},
            {"id": "(S)-thalidomide", "smiles": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"},
        ),
    }
    return (EXAMPLES,)


@app.cell
def _(EXAMPLES, mo):
    pick = mo.ui.dropdown(list(EXAMPLES), value=next(iter(EXAMPLES)), label="example pair")
    atoms = mo.ui.dropdown(["elements", "any"], value="elements", label="MCS atoms")
    mo.hstack([pick, atoms], justify="start")
    return atoms, pick


@app.cell
def _(EXAMPLES, mo, pick):
    _a, _b = EXAMPLES[pick.value]
    smi_a = mo.ui.text(_a["smiles"], label="A", full_width=True)
    smi_b = mo.ui.text(_b["smiles"], label="B", full_width=True)
    mo.vstack([smi_a, smi_b])
    return smi_a, smi_b


@app.cell
def _(EXAMPLES, pick, smi_a, smi_b):
    _a, _b = EXAMPLES[pick.value]
    a = {**_a, "smiles": smi_a.value}
    b = {**_b, "smiles": smi_b.value}
    value_cols = ["pEC50"] if "pEC50" in a else []
    return a, b, value_cols


@app.cell
def _(MolPair, a, atoms, b, mo, value_cols):
    mo.vstack(
        [
            mo.ui.anywidget(
                MolPair(
                    a,
                    b,
                    value_cols=value_cols,
                    value_ranges={"pEC50": (1.5, 8)},
                    show_common=True,
                    view="common",
                    mcs={"atoms": atoms.value},
                )
            ),
            mo.md(
                "Without captions or copy icons (`show_formula=False, show_smiles=False, copy_smiles=False`):"
            ),
            mo.ui.anywidget(
                MolPair(
                    a,
                    b,
                    value_cols=value_cols,
                    show_formula=False,
                    show_smiles=False,
                    copy_smiles=False,
                )
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
