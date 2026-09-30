# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "molwidgets @ git+https://github.com/N283T/openadmet-marimo",
#     "polars>=1.30",
#     "numpy>=2",
#     "altair>=5.5",
#     "lightgbm>=4.5",
#     "scikit-learn>=1.7",  # required by lightgbm's sklearn API
#     "scipy>=1.14",
#     "rdkit>=2025.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Do you really know your ECFP4?")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Do you really know your ECFP4?
    ### What it encodes and where it fails, tested on PXR data

    When you feed molecules to a machine-learning model, you have probably written something like this:

    ```python
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    ```

    That is **ECFP4**, the most widely used molecular representation in cheminformatics. It is quick to compute with almost nothing to tune, and it works well for a lot of similarity search and QSAR. When a new method is benchmarked, ECFP4 is usually the first thing it is compared against.

    Few people can say what those 2048 bits record about a molecule and what they leave out, though. If you don't know where ECFP4 is weak, you won't know why it fails when you hit a dataset where it does.

    /// admonition | A note on names
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) and RDKit's **Morgan fingerprint** are the same thing. The number in ECFP*n* is the *diameter* of the neighbourhood around each atom. RDKit asks for the radius instead, so ECFP4 is `radius=2`.
    ///

    The notebook has two parts.

    **Part 1 · How it works**: how ECFP4 turns a molecule into bits.

    * What ECFP4 computes
    * 1 · A closer look at ECFP4

    **Part 2 · Where it fails**: a dataset picked because ECFP4 does poorly on it (OpenADMET's PXR induction data), to see where and why it breaks.

    * 2 · A dataset where ECFP4 struggles: PXR
    * 3 · Where ECFP4 breaks: the similarity principle, activity cliffs, identical fingerprints, whole-molecule properties
    * 4 · Inside the model: what a LightGBM trained on ECFP4 learned
    * 5 · Model lab: counts, fold size, chirality and descriptors
    * 6 · Take-aways

    ---

    ## Part 1 · How ECFP4 works

    Start with the 80-second video on what ECFP4 computes and how it behaves.
    """)
    return


@app.cell(hide_code=True)
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In short, ECFP4 has these properties:

    * the fingerprint only records which substructures are present
    * how often a substructure occurs, where it sits and what the molecule is like as a whole are all lost
    * folding pushes unrelated substructures onto the same bit
    * Tanimoto similarity counts shared bits, collisions included

    /// admonition | Substructures and environments
    An ECFP4 substructure is circular: one atom plus every atom and bond within the radius around it. It is also called an atom environment, which is what "environment" and `# envs` mean in the widgets.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The widget below runs the same steps on any molecule. Type a SMILES or pick an example, then step through with **next** or press **play**.
    """)
    return


@app.cell(hide_code=True)
def _(ECFPStepper, mo):
    ecfp_stepper = mo.ui.anywidget(ECFPStepper())
    ecfp_stepper
    return (ecfp_stepper,)


@app.cell
def _():
    import altair as alt
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
    from rdkit.Chem import rdMolDescriptors as rdmd
    from scipy.stats import spearmanr

    alt.data_transformers.disable_max_rows()  # a few charts plot all ~4.6k compounds

    from molwidgets import (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MorganBitTiles,
        MorganExplorer,
        census_for,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )

    return (
        BitAtlas,
        BitImportance,
        Chem,
        Crippen,
        Descriptors,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MorganBitTiles,
        MorganExplorer,
        alt,
        census_for,
        fingerprint_matrix,
        lgb,
        np,
        pl,
        rdFingerprintGenerator,
        rdmd,
        spearmanr,
        standardize_smiles,
        tanimoto_matrix,
    )


@app.cell
def _(pl):
    from pathlib import Path

    HF = "https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main/"
    FILES = {
        "train": "pxr-challenge_TRAIN.csv",
        "test_p1": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "test_p2": "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
    }
    CI_LO = "pEC50_ci.lower (-log10(molarity))"
    CI_HI = "pEC50_ci.upper (-log10(molarity))"

    def _read(name: str) -> pl.DataFrame:
        local = Path("data") / FILES[name]  # use a local copy when present, else download
        return pl.read_csv(local if local.exists() else HF + FILES[name])

    raw = pl.concat(
        [
            _read(key)
            .select(
                pl.col("Molecule Name").alias("id"),
                pl.col("SMILES").alias("smiles_raw"),
                "pEC50",
                (pl.col(CI_HI) - pl.col(CI_LO)).alias("ci_width"),
            )
            .with_columns(pl.lit("train" if key == "train" else "test").alias("split"))
            for key in FILES
        ]
    )
    return (raw,)


@app.cell
def _(Chem, pl, raw, standardize_smiles):
    # largest fragment, neutralized, canonical; stereo kept
    data = raw.with_columns(
        pl.col("smiles_raw").map_elements(standardize_smiles, return_dtype=pl.Utf8).alias("smiles")
    ).with_row_index("row")

    # "changed" = the standardized parent differs from the input as a structure, not just in notation
    _canon_raw = raw["smiles_raw"].map_elements(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)), return_dtype=pl.Utf8
    )
    hygiene = {
        "invalid": data["smiles"].null_count(),
        "changed": int((_canon_raw != data["smiles"]).sum()),
        "dup_within": data.height - data["smiles"].n_unique(),
        "overlap": int(
            data.filter(pl.col("split") == "test")["smiles"]
            .is_in(data.filter(pl.col("split") == "train")["smiles"].implode())
            .sum()
        ),
    }
    train = data.filter(pl.col("split") == "train")
    test = data.filter(pl.col("split") == "test")
    return data, hygiene, test, train


@app.cell(hide_code=True)
def _(Chem, census_for, mo, np, rdFingerprintGenerator, train):
    _envs = census_for(train["smiles"].to_list(), 2, 2048).n_envs
    # distinct radius-0..2 environments per molecule, before folding
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    _per_mol = np.median(
        [
            len(_gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements())
            for s in train["smiles"]
        ]
    )
    mo.md(
        f"""
    ## 1 · A closer look at ECFP4

    Here we look at ECFP4 on molecules from the PXR dataset used in Part 2.

    A compound in this dataset has **{_per_mol:.0f} distinct substructures** (median). Across all {train.height:,} train compounds there are **{int(_envs.sum()):,}**, and folding them into 2048 bits puts **{_envs[_envs > 0].mean():.0f}** into each bit on average. No bit holds just one.

    Pick a compound in the grid to see the molecule (left) and its bits (right) below.

    * **hover a bit**: highlights where in the molecule it comes from
    * **red**: the bit is shared with another substructure of the same molecule (a collision)
    * **collisions only**: show only the colliding bits
    * **grey badge**: how many other substructures in the dataset share the bit
    * **click a row**: list those substructures
    * **radius / fold to**: rebuild the bits
    """
    )
    return


@app.cell
def _(MolGrid, data, mo, pl):
    # the benzene → pyridine pair from section 4 first, then everything by potency
    _pair = ["OADMET-0006254", "OADMET-0002810"]
    _rows = data.select("id", "smiles", "split", "pEC50")
    grid = mo.ui.anywidget(
        MolGrid(
            pl.concat(
                [
                    _rows.filter(pl.col("id").is_in(_pair)).sort("pEC50"),
                    _rows.filter(~pl.col("id").is_in(_pair)).sort("pEC50", descending=True),
                ]
            ),
            subset=["split", "pEC50"],
            color_by="pEC50",
            selection_mode="single",
            selection=["OADMET-0006254"],
            page_size=12,
            cell_size=150,
        )
    )
    grid
    return (grid,)


@app.cell
def _(MorganBitTiles, data, grid, mo, pl, train):
    _sel = grid.value.get("selection") or ["OADMET-0006254"]
    _row = data.filter(pl.col("id") == _sel[0]).row(0, named=True)
    mo.ui.anywidget(
        MorganBitTiles(
            _row["smiles"],
            reference=train["smiles"].to_list(),
            ids=train["id"].to_list(),
            label=f"{_row['id']} · {_row['split']} · pEC50 {_row['pEC50']:.2f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to notice:

    * **OADMET-0006254 collides with itself**
      * at 2048 bits the quaternary carbon `C(C)(S)(C)C` and the aromatic `c(N)(c)c` both land on bit 381
      * press collisions only to see it; at 8192 bits the collision is gone
    * **radius-0 bits are set by many molecules**
      * a single atom turns up in all kinds of molecules
      * bit 80 (`[C;D2;H2]`, a CH₂) is set in more than 60% of the train compounds
      * in a carboxylic acid the carbonyl carbon `[C;D3;H0]` and the hydroxyl oxygen `[O;D1;H1]` fall on the same bit, 807
    * **a substructure that occurs several times gets one row**
      * three methyls or four aromatic CH are shown once, as ×3 or ×4
      * the bit vector keeps no counts

    An ECFP4 bit records only whether a substructure is present, and one bit is often shared by unrelated substructures.
    """)
    return


@app.cell(hide_code=True)
def _(BitAtlas, mo, train):
    mo.vstack(
        [
            mo.md("Across the whole train set, each bit holds substructures like these."),
            mo.ui.anywidget(BitAtlas(train["smiles"].to_list(), ids=train["id"].to_list())),
        ]
    )
    return


@app.cell(hide_code=True)
def _(hygiene, mo, test, train):
    mo.vstack(
        [
            mo.md(r"""
    ---

    ## Part 2 · Where ECFP4 fails

    ## 2 · A dataset where ECFP4 struggles: PXR

    The **pregnane X receptor (PXR)** is a nuclear receptor that senses foreign molecules and turns up CYP3A4, P-gp and other clearance genes. A drug that activates PXR can speed up the metabolism of other drugs, so PXR induction is a common drug–drug interaction (DDI) risk. OpenADMET measured it for more than 11,000 compounds and ran a [blind challenge](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/) [on Hugging Face](https://huggingface.co/spaces/openadmet/pxr-challenge) to predict **pEC50** for 513 new ones.

    In that challenge, models built on ECFP4 kept ending up near the bottom. In my own entry (4th of 95), a LightGBM on ECFP4 alone reached a CV MAE of about 0.57, while ensembles of descriptors and embeddings got below 0.40. Other teams saw the same thing.

    The data are [OpenADMET's PXR release on Hugging Face](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY-4.0). We use the **train** set and the full **test** set, whose labels were published after the challenge. Every SMILES is standardized the same way: largest fragment, neutralized, stereochemistry kept, canonical SMILES.
    """),
            mo.hstack(
                [
                    mo.stat(f"{train.height:,}", label="train"),
                    mo.stat(f"{test.height:,}", label="test"),
                    mo.stat(str(hygiene["changed"]), label="neutralized"),
                    mo.stat(str(hygiene["dup_within"]), label="duplicates after standardization"),
                    mo.stat(str(hygiene["overlap"]), label="train–test overlap"),
                ],
                widths="equal",
                gap=0.5,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 513 test compounds are not a random sample. OpenADMET took the **63** hits from its screen that were potent (EC50 ≤ 1 µM) and inactive in cells lacking PXR, and bought Enamine analogues with an **ECFP4 Tanimoto similarity above 0.4** to them. This is plain hit expansion, the usual way to follow up a screen.

    Two things follow from how the set was built:

    * every test compound has a similar compound in train
    * that similar compound is usually potent

    For each compound we find its **nearest neighbour (NN)** in train by ECFP4 Tanimoto (2048 bits; train compounds skip themselves).
    """)
    return


@app.cell
def _(fingerprint_matrix, np, test, train, tanimoto_matrix):
    X_train = fingerprint_matrix(train["smiles"].to_list(), radius=2, n_bits=2048)
    X_test = fingerprint_matrix(test["smiles"].to_list(), radius=2, n_bits=2048)

    S_train = tanimoto_matrix(X_train)
    np.fill_diagonal(S_train, 0.0)  # leave-one-out: a compound is not its own neighbour
    S_test = tanimoto_matrix(X_test, X_train)

    y_train = train["pEC50"].to_numpy()
    y_test = test["pEC50"].to_numpy()
    nn_test = S_test.argmax(1)
    return S_test, S_train, X_train, nn_test, y_test, y_train


@app.cell(hide_code=True)
def _(S_test, S_train, alt, mo, nn_test, np, pl, test, y_test, y_train):
    _dens = pl.concat(
        [
            pl.DataFrame({"NN Tanimoto": S_train.max(1), "set": "train → rest of train"}),
            pl.DataFrame({"NN Tanimoto": S_test.max(1), "set": "test → train"}),
        ]
    )
    _density = (
        alt.Chart(_dens)
        .transform_density(
            "NN Tanimoto", groupby=["set"], as_=["NN Tanimoto", "density"], extent=[0, 1]
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("NN Tanimoto:Q", title="Tanimoto to nearest training neighbour"),
            y=alt.Y("density:Q", stack=None),
            color=alt.Color(
                "set:N",
                scale=alt.Scale(
                    range=["#3b82f6", "#f59e0b"],
                    domain=["train → rest of train", "test → train"],
                ),
                legend=alt.Legend(orient="top-right", title=None, fillColor="white", padding=4),
            ),
        )
        .properties(height=220, width=300)
    )
    _pts = pl.DataFrame(
        {
            "id": test["id"],
            "NN Tanimoto": S_test.max(1),
            "test pEC50": y_test,
            "NN pEC50": y_train[nn_test],
        }
    )
    _scatter = (
        alt.Chart(_pts)
        .mark_circle(size=24, opacity=0.6)
        .encode(
            x=alt.X("NN Tanimoto:Q", scale=alt.Scale(domain=[0.2, 1])),
            y=alt.Y("test pEC50:Q", scale=alt.Scale(domain=[1.5, 7.5])),
            color=alt.Color(
                "NN pEC50:Q",
                scale=alt.Scale(range=["#2563eb", "#dc2626"], interpolate="hcl"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=["id", alt.Tooltip("NN Tanimoto:Q", format=".2f"), "test pEC50", "NN pEC50"],
        )
        .properties(height=220, width=300)
    )
    _nn = y_train[nn_test]
    mo.vstack(
        [
            mo.hstack([_density, _scatter], justify="start", gap=2),
            mo.md(
                f"""
    **Left: similarity to the nearest neighbour**

    * the NN similarity from test to train (median **{np.median(S_test.max(1)):.2f}**) is higher than within train (**{np.median(S_train.max(1)):.2f}**)
    * the test set sits inside the applicability domain of train

    **Right: test pEC50 against the nearest neighbour's pEC50**

    * the neighbours are mostly potent (mean pEC50 **{_nn.mean():.2f}**, train overall {y_train.mean():.2f}, **{(_nn >= 5.5).mean():.0%}** at 5.5 or above)
    * the test compounds themselves still range from {y_test.min():.1f} to {y_test.max():.1f}
    * this is SAR exploration around the hits, and ECFP4 Tanimoto cannot tell you whether a close analogue keeps the activity
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(1, 50, value=1, step=1, label="neighbours k", show_value=True)
    mo.md(
        f"""
    The simplest fingerprint model there is: predict each test compound's pEC50 as the mean of its **k most similar** train compounds. {k_slider}
    """
    )
    return (k_slider,)


@app.cell(hide_code=True)
def _(S_test, alt, k_slider, mo, np, pl, spearmanr, test, y_test, y_train):
    _k = k_slider.value
    _idx = np.argsort(-S_test, axis=1)[:, :_k]
    _pred = y_train[_idx].mean(1)
    _rand = np.abs(y_test[:, None] - y_train[None, :]).mean()
    _df = pl.DataFrame(
        {
            "true pEC50": y_test,
            "kNN prediction": _pred,
            "NN similarity": S_test.max(1),
            "id": test["id"],
        }
    )
    _lim = [1.5, 7.5]
    _pts = (
        alt.Chart(_df)
        .mark_circle(size=28, opacity=0.6)
        .encode(
            x=alt.X("kNN prediction:Q", scale=alt.Scale(domain=_lim)),
            y=alt.Y("true pEC50:Q", scale=alt.Scale(domain=_lim)),
            color=alt.Color(
                "NN similarity:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                "id",
                alt.Tooltip("true pEC50:Q", format=".2f"),
                alt.Tooltip("kNN prediction:Q", format=".2f"),
                alt.Tooltip("NN similarity:Q", format=".2f"),
            ],
        )
    )
    _diag = (
        alt.Chart(pl.DataFrame({"x": _lim, "y": _lim}))
        .mark_line(color="#9ca3af", strokeDash=[4, 4])
        .encode(x="x", y="y")
    )
    _rho = spearmanr(_pred, y_test)[0]
    _mae = np.abs(_pred - y_test).mean()
    mo.vstack(
        [
            mo.hstack(
                [
                    (_diag + _pts).properties(width=330, height=300),
                    mo.vstack(
                        [
                            mo.stat(f"{_mae:.2f}", label=f"MAE ({_k}-NN)"),
                            mo.stat(f"{_rho:.2f}", label="Spearman ρ"),
                            mo.stat(
                                f"{_rand:.2f}",
                                label="MAE when predicting with a random train compound",
                            ),
                        ]
                    ),
                ],
                widths=[1.1, 1],
                align="center",
            ),
            mo.md("""
    * with k = 1, ρ ≈ 0: almost no ranking power
    * the MAE is about the same as predicting with a random train compound
    * a larger k lowers the MAE, but only because the predictions drift toward the mean (the points collapse into a vertical band)

    The test set was built from ECFP4 neighbours of train, yet the nearest neighbour's activity says almost nothing about a test compound.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Where ECFP4 breaks

    ### 3a · Measuring the similarity principle

    The similar property principle says that the more alike two molecules are, the smaller their difference in activity should be. We bin all ~8.5 million train pairs by Tanimoto similarity and look at |Δ pEC50| in each bin.
    """)
    return


@app.cell
def _(S_train, np, y_train):
    _iu = np.triu_indices(len(y_train), k=1)
    _sim = S_train[_iu]
    _dy = np.abs(y_train[_iu[0]] - y_train[_iu[1]])
    _edges = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    _which = np.clip(np.digitize(_sim, _edges) - 1, 0, len(_edges) - 2)
    similarity_curve = [
        {
            "sim_lo": float(_edges[b]),
            "sim_mid": float(_edges[b] + 0.025),
            "pairs": int((_which == b).sum()),
            "mean_dy": float(_dy[_which == b].mean()),
            "q90_dy": float(np.quantile(_dy[_which == b], 0.9)),
            "frac_gt1": float((_dy[_which == b] > 1).mean()),
        }
        for b in range(len(_edges) - 1)
        if (_which == b).sum() >= 5
    ]
    random_pair_dy = float(_dy.mean())

    # Keep only reasonably similar pairs for the cliff browser below.
    _keep = _sim >= 0.4
    pair_i, pair_j = _iu[0][_keep], _iu[1][_keep]
    pair_sim, pair_dy = _sim[_keep], _dy[_keep]
    return pair_dy, pair_i, pair_j, pair_sim, random_pair_dy, similarity_curve


@app.cell(hide_code=True)
def _(alt, mo, pair_dy, pair_sim, pl, random_pair_dy, similarity_curve):
    _df = pl.DataFrame(similarity_curve)
    _base = alt.Chart(_df).encode(
        x=alt.X(
            "sim_mid:Q", title="Tanimoto similarity of the pair", scale=alt.Scale(domain=[0, 1])
        )
    )
    _band = (
        _base.transform_calculate(zero="0")
        .mark_area(opacity=0.18, color="#d6336c")
        .encode(y=alt.Y("q90_dy:Q"), y2="zero:Q")
    )
    _line = _base.mark_line(point=True, color="#d6336c").encode(
        y=alt.Y("mean_dy:Q", title="|Δ pEC50| (mean, shaded to 90th pct.)"),
        tooltip=[
            alt.Tooltip("sim_lo:Q", title="bin from", format=".2f"),
            alt.Tooltip("pairs:Q", format=","),
            alt.Tooltip("mean_dy:Q", format=".2f"),
            alt.Tooltip("frac_gt1:Q", title="share with |Δ| > 1", format=".0%"),
        ],
    )
    _rule = (
        alt.Chart(pl.DataFrame({"y": [random_pair_dy]}))
        .mark_rule(strokeDash=[5, 4], color="#6b7280")
        .encode(y="y:Q")
    )
    _hi = pair_sim >= 0.5
    mo.vstack(
        [
            mo.hstack(
                [
                    (_band + _line + _rule).properties(height=260, width=360),
                    mo.vstack(
                        [
                            mo.stat(f"{random_pair_dy:.2f}", label="|Δ pEC50|, random pairs"),
                            mo.stat(
                                f"{pair_dy[_hi].mean():.2f}", label="|Δ pEC50|, Tanimoto ≥ 0.5"
                            ),
                            mo.stat(
                                f"{(pair_dy[_hi] > 1).mean():.0%}",
                                label="pairs ≥ 0.5 that differ more than 10-fold",
                            ),
                        ]
                    ),
                ],
                widths=[1.1, 1],
                align="center",
            ),
            mo.md(f"""
    **On PXR, looking alike says little about being equally active.** Pairs with Tanimoto ≥ 0.5 still differ by {pair_dy[_hi].mean() / random_pair_dy:.0%} of what random pairs do, and {(pair_dy[_hi] > 1).mean():.0%} of them differ more than 10-fold in potency.

    * line: mean |Δ pEC50| per bin; shading: up to the 90th percentile; dashed line: random pairs
    * hover a point for the number of pairs in the bin

    The premise of an ECFP4 model, that similar molecules have similar activity, holds only loosely for PXR.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(MolPair, mo, train):
    # a clean cliff: both compounds well measured (95% CI < 1 log unit), one methyl apart
    cliff_example = ["OADMET-0001944", "OADMET-0002007"]
    _rows = [train.filter(train["id"] == i).row(0, named=True) for i in cliff_example]
    _dy = abs(_rows[1]["pEC50"] - _rows[0]["pEC50"])
    mo.vstack(
        [
            mo.md(f"""
    ### 3b · Browsing activity cliffs

    An **activity cliff** is a pair of compounds that look alike but differ a lot in activity (here pEC50). The two below differ only by one methyl on the benzene ring (the common part is blue), yet their EC50 values are about {10**_dy:.0f}-fold apart.
    """),
            mo.ui.anywidget(
                MolPair(
                    _rows[0],
                    _rows[1],
                    value_cols=["pEC50"],
                    # the dumbbell's axis: every train compound's pEC50
                    value_ranges={"pEC50": (train["pEC50"].min(), train["pEC50"].max())},
                    properties=["cLogP"],
                    show_common=True,
                )
            ),
        ]
    )
    return (cliff_example,)


@app.cell(hide_code=True)
def _(mo, np):
    def _curve_panel(title, ec50, color, note):
        # x: log concentration mapped to pixels; tested range is 40..230
        w, h, x0, x1, top, bot = 350, 190, 40, 230, 30, 160
        xs = np.linspace(20, 340, 140)
        ys = bot - (bot - top) / (1 + np.exp(-(xs - ec50) / 18))
        inside = (xs >= x0) & (xs <= x1)
        solid = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[inside], ys[inside]))
        dashed = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[xs >= x1], ys[xs >= x1]))
        pts = np.linspace(x0 + 10, x1 - 5, 7)
        dots = "".join(
            f"<circle cx='{x:.1f}' cy='{bot - (bot - top) / (1 + np.exp(-(x - ec50) / 18)) + d:.1f}' r='3.2' fill='{color}'/>"
            for x, d in zip(pts, [2, -3, 3, -2, 2, -3, 1])
        )
        mid = (top + bot) / 2
        known = ec50 <= x1
        marker = (
            f"<line x1='{ec50}' y1='{mid}' x2='{ec50}' y2='{bot}' stroke='{color}' stroke-dasharray='3 3'/>"
            f"<circle cx='{ec50}' cy='{mid}' r='4.5' fill='none' stroke='{color}' stroke-width='2'/>"
            f"<text x='{ec50}' y='{bot + 14}' text-anchor='middle' font-size='11' fill='{color}'>EC50</text>"
            if known
            else f"<line x1='{ec50 - 35}' y1='{mid}' x2='{ec50 + 35}' y2='{mid}' stroke='{color}' stroke-width='2'/>"
            f"<line x1='{ec50 - 35}' y1='{mid - 6}' x2='{ec50 - 35}' y2='{mid + 6}' stroke='{color}' stroke-width='2'/>"
            f"<line x1='{ec50 + 35}' y1='{mid - 6}' x2='{ec50 + 35}' y2='{mid + 6}' stroke='{color}' stroke-width='2'/>"
            f"<text x='{ec50 + 4}' y='{mid + 22}' font-size='11' fill='{color}'>EC50 ?</text>"
        )
        return f"""
    <svg viewBox='0 0 {w} {h + 22}' width='{w}' style='max-width:100%;font-family:system-ui,sans-serif'>
      <text x='{w / 2}' y='14' text-anchor='middle' font-size='13' font-weight='600' fill='currentColor'>{title}</text>
      <rect x='{x0}' y='{top - 8}' width='{x1 - x0}' height='{bot - top + 8}' fill='#9ca3af' opacity='0.15'/>
      <text x='{(x0 + x1) / 2}' y='{top + 4}' text-anchor='middle' font-size='10' fill='#6b7280'>tested range</text>
      <line x1='20' y1='{bot}' x2='{w - 8}' y2='{bot}' stroke='currentColor' opacity='0.6'/>
      <line x1='20' y1='{top - 10}' x2='20' y2='{bot}' stroke='currentColor' opacity='0.6'/>
      <text x='{w - 8}' y='{bot + 28}' text-anchor='end' font-size='10' fill='#6b7280'>log concentration →</text>
      <text x='14' y='{top - 14}' font-size='10' fill='#6b7280'>response</text>
      <polyline points='{solid}' fill='none' stroke='{color}' stroke-width='2.2'/>
      <polyline points='{dashed}' fill='none' stroke='{color}' stroke-width='2' stroke-dasharray='5 4' opacity='0.8'/>
      {dots}{marker}
      <text x='{w / 2}' y='{h + 20}' text-anchor='middle' font-size='11' fill='#6b7280'>{note}</text>
    </svg>"""

    mo.vstack(
        [
            mo.md(r"""
    First, a word on how pEC50 is measured. It comes from fitting an S-shaped dose–response curve to the responses at several concentrations. A potent compound reaches a plateau inside the tested range, so the midpoint of the curve (EC50) is well defined. A weak compound is still rising at the highest concentration, so its EC50 is an extrapolation.
    """),
            mo.hstack(
                [
                    mo.Html(
                        _curve_panel(
                            "strong compound",
                            120,
                            "#1c7ed6",
                            "plateau inside the range → EC50 is pinned down",
                        )
                    ),
                    mo.Html(
                        _curve_panel(
                            "weak compound", 275, "#d6336c", "no plateau → EC50 is extrapolated"
                        )
                    ),
                ],
                justify="center",
                gap=2,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, pl, train):
    min_sim = mo.ui.slider(0.4, 0.9, value=0.55, step=0.05, label="min Tanimoto", show_value=True)
    min_dy = mo.ui.slider(0.5, 3.0, value=1.5, step=0.25, label="min |Δ pEC50|", show_value=True)
    _bins = (
        train.with_columns(
            pl.col("pEC50")
            .cut([3, 4, 5, 6], labels=["< 3", "3–4", "4–5", "5–6", "≥ 6"])
            .alias("pEC50 bin")
        )
        .group_by("pEC50 bin")
        .agg(pl.len().alias("n"), pl.col("ci_width").median().alias("median 95% CI width"))
        .sort("pEC50 bin")
    )
    _low = _bins.filter(pl.col("pEC50 bin") == "< 3")
    _high = _bins.filter(pl.col("pEC50 bin") == "≥ 6")
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{_low['median 95% CI width'].item():.1f}",
                        label="CI width, pEC50 < 3 (median)",
                    ),
                    mo.stat(
                        f"{_high['median 95% CI width'].item():.1f}",
                        label="CI width, pEC50 ≥ 6 (median)",
                    ),
                    mo.stat(
                        f"{_low['n'].item()} ({_low['n'].item() / train.height:.0%})",
                        label="train compounds with pEC50 < 3",
                    ),
                ],
                widths="equal",
                gap=0.5,
            ),
            mo.md(r"""
    The pEC50 of a weak compound is a rough estimate. If one side of a cliff sits in that range, part of the difference is measurement uncertainty rather than structure.

    Set what "similar" and "different" mean with the sliders, pick a pair from the table, and compare the two fingerprints bit by bit below. The only bits a fingerprint model can use to explain the difference are the ones present in just one molecule, and they are usually a few common substructures.

    The table is sorted by **SALI** (structure–activity landscape index, Guha & Van Drie 2008): SALI = |Δ pEC50| / (1 − Tanimoto). It grows when the activity gap is large and the structural difference small, so the steepest cliffs come first.

    * **only A / only B**: show the bits present in one molecule only
    * **# mols / # envs**: molecules that set the bit / distinct substructures in it
    * **Δ pEC50**: mean pEC50 of molecules with the bit minus those without

    /// details | How Δ pEC50 is calculated
    The train compounds (4,139) are split by whether the bit is set in their fingerprint, at the radius and bit count chosen in the widget, and Δ is the difference between the two mean pEC50 values. No model is involved.

    * bits tend to be set together, for example across a scaffold series, so Δ is not the effect of that bit alone
    * a small group gives extreme values (a bit set in almost every compound has only ~100 compounds without it)
    * for a colliding bit, Δ averages over every substructure in it
    ///
    """),
            mo.hstack([min_sim, min_dy], justify="start", gap=2),
        ]
    )
    return min_dy, min_sim


@app.cell
def _(Chem, Crippen, min_dy, min_sim, np, pair_dy, pair_i, pair_j, pair_sim, pl, train):
    _m = (pair_sim >= min_sim.value) & (pair_dy >= min_dy.value)
    _logp = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in train["smiles"]])
    _ids, _y = train["id"].to_numpy(), train["pEC50"].to_numpy()
    _a, _b = pair_i[_m], pair_j[_m]
    # Order each pair so that A is the more potent compound.
    _swap = _y[_a] < _y[_b]
    _a, _b = np.where(_swap, _b, _a), np.where(_swap, _a, _b)
    cliffs = (
        pl.DataFrame(
            {
                "A": _ids[_a],
                "B": _ids[_b],
                "Tanimoto": pair_sim[_m].astype(float).round(3),
                "pEC50 A": _y[_a],
                "pEC50 B": _y[_b],
                "Δ pEC50": pair_dy[_m].round(2),
                "Δ logP (A−B)": (_logp[_a] - _logp[_b]).round(2),
                "B CI width": train["ci_width"].to_numpy()[_b].round(2),
            }
        )
        .with_columns((pl.col("Δ pEC50") / (1 - pl.col("Tanimoto") + 1e-3)).round(1).alias("SALI"))
        .sort("SALI", descending=True)
        .with_row_index("rank", offset=1)
    )
    cliffs = cliffs.select("rank", pl.exclude("rank", "SALI"), "SALI")
    return (cliffs,)


@app.cell
def _(cliff_example, cliffs, mo):
    # where the example pair from 3b sits in the table (0 when the sliders leave it out)
    _rank = next(
        (
            r
            for r, a, b in cliffs.select("rank", "A", "B").iter_rows()
            if {a, b} == set(cliff_example)
        ),
        0,
    )
    cliff_table = mo.ui.table(
        cliffs,
        selection="single",
        initial_selection=[0] if cliffs.height else None,
        page_size=6,
        freeze_columns_right=["SALI"],
        label=f"{cliffs.height:,} pairs · highest SALI first"
        + (f" · the example above is rank {_rank}" if _rank else ""),
    )
    cliff_table
    return (cliff_table,)


@app.cell
def _(MorganExplorer, cliff_table, mo, train):
    _sel = cliff_table.value
    if _sel is None or len(_sel) == 0:
        cliff_explorer = mo.md("_Select a pair in the table above._")
    else:
        _r = _sel.row(0, named=True)
        _smi = dict(zip(train["id"], train["smiles"]))
        cliff_explorer = mo.ui.anywidget(
            MorganExplorer(
                [
                    {"id": _r["A"], "smiles": _smi[_r["A"]], "label": f"pEC50 {_r['pEC50 A']:.2f}"},
                    {"id": _r["B"], "smiles": _smi[_r["B"]], "label": f"pEC50 {_r['pEC50 B']:.2f}"},
                ],
                reference=train["smiles"].to_list(),
                y=train["pEC50"].to_numpy(),
                y_label="pEC50",
                pair_note=f"Δ pEC50 {_r['Δ pEC50']:.2f} · Δ logP {_r['Δ logP (A−B)']:+.2f}",
            )
        )
    cliff_explorer
    return


@app.cell(hide_code=True)
def _(cliffs, mo, pl):
    _n = cliffs.height
    _noisy = cliffs.filter(pl.col("B CI width") > 1.5).height if _n else 0
    _lip = cliffs.filter(pl.col("Δ logP (A−B)") > 0).height if _n else 0
    mo.md(
        f"""
    Among these {_n} pairs:

    * in {_noisy}, the weaker compound's CI is wider than 1.5 log units, so part of the difference is measurement uncertainty
    * in {_lip} ({_lip / max(_n, 1):.0%}), the more potent compound also has the higher calculated logP: a whole-molecule property is at work, not any single bit

    Part of a cliff's gap comes from what a fingerprint cannot represent: measurement uncertainty and whole-molecule properties.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c · Different molecules, identical fingerprints

    If two different molecules produce exactly the same bit vector, any model built on those bits **must** predict the same value for both. Below are all such groups in train, with the reason the fingerprint cannot tell them apart.
    """)
    return


@app.cell
def _(Chem, X_train, mo, np, pl, rdFingerprintGenerator, train):
    from collections import defaultdict

    def why_identical(smiles: list[str]) -> str:
        """Classify a group of molecules that share a Morgan bit vector."""
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        flat = {Chem.MolToSmiles(m, isomericSmiles=False) for m in mols}
        if len(flat) > 1:
            # Different constitution, same set of radius-2 environments: only repeat counts differ.
            return "ring size / chain length"
        # Count specified stereo elements (tetrahedral centres and double-bond geometry).
        n_specified = {
            sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms())
            + sum(b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds())
            for m in mols
        }
        return "stereo specified / unspecified" if len(n_specified) > 1 else "stereoisomers"

    _groups = defaultdict(list)
    for _i, _fp in enumerate(X_train):
        _groups[_fp.tobytes()].append(_i)
    _twins = sorted(
        (g for g in _groups.values() if len(g) > 1),
        key=lambda g: -np.ptp(train["pEC50"].to_numpy()[g]),
    )
    _smiles = train["smiles"].to_list()
    _rows = [
        {**train.row(i, named=True), "group": gi + 1, "why": why_identical([_smiles[k] for k in g])}
        for gi, g in enumerate(_twins)
        for i in g
    ]
    fp_twins = pl.DataFrame(_rows).select("id", "smiles", "group", "why", "pEC50", "ci_width")
    _summary = fp_twins.group_by("group", "why").agg(
        (pl.col("pEC50").max() - pl.col("pEC50").min()).alias("range")
    )
    _n_stereo = _summary.filter(pl.col("why") != "ring size / chain length").height
    _chiral_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )

    def _splits(g):
        return (
            len(
                {
                    _chiral_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(_smiles[k])).tobytes()
                    for k in g
                }
            )
            > 1
        )

    _n_split = sum(
        _splits(g)
        for g in _twins
        if why_identical([_smiles[k] for k in g]) != "ring size / chain length"
    )

    # one row per group: stereo groups first, then ring size / chain length; largest spread first
    _y = train["pEC50"].to_numpy()
    _ids = train["id"].to_list()
    twin_smiles = dict(zip(_ids, _smiles))
    _table_rows = []
    for _g in sorted(
        _twins,
        key=lambda g: (
            why_identical([_smiles[k] for k in g]) == "ring size / chain length",
            -np.ptp(_y[g]),
        ),
    ):
        _why = why_identical([_smiles[k] for k in _g])
        _a, _b = sorted(_g, key=lambda k: -_y[k])[:2]
        _table_rows.append(
            {
                "why": _why,
                "A": _ids[_a],
                "B": _ids[_b],
                "pEC50 A": _y[_a],
                "pEC50 B": _y[_b],
                "Δ pEC50": round(float(_y[_a] - _y[_b]), 2),
                "includeChirality": "—"
                if _why == "ring size / chain length"
                else ("splits" if _splits(_g) else "no split"),
            }
        )
    twin_table = mo.ui.table(
        pl.DataFrame(_table_rows),
        selection="single",
        initial_selection=[0],
        page_size=10,
        label="groups sharing a fingerprint (select a row to compare below)",
    )
    mo.vstack(
        [
            mo.md(
                f"""
    **{len(_twins)} groups** ({fp_twins.height} train compounds) share a fingerprint, and pEC50 differs by up to **{_summary["range"].max():.2f}** within a group. There are two reasons.

    **Stereochemistry ({_n_stereo} groups)**

    * RDKit's ECFP4 ignores chirality and E/Z geometry unless you pass `includeChirality=True`
    * each group is one compound recorded once with its stereo specified and once without
    * examples: lansoprazole and dexlansoprazole (Δ 0.94), bupivacaine and levobupivacaine, rifampicin with and without E/Z labels
    * `includeChirality=True` separates {_n_split} of the {_n_stereo} groups; the sulfoxide stereocentre of lansoprazole is not picked up either way

    **Ring size / chain length ({len(_twins) - _n_stereo} groups)**

    * examples: cyclohexylamine and cycloheptylamine, azepane and azocane, nonanoic and palmitic acid
    * within radius 2 every atom sees the same surroundings, so the set of substructures is identical and only their counts differ
    * a bit vector keeps no counts and cannot tell them apart; a count fingerprint can

    Molecules with the same bit vector get the same prediction from any model; ECFP4 does not see stereo (by default) or counts.
    """
            ),
            twin_table,
        ]
    )
    return twin_smiles, twin_table


@app.cell
def _(MorganExplorer, mo, train, twin_smiles, twin_table):
    _sel = twin_table.value
    if _sel is None or len(_sel) == 0:
        twin_explorer = mo.md("_Select a group in the table above._")
    else:
        _r = _sel.row(0, named=True)
        twin_explorer = mo.ui.anywidget(
            MorganExplorer(
                [
                    {
                        "id": _r["A"],
                        "smiles": twin_smiles[_r["A"]],
                        "label": f"pEC50 {_r['pEC50 A']:.2f}",
                    },
                    {
                        "id": _r["B"],
                        "smiles": twin_smiles[_r["B"]],
                        "label": f"pEC50 {_r['pEC50 B']:.2f}",
                    },
                ],
                reference=train["smiles"].to_list(),
                y=train["pEC50"].to_numpy(),
                y_label="pEC50",
                pair_note=f"Δ pEC50 {_r['Δ pEC50']:.2f}",
                stereo_labels=True,
            )
        )
    twin_explorer
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d · What a bag of substructures cannot express

    PXR's ligand-binding pocket is large, flexible and hydrophobic, and it is known to accept very different scaffolds. If binding depends less on specific substructures than on how lipophilic and how large the whole molecule is, a representation built from the presence of local substructures is a poor fit.
    """)
    return


@app.cell
def _(Chem, Crippen, Descriptors, np, rdmd, test, train):
    DESCRIPTORS = {
        "MolLogP": Crippen.MolLogP,
        "MolWt": Descriptors.MolWt,
        "TPSA": rdmd.CalcTPSA,
        "HBD": rdmd.CalcNumHBD,
        "HBA": rdmd.CalcNumHBA,
        "RotBonds": rdmd.CalcNumRotatableBonds,
        "Rings": rdmd.CalcNumRings,
        "AromaticRings": rdmd.CalcNumAromaticRings,
        "FractionCSP3": rdmd.CalcFractionCSP3,
        "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
        "Heteroatoms": rdmd.CalcNumHeteroatoms,
        "Halogens": lambda m: sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms()),
        "MolMR": Crippen.MolMR,
        "StereoCenters": lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
    }

    def descriptor_matrix(smiles):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        return np.array([[f(m) for f in DESCRIPTORS.values()] for m in mols], dtype=float)

    D_train = descriptor_matrix(train["smiles"].to_list())
    D_test = descriptor_matrix(test["smiles"].to_list())
    return DESCRIPTORS, D_test, D_train


@app.cell(hide_code=True)
def _(DESCRIPTORS, mo):
    prop_pick = mo.ui.dropdown(list(DESCRIPTORS), value="MolLogP", label="descriptor")
    prop_pick
    return (prop_pick,)


@app.cell(hide_code=True)
def _(DESCRIPTORS, D_train, alt, mo, pl, prop_pick, spearmanr, y_train):
    _j = list(DESCRIPTORS).index(prop_pick.value)
    _rhos = sorted(
        ((name, spearmanr(D_train[:, k], y_train)[0]) for k, name in enumerate(DESCRIPTORS)),
        key=lambda t: -abs(t[1]),
    )
    _df = pl.DataFrame({prop_pick.value: D_train[:, _j], "pEC50": y_train})
    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X(f"{prop_pick.value}:Q", bin=alt.Bin(maxbins=40)),
            y=alt.Y("pEC50:Q", bin=alt.Bin(maxbins=30)),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=None),
        )
        .properties(width=320, height=250)
    )
    _bars = (
        alt.Chart(
            pl.DataFrame({"descriptor": [n for n, _ in _rhos], "Spearman ρ": [r for _, r in _rhos]})
        )
        .mark_bar()
        .encode(
            y=alt.Y("descriptor:N", sort=None, title=None),
            x=alt.X("Spearman ρ:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
            color=alt.condition(
                alt.datum.descriptor == prop_pick.value, alt.value("#d6336c"), alt.value("#9ca3af")
            ),
        )
        .properties(width=220, height=250)
    )
    mo.hstack([_chart, _bars], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(
    D_test, D_train, S_test, mo, np, pair_dy, pair_i, pair_j, pair_sim, spearmanr, y_test, y_train
):
    _m = pair_sim >= 0.5
    _rho_pairs = spearmanr(abs(D_train[pair_i[_m], 0] - D_train[pair_j[_m], 0]), pair_dy[_m])[0]
    _rho_logp = spearmanr(D_test[:, 0], y_test)[0]  # column 0 = MolLogP
    _order = np.argsort(-S_test, axis=1)

    def _knn_rho(k):
        return spearmanr(y_train[_order[:, :k]].mean(1), y_test)[0]

    mo.vstack(
        [
            mo.md(
                "**A single whole-molecule number ranks the test set better than an ECFP4 kNN.**"
            ),
            mo.hstack(
                [
                    mo.stat(f"{_rho_logp:.2f}", label="Spearman ρ, logP alone (test)"),
                    mo.stat(f"{_knn_rho(1):.2f}", label="ρ, ECFP4 kNN with k = 1 (test)"),
                    mo.stat(f"{_knn_rho(50):.2f}", label="ρ, ECFP4 kNN with k = 50 (test)"),
                ],
                widths="equal",
                gap=0.5,
            ),
            mo.md(f"""
    * left: the chosen descriptor against pEC50; right: Spearman ρ of each of the 14 descriptors with pEC50 (train)
    * among pairs with Tanimoto ≥ 0.5, the logP difference tracks the pEC50 difference, if weakly (ρ = {_rho_pairs:.2f})
    * a bit vector has no axis for "slightly more lipophilic": adding a methyl either sets a bit or it doesn't

    Continuous whole-molecule properties such as lipophilicity cannot be expressed as a set of substructures that are present or absent.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, model_scores):
    mo.md(f"""
    ## 4 · Inside the model

    What does the model most people train first, **LightGBM on 2048-bit ECFP4** (the same baseline as OpenADMET's challenge tutorial), actually learn? Two ways to look:

    * **feature importance** (total gain per bit): which bits the trees split on most
    * **TreeSHAP** (LightGBM's `pred_contrib=True`): how much each bit raised or lowered this molecule's prediction; spreading a bit's contribution over the atoms that set it gives a per-atom map (the idea behind Riniker & Landrum's similarity maps)

    On the test set this model reaches MAE **{model_scores["MAE"]:.2f}** and Spearman ρ **{model_scores["rho"]:.2f}**. It is the ECFP4-bit baseline of the Model lab in section 5, where its predictions are plotted against the truth.
    """)
    return


@app.cell
def _(fingerprint_matrix, lgb, mo, np, spearmanr, test, train, y_test, y_train):
    N_BITS = 2048  # the usual default, as in most first models
    Xm_train = fingerprint_matrix(train["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    Xm_test = fingerprint_matrix(test["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    with mo.status.spinner(f"Training LightGBM on {N_BITS}-bit ECFP4…"):
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            random_state=0,
            verbose=-1,
        ).fit(Xm_train, y_train)
    pred_test = model.predict(Xm_test)
    model_scores = {
        "MAE": float(np.abs(pred_test - y_test).mean()),
        "rho": float(spearmanr(pred_test, y_test)[0]),
    }
    return N_BITS, Xm_test, Xm_train, model, model_scores, pred_test


@app.cell
def _(N_BITS, Xm_train, model, np):
    # per-bit importance of the model: LightGBM gain, and TreeSHAP over the training set
    bit_gain = model.booster_.feature_importance("gain")
    _abs, _on_sum = np.zeros(N_BITS), np.zeros(N_BITS)
    for _s in range(0, len(Xm_train), 500):  # in chunks: at 8192 bits the full matrix is large
        _x = Xm_train[_s : _s + 500]
        _c = model.predict(_x, pred_contrib=True)[:, :-1]  # last column = expected value
        _abs += np.abs(_c).sum(0)
        _on_sum += (_c * _x).sum(0)
    bit_shap = _abs / len(Xm_train)
    _n_on = Xm_train.sum(0)
    bit_effect = np.divide(_on_sum, _n_on, out=np.zeros(N_BITS), where=_n_on > 0)
    return bit_effect, bit_gain, bit_shap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The most important bits, and what is inside them

    The table below ranks the model's bits by importance.

    * **gain** / **mean |SHAP|**: the importance (share of the total over all bits); rank by either
    * **mean SHAP (bit on)**: the bit's average contribution in the molecules that set it; red raises the prediction, blue lowers it
    * **main substructure**: the bit's most common substructure, and the share of the bit's molecules that contain it
    * click a row to see the substructures inside the bit and the molecules that set it
    * click a column header again to flip the order and see the bits the model never used (gain 0)

    /// details | Definitions
    * **gain**: total loss reduction from the splits on the bit during training (LightGBM [`feature_importance(importance_type="gain")`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance))
    * **mean |SHAP|**: the absolute TreeSHAP contribution of the bit (LightGBM [`predict(pred_contrib=True)`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict)), averaged over all train molecules ([SHAP](https://shap.readthedocs.io/en/latest/))
    * **mean SHAP (bit on)**: the same contribution, signed, averaged over the molecules that set the bit
    * bits with gain 0 are ordered by how many molecules set them
    ///
    """)
    return


@app.cell
def _(BitImportance, bit_effect, bit_gain, bit_shap, mo, train):
    bit_importance = mo.ui.anywidget(
        BitImportance(
            train["smiles"].to_list(),
            importance={"gain": bit_gain, "mean |SHAP|": bit_shap},
            effect=bit_effect,
            effect_label="mean SHAP (bit on)",
            ids=train["id"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    bit_importance
    return


@app.cell(hide_code=True)
def _(N_BITS, Xm_test, bit_gain, census_for, mo, np, pl, pred_test, spearmanr, train, y_test):
    _census = census_for(train["smiles"].to_list(), 2, N_BITS)
    _gain = bit_gain
    _top = np.argsort(-_gain)[:15]
    _top_bits = pl.DataFrame(
        {
            "rank": np.arange(1, 16),
            "bit": _top.astype(int),
            "gain": (_gain[_top] / _gain.sum()).round(4),
            "# environments in bit": _census.n_envs[_top].astype(int),
            "# training molecules with bit": _census.on[:, _top].sum(0).astype(int),
            # share of the molecules setting the bit that contain its most common substructure
            "top substructure share": [
                round(_census.examples[int(b)][0]["count"] / max(int(_census.on[:, b].sum()), 1), 3)
                for b in _top
            ],
        }
    )
    _envs = np.median(_top_bits["# environments in bit"])
    _n_on = _census.on.sum(0)
    _unused = np.flatnonzero(_gain == 0)
    _n0, _n0_bit = len(_unused), int(_unused[np.argmax(_n_on[_unused])]) if len(_unused) else -1
    _n0_mols = int(_n_on[_n0_bit]) if len(_unused) else 0
    # does a test molecule with many ignored bits get a worse prediction?
    _k = (Xm_test[:, _unused] > 0).sum(1)
    _err = np.abs(pred_test - y_test)
    _rho0 = spearmanr(_k, _err)[0]
    _mae_lo, _mae_hi = _err[_k <= 2].mean(), _err[_k >= 7].mean()
    _share = np.median(_top_bits["top substructure share"])
    _mixed = _top_bits.filter(pl.col("top substructure share") < 0.8)["bit"].to_list()
    _mixed_note = (
        f"* the exceptions are bits where a second substructure is also common ({' '.join(map(str, _mixed))}); for these the importance cannot be pinned on one substructure"
        if _mixed
        else "* in all top 15 bits the most common substructure accounts for at least 80%"
    )
    mo.vstack(
        [
            mo.md("""
    **Each top bit holds many substructures, but in most of them one substructure dominates.**
    """),
            mo.hstack(
                [
                    mo.stat(f"{_envs:.0f}", label="substructures per top-15 bit (median)"),
                    mo.stat(
                        f"{_share:.0%}",
                        label="share of the most common substructure (median)",
                    ),
                ],
                widths="equal",
                gap=0.5,
            ),
            mo.md(f"""
    * at {N_BITS} bits a top-15 bit holds about {_envs:.0f} substructures, yet in the median bit {_share:.0%} of the molecules that set it contain its most common one
    {_mixed_note}
    * the model never uses {_n0:,} bits; the most frequent of them is set in {_n0_mols:,} molecules (bit {_n0_bit}). Retraining without them gives about the same test score (MAE 0.59)
    * test molecules with more of these bits have slightly larger errors (Spearman {_rho0:.2f}; MAE {_mae_lo:.2f} with 0–2 of them, {_mae_hi:.2f} with 7 or more)

    Importance belongs to bits, but a top bit is usually one substructure, so it can be read as that substructure's importance; for the exceptions, check what is inside.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading one prediction

    The pair below is the second-hardest test compound of the whole challenge, **OADMET-0006254** (pEC50 2.06, the molecule that collided with itself in section 1), next to its nearest train neighbour **OADMET-0002810** (pEC50 5.95). The only change is a benzene CH turned into a pyridine N. According to OpenADMET's analysis, a related co-crystal structure shows this nitrogen hydrogen-bonding to SER247, which probably changes how the ligand sits in the pocket. Every Tier-1 team over-predicted this compound.

    Atom colours show the model's TreeSHAP contributions (red raises the predicted pEC50, blue lowers it). Click a bit to show its substructure instead. The **SHAP** columns give each bit's contribution for A and B, and the menu offers the other worst-predicted test compounds.
    """)
    return


@app.cell
def _(S_test, nn_test, pl, pred_test, test, train):
    cliff_pairs = (
        pl.DataFrame(
            {
                "test id": test["id"],
                "test pEC50": test["pEC50"],
                "predicted": pred_test.round(2),
                "NN id": train["id"].to_numpy()[nn_test],
                "NN pEC50": train["pEC50"].to_numpy()[nn_test],
                "Tanimoto": S_test.max(1).astype(float).round(3),
            }
        )
        .with_columns((pl.col("predicted") - pl.col("test pEC50")).abs().round(2).alias("|error|"))
        .sort("|error|", descending=True)
    )
    return (cliff_pairs,)


@app.cell(hide_code=True)
def _(cliff_pairs, mo, pl):
    _worst = cliff_pairs.head(8)
    if "OADMET-0006254" not in _worst["test id"]:
        _worst = pl.concat(
            [cliff_pairs.filter(pl.col("test id") == "OADMET-0006254"), _worst.head(7)]
        )
    _labels = {
        f"{r['test id']} (true {r['test pEC50']:.2f}, predicted {r['predicted']:.2f}) vs {r['NN id']}": r[
            "test id"
        ]
        for r in _worst.iter_rows(named=True)
    }
    pair_pick = mo.ui.dropdown(
        _labels,
        value=next(k for k in _labels if k.startswith("OADMET-0006254")),
        label="pair (test compound and its nearest train neighbour)",
    )
    pair_pick
    return (pair_pick,)


@app.cell
def _(
    MorganExplorer,
    N_BITS,
    cliff_pairs,
    data,
    fingerprint_matrix,
    mo,
    model,
    np,
    pair_pick,
    pl,
    train,
):
    _row = cliff_pairs.filter(pl.col("test id") == pair_pick.value).row(0, named=True)
    _smi = dict(zip(data["id"], data["smiles"]))
    _ids = [_row["test id"], _row["NN id"]]
    pair_X = fingerprint_matrix([_smi[i] for i in _ids], 2, N_BITS).astype(np.float32)
    pair_contrib = model.predict(pair_X, pred_contrib=True)  # last column = expected value
    _pred = pair_contrib.sum(1)
    _maps = [{str(b): float(c[b]) for b in np.flatnonzero(x)} for c, x in zip(pair_contrib, pair_X)]
    _y = dict(zip(data["id"], data["pEC50"]))
    pair_y = [_y[i] for i in _ids]
    pair_smiles = [_smi[i] for i in _ids]
    shap_explorer = mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _ids[0],
                    "smiles": _smi[_ids[0]],
                    "label": f"test · true {_y[_ids[0]]:.2f} · predicted {_pred[0]:.2f}",
                },
                {
                    "id": _ids[1],
                    "smiles": _smi[_ids[1]],
                    "label": f"train · true {_y[_ids[1]]:.2f} · predicted {_pred[1]:.2f}",
                },
            ],
            reference=train["smiles"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
            n_bits=N_BITS,
            contributions=_maps,
            contrib_label="SHAP",
            contrib_radius=2,
            contrib_n_bits=N_BITS,
            pair_note=f"baseline (mean prediction) {pair_contrib[0, -1]:.2f}",
        )
    )
    shap_explorer
    return pair_X, pair_contrib, pair_smiles, pair_y


@app.cell(hide_code=True)
def _(N_BITS, census_for, mo, pair_X, pair_contrib, pair_smiles, pair_y, train):
    from molwidgets import molecule_bit_tiles

    _a, _b = pair_X.astype(bool)
    _diff = pair_contrib[0, :-1] - pair_contrib[1, :-1]  # per-feature A − B
    _differ, _shared, _neither = _a ^ _b, _a & _b, ~(_a | _b)
    _gap = _diff.sum()
    _measured = pair_y[0] - pair_y[1]
    _lift = pair_contrib[0, :-1][_shared].sum()  # what the shared bits add to A's prediction
    # substructures only A has: how often does each appear in train?
    _census = census_for(train["smiles"].to_list(), 2, N_BITS)
    _only_a = [
        t
        for t in molecule_bit_tiles(pair_smiles[0], 2, N_BITS)
        if _a[t["bit"]] and not _b[t["bit"]]
    ]
    _train_count = [
        next((e["count"] for e in _census.examples.get(t["bit"], []) if e["uid"] == t["uid"]), 0)
        for t in _only_a
    ]
    _rare = sum(c <= 3 for c in _train_count)
    mo.md(
        f"""
    **Where does the prediction gap come from?** The model predicts A − B = **{_gap:+.2f}** (measured: {_measured:+.2f}). TreeSHAP splits this gap over the bits: the **{int(_differ.sum())}** bits that differ between A and B contribute **{_diff[_differ].sum():+.2f}**, the **{int(_shared.sum())}** shared bits **{_diff[_shared].sum():+.2f}**, and bits absent from both **{_diff[_neither].sum():+.2f}**.

    * a small structural change flips {int(_differ.sum())} bits, because every substructure within radius 2 of the changed atom changes
    * even so, the predicted gap is only {abs(_gap):.2f}, far from the measured {abs(_measured):.2f}; the shared bits push A's prediction up by {_lift:+.2f}
    * {_rare} of the {len(_only_a)} substructures only A has appear in 3 or fewer train compounds, so the weight on their bits was learned from other substructures in the same bits
    * the atom map can only show bits that are set; absent bits matter to the trees too, but cannot be drawn on atoms

    The model reproduced only {abs(_gap) / abs(_measured):.0%} of the measured difference.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Model lab

    Each weakness above has a matching fix:

    * **count fingerprint**: keeps occurrence counts (3c)
    * **more bits**: fewer collisions (1)
    * **include chirality**: tells stereoisomers apart (3c)
    * **whole-molecule descriptors**: cover what substructures cannot (3d)

    LightGBM is trained on train and scored on the 513 test compounds. The three reference settings are precomputed; change the settings and press **Train** to add a row to the scoreboard. The "precomputed" row uses CheMeleon (see below) and was computed outside the notebook under the same conditions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lab_features = mo.ui.multiselect(
        ["ECFP bit", "ECFP count", "RDKit descriptors"],
        value=["ECFP count"],
        label="features",
    )
    lab_radius = mo.ui.dropdown(
        {"ECFP2 (radius 1)": 1, "ECFP4 (radius 2)": 2, "ECFP6 (radius 3)": 3},
        value="ECFP4 (radius 2)",
        label="radius",
    )
    lab_bits = mo.ui.dropdown(
        {"256": 256, "1024": 1024, "2048": 2048, "8192": 8192}, value="2048", label="bits"
    )
    lab_chiral = mo.ui.checkbox(label="include chirality")
    lab_run = mo.ui.run_button(label="Train")
    mo.hstack(
        [lab_features, lab_radius, lab_bits, lab_chiral, lab_run], justify="start", gap=1, wrap=True
    )
    return lab_bits, lab_chiral, lab_features, lab_radius, lab_run


@app.cell
def _(Chem, mo, np, test, train):
    from rdkit.Chem import Descriptors as _Descriptors

    def rdkit_descriptors(smiles):
        """All RDKit 2D descriptors (~217); non-finite values become NaN (LightGBM handles them)."""
        rows = [
            list(_Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)).values()) for s in smiles
        ]
        out = np.array(rows, dtype=float)
        out[~np.isfinite(out)] = np.nan
        return out

    with mo.status.spinner("Computing RDKit descriptors…"):
        R_train = rdkit_descriptors(train["smiles"].to_list())
        R_test = rdkit_descriptors(test["smiles"].to_list())
    return R_test, R_train


@app.cell
def _(
    Chem,
    R_test,
    R_train,
    S_test,
    lgb,
    np,
    rdFingerprintGenerator,
    spearmanr,
    test,
    train,
    y_test,
    y_train,
):
    def featurize(smiles, parts, radius, n_bits, chiral):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=chiral
        )
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        blocks = []
        if "ECFP bit" in parts:
            blocks.append(np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32))
        if "ECFP count" in parts:
            blocks.append(
                np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
            )
        return blocks

    def evaluate(parts, radius=2, n_bits=2048, chiral=False, seed=0):
        """Fit LightGBM on train, return test metrics and predictions."""
        tr = featurize(train["smiles"].to_list(), parts, radius, n_bits, chiral)
        te = featurize(test["smiles"].to_list(), parts, radius, n_bits, chiral)
        if "RDKit descriptors" in parts:
            tr.append(R_train)
            te.append(R_test)
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            verbose=-1,
        ).fit(np.hstack(tr), y_train)
        pred = model.predict(np.hstack(te))
        fp_desc = " + ".join(p.replace("ECFP", f"ECFP{2 * radius}") for p in parts)
        if any(p.startswith("ECFP") for p in parts):
            fp_desc += f" · {n_bits} bit{' · chiral' if chiral else ''}"
        return {
            "features": fp_desc,
            "MAE": round(float(np.abs(pred - y_test).mean()), 3),
            "Spearman ρ": round(float(spearmanr(pred, y_test)[0]), 3),
            "pred. SD": round(float(pred.std()), 2),
            "_pred": pred,
        }

    nn_sim_test = S_test.max(1)
    return evaluate, nn_sim_test


@app.cell
def _(evaluate, mo):
    with mo.status.spinner("Training the three reference models…"):
        baseline_runs = [
            evaluate(["ECFP bit"]),
            evaluate(["RDKit descriptors"]),
            evaluate(["ECFP bit", "RDKit descriptors"]),
        ]
    return (baseline_runs,)


@app.cell
def _(np, pl, spearmanr, test, y_test):
    # CheMeleon (a GNN pretrained to predict Mordred descriptors) needs PyTorch, so its runs were
    # computed outside the notebook with the same split and LightGBM settings (results/)
    from pathlib import Path as _Path

    _file = "chemeleon_test_predictions.csv"
    _local = _Path("results") / _file
    _url = "https://raw.githubusercontent.com/N283T/openadmet-marimo/main/results/" + _file
    try:
        _preds = test.select("id").join(pl.read_csv(_local if _local.exists() else _url), on="id")
    except (OSError, pl.exceptions.PolarsError):
        _preds = None  # offline and no local copy: the scoreboard shows only the notebook's runs
    reference_runs = []
    for _col in [] if _preds is None else _preds.columns[1:]:
        _pred = _preds[_col].to_numpy()
        reference_runs.append(
            {
                "features": _col,
                "MAE": round(float(np.abs(_pred - y_test).mean()), 3),
                "Spearman ρ": round(float(spearmanr(_pred, y_test)[0]), 3),
                "pred. SD": round(float(_pred.std()), 2),
                "_pred": _pred,
            }
        )
    return (reference_runs,)


@app.cell
def _(mo):
    get_runs, set_runs = mo.state([])
    return get_runs, set_runs


@app.cell
def _(evaluate, lab_bits, lab_chiral, lab_features, lab_radius, lab_run, mo, set_runs):
    mo.stop(not lab_run.value)
    mo.stop(not lab_features.value, mo.md("Pick at least one feature family.").callout(kind="warn"))
    with mo.status.spinner("Training…"):
        _res = evaluate(lab_features.value, lab_radius.value, lab_bits.value, lab_chiral.value)
    set_runs(lambda runs: [*runs, _res])
    return


@app.cell(hide_code=True)
def _(baseline_runs, get_runs, mo, pl, reference_runs, y_test):
    all_runs = baseline_runs + reference_runs + get_runs()
    _nb, _nr = len(baseline_runs), len(reference_runs)
    _board = pl.DataFrame(
        [
            {k: v for k, v in r.items() if not k.startswith("_")}
            | {"source": "reference" if i < _nb else "precomputed" if i < _nb + _nr else "yours"}
            for i, r in enumerate(all_runs)
        ]
    )
    run_pick = mo.ui.table(
        _board,
        selection="single",
        # the newest run of your own, else the best baseline
        initial_selection=[len(all_runs) - 1 if get_runs() else _nb - 1],
        label=f"Scoreboard (SD of the true test pEC50 = {y_test.std():.2f}; pick a row to inspect it)",
        page_size=8,
    )
    run_pick
    return all_runs, run_pick


@app.cell(hide_code=True)
def _(all_runs, alt, mo, nn_sim_test, pl, run_pick, test, y_test):
    _i = (
        run_pick.value.select(pl.col("features")).to_series().to_list()[0]
        if run_pick.value is not None and len(run_pick.value)
        else None
    )
    _run = next((r for r in reversed(all_runs) if r["features"] == _i), all_runs[-1])
    _pred = _run["_pred"]
    _df = pl.DataFrame(
        {
            "id": test["id"],
            "true pEC50": y_test,
            "predicted": _pred,
            "NN similarity": nn_sim_test,
            "error": _pred - y_test,
        }
    ).with_columns(
        pl.col("true pEC50").cut([4, 5, 6], labels=["< 4", "4–5", "5–6", "≥ 6"]).alias("true bin")
    )
    _lim = [1.5, 7.5]
    _scatter = (
        alt.Chart(_df)
        .mark_circle(size=26, opacity=0.6)
        .encode(
            x=alt.X("predicted:Q", scale=alt.Scale(domain=_lim)),
            y=alt.Y("true pEC50:Q", scale=alt.Scale(domain=_lim)),
            color=alt.Color("NN similarity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[
                "id",
                alt.Tooltip("true pEC50:Q", format=".2f"),
                alt.Tooltip("predicted:Q", format=".2f"),
            ],
        )
        .properties(width=280, height=260, title=_run["features"])
    ) + alt.Chart(pl.DataFrame({"x": _lim, "y": _lim})).mark_line(
        color="#9ca3af", strokeDash=[4, 4]
    ).encode(x="x", y="y")
    _bias = (
        alt.Chart(_df)
        .mark_boxplot(extent="min-max", size=26, color="#d6336c")
        .encode(
            x=alt.X("true bin:N", sort=["< 4", "4–5", "5–6", "≥ 6"], title="true pEC50"),
            y=alt.Y("error:Q", title="prediction − truth"),
        )
        .properties(width=220, height=260, title="error by true potency")
    )
    _zero = alt.Chart(pl.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
    mo.vstack(
        [
            mo.hstack([_scatter, _bias + _zero], justify="start", gap=2),
            mo.md(f"""
    **Every model pulls its predictions toward the mean.**

    * the predictions have an SD of {_pred.std():.2f} against {y_test.std():.2f} for the truth
    * weak compounds are predicted too potent and potent ones too weak (box plot on the right)
    * of the three reference models, ECFP4 alone has the narrowest spread
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Worth trying (numbers are test MAE):

    * **bits → counts**: adds occurrence counts and separates the ring-size groups from 3c; 0.59 → 0.55
    * **fold size**: 256 bits is clearly worse (0.64), while 2048 and 8192 are close (0.59 → 0.58)
    * **include chirality**: few compounds change, so the score hardly moves, but the stereo groups from 3c become separable
    * **RDKit descriptors**: the 217 descriptors alone reach 0.56, better than ECFP4 (0.59); combined with ECFP4, 0.53
    * **a pretrained GNN's embeddings** (for reference): with the output of [CheMeleon](https://github.com/JacksonBurns/chemeleon), a GNN pretrained to predict Mordred descriptors, as features: [0.54](https://github.com/N283T/openadmet-marimo/blob/main/results/chemeleon_lightgbm.csv)

    Combining substructures (a fingerprint) with whole-molecule properties (descriptors) helps most. CheMeleon, a GNN pretrained to predict descriptors, is in a sense that combination too, and pretrained GNNs also helped in [my challenge report](https://n283t.github.io/openadmet-pxr-model-report/). Still, every model pulls its predictions toward the mean.

    ## 6 · Take-aways

    **About ECFP4**

    * **ECFP4 records which local substructures are present**: counts, stereo (by default) and whole-molecule properties are lost
    * **folding puts unrelated substructures on the same bit**: although a bit that is set often is usually dominated by one substructure
    * **a high Tanimoto does not guarantee similar activity**

    **What happened on PXR**

    * **the test set is close to train, yet the nearest neighbour's activity is of little use**: the test set was built from ECFP4 neighbours, but kNN with k = 1 has a rank correlation near 0
    * **much of the activity depends on whole-molecule properties**: logP alone ranks the test set better than an ECFP4 kNN, and a model on RDKit descriptors alone beats one on ECFP4 alone
    * **some cliffs are measurement uncertainty**: the pEC50 of a weak compound is extrapolated

    **When you use ECFP4**

    * for regression, try counts as well (better than bits on PXR); in a colliding bin, though, the counts of different substructures add up
    * switch on chirality when stereo matters
    * combine it with whole-molecule descriptors
    * before interpreting a bit, check which substructures are in it

    ECFP4 is still a solid place to start. Knowing what is inside it and where it fails tells you what to suspect, and what to add, when it doesn't work on your data.

    ---

    ### About this notebook

    * **Data:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY-4.0): the train set plus the phase 1 and phase 2 unblinded test labels. The test-set design and the analysis of the hardest compounds come from references [4]–[6].
    * **Widgets:** `ECFPMovie`, `ECFPStepper`, `MolGrid`, `MorganBitTiles`, `BitAtlas` and `MorganExplorer` are anywidget components written for this notebook ([source](https://github.com/N283T/openadmet-marimo)).
    * **AI use:** I used Claude (Anthropic) as a coding assistant for the widgets, the video and the notebook scaffolding. The question, the choice of analyses and the interpretation come from my own work on the PXR challenge, and every number shown is computed live in this notebook.

    For what other participants did, the [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/) has a table of the methods used by the 28 Tier 1 teams, and the [results post](https://openadmet.ghost.io/its-the-end-of-the-pxr-challenge-as-we-know-it-and-i-feel-fine/) links each team's model report. Mine (4th in the activity track) is [here](https://n283t.github.io/openadmet-pxr-model-report/). As you might guess, no fingerprint model made it into my final ensemble.

    ### References

    1. Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742–754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
    2. Morgan, H. L. The Generation of a Unique Machine Description for Chemical Structures. *J. Chem. Doc.* **1965**, 5, 107–113. [doi:10.1021/c160017a018](https://doi.org/10.1021/c160017a018)
    3. Riniker, S.; Landrum, G. A. Similarity maps – a visualization strategy for molecular fingerprints and machine-learning methods. *J. Cheminform.* **2013**, 5, 43. [doi:10.1186/1758-2946-5-43](https://doi.org/10.1186/1758-2946-5-43)
    4. OpenADMET. [Announcing the next OpenADMET Blind Challenge: Predicting PXR Induction](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/) (2026-03-17)
    5. OpenADMET. [Predicting PXR Induction - We have liftoff](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/) (2026-04-01)
    6. OpenADMET. [Don't Look Back in Error: What we learned predicting PXR induction (Part I)](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/) (2026-08-27)
    """)
    return


if __name__ == "__main__":
    app.run()
