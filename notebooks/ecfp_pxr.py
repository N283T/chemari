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
app = marimo.App(width="medium", app_title="Similar, but not the same: ECFP4 on PXR")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Similar, but not the same
    ### ECFP4 from the inside, and why it stumbles on PXR induction

    The **pregnane X receptor (PXR)** is a nuclear receptor that senses foreign molecules and switches on
    CYP3A4, P-gp and friends. A drug that activates PXR can speed up the metabolism of *other* drugs, so
    PXR induction is a classic drug–drug-interaction liability. OpenADMET measured it for >11,000
    compounds and ran a blind challenge asking for **pEC50** predictions on 513 new ones.

    In that challenge, models built on **ECFP4 (Morgan) fingerprints**, the workhorse of QSAR, were
    consistently among the weakest. In my own entry (4th of 95), a Morgan-only LightGBM landed around
    0.57 CV MAE while descriptor/embedding ensembles got below 0.40, and other teams reported the same
    ordering. This notebook asks **why**.

    First, eighty seconds on what ECFP4 actually computes. Press play:
    """)
    return


@app.cell(hide_code=True)
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That is the whole algorithm. Keep three things from it, because the rest of the notebook is about
    them: the fingerprint records **which** substructures occur (not how often, not where, not the
    molecule as a whole); **folding** puts unrelated substructures on the same bit; and **Tanimoto**
    similarity counts shared bits, collisions included.

    Now we take it to the PXR data:

    1. **ECFP4 on a real molecule**: the same picture for any compound in the dataset
    2. **The data, and how the test set was built**: every test compound is an ECFP4 neighbour of a hit
    3. **Where ECFP4 breaks**: the similarity principle, activity cliffs, identical fingerprints, and
       what a bag of substructures cannot express
    4. **Inside the model**: what a LightGBM on ECFP4 bits learns, bit by bit and atom by atom
    5. **Model lab**: counts, fold size, chirality and descriptors, tested on the unblinded test set
    6. **Take-aways**

    /// admonition | Names you will see
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) and the **Morgan fingerprint**
    (RDKit's name) are the same idea. The number in ECFP*n* is the *diameter*: **ECFP4 = Morgan radius 2**.
    ///
    """)
    return


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
        ECFPMovie,
        MolGrid,
        MorganBitTiles,
        MorganExplorer,
        bit_gallery,
        census_for,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )

    return (
        Chem,
        Crippen,
        Descriptors,
        ECFPMovie,
        MolGrid,
        MorganBitTiles,
        MorganExplorer,
        alt,
        bit_gallery,
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
def _(census_for, mo, train):
    _envs = census_for(train["smiles"].to_list(), 2, 2048).n_envs
    mo.md(
        f"""
    ## 1 · ECFP4 on a real molecule

    The movie used a five-atom toy with 10 environments. A drug-like PXR compound has around 45, and
    the {train.height:,} training molecules together contain **{int(_envs.sum()):,}** distinct ECFP4
    environments for 2048 bits: about **{_envs[_envs > 0].mean():.0f} per bit**, and every bit is shared.

    Pick any compound in the grid (filter by id, sort by pEC50, or type a SMARTS such as `c1ccncc1`); the
    first two are a benzene → pyridine pair we will meet again in section 4.
    Below it, the molecule sits on the left and its bits on the right, each drawn like the cards in the
    movie with RDKit's `DrawMorganEnv` (the colour key sits above the widget). **Hover a bit** to light up
    where it comes from. A **red edge** marks a collision inside the molecule; the **badge** counts the
    other substructures that share the bit across the dataset. Click a row to see them.
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
    Things to look for:

    * **The default compound, OADMET-0006254, collides with itself.** At 2048 bits its quaternary
      carbon `C(C)(S)(C)C` and an aromatic `c(N)(c)c` land on the same bit. Remember this molecule: it
      comes back in section 4 as one of the hardest test compounds of the whole challenge.
    * **Radius-0 bits are shared by everything.** A bare atom type does not depend on the rest of the
      molecule, so its bit is also set by hundreds of unrelated environments across the dataset. Try any
      carboxylic acid: the carbonyl carbon `[C;D3;H0]` and the hydroxyl oxygen `[O;D1;H1]` share bit 807.
    * **Repeats collapse.** Three methyls or four aromatic CH give one tile each (×3, ×4). Switch the
      widget to 8192 bits and most red frames vanish.
    """)
    return


@app.cell(hide_code=True)
def _(hygiene, mo, test, train):
    mo.vstack(
        [
            mo.md(r"""
    ## 2 · The data, and how the test set was built

    We use the public OpenADMET PXR release on Hugging Face (CC-BY-4.0): the **training set** and the
    **full test set**, whose labels were unblinded after the challenge. Every SMILES is standardized the
    same way: largest fragment, neutralized, stereochemistry kept, canonical SMILES.
    """),
            mo.hstack(
                [
                    mo.stat(f"{train.height:,}", label="training compounds"),
                    mo.stat(f"{test.height:,}", label="test compounds (unblinded)"),
                    mo.stat(str(hygiene["changed"]), label="charged / salt forms neutralized"),
                    mo.stat(str(hygiene["dup_within"]), label="duplicates after standardization"),
                    mo.stat(str(hygiene["overlap"]), label="test compounds also in train"),
                ],
                widths="equal",
                gap=0.5,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, pl, train):
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
    _chart = (
        alt.Chart(_bins)
        .mark_bar(color="#6b7280")
        .encode(
            x=alt.X("pEC50 bin:N", sort=None),
            y=alt.Y("median 95% CI width:Q", title="median 95% CI width (log units)"),
            tooltip=["pEC50 bin", "n", alt.Tooltip("median 95% CI width:Q", format=".2f")],
        )
        .properties(height=180, width=300)
    )
    _low = _bins.filter(pl.col("pEC50 bin") == "< 3")
    mo.accordion(
        {
            "Not all labels are equal: weak compounds carry wide confidence intervals": mo.hstack(
                [
                    _chart,
                    mo.md(
                        f"""
    pEC50 comes from a dose–response fit. Weak compounds never reach a plateau, so their EC50 is an
    extrapolation: below pEC50 3 the median confidence interval is
    **{_low["median 95% CI width"].item():.1f} log units wide**, versus ~0.2–0.3 for potent compounds.
    That is **{_low["n"].item()} training compounds** ({_low["n"].item() / train.height:.0%}) whose exact
    value is mostly noise. Keep it in mind when we look at activity cliffs.
    """
                    ),
                ],
                widths=[1, 1.4],
                align="center",
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The 513 test compounds are not a random sample. OpenADMET took the **63** compounds that were potent
    (EC50 ≤ 1 µM) *and* selective in the PXR-null counter-screen, and bought Enamine analogues with an
    **ECFP4 Tanimoto similarity > 0.4** to them
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)).
    It is a hit-expansion set, exactly how a medicinal chemist follows up a screen, and it was built with
    the very fingerprint from the movie.

    So two things are true by construction: every test compound has a close relative in the training
    data, and that relative is usually potent. For every compound we find its **nearest neighbour (NN)**
    in the training set by ECFP4 Tanimoto (2048 bits; training compounds exclude themselves).
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
                    range=["#3b82f6", "#f59e0b"], domain=["train → rest of train", "test → train"]
                ),
                legend=alt.Legend(orient="top", title=None),
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
                "NN pEC50:Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(orient="top")
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
    **Left:** as designed, the test set is *closer* to the training data (median NN similarity
    **{np.median(S_test.max(1)):.2f}**) than the training compounds are to each other
    (**{np.median(S_train.max(1)):.2f}**). Being out of domain is not the problem here.

    **Right:** each test compound against its nearest neighbour, coloured by the neighbour's potency.
    The neighbours are overwhelmingly potent (mean pEC50 **{_nn.mean():.2f}** against
    **{y_train.mean():.2f}** for the training set; **{(_nn >= 5.5).mean():.0%}** have pEC50 ≥ 5.5), yet the
    test compounds themselves spread from ~2 to ~7. This is an **SAR exploration around the hits**: the
    real question is which small changes keep the activity, and ECFP4 Tanimoto cannot tell you that.
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
    The most direct fingerprint model there is makes the point: predict each test compound's pEC50 as
    the mean of its **k most similar** training compounds. {k_slider}
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
    mo.hstack(
        [
            (_diag + _pts).properties(width=330, height=300),
            mo.vstack(
                [
                    mo.stat(f"{_mae:.2f}", label=f"MAE, {_k}-NN"),
                    mo.stat(f"{_rho:.2f}", label="Spearman ρ"),
                    mo.stat(f"{_rand:.2f}", label="MAE if you guess a random training compound"),
                    mo.md(
                        "With **k = 1**, the nearest neighbour has essentially no ranking power "
                        "(ρ ≈ 0), and its pEC50 is off by as much as a randomly chosen training compound. "
                        "Averaging more neighbours helps only by shrinking every prediction toward the "
                        "mean. Drag the slider and watch the cloud collapse into a vertical band."
                    ),
                ]
            ),
        ],
        widths=[1.1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Where ECFP4 breaks

    ### 3a · The similarity principle, measured

    The *similar property principle* says that the activity difference between two molecules should
    shrink as their structural similarity grows. We can measure it directly on all ~8.5 million training
    pairs: bin the pairs by Tanimoto similarity and look at the typical |Δ pEC50| in each bin.
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
def _(alt, mo, pl, random_pair_dy, similarity_curve):
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
    _hi = _df.filter(pl.col("sim_lo") >= 0.5)
    mo.hstack(
        [
            (_band + _line + _rule).properties(height=260, width=360),
            mo.md(
                f"""
    The dashed line is a random pair (**{random_pair_dy:.2f}** log units). The curve does go down, so
    similarity is not useless, but slowly. Even among pairs with Tanimoto ≥ 0.5 the typical difference
    is **{(_hi["mean_dy"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.2f}** log units, and
    **{(_hi["frac_gt1"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.0%}** of pairs differ by more than
    10-fold in potency.

    Hover the points for the counts: there are very few truly close pairs, so most of the data lives in
    the region where the fingerprint says "somewhat similar" and PXR says "could be anything".
    """
            ),
        ],
        widths=[1.1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    min_sim = mo.ui.slider(0.4, 0.9, value=0.55, step=0.05, label="min Tanimoto", show_value=True)
    min_dy = mo.ui.slider(0.5, 3.0, value=1.5, step=0.25, label="min |Δ pEC50|", show_value=True)
    mo.md(
        f"""
    ### 3b · Browse the activity cliffs

    An **activity cliff** is a pair that looks alike but behaves differently. Choose what "alike" and
    "differently" mean, pick a pair from the table, and compare the two fingerprints bit by bit.
    Bits present in only one molecule are all a fingerprint model *can* use to explain the difference,
    and they are often a handful of generic environments.

    {mo.hstack([min_sim, min_dy], justify="start", gap=2)}
    """
    )
    return min_dy, min_sim


@app.cell
def _(Crippen, Chem, min_dy, min_sim, np, pair_dy, pair_i, pair_j, pair_sim, pl, train):
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
    )
    return (cliffs,)


@app.cell
def _(cliffs, mo):
    cliff_table = mo.ui.table(
        cliffs,
        selection="single",
        initial_selection=[0] if cliffs.height else None,
        page_size=6,
        label=f"{cliffs.height:,} pairs · SALI = |Δ pEC50| / (1 − similarity)",
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
    Two things to look for while browsing:

    * **Is the weak partner a real measurement?** In {_noisy} of these {_n} pairs, the weaker compound
      has a CI wider than 1.5 log units: part of the "cliff" is assay noise at the bottom of the scale.
    * **Which one is greasier?** In {_lip} of {_n} pairs ({_lip / max(_n, 1):.0%}) the more potent
      compound also has the higher calculated logP. A property of the *whole molecule* is doing work that
      no single bit can express (3d).
    """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c · Different molecules, identical fingerprints

    If two different molecules produce exactly the same bit vector, any model built on those bits
    **must** predict the same value for both. Here is every such group in the training set, with the
    reason the fingerprint cannot tell its members apart.
    """)
    return


@app.cell
def _(Chem, MolGrid, X_train, mo, np, pl, rdFingerprintGenerator, train):
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
        return "stereo specified vs unspecified" if len(n_specified) > 1 else "stereoisomers"

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
    twin_grid = mo.ui.anywidget(
        MolGrid(
            fp_twins,
            subset=["group", "why", "pEC50", "ci_width"],
            color_by="pEC50",
            page_size=14,
            cell_size=150,
        )
    )
    _n_stereo = _summary.filter(pl.col("why") != "ring size / chain length").height
    _chiral_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    _n_split = sum(
        len(
            {_chiral_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(_smiles[k])).tobytes() for k in g}
        )
        > 1
        for g in _twins
        if why_identical([_smiles[k] for k in g]) != "ring size / chain length"
    )
    mo.vstack(
        [
            mo.md(
                f"""
    **{len(_twins)} groups** ({fp_twins.height} training compounds) collapse onto a shared fingerprint;
    potency inside a group differs by up to **{_summary["range"].max():.2f}** log units. Two reasons:

    * **Stereochemistry** ({_n_stereo} groups). RDKit Morgan fingerprints ignore chirality and
      double-bond geometry unless you pass `includeChirality=True`. In these groups one record has its
      stereocentre or double bond *specified* and the other does not: a single enantiomer next to what is
      most likely the racemate (lansoprazole / dexlansoprazole, bupivacaine / levobupivacaine), or the same
      drug registered with and without E/Z labels (rifampicin, the textbook PXR agonist). The lansoprazole
      pair differs by ~1 log unit, a real question for a chemist and invisible to the fingerprint.
      Switching on chirality separates {_n_split} of these {_n_stereo} groups; the sulfoxide stereocentre
      is not picked up either way.
    * **Ring size / chain length** ({len(_twins) - _n_stereo} groups). Cyclohexyl vs cycloheptyl amine,
      azepane vs azocane, nonanoic vs palmitic acid: every atom sees the same neighbourhood within radius
      2, so the *set* of environments is identical and only *how many times* each occurs differs. A bit
      vector stores presence, not counts; a count fingerprint separates these (try it in the lab).
    """
            ),
            twin_grid,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d · What a bag of substructures cannot express

    PXR's ligand-binding pocket is large, flexible and hydrophobic; it is famous for accepting very
    different scaffolds. If binding is driven by *how greasy and how big* a molecule is rather than by a
    specific pharmacophore, then a bag of local substructures is the wrong language.
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
def _(D_train, mo, pair_dy, pair_i, pair_j, pair_sim, spearmanr):
    _m = pair_sim >= 0.5
    _dlogp = abs(D_train[pair_i[_m], 0] - D_train[pair_j[_m], 0])
    _rho = spearmanr(_dlogp, pair_dy[_m])[0]
    mo.md(
        f"""
    A single number, Crippen logP, ranks training compounds about as well as the fingerprint kNN ranked
    the test set. And *within* the similar pairs (Tanimoto ≥ 0.5), the logP difference tracks the potency
    difference (Spearman ρ = **{_rho:.2f}**): part of what looks like a cliff to the fingerprint is a
    smooth lipophilicity trend. A folded bit vector has no axis for "a bit more lipophilic"; adding a
    methyl either flips a bit or it doesn't.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Inside the model

    Let's train the model most people would train first, **LightGBM on 2048-bit ECFP4** (the same
    baseline as OpenADMET's challenge tutorial), and ask it what it learned. Two standard tools:

    * **Global feature importance** (total gain per bit): which bits the trees split on most.
    * **Local attributions** (TreeSHAP, via LightGBM's `pred_contrib=True`): how much each bit pushed
      *this* molecule's prediction up or down. Spreading each bit's contribution over the atoms that set
      it gives a per-atom map, in the spirit of Riniker & Landrum's similarity maps.

    Choose the fold size and train. Watch what happens to the "top bits" as you change it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_bits = mo.ui.radio(
        {"2048 bits (the usual default)": 2048, "8192 bits": 8192},
        value="2048 bits (the usual default)",
        label="fold size for the model",
        inline=True,
    )
    model_bits
    return (model_bits,)


@app.cell
def _(fingerprint_matrix, lgb, model_bits, mo, np, spearmanr, test, train, y_test, y_train):
    N_BITS = model_bits.value
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
    return N_BITS, model, model_scores, pred_test


@app.cell(hide_code=True)
def _(N_BITS, alt, census_for, mo, model, model_scores, np, pl, train):
    _census = census_for(train["smiles"].to_list(), 2, N_BITS)
    _gain = model.booster_.feature_importance("gain")
    _top = np.argsort(-_gain)[:15]
    top_bits = pl.DataFrame(
        {
            "rank": np.arange(1, 16),
            "bit": _top.astype(int),
            "gain": (_gain[_top] / _gain.sum()).round(4),
            "# environments in bit": _census.n_envs[_top].astype(int),
            "# training molecules with bit": _census.on[:, _top].sum(0).astype(int),
        }
    )
    _chart = (
        alt.Chart(top_bits)
        .mark_bar()
        .encode(
            y=alt.Y("bit:N", sort=None, title=f"top bits ({N_BITS})"),
            x=alt.X("# environments in bit:Q", title="distinct substructures sharing the bit"),
            color=alt.Color(
                "gain:Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="share of gain")
            ),
            tooltip=[
                "rank",
                "bit",
                alt.Tooltip("gain:Q", format=".1%"),
                "# environments in bit",
                "# training molecules with bit",
            ],
        )
        .properties(width=340, height=300)
    )
    mo.hstack(
        [
            _chart,
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.stat(f"{model_scores['MAE']:.2f}", label="test MAE"),
                            mo.stat(f"{model_scores['rho']:.2f}", label="test Spearman ρ"),
                            mo.stat(
                                f"{np.median(top_bits['# environments in bit']):.0f}",
                                label="median substructures per top-15 bit",
                            ),
                        ]
                    ),
                    mo.md(
                        f"""
    Each bar is one of the 15 most important bits; its length is the number of **different**
    substructures from the training set that were folded into it. At {N_BITS} bits the model's
    favourite features are each a mixture of about
    **{np.median(top_bits["# environments in bit"]):.0f}** unrelated environments. "Bit {top_bits["bit"][0]}
    is the most important feature" is not an explanation: it is a pointer to a bag of substructures.
    Switch to 8192 bits above and the bags shrink to a few members each, while the accuracy barely
    moves. Pick a bit below to see what is inside it.
    """
                    ),
                ]
            ),
        ],
        widths=[1, 1.1],
        align="center",
    )
    return (top_bits,)


@app.cell(hide_code=True)
def _(mo, top_bits):
    top_bit_pick = mo.ui.dropdown(
        {f"#{r} · bit {b}": int(b) for r, b in zip(top_bits["rank"], top_bits["bit"])},
        value=f"#1 · bit {top_bits['bit'][0]}",
        label="look inside a top bit",
    )
    top_bit_pick
    return (top_bit_pick,)


@app.cell(hide_code=True)
def _(N_BITS, bit_gallery, mo, top_bit_pick, train):
    mo.Html(
        bit_gallery(
            top_bit_pick.value,
            train["smiles"].to_list(),
            ids=train["id"].to_list(),
            radius=2,
            n_bits=N_BITS,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading one prediction

    Now zoom into single molecules. The pair below is the second-hardest test compound of the whole
    challenge, **OADMET-0006254** (pEC50 2.06, the self-colliding molecule from section 1), next to its
    nearest training neighbour **OADMET-0002810** (pEC50 5.95). The only change is a benzene CH →
    pyridine N. According to the OpenADMET analysis, a related co-crystal structure shows that nitrogen
    H-bonding to SER247, which likely changes how the ligand sits in the pocket. Every Tier-1 team
    over-predicted this compound.

    Atom colours show the model's TreeSHAP attribution (red raises the predicted pEC50, blue lowers it).
    Click a bit to see its environments instead; the table's **SHAP** columns show each bit's
    contribution for A and B. The menu offers the other worst-predicted test compounds.
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
        label="pair (test compound vs its nearest training neighbour)",
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
    return pair_X, pair_contrib, pair_y


@app.cell(hide_code=True)
def _(mo, pair_X, pair_contrib, pair_y):
    _a, _b = pair_X.astype(bool)
    _diff = pair_contrib[0, :-1] - pair_contrib[1, :-1]  # per-feature A − B
    _differ, _shared, _neither = _a ^ _b, _a & _b, ~(_a | _b)
    _gap = _diff.sum()
    mo.md(
        f"""
    **Where does the prediction gap come from?** The model predicts A − B = **{_gap:+.2f}**
    (measured: {pair_y[0] - pair_y[1]:+.2f}). TreeSHAP splits that gap exactly over the features:
    the **{int(_differ.sum())}** bits that differ between A and B account for **{_diff[_differ].sum():+.2f}**,
    the **{int(_shared.sum())}** bits they share for **{_diff[_shared].sum():+.2f}** (trees split on
    combinations, so a shared bit can matter more in one context than the other), and bits absent from
    both for **{_diff[_neither].sum():+.2f}**. The atom map shows only bits that are *present*; absences
    matter to trees too, but they cannot be drawn on atoms.

    * **The change is visible only as a few bits.** A single atom swap flips a handful of environments
      (the *only A* / *only B* rows). The model sees them, but in the training data each of those bits
      also stands for many other substructures (**# envs**), so its learned effect is an average over
      everything folded into it. In the benzene → pyridine pair, the new nitrogen barely lights up.
    * **Similar fingerprints give similar predictions.** Most attribution sits on shared bits. The model
      is doing exactly what ECFP asks it to do; the activity cliff is simply not in the representation.
    * **An attribution on a colliding bit is ambiguous.** A red atom means "the model likes this bit",
      not "the model likes this group", when the bit holds a dozen substructures. Retrain at 8192 bits
      and compare.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Model lab

    Everything above points at the same few fixes: a **count** fingerprint to keep "how much", more bits
    to reduce collisions, chirality where stereo matters, and a few **whole-molecule descriptors** for
    what a bag of substructures cannot express. Test them here: LightGBM on the training set, scored on
    the 513 unblinded test compounds. The three reference configurations are pre-computed; change the
    settings and press **Train** to add your own row to the scoreboard.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lab_features = mo.ui.multiselect(
        ["Morgan bits", "Morgan counts", "14 descriptors"],
        value=["Morgan counts"],
        label="features",
    )
    lab_radius = mo.ui.dropdown({"1": 1, "2": 2, "3": 3}, value="2", label="radius")
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
def _(
    Chem,
    D_test,
    D_train,
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
        if "Morgan bits" in parts:
            blocks.append(np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32))
        if "Morgan counts" in parts:
            blocks.append(
                np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
            )
        return blocks

    def evaluate(parts, radius=2, n_bits=2048, chiral=False, seed=0):
        """Fit LightGBM on train, return test metrics and predictions."""
        tr = featurize(train["smiles"].to_list(), parts, radius, n_bits, chiral)
        te = featurize(test["smiles"].to_list(), parts, radius, n_bits, chiral)
        if "14 descriptors" in parts:
            tr.append(D_train)
            te.append(D_test)
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
        fp_desc = " + ".join(parts)
        if any(p.startswith("Morgan") for p in parts):
            fp_desc += f" (r{radius}, {n_bits}{', chiral' if chiral else ''})"
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
            evaluate(["Morgan bits"]),
            evaluate(["14 descriptors"]),
            evaluate(["Morgan bits", "14 descriptors"]),
        ]
    return (baseline_runs,)


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
def _(baseline_runs, get_runs, mo, pl, y_test):
    all_runs = baseline_runs + get_runs()
    _board = pl.DataFrame(
        [
            {k: v for k, v in r.items() if not k.startswith("_")}
            | {"source": "reference" if i < len(baseline_runs) else "yours"}
            for i, r in enumerate(all_runs)
        ]
    )
    run_pick = mo.ui.table(
        _board,
        selection="single",
        initial_selection=[len(all_runs) - 1],
        label=f"Scoreboard (true pEC50 SD on test = {y_test.std():.2f}; pick a row to inspect it)",
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
            mo.md(
                f"Predictions span an SD of **{_pred.std():.2f}** against **{y_test.std():.2f}** for the truth. "
                "Every model here regresses toward the middle; the question is how much. Weak compounds are "
                "predicted too potent, potent compounds too weak, and fingerprint-only models compress the most."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things worth trying:

    * **Bits → counts** at the same radius and length. Counts add "how much" information for free, and
      they separate the ring-size twins from 3c.
    * **256 → 8192 bits.** Fewer collisions help a little. Collisions are real (section 1), but they are
      not the main story.
    * **Include chirality.** Only a handful of compounds change, so the score barely moves; for the
      stereoisomer groups in 3c it is the difference between "impossible" and "possible".
    * **Morgan + 14 descriptors.** Fourteen simple whole-molecule numbers repair much of what 2048 bits
      miss.

    ## 6 · Take-aways

    1. **ECFP4 is a bag of local substructures.** It records which environments occur, folds them into a
       fixed number of bits, and forgets counts, stereochemistry by default, and the molecule as a whole.
    2. **The PXR test set is in domain by design.** Every test compound was picked as an ECFP4 neighbour
       of a potent hit, so it is closer to the training data than training compounds are to each other.
    3. **Similarity does not transfer activity here.** The nearest neighbour's pEC50 is about as
       informative as a random training compound's; the similarity–activity curve falls, but slowly.
    4. **Much of the signal is global.** Lipophilicity and size explain a share of the variance that a
       bag of environments cannot express, and they account for part of what looks like cliffs. Some
       "cliffs" are simply assay floor.
    5. **Fingerprints are still useful, as one ingredient.** Prefer counts for regression, use more bits
       (or sparse features) if you want to interpret individual bits, switch on chirality when stereo
       matters, and put a few descriptors next to the bits. The challenge's best models went further with
       learned embeddings and structure-based features.
    6. **Interpret with care.** Map a bit back to *all* the environments it contains before telling a
       story about it, and read per-atom attributions across a matched pair.

    ---

    ### About this notebook

    * **Data:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0), training set plus the phase 1 and phase 2 unblinded test labels. Test-set design and the
      analysis of the hardest compounds are from OpenADMET's
      [challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)
      and [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/).
    * **References:** Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50, 742 (ECFP); Morgan, *J. Chem. Doc.*
      1965, 5, 107; Riniker & Landrum, *J. Cheminform.* 2013, 5, 43 (similarity maps).
    * **Widgets:** `ECFPMovie`, `MolGrid`, `MorganBitTiles` and `MorganExplorer` are custom anywidget
      components written for this notebook ([source](https://github.com/N283T/openadmet-marimo)).
      Molecules are drawn in the browser with RDKit.js; fingerprints, bit environments and collisions are
      computed with RDKit in Python. The movie's identifiers are real RDKit values.
    * **Related:** Pat Walters' [marimo-chem-utils](https://github.com/PatWalters/marimo_chem_utils) and
      [practical cheminformatics tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials);
      [mols2grid](https://github.com/cbouy/mols2grid), which inspired the grid.
    * **AI use:** Built together with Claude (Anthropic) as a coding assistant for the widgets, the movie
      and the notebook scaffolding. The question, analysis choices and interpretation come from my own
      PXR challenge work, and every number shown is computed live in this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
