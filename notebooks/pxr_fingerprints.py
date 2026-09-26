# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "molwidgets @ git+https://github.com/N283T/openadmet-marimo",
#     "polars>=1.30",
#     "numpy>=2",
#     "altair>=5.5",
#     "lightgbm>=4.5",
#     "scipy>=1.14",
#     "rdkit>=2025.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Similar, but not the same: Morgan fingerprints on PXR")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Similar, but not the same
    ### Why Morgan fingerprints stumble on PXR induction

    The **pregnane X receptor (PXR)** is a nuclear receptor that senses foreign molecules and switches on
    CYP3A4, P-gp and friends. A drug that activates PXR can speed up the metabolism of *other* drugs, so
    PXR induction is a classic drug–drug-interaction liability. OpenADMET measured it for >11,000
    compounds and ran a blind challenge asking for **pEC50** predictions on 513 new ones.

    In that challenge, models built on **Morgan (ECFP-like) fingerprints** — the workhorse of QSAR —
    were consistently among the weakest inputs. In my own entry (4th of 95), a Morgan-only
    LightGBM landed around 0.57 CV MAE while descriptor/embedding ensembles got below 0.40, and
    other teams reported the same ordering. This notebook asks **why**.

    Fingerprint models are nearest-neighbour machines at heart: they assume that *molecules which
    share substructures share activity*. We will test that assumption directly on the PXR data and
    look, bit by bit, at where it breaks.

    /// details | What you will build intuition for
    1. What a Morgan bit actually encodes (and what it throws away)
    2. How similar the test set really is to the training set
    3. How well structural similarity predicts PXR activity — the *similarity principle*, measured
    4. Three blind spots: stereochemistry, "how much" vs "whether", and whole-molecule properties
    5. A small model lab to test the ideas yourself
    ///

    The interactive pieces (molecule grid and fingerprint explorer) are custom
    [anywidget](https://anywidget.dev) components from the `molwidgets` package that ships with this
    notebook.
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
        MolGrid,
        MorganExplorer,
        census_for,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )

    return (
        Chem,
        Crippen,
        Descriptors,
        MolGrid,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The data, and some chemical hygiene

    We use the public OpenADMET PXR release on Hugging Face (CC-BY-4.0): the **training set** and
    the **full test set**, whose labels were unblinded after the challenge in two phases. Every
    SMILES is standardized the same way: keep the largest fragment, neutralize charges, keep
    stereochemistry, write canonical SMILES.
    """)
    return


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
        "stereo": int(data["smiles"].str.contains("@").sum()),
    }
    train = data.filter(pl.col("split") == "train")
    test = data.filter(pl.col("split") == "test")
    return data, hygiene, test, train


@app.cell(hide_code=True)
def _(hygiene, mo, test, train):
    mo.hstack(
        [
            mo.stat(f"{train.height:,}", label="training compounds"),
            mo.stat(f"{test.height:,}", label="test compounds (unblinded)"),
            mo.stat(str(hygiene["invalid"]), label="unparsable SMILES"),
            mo.stat(str(hygiene["changed"]), label="charged / salt forms neutralized"),
            mo.stat(str(hygiene["dup_within"]), label="duplicates after standardization"),
            mo.stat(str(hygiene["overlap"]), label="test compounds also in train"),
        ],
        widths="equal",
        gap=0.5,
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
    mo.hstack(
        [
            _chart,
            mo.md(
                f"""
    **Not all labels are equal.** pEC50 comes from a dose–response fit. Weak compounds never reach
    a plateau, so their EC50 is an extrapolation: below pEC50 3 the median confidence interval is
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["median 95% CI width"].item():.1f} log units wide**,
    versus ~0.2–0.3 for potent compounds. That is
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item()} training compounds** ({_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item() / train.height:.0%})
    whose exact value is mostly noise — keep this in mind when we look at "activity cliffs" later.
    """
            ),
        ],
        widths=[1, 1.4],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Anatomy of a Morgan fingerprint

    A Morgan fingerprint describes a molecule as a *bag of circular atom environments*. Every atom
    starts with an identifier built from its element, degree, hydrogen count, charge and ring
    membership (**radius 0**). Each iteration merges in the identifiers of the neighbours, so
    radius 1 sees an atom plus its bonded neighbours, radius 2 the neighbours of those, and so on.
    Every environment is hashed to a 32-bit integer, then **folded** into a fixed-length bit vector
    with `hash % n_bits`.

    Two consequences matter for everything that follows:

    * The fingerprint records **whether** an environment occurs, not where, how often (for bit
      vectors), or what the whole molecule looks like.
    * Folding means unrelated environments can land on the **same bit** — a *collision*.

    Pick any compound in the grid (filter by id, sort by pEC50, or type a SMARTS such as
    `c1ccncc1`), then click rows in the explorer to highlight the atoms that set each bit. For every
    bit the table also shows how many training compounds carry it, how many *different*
    substructures land on it (**# envs**) and how the mean pEC50 changes when it is on. Select a bit
    to see those colliding substructures side by side.
    """)
    return


@app.cell
def _(MolGrid, data, mo, pl):
    grid = mo.ui.anywidget(
        MolGrid(
            data.select("id", "smiles", "split", "pEC50").sort("pEC50", descending=True),
            subset=["split", "pEC50"],
            color_by="pEC50",
            selection_mode="single",
            selection=[data.sort("pEC50", descending=True)["id"][0]],
            page_size=12,
            cell_size=165,
        )
    )
    grid
    return (grid,)


@app.cell
def _(MorganExplorer, data, grid, mo, pl, train):
    _sel = grid.value.get("selection") or [train.sort("pEC50", descending=True)["id"][0]]
    _row = data.filter(pl.col("id") == _sel[0]).row(0, named=True)
    explorer = mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _row["id"],
                    "smiles": _row["smiles"],
                    "label": f"{_row['split']} · pEC50 {_row['pEC50']:.2f}",
                }
            ],
            reference=train["smiles"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    explorer
    return (explorer,)


@app.cell(hide_code=True)
def _(census_for, explorer, mo, np, train):
    _r, _n = explorer.value.get("radius", 2), explorer.value.get("n_bits", 2048)
    _census = census_for(train["smiles"].to_list(), _r, _n)
    _envs = _census.n_envs
    _bit = explorer.value.get("selected_bit", -1)
    _sel = (
        f"Selected bit **{_bit}** is set by **{len(explorer.value.get('bit_examples', []))}** different "
        "substructures in the training set (gallery above)."
        if _bit >= 0
        else "Click a row in the bit table: the selected bit flows back into Python and updates this text."
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{int(_envs.sum()):,}",
                        label=f"distinct radius ≤ {_r} environments in training set",
                    ),
                    mo.stat(f"{int((_envs > 0).sum()):,} / {_n:,}", label="bits used"),
                    mo.stat(f"{_envs[_envs > 0].mean():.1f}", label="environments per used bit"),
                    mo.stat(f"{(_envs > 1).mean():.0%}", label="bits shared by ≥ 2 environments"),
                ],
                widths="equal",
            ),
            mo.md(
                f"With {_n:,} bits, the training set's {int(_envs.sum()):,} environments have nowhere to go "
                "but on top of each other. Switch the explorer to 4096 bits or radius 1 and watch these "
                "numbers move. " + _sel
            ),
        ]
    ).callout(kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · How the test set was built

    The 513 test compounds are not a random sample. OpenADMET took the **63** compounds that were potent
    (EC50 ≤ 1 µM) *and* selective in the PXR-null counter-screen, and bought Enamine analogues with an
    **ECFP4 Tanimoto similarity > 0.4** to them
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)).
    It is a hit-expansion set, exactly how a medicinal chemist would follow up a screen.

    So two things are true by construction: every test compound has a close relative in the training
    data, and that relative is usually potent. Let's confirm it, and then ask the question that actually
    matters: **given a close, potent neighbour, can structural similarity tell which analogues keep the
    activity?**

    For every compound we find its **nearest neighbour (NN)** in the training set by Tanimoto
    similarity on Morgan fingerprints (radius 2, 2048 bits). For training compounds we exclude the
    compound itself.
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
    nn_train = S_train.argmax(1)
    nn_test = S_test.argmax(1)
    return S_test, S_train, X_test, X_train, nn_test, nn_train, y_test, y_train


@app.cell(hide_code=True)
def _(S_test, S_train, alt, mo, np, pl):
    _df = pl.concat(
        [
            pl.DataFrame({"NN Tanimoto": S_train.max(1), "set": "train → rest of train"}),
            pl.DataFrame({"NN Tanimoto": S_test.max(1), "set": "test → train"}),
        ]
    )
    _chart = (
        alt.Chart(_df)
        .transform_density(
            "NN Tanimoto", groupby=["set"], as_=["NN Tanimoto", "density"], extent=[0, 1]
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("NN Tanimoto:Q", title="Tanimoto similarity to nearest training neighbour"),
            y=alt.Y("density:Q", stack=None),
            color=alt.Color(
                "set:N",
                scale=alt.Scale(
                    range=["#3b82f6", "#f59e0b"], domain=["train → rest of train", "test → train"]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(height=200, width=640)
    )
    mo.vstack(
        [
            _chart,
            mo.hstack(
                [
                    mo.stat(
                        f"{np.median(S_train.max(1)):.2f}",
                        label="median NN similarity, train → train",
                    ),
                    mo.stat(
                        f"{np.median(S_test.max(1)):.2f}",
                        label="median NN similarity, test → train",
                    ),
                    mo.stat(
                        f"{(S_test.max(1) >= 0.5).mean():.0%}",
                        label="test compounds with a NN ≥ 0.5",
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                "As designed, the test set is **closer** to the training data than the training compounds "
                "are to each other. Being out of domain is not the problem here. The question is whether "
                "similarity carries the activity, and the most direct way to find out is to predict with it."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(1, 50, value=1, step=1, label="neighbours k", show_value=True)
    mo.md(
        f"""
    So let's use the most direct fingerprint model there is: predict each test compound's pEC50 as the
    mean of its **k most similar** training compounds. {k_slider}
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
def _(alt, mo, nn_test, pl, y_test, y_train):
    _df = pl.concat(
        [
            pl.DataFrame({"pEC50": y_train, "set": "all training compounds"}),
            pl.DataFrame(
                {
                    "pEC50": y_train[nn_test],
                    "set": "nearest training neighbour of each test compound",
                }
            ),
            pl.DataFrame({"pEC50": y_test, "set": "test compounds (truth)"}),
        ]
    )
    _chart = (
        alt.Chart(_df)
        .transform_density("pEC50", groupby=["set"], as_=["pEC50", "density"], extent=[1.5, 8])
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("pEC50:Q"),
            y=alt.Y("density:Q"),
            color=alt.Color(
                "set:N",
                scale=alt.Scale(
                    domain=[
                        "all training compounds",
                        "nearest training neighbour of each test compound",
                        "test compounds (truth)",
                    ],
                    range=["#9ca3af", "#3b82f6", "#f59e0b"],
                ),
                legend=alt.Legend(orient="top", title=None, direction="vertical"),
            ),
        )
        .properties(height=220, width=380)
    )
    _nn = y_train[nn_test]
    mo.vstack(
        [
            mo.md("### Who are the neighbours?"),
            mo.hstack(
                [
                    _chart,
                    mo.md(
                        f"""
    The vertical band in the scatter above is the test-set design showing through. The nearest
    neighbours of test compounds are overwhelmingly **potent** training compounds: their mean pEC50 is
    **{_nn.mean():.2f}**, against **{y_train.mean():.2f}** for the training set as a whole, and
    **{(_nn >= 5.5).mean():.0%}** of them have pEC50 ≥ 5.5. That is expected, since the analogues were
    chosen around the hits.

    What matters is that the test compounds themselves do not follow. Their potency spreads from ~2 to ~7
    (mean **{y_test.mean():.2f}**): an **SAR exploration around the hits**, where every compound has a
    close, potent relative *by construction* and the real question is which small changes keep the
    activity. That is exactly the question a similarity-based model cannot answer, and the one the
    OpenADMET [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
    identifies as the shared failure point of every top team.
    """
                    ),
                ],
                widths=[1, 1.1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The similarity principle, measured

    The *similar property principle* says that the activity difference between two molecules should
    shrink as their structural similarity grows. We can measure it directly on all
    ~8.5 million training pairs: bin the pairs by Tanimoto similarity and look at the typical
    |Δ pEC50| in each bin.
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
    The dashed line is a random pair (**{random_pair_dy:.2f}** log units). The curve does go down —
    similarity is not useless — but slowly. Even among pairs with Tanimoto ≥ 0.5, the typical
    difference is **{(_hi["mean_dy"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.2f}** log units and
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
    ### Browse the cliffs

    An **activity cliff** is a pair that looks alike but behaves differently. Choose what "alike" and
    "differently" mean, pick a pair from the table, and compare the two fingerprints bit by bit.
    Bits present in only one molecule are exactly what a fingerprint model *can* use to explain the
    difference — often they are a handful of generic environments.

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
    train_logp = _logp
    return cliffs, train_logp


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
def _(MorganExplorer, cliff_table, mo, pl, train):
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
      has a CI wider than 1.5 log units — part of the "cliff" is assay noise at the bottom of the scale.
    * **Which one is greasier?** In {_lip} of {_n} pairs ({_lip / max(_n, 1):.0%}) the more potent
      compound also has the higher calculated logP. A property of the *whole molecule* is doing work that
      no single bit can express. That is the next section.
    """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · What the bits cannot see

    ### 5a · Different molecules, identical fingerprints

    If two different molecules produce exactly the same bit vector, any model built on those bits
    **must** predict the same value for both. Let's find every such group in the training set and ask
    *why* the fingerprint cannot tell them apart.
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
    potency inside a group differs by up to **{_summary["range"].max():.2f}** log units. There are two
    different reasons:

    * **Stereochemistry** ({_n_stereo} groups). RDKit Morgan fingerprints ignore chirality and
      double-bond geometry unless you pass `includeChirality=True`. Look closely: in these groups one
      record has its stereocentre or double bond *specified* and the other does not — a single
      enantiomer next to what is most likely the racemate (lansoprazole / dexlansoprazole,
      bupivacaine / levobupivacaine), or the same drug registered twice with and without E/Z labels
      (rifampicin, the textbook PXR agonist). The lansoprazole pair differs by ~1 log unit, which is a
      real question for a chemist and invisible to the fingerprint. Switching on chirality separates
      {_n_split} of these {_n_stereo} groups; the sulfoxide stereocentre is not picked up either way.
    * **Ring size / chain length** ({len(_twins) - _n_stereo} groups). Cyclohexyl vs cycloheptyl amine,
      azepane vs azocane, nonanoic vs palmitic acid: every atom sees the same neighbourhood within
      radius 2, so the *set* of environments is identical — only *how many times* each occurs differs.
      A bit vector
      stores presence, not counts; a count fingerprint separates these (see 5c).
    """
            ),
            twin_grid,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b · Whole-molecule properties

    PXR's ligand-binding pocket is large, flexible and hydrophobic; it is famous for accepting
    very different scaffolds. If binding is driven by *how greasy and how big* a molecule is rather
    than by a specific pharmacophore, then a bag of local substructures is the wrong language.
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
def _(DESCRIPTORS, D_train, mo):
    prop_pick = mo.ui.dropdown(list(DESCRIPTORS), value="MolLogP", label="descriptor")
    prop_pick
    return (prop_pick,)


@app.cell(hide_code=True)
def _(DESCRIPTORS, D_train, alt, mo, np, pl, prop_pick, spearmanr, y_train):
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
    A single number — Crippen logP — ranks training compounds about as well as the fingerprint kNN
    ranked the test set. And *within* the similar pairs (Tanimoto ≥ 0.5), the logP difference
    tracks the potency difference (Spearman ρ = **{_rho:.2f}**): part of what looks like a cliff to
    the fingerprint is a smooth lipophilicity trend. A folded bit vector has no axis for "a bit more
    lipophilic"; adding a methyl either flips a bit or it doesn't.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5c · "Whether" versus "how much"

    A binary bit says an environment is present. A **count** fingerprint records how many times — and
    counts summed over a molecule are a crude measure of size and lipophilicity. If the hypothesis
    above is right, counts should recover some of the lost signal. Test it in the lab below.

    ## 6 · Model lab

    Train a LightGBM model (the same kind of baseline as OpenADMET's challenge tutorial) on the training
    set and score it on the 513 unblinded test compounds. The three default configurations are
    pre-computed; change the settings and press **Train** to add your own row to the scoreboard.
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
def _(all_runs, alt, mo, nn_sim_test, np, pl, run_pick, test, y_test):
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
                "predicted too potent, potent compounds too weak — and fingerprint-only models compress the most."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things worth trying in the lab:

    * **Bits → counts** at the same radius and length. Counts add "how much" information for free.
    * **256 → 8192 bits.** Fewer collisions help a little. Collisions are real (see the explorer),
      but they are not the main story.
    * **Include chirality.** Only a handful of compounds change, so the score barely moves — but for
      the stereoisomer pairs in 5a it is the difference between "impossible" and "possible".
    * **Morgan + 14 descriptors.** Fourteen simple whole-molecule numbers repair much of what 2048
      bits miss.

    ## 7 · Take-aways

    1. **The PXR test set is in domain by design.** It is a hit-expansion set: every test compound was
       picked as an ECFP4 neighbour of a potent hit, so it is closer to the training data than training
       compounds are to each other, and its nearest neighbour is usually potent.
    2. **Similarity does not transfer activity here.** The nearest training neighbour's pEC50 is about
       as informative as a random training compound's. The similarity–activity curve falls, but slowly.
    3. **Much of the signal is global.** Lipophilicity and size explain a share of the variance that a
       bag of local environments cannot express, and they account for part of what looks like cliffs.
    4. **Some "cliffs" are assay floor.** Weak compounds carry very wide confidence intervals; treat
       differences at the bottom of the scale with suspicion.
    5. **Fingerprints are still useful — as one ingredient.** Counts, larger folds, chirality and a few
       descriptors each recover part of the gap; the challenge's best models went further with learned
       embeddings and structure-based features.

    ---

    ### About this notebook

    * **Data:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0), training set plus the phase 1 and phase 2 unblinded test labels.
    * **Widgets:** `MolGrid` and `MorganExplorer` are custom anywidget components written for this
      notebook ([source](https://github.com/N283T/openadmet-marimo)). Molecules are drawn in the browser
      with RDKit.js; fingerprints, bit environments and collisions are computed with RDKit in Python.
    * **Related:** Pat Walters' [marimo-chem-utils](https://github.com/PatWalters/marimo_chem_utils) and
      [practical cheminformatics tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials);
      [mols2grid](https://github.com/cbouy/mols2grid), which inspired the grid.
    * **AI use:** Built together with Claude (Anthropic) as a coding assistant for the widgets and
      notebook scaffolding. The question, analysis choices and interpretation come from my own
      PXR challenge work, and every number shown is computed live in this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
