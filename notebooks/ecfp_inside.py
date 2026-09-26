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
app = marimo.App(width="medium", app_title="Inside ECFP4: a hands-on guide to Morgan fingerprints")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inside ECFP4
    ### A hands-on guide to Morgan fingerprints — what they encode, what they lose, and what that means for your model

    If you have trained a QSAR model, you have probably typed something like

    ```python
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    ```

    and moved on. ECFP4 is the default molecular representation in cheminformatics: it powers
    similarity search, clustering, library design and countless ML baselines. It even defined the
    test set of the OpenADMET PXR challenge: the 513 test compounds were bought from Enamine
    because their **ECFP4 Tanimoto similarity to a potent, selective PXR hit was above 0.4**
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)).

    Yet few people could say what bit 1380 of that vector *means*, or how many different
    substructures share it. This notebook opens the box:

    1. **The algorithm**, one iteration at a time, with a from-scratch implementation you can read
    2. **Folding and collisions** — the step everybody forgets
    3. **Blind spots** — counts, stereochemistry, ring size
    4. **Similarity** — what "Tanimoto > 0.4" does and does not promise
    5. **ECFP inside a model** — LightGBM feature importance and per-atom attributions, and why
       collisions make them hard to read
    6. **A cheat sheet** of strengths, weaknesses and sensible defaults

    Examples come from the OpenADMET PXR induction dataset. The interactive pieces are custom
    [anywidget](https://anywidget.dev) components from the `molwidgets` package written for this notebook.

    /// admonition | Names you will see
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) and the **Morgan fingerprint**
    (RDKit's name, after Morgan's 1965 canonicalization algorithm) are the same idea. The number in
    ECFP*n* is the *diameter*: **ECFP4 = Morgan radius 2**, ECFP6 = radius 3. **FCFP** uses
    pharmacophoric atom features (donor, acceptor, aromatic, …) instead of element-level invariants.
    ///
    """)
    return


@app.cell
def _():
    import inspect

    import altair as alt
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from scipy.stats import spearmanr

    alt.data_transformers.disable_max_rows()

    from molwidgets import (
        ECFPStepper,
        MolGrid,
        MorganExplorer,
        census_for,
        ecfp_trace,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )
    from molwidgets import ecfp as ecfp_module

    return (
        Chem,
        ECFPStepper,
        MolGrid,
        MorganExplorer,
        alt,
        census_for,
        ecfp_module,
        ecfp_trace,
        fingerprint_matrix,
        inspect,
        lgb,
        np,
        pl,
        rdFingerprintGenerator,
        spearmanr,
        standardize_smiles,
        tanimoto_matrix,
    )


@app.cell
def _(pl, standardize_smiles):
    from pathlib import Path

    HF = "https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main/"
    FILES = {
        "train": "pxr-challenge_TRAIN.csv",
        "test_p1": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "test_p2": "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
    }

    def _read(name: str) -> pl.DataFrame:
        local = Path("data") / FILES[name]  # use a local copy when present, else download
        return pl.read_csv(local if local.exists() else HF + FILES[name])

    data = pl.concat(
        [
            _read(key)
            .select(
                pl.col("Molecule Name").alias("id"), pl.col("SMILES").alias("smiles_raw"), "pEC50"
            )
            .with_columns(pl.lit("train" if key == "train" else "test").alias("split"))
            for key in FILES
        ]
    ).with_columns(
        # largest fragment, neutralized, canonical; stereo kept
        pl.col("smiles_raw").map_elements(standardize_smiles, return_dtype=pl.Utf8).alias("smiles")
    )
    train = data.filter(pl.col("split") == "train")
    test = data.filter(pl.col("split") == "test")
    return data, test, train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The algorithm, one iteration at a time

    ECFP builds a molecule's description from the inside out.

    **Iteration 0 — atom invariants.** Each heavy atom gets an integer identifier by hashing a small
    tuple of properties. RDKit uses six: atomic number, degree (including Hs), total hydrogen count,
    formal charge, isotope and whether the atom is in a ring. Two carbons with the same six numbers get
    the same identifier, wherever they sit in the molecule.

    **Iteration *r*.** Each atom's new identifier is the hash of its own previous identifier plus the
    sorted list of *(bond order, neighbour identifier)* pairs. After one iteration an identifier
    describes the atom and its neighbours (a radius-1 environment); after two, the neighbours'
    neighbours; and so on.

    **Collecting features.** Every identifier produced along the way is a feature — *unless* it
    describes exactly the same set of bonds as an environment already collected. Those duplicates
    are dropped, as are atoms whose environment stopped growing (it already covers everything it can
    reach). The result is a *set* of integers: the unfolded ECFP.

    Choose a molecule and step through it. Watch symmetric atoms keep the same colour, click an atom
    to see exactly what was hashed, and switch radius to see environments grow and duplicates vanish.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    EXAMPLES = {
        "paracetamol (small, symmetric ring)": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · potent PXR agonist (pEC50 5.95)": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · its pyridine analogue (pEC50 2.06)": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",
        "cyclohexylamine": "NC1CCCCC1",
        "cycloheptylamine": "NC1CCCCCC1",
    }
    example_pick = mo.ui.dropdown(
        EXAMPLES, value="paracetamol (small, symmetric ring)", label="molecule"
    )
    custom_smiles = mo.ui.text(placeholder="…or paste a SMILES", label="", full_width=False)
    mo.hstack([example_pick, custom_smiles], justify="start", gap=1)
    return custom_smiles, example_pick


@app.cell
def _(Chem, ECFPStepper, custom_smiles, example_pick, mo):
    _smi = custom_smiles.value.strip() or example_pick.value
    if Chem.MolFromSmiles(_smi) is None:
        stepper = mo.md(f"`{_smi}` is not a valid SMILES.").callout(kind="warn")
    else:
        stepper = mo.ui.anywidget(ECFPStepper(_smi, max_radius=3, radius=1, n_bits=64))
    stepper
    return (stepper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Things to notice:

    * **Iteration 0 is coarse.** In paracetamol, the four aromatic CH carbons share one identifier. ECFP
      starts from *chemistry-free* labels: nothing about pharmacophores, just element-level counts.
    * **Symmetry survives.** Atoms related by symmetry keep equal identifiers at every radius, so
      they contribute a single feature: the fingerprint records that an environment occurs, not how
      many atoms share it.
    * **Duplicates are real.** At higher radius, neighbouring atoms often end up covering the same bonds;
      only one of them is kept. Small molecules stop growing after one or two iterations.
    * **The fold panel is where information is lost.** Switch to 16 or 64 bits and red cells appear:
      different environments land on the same bit. We will measure how often that happens at 2048.

    ### The whole algorithm fits on one screen

    The stepper is driven by the function below: a plain-Python ECFP written for readability. It uses
    `blake2b` instead of RDKit's internal hash, so identifiers are different numbers, but the logic
    — invariants, neighbour ordering, duplicate removal — follows RDKit.
    """)
    return


@app.cell(hide_code=True)
def _(ecfp_module, inspect, mo):
    mo.accordion(
        {
            "Show the implementation (`molwidgets/ecfp.py`)": mo.md(
                "```python\n"
                + inspect.getsource(ecfp_module.atom_invariant)
                + "\n\n"
                + inspect.getsource(ecfp_module.ecfp_trace)
                + "```"
            )
        }
    )
    return


@app.cell
def _(Chem, ecfp_trace, rdFingerprintGenerator, train):
    # Does the readable version produce the same *number* of distinct features as RDKit?
    def _count_matches(radius: int) -> int:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
        return sum(
            len(ecfp_trace(s, radius).identifiers())
            == len(gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements())
            for s in train["smiles"]
        )

    validation = {r: _count_matches(r) for r in (1, 2)}
    return (validation,)


@app.cell(hide_code=True)
def _(mo, train, validation):
    mo.md(
        f"""
    **Check against RDKit.** For every one of the {train.height:,} training molecules, the readable
    implementation yields the same number of distinct features as RDKit's
    `GetSparseCountFingerprint`: **{validation[1]:,} / {train.height:,}** at radius 1 and
    **{validation[2]:,} / {train.height:,}** at radius 2. (When two atoms cover identical bonds, *which*
    one is kept depends on the ordering of hash values, so the chosen centre atom can differ; the
    features themselves do not.)
    """
    ).callout(kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Folding: from identifiers to 2048 bits

    The unfolded fingerprint is a set of 32-bit integers — about four billion possible values.
    Machine-learning libraries want a fixed-length vector, so RDKit **folds** each identifier onto a
    bit with `identifier % n_bits`. That is a hash table with no collision handling. Two unrelated
    environments that land on the same bit become indistinguishable, forever.

    How likely is that? It is the birthday problem. A molecule with *k* distinct features folded into
    *n* bits avoids every collision with probability

    $$P(\text{no collision}) = \prod_{i=0}^{k-1}\left(1 - \frac{i}{n}\right) \approx e^{-k(k-1)/2n}.$$

    A drug-like molecule has ~40–60 ECFP4 features. With *k* = 50 and *n* = 2048 that is
    $e^{-0.6} \approx 0.55$: **nearly half of all molecules have at least one collision inside
    themselves**, before we even compare molecules.
    """)
    return


@app.cell
def _(Chem, np, rdFingerprintGenerator, train):
    # Distinct ECFP4 identifiers per training molecule (unfolded).
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    unfolded = [
        np.fromiter(
            _gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements().keys(),
            dtype=np.int64,
        )
        for s in train["smiles"]
    ]
    n_features = np.array([len(u) for u in unfolded])
    return n_features, unfolded


@app.cell(hide_code=True)
def _(mo):
    fold_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    fold_pick = mo.ui.slider(steps=fold_sizes, value=2048, label="n_bits", show_value=True)
    fold_pick
    return fold_pick, fold_sizes


@app.cell(hide_code=True)
def _(alt, fold_pick, fold_sizes, mo, n_features, np, pl, unfolded):
    def _with_collision(n: int) -> float:
        return float(np.mean([len(np.unique(u % n)) < len(u) for u in unfolded]))

    def _theory(n: int) -> float:
        k = n_features[:, None]
        i = np.arange(n_features.max())[None, :]
        p_none = np.prod(np.where(i < k, 1 - i / n, 1.0), axis=1)
        return float(1 - p_none.mean())

    _rows = [
        {"n_bits": n, "share of molecules": v, "source": src}
        for n in fold_sizes
        for v, src in ((_with_collision(n), "PXR training set"), (_theory(n), "birthday formula"))
    ]
    _df = pl.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "n_bits:Q", scale=alt.Scale(type="log", base=2), axis=alt.Axis(values=fold_sizes)
            ),
            y=alt.Y(
                "share of molecules:Q",
                title="molecules with ≥ 1 internal collision",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(range=["#9ca3af", "#d6336c"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash("source:N", legend=None),
        )
        .properties(width=380, height=240)
    )
    _rule = (
        alt.Chart(pl.DataFrame({"n_bits": [fold_pick.value]}))
        .mark_rule(color="#1f2328")
        .encode(x="n_bits:Q")
    )
    _here = _with_collision(fold_pick.value)
    mo.hstack(
        [
            _chart + _rule,
            mo.vstack(
                [
                    mo.stat(
                        f"{np.median(n_features):.0f}",
                        label="median distinct ECFP4 features per molecule",
                    ),
                    mo.stat(
                        f"{_here:.0%}",
                        label=f"molecules with an internal collision at {fold_pick.value} bits",
                    ),
                    mo.md(
                        "The simple formula tracks the real data closely. Doubling the length roughly "
                        "halves the collision rate; you need ~16k bits before collisions become rare."
                    ),
                ]
            ),
        ],
        widths=[1.2, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Across a dataset, every bit is shared

    Inside one molecule collisions are occasional. Across a whole dataset they are universal: the
    training set contains tens of thousands of distinct environments and 2048 bits to put them in.
    The explorer below shows, for any molecule, which environments set each bit and **how many
    different substructures from the whole training set land on that bit (# envs)**. Click a row to
    see them side by side.
    """)
    return


@app.cell
def _(census_for, mo, np, train):
    _c = census_for(train["smiles"].to_list(), 2, 2048)
    _envs = _c.n_envs
    mo.hstack(
        [
            mo.stat(
                f"{int(_envs.sum()):,}", label="distinct ECFP4 environments in 4,139 molecules"
            ),
            mo.stat(f"{_envs[_envs > 0].mean():.1f}", label="environments per bit at 2048 bits"),
            mo.stat(f"{int(_envs.max())}", label="most crowded bit"),
            mo.stat(f"{(_envs > 1).mean():.0%}", label="bits shared by ≥ 2 environments"),
        ],
        widths="equal",
    )
    return


@app.cell
def _(MorganExplorer, mo, train):
    _i = train["pEC50"].arg_max()
    fold_explorer = mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": train["id"][_i],
                    "smiles": train["smiles"][_i],
                    "label": f"pEC50 {train['pEC50'][_i]:.2f}",
                }
            ],
            reference=train["smiles"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    fold_explorer
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Blind spots

    Folding loses information by accident. Some information is never collected in the first place.
    Each row below is a pair of *different* molecules; compare their Tanimoto similarity under four
    common settings.
    """)
    return


@app.cell
def _(Chem, pl, rdFingerprintGenerator):
    PAIRS = {
        "cyclohexyl- vs cycloheptylamine": ("NC1CCCCC1", "NC1CCCCCC1"),
        "nonanoic vs palmitic acid": ("CCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCC(=O)O"),
        "(R)- vs (S)-ibuprofen": (
            "C[C@@H](C(=O)O)c1ccc(CC(C)C)cc1",
            "C[C@H](C(=O)O)c1ccc(CC(C)C)cc1",
        ),
        "dexlansoprazole vs lansoprazole (stereo unspecified)": (
            "Cc1c(OCC(F)(F)F)ccnc1C[S@@](=O)c1nc2ccccc2[nH]1",
            "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1",
        ),
        "biphenyl vs terphenyl": ("c1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"),
        "benzene → pyridine (PXR pair from §5)": (
            "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
            "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",
        ),
    }

    def _tani(a, b, gen):
        x, y = (gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(bool) for s in (a, b))
        return float((x & y).sum() / (x | y).sum())

    def _tani_counts(a, b):
        g = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        x, y = (
            g.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements() for s in (a, b)
        )
        keys = set(x) | set(y)
        num = sum(min(x.get(k, 0), y.get(k, 0)) for k in keys)
        return num / sum(max(x.get(k, 0), y.get(k, 0)) for k in keys)

    _ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _ecfp_chiral = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    _fcfp = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    blind_spots = pl.DataFrame(
        [
            {
                "pair": name,
                "ECFP4 bits": round(_tani(a, b, _ecfp), 3),
                "ECFP4 counts": round(_tani_counts(a, b), 3),
                "bits + chirality": round(_tani(a, b, _ecfp_chiral), 3),
                "FCFP4 bits": round(_tani(a, b, _fcfp), 3),
            }
            for name, (a, b) in PAIRS.items()
        ]
    )
    blind_spots
    return (blind_spots,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **"How many" is invisible to bits.** Ring size and chain length beyond the radius produce the
      *same set* of environments, only repeated more often: Tanimoto 1.0 on bits, clearly below 1 on
      counts. Size is a strong driver of many ADMET endpoints (PXR's pocket is large and greasy), so
      this is not a corner case.
    * **Stereochemistry is off by default.** Enantiomers are identical unless you pass
      `includeChirality=True` — and even then some stereocentres (the lansoprazole sulfoxide) are
      not captured. The PXR training set contains exactly such pairs, with potency differences near a
      log unit.
    * **FCFP trades detail for pharmacophores.** It merges atoms with the same role (e.g. any
      aromatic carbon), which helps scaffold hopping but can hide changes that matter. Check the
      benzene → pyridine pair: one aromatic CH becomes an N.

    ## 4 · Similarity: what "Tanimoto > 0.4" promises

    Tanimoto similarity on ECFP4 is the share of set bits two molecules have in common. It is the
    workhorse of analogue searching, and it is how the PXR test set was built. Below, every test
    compound is placed by its similarity to its nearest **potent** training compound (pEC50 ≥ 6)
    against its own measured potency.
    """)
    return


@app.cell(hide_code=True)
def _(alt, fingerprint_matrix, mo, np, pl, tanimoto_matrix, test, train):
    _potent = train.filter(pl.col("pEC50") >= 6)
    _S = tanimoto_matrix(
        fingerprint_matrix(test["smiles"].to_list(), 2, 2048),
        fingerprint_matrix(_potent["smiles"].to_list(), 2, 2048),
    )
    _best = _S.argmax(1)
    _df = pl.DataFrame(
        {
            "id": test["id"],
            "Tanimoto to nearest potent hit": _S.max(1),
            "test pEC50": test["pEC50"],
            "hit pEC50": _potent["pEC50"].to_numpy()[_best],
        }
    )
    _chart = (
        alt.Chart(_df)
        .mark_circle(size=28, opacity=0.6, color="#3b82f6")
        .encode(
            x=alt.X("Tanimoto to nearest potent hit:Q", scale=alt.Scale(domain=[0.2, 1])),
            y=alt.Y("test pEC50:Q", scale=alt.Scale(domain=[1.5, 7.5])),
            tooltip=[
                "id",
                alt.Tooltip("Tanimoto to nearest potent hit:Q", format=".2f"),
                "test pEC50",
                "hit pEC50",
            ],
        )
        .properties(width=380, height=260)
    )
    _hit_band = (
        alt.Chart(pl.DataFrame({"y": [6.0]}))
        .mark_rule(strokeDash=[4, 4], color="#d6336c")
        .encode(y="y:Q")
    )
    _hi = _df.filter(pl.col("Tanimoto to nearest potent hit") >= 0.5)
    mo.hstack(
        [
            _chart + _hit_band,
            mo.md(
                f"""
    The test compounds were chosen as ECFP4 neighbours (> 0.4) of **63** hits that were both potent
    and selective in the counter-screen; after re-measurement the launch post counts 46 such hits.
    Here we simply use all {_potent.height} training compounds with pEC50 ≥ 6, and RDKit's ECFP4 is not
    bit-identical to every vendor's implementation, so **{(_df["Tanimoto to nearest potent hit"] > 0.4).mean():.0%}**
    of test compounds clear 0.4 in this plot. Either way, their own potency spans the whole assay range (dashed line:
    pEC50 6). Among the {_hi.height} compounds with Tanimoto ≥ 0.5 to a current hit, only
    **{(_hi["test pEC50"] >= 6).mean():.0%}** are hits themselves and
    **{(_hi["test pEC50"] < 4).mean():.0%}** are essentially inactive (pEC50 < 4).

    That is not a failure of ECFP. Similarity search is *supposed* to return the neighbourhood of a hit,
    and medicinal chemists buy analogues precisely to find out which ones keep the activity. What ECFP4
    Tanimoto cannot tell you is **which** of those small changes matter — and a model built on the same
    bits inherits the same blindness. The OpenADMET
    [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
    found that the hardest test compounds for all top teams were exactly these activity cliffs.
    """
            ),
        ],
        widths=[1.1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · ECFP inside a model

    Let's train the model most people would train first — **LightGBM on 2048-bit ECFP4**, the same
    baseline OpenADMET's challenge tutorial uses — and then ask it what it learned. Two standard tools:

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
def _(fingerprint_matrix, lgb, model_bits, mo, np, spearmanr, test, train):
    N_BITS = model_bits.value
    X_train = fingerprint_matrix(train["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    X_test = fingerprint_matrix(test["smiles"].to_list(), 2, N_BITS).astype(np.float32)
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
        ).fit(X_train, train["pEC50"].to_numpy())
    pred_test = model.predict(X_test)
    model_scores = {
        "MAE": float(np.abs(pred_test - test["pEC50"].to_numpy()).mean()),
        "rho": float(spearmanr(pred_test, test["pEC50"].to_numpy())[0]),
    }
    return N_BITS, X_test, model, model_scores, pred_test


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
    Switch to 8192 bits above and the bags shrink to a few members each — while the accuracy barely moves.
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
    top_bit_table = mo.ui.table(
        top_bits,
        selection="single",
        initial_selection=[0],
        page_size=5,
        label="Pick a top bit to see what is inside it",
    )
    top_bit_table
    return (top_bit_table,)


@app.cell
def _(Chem, MolGrid, N_BITS, census_for, mo, pl, top_bit_table, train):
    from molwidgets.chem import env_atoms_bonds

    _bit = (
        int(top_bit_table.value["bit"][0])
        if len(top_bit_table.value)
        else int(top_bit_table.data["bit"][0])
    )
    _smiles = train["smiles"].to_list()
    _ids = train["id"].to_list()
    _examples = census_for(_smiles, 2, N_BITS).examples.get(_bit, [])
    _rows, _hl = [], {}
    for _k, _ex in enumerate(_examples):
        _key = f"{_k + 1}. {_ids[_ex['mol_index']]}"
        _atoms, _bonds = env_atoms_bonds(
            Chem.MolFromSmiles(_smiles[_ex["mol_index"]]), _ex["center"], _ex["radius"]
        )
        _rows.append(
            {
                "env": _key,
                "smiles": _smiles[_ex["mol_index"]],
                "substructure": _ex["smiles"],
                "radius": _ex["radius"],
                "molecules": _ex["count"],
            }
        )
        _hl[_key] = {"atoms": _atoms, "bonds": _bonds}
    mo.vstack(
        [
            mo.md(
                f"**Bit {_bit}** is set by **{len(_examples)}** distinct environments in the training set. "
                "Each card shows one of them (highlighted) in an example molecule; *molecules* is how many "
                "training compounds contain that environment."
            ),
            MolGrid(
                pl.DataFrame(_rows),
                id_col="env",
                subset=["substructure", "radius", "molecules"],
                highlights=_hl,
                sort_by="molecules",
                page_size=8,
                cell_size=170,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading one prediction

    Now zoom into single molecules. The pair below is the second-hardest test compound of the whole
    challenge, **OADMET-0006254** (pEC50 2.06), next to its nearest training neighbour
    **OADMET-0002810** (pEC50 5.95). The only change is a benzene CH → pyridine N. According to the
    OpenADMET analysis, a related co-crystal structure shows that nitrogen H-bonding to SER247, which
    likely changes how the ligand sits in the pocket. Every Tier-1 team over-predicted this compound.

    Atom colours show the model's TreeSHAP attribution (red raises the predicted pEC50, blue lowers it).
    Click a bit to see its environments instead; the table's **SHAP** columns show each bit's
    contribution for A and B. You can also pick another test compound and its nearest training
    neighbour from the list.
    """)
    return


@app.cell
def _(fingerprint_matrix, np, pl, pred_test, tanimoto_matrix, test, train):
    _S = tanimoto_matrix(
        fingerprint_matrix(test["smiles"].to_list(), 2, 2048),
        fingerprint_matrix(train["smiles"].to_list(), 2, 2048),
    )
    _nn = _S.argmax(1)
    cliff_pairs = (
        pl.DataFrame(
            {
                "test id": test["id"],
                "test pEC50": test["pEC50"],
                "predicted": pred_test.round(2),
                "NN id": train["id"].to_numpy()[_nn],
                "NN pEC50": train["pEC50"].to_numpy()[_nn],
                "Tanimoto": _S.max(1).astype(float).round(3),
            }
        )
        .with_columns((pl.col("predicted") - pl.col("test pEC50")).abs().round(2).alias("|error|"))
        .sort("|error|", descending=True)
    )
    _default = (
        cliff_pairs["test id"].to_list().index("OADMET-0006254")
        if "OADMET-0006254" in cliff_pairs["test id"]
        else 0
    )
    pair_default = int(np.int64(_default))
    return cliff_pairs, pair_default


@app.cell(hide_code=True)
def _(cliff_pairs, mo, pair_default):
    pair_table = mo.ui.table(
        cliff_pairs,
        selection="single",
        initial_selection=[pair_default],
        page_size=5,
        label="Test compounds and their nearest training neighbour, largest model error first",
    )
    pair_table
    return (pair_table,)


@app.cell
def _(MorganExplorer, N_BITS, data, fingerprint_matrix, mo, model, np, pair_table, pl, train):
    _row = (
        pair_table.value.row(0, named=True)
        if len(pair_table.value)
        else pair_table.data.row(0, named=True)
    )
    _smi = dict(zip(data["id"], data["smiles"]))
    _ids = [_row["test id"], _row["NN id"]]
    _X = fingerprint_matrix([_smi[i] for i in _ids], 2, N_BITS).astype(np.float32)
    pair_X = _X
    pair_contrib = model.predict(_X, pred_contrib=True)  # last column = expected value
    _contrib = pair_contrib
    _pred = _contrib.sum(1)
    _maps = [{str(b): float(c[b]) for b in np.flatnonzero(x)} for c, x in zip(_contrib, _X)]
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
            pair_note=f"baseline (mean prediction) {_contrib[0, -1]:.2f}",
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
    the **{int(_shared.sum())}** bits they share for **{_diff[_shared].sum():+.2f}** (trees split on combinations,
    so a shared bit can matter more in one context than the other), and bits absent from both for
    **{_diff[_neither].sum():+.2f}**. The atom map shows only bits that are *present*; absences matter to trees
    too, but they cannot be drawn on atoms.

    What the attribution map can and cannot tell you:

    * **The change is visible only as a few bits.** A single atom swap flips a handful of environments
      (the *only A* / *only B* rows). The model sees them — but in the training data each of those bits
      also stands for many other substructures (**# envs**), so its learned effect is an average over
      everything folded into it. In the benzene → pyridine pair, the new nitrogen barely lights up.
    * **Similar fingerprints give similar predictions.** Most attribution sits on shared bits. The model is
      doing exactly what ECFP asks it to do; the activity cliff is simply not in the representation.
    * **An attribution on a colliding bit is ambiguous.** A red atom means "the model likes this bit", not
      "the model likes this group", when the bit holds a dozen substructures. Retrain at 8192 bits and compare.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Cheat sheet

    **Strengths**

    * Fast, deterministic, needs no training data.
    * Describes local substructure well; excellent for similarity search, clustering and analogue retrieval.
    * A strong baseline for SAR within a congeneric series.

    **Weaknesses**

    * Folding collisions: roughly half of drug-like molecules collide with themselves at 2048 bits, and
      every bit is shared across a dataset.
    * Bit vectors record presence, not counts: ring size, chain length and repeated groups disappear.
    * Stereochemistry is ignored by default.
    * No global shape, size or physicochemistry; every environment counts the same.
    * Hashed bits are hard to interpret.

    **Defaults worth changing**

    * Prefer **count** fingerprints for regression.
    * Use ≥ 4096 bits, or unfolded/sparse features, if you want to *interpret* individual bits.
    * Pass `includeChirality=True` when stereo matters.
    * Add a few whole-molecule descriptors (logP, size, TPSA) next to the bits.

    **When interpreting a model**

    * Map each bit back to *all* the environments it contains before telling a story about it.
    * Read per-atom attributions across a matched pair, not for a single molecule in isolation.

    ---

    ### About this notebook

    * **Data:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0). Test-set design and the analysis of the hardest compounds are from OpenADMET's
      [challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/),
      [launch post](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/) and
      [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/).
    * **References:** Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50, 742 (ECFP); Morgan, *J. Chem. Doc.* 1965, 5, 107;
      Riniker & Landrum, *J. Cheminform.* 2013, 5, 43 (similarity maps).
    * **Widgets:** `ECFPStepper`, `MorganExplorer` and `MolGrid` are custom anywidget components written for
      this notebook ([source](https://github.com/N283T/openadmet-marimo)); `molwidgets.ecfp` is the readable
      ECFP implementation shown above.
    * **Companion notebook:** *Similar, but not the same* digs into why fingerprint models struggle on this dataset.
    * **AI use:** Built together with Claude (Anthropic) as a coding assistant for the widgets and notebook
      scaffolding. The questions, analysis choices and interpretation come from my own PXR challenge work;
      every number is computed live in this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
