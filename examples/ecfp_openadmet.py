# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "chemari @ git+https://github.com/N283T/chemari@v0.1.0",
#     "polars>=1.30",
#     "numpy>=2",
#     "altair>=5.5",
#     "duckdb>=1.1",
#     "pyarrow>=18",
#     "sqlglot>=26",
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

    When you feed molecules to a machine-learning model, you have probably written something like this:

    ```python
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    ```

    That is **ECFP4**<sup><a href="#ref-1">1</a></sup>, the most widely used molecular representation in cheminformatics. It is quick to compute with almost nothing to tune, and it works well for a lot of similarity search and QSAR. When a new method is benchmarked, ECFP4 is usually the first thing it is compared against.

    Few people can say what those 2048 bits record about a molecule and what they leave out, though. Fingerprints built with the same settings can look very different from one dataset to another. This notebook uses widgets to look inside ECFP4 and build that understanding.

    /// admonition | A note on names
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) and RDKit's **Morgan fingerprint** are the same thing. The number in ECFP*n* is the *diameter* of the neighbourhood around each atom. RDKit asks for the radius instead, so ECFP4 is `radius=2`.
    ///

    The notebook has three parts.

    * **[Part 1](#part-1) · Inside ECFP4**: the algorithm and its properties
    * **[Part 2](#part-2) · ECFP4 on datasets**: what ECFP4 looks like inside a dataset
    * **[Part 3](#part-3) · Extra**: comparing datasets to get an overview

    The suggested reading order:

    1. Read [Part 1](#part-1) and [Part 2](#part-2) to learn how ECFP4 works and how to read the widgets (in Part 2, start with a single endpoint)
    2. Browse all 16 endpoints in [Part 3](#part-3)
    3. [Pick](#picker) an endpoint that caught your eye and look inside it in Part 2
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import duckdb
    import numpy as np
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
        census_for,
    )
    from chemari.examples import openadmet as bench

    _ = alt.data_transformers.disable_max_rows()
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
        Path,
        alt,
        bench,
        census_for,
        duckdb,
        np,
        pl,
    )


@app.cell
def _(Path, bench, mo, pl):
    # precomputed tables: local copy, else GitHub; whatever is missing is computed here
    _where = bench.open_tables(Path("results/precomputed"))
    _missing = [t for t in bench.TABLES if t not in _where]
    if _missing:
        with mo.status.spinner(
            f"No precomputed tables found, computing them here ({', '.join(_missing)})… this takes a few minutes"
        ):
            _tables = {t: [] for t in bench.TABLES}
            _raw = {}
            for _task in bench.TASKS:
                _raw.setdefault(_task.dataset, bench.load_raw(_task.dataset))
                _mol = bench.load_task(_task, raw=_raw[_task.dataset])
                _tables["molecules"].append(_mol)
                for _k, _v in bench.compute_task(_mol).items():
                    _tables[_k].append(_v)
            _computed = {k: pl.concat(v, how="diagonal_relaxed") for k, v in _tables.items()}
    else:
        _computed = {}
    molecules, neighbours, predictions, metrics, shap, gain, bitlen = (
        _computed[t] if t in _computed else pl.read_parquet(_where[t]) for t in bench.TABLES
    )
    tasks = pl.DataFrame(
        [
            {
                "task": t.key,
                "dataset": t.dataset,
                "endpoint": t.endpoint,
                "label": t.label,
                "what": t.note,
            }
            for t in bench.TASKS
        ]
    )
    _sources = "\n".join(
        f"* `{k}`: {'computed here' if k in _computed else _where[k]}" for k in bench.TABLES
    )
    mo.accordion(
        {
            "About loading the data": mo.md(
                "The slow parts (descriptors, model training, TreeSHAP) are precomputed and stored as parquet "
                "files, which are loaded from `results/precomputed/` in the repository. "
                "If the files are not found, the same code (`chemari.examples.openadmet`) computes them here. "
                "The loaded tables are aggregated with SQL in DuckDB.\n\nLoaded from:\n\n"
                + _sources
            )
        }
    )
    return bitlen, gain, metrics, molecules, neighbours, predictions, shap, tasks


@app.cell
def _(duckdb, metrics, neighbours, tasks):
    # the notebook's own DuckDB connection for its SQL cells (the default connection is also
    # used by marimo's server to parse SQL; sharing it can deadlock the two)
    db = duckdb.connect()
    db.register("metrics", metrics)
    db.register("neighbours", neighbours)
    _ = db.register("tasks", tasks)
    return (db,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## <span id="part-1"></span>Part 1 · Inside ECFP4

    Start with an 80-second video that shows how ECFP4 turns a molecule into bits.
    """)
    return


@app.cell(hide_code=True)
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The widget below runs the same steps on any molecule. Type a SMILES or pick an example, then step through with **next** or press **play**.
    """)
    return


@app.cell(hide_code=True)
def _(ECFPStepper, mo):
    mo.ui.anywidget(ECFPStepper())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here are the terms used in the video and the widget above.

    * **radius**: how many bonds out from an atom ECFP4 looks. It grows from 0 in steps of 1, and ECFP4 goes up to 2
    * **substructure (environment)**: the atoms and bonds within the radius around a central atom
    * **folding**: wrapping the fingerprint into a fixed length (usually 2048 bits for ECFP4)
    * **bit**: one element of the fingerprint. It is 1 if the molecule has a substructure that maps to that position
    * **collision**: different substructures landing on the same bit

    ### A closer look at ECFP4

    The widget below visualizes the ECFP4 that is actually generated. Select a bit on the right to highlight its substructure on the molecule and to see whether it has a collision. Change the molecule, the radius or the folding (number of bits) and try different settings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tiles_smiles = mo.ui.text("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", label="SMILES", full_width=True)
    tiles_smiles
    return (tiles_smiles,)


@app.cell(hide_code=True)
def _(MorganBitTiles, mo, tiles_smiles):
    mo.ui.anywidget(MorganBitTiles(tiles_smiles.value, label=""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Pitfalls of ECFP4

    Besides the collisions in the video, ECFP4 has other pitfalls. Here we use a widget that compares the fingerprints of two molecules to look at them.

    ### Pitfall 1 · Different molecules, identical fingerprints

    In a similarity search, several different molecules sometimes come back with a similarity of 1.0. This happens when:

    * the stereochemistry differs
    * the ring size or chain length differs
    """)
    return


@app.cell
def _():
    # names checked against PubChem (structure, and stereo for the enantiomers)
    SAME_FP = [
        {
            "label": "cyclohexylamine / cycloheptylamine (ring size)",
            "a": "NC1CCCCC1",
            "b": "NC1CCCCCC1",
            "name_a": "cyclohexylamine",
            "name_b": "cycloheptylamine",
        },
        {
            "label": "azepane / azocane (ring size)",
            "a": "C1CCCNCC1",
            "b": "C1CCCNCCC1",
            "name_a": "azepane",
            "name_b": "azocane",
        },
        {
            "label": "nonanoic acid / palmitic acid (chain length)",
            "a": "CCCCCCCCC(=O)O",
            "b": "CCCCCCCCCCCCCCCC(=O)O",
            "name_a": "nonanoic acid",
            "name_b": "palmitic acid",
        },
        {
            "label": "(R)- / (S)-thalidomide (stereo)",
            "a": "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "b": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "name_a": "(R)-thalidomide",
            "name_b": "(S)-thalidomide",
        },
        {
            "label": "lansoprazole / dexlansoprazole (stereo)",
            "a": "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1",
            "b": "Cc1c(OCC(F)(F)F)ccnc1C[S@@](=O)c1nc2ccccc2[nH]1",
            "name_a": "lansoprazole",
            "name_b": "dexlansoprazole",
        },
    ]
    CLOSE_BUT_FAR = [
        {
            "label": "N-methylacetamide / N-ethylacetamide (from the movie, CH₃ → C₂H₅)",
            "a": "CC(=O)NC",
            "b": "CC(=O)NCC",
            "name_a": "N-methylacetamide",
            "name_b": "N-ethylacetamide",
        },
        {
            "label": "phenethylamine / 4-pyridylethylamine (CH → N)",
            "a": "NCCc1ccccc1",
            "b": "NCCc1ccncc1",
            "name_a": "phenethylamine",
            "name_b": "4-pyridylethylamine",
        },
        {
            "label": "diazepam / nordazepam (N-CH₃ → N-H)",
            "a": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
            "b": "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1",
            "name_a": "diazepam",
            "name_b": "nordazepam",
        },
        {
            "label": "paracetamol / phenacetin (OH → OEt)",
            "a": "CC(=O)Nc1ccc(O)cc1",
            "b": "CCOc1ccc(NC(C)=O)cc1",
            "name_a": "paracetamol",
            "name_b": "phenacetin",
        },
        {
            "label": "OADMET-0001944 / OADMET-0002007 (PXR, one CH₃)",
            "a": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
            "b": "CCN(CC)CC(=O)Nc1c(C)cc(C)cc1C",
            "name_a": "OADMET-0001944",
            "name_b": "OADMET-0002007",
        },
    ]
    return CLOSE_BUT_FAR, SAME_FP


@app.cell(hide_code=True)
def _(SAME_FP, mo):
    # one example per cause, for the MolPair right below
    _by_label = {e["label"]: e for e in SAME_FP}
    same_kind = mo.ui.dropdown(
        {
            "stereo": _by_label["(R)- / (S)-thalidomide (stereo)"],
            "ring size": _by_label["cyclohexylamine / cycloheptylamine (ring size)"],
            "chain length": _by_label["nonanoic acid / palmitic acid (chain length)"],
        },
        value="stereo",
        label="example",
    )
    same_kind
    return (same_kind,)


@app.cell(hide_code=True)
def _(MolPair, mo, same_kind):
    _e = same_kind.value
    mo.ui.anywidget(
        MolPair(
            {"id": _e["name_a"], "smiles": _e["a"]},
            {"id": _e["name_b"], "smiles": _e["b"]},
            show_smiles=False,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When the ring size or chain length differs, the molecular composition itself differs, yet the molecules are treated as the same. In a similarity search you can spot this by eye. In machine learning the model returns exactly the same prediction for different molecules, so the risk is larger.

    The reason is simple: the default ECFP only records whether a substructure is present.

    * Stereochemistry is ignored → `includeChirality=True` tells the pair apart
    * A substructure counts as 1 however often it occurs (a longer ring or chain adds nothing) → a count fingerprint, which records the number of occurrences, tells the pair apart

    Pick a pair from the table below, then switch to count or turn chirality on and off in the widget. Watch how the similarity at the bottom left and the bit list on the right change.
    """)
    return


@app.cell(hide_code=True)
def _(SAME_FP, bench, mo, np, pl):
    def _separates(e):
        # which setting tells the pair apart (checked here with RDKit)
        out = [
            name
            for name, kw in [("count", {"count": True}), ("chirality", {"chirality": True})]
            if not np.array_equal(*bench.fingerprints([e["a"], e["b"]], **kw))
        ]
        return " / ".join(out) or "neither"

    same_table = mo.ui.table(
        pl.DataFrame(
            [
                {
                    "A": e["name_a"],
                    "B": e["name_b"],
                    "difference": e["label"].rsplit("(", 1)[1].rstrip(")"),
                    "separated by": _separates(e),
                }
                for e in SAME_FP
            ]
        ),
        selection="single",
        initial_selection=[0],
        label="pairs with the same fingerprint",
    )
    same_table
    return (same_table,)


@app.cell(hide_code=True)
def _(MorganExplorer, SAME_FP, mo, same_table):
    _sel = same_table.value
    _name = _sel["A"][0] if _sel is not None and len(_sel) else SAME_FP[0]["name_a"]
    _e = next(e for e in SAME_FP if e["name_a"] == _name)
    mo.ui.anywidget(
        MorganExplorer(
            [{"id": _e["name_a"], "smiles": _e["a"]}, {"id": _e["name_b"], "smiles": _e["b"]}],
            stereo_labels=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Neither is a cure-all, though.

    * Some stereo differences are not separated even with chirality (lansoprazole and dexlansoprazole differ in the stereo of the sulfoxide sulfur, which ECFP does not distinguish)
    * With counts, a bit with a collision also adds up the counts of the different substructures on it, so the downside of a collision can be larger than usual

    Collisions can be reduced with more bits, but the extra bits sometimes bring little. Choose the settings that suit your dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pitfall 2 · Similar molecules, lower similarity than expected

    The opposite also happens. Some pairs look very similar but have a Tanimoto of only 0.4 to 0.7, lower than intuition suggests. ECFP4 looks out to radius 2 around every atom, so changing one atom changes all the substructures around it, and more than 10 bits can be swapped.

    In the chain-length and ring-size examples of the previous section, extra carbons did not change the fingerprint. The difference between the two cases is where in the structure the change happens.

    * Adding the same unit in the middle of a long chain or a large ring → the substructures within radius 2 are ones that already exist → no new bit is set
    * Replacing an atom, or changing a substituent or the end of a chain → every substructure within radius 2 of that spot is new
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span id="far-pair"></span>Pick a pair from the table below. The two structures and the fingerprint comparison are shown in turn. The fingerprint comparison lists on the right the bits that only one of the two has (differ). You can see that the substructures around the changed atom are swapped out together.
    """)
    return


@app.cell(hide_code=True)
def _(CLOSE_BUT_FAR, SAME_FP, bench, mo, pl):
    def _bits(e):
        a, b = bench.fingerprints([e["a"], e["b"]]).astype(bool)
        return round(float((a & b).sum() / (a | b).sum()), 2), int((a ^ b).sum())

    # the chain-length pair from the previous section, for comparison (no bit changes)
    far_pairs = CLOSE_BUT_FAR + [e for e in SAME_FP if "chain length" in e["label"]]

    far_table = mo.ui.table(
        pl.DataFrame(
            [
                {
                    "A": e["name_a"],
                    "B": e["name_b"],
                    "difference": e["label"].rsplit("(", 1)[1].rstrip(")"),
                    "Tanimoto": t,
                    "bits that differ": n,
                }
                for e in far_pairs
                for t, n in [_bits(e)]
            ]
        ),
        selection="single",
        initial_selection=[0],
        label="pairs that look alike but have a low Tanimoto",
    )
    far_table
    return far_pairs, far_table


@app.cell(hide_code=True)
def _(far_pairs, far_table):
    # the pair picked in the table, for the two widgets below
    _sel = far_table.value
    _name = _sel["A"][0] if _sel is not None and len(_sel) else far_pairs[0]["name_a"]
    far_pair = next(e for e in far_pairs if e["name_a"] == _name)
    return (far_pair,)


@app.cell(hide_code=True)
def _(MolPair, far_pair, mo):
    mo.ui.anywidget(
        MolPair(
            {"id": far_pair["name_a"], "smiles": far_pair["a"]},
            {"id": far_pair["name_b"], "smiles": far_pair["b"]},
            show_smiles=False,
            show_common=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(MorganExplorer, far_pair, mo):
    _e = far_pair
    mo.ui.anywidget(
        MorganExplorer(
            [{"id": _e["name_a"], "smiles": _e["a"]}, {"id": _e["name_b"], "smiles": _e["b"]}],
            row_filter="differ",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This problem is hard to avoid by changing ECFP settings alone. Some options:

    * try another fingerprint
    * compare with subgraph-based methods such as scaffolds or MCES
    * add descriptors

    Switch the [widget that lines up the structures](#far-pair) to its similarity tab to compare the chosen pair with several methods and similarity coefficients. The values differ from method to method, so do not compare numbers across methods. Instead, watch how the values move when you change the pair within one method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    None of these is a definitive solution, and what counts as "similar" is in the end close to a matter of judgment. Screening, clustering and cross-validation splits often cut at a similarity threshold mechanically. Try a few and choose the method and threshold that fit your purpose.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## <span id="part-2"></span>Part 2 · ECFP4 on datasets
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Three datasets

    The values in all three datasets were measured in real drug discovery projects. Train/test uses the split from the challenges. Compounds in test whose structure is also in train are dropped from test, and duplicate structures within a split have their values averaged. Values measured on a ratio scale (solubility, clearance, permeability, unbound fraction and so on) are converted to log10(x + 1).
    """)
    return


@app.cell
def _(db, mo):
    _overview = mo.sql(
        """
        SELECT t.dataset, t.endpoint, t.what AS measured, m.n_train, m.n_test
        FROM tasks t JOIN metrics m USING (task)
        ORDER BY t.dataset DESC, m.n_train DESC
        """,
        engine=db,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **PXR**: activation of PXR (a nuclear receptor that regulates the expression of drug-metabolizing enzymes) ([blog](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/))
    * **ASAP**: antiviral drug discovery (ASAP Discovery). Inhibition of the MERS-CoV / SARS-CoV-2 main protease, and ADMET ([blog](https://polarishub.io/blog/antiviral-competition))
    * **ExpansionRx**: ADMET from an RNA-targeted drug discovery program (Expansion Therapeutics) ([blog](https://openadmet.ghost.io/expansionrx-openadmet-blind-challenge/))
    """)
    return


@app.cell(hide_code=True)
def _(mo, tasks):
    _options = {f"{r['dataset']} · {r['endpoint']}": r["task"] for r in tasks.iter_rows(named=True)}
    task_pick = mo.ui.dropdown(_options, value="PXR · pEC50", label="dataset · endpoint")
    mo.vstack(
        [
            mo.md(r"""
    <span id="picker"></span>Choose a dataset and an endpoint below. Whichever you choose, ECFP4 is examined in the same way. See the blogs above for the background of each dataset. Try switching between datasets to see how they differ.
    """),
            task_pick,
        ]
    )
    return (task_pick,)


@app.cell
def _(bench, molecules, pl, task_pick):
    task = bench.TASK[task_pick.value]
    mols = molecules.filter(pl.col("task") == task.key)
    train = mols.filter(pl.col("split") == "train")
    test = mols.filter(pl.col("split") == "test")
    return mols, task, test, train


@app.cell(hide_code=True)
def _(alt, mo, mols, task, test, train):
    _hist = (
        alt.Chart(mols.select("y", "split"))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("y:Q", bin=alt.Bin(maxbins=40), title=task.label),
            y=alt.Y("count():Q", stack=None, title="compounds"),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(domain=["train", "test"], range=["#1c7ed6", "#f08c00"]),
                title=None,
            ),
        )
        .properties(height=200, width=420)
    )
    mo.vstack(
        [
            mo.md(f"""
    ### <span id="sec-2-1"></span>2.1 · What is in the dataset

    **{task.dataset} · {task.endpoint}**: {task.note}. {train.height:,} train compounds and {test.height:,} test compounds.
    """),
            mo.hstack(
                [
                    _hist,
                    mo.vstack(
                        [
                            mo.stat(f"{train['y'].std():.2f}", label=f"SD of {task.label} (train)"),
                            mo.stat(f"{test['y'].std():.2f}", label="SD (test)"),
                        ]
                    ),
                ],
                widths=[1.4, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(MolGrid, mo, mols, pl):
    mol_grid = mo.ui.anywidget(
        MolGrid(
            mols.sort("y", descending=True).select("id", "smiles", pl.col("y").round(2), "split"),
            color_by="y",
            show_legend=False,
            group_by="split",
            page_size=12,
            selection_mode="pair",
        )
    )
    mo.vstack(
        [
            mo.md(
                "Take a look at the compounds in the dataset. In the grid below you can filter to train or test only, "
                "and search by substructure or similarity. Selecting a compound shows information such as molecular weight, "
                "and selecting two lets you compare them."
            ),
            mol_grid,
        ]
    )
    return (mol_grid,)


@app.cell(hide_code=True)
def _(MolPair, mo, mol_grid, mols, pl, task):
    _picked = [
        {**r, task.label: r["y"]}
        for i in mol_grid.value.get("selection", [])
        for r in mols.filter(pl.col("id") == i).iter_rows(named=True)
    ]
    if _picked:
        _out = mo.ui.anywidget(
            MolPair(
                *[{**r, "id": f"{r['id']} ({r['split']})"} for r in _picked[:2]],
                value_cols=[task.label],
                value_ranges={task.label: (float(mols["y"].min()), float(mols["y"].max()))},
                properties=["MW", "cLogP", "TPSA", "HBD", "HBA"],
            )
        )
    else:
        _out = mo.callout(
            mo.md(
                "Select a compound in the grid above to show it here. Select two to compare them."
            ),
            kind="info",
        )
    _out
    return


@app.cell(hide_code=True)
def _(census_for, mo, np, task, train):
    from rdkit import Chem as _Chem
    from rdkit.Chem import rdFingerprintGenerator as _rfg

    _census = census_for(train["smiles"].to_list(), 2, 2048)
    _envs = _census.n_envs
    # per molecule: distinct radius 0–2 environments before folding, and bits set after folding
    _gen = _rfg.GetMorganGenerator(radius=2, fpSize=2048)
    _mols = [_Chem.MolFromSmiles(s) for s in train["smiles"]]
    _n_envs = np.array([len(_gen.GetSparseCountFingerprint(m).GetNonzeroElements()) for m in _mols])
    _n_bits = np.array([_gen.GetFingerprint(m).GetNumOnBits() for m in _mols])
    mo.md(f"""
    ### <span id="sec-2-2"></span>2.2 · ECFP4 on {task.dataset} · {task.endpoint}

    The collisions seen in [Part 1](#part-1) come in two kinds.

    * **Within a molecule**: different substructures of the same molecule land on the same bit. The molecule then has one fewer bit set
    * **Across the dataset**: different substructures of different molecules land on the same bit. A set bit then does not tell you which substructure caused it<sup><a href="#ref-2">2</a></sup>

    In {task.dataset} · {task.endpoint}:

    * **{np.median(_n_envs):.0f} distinct substructures** per compound (median)
    * **{(_n_bits < _n_envs).mean():.0%}** of the train compounds have a collision within the molecule
    * The {train.height:,} train compounds have **{int(_envs.sum()):,} distinct substructures** in total. Folding into 2048 bits puts an average of **{_envs[_envs > 0].mean():.0f}** on each bit<sup><a href="#ref-3">3</a></sup>
    * Purity (among the compounds that have a bit set, the share that carry the bit's most common substructure) is **{_census.purity()[1]:.0%}** on average

    Substructures are told apart by the identifier RDKit assigns before folding. Different substructures that get the same identifier are not counted here.

    """)
    return


@app.cell(hide_code=True)
def _(BitAtlas, mo, train):
    mo.vstack(
        [
            mo.md(
                "Below are all the bits of the train set. Select a row to see the substructures that map to that bit."
            ),
            mo.ui.anywidget(BitAtlas(train["smiles"].to_list(), ids=train["id"].to_list())),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A single compound can show both kinds of collision. Select a compound in the grid.

    * Red: a collision within the molecule
    * Grey badge: the number of other substructures in the dataset that share the bit. Select a bit and those substructures are shown below
    """)
    return


@app.cell(hide_code=True)
def _(MolGrid, mo, mols, pl, test):
    bits_grid = mo.ui.anywidget(
        MolGrid(
            mols.sort("y", descending=True).select("id", "smiles", pl.col("y").round(2), "split"),
            color_by="y",
            show_legend=False,
            group_by="split",
            page_size=6,  # one row: the bit view below is the point here
            selection_mode="single",
            selection=[test.sort("y", descending=True)["id"][0]],
        )
    )
    bits_grid
    return (bits_grid,)


@app.cell(hide_code=True)
def _(MorganBitTiles, bits_grid, mo, mols, pl, task, train):
    _sel = bits_grid.value.get("selection") or []
    if _sel:
        _row = mols.filter(pl.col("id") == _sel[0]).row(0, named=True)
        _out = mo.ui.anywidget(
            MorganBitTiles(
                _row["smiles"],
                reference=train["smiles"].to_list(),
                ids=train["id"].to_list(),
                label=f"{_row['id']} · {_row['split']} · {task.label} {_row['y']:.2f}",
            )
        )
    else:
        _out = mo.callout(
            mo.md("Select a compound in the grid above to show it here."), kind="info"
        )
    _out
    return


@app.cell(hide_code=True)
def _(bench, mo, mols, np, pl, task):
    # groups of molecules (train and test together) whose bit vectors are identical, and whether
    # counts or chirality tell them apart
    _smi = mols["smiles"].to_list()
    _keys = {
        name: [row.tobytes() for row in bench.fingerprints(_smi, **kw).astype(np.uint16)]
        for name, kw in [("bit", {}), ("count", {"count": True}), ("chiral", {"chirality": True})]
    }
    _groups: dict[bytes, list[int]] = {}
    for _i, _k in enumerate(_keys["bit"]):
        _groups.setdefault(_k, []).append(_i)
    _y = mols["y"].to_numpy()
    _rows = []
    for _g in (g for g in _groups.values() if len(g) > 1):
        _by = [k for k in ("count", "chiral") if len({_keys[k][i] for i in _g}) > 1]
        _lo, _hi = min(_g, key=lambda i: _y[i]), max(_g, key=lambda i: _y[i])
        _rows.append(
            {
                "compounds": " / ".join(mols["id"][i] for i in _g[:3])
                + (" …" if len(_g) > 3 else ""),
                "n": len(_g),
                f"Δ {task.label}": round(float(_y[_hi] - _y[_lo]), 2),
                "separated by": " / ".join(
                    {"count": "count", "chiral": "chirality"}[k] for k in _by
                )
                or "neither",
                # the two members furthest apart in value, for the comparison below
                "_a": mols["id"][_lo],
                "_b": mols["id"][_hi],
            }
        )
    twins = (
        pl.DataFrame(_rows).sort(f"Δ {task.label}", descending=True) if _rows else pl.DataFrame()
    )
    _n = {
        k: int((twins["separated by"].str.contains(k)).sum()) if twins.height else 0
        for k in ("count", "chirality", "neither")
    }
    _numbers = (
        f"""
    * **{twins.height} groups** of compounds share a fingerprint, **{int(twins["n"].sum())} compounds** in total
    * **{_n["count"]} groups** can be told apart by count, **{_n["chirality"]} groups** by chirality, and **{_n["neither"]} groups** by neither
    * The largest difference in {task.label} within a group with the same fingerprint is **{twins[f"Δ {task.label}"].max():.2f}**
    """
        if twins.height
        else """
    * No compounds share a fingerprint
    """
    )
    mo.md(f"""
    ### <span id="sec-2-3"></span>2.3 · Different molecules, identical fingerprints

    This is Pitfall 1 from [Part 1](#part-1). A model that uses only the fingerprint predicts the same value for compounds with the same fingerprint.

    In {task.dataset} · {task.endpoint}:
    {_numbers}""")
    return (twins,)


@app.cell(hide_code=True)
def _(mo, twins):
    twins_table = (
        mo.ui.table(
            twins.drop("_a", "_b"),
            selection="single",
            initial_selection=[0],
            page_size=6,
            label="groups of compounds with the same fingerprint",
        )
        if twins.height
        else None
    )
    twins_table
    return (twins_table,)


@app.cell(hide_code=True)
def _(MorganExplorer, mo, mols, pl, task, twins, twins_table):
    if twins_table is None:
        _out = None
    else:
        _sel = twins_table.value
        _key = _sel["compounds"][0] if _sel is not None and len(_sel) else twins["compounds"][0]
        _row = twins.filter(pl.col("compounds") == _key).row(0, named=True)
        _pair = [mols.filter(pl.col("id") == _row[k]).row(0, named=True) for k in ("_a", "_b")]
        _out = mo.ui.anywidget(
            MorganExplorer(
                [
                    {
                        "id": r["id"],
                        "smiles": r["smiles"],
                        "label": f"{r['split']} · {task.label} {r['y']:.2f}",
                    }
                    for r in _pair
                ],
                stereo_labels=True,
            )
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-2-4"></span>2.4 · Nearest neighbour (NN)

    From here we look at similarity. For each test compound, the train compound with the highest ECFP4 Tanimoto similarity is called its **nearest neighbour (NN)**.

    The similarity to the NN shows whether train contains structures like the test compound. If the distribution below sits to the right, test is close to the structures in train. If it sits to the left, test has many structures that train does not.
    """)
    return


@app.cell(hide_code=True)
def _(alt, metrics, mo, neighbours, pl, task):
    _nn = neighbours.filter(pl.col("task") == task.key).select("tanimoto")
    _hist = (
        alt.Chart(_nn)
        .mark_bar(color="#1c7ed6", opacity=0.8)
        .encode(
            x=alt.X(
                "tanimoto:Q",
                bin=alt.Bin(extent=[0, 1], step=0.05),
                title="Tanimoto to the NN (ECFP4)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("count():Q", title="test compounds"),
        )
        .properties(height=220, width=420)
    )
    _m = metrics.filter(pl.col("task") == task.key).row(0, named=True)
    mo.hstack(
        [
            _hist,
            mo.vstack(
                [
                    mo.stat(f"{_m['nn_tanimoto_median']:.2f}", label="median Tanimoto to the NN"),
                    mo.stat(f"{_m['share_nn_ge_06']:.0%}", label="test compounds with NN ≥ 0.6"),
                    mo.stat(
                        f"{_m['n_identical_fp']}",
                        label="test compounds sharing a train fingerprint",
                    ),
                ]
            ),
        ],
        widths=[1.4, 1],
        align="center",
    )
    return


@app.cell
def _(neighbours, np, pl, task):
    # every test compound with its NN: the similarity, and how far apart their values are
    nn_pairs = neighbours.filter(pl.col("task") == task.key).with_columns(
        (pl.col("y_test") - pl.col("y_nn")).alias("delta"),
        (pl.col("y_test") - pl.col("y_nn")).abs().alias("dy"),
    )
    # the same difference for random train–test pairs (NN values shuffled)
    random_dy = float(
        np.abs(
            np.random.default_rng(0).permutation(nn_pairs["y_nn"].to_numpy())
            - nn_pairs["y_test"].to_numpy()
        ).mean()
    )
    # activity cliffs: similar by ECFP4, yet as far apart in value as unrelated compounds
    cliffs = nn_pairs.filter((pl.col("tanimoto") >= 0.6) & (pl.col("dy") >= random_dy))
    return cliffs, nn_pairs, random_dy


@app.cell(hide_code=True)
def _(mo, nn_pairs, task):
    mo.md(f"""
    ### <span id="sec-2-5"></span>2.5 · The similarity principle and activity cliffs

    In the analysis of compounds, for example in drug discovery, the similarity principle and its exception, the activity cliff, are a pair of key concepts.

    * **Similarity principle**: the idea that structurally similar molecules have similar properties<sup><a href="#ref-4">4</a></sup>. It underlies similarity search and predictions that use the values of similar compounds
    * **Activity cliff**: a pair of compounds that are structurally very similar but differ greatly in activity<sup><a href="#ref-5">5</a></sup>. A well-known example is the "magic methyl": adding a single methyl group can sometimes change the activity by 100 times or more<sup><a href="#ref-6">6</a></sup>

    The figure below shows what this looks like for {task.dataset} · {task.endpoint}.

    * Dots: test compounds ({nn_pairs.height:,})
    * x axis: similarity to the NN
    * y axis: difference in {task.label} from the NN, |Δ|
    * Red line: mean |Δ| in each similarity bin
    * Dashed line: mean |Δ| of random train–test pairs

    If the similarity principle holds, the dots gather lower toward the right. Here an activity cliff is a pair with a similarity of 0.6 or more and a |Δ| at or above the mean of random pairs (the coloured region at the top right of the figure).
    """)
    return


@app.cell(hide_code=True)
def _(alt, cliffs, mo, nn_pairs, pl, random_dy, task):
    _curve = (
        nn_pairs.with_columns((pl.col("tanimoto") * 10).floor().clip(0, 9).alias("bin"))
        .group_by("bin")
        .agg(pl.len().alias("n"), pl.col("dy").mean().alias("mean_dy"))
        .filter(pl.col("n") >= 5)
        .with_columns(((pl.col("bin") + 0.5) / 10).alias("sim"))
        .sort("bin")
    )
    _top = float(nn_pairs["dy"].max()) * 1.05
    _x = alt.X("tanimoto:Q", title="Tanimoto to the NN (ECFP4)", scale=alt.Scale(domain=[0, 1]))
    _y = alt.Y("dy:Q", title=f"|Δ {task.label}| (test − NN)", scale=alt.Scale(domain=[0, _top]))
    _region = (
        alt.Chart(pl.DataFrame({"x": [0.6], "x2": [1.0], "y": [random_dy], "y2": [_top]}))
        .mark_rect(color="#d6336c", opacity=0.08)
        .encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    )
    _label = (
        alt.Chart(pl.DataFrame({"x": [0.99], "y": [_top * 0.97], "t": ["activity cliff"]}))
        .mark_text(align="right", baseline="top", color="#d6336c", fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="t:N")
    )
    _points = (
        alt.Chart(nn_pairs.select("test_id", "nn_id", "tanimoto", "dy"))
        .mark_circle(size=18, opacity=0.35, color="#1c7ed6")
        .encode(
            x=_x,
            y=_y,
            tooltip=[
                alt.Tooltip("test_id:N", title="test"),
                alt.Tooltip("nn_id:N", title="NN"),
                alt.Tooltip("tanimoto:Q", format=".2f"),
                alt.Tooltip("dy:Q", title="|Δ|", format=".2f"),
            ],
        )
    )
    _line = (
        alt.Chart(_curve)
        .mark_line(point=True, color="#d6336c")
        .encode(
            x="sim:Q",
            y="mean_dy:Q",
            tooltip=[
                alt.Tooltip("n:Q", title="compounds"),
                alt.Tooltip("mean_dy:Q", title="mean |Δ|", format=".2f"),
            ],
        )
    )
    _rule = (
        alt.Chart(pl.DataFrame({"y": [random_dy]}))
        .mark_rule(strokeDash=[5, 4], color="#6b7280")
        .encode(y="y:Q")
    )
    mo.hstack(
        [
            (_region + _points + _rule + _line + _label).properties(height=280, width=430),
            mo.vstack(
                [
                    mo.stat(f"{nn_pairs['dy'].mean():.2f}", label="mean |Δ| to the NN"),
                    mo.stat(f"{random_dy:.2f}", label="mean |Δ| of random pairs"),
                    mo.stat(f"{cliffs.height}", label="activity cliffs"),
                ]
            ),
        ],
        widths=[1.5, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo, task):
    mo.md(rf"""
    The table below lists the activity cliff pairs. Select a row to compare the two compounds in two ways.

    * **Molecules**: structures, measured values and properties side by side
    * **Fingerprints**: the bits that differ between the two. The `Δ {task.label}` column is the mean {task.label} of train compounds with the bit set minus the mean of those without it. A positive value means compounds with that bit tend to have higher values
    """)
    return


@app.cell(hide_code=True)
def _(cliffs, mo, pl):
    _pairs = cliffs.sort("dy", descending=True).select(
        pl.col("test_id").alias("test"),
        pl.col("nn_id").alias("NN (train)"),
        pl.col("tanimoto").round(2).alias("Tanimoto"),
        pl.col("y_test").round(2).alias("value (test)"),
        pl.col("y_nn").round(2).alias("value (NN)"),
        pl.col("delta").round(2).alias("Δ"),
    )
    cliff_table = mo.ui.table(
        _pairs,
        selection="single",
        initial_selection=[0] if _pairs.height else [],
        page_size=6,
        label="activity cliff pairs",
    )
    cliff_table
    return (cliff_table,)


@app.cell(hide_code=True)
def _(MolPair, MorganExplorer, cliff_table, mo, mols, pl, task, train):
    _sel = cliff_table.value
    if _sel is None or len(_sel) == 0:
        _out = mo.callout(
            mo.md("This endpoint has no pairs that count as activity cliffs."), kind="info"
        )
    else:
        _r = _sel.row(0, named=True)
        _rows = [
            mols.filter(pl.col("id") == i).row(0, named=True)
            for i in (_r["test"], _r["NN (train)"])
        ]
        _out = mo.ui.tabs(
            {
                "Molecules": mo.ui.anywidget(
                    MolPair(
                        *[{**r, task.label: r["y"]} for r in _rows],
                        value_cols=[task.label],
                        properties=["MW", "cLogP", "TPSA", "HBD", "HBA"],
                    )
                ),
                "Fingerprints": mo.ui.anywidget(
                    MorganExplorer(
                        [
                            {
                                "id": r["id"],
                                "smiles": r["smiles"],
                                "label": f"{r['split']} · {task.label} {r['y']:.2f}",
                            }
                            for r in _rows
                        ],
                        reference=train["smiles"].to_list(),
                        y=train["y"].to_numpy(),
                        y_label=task.label,
                        row_filter="differ",
                    )
                ),
            }
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-2-6"></span>2.6 · Models

    We build models that predict the measured value from ECFP4 features, using LightGBM. For comparison, models that differ only in their features are also trained.

    * **ECFP4 bit**: 2048 bit
    * **ECFP4 count**: the same 2048 dimensions, but keeping the number of occurrences
    * **RDKit desc**: 217 RDKit 2D descriptors (whole-molecule properties such as molecular weight, logP and TPSA)
    * **bit + desc**: both side by side

    Every model is trained on train and predicts test. As a baseline we also include using the measured value of the NN from [2.4](#sec-2-4) directly as the prediction (**NN value**). It shows how far the similarity principle alone goes.

    The scatter plot under the bar chart shows the chosen model's predictions on test. Blue dots are compounds with a similarity of 0.6 or more to the NN (the range counted as "similar" in [2.5](#sec-2-5)), and triangles are the test compounds in the activity cliffs of [2.5](#sec-2-5). Hover over a dot to show that compound on the right. Click to pin it.
    """)
    return


@app.cell(hide_code=True)
def _(alt, bench, metrics, mo, nn_pairs, pl, task):
    _m = metrics.filter(pl.col("task") == task.key).row(0, named=True)
    _order = ["NN value", *bench.FEATURES]
    _yt, _yn = nn_pairs["y_test"].to_numpy(), nn_pairs["y_nn"].to_numpy()
    _nn = {
        "rho": _m["rho_1nn"],
        "r2": float(1 - ((_yn - _yt) ** 2).sum() / ((_yt - _yt.mean()) ** 2).sum()),
        "mae": float(abs(_yn - _yt).mean()),
    }
    _colours = alt.Scale(
        domain=_order, range=["#adb5bd", "#4c78a8", "#f58518", "#e45756", "#72b7b2"]
    )

    def _room(scores):
        # the data's range (with zero) plus room on the right for the value labels
        _lo, _hi = min(0.0, scores.min()), max(0.0, scores.max())
        return [_lo, _hi + 0.2 * (_hi - _lo)]

    def _panel(key, title, first, domain=None):
        _bars = pl.DataFrame(
            {
                "features": _order,
                "score": [_nn[key]] + [_m[f"{key} {f}"] for f in bench.FEATURES],
            }
        )
        _bar = (
            alt.Chart(_bars, title=alt.Title(title, fontSize=12, anchor="start"))
            .mark_bar()
            .encode(
                y=alt.Y(
                    "features:N",
                    sort=_order,
                    title=None,
                    axis=alt.Axis(labels=first, ticks=first, domain=first),
                ),
                x=alt.X(
                    "score:Q",
                    title=None,
                    scale=alt.Scale(domain=domain or _room(_bars["score"]), nice=False),
                ),
                color=alt.Color("features:N", legend=None, scale=_colours),
                tooltip=["features", alt.Tooltip("score:Q", title=title, format=".2f")],
            )
            .properties(height=170, width=265)
        )
        # labels of negative bars (R² of the NN value) sit right of zero
        _text = (
            _bar.mark_text(align="left", dx=3, fontSize=11)
            .transform_calculate(at="max(datum.score, 0)")
            .encode(x="at:Q", text=alt.Text("score:Q", format=".2f"), color=alt.value("#495057"))
        )
        return _bar + _text

    mo.hstack(
        [
            alt.hconcat(
                _panel("rho", "Spearman ρ", True, [0, 1]),
                _panel("r2", "R²", False),
                _panel("mae", "MAE (lower is better)", False),
                spacing=28,
            ).resolve_scale(color="shared")
        ],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(bench, mo):
    model_pick = mo.ui.radio(
        ["NN value", *bench.FEATURES], value="ECFP4 bit", inline=True, label="model"
    )
    return (model_pick,)


@app.cell(hide_code=True)
def _(MolScatter, cliffs, mo, model_pick, mols, nn_pairs, pl, predictions, task):
    _smi = dict(zip(mols["id"], mols["smiles"]))
    # "NN value" predicts each test compound with its NN's measured value
    _pred = (
        nn_pairs.select(
            pl.col("test_id").alias("id"), pl.col("y_test").alias("y"), pl.col("y_nn").alias("pred")
        )
        if model_pick.value == "NN value"
        else predictions.filter(
            (pl.col("task") == task.key)
            & (pl.col("split") == "test")
            & (pl.col("features") == model_pick.value)
        ).select("id", "y", "pred")
    )
    _p = (
        _pred.join(nn_pairs.select(pl.col("test_id").alias("id"), "nn_id", "tanimoto"), on="id")
        .with_columns(
            pl.col("id").replace_strict(_smi).alias("smiles"),
            pl.col("nn_id").replace_strict(_smi).alias("nn_smiles"),
            pl.col("id").is_in(cliffs["test_id"].implode()).alias("cliff"),
            pl.when(pl.col("tanimoto") >= 0.6)
            .then(pl.lit("≥ 0.6"))
            .otherwise(pl.lit("< 0.6"))
            .alias("NN"),
        )
        .sort("tanimoto")
    )
    _scatter = mo.ui.anywidget(
        MolScatter(
            _p.select(
                "id",
                "smiles",
                pl.col("y").alias("measured"),
                pl.col("pred").alias("predicted"),
                pl.col("tanimoto").alias("Tanimoto to the NN"),
                "cliff",
                "NN",
                "nn_id",
                "nn_smiles",
            ),
            x="measured",
            y="predicted",
            x_label=f"measured {task.label}",
            y_label=f"predicted ({model_pick.value})",
            color_by="NN",
            color_label="Tanimoto to the NN",
            color_map={"≥ 0.6": "#1c7ed6", "< 0.6": "#b8c2cc"},
            mark_by="cliff",
            mark_label="activity cliff",
            diagonal=True,
            same_axes=True,
            card_title="test compound",
            info_title=f"prediction ({model_pick.value})",
            axis_fields=["measured", "predicted"],
            partner_id_col="nn_id",
            partner_smiles_col="nn_smiles",
            partner_label="NN (train)",
            partner_fields=["Tanimoto to the NN"],
        )
    )
    mo.vstack([model_pick, _scatter])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-2-7"></span>2.7 · Feature importance

    The bits the ECFP4 bit model uses are ranked by two measures.

    * **gain**: the total error reduction from the splits that LightGBM's trees make on that bit ([`feature_importance(importance_type="gain")`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance))
    * **mean |SHAP|**: the absolute TreeSHAP value (how much each bit raised or lowered the prediction for that compound), averaged over all train compounds. Compounds where the bit is not set are included

    Click a row to show the substructures that map to that bit and the compounds that have it set, below.
    """)
    return


@app.cell(hide_code=True)
def _(BitImportance, gain, mo, np, pl, shap, task, train):
    _s = shap.filter(
        (pl.col("task") == task.key)
        & (pl.col("bit") >= 0)
        & pl.col("id").is_in(train["id"].implode())
    )
    # the mean SHAP of each bit over the train compounds that set it
    _agg = _s.group_by("bit").agg(pl.col("shap").mean().alias("on"))
    _eff = np.zeros(2048)
    _eff[_agg["bit"].to_numpy()] = _agg["on"].to_numpy()
    _gain = gain.filter(pl.col("task") == task.key).sort("bit")
    mo.ui.anywidget(
        BitImportance(
            train["smiles"].to_list(),
            importance={
                "gain": _gain["gain"].to_numpy(),
                "mean |SHAP|": _gain["shap_abs"].to_numpy(),
            },
            effect=_eff,
            effect_label="mean SHAP (bit on)",
            ids=train["id"].to_list(),
            y=train["y"].to_numpy(),
            y_label=task.label,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, neighbours, pl, task):
    _nb = neighbours.filter(pl.col("task") == task.key).sort(
        (pl.col("y_test") - pl.col("y_nn")).abs(), descending=True
    )
    shap_pick = mo.ui.dropdown(
        {
            f"{r['test_id']} (Tanimoto {r['tanimoto']:.2f})": r["test_id"]
            for r in _nb.head(30).iter_rows(named=True)
        },
        value=None
        if _nb.height == 0
        else f"{_nb['test_id'][0]} (Tanimoto {_nb['tanimoto'][0]:.2f})",
        label="test compound (largest difference from the NN first)",
    )
    mo.vstack(
        [
            mo.md(r"""
    This breaks down a single prediction. A test compound and its NN in train are shown side by side. The colour of each atom is the bit's TreeSHAP contribution, split evenly among the atoms of the substructures on that bit (red raises the prediction, blue lowers it).
    """),
            shap_pick,
        ]
    )
    return (shap_pick,)


@app.cell(hide_code=True)
def _(MorganExplorer, mo, mols, neighbours, pl, predictions, shap, shap_pick, task, train):
    _r = neighbours.filter(
        (pl.col("task") == task.key) & (pl.col("test_id") == shap_pick.value)
    ).row(0, named=True)
    _ids = [_r["test_id"], _r["nn_id"]]
    _smi = dict(zip(mols["id"], mols["smiles"]))
    _y = dict(zip(mols["id"], mols["y"]))
    _pred = dict(
        predictions.filter((pl.col("task") == task.key) & (pl.col("features") == "ECFP4 bit"))
        .select("id", "pred")
        .iter_rows()
    )
    _maps = []
    for _i in _ids:
        _c = shap.filter((pl.col("task") == task.key) & (pl.col("id") == _i) & (pl.col("bit") >= 0))
        _maps.append({str(b): float(v) for b, v in _c.select("bit", "shap").iter_rows()})
    mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _ids[0],
                    "smiles": _smi[_ids[0]],
                    "label": f"test · measured {_y[_ids[0]]:.2f} · predicted {_pred[_ids[0]]:.2f}",
                },
                {
                    "id": _ids[1],
                    "smiles": _smi[_ids[1]],
                    # no prediction here: the stored one is out of fold, the SHAP is not
                    "label": f"train · measured {_y[_ids[1]]:.2f}",
                },
            ],
            reference=train["smiles"].to_list(),
            y=train["y"].to_numpy(),
            y_label=task.label,
            contributions=_maps,
            contrib_label="SHAP",
            contrib_radius=2,
            contrib_n_bits=2048,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, task):
    mo.md(f"""
    ---

    ## <span id="part-3"></span>Part 3 · Extra: comparing datasets

    [Part 2](#part-2) picked one endpoint and looked inside it. As an extra, this part lines up all 16 endpoints for comparison. If one catches your eye, switch to it with the [selection above](#picker) and look inside it with the Part 2 widgets. In the figures, the endpoint you have selected, {task.dataset} · {task.endpoint}, is in bold.

    ### <span id="sec-3-1"></span>3.1 · Substructures on a bit

    This compares the dataset-wide collisions seen in [2.2](#sec-2-2) across datasets. `bits` changes the number of bits. Switching to `endpoint` shows the distribution for each endpoint.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    load_unit = mo.ui.radio(["dataset", "endpoint"], value="dataset", inline=True, label="per")
    load_bits = mo.ui.radio(
        {"1024": 1024, "2048": 2048, "4096": 4096, "8192": 8192},
        value="2048",
        inline=True,
        label="bits",
    )
    return load_bits, load_unit


@app.cell(hide_code=True)
def _(bench, census_for, molecules, np, pl):
    # each dataset's whole train set, and every endpoint's train set
    train_sets = {}
    for _t in bench.TASKS:
        _smi = molecules.filter((pl.col("task") == _t.key) & (pl.col("split") == "train"))[
            "smiles"
        ].to_list()
        train_sets[("endpoint", f"{_t.dataset} · {_t.endpoint}")] = _smi
        train_sets.setdefault(("dataset", _t.dataset), []).extend(_smi)
    # a compound measured for several endpoints counts once
    train_sets = {k: list(dict.fromkeys(v)) for k, v in train_sets.items()}

    def bit_load_of(smiles, n_bits):
        """The 2.2 census of one train set: its numbers, and substructures on a bit → bits."""
        _c = census_for(smiles, 2, n_bits)
        _e = _c.n_envs
        _k, _n = np.unique(_e, return_counts=True)
        return {
            "n_train": len(smiles),
            "substructures": int(_e.sum()),
            "per_bit": float(_e[_e > 0].mean()),
            "empty_bits": int((_e == 0).sum()),
            "few": float((_e <= 5).mean()),  # share of bits with at most 5 substructures
            "purity": _c.purity()[1],
        }, pl.DataFrame({"envs": _k, "bits": _n})

    return bit_load_of, train_sets


@app.cell(hide_code=True)
def _(bit_load_of, load_bits, load_unit, mo, pl, train_sets):
    _rows, _hist = [], []
    with mo.status.spinner("Counting substructures…"):
        for (_unit, _name), _smi in train_sets.items():
            if _unit == load_unit.value:
                _row, _h = bit_load_of(_smi, load_bits.value)
                _rows.append({"name": _name, **_row})
                _hist.append(_h.with_columns(pl.lit(_name).alias("name")))
    bit_load = pl.DataFrame(_rows).sort("per_bit")
    bit_load_hist = pl.concat(_hist)
    return bit_load, bit_load_hist


@app.cell(hide_code=True)
def _(alt, bit_load, bit_load_hist, dataset_colours, load_bits, load_unit, mo, pl, task):
    _by_dataset = load_unit.value == "dataset"
    _sel = task.dataset if _by_dataset else f"{task.dataset} · {task.endpoint}"
    _load = bit_load
    _d = bit_load_hist
    _w, _h = (290, 220) if _by_dataset else (190, 130)
    _cols = 3 if _by_dataset else 4
    _top = int(_d["bits"].max())
    _n = int(_load["n_train"].max())
    _most = int(_d["envs"].max())

    def _panel(i, r):
        _first = i % _cols == 0
        # endpoint names are long: the dataset goes on the line below
        _ds_name, _, _head = r["name"].rpartition(" · ")
        _bars = (
            alt.Chart(
                _d.filter(pl.col("name") == r["name"]),
                title=alt.Title(
                    _head,
                    subtitle=_ds_name or alt.Undefined,
                    fontSize=14 if _by_dataset else 13,
                    fontWeight="bold" if r["name"] == _sel else "normal",
                    subtitleFontSize=11,
                    subtitleColor="#495057",
                    anchor="start",
                ),
            )
            .mark_bar(width={"band": 0.85})
            .encode(
                x=alt.X(
                    "envs:O",
                    title="substructures on a bit",
                    axis=alt.Axis(
                        values=list(range(0, _most + 1, 5 if _most <= 30 else 10)), labelAngle=0
                    ),
                    scale=alt.Scale(domain=list(range(_most + 1))),
                ),
                y=alt.Y(
                    "bits:Q",
                    title="bits" if _first else None,
                    scale=alt.Scale(domain=[0, _top]),
                    axis=alt.Axis(labels=_first, ticks=_first),
                ),
                # the dataset colours of 3.2
                color=alt.value(dataset_colours[_ds_name or _head]),
                tooltip=[
                    alt.Tooltip("envs:O", title="substructures on a bit"),
                    alt.Tooltip("bits:Q", title="bits"),
                ],
            )
        )
        # the numbers in a box in the empty top-right corner, clear of the grid lines
        _big, _small = (17, 12.5) if _by_dataset else (12.5, 11)
        _pad, _right = 7, _w - 6
        # wide enough for the longest line (about 0.6 em per character), the same in every panel
        _box_w = 2 * _pad + max(len(f"{_n:,} compounds") * _small, len("10.6 per bit") * _big) * 0.6
        _lines = [
            (f"{r['per_bit']:.1f} per bit", _big, "bold", 6 + _pad),
            (f"purity {r['purity']:.0%}", _small, "normal", 6 + _pad + _big + 4),
            (f"{r['n_train']:,} compounds", _small, "normal", 6 + _pad + _big + _small + 8),
        ]
        _box = (
            alt.Chart(pl.DataFrame({"x": [0]}))
            .mark_rect(fill="white", stroke="#ced4da", strokeWidth=1, cornerRadius=4, opacity=0.95)
            .encode(
                x=alt.value(_right - _box_w),
                x2=alt.value(_right),
                y=alt.value(6),
                y2=alt.value(6 + 2 * _pad + _big + 2 * _small + 8),
            )
        )
        _labels = [
            alt.Chart(pl.DataFrame({"t": [t]}))
            .mark_text(
                align="right", baseline="top", fontSize=size, fontWeight=weight, color="#212529"
            )
            .encode(x=alt.value(_right - _pad), y=alt.value(y), text="t:N")
            for t, size, weight, y in _lines
        ]
        return alt.layer(_bars, _box, *_labels).properties(width=_w, height=_h)

    _chart = alt.concat(
        *[_panel(i, r) for i, r in enumerate(_load.iter_rows(named=True))],
        columns=_cols,
        spacing=24 if _by_dataset else 18,
    )
    mo.vstack(
        [
            mo.hstack([load_unit, load_bits], justify="start", gap=2),
            mo.hstack([_chart], justify="center"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bit_load_of, mo, train_sets):
    # the text is about 2048 bits, whatever the switch above shows
    _at = {
        d: bit_load_of(train_sets[("dataset", d)], 2048)[0] for d in ("PXR", "ASAP", "ExpansionRx")
    }
    _pxr, _asap, _exp = _at["PXR"], _at["ASAP"], _at["ExpansionRx"]
    _pxr_long = bit_load_of(train_sets[("dataset", "PXR")], 8192)[0]
    mo.md(f"""
    With 2048 bits, these trends appear.

    * **PXR**
        * No empty bits
        * The most substructures per bit (an average of {_pxr["per_bit"]:.1f})
        * The lowest purity ({_pxr["purity"]:.0%})
    * **ASAP and ExpansionRx**
        * Most bits hold 5 or fewer substructures (ASAP {_asap["few"]:.0%}, ExpansionRx {_exp["few"]:.0%}), against only {_pxr["few"]:.0%} for PXR
        * ExpansionRx has more compounds than PXR, but only about {_exp["substructures"] / _pxr["substructures"]:.0%} as many distinct substructures

    The PXR compounds are structurally quite diverse. More bits move the distribution to the left (toward 0), but even at 8192 bits PXR still has an average of {_pxr_long["per_bit"]:.1f} substructures per bit.

    Which substructures sit on a given bit can be seen in the table in [2.2](#sec-2-2). Switch between PXR and ASAP with the [selection above](#picker) to see the difference.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-2"></span>3.2 · Collisions and accuracy

    Does the model get more accurate if more bits reduce collisions? We retrained the ECFP4 bit model at each number of bits.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # checks or unchecks every endpoint below
    bitlen_all = mo.ui.checkbox(value=True, label="all")
    return (bitlen_all,)


@app.cell(hide_code=True)
def _(bitlen_all, mo):
    dataset_colours = {"ASAP": "#0ca678", "ExpansionRx": "#7048e8", "PXR": "#e8590c"}
    # the same per dataset (rebuilt when the overall "all" flips)
    bitlen_all_of = mo.ui.dictionary(
        {d: mo.ui.checkbox(value=bitlen_all.value, label="all") for d in dataset_colours}
    )
    return bitlen_all_of, dataset_colours


@app.cell(hide_code=True)
def _(bench, bitlen_all_of, mo):
    # one checkbox per endpoint: which lines the charts below show (rebuilt when an "all" flips)
    bitlen_show = mo.ui.dictionary(
        {
            t.key: mo.ui.checkbox(value=bitlen_all_of.value[t.dataset], label=t.endpoint)
            for t in bench.TASKS
        }
    )
    # what the two charts show: one fingerprint measure, one test score
    bitlen_fp = mo.ui.radio(
        ["substructures per bit", "purity", "empty bits"],
        value="substructures per bit",
        inline=True,
        label="fingerprint",
    )
    bitlen_score = mo.ui.radio(
        ["Spearman ρ", "R²", "MAE"], value="Spearman ρ", inline=True, label="test score"
    )
    return bitlen_fp, bitlen_score, bitlen_show


@app.cell(hide_code=True)
def _(
    alt,
    bench,
    bitlen,
    bitlen_all,
    bitlen_all_of,
    bitlen_fp,
    bitlen_score,
    bitlen_show,
    dataset_colours,
    mo,
    pl,
    task,
    tasks,
):
    # the ECFP4 bit model refitted at each length (precomputed), one line per endpoint
    _d = (
        bitlen.join(tasks.select("task", "dataset", "endpoint"), on="task")
        .with_columns(
            (pl.col("dataset") + " · " + pl.col("endpoint")).alias("name"),
            (pl.col("task") == task.key).alias("selected"),
            (pl.col("empty_bits") / pl.col("n_bits")).alias("empty"),
        )
        .filter(pl.col("task").is_in([k for k, on in bitlen_show.value.items() if on]))
    )
    _colour = alt.Color(
        "dataset:N",
        scale=alt.Scale(domain=list(dataset_colours), range=list(dataset_colours.values())),
        legend=None,
    )

    def _lines(field, title, fmt, domain=None):
        _base = alt.Chart(_d, title=alt.Title(title, fontSize=13, anchor="start")).encode(
            x=alt.X("n_bits:O", title="bits", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                f"{field}:Q",
                title=None,
                scale=alt.Scale(domain=domain) if domain else alt.Scale(zero=False),
                axis=alt.Axis(format=fmt),
            ),
            color=_colour,
            detail="name:N",
            tooltip=[
                alt.Tooltip("name:N", title="endpoint"),
                alt.Tooltip("n_bits:O", title="bits"),
                alt.Tooltip(f"{field}:Q", title=title, format=fmt),
            ],
        )
        # the selected endpoint thicker, on top
        return (
            _base.mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=1.5).transform_filter(
                "!datum.selected"
            )
            + _base.mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3.5).transform_filter(
                "datum.selected"
            )
        ).properties(width=400, height=260)

    # the checkboxes as a two-column grid: the dataset (in the colour of its lines), then its
    # endpoints, which wrap inside their own column
    _rows = "".join(
        f'<div style="font-weight:600;white-space:nowrap"><span style="color:{c}">●</span> {d}</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:4px 18px">'
        + f'<span style="margin-right:10px">{bitlen_all_of[d]}</span>'
        + "".join(f"{bitlen_show[t.key]}" for t in bench.TASKS if t.dataset == d)
        + "</div>"
        for d, c in dataset_colours.items()
    )
    _boxes = mo.Html(
        '<div style="display:grid;grid-template-columns:max-content 1fr;gap:8px 20px;'
        "align-items:start;padding:10px 14px;border:1px solid var(--slate-4, #e5e7eb);"
        f'border-radius:8px;font-size:0.9rem"><div></div><div>{bitlen_all}</div>{_rows}</div>'
    )
    _fp = {
        "substructures per bit": ("per_bit", "substructures per bit", ".1f"),
        "purity": ("purity", "purity", ".0%"),
        "empty bits": ("empty", "empty bits", ".0%", [0, 1]),
    }
    _score = {
        "Spearman ρ": ("rho", "Spearman ρ (test)", ".2f", [0, 1]),
        "R²": ("r2", "R² (test)", ".2f"),
        "MAE": ("mae", "MAE (test, lower is better)", ".2f"),
    }
    # share of empty bits at the longest length, median over the endpoints
    _empty_long = (
        bitlen.filter(pl.col("n_bits") == 8192)
        .select((pl.col("empty_bits") / pl.col("n_bits")).median())
        .item()
    )
    _pxr = {
        r["n_bits"]: r for r in bitlen.filter(pl.col("task") == "pxr/pEC50").iter_rows(named=True)
    }
    mo.vstack(
        [
            mo.md(
                f"Each line is one endpoint. The thick line is the one you have selected, {task.dataset} · {task.endpoint}."
            ),
            _boxes,
            # one fingerprint measure beside one test score, each under its own switch
            mo.hstack(
                [
                    mo.vstack([bitlen_fp, _lines(*_fp[bitlen_fp.value])], align="center"),
                    mo.vstack([bitlen_score, _lines(*_score[bitlen_score.value])], align="center"),
                ],
                justify="center",
                gap=2,
            ),
            mo.md(
                f"""
    * Fewer substructures per bit, and higher purity
    * More empty bits. Information does not grow in step with the number of bits (at 8192 bits, a median of {_empty_long:.0%} of bits are empty across the 16 endpoints)
    * Model accuracy levels off at some number of bits
        * PXR and the ExpansionRx endpoints with many compounds: around 2048 bits
        * ASAP: unchanged from 1024 bits

    With more bits, substructures that shared a bit are separated and the fingerprint holds more information. There could be several reasons why accuracy still levels off. I think the added information does not translate much into accuracy.

    * Most of the separated substructures are rare ones. In PXR, of {_pxr[2048]["substructures"]:,} substructures only {_pxr[2048]["frequent_substructures"]:,} occur in 20 or more compounds, and a few compounds are not enough to learn what such a substructure does to the value (select a bit in the table in [2.2](#sec-2-2) to see the number of compounds)
    * The main causes of wrong predictions lie elsewhere. A test set far from train ([2.4](#sec-2-4)) and activity cliffs ([2.5](#sec-2-5)) do not go away with more bits
    """
            ),
            mo.callout(
                mo.md(
                    "With the current setting (`min_child_samples=20`), LightGBM only makes splits that leave 20 or more compounds on both sides. "
                    "Rare substructures stay unused for prediction even after they are split onto separate bits."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-3"></span>3.3 · Closeness to train and accuracy

    If train has compounds similar to a test compound, their measured values are a clue for the prediction. This lines up the similarity to the NN from [2.4](#sec-2-4) and the prediction accuracy from [2.6](#sec-2-6) for the 16 endpoints.
    """)
    return


@app.cell
def _(db, mo):
    nn_by_task = mo.sql(
        """
        SELECT
            t.dataset || ' · ' || t.endpoint AS name, n.task,
            min(n.tanimoto) AS lo, quantile_cont(n.tanimoto, 0.25) AS q1,
            median(n.tanimoto) AS med, quantile_cont(n.tanimoto, 0.75) AS q3, max(n.tanimoto) AS hi,
            avg((n.tanimoto >= 0.6)::INT) AS share_close
        FROM neighbours n JOIN tasks t USING (task)
        GROUP BY ALL
        ORDER BY med DESC
        """,
        output=False,
        engine=db,
    )
    return (nn_by_task,)


@app.cell
def _(db, mo):
    # every score of every model, one row each; best marks the top model per endpoint
    # (the lowest for MAE). NN value's R² and MAE come from the neighbours table
    scores = mo.sql(
        """
        WITH nn AS (
            SELECT
                n.task,
                1 - sum(power(n.y_nn - n.y_test, 2)) / sum(power(n.y_test - a.mean, 2)) AS r2,
                avg(abs(n.y_nn - n.y_test)) AS mae
            FROM neighbours n
            JOIN (SELECT task, avg(y_test) AS mean FROM neighbours GROUP BY task) a USING (task)
            GROUP BY n.task
        ),
        long AS (
            UNPIVOT (
                SELECT
                    m.task, m.rho_1nn AS "rho NN value", nn.r2 AS "r2 NN value",
                    nn.mae AS "mae NN value", COLUMNS('^(rho|r2|mae|pair_slope) ')
                FROM metrics m JOIN nn USING (task)
            )
            ON COLUMNS(* EXCLUDE task) INTO NAME metric VALUE score
        )
        SELECT
            t.dataset || ' · ' || t.endpoint AS name, l.task,
            split_part(l.metric, ' ', 1) AS stat,
            substr(l.metric, strpos(l.metric, ' ') + 1) AS model,
            l.score,
            CASE WHEN stat = 'mae'
                THEN l.score = min(l.score) OVER (PARTITION BY l.task, stat)
                ELSE l.score = max(l.score) OVER (PARTITION BY l.task, stat)
            END AS best
        FROM long l JOIN tasks t USING (task)
        """,
        output=False,
        engine=db,
    )
    return (scores,)


@app.cell(hide_code=True)
def _(alt, nn_by_task, task):
    # rows in the same order in every chart of Part 3; the selected endpoint in bold
    rows = nn_by_task["name"].to_list()
    _sel = f"{task.dataset} · {task.endpoint}"
    row_axis = alt.Axis(
        labelFontWeight=alt.expr(f"datum.value == '{_sel}' ? 'bold' : 'normal'"),
        labelLimit=260,
    )
    return row_axis, rows


@app.cell(hide_code=True)
def _(alt, mo, neighbours, nn_by_task, pl, predictions, row_axis, rows, scores):
    _y = alt.Y("name:N", sort=rows, title=None, axis=row_axis)
    _x = alt.X("lo:Q", title="Tanimoto to the NN (ECFP4)", scale=alt.Scale(domain=[0, 1]))
    _base = alt.Chart(nn_by_task).encode(y=_y)
    _box = (
        _base.mark_rule(color="#868e96").encode(x=_x, x2="hi:Q")
        + _base.mark_bar(size=12, color="#adb5bd").encode(
            x="q1:Q",
            x2="q3:Q",
            tooltip=[
                "name",
                alt.Tooltip("med:Q", title="median", format=".2f"),
                alt.Tooltip("q1:Q", format=".2f"),
                alt.Tooltip("q3:Q", format=".2f"),
                alt.Tooltip("share_close:Q", title="NN ≥ 0.6", format=".0%"),
            ],
        )
        + _base.mark_tick(color="#212529", size=12, thickness=2).encode(x="med:Q")
    ).properties(
        width=300, height=380, title=alt.Title("similarity to the NN", fontSize=12, anchor="start")
    )
    # NN value → ECFP4 bit, per endpoint
    _two = scores.filter(
        (pl.col("stat") == "rho") & pl.col("model").is_in(["NN value", "ECFP4 bit"])
    )
    _dy = alt.Y(
        "name:N", sort=rows, title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)
    )
    _db = alt.Chart(_two).encode(y=_dy)
    _dumbbell = (
        _db.mark_line(color="#ced4da", strokeWidth=2).encode(x="score:Q", detail="name:N")
        + _db.mark_circle(size=80, opacity=1).encode(
            x=alt.X("score:Q", title="Spearman ρ (test)", scale=alt.Scale(domain=[-0.1, 1])),
            color=alt.Color(
                "model:N",
                # the colours of 2.6
                scale=alt.Scale(domain=["NN value", "ECFP4 bit"], range=["#adb5bd", "#4c78a8"]),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["name", "model", alt.Tooltip("score:Q", title="ρ", format=".2f")],
        )
    ).properties(
        width=300, height=380, title=alt.Title("NN value → ECFP4 bit", fontSize=12, anchor="start")
    )
    # how far the model's rho is above NN value, per endpoint
    _rho = scores.filter(pl.col("stat") == "rho").pivot(on="model", index="task", values="score")
    _gain = dict(_rho.select("task", pl.col("ECFP4 bit") - pl.col("NN value")).iter_rows())
    # the model's test error for compounds with and without a close NN, per endpoint
    _err = (
        neighbours.join(
            predictions.filter(
                (pl.col("features") == "ECFP4 bit") & (pl.col("split") == "test")
            ).select("task", pl.col("id").alias("test_id"), "pred"),
            on=["task", "test_id"],
        )
        .group_by("task", (pl.col("tanimoto") >= 0.6).alias("close"))
        .agg((pl.col("pred") - pl.col("y_test")).abs().mean().alias("mae"))
        .pivot(on="close", index="task", values="mae")
    )
    _closer_better = int((_err["true"] < _err["false"]).sum())
    _pxr = _err.filter(pl.col("task") == "pxr/pEC50").row(0, named=True)
    _pxr_mae = {True: _pxr["true"], False: _pxr["false"]}
    mo.vstack(
        [
            mo.md(
                """
    * Left: similarity between the test compounds and their NN
    * Right: accuracy comparison (Spearman ρ). Using the NN's measured value directly as the prediction (NN value), and the ECFP4 bit model
    """
            ),
            mo.hstack([alt.hconcat(_box, _dumbbell, spacing=24)], justify="center"),
            mo.md(
                f"""
    * ASAP pIC50: almost every test compound has a similar NN. NN value alone gives ρ of 0.61 and 0.75
    * PXR: there are almost no similar NNs. NN value has a ρ of 0.05, but the model has 0.61
    * ASAP KSOL: the model falls below NN value. Measured values cluster near the upper limit, so ranks are hard to separate

    Across endpoints, the higher the similarity to the NN, the higher the ρ of NN value. The model's margin over NN value is larger for endpoints with lower similarity (PXR {_gain["pxr/pEC50"]:+.2f}, ASAP pIC50 {_gain["asap/mers"]:+.2f} and {_gain["asap/sars2"]:+.2f}).

    Within a single endpoint too, the model's error is smaller for test compounds that have a similar NN. This holds for {_closer_better} of the 16 endpoints. In PXR, the MAE is {_pxr_mae[True]:.2f} for compounds with a similarity of 0.6 or more and {_pxr_mae[False]:.2f} for those below 0.6.

    The NN of each test compound and its structure can be seen in [2.4](#sec-2-4).
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-4"></span>3.4 · Features and accuracy

    This lines up the five predictions from [2.6](#sec-2-6) across the 16 endpoints.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    score_pick = mo.ui.radio(
        {"Spearman ρ": "rho", "R²": "r2", "MAE": "mae"},
        value="Spearman ρ",
        inline=True,
        label="test score",
    )
    score_view = mo.ui.radio(
        {"score": "score", "Δ from ECFP4 bit": "delta"}, value="score", inline=True, label="show"
    )
    return score_pick, score_view


@app.cell(hide_code=True)
def _(alt, pl, row_axis, rows, scores):
    def score_heat(stat, models, title, scheme="viridis", best=None, baseline=None):
        """Endpoints × models, each cell coloured from the worst to the best score of the table
        (the lowest is best for MAE); ``best`` adds a column naming the best model of each row.

        ``baseline``: a model name; the cells then show each score minus that model's score in
        the same row, on a diverging scale (blue = better than the baseline, red = worse)."""
        _d = scores.filter((pl.col("stat") == stat) & pl.col("model").is_in(models))
        _flip = -1 if stat == "mae" else 1
        if baseline is None:
            _lo, _hi = _d["score"].min(), _d["score"].max()
            _shade = (pl.col("score") - _lo) / (_hi - _lo)
            _d = _d.with_columns(
                pl.col("score").alias("shown"),
                (1 - _shade if stat == "mae" else _shade).alias("shade"),
            )
            _scale, _fmt = alt.Scale(domain=[0, 1], scheme=scheme), ".2f"
            # dark cells get white numbers; viridis is dark at the low end, the others at the high
            _dark = "datum.shade < 0.62" if scheme == "viridis" else "datum.shade > 0.6"
        else:
            _base_score = _d.filter(pl.col("model") == baseline).select(
                "task", pl.col("score").alias("base")
            )
            _d = _d.join(_base_score, on="task").with_columns(
                (pl.col("score") - pl.col("base")).alias("shown")
            )
            _d = _d.with_columns((_flip * pl.col("shown")).alias("shade"))
            _most = float(_d["shade"].abs().max())
            _scale, _fmt = alt.Scale(domain=[-_most, 0, _most], scheme="redblue"), "+.2f"
            _dark = f"abs(datum.shade) > {0.6 * _most}"
        _base = alt.Chart(_d).encode(
            x=alt.X("model:N", sort=models, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("name:N", sort=rows, title=None, axis=row_axis),
        )
        _rect = _base.mark_rect().encode(
            color=alt.Color("shade:Q", scale=_scale, legend=None),
            tooltip=["name", "model", alt.Tooltip("score:Q", title=title, format=".2f")],
        )
        _ink = alt.condition(_dark, alt.value("white"), alt.value("#212529"))
        # the best model of each row in bold
        _text = [
            _base.mark_text(fontSize=11, fontWeight=w)
            .transform_filter(f"{'' if b else '!'}datum.best")
            .encode(text=alt.Text("shown:Q", format=_fmt), color=_ink)
            for b, w in ((True, "bold"), (False, "normal"))
        ]
        _heat = alt.layer(_rect, *_text).properties(width=82 * len(models), height=380)
        if best is None:
            return _heat
        _name = (
            alt.Chart(_d.filter(pl.col("best")), title=alt.Title("best", fontSize=11))
            .mark_text(align="left", fontSize=12, fontWeight="bold")
            .encode(
                x=alt.value(6),
                y=alt.Y("name:N", sort=rows, title=None, axis=None),
                text="model:N",
                color=alt.Color("model:N", scale=best, legend=None),
            )
            .properties(width=90, height=380)
        )
        return alt.hconcat(_heat, _name, spacing=4).resolve_scale(color="independent")

    return (score_heat,)


@app.cell(hide_code=True)
def _(alt, mo, score_heat, score_pick, score_view):
    _models = ["NN value", "ECFP4 bit", "ECFP4 count", "RDKit desc", "bit + desc"]
    # the model colours of 2.6
    _colours = alt.Scale(
        domain=_models, range=["#adb5bd", "#4c78a8", "#f58518", "#e45756", "#72b7b2"]
    )
    _title = {"rho": "Spearman ρ", "r2": "R²", "mae": "MAE"}[score_pick.value]
    mo.vstack(
        [
            mo.hstack([score_pick, score_view], justify="start", gap=2),
            mo.hstack(
                [
                    score_heat(
                        score_pick.value,
                        _models,
                        _title,
                        best=_colours,
                        baseline="ECFP4 bit" if score_view.value == "delta" else None,
                    )
                ],
                justify="center",
            ),
            mo.md(
                """
    For almost no endpoint is the ECFP4 bit model alone the best. For many endpoints, adding descriptors improves accuracy.

    The size of the gain differs by endpoint, though. For the same LogD, going from ECFP4 bit to bit + desc raises ρ by +0.42 in ASAP and by +0.03 in ExpansionRx. Which features fit depends on the dataset as well as on the kind of value measured. Comparing features and settings shows the tendencies of each dataset.

    This does not mean ECFP4 is inaccurate. A model that is weak on its own can still improve an ensemble if its errors differ from those of the other models. The scatter plot in [2.6](#sec-2-6) shows which compounds are mispredicted for the selected endpoint.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-5"></span>3.5 · Activity cliffs and accuracy

    The activity cliffs in [2.5](#sec-2-5) were test compounds that are structurally close to their NN but differ greatly in value. This compares the model's error on those compounds across the 16 endpoints. The comparison group is the compounds that also have a similarity of 0.6 or more to the NN but are not activity cliffs.
    """)
    return


@app.cell
def _(bench, neighbours, np, pl, predictions):
    # test compounds with a close NN, split as in 2.5 into activity cliffs and the rest, and
    # each model's test error (MAE) on the two groups
    _parts = []
    for _t in bench.TASKS:
        _nb = neighbours.filter(pl.col("task") == _t.key)
        _dy = (_nb["y_test"] - _nb["y_nn"]).abs().to_numpy()
        _random = np.abs(
            np.random.default_rng(0).permutation(_nb["y_nn"].to_numpy()) - _nb["y_test"].to_numpy()
        ).mean()
        _parts.append(
            _nb.with_columns(pl.Series("cliff", _dy >= _random)).filter(pl.col("tanimoto") >= 0.6)
        )
    cliff_error = (
        pl.concat(_parts)
        .join(
            predictions.filter(pl.col("split") == "test").select(
                "task", pl.col("id").alias("test_id"), "features", "pred"
            ),
            on=["task", "test_id"],
        )
        .group_by("task", "features", "cliff")
        .agg(pl.len().alias("n"), (pl.col("pred") - pl.col("y_test")).abs().mean().alias("mae"))
    )
    return (cliff_error,)


@app.cell(hide_code=True)
def _(bench, mo):
    cliff_model = mo.ui.radio(bench.FEATURES, value="ECFP4 bit", inline=True, label="model")
    return (cliff_model,)


@app.cell(hide_code=True)
def _(alt, cliff_error, cliff_model, mo, pl, row_axis, rows, tasks):
    _d = (
        cliff_error.filter(pl.col("features") == cliff_model.value)
        .join(tasks.select("task", "dataset", "endpoint"), on="task")
        .with_columns(
            (pl.col("dataset") + " · " + pl.col("endpoint")).alias("name"),
            pl.when(pl.col("cliff"))
            .then(pl.lit("activity cliff"))
            .otherwise(pl.lit("other compounds with a close NN"))
            .alias("group"),
        )
    )
    _base = alt.Chart(_d).encode(y=alt.Y("name:N", sort=rows, title=None, axis=row_axis))
    _chart = (
        _base.mark_line(color="#ced4da", strokeWidth=2).encode(x="mae:Q", detail="name:N")
        + _base.mark_circle(size=90, opacity=1).encode(
            x=alt.X("mae:Q", title=f"MAE (test, {cliff_model.value})"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["other compounds with a close NN", "activity cliff"],
                    range=["#adb5bd", "#d6336c"],
                ),
                title=None,
                legend=alt.Legend(orient="top", labelLimit=300),
            ),
            tooltip=[
                "name",
                "group",
                alt.Tooltip("n:Q", title="compounds"),
                alt.Tooltip("mae:Q", title="MAE", format=".2f"),
            ],
        )
    ).properties(width=460, height=380)
    _wide = _d.pivot(on="cliff", index="task", values="mae")
    _ratio = _wide["true"] / _wide["false"]
    _larger = int((_ratio > 1).sum())
    mo.vstack(
        [
            cliff_model,
            mo.hstack([_chart], justify="center"),
            mo.md(
                f"""
    For activity cliff compounds, the error is larger in {_larger} of the 16 endpoints. The error is {_ratio.median():.1f} times that of the other compounds (median). Changing the features does not change this trend.

    The structures of the activity cliff pairs and the bits that differ between the two can be seen in [2.5](#sec-2-5).
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## <span id="summary"></span>Summary

    This notebook looked at ECFP4 from the inside, not only through numbers. If you have been using ECFP4 without much thought, much of this may have been new. Seeing the structures directly makes the mechanism easier to grasp.

    [Part 2](#part-2) and [Part 3](#part-3) looked at ECFP4 inside datasets. What is interesting is that the number of collisions and similar quantities differ greatly from dataset to dataset. You cannot see this just by using ECFP4. PXR in particular looked quite different inside from the other two. I took part in the PXR challenge myself, and I did not notice this difference at the time.

    Checking feature importance after training a model is a common step. With fingerprints, though, what a bit stands for is hard to see, which made the check difficult. The widget in [2.7](#sec-2-7) shows the substructures of an important bit and the compounds that have it set right on the spot. It turns out that an important bit is not always a meaningful structure and is often a common substructure instead (this depends on the dataset). Comparing several datasets makes the difference clear.

    The results in this notebook come from these datasets and do not necessarily hold in general. Please use this notebook and the widgets ([`chemari`](https://github.com/N283T/chemari)) to deepen your understanding of ECFP4, or to examine ECFP4 on your own datasets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## About this notebook

    * **Data**: [PXR challenge](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY-4.0), [ASAP-Polaris-OpenADMET antiviral challenge](https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge) (MIT), [OpenADMET-ExpansionRx challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) (CC-BY-4.0)
    * **Precomputed tables**: `dev/precompute.py` → `results/precomputed/`. The computation code is `chemari.examples.openadmet`
    * **Widgets**: the widgets in this notebook (`ECFPMovie`, `ECFPStepper`, `MorganBitTiles`, `MolGrid`, `MolPair`, `BitAtlas`, `BitImportance`, `MolScatter`, `MorganExplorer`) are published as a package, [CheMari](https://github.com/N283T/chemari). A PyPI release and more widgets are under consideration
    * **AI use**: I used Claude (Anthropic) as a coding assistant for the widgets, the video and the notebook scaffolding. The questions I ask, the choice of analyses and the interpretation are my own.

    ## References

    1. <span id="ref-1"></span>Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742–754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
    2. <span id="ref-2"></span>Virany, W.; Tripp, A. Hash Collisions in Molecular Fingerprints: Effects on Property Prediction and Bayesian Optimization. AI for Science workshop, NeurIPS 2025. [arXiv:2511.17078](https://arxiv.org/abs/2511.17078) (collisions make similarity look higher, and what that does to prediction)
    3. <span id="ref-3"></span>Gütlein, M.; Kramer, S. Filtered circular fingerprints improve either prediction or runtime performance while retaining interpretability. *J. Cheminform.* **2016**, 8, 60. [doi:10.1186/s13321-016-0173-z](https://doi.org/10.1186/s13321-016-0173-z) (calls the number of substructures per bit the bit-load)
    4. <span id="ref-4"></span>Johnson, M. A.; Maggiora, G. M. (eds.) *Concepts and Applications of Molecular Similarity*. Wiley, **1990**
    5. <span id="ref-5"></span>Maggiora, G. M. On Outliers and Activity Cliffs — Why QSAR Often Disappoints. *J. Chem. Inf. Model.* **2006**, 46, 1535. [doi:10.1021/ci060117s](https://doi.org/10.1021/ci060117s)
    6. <span id="ref-6"></span>Schönherr, H.; Cernak, T. Profound Methyl Effects in Drug Discovery and a Call for New C–H Methylation Reactions. *Angew. Chem. Int. Ed.* **2013**, 52, 12256–12267. [doi:10.1002/anie.201303207](https://doi.org/10.1002/anie.201303207)
    """)
    return


if __name__ == "__main__":
    app.run()
