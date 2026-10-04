"""ECFP4 across OpenADMET datasets: loading, features, models and the per-task numbers.

Used both by ``dev/precompute.py`` (writes ``results/precomputed/*.parquet``) and by the notebook,
which reads those files and falls back to computing a task here when a file is missing.

Needs polars, lightgbm, scikit-learn and scipy on top of the package's own dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..chem import _generator, bit_census, standardize_smiles

HF = "https://huggingface.co/datasets/openadmet/"
# the tables of the release this package was installed from, so a published notebook keeps
# reading the data it was written against
PRECOMPUTED_URL = (
    f"https://raw.githubusercontent.com/N283T/chemari/v{__version__}/results/precomputed/"
)
TABLES = ["molecules", "neighbours", "predictions", "metrics", "shap", "gain", "bitlen"]
# fingerprint lengths compared in the "bitlen" table
LENGTHS = [1024, 2048, 4096, 8192]
FEATURES = ["ECFP4 bit", "ECFP4 count", "RDKit desc", "bit + desc"]


@dataclass(frozen=True)
class Task:
    key: str  # e.g. "pxr/pEC50"
    dataset: str  # PXR, ASAP, ExpansionRx
    endpoint: str  # column in the source file
    label: str  # axis label after the transform
    log10: bool  # ratio-scale endpoint: log10(x + 1) before modelling
    note: str  # what is measured, for the notebook


TASKS = [
    Task("pxr/pEC50", "PXR", "pEC50", "pEC50", False, "EC50 of PXR activation (reporter assay)"),
    Task(
        "asap/mers",
        "ASAP",
        "pIC50 (MERS-CoV Mpro)",
        "pIC50",
        False,
        "inhibition of the MERS-CoV main protease",
    ),
    Task(
        "asap/sars2",
        "ASAP",
        "pIC50 (SARS-CoV-2 Mpro)",
        "pIC50",
        False,
        "inhibition of the SARS-CoV-2 main protease",
    ),
    Task("asap/logd", "ASAP", "LogD", "LogD", False, "distribution coefficient (pH 7.4)"),
    Task("asap/ksol", "ASAP", "KSOL", "log10(KSOL + 1) (µM)", True, "kinetic solubility"),
    Task(
        "asap/hlm",
        "ASAP",
        "HLM",
        "log10(HLM + 1) (µL/min/mg)",
        True,
        "human liver microsome stability",
    ),
    Task(
        "asap/mlm",
        "ASAP",
        "MLM",
        "log10(MLM + 1) (µL/min/mg)",
        True,
        "mouse liver microsome stability",
    ),
    Task(
        "asap/mdr1",
        "ASAP",
        "MDR1-MDCKII",
        "log10(Papp + 1) (10⁻⁶ cm/s)",
        True,
        "MDR1-MDCKII permeability",
    ),
    Task("expansion/logd", "ExpansionRx", "LogD", "LogD", False, "distribution coefficient"),
    Task(
        "expansion/ksol", "ExpansionRx", "KSOL", "log10(KSOL + 1) (µM)", True, "kinetic solubility"
    ),
    Task(
        "expansion/hlm",
        "ExpansionRx",
        "HLM CLint",
        "log10(CLint + 1) (mL/min/kg)",
        True,
        "human liver microsome CLint",
    ),
    Task(
        "expansion/mlm",
        "ExpansionRx",
        "MLM CLint",
        "log10(CLint + 1) (mL/min/kg)",
        True,
        "mouse liver microsome CLint",
    ),
    Task(
        "expansion/caco2",
        "ExpansionRx",
        "Caco-2 Permeability Papp A>B",
        "log10(Papp + 1) (10⁻⁶ cm/s)",
        True,
        "Caco-2 permeability (A→B)",
    ),
    Task(
        "expansion/efflux",
        "ExpansionRx",
        "Caco-2 Permeability Efflux",
        "log10(efflux ratio + 1)",
        True,
        "Caco-2 efflux ratio",
    ),
    Task(
        "expansion/mppb",
        "ExpansionRx",
        "MPPB",
        "log10(% unbound + 1)",
        True,
        "unbound fraction in mouse plasma",
    ),
    Task(
        "expansion/mbpb",
        "ExpansionRx",
        "MBPB",
        "log10(% unbound + 1)",
        True,
        "unbound fraction in mouse brain tissue",
    ),
]
TASK = {t.key: t for t in TASKS}
# the same notes for the Japanese notebook
NOTES_JA = {
    "pxr/pEC50": "PXR 活性化 (レポーター) の EC50",
    "asap/mers": "MERS-CoV メインプロテアーゼ阻害",
    "asap/sars2": "SARS-CoV-2 メインプロテアーゼ阻害",
    "asap/logd": "分配係数 (pH 7.4)",
    "asap/ksol": "速度論的溶解度",
    "asap/hlm": "ヒト肝ミクロソーム安定性",
    "asap/mlm": "マウス肝ミクロソーム安定性",
    "asap/mdr1": "MDR1-MDCKII 膜透過",
    "expansion/logd": "分配係数",
    "expansion/ksol": "速度論的溶解度",
    "expansion/hlm": "ヒト肝ミクロソーム CLint",
    "expansion/mlm": "マウス肝ミクロソーム CLint",
    "expansion/caco2": "Caco-2 膜透過 (A→B)",
    "expansion/efflux": "Caco-2 排出比",
    "expansion/mppb": "マウス血漿タンパク非結合率",
    "expansion/mbpb": "マウス脳組織非結合率",
}

_SOURCES = {
    "PXR": {
        "train": ["pxr-challenge-train-test/resolve/main/pxr-challenge_TRAIN.csv"],
        "test": [
            "pxr-challenge-train-test/resolve/main/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
            "pxr-challenge-train-test/resolve/main/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
        ],
    },
    "ASAP": {
        "potency": ["ASAP_Polaris_OpenADMET_challenge/resolve/main/Potency.csv"],
        "admet": ["ASAP_Polaris_OpenADMET_challenge/resolve/main/ADMET.csv"],
    },
    "ExpansionRx": {
        "train": ["openadmet-expansionrx-challenge-data/resolve/main/expansion_data_train.csv"],
        "test": ["openadmet-expansionrx-challenge-data/resolve/main/expansion_data_test.csv"],
    },
}


def _read(path: str, data_dir: Path | None):
    import polars as pl

    local = data_dir / Path(path).name if data_dir is not None else None
    for candidate in [
        local,
        data_dir / "other" / path.replace("/resolve/main/", "_") if data_dir else None,
    ]:
        if candidate is not None and candidate.exists():
            return pl.read_csv(candidate, infer_schema_length=10000)
    return pl.read_csv(HF + path, infer_schema_length=10000)


def load_raw(dataset: str, data_dir: Path | None = None):
    """Rows of one dataset: id, smiles_raw, split, every endpoint column (and ci_width for PXR)."""
    import polars as pl

    src = _SOURCES[dataset]
    if dataset == "PXR":
        lo, hi = "pEC50_ci.lower (-log10(molarity))", "pEC50_ci.upper (-log10(molarity))"
        return pl.concat(
            [
                _read(f, data_dir).select(
                    pl.col("Molecule Name").alias("id"),
                    pl.col("SMILES").alias("smiles_raw"),
                    pl.lit(split).alias("split"),
                    "pEC50",
                    (pl.col(hi) - pl.col(lo)).alias("ci_width"),
                )
                for split, files in src.items()
                for f in files
            ]
        )
    if dataset == "ASAP":
        frames = []
        for f in src["potency"] + src["admet"]:
            df = _read(f, data_dir).with_columns(
                pl.col("CXSMILES").str.split(" ").list.first().alias("smiles_raw"),
                pl.col("Set").str.to_lowercase().alias("split"),
                pl.col("Molecule Name").alias("id"),
            )
            frames.append(df.drop("CXSMILES", "Set", "Molecule Name"))
        potency, admet = frames
        # keep the row order fixed: it reaches the models through subsampling
        return potency.join(
            admet,
            on=["id", "smiles_raw", "split"],
            how="full",
            coalesce=True,
            maintain_order="left_right",
        )
    return pl.concat(
        [
            _read(f, data_dir)
            .rename({"Molecule Name": "id", "SMILES": "smiles_raw"})
            .with_columns(pl.lit(split).alias("split"))
            for split, files in src.items()
            for f in files
        ],
        how="diagonal_relaxed",
    )


def load_task(task: Task | str, data_dir: Path | None = None, raw: Any = None):
    """One endpoint, ready to model: id, smiles (standardised), split, y (transformed), ci_width.

    Rows without a value are dropped, ratio-scale endpoints become log10(x + 1), duplicate
    structures within a split are averaged, and test compounds whose structure is also in
    train are dropped.
    """
    import polars as pl

    task = TASK[task] if isinstance(task, str) else task
    df = raw if raw is not None else load_raw(task.dataset, data_dir)
    if "ci_width" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("ci_width"))
    df = df.select(
        "id", "smiles_raw", "split", pl.col(task.endpoint).cast(pl.Float64).alias("y"), "ci_width"
    )
    df = df.drop_nulls("y")
    if task.log10:  # log10(x + 1): keeps the zeros (e.g. a CLint below the assay's limit)
        df = df.filter(pl.col("y") >= 0).with_columns((pl.col("y") + 1).log10())
    df = df.with_columns(
        pl.col("smiles_raw").map_elements(standardize_smiles, return_dtype=pl.Utf8).alias("smiles")
    ).drop_nulls("smiles")
    df = df.group_by("split", "smiles", maintain_order=True).agg(
        pl.col("id").first(), pl.col("y").mean(), pl.col("ci_width").max()
    )
    train = df.filter(pl.col("split") == "train")
    test = df.filter(
        (pl.col("split") == "test") & ~pl.col("smiles").is_in(train["smiles"].implode())
    )
    return pl.concat([train, test]).select(
        pl.lit(task.key).alias("task"), "id", "smiles", "split", "y", "ci_width"
    )


# ---------------------------------------------------------------- features and models

_DESC: dict[str, np.ndarray] = {}


def descriptors(smiles: list[str]) -> np.ndarray:
    """All RDKit 2D descriptors (``Descriptors.CalcMolDescriptors``), cached per SMILES."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    for s in smiles:
        if s not in _DESC:
            vals = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
            _DESC[s] = np.array(list(vals.values()), dtype=np.float64)
    out = np.array([_DESC[s] for s in smiles])
    return np.nan_to_num(np.clip(out, -1e6, 1e6)).astype(np.float32)


def fingerprints(
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    count: bool = False,
    chirality: bool = False,
) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = (
        rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=True
        )
        if chirality
        else _generator(radius, n_bits)
    )
    get = gen.GetCountFingerprintAsNumPy if count else gen.GetFingerprintAsNumPy
    return np.array([get(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)


def features(smiles: list[str]) -> dict[str, np.ndarray]:
    bit = fingerprints(smiles)
    desc = descriptors(smiles)
    return {
        "ECFP4 bit": bit,
        "ECFP4 count": fingerprints(smiles, count=True),
        "RDKit desc": desc,
        "bit + desc": np.hstack([bit, desc]),
    }


def make_model():
    """The notebook's LightGBM."""
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        colsample_bytree=0.5,
        subsample=0.8,
        subsample_freq=1,
        random_state=0,
        verbose=-1,
    )


def tanimoto(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """(n_test, n_train) Tanimoto of binary fingerprints."""
    a, b = train > 0, test > 0
    inter = b.astype(np.int32) @ a.T.astype(np.int32)
    return inter / np.maximum(b.sum(1)[:, None] + a.sum(1)[None, :] - inter, 1)


def out_of_fold(x: np.ndarray, y: np.ndarray, folds: int = 5) -> np.ndarray:
    from sklearn.model_selection import KFold

    oof = np.zeros(len(y))
    for fit, held in KFold(folds, shuffle=True, random_state=0).split(x):
        oof[held] = make_model().fit(x[fit], y[fit]).predict(x[held])
    return oof


# ---------------------------------------------------------------- one task, all tables


def bit_gain(key: str, model: Any, contrib: np.ndarray) -> Any:
    """Two importances per bit of a fitted ECFP4 bit model, as a (task, bit, gain, shap_abs)
    table: LightGBM's split gain, and the mean |TreeSHAP| over the train compounds.

    ``contrib``: the model's TreeSHAP on the train compounds, (n_train, n_bits). The mean runs
    over every compound, with the bit on or off (an off bit has a contribution too).
    """
    import polars as pl

    gain = model.booster_.feature_importance(importance_type="gain")
    return pl.DataFrame(
        {
            "task": key,
            "bit": np.arange(len(gain), dtype=np.int32),
            "gain": gain.astype(np.float32),
            "shap_abs": np.abs(contrib).mean(0).astype(np.float32),
        }
    )


def compute_gain(mol):
    """Only the "gain" table for one task: refits the ECFP4 bit model of ``compute_task`` (same
    data, same seed, so the same model) without the rest."""
    import polars as pl

    tr = mol.filter(pl.col("split") == "train")
    x = fingerprints(tr["smiles"].to_list())
    model = make_model().fit(x, tr["y"].to_numpy())
    return bit_gain(mol["task"][0], model, model.predict(x, pred_contrib=True)[:, :-1])


def compute_bitlen(mol, lengths: list[int] | None = None):
    """The "bitlen" table for one task: the ECFP4 bit model refitted at each fingerprint length,
    with its test scores and how crowded the train set's bits are at that length (distinct
    substructures, mean per non-empty bit, empty bits, purity).

    ``usable_bits`` counts the bits LightGBM can split on: on in at least ``min_child_samples``
    train compounds and off in at least as many. ``frequent_substructures`` is the same count
    for the substructures before folding (the same at every length)."""
    import polars as pl
    from scipy.stats import spearmanr

    tr, te = mol.filter(pl.col("split") == "train"), mol.filter(pl.col("split") == "test")
    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    rows, least = [], make_model().min_child_samples

    def splittable(n_on: np.ndarray) -> int:
        return int(((n_on >= least) & (n_on <= len(ytr) - least)).sum())

    for n_bits in lengths or LENGTHS:
        x = fingerprints(tr["smiles"].to_list(), n_bits=n_bits)
        model = make_model().fit(x, ytr)
        p = model.predict(fingerprints(te["smiles"].to_list(), n_bits=n_bits))
        census = bit_census(tr["smiles"].to_list(), 2, n_bits)
        envs = census.n_envs
        rows.append(
            {
                "task": mol["task"][0],
                "n_bits": n_bits,
                "rho": float(spearmanr(p, yte)[0]),
                "r2": float(1 - ((p - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()),
                "mae": float(np.abs(p - yte).mean()),
                "substructures": int(envs.sum()),
                "per_bit": float(envs[envs > 0].mean()),
                "empty_bits": int((envs == 0).sum()),
                "purity": float(census.purity()[1]),
                "usable_bits": splittable(x.sum(0)),
                "frequent_substructures": splittable(
                    np.array([e["count"] for envs in census.examples.values() for e in envs])
                ),
            }
        )
    return pl.DataFrame(rows)


def compute_task(mol):
    """Every precomputed table for one task, from its ``load_task`` rows.

    Returns {"neighbours", "predictions", "metrics", "shap", "gain", "bitlen"} as polars
    DataFrames.
    """
    import polars as pl
    from scipy.stats import pearsonr, spearmanr

    key = mol["task"][0]
    tr, te = mol.filter(pl.col("split") == "train"), mol.filter(pl.col("split") == "test")
    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    ftr, fte = features(tr["smiles"].to_list()), features(te["smiles"].to_list())

    sim = tanimoto(ftr["ECFP4 bit"], fte["ECFP4 bit"])
    nn, nn_sim = sim.argmax(1), sim.max(1)
    neighbours = pl.DataFrame(
        {
            "task": key,
            "test_id": te["id"],
            "nn_id": tr["id"][nn],
            "tanimoto": nn_sim,
            "y_test": yte,
            "y_nn": ytr[nn],
        }
    )

    preds, metrics = [], {"task": key, "n_train": len(ytr), "n_test": len(yte)}
    metrics |= {
        "test_sd": float(yte.std()),
        "nn_tanimoto_median": float(np.median(nn_sim)),
        "share_nn_ge_06": float(np.mean(nn_sim >= 0.6)),
        "n_identical_fp": int((nn_sim >= 0.999).sum()),
        "rho_1nn": float(spearmanr(ytr[nn], yte)[0]),
    }
    shap = gain = None
    for name in FEATURES:
        model = make_model().fit(ftr[name], ytr)
        p_test = model.predict(fte[name])
        p_oof = out_of_fold(ftr[name], ytr)
        preds.append(
            pl.DataFrame(
                {
                    "task": key,
                    "features": name,
                    "id": pl.concat([te["id"], tr["id"]]),
                    "split": ["test"] * len(yte) + ["train"] * len(ytr),
                    "y": np.concatenate([yte, ytr]),
                    "pred": np.concatenate([p_test, p_oof]),
                }
            )
        )
        metrics[f"rho {name}"] = float(spearmanr(p_test, yte)[0])
        metrics[f"r2 {name}"] = float(
            1 - ((p_test - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
        )
        metrics[f"mae {name}"] = float(np.abs(p_test - yte).mean())
        dt, dp = yte - ytr[nn], p_test - p_oof[nn]
        metrics[f"pair_slope {name}"] = float(np.polyfit(dt, dp, 1)[0])
        metrics[f"pair_r {name}"] = float(pearsonr(dp, dt)[0])
        if name == "ECFP4 bit":
            # TreeSHAP of the bit model on test and train compounds, on-bits only (long format)
            x = np.vstack([fte[name], ftr[name]])
            c = model.predict(x, pred_contrib=True)
            gain = bit_gain(key, model, c[len(yte) :, :-1])
            ids = pl.concat([te["id"], tr["id"]]).to_numpy()
            rows, bits = np.nonzero(x > 0)
            shap = pl.DataFrame(
                {
                    "task": key,
                    "id": ids[rows],
                    "bit": bits.astype(np.int32),
                    "shap": c[rows, bits].astype(np.float32),
                }
            ).vstack(
                pl.DataFrame(
                    {
                        "task": key,
                        "id": ids,
                        "bit": np.full(len(ids), -1, dtype=np.int32),  # -1 = expected value
                        "shap": c[:, -1].astype(np.float32),
                    }
                )
            )
    return {
        "neighbours": neighbours,
        "predictions": pl.concat(preds),
        "metrics": pl.DataFrame([metrics]),
        "shap": shap,
        "gain": gain,
        "bitlen": compute_bitlen(mol),
    }


def precompute(
    out_dir: Path,
    data_dir: Path | None = None,
    tasks: list[Task] | None = None,
    only: str | None = None,
) -> None:
    """Write results/precomputed/<table>.parquet for every task.

    ``only``: "gain" or "bitlen" writes just that table, refitting only the models it needs. It
    takes its rows from the molecules.parquet already in ``out_dir``, so the refitted models see
    the same rows in the same order as the ones behind the other tables.
    """
    import polars as pl

    single = {"gain": compute_gain, "bitlen": compute_bitlen}
    tables: dict[str, list] = {t: [] for t in ([only] if only else TABLES)}
    stored = pl.read_parquet(out_dir / "molecules.parquet") if only else None
    raws: dict[str, Any] = {}
    for task in tasks or TASKS:
        if only is not None and stored is not None:
            mol = stored.filter(pl.col("task") == task.key)
            tables[only].append(single[only](mol))
            print(task.key, mol.height, flush=True)
            continue
        if task.dataset not in raws:
            raws[task.dataset] = load_raw(task.dataset, data_dir)
        mol = load_task(task, raw=raws[task.dataset])
        tables["molecules"].append(mol)
        for name, df in compute_task(mol).items():
            tables[name].append(df)
        print(task.key, mol.height, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, parts in tables.items():
        pl.concat(parts, how="diagonal_relaxed").write_parquet(
            out_dir / f"{name}.parquet", compression="zstd"
        )


def open_tables(local_dir: Path | None = None, url: str = PRECOMPUTED_URL) -> dict[str, str]:
    """Where each precomputed table can be read from: a local file, else the GitHub copy.

    Returns {table: path or URL}; a table that is not reachable is left out (compute it with
    ``compute_task`` instead).
    """
    import urllib.request

    out = {}
    for name in TABLES:
        if local_dir is not None and (local_dir / f"{name}.parquet").exists():
            out[name] = str(local_dir / f"{name}.parquet")
            continue
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url + f"{name}.parquet", method="HEAD"), timeout=10
            ):
                out[name] = url + f"{name}.parquet"
        except OSError:
            pass
    return out
