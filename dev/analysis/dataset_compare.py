"""Is PXR hard for ECFP4, or hard for everything? The same checks on other OpenADMET datasets.

Datasets (their own train/test splits, test labels public):

* PXR challenge: pEC50
* ASAP / Polaris / OpenADMET antiviral challenge: Mpro pIC50 (MERS-CoV, SARS-CoV-2) and ADMET
* OpenADMET-ExpansionRx challenge: ADMET of a lead-optimisation program

Per endpoint:

* LightGBM (the notebook's settings) on ECFP4 bit, ECFP4 count, RDKit descriptors, bit + descriptors:
  test Spearman, R², MAE
* 1-NN (ECFP4 Tanimoto): test Spearman, and by neighbour similarity
* test compound vs nearest training neighbour: how much of the measured difference each model
  predicts (slope of Δpred on Δtrue; neighbour predicted out of fold)

Endpoints measured on a ratio scale (solubility, clearance, permeability, % unbound, IC50 in µM)
are log10-transformed; LogD and pIC50/pEC50 are used as they are. Duplicate structures within a
split are averaged. Writes results/dataset_compare/metrics.csv and summary.md.

    uv run python dev/analysis/dataset_compare.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

from molwidgets import standardize_smiles

sys.path.insert(0, str(Path(__file__).parent))
from pair_shap import ROOT, make_model

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]
OUT = ROOT / "results" / "dataset_compare"
OTHER = ROOT / "data" / "other"
HF = "https://huggingface.co/datasets/openadmet/"
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def read(local: Path, remote: str) -> pl.DataFrame:
    if local.exists():
        return pl.read_csv(local, infer_schema_length=10000)
    return pl.read_csv(HF + remote, infer_schema_length=10000)


def tasks() -> list[tuple[str, str, pl.DataFrame, pl.DataFrame, bool]]:
    """(dataset, endpoint, train, test, log10) with columns smiles, y."""
    out = []
    # PXR
    pxr = {
        "train": ["pxr-challenge_TRAIN.csv"],
        "test": [
            "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
            "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
        ],
    }
    parts = {
        k: pl.concat(
            [
                read(ROOT / "data" / f, f"pxr-challenge-train-test/resolve/main/{f}").select(
                    pl.col("SMILES").alias("smiles"), pl.col("pEC50").alias("y")
                )
                for f in fs
            ]
        )
        for k, fs in pxr.items()
    }
    out.append(("PXR", "pEC50", parts["train"], parts["test"], False))
    # ASAP
    asap = "ASAP_Polaris_OpenADMET_challenge"
    for fname, cols in [
        ("Potency.csv", {"pIC50 (MERS-CoV Mpro)": False, "pIC50 (SARS-CoV-2 Mpro)": False}),
        ("ADMET.csv", {"LogD": False, "KSOL": True, "HLM": True, "MLM": True, "MDR1-MDCKII": True}),
    ]:
        df = read(OTHER / f"{asap}_{fname}", f"{asap}/resolve/main/{fname}").with_columns(
            pl.col("CXSMILES").str.split(" ").list.first().alias("smiles")
        )
        for c, log in cols.items():
            sel = df.select("smiles", pl.col(c).alias("y"), "Set")
            out.append(
                (
                    "ASAP",
                    c,
                    sel.filter(pl.col("Set") == "Train").drop("Set"),
                    sel.filter(pl.col("Set") == "Test").drop("Set"),
                    log,
                )
            )
    # ExpansionRx
    exp = "openadmet-expansionrx-challenge-data"
    split = {
        k: read(
            OTHER / f"{exp}_expansion_data_{k}.csv", f"{exp}/resolve/main/expansion_data_{k}.csv"
        )
        for k in ["train", "test"]
    }
    for c in [
        "LogD",
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A>B",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
    ]:
        tr, te = (
            split[k].select(pl.col("SMILES").alias("smiles"), pl.col(c).alias("y"))
            for k in ["train", "test"]
        )
        out.append(("ExpansionRx", c, tr, te, c != "LogD"))
    return out


def clean(df: pl.DataFrame, log: bool) -> pl.DataFrame:
    df = df.drop_nulls().with_columns(pl.col("y").cast(pl.Float64))
    if log:
        df = df.filter(pl.col("y") > 0).with_columns(pl.col("y").log10())
    df = df.with_columns(
        pl.col("smiles").map_elements(standardize_smiles, return_dtype=pl.Utf8)
    ).drop_nulls()
    return df.group_by("smiles", maintain_order=True).agg(pl.col("y").mean())


_desc_cache: dict[str, np.ndarray] = {}


def features(smiles: list[str]) -> dict[str, np.ndarray]:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    bit = np.array([GEN.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    count = np.array([GEN.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
    desc = []
    for s, m in zip(smiles, mols):
        if s not in _desc_cache:
            _desc_cache[s] = np.array(
                list(Descriptors.CalcMolDescriptors(m).values()), dtype=np.float64
            )
        desc.append(_desc_cache[s])
    d = np.nan_to_num(np.clip(np.array(desc), -1e6, 1e6)).astype(np.float32)
    return {
        "ECFP4 bit": bit,
        "ECFP4 count": count,
        "RDKit desc": d,
        "bit + desc": np.hstack([bit, d]),
    }


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a > 0, b > 0
    inter = b.astype(np.int32) @ a.T.astype(np.int32)
    return inter / (b.sum(1)[:, None] + a.sum(1)[None, :] - inter)


def pair_slope(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, nn: np.ndarray
) -> tuple[float, float]:
    """Slope and r of Δpred on Δtrue for test vs nearest neighbour; neighbour out of fold."""
    oof = np.zeros(len(ytr))
    for fit, held in KFold(5, shuffle=True, random_state=0).split(xtr):
        oof[held] = make_model().fit(xtr[fit], ytr[fit]).predict(xtr[held])
    pte = make_model().fit(xtr, ytr).predict(xte)
    dt, dp = yte - ytr[nn], pte - oof[nn]
    return float(np.polyfit(dt, dp, 1)[0]), float(pearsonr(dp, dt)[0])


def main() -> None:
    rows = []
    for ds, ep, tr, te, log in tasks():
        tr, te = clean(tr, log), clean(te, log)
        te = te.filter(~pl.col("smiles").is_in(tr["smiles"].implode()))
        if tr.height < 100 or te.height < 30:
            continue
        ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
        ftr, fte = features(tr["smiles"].to_list()), features(te["smiles"].to_list())
        sim = tanimoto(ftr["ECFP4 bit"], fte["ECFP4 bit"])
        nn, nn_sim = sim.argmax(1), sim.max(1)
        row = {
            "dataset": ds,
            "endpoint": ep,
            "log10": log,
            "n_train": tr.height,
            "n_test": te.height,
            "test_sd": float(yte.std()),
            "nn_tanimoto_median": float(np.median(nn_sim)),
            "share_nn_ge_0.6": float(np.mean(nn_sim >= 0.6)),
            "1nn_rho": float(spearmanr(ytr[nn], yte)[0]),
            "1nn_rho_sim_ge_0.6": float(spearmanr(ytr[nn][nn_sim >= 0.6], yte[nn_sim >= 0.6])[0])
            if (nn_sim >= 0.6).sum() >= 10
            else np.nan,
        }
        for name in ftr:
            pred = make_model().fit(ftr[name], ytr).predict(fte[name])
            row[f"rho {name}"] = float(spearmanr(pred, yte)[0])
            row[f"r2 {name}"] = float(
                1 - ((pred - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
            )
            row[f"mae {name}"] = float(np.abs(pred - yte).mean())
        for name in ["ECFP4 bit", "RDKit desc"]:
            slope, r = pair_slope(ftr[name], ytr, fte[name], yte, nn)
            row[f"pair slope {name}"], row[f"pair r {name}"] = slope, r
        rows.append(row)
        print(ds, ep, {k: round(v, 2) for k, v in row.items() if isinstance(v, float)}, flush=True)

    df = pl.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.write_csv(OUT / "metrics.csv")
    (OUT / "summary.md").write_text(summarise(df))
    print((OUT / "summary.md").read_text())


def summarise(df: pl.DataFrame) -> str:
    def md(d: pl.DataFrame) -> str:
        cols = d.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in d.iter_rows():
            lines.append(
                "| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r) + " |"
            )
        return "\n".join(lines)

    perf = df.select(
        "dataset",
        "endpoint",
        "n_train",
        "n_test",
        pl.col("rho ECFP4 bit").alias("ρ bit"),
        pl.col("rho ECFP4 count").alias("ρ count"),
        pl.col("rho RDKit desc").alias("ρ desc"),
        pl.col("rho bit + desc").alias("ρ bit+desc"),
        pl.col("r2 ECFP4 bit").alias("R² bit"),
        pl.col("r2 RDKit desc").alias("R² desc"),
    )
    near = df.select(
        "dataset",
        "endpoint",
        pl.col("nn_tanimoto_median").alias("NN Tanimoto (median)"),
        pl.col("share_nn_ge_0.6").alias("share NN ≥ 0.6"),
        pl.col("1nn_rho").alias("1-NN ρ"),
        pl.col("1nn_rho_sim_ge_0.6").alias("1-NN ρ (NN ≥ 0.6)"),
        pl.col("pair slope ECFP4 bit").alias("pair slope bit"),
        pl.col("pair slope RDKit desc").alias("pair slope desc"),
        pl.col("pair r ECFP4 bit").alias("pair r bit"),
        pl.col("pair r RDKit desc").alias("pair r desc"),
    )
    return f"""# ECFP4 on PXR vs other OpenADMET datasets

Generated by `dev/analysis/dataset_compare.py`. LightGBM with the notebook's settings; each
dataset's own train/test split; test compounds also in train are dropped.

## Test performance (Spearman ρ, R²)

{md(perf)}

## Nearest neighbours and pair differences

* pair slope: slope of (predicted test − neighbour) on (measured test − neighbour); 1 = the
  model reproduces the size of the difference, 0 = it predicts no difference. Neighbour
  predicted out of fold.

{md(near)}
"""


if __name__ == "__main__":
    main()
