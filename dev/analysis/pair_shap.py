"""Why does the ECFP4 model miss on PXR? Test compound vs its nearest training neighbour.

For each test compound and its ECFP4 nearest neighbour in train, compare the measured
difference in pEC50 with the model's predicted difference, and split the predicted difference
with TreeSHAP into the bits the two molecules do not share and the bits they share.

The training neighbour's prediction is taken out of fold (5-fold CV on train): the model that
predicts a neighbour never saw it, so the predicted difference does not borrow the neighbour's
own label. The in-sample version is kept for comparison.

Same data preparation and LightGBM settings as the former PXR notebook
(examples/ecfp_pxr.py, section 4; removed).
Writes results/pair_shap/pairs.csv and results/pair_shap/summary.md.

    uv run python dev/analysis/pair_shap.py
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

from chemari import census_for, fingerprint_matrix, molecule_bit_tiles, standardize_smiles

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "pair_shap"
N_BITS = 2048
CI_LO, CI_HI = "pEC50_ci.lower (-log10(molarity))", "pEC50_ci.upper (-log10(molarity))"
FILES = {
    "train": "pxr-challenge_TRAIN.csv",
    "test_p1": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    "test_p2": "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
}


def load() -> tuple[pl.DataFrame, pl.DataFrame]:
    raw = pl.concat(
        [
            pl.read_csv(ROOT / "data" / f)
            .select(
                pl.col("Molecule Name").alias("id"),
                pl.col("SMILES").alias("smiles_raw"),
                "pEC50",
                (pl.col(CI_HI) - pl.col(CI_LO)).alias("ci_width"),
            )
            .with_columns(pl.lit("train" if k == "train" else "test").alias("split"))
            for k, f in FILES.items()
        ]
    )
    data = raw.with_columns(
        pl.col("smiles_raw").map_elements(standardize_smiles, return_dtype=pl.Utf8).alias("smiles")
    )
    return data.filter(pl.col("split") == "train"), data.filter(pl.col("split") == "test")


def make_model() -> lgb.LGBMRegressor:
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


def main() -> None:
    train, test = load()
    xtr = fingerprint_matrix(train["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    xte = fingerprint_matrix(test["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    ytr, yte = train["pEC50"].to_numpy(), test["pEC50"].to_numpy()
    model = make_model().fit(xtr, ytr)

    # nearest training neighbour of each test compound (ECFP4 Tanimoto)
    a, b = xtr > 0, xte > 0
    inter = b.astype(np.int32) @ a.T.astype(np.int32)
    sim_all = inter / (b.sum(1)[:, None] + a.sum(1)[None, :] - inter)
    nn, sim = sim_all.argmax(1), sim_all.max(1)

    # out-of-fold TreeSHAP for every training compound
    contrib_oof = np.zeros((len(xtr), N_BITS + 1), dtype=np.float64)
    for fit_idx, held in KFold(5, shuffle=True, random_state=0).split(xtr):
        m = make_model().fit(xtr[fit_idx], ytr[fit_idx])
        contrib_oof[held] = m.predict(xtr[held], pred_contrib=True)
    c_test = model.predict(xte, pred_contrib=True)
    c_nn_fit = model.predict(xtr[nn], pred_contrib=True)
    c_nn = contrib_oof[nn]

    d_true = yte - ytr[nn]
    d_pred = c_test.sum(1) - c_nn.sum(1)  # out-of-fold neighbour
    d_pred_fit = c_test.sum(1) - c_nn_fit.sum(1)  # neighbour predicted in sample
    only = b ^ a[nn]
    shared = b & a[nn]
    # NB: the test model and the fold model differ, so their SHAP values are compared per bit
    per_bit = (c_test - c_nn)[:, :-1]
    from_only = (per_bit * only).sum(1)
    from_shared = (per_bit * shared).sum(1)
    from_rest = per_bit.sum(1) - from_only - from_shared
    base_shift = (c_test - c_nn)[:, -1]

    # bits only the test compound sets whose own substructure is rare in train (<= 3 molecules):
    # the weight of such a bit was learned from other substructures folded into it
    census = census_for(train["smiles"].to_list(), 2, N_BITS)
    count = {(bit, e["uid"]): e["count"] for bit, ex in census.examples.items() for e in ex}
    borrowed = np.zeros(len(xte))
    for i, smi in enumerate(test["smiles"].to_list()):
        own: dict[int, int] = {}
        for t in molecule_bit_tiles(smi, 2, N_BITS):
            own[t["bit"]] = max(own.get(t["bit"], 0), count.get((t["bit"], t["uid"]), 0))
        bits = np.flatnonzero(b[i] & ~a[nn[i]])
        total = np.abs(per_bit[i, bits]).sum()
        rare = [x for x in bits if own.get(x, 0) <= 3]
        borrowed[i] = np.abs(per_bit[i, rare]).sum() / total if total > 0 else 0.0

    pairs = pl.DataFrame(
        {
            "test_id": test["id"],
            "nn_id": train["id"][nn],
            "tanimoto": sim.round(3),
            "pEC50_test": yte,
            "pEC50_nn": ytr[nn],
            "ci_test": test["ci_width"],
            "ci_nn": train["ci_width"][nn],
            "d_true": d_true.round(3),
            "d_pred": d_pred.round(3),
            "d_pred_in_sample": d_pred_fit.round(3),
            "from_unshared_bits": from_only.round(3),
            "from_shared_bits": from_shared.round(3),
            "from_absent_bits": from_rest.round(3),
            "baseline_shift": base_shift.round(3),
            "n_unshared_bits": only.sum(1),
            "borrowed_share": borrowed.round(3),
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.write_csv(OUT / "pairs.csv")
    (OUT / "summary.md").write_text(summarise(pairs))
    print((OUT / "summary.md").read_text())


def summarise(p: pl.DataFrame) -> str:
    dt, dp, dpf = p["d_true"].to_numpy(), p["d_pred"].to_numpy(), p["d_pred_in_sample"].to_numpy()
    sim = p["tanimoto"].to_numpy()
    big = np.abs(dt) > 1

    def stats(m: np.ndarray) -> str:
        if m.sum() < 5:
            return f"{m.sum()} | – | – | – | –"
        agree = np.mean(np.sign(dp[m & big]) == np.sign(dt[m & big])) if (m & big).any() else np.nan
        return (
            f"{m.sum()} | {pearsonr(dp[m], dt[m])[0]:.2f} | {np.polyfit(dt[m], dp[m], 1)[0]:.2f} | "
            f"{np.abs(dt[m]).mean():.2f} / {np.abs(dp[m]).mean():.2f} | {agree:.2f}"
        )

    head = "| pairs | n | r(Δpred, Δtrue) | slope Δpred~Δtrue | mean \\|Δtrue\\| / \\|Δpred\\| | sign agrees (\\|Δtrue\\|>1) |\n|---|---|---|---|---|---|\n"
    rows = [("all", np.ones(len(dt), bool))]
    rows += [
        (f"Tanimoto {lo}–{hi}", (sim >= lo) & (sim < hi))
        for lo, hi in [(0.3, 0.5), (0.5, 0.6), (0.6, 1.01)]
    ]
    ci_ok = (p["ci_test"].to_numpy() < 1) & (p["ci_nn"].to_numpy() < 1)
    rows += [("both CI < 1", ci_ok), ("either CI ≥ 1", ~ci_ok)]
    bs = p["borrowed_share"].to_numpy()
    rows += [("borrowed share < 0.3", bs < 0.3), ("borrowed share ≥ 0.3", bs >= 0.3)]
    table = head + "\n".join(f"| {name} | {stats(m)} |" for name, m in rows)

    un, sh = p["from_unshared_bits"].to_numpy(), p["from_shared_bits"].to_numpy()
    share_un = np.median(np.abs(un) / (np.abs(un) + np.abs(sh) + 1e-9))
    return f"""# Test compound vs nearest training neighbour: measured vs predicted difference

Model: LightGBM on 2048-bit ECFP4, as in section 4. Neighbour predicted out of fold (5-fold).
Δ = test − neighbour. Generated by `dev/analysis/pair_shap.py`.

{table}

* In-sample neighbour instead (for comparison): r(Δpred, Δtrue) = {pearsonr(dpf, dt)[0]:.2f}, slope {np.polyfit(dt, dpf, 1)[0]:.2f}
* Share of |Δpred| carried by the bits the two molecules do not share (median): {share_un:.2f}
* Spearman(test prediction, test pEC50) for reference: see section 4 of the notebook
* Median number of bits that differ: {int(np.median(p["n_unshared_bits"]))}
"""


if __name__ == "__main__":
    main()
