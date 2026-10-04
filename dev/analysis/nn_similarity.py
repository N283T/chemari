"""How close is "close" for ECFP4 on PXR? Test compounds vs their nearest training neighbour.

* 1-NN error by Tanimoto of the neighbour, and with poorly measured compounds removed
* how many bits a test compound and its neighbour share / differ, and how generic the shared
  bits are; how many bits one methyl changes
* are the bits most correlated with pEC50 stand-ins for logP?

Same data preparation as examples/ecfp_pxr.py. Writes results/nn_similarity/summary.md.

    uv run python dev/analysis/nn_similarity.py
"""

import numpy as np
from pair_shap import ROOT, load  # same data preparation
from rdkit import Chem
from rdkit.Chem import Crippen
from scipy.stats import spearmanr

from chemari import fingerprint_matrix

OUT = ROOT / "results" / "nn_similarity"


def main() -> None:
    train, test = load()
    xtr = fingerprint_matrix(train["smiles"].to_list(), 2, 2048) > 0
    xte = fingerprint_matrix(test["smiles"].to_list(), 2, 2048) > 0
    ytr, yte = train["pEC50"].to_numpy(), test["pEC50"].to_numpy()
    citr, cite = train["ci_width"].to_numpy(), test["ci_width"].to_numpy()
    inter = xte.astype(np.int32) @ xtr.T.astype(np.int32)
    s_all = inter / (xte.sum(1)[:, None] + xtr.sum(1)[None, :] - inter)
    nn, sim = s_all.argmax(1), s_all.max(1)
    err = np.abs(ytr[nn] - yte)
    rng = np.random.default_rng(0)
    rand = np.abs(rng.choice(ytr, 20000) - rng.choice(yte, 20000)).mean()
    good = (cite < 1) & (citr[nn] < 1)

    lines = ["# Nearest training neighbour of each test compound (ECFP4, 2048 bit)", ""]
    lines += [
        f"* test compounds: {len(yte)}; nearest-neighbour Tanimoto median {np.median(sim):.2f}",
        f"* test CI width: median {np.nanmedian(cite):.2f}, share > 1.5: {np.mean(cite > 1.5):.1%}",
        f"* random train–test pair: mean |Δ pEC50| {rand:.2f}",
        "",
        "| neighbour Tanimoto | n | mean \\|Δ pEC50\\| (1-NN error) | Spearman(1-NN, true) | same, both CI < 1 (n / ρ) |",
        "|---|---|---|---|---|",
    ]
    for lo, hi in [(0.0, 1.01), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]:
        m = (sim >= lo) & (sim < hi)
        g = m & good
        rho = spearmanr(ytr[nn][m], yte[m])[0] if m.sum() > 4 else np.nan
        rho_g = spearmanr(ytr[nn][g], yte[g])[0] if g.sum() > 4 else np.nan
        label = "all" if lo == 0 else f"{lo}–{min(hi, 1.0)}"
        lines.append(
            f"| {label} | {m.sum()} | {err[m].mean():.2f} | {rho:.2f} | {g.sum()} / {rho_g:.2f} |"
        )

    # what a Tanimoto of 0.4–0.6 is made of
    m = (sim >= 0.4) & (sim < 0.6)
    idx = np.flatnonzero(m)
    shared = xte[idx] & xtr[nn[idx]]
    differ = xte[idx] ^ xtr[nn[idx]]
    common = xtr.mean(0) >= 0.2
    generic = np.array([(s & common).sum() / max(s.sum(), 1) for s in shared])
    one_methyl = fingerprint_matrix(
        ["CCN(CC)CC(=O)Nc1c(C)cccc1C", "CCN(CC)CC(=O)Nc1c(C)cc(C)cc1C"], 2, 2048
    ).astype(bool)
    lines += [
        "",
        "## What Tanimoto 0.4–0.6 is made of",
        "",
        (
            f"* pairs: {m.sum()}; bits on in the test compound (median) {np.median(xte[idx].sum(1)):.0f}, "
            f"shared {np.median(shared.sum(1)):.0f}, differing {np.median(differ.sum(1)):.0f}"
        ),
        f"* shared bits that are set in ≥ 20% of train (generic pieces): {generic.mean():.0%} on average",
        (
            f"* one methyl on the benzene (3b pair): {int((one_methyl[0] ^ one_methyl[1]).sum())} of "
            f"{int(one_methyl[0].sum())} bits change, Tanimoto "
            f"{(one_methyl[0] & one_methyl[1]).sum() / (one_methyl[0] | one_methyl[1]).sum():.2f}"
        ),
    ]

    # bits most correlated with pEC50: do they just track logP?
    logp = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in train["smiles"]])  # ty: ignore[unresolved-attribute]
    freq = xtr.mean(0)
    ok = np.flatnonzero((freq > 0.02) & (freq < 0.98))
    r_y = np.array([spearmanr(xtr[:, j], ytr)[0] for j in ok])
    top = ok[np.argsort(-np.abs(r_y))[:20]]
    r_top = np.array([spearmanr(xtr[:, j], ytr)[0] for j in top])
    r_lp = np.array([spearmanr(xtr[:, j], logp)[0] for j in top])
    resid = ytr - np.polyval(np.polyfit(logp, ytr, 1), logp)
    r_res = np.array([spearmanr(xtr[:, j], resid)[0] for j in top])
    lines += [
        "",
        "## Are the pEC50-correlated bits stand-ins for logP?",
        "",
        f"* logP vs pEC50: Spearman {spearmanr(logp, ytr)[0]:.2f} (train)",
        f"* top 20 bits by |Spearman with pEC50| (set in 2–98% of train): mean |ρ| {np.abs(r_top).mean():.2f}",
        f"* same sign with logP: {np.mean(np.sign(r_lp) == np.sign(r_top)):.0%}; mean |ρ with logP| {np.abs(r_lp).mean():.2f}",
        f"* vs pEC50 after removing a linear logP trend: mean |ρ| {np.abs(r_res).mean():.2f}",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
