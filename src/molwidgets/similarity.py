"""Similarity of two molecules by several representations and coefficients.

Fingerprints are compared with Tanimoto, Dice or cosine (all three written for count vectors, so
they reduce to the usual formulas on bits). MCES is RDKit's RASCAL (Johnson similarity of the
maximum common edge subgraph). "properties" compares whole-molecule descriptors: 1 / (1 + the
RMS difference of their z-scores).

Values of different methods sit on different scales (MACCS gives most unrelated pairs 0.4–0.7,
ECFP4 most below 0.3), so they should not be compared row against row.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem import rdMolDescriptors, rdRascalMCES
from rdkit.Chem.Scaffolds import MurckoScaffold

METRICS = ["Tanimoto", "Dice", "cosine"]


def _gen_fp(gen: Any, count: bool = False) -> Callable[[Chem.Mol], np.ndarray]:
    if count:
        return lambda m: gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return lambda m: gen.GetFingerprintAsNumPy(m).astype(np.float32)


def _maccs(m: Chem.Mol) -> np.ndarray:
    arr = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(rdMolDescriptors.GetMACCSKeysFingerprint(m), arr)
    return arr


@lru_cache(maxsize=1)
def _fingerprints() -> dict[str, tuple[str, Callable[[Chem.Mol], np.ndarray]]]:
    """key → (label, mol → vector)."""
    morgan = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    return {
        "ecfp4": ("ECFP4", _gen_fp(morgan)),
        "ecfp4_count": ("ECFP4 count", _gen_fp(morgan, count=True)),
        "ecfp6": ("ECFP6", _gen_fp(rfg.GetMorganGenerator(radius=3, fpSize=2048))),
        "fcfp4": (
            "FCFP4",
            _gen_fp(
                rfg.GetMorganGenerator(
                    radius=2, fpSize=2048, atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen()
                )
            ),
        ),
        "atompair": ("Atom pair", _gen_fp(rfg.GetAtomPairGenerator(fpSize=2048))),
        "torsion": (
            "Topological torsion",
            _gen_fp(rfg.GetTopologicalTorsionGenerator(fpSize=2048)),
        ),
        "rdkit": (
            "RDKit (paths ≤ 7 bonds)",
            _gen_fp(rfg.GetRDKitFPGenerator(fpSize=2048, maxPath=7)),
        ),
        "maccs": ("MACCS keys", _maccs),
    }


METHODS = [*_fingerprints.__wrapped__().keys(), "mces", "properties"]
OTHER_LABELS = {"mces": "MCES (RASCAL, Johnson)", "properties": "properties (11 descriptors)"}


def coefficient(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """Row-wise similarity of two equally shaped (n, d) or (d,) non-negative arrays."""
    lo = np.minimum(a, b).sum(-1)
    if metric == "Tanimoto":
        den = np.maximum(a, b).sum(-1)
    elif metric == "Dice":
        lo, den = 2 * lo, a.sum(-1) + b.sum(-1)
    elif metric == "cosine":
        lo, den = (a * b).sum(-1), np.sqrt((a * a).sum(-1) * (b * b).sum(-1))
    else:
        raise ValueError(f"unknown metric {metric!r}; use one of {METRICS}")
    return np.where(den > 0, lo / np.where(den > 0, den, 1), 0.0)


def mces_similarity(a: Chem.Mol, b: Chem.Mol, timeout: int = 1) -> float:
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.0
    opts.timeout = timeout
    found = rdRascalMCES.FindMCES(a, b, opts)
    return float(found[0].similarity) if found else 0.0


def _props(m: Chem.Mol) -> np.ndarray:
    from .widgets import _pair_properties

    return np.array([float(fn(m)) for _, _, fn, _, _ in _pair_properties()])


def _prop_scale(reference: Sequence[Chem.Mol] | None) -> np.ndarray:
    """Standard deviation per property: from the reference set, else a quarter of the typical
    drug-like range."""
    from .widgets import _pair_properties

    if reference:
        sd = np.array([_props(m) for m in reference]).std(0)
        if np.all(sd > 0):
            return sd
    return np.array([(hi - lo) / 4 for *_, (lo, hi) in _pair_properties()])


def property_similarity(pa: np.ndarray, pb: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (pa - pb) / scale
    return 1 / (1 + np.sqrt((z * z).mean(-1)))


def _scaffold(m: Chem.Mol, generic: bool) -> str:
    core = MurckoScaffold.GetScaffoldForMol(m)
    if generic:
        core = MurckoScaffold.MakeScaffoldGeneric(core)
    return Chem.MolToSmiles(core)


def pair_similarities(
    a: Chem.Mol, b: Chem.Mol, methods: Sequence[str], scale: np.ndarray
) -> dict[str, dict[str, float]]:
    """method → {metric: value}; MCES and properties have one value (key "")."""
    fps = _fingerprints()
    out: dict[str, dict[str, float]] = {}
    for key in methods:
        if key in fps:
            va, vb = fps[key][1](a), fps[key][1](b)
            out[key] = {m: float(coefficient(va, vb, m)) for m in METRICS}
        elif key == "mces":
            out[key] = {"": mces_similarity(a, b)}
        elif key == "properties":
            out[key] = {"": float(property_similarity(_props(a), _props(b), scale))}
    return out


def similarity_table(
    a: Chem.Mol, b: Chem.Mol, methods: Sequence[str] | None = None
) -> dict[str, Any]:
    """Everything MolPair's "similarity" view shows, as JSON-ready data."""
    methods = list(METHODS if methods is None else methods)
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown similarity method(s) {unknown}; built-in: {METHODS}")
    values = pair_similarities(a, b, methods, _prop_scale(None))
    fps = _fingerprints()
    return {
        "metrics": METRICS,
        "rows": [
            {"key": k, "label": fps[k][0] if k in fps else OTHER_LABELS[k], "values": values[k]}
            for k in methods
        ],
        # None when either molecule has no ring (no scaffold to compare)
        "scaffold": {
            kind: None
            if not (sa := _scaffold(a, generic)) or not (sb := _scaffold(b, generic))
            else sa == sb
            for kind, generic in [("murcko", False), ("generic", True)]
        },
    }
