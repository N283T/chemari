"""RDKit helpers shared by the widgets: standardization and Morgan bit bookkeeping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # ty: ignore[unresolved-attribute]


@lru_cache(maxsize=1)
def _standardizers():
    return (
        rdMolStandardize.LargestFragmentChooser(),
        rdMolStandardize.Uncharger(),
    )


def standardize_smiles(smiles: str) -> str | None:
    """Return the canonical SMILES of the neutral parent (largest fragment), or None.

    Stereochemistry is kept. Salts/solvents are stripped by keeping the largest
    fragment, and charges are neutralized where a neutral form exists.
    """
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    chooser, uncharger = _standardizers()
    mol = uncharger.uncharge(chooser.choose(mol))
    return Chem.MolToSmiles(mol)


@lru_cache(maxsize=16)
def _generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def env_atoms_bonds(mol: Chem.Mol, center: int, radius: int) -> tuple[list[int], list[int]]:
    """Atoms and bonds that make up the circular environment of `center` at `radius`."""
    if radius == 0:
        return [center], []
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center))
    atoms = {center}
    for b in bonds:
        bond = mol.GetBondWithIdx(b)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return sorted(atoms), bonds


def env_smiles(mol: Chem.Mol, center: int, radius: int) -> str:
    """A readable SMILES for the environment (atoms outside the env are dropped)."""
    atoms, bonds = env_atoms_bonds(mol, center, radius)
    if not bonds:
        # Radius 0 = the atom invariants Morgan hashes: element, degree, Hs, charge, ring.
        atom = mol.GetAtomWithIdx(center)
        sym = atom.GetSymbol().lower() if atom.GetIsAromatic() else atom.GetSymbol()
        label = f"[{sym};D{atom.GetDegree()};H{atom.GetTotalNumHs()}"
        if atom.GetFormalCharge():
            label += f";{atom.GetFormalCharge():+d}"
        return label + (";R]" if atom.IsInRing() else "]")
    return Chem.MolFragmentToSmiles(
        mol, atomsToUse=atoms, bondsToUse=bonds, rootedAtAtom=center, canonical=True
    )


def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> list[dict]:
    """All set bits of a folded Morgan fingerprint with the environments that set them.

    Each entry: {"bit", "envs": [{"center", "radius", "atoms", "bonds", "smiles"}]}.
    A bit with more than one distinct env SMILES is a collision *inside* this molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    _generator(radius, n_bits).GetFingerprint(mol, additionalOutput=ao)
    out = []
    for bit, envs in sorted(ao.GetBitInfoMap().items()):
        rows = []
        for center, rad in envs:
            atoms, bonds = env_atoms_bonds(mol, center, rad)
            rows.append(
                {
                    "center": center,
                    "radius": rad,
                    "atoms": atoms,
                    "bonds": bonds,
                    "smiles": env_smiles(mol, center, rad),
                }
            )
        out.append({"bit": int(bit), "envs": rows})
    return out


def fingerprint_matrix(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Binary Morgan fingerprints as a (n, n_bits) uint8 matrix."""
    gen = _generator(radius, n_bits)
    mat = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mat[i] = gen.GetFingerprintAsNumPy(mol)
    return mat


def tanimoto_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Tanimoto similarity between binary fingerprint matrices."""
    b = a if b is None else b
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    inter = a32 @ b32.T
    union = a32.sum(1)[:, None] + b32.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


@dataclass
class BitCensus:
    """Dataset-level view of each folded bit: which distinct environments land in it.

    Folded Morgan bits are `hash(environment) % n_bits`, so different substructures
    can share a bit (a collision). `examples[bit]` keeps one representative env per
    distinct unfolded identifier.
    """

    radius: int
    n_bits: int
    n_mols: int
    on: np.ndarray  # (n_mols, n_bits) uint8
    examples: dict[int, list[dict]] = field(default_factory=dict)

    @property
    def n_envs(self) -> np.ndarray:
        counts = np.zeros(self.n_bits, dtype=int)
        for bit, envs in self.examples.items():
            counts[bit] = len(envs)
        return counts

    def stats(self, y: np.ndarray | None = None) -> dict[int, dict]:
        """Per-bit frequency, collision count and (optionally) activity contrast."""
        freq = self.on.sum(0)
        n_envs = self.n_envs
        out: dict[int, dict] = {}
        for bit in np.flatnonzero(freq):
            row = {"n_on": int(freq[bit]), "n_envs": int(n_envs[bit])}
            if y is not None:
                mask = self.on[:, bit].astype(bool)
                row["mean_on"] = float(np.nanmean(y[mask]))
                row["mean_off"] = float(np.nanmean(y[~mask])) if (~mask).any() else None
            out[int(bit)] = row
        return out


def bit_census(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> BitCensus:
    """Build a BitCensus using unfolded Morgan identifiers to detect collisions."""
    sparse_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    folded_gen = _generator(radius, n_bits)
    on = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    seen: dict[int, dict[int, dict]] = defaultdict(dict)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        on[i] = folded_gen.GetFingerprintAsNumPy(mol)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        sparse_gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
        for uid, envs in ao.GetBitInfoMap().items():
            bit = uid % n_bits
            if uid not in seen[bit]:
                center, rad = envs[0]
                seen[bit][uid] = {"mol_index": i, "center": center, "radius": rad, "count": 0}
            seen[bit][uid]["count"] += 1
    examples: dict[int, list[dict]] = {}
    for bit, by_uid in seen.items():
        rows = []
        for uid, ex in sorted(by_uid.items(), key=lambda kv: -kv[1]["count"]):
            mol = Chem.MolFromSmiles(smiles[ex["mol_index"]])
            rows.append(
                {**ex, "uid": int(uid), "smiles": env_smiles(mol, ex["center"], ex["radius"])}
            )
        examples[bit] = rows
    return BitCensus(radius=radius, n_bits=n_bits, n_mols=len(smiles), on=on, examples=examples)


def substructure_bits(
    query: str,
    smiles: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    max_examples: int = 6,
) -> dict:
    """Which folded Morgan bits a substructure sets, across a set of molecules.

    For every molecule that matches ``query`` (SMARTS, or SMILES as a fallback), each Morgan
    environment centred on a matched atom is classified as **inside** (all its atoms lie in the
    match: the pattern's own bits) or **context** (it reaches outside the match, so it also depends
    on the neighbours). Returns per-bit counts plus a few example molecules with atom maps.
    """
    q = Chem.MolFromSmarts(query) if query else None
    if q is None and query:
        q = Chem.MolFromSmiles(query)
    if q is None or q.GetNumAtoms() == 0:
        return {"valid": False, "n_match": 0, "bits": [], "examples": []}
    gen = _generator(radius, n_bits)
    per_bit: dict[int, dict] = {}
    examples: list[dict] = []
    n_match = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        match = mol.GetSubstructMatch(q)
        if not match:
            continue
        n_match += 1
        matched = set(match)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetFingerprint(mol, additionalOutput=ao)
        envs_here: dict[int, list[dict]] = {}
        for bit, envs in ao.GetBitInfoMap().items():
            for center, rad in envs:
                if center not in matched:
                    continue
                atoms, bonds = env_atoms_bonds(mol, center, rad)
                envs_here.setdefault(int(bit), []).append(
                    {
                        "center": center,
                        "radius": rad,
                        "atoms": atoms,
                        "bonds": bonds,
                        "kind": "inside" if set(atoms) <= matched else "context",
                    }
                )
        for bit, envs in envs_here.items():
            row = per_bit.setdefault(
                bit, {"bit": bit, "n_with": 0, "n_inside": 0, "radius": 99, "_envs": {}}
            )
            row["n_with"] += 1
            if any(e["kind"] == "inside" for e in envs):
                row["n_inside"] += 1
            row["radius"] = min(row["radius"], *(e["radius"] for e in envs))
            for e in envs:  # every distinct environment of the pattern that lands on this bit
                text = env_smiles(mol, e["center"], e["radius"])
                row["_envs"][text] = row["_envs"].get(text, 0) + 1
        if len(examples) < max_examples:
            examples.append(
                {"index": i, "smiles": smi, "match": sorted(matched), "bits": envs_here}
            )
    bits = []
    for row in per_bit.values():
        row["kind"] = "inside" if row["n_inside"] == row["n_with"] else "context"
        envs = sorted(row.pop("_envs").items(), key=lambda kv: -kv[1])
        # context environments vary by molecule: show the most common few
        row["env"] = " | ".join(t for t, _ in envs[: (4 if row["kind"] == "inside" else 2)])
        row["n_pattern_envs"] = len(envs)
        bits.append(row)
    bits.sort(key=lambda r: (r["kind"] != "inside", -r["n_with"], r["radius"]))
    return {"valid": True, "n_match": n_match, "bits": bits, "examples": examples}
