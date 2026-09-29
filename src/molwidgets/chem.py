"""RDKit helpers shared by the widgets: standardization and Morgan bit bookkeeping."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

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


def molecule_bit_tiles(smiles: str, radius: int = 2, n_bits: int = 2048) -> list[dict]:
    """One entry per distinct Morgan identifier of a molecule, with the folded bit it sets.

    Symmetric atoms that share an identifier are merged (``count``); two *different*
    identifiers that fold onto the same bit are an in-molecule collision (``collides``).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    rdFingerprintGenerator.GetMorganGenerator(radius=radius).GetSparseCountFingerprint(
        mol, additionalOutput=ao
    )
    tiles: list[dict[str, Any]] = []
    for uid, envs in ao.GetBitInfoMap().items():
        center, rad = min(envs, key=lambda e: (e[1], e[0]))
        where = []  # every occurrence, so all matching atoms can be highlighted
        for c, r in envs:
            atoms, bonds = env_atoms_bonds(mol, c, r)
            where.append({"center": int(c), "atoms": atoms, "bonds": bonds})
        tiles.append(
            {
                "uid": int(uid),
                "bit": int(uid % n_bits),
                "radius": int(rad),
                "center": int(center),
                "count": len(envs),
                "env": env_smiles(mol, center, rad),
                "where": where,
            }
        )
    bits = [int(uid % n_bits) for uid in ao.GetBitInfoMap()]
    for t in tiles:
        t["collides"] = bits.count(int(t["uid"]) % n_bits) > 1
    tiles.sort(key=lambda t: (t["radius"], t["bit"], t["uid"]))
    return tiles


def find_mcs(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    timeout: float = 2.0,
    atoms: str = "elements",
    bonds: str = "order",
    ring_matches_ring: bool = True,
    complete_rings: bool = True,
) -> dict[str, Any]:
    """Maximum common substructure of two molecules, with RDKit's FMCS options by name.

    ``atoms``: "elements" (an atom matches only the same element) or "any" (any element, so a
    ring CH that became N stays matched). ``bonds``: "order" (aromatic matches aromatic),
    "order_exact" or "any". ``ring_matches_ring``: ring atoms/bonds match only ring ones;
    ``complete_rings``: a ring is matched whole or not at all.

    Returns ``pairs`` (atom pairs ``[(a, b), ...]``), ``bonds_a`` / ``bonds_b`` (matched bonds),
    ``smarts`` (the common part as SMARTS) and ``timed_out``; empty when nothing is shared.
    With ``atoms="any"``, of B's symmetric matches the one pairing the most identical elements
    is kept.
    """
    from rdkit.Chem import rdFMCS

    atom_cmp = {
        "elements": rdFMCS.AtomCompare.CompareElements,
        "any": rdFMCS.AtomCompare.CompareAny,
    }
    bond_cmp = {
        "order": rdFMCS.BondCompare.CompareOrder,
        "order_exact": rdFMCS.BondCompare.CompareOrderExact,
        "any": rdFMCS.BondCompare.CompareAny,
    }
    res = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=int(max(1, timeout)),
        ringMatchesRingOnly=ring_matches_ring,
        completeRingsOnly=complete_rings,
        atomCompare=atom_cmp[atoms],
        bondCompare=bond_cmp[bonds],
    )
    empty: dict[str, Any] = {
        "pairs": [],
        "bonds_a": [],
        "bonds_b": [],
        "smarts": "",
        "timed_out": bool(res.canceled),
    }
    query = Chem.MolFromSmarts(res.smartsString) if res.numAtoms else None
    if query is None:
        return empty
    hit_a = mol_a.GetSubstructMatch(query)
    if atoms == "any":
        hit_b = max(
            mol_b.GetSubstructMatches(query, uniquify=False, maxMatches=500),
            key=lambda h: sum(
                mol_a.GetAtomWithIdx(i).GetAtomicNum() == mol_b.GetAtomWithIdx(j).GetAtomicNum()
                for i, j in zip(hit_a, h)
            ),
            default=(),
        )
    else:
        hit_b = mol_b.GetSubstructMatch(query)
    if not hit_a or not hit_b:
        return empty
    matched: list[list[int]] = [[], []]
    for qb in query.GetBonds():
        for k, (m, hit) in enumerate(zip((mol_a, mol_b), (hit_a, hit_b))):
            b = m.GetBondBetweenAtoms(hit[qb.GetBeginAtomIdx()], hit[qb.GetEndAtomIdx()])
            if b is not None:
                matched[k].append(b.GetIdx())
    return {
        **empty,
        "pairs": [(int(i), int(j)) for i, j in zip(hit_a, hit_b)],
        "bonds_a": matched[0],
        "bonds_b": matched[1],
        "smarts": res.smartsString,
    }


def common_substructure(
    mol_a: Chem.Mol, mol_b: Chem.Mol, timeout: float = 2.0
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Maximum common substructure of two molecules (elements and bond orders must match,
    rings only with rings and only whole). Returns the atom pairs ``[(a, b), ...]`` and the
    matched bonds of A and of B; all empty when nothing is shared. See :func:`find_mcs` for
    the other options."""
    r = find_mcs(mol_a, mol_b, timeout)
    return r["pairs"], r["bonds_a"], r["bonds_b"]
