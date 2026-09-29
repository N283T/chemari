"""A readable re-implementation of the ECFP / Morgan algorithm, for teaching.

It follows RDKit's Morgan fingerprint step by step (same atom invariants, same neighbour
ordering, same duplicate-environment removal) but uses a different hash function, so the
identifiers are different numbers while the *set* of features is the same.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from rdkit import Chem

INVARIANT_NAMES = ("atomic number", "degree", "total Hs", "formal charge", "isotope", "in ring")


def small_hash(obj) -> int:
    """Deterministic 32-bit hash (stand-in for RDKit's boost::hash_combine)."""
    return int.from_bytes(hashlib.blake2b(repr(obj).encode(), digest_size=4).digest(), "little")


def atom_invariant(atom: Chem.Atom) -> tuple[int, ...]:
    """RDKit's default (ECFP) atom invariants."""
    return (
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        atom.GetIsotope(),
        int(atom.IsInRing()),
    )


@dataclass
class AtomStep:
    """One atom at one iteration."""

    atom: int
    radius: int
    identifier: int
    recipe: tuple  # what was hashed to get `identifier`
    atoms: list[int]
    bonds: list[int]
    status: str  # "new", "duplicate", "no growth"
    duplicate_of: int | None = None


@dataclass
class ECFPTrace:
    smiles: str
    radius: int
    steps: list[list[AtomStep]] = field(default_factory=list)  # steps[r][atom]

    @property
    def features(self) -> list[AtomStep]:
        return [s for layer in self.steps for s in layer if s.status == "new"]

    def identifiers(self) -> set[int]:
        return {s.identifier for s in self.features}


def _atoms_of(mol: Chem.Mol, center: int, bonds: frozenset[int]) -> list[int]:
    atoms = {center}
    for b in bonds:
        bond = mol.GetBondWithIdx(b)
        atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return sorted(atoms)


def ecfp_trace(smiles: str, radius: int = 2) -> ECFPTrace:
    """Run the Morgan algorithm and record every intermediate identifier.

    Iteration 0 hashes each atom's invariants. Iteration r hashes the atom's previous
    identifier together with the (bond order, identifier) pairs of its neighbours, sorted.
    An environment is kept as a feature only if its set of bonds has not been seen before;
    when several atoms cover the same bonds in one iteration, only one of them is kept.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES: {smiles}")
    trace = ECFPTrace(smiles=smiles, radius=radius)

    ids = [small_hash(atom_invariant(a)) for a in mol.GetAtoms()]
    env = [frozenset[int]() for _ in mol.GetAtoms()]
    trace.steps.append(
        [
            AtomStep(a.GetIdx(), 0, ids[a.GetIdx()], atom_invariant(a), [a.GetIdx()], [], "new")
            for a in mol.GetAtoms()
        ]
    )

    seen: dict[frozenset[int], int] = {}  # bond set -> atom that first emitted it
    for r in range(1, radius + 1):
        new_ids, new_env, layer = [], [], []
        for a in mol.GetAtoms():
            i = a.GetIdx()
            nbrs = tuple(
                sorted(
                    (int(b.GetBondTypeAsDouble() * 2), ids[b.GetOtherAtomIdx(i)])
                    for b in a.GetBonds()
                )
            )
            recipe = (r, ids[i], nbrs)
            bonds = set(env[i])
            for b in a.GetBonds():
                bonds.add(b.GetIdx())
                bonds |= env[b.GetOtherAtomIdx(i)]
            new_ids.append(small_hash(recipe))
            new_env.append(frozenset(bonds))
            layer.append(
                AtomStep(
                    i, r, new_ids[-1], recipe, _atoms_of(mol, i, new_env[-1]), sorted(bonds), ""
                )
            )
        # Deterministic tie-break among atoms covering the same bonds this round.
        for step in sorted(layer, key=lambda s: (s.bonds, s.identifier, s.atom)):
            key = frozenset(step.bonds)
            if not key or key == env[step.atom]:
                step.status = "no growth"
            elif key in seen:
                step.status = "duplicate"
                step.duplicate_of = seen[key]
            else:
                step.status = "new"
                seen[key] = step.atom
        trace.steps.append(layer)
        ids, env = new_ids, new_env
    return trace


_LABELS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _label(n: int) -> str:
    return _LABELS[n] if n < len(_LABELS) else _LABELS[n % len(_LABELS)] + str(n // len(_LABELS))


def _bit_info(mol: Chem.Mol, radius: int, redundant: bool) -> dict[tuple[int, int], int]:
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, includeRedundantEnvironments=redundant
    )
    out = rdFingerprintGenerator.AdditionalOutput()
    out.AllocateBitInfoMap()
    gen.GetSparseCountFingerprint(mol, additionalOutput=out)
    return {(a, r): ident for ident, envs in out.GetBitInfoMap().items() for a, r in envs}


def ecfp_story(smiles: str, radius: int = 2) -> dict:
    """Every step of ECFP for one molecule, as plain JSON-able data (drives ECFPStepper).

    Returns ``{"smiles", "radius", "layers"}``; ``layers[r][atom]`` holds the identifier, its
    letter, the environment's atoms and bonds, and whether it was kept ("new"), dropped as a
    duplicate (with ``dup_of``) or dropped because it stopped growing ("no growth"). Radius 0
    rows carry the six invariants (``inv``), later ones the sorted ``nbrs`` [bond order, atom].

    Identifiers are RDKit's own (``includeRedundantEnvironments=True`` yields one for every atom
    and radius, including the environments RDKit later drops); which ones are kept comes from the
    ordinary generator. Bond sets and neighbour lists come from :func:`ecfp_trace`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot parse SMILES: {smiles}")
    every = _bit_info(mol, radius, redundant=True)
    kept = set(_bit_info(mol, radius, redundant=False))
    trace = ecfp_trace(smiles, radius)

    labels: dict[int, str] = {}
    first_env: dict[frozenset[int], tuple[int, int]] = {}
    layers = []
    for r, layer in enumerate(trace.steps):
        # letters go to the kept identifiers first, as in the movie; dropped ones come last
        for s in sorted(layer, key=lambda s: (s.atom, r) not in kept):
            labels.setdefault(every[(s.atom, r)], _label(len(labels)))
        rows = [
            {
                "atom": s.atom,
                "id": every[(s.atom, r)],
                "label": labels[every[(s.atom, r)]],
                "atoms": s.atoms,
                "bonds": s.bonds,
            }
            for s in layer
        ]
        for row, s in zip(rows, layer):
            key = frozenset(s.bonds)
            if (s.atom, r) in kept:
                row["status"] = "new"
                first_env.setdefault(key, (s.atom, r))
            elif r and (not key or key == frozenset(trace.steps[r - 1][s.atom].bonds)):
                row["status"] = "no growth"
            else:
                row["status"] = "duplicate"
        for row, s in zip(rows, layer):
            if row["status"] == "duplicate":
                a, rr = first_env[frozenset(s.bonds)]
                row["dup_of"] = labels[every[(a, rr)]]
            if r == 0:
                row["inv"] = list(s.recipe)
            else:
                prev = {step.atom: every[(step.atom, r - 1)] for step in trace.steps[r - 1]}
                # RDKit sorts the neighbours by (bond type, identifier) before hashing; the bond
                # type is its enum value, so aromatic (12) comes after triple (3)
                around = sorted(
                    mol.GetAtomWithIdx(s.atom).GetBonds(),
                    key=lambda b: (int(b.GetBondType()), prev[b.GetOtherAtomIdx(s.atom)]),
                )
                row["nbrs"] = [[b.GetBondTypeAsDouble(), b.GetOtherAtomIdx(s.atom)] for b in around]
        layers.append(rows)

    return {"smiles": smiles, "radius": radius, "layers": layers}
