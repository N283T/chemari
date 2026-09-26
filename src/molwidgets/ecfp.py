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
