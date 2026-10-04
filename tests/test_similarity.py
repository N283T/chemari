import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from chemari.similarity import METHODS, coefficient, similarity_table

A, B = "CC(=O)Nc1ccc(O)cc1", "CCOc1ccc(NC(C)=O)cc1"


def test_coefficients_match_rdkit_on_bits():
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fa, fb = (gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in (A, B))
    va, vb = (gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(float) for s in (A, B))
    assert coefficient(va, vb, "Tanimoto") == pytest.approx(DataStructs.TanimotoSimilarity(fa, fb))
    assert coefficient(va, vb, "Dice") == pytest.approx(DataStructs.DiceSimilarity(fa, fb))
    assert coefficient(va, vb, "cosine") == pytest.approx(DataStructs.CosineSimilarity(fa, fb))


def test_coefficient_of_empty_vectors_is_zero():
    z = np.zeros(8)
    assert coefficient(z, z, "Tanimoto") == 0


def test_similarity_table():
    t = similarity_table(Chem.MolFromSmiles(A), Chem.MolFromSmiles(B))
    assert [r["key"] for r in t["rows"]] == METHODS
    for r in t["rows"]:
        for v in r["values"].values():
            assert 0 <= v <= 1
    assert t["scaffold"] == {"murcko": True, "generic": True}


def test_similarity_table_acyclic():
    t = similarity_table(
        Chem.MolFromSmiles("CC(=O)NC"), Chem.MolFromSmiles("CC(=O)NCC"), ["ecfp4", "mces"]
    )
    assert [r["key"] for r in t["rows"]] == ["ecfp4", "mces"]
    assert t["scaffold"]["murcko"] is None


def test_molpair_single_molecule():
    from chemari import MolPair

    w = MolPair({"id": "x", "smiles": A, "y": 1.5}, value_cols=["y"])
    assert len(w.data["sides"]) == 1 and w.data["similarity"] is None
    assert w.data["sides"][0]["values"] == {"y": 1.5}
    w.set_pair(A, B)
    assert len(w.data["sides"]) == 2 and 0 < w.data["similarity"] < 1


def test_molgrid_similarity_search():
    from chemari import MolGrid

    g = MolGrid(
        [
            {"id": "a", "smiles": "CCO"},
            {"id": "b", "smiles": "CCCO"},
            {"id": "c", "smiles": "c1ccccc1"},
        ]
    )
    assert g.similarity == {}
    g.similarity_query = "CCO"
    v = g.similarity["values"]
    assert v["a"] == 1.0 and v["a"] > v["b"] > v["c"]
    g.similarity_query = "not a smiles"
    assert g.similarity == {"error": True}
