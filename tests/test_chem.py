import numpy as np

from molwidgets import (
    MorganExplorer,
    bit_census,
    fingerprint_matrix,
    morgan_bits,
    standardize_smiles,
    tanimoto_matrix,
)


def test_standardize_strips_salt_and_neutralizes():
    assert standardize_smiles("CC(=O)[O-].[Na+]") == "CC(=O)O"
    assert standardize_smiles("not a smiles") is None


def test_standardize_keeps_stereo():
    smi = standardize_smiles("C[C@H](N)C(=O)O")
    assert smi is not None and "@" in smi


def test_morgan_bits_match_fingerprint():
    smi = "c1ccccc1CCN"
    bits = {b["bit"] for b in morgan_bits(smi, 2, 1024)}
    fp = fingerprint_matrix([smi], 2, 1024)[0]
    assert bits == set(np.flatnonzero(fp))


def test_radius0_label_encodes_invariants():
    envs = [e for b in morgan_bits("CCO", 0, 2048) for e in b["envs"]]
    assert "[O;D1;H1]" in {e["smiles"] for e in envs}


def test_census_folding_matches_folded_fp():
    smiles = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "C1CCNCC1"]
    census = bit_census(smiles, 2, 64)
    assert (census.on == fingerprint_matrix(smiles, 2, 64)).all()
    assert census.n_envs.sum() >= census.on.any(0).sum()


def test_ring_size_homologues_share_bits():
    fps = fingerprint_matrix(["NC1CCCCC1", "NC1CCCCCC1"], 2, 2048)
    assert tanimoto_matrix(fps)[0, 1] == 1.0


def test_explorer_pair_payload_and_examples():
    ref = ["CCO", "CCN", "c1ccccc1O"]
    w = MorganExplorer(
        [{"id": "a", "smiles": "CCO"}, {"id": "b", "smiles": "CCN"}], reference=ref, y=[1, 2, 3]
    )
    assert len(w.payload) == 2
    bit = w.payload[0]["bits"][0]["bit"]
    w.selected_bit = bit
    assert w.bit_examples and all("parent_smiles" in e for e in w.bit_examples)
