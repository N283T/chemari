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


def test_ecfp_trace_matches_rdkit_feature_count():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from molwidgets import ecfp_trace

    for smi in ["CC(=O)Nc1ccc(O)cc1", "C1CCNCC1", "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1"]:
        for r in (1, 2, 3):
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=r)
            rd = gen.GetSparseCountFingerprint(Chem.MolFromSmiles(smi)).GetNonzeroElements()
            assert len(ecfp_trace(smi, r).identifiers()) == len(rd)


def test_ecfp_trace_symmetric_atoms_share_identifiers():
    from molwidgets import ecfp_trace

    layer0 = ecfp_trace("c1ccccc1", 1).steps[0]
    assert len({s.identifier for s in layer0}) == 1


def test_stepper_renders_click_targets():
    from molwidgets import ECFPStepper

    w = ECFPStepper("CCO", max_radius=2)
    assert w.steps[1][1]["recipe_text"].startswith("B = hash(b |")
    assert w.svg.count('class="es-hit"') == 3
    w.atom, w.radius = 1, 2
    assert "<svg" in w.svg


def test_bit_tiles_merge_symmetry_and_flag_collisions():
    from molwidgets import molecule_bit_tiles

    acid = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"  # ibuprofen
    tiles = molecule_bit_tiles(acid, 2, 2048)
    methyl = [t for t in tiles if t["env"] == "[C;D1;H3]"]
    assert len(methyl) == 1 and methyl[0]["count"] == 3  # symmetric atoms, one identifier
    assert {t["env"] for t in tiles if t["bit"] == 807} == {"[C;D3;H0]", "[O;D1;H1]"}
    assert all(t["collides"] for t in tiles if t["bit"] == 807)
    assert not any(t["collides"] for t in molecule_bit_tiles(acid, 2, 8192) if t["radius"] == 0)


def test_bit_tiles_widget_gallery():
    from molwidgets import MorganBitTiles

    ref = ["CC(=O)O", "CCO", "c1ccccc1O"]
    w = MorganBitTiles("CC(=O)O", reference=ref, ids=["a", "b", "c"])
    assert w.tiles and "<svg" in w.tiles[0]["svg"]
    w.selected = w.tiles[0]["bit"]
    assert w.gallery and any(g["mine"] for g in w.gallery)
