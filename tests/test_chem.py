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
    assert w.svg.count('class="es-hit"') == 3
    w.atom, w.radius = 1, 2
    assert "<svg" in w.svg
    w.smiles = "not a smiles"
    assert w.error and w.steps  # the last good molecule stays


def test_ecfp_story_uses_rdkit_identifiers():
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    from molwidgets.ecfp import ecfp_story

    smi = "CC(=O)NC"  # the movie's molecule: letters a-j are its kept identifiers
    story = ecfp_story(smi, 2)
    kept = {
        row["label"]: row["id"]
        for layer in story["layers"]
        for row in layer
        if row["status"] == "new"
    }
    assert sorted(kept) == list("abcdefghij")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    assert set(kept.values()) == set(
        gen.GetSparseCountFingerprint(Chem.MolFromSmiles(smi)).GetNonzeroElements()
    )
    # the N-methyl carbon at radius 2 covers the same bonds as N at radius 1
    assert story["layers"][2][4]["status"] == "duplicate" and story["layers"][2][4]["dup_of"] == "h"


def test_ecfp_story_sorts_aromatic_bonds_last():
    from molwidgets.ecfp import ecfp_story

    row = ecfp_story("CC(=O)Nc1ccc(O)cc1", 1)["layers"][1][4]  # the aromatic C bonded to N
    assert [order for order, _ in row["nbrs"]] == [1.0, 1.5, 1.5]


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


def test_movie_uses_real_rdkit_identifiers():
    import re

    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as G

    from molwidgets import ECFPMovie

    page = ECFPMovie().page
    assert "Inside ECFP4" in page
    ids = {
        k: int(v)
        for k, v in re.findall(r"(\w): (\d{6,})", page.split("const IDS = {")[1].split("};")[0])
    }
    # (atom, radius) that each letter stands for in the storyboard
    where = {"a": (0, 0), "b": (1, 0), "c": (2, 0), "d": (3, 0), "e": (0, 1),
             "f": (1, 1), "g": (2, 1), "h": (3, 1), "i": (4, 1), "j": (3, 2)}  # fmt: skip
    ao = G.AdditionalOutput()
    ao.AllocateBitInfoMap()
    G.GetMorganGenerator(radius=2).GetSparseCountFingerprint(
        Chem.MolFromSmiles("CC(=O)NC"), additionalOutput=ao
    )
    info = ao.GetBitInfoMap()
    assert set(info) == set(ids.values())
    for key, env in where.items():
        assert env in info[ids[key]]
    # the comparison molecule's 16-bit vector
    bits = G.GetMorganGenerator(radius=2, fpSize=16).GetFingerprint(Chem.MolFromSmiles("CC(=O)NCC"))
    assert "[0, 1, 5, 6, 7, 8, 9, 10, 11, 13, 14]" in page
    assert list(bits.GetOnBits()) == [0, 1, 5, 6, 7, 8, 9, 10, 11, 13, 14]


def test_widgets_bump_rev_after_recomputing():
    from molwidgets import BitAtlas, ECFPStepper

    w = ECFPStepper("CCO")
    before, svg = w.rev, w.svg
    w.radius = 1
    assert w.rev > before and w.svg != svg  # the drawing is ready by the time rev moves
    atlas = BitAtlas(["CCO", "c1ccccc1O", "CC(=O)N"])
    before = atlas.rev
    atlas.sort = "envs"
    assert atlas.rev > before


def test_bit_importance_ranks_bits_and_shows_their_substructures():
    import numpy as np

    from molwidgets import BitImportance

    gain = np.zeros(64)
    gain[[5, 9]] = [1.0, 3.0]
    w = BitImportance(["CCO", "c1ccccc1O", "CC(=O)N"], importance={"gain": gain}, effect=gain - 1)
    assert w.sort == "gain" and w.n_bits == 64
    assert [r["bit"] for r in w.rows[:2]] == [9, 5]
    assert (
        w.rows[0]["rank"] == 1
        and w.rows[0]["scores"]["gain"] == 0.75
        and w.rows[0]["effect"] == 2.0
    )
    assert w.selected == 9 and w.detail["bit"] == 9  # the top bit is open from the start
    busy = next(r["bit"] for r in w.rows if r["n_envs"])
    w.selected = busy
    assert len(w.detail["envs"]) >= 1 and "<svg" in w.detail["envs"][0]["svg"]
    assert w.mols["total"] == w.detail["n_mols"] and w.mols["items"][0]["hits"]  # atoms to light up
    w.mol_filter = w.detail["envs"][0]["uid"]
    assert w.mols["total"] == w.detail["envs"][0]["count"]
    w.descending = False  # unused bits first, most frequent first
    assert w.rows[0]["scores"]["gain"] == 0
    w.page_size = 10
    w.page = 1
    assert w.rows[0]["rank"] == 11 and len(w.order) == 64  # every bit is ranked, a page at a time


def test_common_substructure_maps_atoms_and_bonds():
    from rdkit import Chem

    from molwidgets import common_substructure

    a, b = Chem.MolFromSmiles("Cc1ccccc1NC(C)=O"), Chem.MolFromSmiles("Cc1ccc(C)cc1NC(C)=O")
    pairs, bonds_a, bonds_b = common_substructure(a, b)
    assert len(pairs) == a.GetNumAtoms() and len(bonds_a) == len(bonds_b) == a.GetNumBonds()
    assert all(a.GetAtomWithIdx(i).GetSymbol() == b.GetAtomWithIdx(j).GetSymbol() for i, j in pairs)
    assert common_substructure(Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("[Na+]")) == (
        [],
        [],
        [],
    )


def test_mol_pair_widget():
    from molwidgets import MolPair

    w = MolPair(
        {"id": "a", "smiles": "CC(=O)Nc1ccc(O)cc1", "pEC50": 5.0},
        {"id": "b", "smiles": "CCOc1ccc(NC(C)=O)cc1", "pEC50": 6.0},
        value_cols=["pEC50"],
    )
    d = w.data
    assert not d["searched"] and d["mcs_atoms"] == 0 and 0 < d["similarity"] < 1
    w.show_common = True
    d = w.data
    assert d["searched"] and d["mcs_atoms"] == 11
    assert all("svg_common" in s for s in d["sides"])
    w.show_common = False  # the browser swaps drawings; Python does not recompute
    assert w.data is d
    assert d["sides"][1]["values"] == {"pEC50": 6.0}
    assert d["sides"][0]["props"]["HBD"] == 2
    w.align = True
    assert w.data["mcs_atoms"] == 11
    assert d["property_meta"][0]["range"] == [0, 600] and d["value_ranges"] == {}
    w2 = MolPair("CCO", "CCN", value_ranges={"pEC50": (3, 9)}, show_formula=False)
    assert w2.data["value_ranges"] == {"pEC50": [3.0, 9.0]} and not w2.show_formula
    d = w.data
    w.view = "common"  # already searched: no recompute
    assert w.data is d
    # paracetamol → phenacetin: the OH's hydrogen becomes an ethyl
    (site,) = d["edits"]["sites"]
    assert site["label"] == "R1" and site["smiles"] == ["", "CC[*:1]"] and site["change"] is None
    assert d["edits"]["core_smiles"] == "CC(=O)Nc1ccc(O[*:1])cc1"
    (site,) = MolPair(
        "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
        "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
        view="common",
    ).data["edits"]["sites"]
    assert site["kind"] == "stereo" and tuple(site["change"]) == ("R", "S")
    # benzene → pyridine: with any-element matching the ring stays common and C → N is a site
    p = MolPair("c1ccccc1CCN", "c1ccncc1CCN", view="common", mcs={"atoms": "any"})
    (site,) = p.data["edits"]["sites"]
    assert (
        p.data["mcs_atoms"] == 9
        and site["kind"] == "element"
        and tuple(site["change"]) == ("C", "N")
    )
    p.mcs = {}  # elements only: the ring is no longer common
    assert p.data["mcs_atoms"] < 9
    w.set_pair("CCO", "not a smiles")
    assert w.data["sides"][1]["valid"] is False and w.data["similarity"] is None


def test_copy_smiles_option():
    from molwidgets import MolGrid, MolPair

    assert MolPair("CCO", "CCN").copy_smiles
    assert not MolPair("CCO", "CCN", copy_smiles=False).copy_smiles
    assert not MolGrid([{"id": "a", "smiles": "CCO"}], copy_smiles=False).copy_smiles


def test_mol_pair_properties():
    import pytest

    from molwidgets import MolPair

    keys = [m["key"] for m in MolPair("CCO", "CCN").data["property_meta"]]
    assert keys == ["MW", "cLogP", "HBD", "HBA"]  # rule of five by default
    w = MolPair(
        "CCO",
        "CCCO",
        properties=[
            "QED",
            ("Carbons", lambda m: sum(a.GetSymbol() == "C" for a in m.GetAtoms()), 0, None),
        ],
    )
    meta = w.data["property_meta"]
    assert [m["key"] for m in meta] == ["QED", "Carbons"] and meta[1]["range"] is None
    assert [s["props"]["Carbons"] for s in w.data["sides"]] == [2, 3]
    with pytest.raises(ValueError, match="unknown property"):
        MolPair("CCO", "CCN", properties=["logP"])


def test_explorer_chirality():
    from molwidgets import MorganExplorer

    r, s = "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1", "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"
    w = MorganExplorer([{"smiles": r}, {"smiles": s}])

    def bits(k):
        return {b["bit"] for b in w.payload[k]["bits"]}

    assert bits(0) == bits(1)
    w.chirality = True
    assert bits(0) != bits(1)


def test_bit_purity():
    from molwidgets import bit_census

    census = bit_census(["CCO", "CCN", "CCO"], n_bits=2048)
    purity, mean = census.purity()
    used = ~np.isnan(purity)
    assert used.sum() == (census.n_envs > 0).sum()
    assert ((purity[used] > 0) & (purity[used] <= 1)).all()
    assert 0 < mean <= 1
