"""CheMari: chemistry widgets for marimo notebooks (molecule grids, pairs, fingerprints)."""

__version__ = "0.1.2"

from .chem import (
    BitCensus,
    bit_census,
    common_substructure,
    find_mcs,
    fingerprint_matrix,
    molecule_bit_tiles,
    morgan_bits,
    standardize_smiles,
    tanimoto_matrix,
)
from .ecfp import ecfp_trace
from .widgets import (
    BitAtlas,
    BitImportance,
    ECFPMovie,
    ECFPStepper,
    MolGrid,
    MolPair,
    MolScatter,
    MorganBitTiles,
    MorganExplorer,
    bit_gallery,
    census_for,
)

__all__ = [
    "BitAtlas",
    "BitCensus",
    "BitImportance",
    "ECFPMovie",
    "ECFPStepper",
    "MolGrid",
    "MolPair",
    "MolScatter",
    "MorganBitTiles",
    "MorganExplorer",
    "bit_census",
    "bit_gallery",
    "census_for",
    "common_substructure",
    "ecfp_trace",
    "find_mcs",
    "fingerprint_matrix",
    "molecule_bit_tiles",
    "morgan_bits",
    "standardize_smiles",
    "tanimoto_matrix",
]
