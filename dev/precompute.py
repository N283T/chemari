"""Precompute the ECFP4-across-datasets tables for examples/ecfp_openadmet_ja.py.

Writes results/precomputed/{molecules,neighbours,predictions,metrics,shap,gain,bitlen}.parquet.
Uses local copies under data/ (and data/other/) when present, else downloads from Hugging Face.
``--only-gain`` / ``--only-bitlen`` refit just the models that table needs and write it alone.

    uv run python dev/precompute.py [--only-gain | --only-bitlen]
"""

import sys
from pathlib import Path

from chemari.examples.openadmet import precompute

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    only = next((a.removeprefix("--only-") for a in sys.argv[1:] if a.startswith("--only-")), None)
    precompute(ROOT / "results" / "precomputed", data_dir=ROOT / "data", only=only)
