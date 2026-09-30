"""Precompute the ECFP4-across-datasets tables for notebooks/ecfp_openadmet_ja.py.

Writes results/precomputed/{molecules,neighbours,predictions,metrics,shap}.parquet. Uses local
copies under data/ (and data/other/) when present, else downloads from Hugging Face.

    uv run python dev/precompute.py
"""

from pathlib import Path

from molwidgets.bench import precompute

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    precompute(ROOT / "results" / "precomputed", data_dir=ROOT / "data")
