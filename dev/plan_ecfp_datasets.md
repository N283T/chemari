# Plan: ECFP4 across OpenADMET datasets (new notebook)

Working title: 「ECFP4 を中身から見る — OpenADMET の 3 つのデータセットで」
File: `notebooks/ecfp_openadmet_ja.py` (Japanese first, English later). The current
`ecfp_pxr*.py` notebooks stay as they are; their cells are reused where they fit.

## Message

ECFP4 is the usual first representation. Looking inside it on several real datasets shows where
it works, where it does not, and why — and the "why" is rarely one thing:

* on a lead-optimisation series whose test compounds sit close to train (ASAP Mpro potency),
  ECFP4 works as nearest neighbours and as model features
* count fingerprints help almost everywhere; for physchem endpoints (LogD, solubility)
  descriptors are strong
* in a setting like PXR (test far from train by design, neighbours differ in activity even when
  close, properties matter), a model that only takes a fingerprint struggles — but the fingerprint
  is not the only reason, and other representations are only a little better
* ECFP4 is the starting point: understanding it tells you what to try next

## Datasets and endpoints

| dataset | endpoints | split | licence |
|---|---|---|---|
| PXR challenge | pEC50 | challenge train / test (test = Enamine analogues of hits) | CC-BY-4.0 |
| ASAP-Polaris antiviral | pIC50 MERS-CoV Mpro, SARS-CoV-2 Mpro; LogD, KSOL, HLM, MLM, MDR1-MDCKII | challenge Train / Test | MIT |
| ExpansionRx | LogD, KSOL, HLM CLint, MLM CLint, Caco-2 Papp, Caco-2 efflux, MPPB, MBPB (MGMB too small) | challenge train / test (full release) | CC-BY-4.0 |

Ratio-scale endpoints log10-transformed; LogD / pIC50 / pEC50 as they are. Standardise SMILES
(largest fragment, neutralise, keep stereo), average duplicates within a split, drop test
compounds that are also in train, report near-duplicates across splits.

## Data layer (precomputed first, compute as fallback)

* `dev/precompute.py` builds everything into `results/precomputed/*.parquet` (committed, so molab
  can fetch them from GitHub raw):
  * `molecules`: task, id, smiles, split, y (transformed), CI width when available
  * `neighbours`: task, test id, nearest train id, Tanimoto (ECFP4 bit and count)
  * `predictions`: task, id, features (bit / count / desc / bit+desc), prediction (test; train out
    of fold)
  * `metrics`: task × features: Spearman, R², MAE; 1-NN ρ; pair slope
  * `shap`: task, id, bit, TreeSHAP of the bit model (long format, on-bits only)
* the notebook opens the parquet files with DuckDB (marimo SQL cells) and aggregates in SQL; if a
  file is missing (local or remote) it computes that piece in the notebook with the same code
  (shared module, e.g. `src/molwidgets/benchmark.py` or a small `ecfp_bench` package)
* the "try your own settings" parts (radius, n_bits, count, chirality) always compute live

## Outline

0. **はじめに** — ECFP4 in two lines of RDKit; the three datasets (table: what is measured, n
   train / test, how the split was made); the dataset / endpoint picker that drives Part 2
1. **ECFP4 の中身** (dataset-independent)
   * movie, ECFPStepper, MorganBitTiles (as now)
   * **new: count** — the same molecule as bit vs count; widgets gain a count mode (Stepper /
     Tiles / Explorer show how many times each environment occurs)
2. **データセットで見る** (everything reacts to the picker)
   1. 概要 — distribution of the endpoint, MolGrid of train / test coloured by it
   2. train と test はどれくらい近いか — nearest-neighbour Tanimoto histogram, with the other
      datasets overlaid (SQL over `neighbours`)
   3. 類似性原理 — |Δy| vs Tanimoto, 1-NN ρ by similarity band, compared across datasets
   4. 似ているのに違う — cliff table + MolPair (common-part view)
   5. 同じ fingerprint — identical bit vectors (stereo / counts), which count / chirality separate
   6. bit の中身 — BitAtlas on the selected dataset (collisions)
   7. モデル — bit / count / descriptors / combination (precomputed metrics, prediction vs
      measured); a small lab to change radius / n_bits / count / chirality live
   8. モデルが見ているもの — BitImportance, SHAP on a test compound and its neighbour
3. **見えたこと** — cross-dataset table (SQL over `metrics`) and the conclusions above; PXR as a
   case study (how pEC50 is measured and its uncertainty; OADMET-0006254)
4. この notebook について / AI の利用 / 参考文献

## Widgets

* reuse: MolGrid, MolPair, MorganExplorer, MorganBitTiles, ECFPStepper, BitAtlas, BitImportance
* add a count mode to Stepper / Tiles / Explorer (extend the existing widgets, no new engine)
* everything must work for any dataset (no PXR-specific assumptions)

## To check before writing

* ASAP Mpro: pair slope ≈ 0 although 1-NN ρ is 0.75 — find out why before using pair slopes
* ExpansionRx transforms (log10 of % unbound, CLint with zeros) and censored values ("raw" file)
* near-duplicates across train / test in each dataset (Chemical Validity)
* runtime on molab for the fallback path (descriptors for ~15k molecules)
