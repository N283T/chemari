# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "molwidgets @ git+https://github.com/N283T/openadmet-marimo",
#     "polars>=1.30",
#     "numpy>=2",
#     "altair>=5.5",
#     "lightgbm>=4.5",
#     "scikit-learn>=1.7",  # required by lightgbm's sklearn API
#     "scipy>=1.14",
#     "rdkit>=2025.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="Inside ECFP4: a hands-on guide to Morgan fingerprints")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inside ECFP4
    ### A hands-on guide to Morgan fingerprints

    QSAR モデルを作るとき、とりあえず Morgan fingerprint (radius 2, 2048 bit) を入れて、
    中身は気にせず先に進んでいませんか？ ECFP4 はケモインフォマティクスの標準的な分子表現で、
    類似検索、クラスタリング、ライブラリ設計、数えきれないほどの ML ベースラインを支えています。

    あまりに当たり前に使われているので、そのベクトルの bit 1380 が何を *意味するか*、そこに何種類の
    部分構造が同居しているかを気にする人はほとんどいません。この notebook では、その箱を開けてみます:

    1. **The algorithm**
    2. **Collisions**
    3. **Blind spots**
    4. **Similarity**
    5. **ECFP inside a model**
    6. **Cheat sheet**

    例には OpenADMET の PXR induction データセットを使います。インタラクティブな部品は、この notebook の
    ために作った [anywidget](https://anywidget.dev) ベースの自作パッケージ `molwidgets` です。

    /// admonition | 出てくる名前
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) と **Morgan fingerprint**
    (RDKit での呼び名。Morgan が 1965 年に発表した canonicalization アルゴリズムに由来) は同じものです。
    ECFP*n* の数字は *直径* を表し、**ECFP4 = Morgan radius 2**、ECFP6 = radius 3 です。**FCFP** は元素
    レベルの invariant の代わりに、薬理作用団的な原子の特徴 (donor, acceptor, aromatic, …) を使います。
    ///
    """)
    return


@app.cell
def _():
    import inspect

    import altair as alt
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from scipy.stats import spearmanr

    alt.data_transformers.disable_max_rows()

    from molwidgets import (
        ECFPMovie,
        ECFPStepper,
        MorganBitTiles,
        MorganExplorer,
        bit_gallery,
        census_for,
        ecfp_trace,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )
    from molwidgets import ecfp as ecfp_module

    return (
        Chem,
        ECFPMovie,
        ECFPStepper,
        MorganBitTiles,
        MorganExplorer,
        alt,
        bit_gallery,
        census_for,
        ecfp_module,
        ecfp_trace,
        fingerprint_matrix,
        inspect,
        lgb,
        np,
        pl,
        rdFingerprintGenerator,
        spearmanr,
        standardize_smiles,
        tanimoto_matrix,
    )


@app.cell
def _(pl, standardize_smiles):
    from pathlib import Path

    HF = "https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main/"
    FILES = {
        "train": "pxr-challenge_TRAIN.csv",
        "test_p1": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "test_p2": "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
    }

    def _read(name: str) -> pl.DataFrame:
        local = Path("data") / FILES[name]  # use a local copy when present, else download
        return pl.read_csv(local if local.exists() else HF + FILES[name])

    data = pl.concat(
        [
            _read(key)
            .select(
                pl.col("Molecule Name").alias("id"), pl.col("SMILES").alias("smiles_raw"), "pEC50"
            )
            .with_columns(pl.lit("train" if key == "train" else "test").alias("split"))
            for key in FILES
        ]
    ).with_columns(
        # largest fragment, neutralized, canonical; stereo kept
        pl.col("smiles_raw").map_elements(standardize_smiles, return_dtype=pl.Utf8).alias("smiles")
    )
    train = data.filter(pl.col("split") == "train")
    test = data.filter(pl.col("split") == "test")
    return data, test, train


@app.cell(hide_code=True)
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The algorithm

    ECFP がやっていることは、ひとことで言えば「各原子のまわりを少しずつ広く見ながら、見えた部分構造に
    番号をつけていく」ことです。手順は 4 つです。

    1. **Atom invariants**: まず原子ごとに番号をつけます。原子番号、結合している原子の数 (H を含む)、H の数、
       電荷、同位体、環に入っているか、の 6 つを 1 つの整数に hash したものです。この 6 つが同じ原子は
       同じ番号になります。
    2. **Iteration**: 隣の原子を取り込んで番号を更新します。自分の番号と、隣の原子の番号 (と結合の種類) を
       まとめてもう一度 hash します。1 回目で「自分 + 隣」(radius 1)、2 回目で「自分 + 隣 + 隣の隣」(radius 2)
       を表す番号になります。ECFP4 は 2 回目まで行います。
    3. **Deduplication**: ここまでに出てきた番号はすべて特徴になります。ただし、同じ原子の範囲を指す番号は
       1 つにまとめます (**duplicate**)。広げても範囲が変わらなかった番号も捨てます (**no growth**)。
       これで「分子に含まれる部分構造の番号の集合」ができます。
    4. **Folding**: 番号は巨大な整数なので、`番号 % 2048` で 2048 個の箱のどれかに入れ、その箱の bit を 1 に
       します。違う部分構造が同じ箱に入っても区別されません。これが **collision** です。

    下の widget で、まずは **isobutane** のまま **next ▶** を押してみてください。番号は長いので短いラベルで
    表示します (iteration 0 は *a, b*、1 回目は *A, B*、2 回目は *A', B'*)。右下の bit vector が、
    最終的にモデルが受け取るものです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    EXAMPLES = {
        "isobutane (ここから)": "CC(C)C",
        "paracetamol (小さく、対称な環)": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · 強活性の PXR agonist (pEC50 5.95)": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · その pyridine analogue (pEC50 2.06)": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",
        "cyclohexylamine": "NC1CCCCC1",
        "cycloheptylamine": "NC1CCCCCC1",
    }
    example_pick = mo.ui.dropdown(EXAMPLES, value="isobutane (ここから)", label="分子")
    custom_smiles = mo.ui.text(placeholder="…または SMILES を貼り付け", label="", full_width=False)
    mo.hstack([example_pick, custom_smiles], justify="start", gap=1)
    return custom_smiles, example_pick


@app.cell
def _(Chem, ECFPStepper, custom_smiles, example_pick, mo):
    _smi = custom_smiles.value.strip() or example_pick.value
    if Chem.MolFromSmiles(_smi) is None:
        stepper = mo.md(f"`{_smi}` は正しい SMILES ではありません。").callout(kind="warn")
    else:
        stepper = mo.ui.anywidget(ECFPStepper(_smi, max_radius=2))
    stepper
    return (stepper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Isobutane の場合**: ステップ 1–4 では 3 つのメチルがすべて **a**、中心の炭素が **b** になります。
    メチルは 3 つありますが、fingerprint に残るのは「*a* がある」という事実だけで、特徴は 2 つです。
    ステップ 5–8 で「CH についたメチル」(**A**) と「メチルを 3 つ持つ CH」(**B**、これで分子全体) が加わります。
    ステップ 9–12 では何も増えません。メチルから 2 つ広げた範囲は *B* とまったく同じ (**duplicate**)、
    中心はもうそれ以上広がらない (**no growth**) からです。最終的に特徴 4 つ、bit 4 つです。

    次は **paracetamol** を選んで **explore** に切り替えてみてください。ベンゼン環の 4 つの CH は iteration 0 では
    同じ番号です。環のどこにいるかは、隣、隣の隣と取り込んでいくことで区別されます。また 64 bit だと、すでに赤枠の bit が
    1 つあります。2 つの違う部分構造が同じ bit に入ってしまい、モデルからは区別できなくなっている箇所です。
    """)
    return


@app.cell
def _(Chem, ecfp_trace, rdFingerprintGenerator, train):
    # Does the readable version produce the same *number* of distinct features as RDKit?
    def _count_matches(radius: int) -> int:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
        return sum(
            len(ecfp_trace(s, radius).identifiers())
            == len(gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements())
            for s in train["smiles"]
        )

    validation = {r: _count_matches(r) for r in (1, 2)}
    return (validation,)


@app.cell(hide_code=True)
def _(ecfp_module, inspect, mo, train, validation):
    mo.accordion(
        {
            "舞台裏: アルゴリズム全体を ~40 行で、RDKit と照合済み": mo.vstack(
                [
                    mo.md(
                        f"""
    Stepper は素の Python で書いた ECFP (`molwidgets.ecfp`) で動いています。RDKit 内部の hash の代わりに
    `blake2b` を使っているので identifier の数値は異なりますが、invariant、隣接原子の並べ方、重複の除去は
    RDKit に従っています: training 分子のうち radius 1 で **{validation[1]:,} / {train.height:,}**、radius 2 で
    **{validation[2]:,} / {train.height:,}** について、RDKit の `GetSparseCountFingerprint` とまったく同じ数の
    distinct な特徴を出します。
    """
                    ),
                    mo.md(
                        "```python\n"
                        + inspect.getsource(ecfp_module.atom_invariant)
                        + "\n\n"
                        + inspect.getsource(ecfp_module.ecfp_trace)
                        + "```"
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Collisions

    情報が失われるのは fold の段階です。1 つの分子について言えば、これは誕生日問題です: *k* 個の distinct な
    特徴を *n* bit に入れたとき、collision が 1 つも起きない確率は

    $$P(\text{no collision}) = \prod_{i=0}^{k-1}\left(1 - \frac{i}{n}\right) \approx e^{-k(k-1)/2n}.$$

    典型的な分子の ECFP4 特徴は ~50 個なので、2048 bit ではおよそ **半数の分子が自分自身の中で衝突します**。
    式と PXR の training set はよく一致しています:
    """)
    return


@app.cell
def _(Chem, np, rdFingerprintGenerator, train):
    # Distinct ECFP4 identifiers per training molecule (unfolded).
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    unfolded = [
        np.fromiter(
            _gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements().keys(),
            dtype=np.int64,
        )
        for s in train["smiles"]
    ]
    n_features = np.array([len(u) for u in unfolded])
    return n_features, unfolded


@app.cell(hide_code=True)
def _(mo):
    fold_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    fold_pick = mo.ui.slider(steps=fold_sizes, value=2048, label="n_bits", show_value=True)
    fold_pick
    return fold_pick, fold_sizes


@app.cell(hide_code=True)
def _(alt, fold_pick, fold_sizes, mo, n_features, np, pl, unfolded):
    def _with_collision(n: int) -> float:
        return float(np.mean([len(np.unique(u % n)) < len(u) for u in unfolded]))

    def _theory(n: int) -> float:
        k = n_features[:, None]
        i = np.arange(n_features.max())[None, :]
        p_none = np.prod(np.where(i < k, 1 - i / n, 1.0), axis=1)
        return float(1 - p_none.mean())

    _rows = [
        {"n_bits": n, "分子の割合": v, "source": src}
        for n in fold_sizes
        for v, src in ((_with_collision(n), "PXR training set"), (_theory(n), "誕生日問題の式"))
    ]
    _df = pl.DataFrame(_rows)
    _chart = (
        alt.Chart(_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "n_bits:Q", scale=alt.Scale(type="log", base=2), axis=alt.Axis(values=fold_sizes)
            ),
            y=alt.Y(
                "分子の割合:Q",
                title="分子内 collision が 1 回以上ある分子",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(range=["#9ca3af", "#d6336c"]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeDash=alt.StrokeDash("source:N", legend=None),
        )
        .properties(width=380, height=240)
    )
    _rule = (
        alt.Chart(pl.DataFrame({"n_bits": [fold_pick.value]}))
        .mark_rule(color="#1f2328")
        .encode(x="n_bits:Q")
    )
    _here = _with_collision(fold_pick.value)
    mo.hstack(
        [
            _chart + _rule,
            mo.vstack(
                [
                    mo.stat(
                        f"{np.median(n_features):.0f}",
                        label="1 分子あたりの distinct な ECFP4 特徴 (中央値)",
                    ),
                    mo.stat(
                        f"{_here:.0%}",
                        label=f"{fold_pick.value} bit で分子内 collision がある分子",
                    ),
                    mo.md(
                        "単純な式が実データをよく再現しています。長さを 2 倍にすると collision 率は"
                        "おおよそ半分になり、collision がまれになるには ~16k bit が必要です。"
                    ),
                ]
            ),
        ],
        widths=[1.2, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(census_for, mo, train):
    _envs = census_for(train["smiles"].to_list(), 2, 2048).n_envs
    mo.md(
        f"""
    データセット全体では collision は避けられません: {train.height:,} 個の training 分子には distinct な ECFP4
    環境が **{int(_envs.sum()):,}** 種類あり、2048 bit に対して **1 bit あたり約 {_envs[_envs > 0].mean():.0f} 種類**、
    すべての bit が共有されています。

    下では、左に分子、右にその bit を並べています。各 bit は RDKit の `DrawMorganEnv` で描いています (青: 中心原子、
    黄: 芳香族、灰: 脂肪族環、薄い灰: 環境がつながる先)。**Bit にカーソルを乗せる** と、それが分子のどこから来た
    かが光ります。**赤い縁** は分子の中での collision、**バッジ** はデータセット全体でその bit を共有している他の
    部分構造の数です — 行をクリックすると、それらが見られます。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    TILE_EXAMPLES = {
        "ibuprofen (ここから)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · 強活性の PXR agonist": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · その pyridine analogue": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",
    }
    tile_pick = mo.ui.dropdown(TILE_EXAMPLES, value="ibuprofen (ここから)", label="分子")
    tile_smiles = mo.ui.text(placeholder="…または SMILES を貼り付け", label="")
    mo.hstack([tile_pick, tile_smiles], justify="start", gap=1)
    return tile_pick, tile_smiles


@app.cell
def _(Chem, MorganBitTiles, mo, tile_pick, tile_smiles, train):
    _smi = tile_smiles.value.strip() or tile_pick.value
    if Chem.MolFromSmiles(_smi) is None:
        bit_tiles = mo.md(f"`{_smi}` は正しい SMILES ではありません。").callout(kind="warn")
    else:
        bit_tiles = mo.ui.anywidget(
            MorganBitTiles(_smi, reference=train["smiles"].to_list(), ids=train["id"].to_list())
        )
    bit_tiles
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    注目してほしい点:

    * **Ibuprofen では bit 807 が 2 回出てくる**: カルボン酸のカルボニル炭素 (`[C;D3;H0]`) とヒドロキシ酸素
      (`[O;D1;H1]`) です。Radius 0 の identifier は分子の残りに依存しないので、2048 bit では *すべての*
      ヒドロキシ酸素が、結合相手が 3 つで H を持たない *すべての* 炭素と bit を共有しています。クリックすると
      その bit の他の部分構造が見られます。8192 bit にすると赤枠は消えます。
    * **繰り返しはまとめられる.** 3 つのメチルと 4 つの芳香族 CH 炭素は、それぞれ 1 枚のタイル (×3、×4) に
      なります。
    * **Radius 0 は結合次数を見ない.** Paracetamol のカルボニル酸素 `[O;D1;H0]` は、スルホニルの酸素と同じ
      identifier です。区別できるのは radius 1 のタイル `O=C` だけです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Blind spots

    Folding は偶然によって情報を失います。一方で、そもそも最初から集められない情報もあります。
    下の各行は *異なる* 分子のペアです。よく使われる 4 つの設定での Tanimoto 類似度を比べてみてください。
    """)
    return


@app.cell
def _(Chem, pl, rdFingerprintGenerator):
    PAIRS = {
        "cyclohexyl- と cycloheptylamine": ("NC1CCCCC1", "NC1CCCCCC1"),
        "nonanoic acid と palmitic acid": ("CCCCCCCCC(=O)O", "CCCCCCCCCCCCCCCC(=O)O"),
        "(R)- と (S)-ibuprofen": (
            "C[C@@H](C(=O)O)c1ccc(CC(C)C)cc1",
            "C[C@H](C(=O)O)c1ccc(CC(C)C)cc1",
        ),
        "dexlansoprazole と lansoprazole (立体未指定)": (
            "Cc1c(OCC(F)(F)F)ccnc1C[S@@](=O)c1nc2ccccc2[nH]1",
            "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1",
        ),
        "biphenyl と terphenyl": ("c1ccc(-c2ccccc2)cc1", "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"),
        "benzene → pyridine (§5 の PXR ペア)": (
            "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
            "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",
        ),
    }

    def _tani(a, b, gen):
        x, y = (gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(bool) for s in (a, b))
        return float((x & y).sum() / (x | y).sum())

    def _tani_counts(a, b):
        g = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        x, y = (
            g.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements() for s in (a, b)
        )
        keys = set(x) | set(y)
        num = sum(min(x.get(k, 0), y.get(k, 0)) for k in keys)
        return num / sum(max(x.get(k, 0), y.get(k, 0)) for k in keys)

    _ecfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    _ecfp_chiral = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    _fcfp = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    blind_spots = pl.DataFrame(
        [
            {
                "ペア": name,
                "ECFP4 bits": round(_tani(a, b, _ecfp), 3),
                "ECFP4 counts": round(_tani_counts(a, b), 3),
                "bits + chirality": round(_tani(a, b, _ecfp_chiral), 3),
                "FCFP4 bits": round(_tani(a, b, _fcfp), 3),
            }
            for name, (a, b) in PAIRS.items()
        ]
    )
    blind_spots
    return (blind_spots,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **「いくつあるか」は bit には見えない.** Radius より外側の環サイズや鎖長の違いは、*同じ集合* の環境を
      生むだけで、違うのは繰り返しの回数です。Bit では Tanimoto 1.0、count では明らかに 1 未満になります。
      サイズは多くの ADMET エンドポイントを強く左右する要因 (PXR のポケットは大きく脂溶性) なので、
      これは例外的なケースではありません。
    * **立体化学はデフォルトでオフ.** `includeChirality=True` を指定しないとエナンチオマーは同一になります。
      指定しても拾われない立体中心 (lansoprazole の sulfoxide) もあります。PXR の training set にはまさに
      このようなペアがあり、活性が ~1 log unit 違います。
    * **FCFP は細部と引き換えに pharmacophore を取る.** 同じ役割の原子 (たとえば芳香族炭素すべて) を
      まとめるので scaffold hopping には役立ちますが、大事な変化を隠すこともあります。芳香族 CH が 1 つ N に
      変わる benzene → pyridine のペアを見てください。

    ## 4 · Similarity

    ECFP4 の Tanimoto 類似度は、2 つの分子が共通して立てている bit の割合です。アナログ探索の主力であり、
    PXR の test set もこれで作られました。下の図では、各 test 化合物を、最も近い **強活性** の training 化合物
    (pEC50 ≥ 6) との類似度と、自身の実測活性でプロットしています。
    """)
    return


@app.cell(hide_code=True)
def _(alt, fingerprint_matrix, mo, np, pl, tanimoto_matrix, test, train):
    _potent = train.filter(pl.col("pEC50") >= 6)
    _S = tanimoto_matrix(
        fingerprint_matrix(test["smiles"].to_list(), 2, 2048),
        fingerprint_matrix(_potent["smiles"].to_list(), 2, 2048),
    )
    _best = _S.argmax(1)
    _df = pl.DataFrame(
        {
            "id": test["id"],
            "最も近い強活性ヒットとの Tanimoto": _S.max(1),
            "test pEC50": test["pEC50"],
            "hit pEC50": _potent["pEC50"].to_numpy()[_best],
        }
    )
    _chart = (
        alt.Chart(_df)
        .mark_circle(size=28, opacity=0.6, color="#3b82f6")
        .encode(
            x=alt.X("最も近い強活性ヒットとの Tanimoto:Q", scale=alt.Scale(domain=[0.2, 1])),
            y=alt.Y("test pEC50:Q", scale=alt.Scale(domain=[1.5, 7.5])),
            tooltip=[
                "id",
                alt.Tooltip("最も近い強活性ヒットとの Tanimoto:Q", format=".2f"),
                "test pEC50",
                "hit pEC50",
            ],
        )
        .properties(width=380, height=260)
    )
    _hit_band = (
        alt.Chart(pl.DataFrame({"y": [6.0]}))
        .mark_rule(strokeDash=[4, 4], color="#d6336c")
        .encode(y="y:Q")
    )
    _hi = _df.filter(pl.col("最も近い強活性ヒットとの Tanimoto") >= 0.5)
    mo.hstack(
        [
            _chart + _hit_band,
            mo.md(
                f"""
    Test 化合物は、強活性かつ counter-screen で選択的だった **63** ヒットの ECFP4 近傍 (> 0.4) として
    選ばれました。再測定の結果、launch post ではそのようなヒットは 46 件とされています。ここでは単純に
    pEC50 ≥ 6 の training 化合物 {_potent.height} 件を使っており、RDKit の ECFP4 はベンダーの実装と bit 単位で
    一致するとは限らないので、この図で 0.4 を超える test 化合物は
    **{(_df["最も近い強活性ヒットとの Tanimoto"] > 0.4).mean():.0%}** です。いずれにせよ、test 化合物自身の
    活性はアッセイの全域に広がっています (点線: pEC50 6)。現在のヒットと Tanimoto ≥ 0.5 の
    {_hi.height} 化合物のうち、自分もヒットなのは **{(_hi["test pEC50"] >= 6).mean():.0%}** だけで、
    **{(_hi["test pEC50"] < 4).mean():.0%}** は実質的に不活性 (pEC50 < 4) です。

    これは ECFP の失敗ではありません。類似検索は、ヒットの近傍を返すのが *本来の役目* です。
    メディシナルケミストは、どれが活性を保つかを知るためにこそ analogue を買います。ECFP4 の Tanimoto に
    わからないのは、その小さな変化の **どれが** 効くのかであり、同じ bit で作ったモデルも同じ盲点を
    受け継ぎます。OpenADMET の
    [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
    でも、上位チーム全員にとって最も難しかった test 化合物は、まさにこうした activity cliff でした。
    """
            ),
        ],
        widths=[1.1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · ECFP inside a model

    たいていの人が最初に作るモデル — OpenADMET のチャレンジ tutorial と同じ **2048-bit ECFP4 の LightGBM** —
    を学習させて、何を学んだのかを聞いてみます。定番の道具は 2 つです:

    * **Global feature importance** (bit ごとの gain の合計): 木がどの bit で最もよく分岐しているか。
    * **Local attribution** (LightGBM の `pred_contrib=True` による TreeSHAP): *この* 分子の予測を各 bit が
      どれだけ押し上げたか・押し下げたか。各 bit の寄与をそれを立てた原子に配分すると、Riniker & Landrum の
      similarity map のような原子ごとのマップになります。

    Fold サイズを選んで学習させてください。切り替えたとき「上位の bit」がどうなるかに注目です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_bits = mo.ui.radio(
        {"2048 bits (よくあるデフォルト)": 2048, "8192 bits": 8192},
        value="2048 bits (よくあるデフォルト)",
        label="モデルの fold サイズ",
        inline=True,
    )
    model_bits
    return (model_bits,)


@app.cell
def _(fingerprint_matrix, lgb, model_bits, mo, np, spearmanr, test, train):
    N_BITS = model_bits.value
    X_train = fingerprint_matrix(train["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    X_test = fingerprint_matrix(test["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    with mo.status.spinner(f"{N_BITS}-bit ECFP4 で LightGBM を学習中…"):
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            random_state=0,
            verbose=-1,
        ).fit(X_train, train["pEC50"].to_numpy())
    pred_test = model.predict(X_test)
    model_scores = {
        "MAE": float(np.abs(pred_test - test["pEC50"].to_numpy()).mean()),
        "rho": float(spearmanr(pred_test, test["pEC50"].to_numpy())[0]),
    }
    return N_BITS, X_test, model, model_scores, pred_test


@app.cell(hide_code=True)
def _(N_BITS, alt, census_for, mo, model, model_scores, np, pl, train):
    _census = census_for(train["smiles"].to_list(), 2, N_BITS)
    _gain = model.booster_.feature_importance("gain")
    _top = np.argsort(-_gain)[:15]
    top_bits = pl.DataFrame(
        {
            "rank": np.arange(1, 16),
            "bit": _top.astype(int),
            "gain": (_gain[_top] / _gain.sum()).round(4),
            "# environments in bit": _census.n_envs[_top].astype(int),
            "# training molecules with bit": _census.on[:, _top].sum(0).astype(int),
        }
    )
    _chart = (
        alt.Chart(top_bits)
        .mark_bar()
        .encode(
            y=alt.Y("bit:N", sort=None, title=f"上位の bit ({N_BITS})"),
            x=alt.X("# environments in bit:Q", title="その bit を共有する異なる部分構造の数"),
            color=alt.Color(
                "gain:Q", scale=alt.Scale(scheme="reds"), legend=alt.Legend(title="gain の割合")
            ),
            tooltip=[
                "rank",
                "bit",
                alt.Tooltip("gain:Q", format=".1%"),
                "# environments in bit",
                "# training molecules with bit",
            ],
        )
        .properties(width=340, height=300)
    )
    mo.hstack(
        [
            _chart,
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.stat(f"{model_scores['MAE']:.2f}", label="test MAE"),
                            mo.stat(f"{model_scores['rho']:.2f}", label="test Spearman ρ"),
                            mo.stat(
                                f"{np.median(top_bits['# environments in bit']):.0f}",
                                label="上位 15 bit の部分構造数 (中央値)",
                            ),
                        ]
                    ),
                    mo.md(
                        f"""
    各バーは重要度上位 15 の bit の 1 つで、長さはその bit に fold された training set の **異なる**
    部分構造の数です。{N_BITS} bit では、モデルのお気に入りの特徴はそれぞれ約
    **{np.median(top_bits["# environments in bit"]):.0f}** 種類の無関係な環境の混合物です。
    「bit {top_bits["bit"][0]} が最も重要な特徴」は説明になっていません。部分構造の詰め合わせを指さしているだけです。
    上で 8192 bit に切り替えると、詰め合わせはそれぞれ数種類まで減ります — 精度はほとんど変わらないのに。
    下で bit を選ぶと、その中身が見られます。
    """
                    ),
                ]
            ),
        ],
        widths=[1, 1.1],
        align="center",
    )
    return (top_bits,)


@app.cell(hide_code=True)
def _(mo, top_bits):
    top_bit_pick = mo.ui.dropdown(
        {f"#{r} · bit {b}": int(b) for r, b in zip(top_bits["rank"], top_bits["bit"])},
        value=f"#1 · bit {top_bits['bit'][0]}",
        label="上位の bit の中身を見る",
    )
    top_bit_pick
    return (top_bit_pick,)


@app.cell(hide_code=True)
def _(N_BITS, bit_gallery, mo, top_bit_pick, train):
    mo.Html(
        bit_gallery(
            top_bit_pick.value,
            train["smiles"].to_list(),
            ids=train["id"].to_list(),
            radius=2,
            n_bits=N_BITS,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Reading one prediction

    ここからは分子単位で見ていきます。下のペアは、チャレンジ全体で 2 番目に難しかった test 化合物
    **OADMET-0006254** (pEC50 2.06) と、その training の nearest neighbour **OADMET-0002810** (pEC50 5.95) です。
    違いは benzene の CH → pyridine の N だけ。OpenADMET の分析によれば、関連する共結晶構造ではその窒素が
    SER247 と水素結合しており、ポケット内でのリガンドの収まり方が変わると考えられます。Tier-1 の全チームが
    この化合物を過大に予測しました。

    原子の色はモデルの TreeSHAP attribution です (赤は予測 pEC50 を上げ、青は下げる)。Bit をクリックすると
    代わりにその環境が表示され、表の **SHAP** 列には A と B それぞれの bit ごとの寄与が出ます。リストから
    予測の外れが大きい他の test 化合物も選べます。
    """)
    return


@app.cell
def _(fingerprint_matrix, np, pl, pred_test, tanimoto_matrix, test, train):
    _S = tanimoto_matrix(
        fingerprint_matrix(test["smiles"].to_list(), 2, 2048),
        fingerprint_matrix(train["smiles"].to_list(), 2, 2048),
    )
    _nn = _S.argmax(1)
    cliff_pairs = (
        pl.DataFrame(
            {
                "test id": test["id"],
                "test pEC50": test["pEC50"],
                "predicted": pred_test.round(2),
                "NN id": train["id"].to_numpy()[_nn],
                "NN pEC50": train["pEC50"].to_numpy()[_nn],
                "Tanimoto": _S.max(1).astype(float).round(3),
            }
        )
        .with_columns((pl.col("predicted") - pl.col("test pEC50")).abs().round(2).alias("|error|"))
        .sort("|error|", descending=True)
    )
    return (cliff_pairs,)


@app.cell(hide_code=True)
def _(cliff_pairs, mo, pl):
    _worst = cliff_pairs.head(8)
    if "OADMET-0006254" not in _worst["test id"]:
        _worst = pl.concat(
            [cliff_pairs.filter(pl.col("test id") == "OADMET-0006254"), _worst.head(7)]
        )
    pair_pick = mo.ui.dropdown(
        {
            f"{r['test id']} (実測 {r['test pEC50']:.2f}, 予測 {r['predicted']:.2f}) vs {r['NN id']}": r[
                "test id"
            ]
            for r in _worst.iter_rows(named=True)
        },
        value=next(
            k
            for k in [
                f"{r['test id']} (実測 {r['test pEC50']:.2f}, 予測 {r['predicted']:.2f}) vs {r['NN id']}"
                for r in _worst.iter_rows(named=True)
            ]
            if k.startswith("OADMET-0006254")
        ),
        label="ペア (test 化合物とその training の nearest neighbour)",
    )
    pair_pick
    return (pair_pick,)


@app.cell
def _(
    MorganExplorer,
    N_BITS,
    cliff_pairs,
    data,
    fingerprint_matrix,
    mo,
    model,
    np,
    pair_pick,
    pl,
    train,
):
    _row = cliff_pairs.filter(pl.col("test id") == pair_pick.value).row(0, named=True)
    _smi = dict(zip(data["id"], data["smiles"]))
    _ids = [_row["test id"], _row["NN id"]]
    _X = fingerprint_matrix([_smi[i] for i in _ids], 2, N_BITS).astype(np.float32)
    pair_X = _X
    pair_contrib = model.predict(_X, pred_contrib=True)  # last column = expected value
    _contrib = pair_contrib
    _pred = _contrib.sum(1)
    _maps = [{str(b): float(c[b]) for b in np.flatnonzero(x)} for c, x in zip(_contrib, _X)]
    _y = dict(zip(data["id"], data["pEC50"]))
    pair_y = [_y[i] for i in _ids]
    shap_explorer = mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _ids[0],
                    "smiles": _smi[_ids[0]],
                    "label": f"test · 実測 {_y[_ids[0]]:.2f} · 予測 {_pred[0]:.2f}",
                },
                {
                    "id": _ids[1],
                    "smiles": _smi[_ids[1]],
                    "label": f"train · 実測 {_y[_ids[1]]:.2f} · 予測 {_pred[1]:.2f}",
                },
            ],
            reference=train["smiles"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
            n_bits=N_BITS,
            contributions=_maps,
            contrib_label="SHAP",
            contrib_radius=2,
            contrib_n_bits=N_BITS,
            pair_note=f"baseline (予測の平均) {_contrib[0, -1]:.2f}",
        )
    )
    shap_explorer
    return pair_X, pair_contrib, pair_y


@app.cell(hide_code=True)
def _(mo, pair_X, pair_contrib, pair_y):
    _a, _b = pair_X.astype(bool)
    _diff = pair_contrib[0, :-1] - pair_contrib[1, :-1]  # per-feature A − B
    _differ, _shared, _neither = _a ^ _b, _a & _b, ~(_a | _b)
    _gap = _diff.sum()
    mo.md(
        f"""
    **予測の差はどこから来るのか?** モデルの予測は A − B = **{_gap:+.2f}** です (実測: {pair_y[0] - pair_y[1]:+.2f})。
    TreeSHAP はこの差を特徴ごとに厳密に分解します: A と B で異なる **{int(_differ.sum())}** bit が
    **{_diff[_differ].sum():+.2f}**、共有している **{int(_shared.sum())}** bit が **{_diff[_shared].sum():+.2f}**
    (木は bit の組み合わせで分岐するので、共有 bit でも文脈によって効き方が変わります)、どちらにも立っていない
    bit が **{_diff[_neither].sum():+.2f}**。原子のマップに描けるのは *立っている* bit だけです。立っていないことも
    木にとっては意味がありますが、原子の上には描けません。

    Attribution マップからわかること、わからないこと:

    * **変化は数個の bit としてしか見えない.** 原子 1 つの置換で変わるのはわずかな環境だけです
      (*only A* / *only B* の行)。モデルはそれを見ていますが、training データではその bit がほかの多くの部分構造も
      表しているので (**# envs**)、学習された効果は fold されたもの全部の平均になります。Benzene → pyridine の
      ペアでは、新しい窒素はほとんど光りません。
    * **似た fingerprint は似た予測になる.** Attribution の大半は共有 bit に乗っています。モデルは ECFP に
      求められたことをそのままやっているだけで、activity cliff はそもそも表現の中にないのです。
    * **Collision している bit の attribution はあいまい.** その bit が十数種類の部分構造を抱えているなら、
      赤い原子の意味は「モデルはこの bit が好き」であって「この官能基が好き」ではありません。8192 bit で学習し直して
      比べてみてください。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Cheat sheet

    **長所**

    * 速く、決定的で、学習データがいらない。
    * 局所的な部分構造をよく表す。類似検索、クラスタリング、アナログ探索にとても強い。
    * 同じシリーズ内の SAR では強力なベースライン。

    **短所**

    * Folding による collision: 2048 bit では drug-like 分子のおよそ半数が自分自身の中で衝突し、データセット全体では
      すべての bit が共有される。
    * Bit vector は有無しか記録しない: 環サイズ、鎖長、繰り返し出てくる基が消える。
    * 立体化学はデフォルトで無視される。
    * 分子全体の形、サイズ、物性がない。どの環境も同じ重みで数えられる。
    * Hash された bit は解釈しにくい。

    **変えたほうがいいデフォルト**

    * 回帰には **count** fingerprint を使う。
    * 個々の bit を *解釈* したいなら 4096 bit 以上、または unfold/sparse な特徴を使う。
    * 立体が効く場面では `includeChirality=True` を指定する。
    * Bit と並べて、分子全体の記述子 (logP、サイズ、TPSA) をいくつか加える。

    **モデルを解釈するとき**

    * ある bit について語る前に、その bit が含む *すべての* 環境に戻って確認する。
    * 原子ごとの attribution は単独の分子ではなく、matched pair で比べて読む。

    ---

    ### About this notebook

    * **データ:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0)。Test set の設計と最難関化合物の分析は、OpenADMET の
      [challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)、
      [launch post](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/)、
      [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
      によります。
    * **参考文献:** Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50, 742 (ECFP); Morgan, *J. Chem. Doc.* 1965, 5, 107;
      Riniker & Landrum, *J. Cheminform.* 2013, 5, 43 (similarity maps)。
    * **ウィジェット:** `ECFPStepper`、`MorganBitTiles`、`MorganExplorer` は、この notebook のために作った anywidget
      コンポーネントです ([ソース](https://github.com/N283T/openadmet-marimo))。`molwidgets.ecfp` が上で見せた
      読める ECFP 実装です。
    * **姉妹 notebook:** *似ているのに、同じじゃない* では、このデータセットで fingerprint モデルが苦戦する理由を
      掘り下げています。
    * **AI の利用:** ウィジェットと notebook の骨組みは、コーディングアシスタントとして Claude (Anthropic) と
      一緒に作りました。問いの設定、分析の選択、解釈は私自身の PXR チャレンジでの経験に基づいています。
      表示している数値はすべて、この notebook の中でその場で計算しています。
    """)
    return


if __name__ == "__main__":
    app.run()
