# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "chemari @ git+https://github.com/N283T/chemari@v0.1.0",
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
app = marimo.App(width="medium", app_title="その ECFP4、理解して使っていますか？")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # その ECFP4、理解して使っていますか？
    ### 中身と弱点を PXR データで確かめる

    分子を機械学習にかけるとき、とりあえずこう書いていないでしょうか。

    ```python
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    ```

    これが **ECFP4**
    です。ケモインフォマティクスで最もよく使われる分子表現で、計算が速く、調整するパラメータもほとんどなく、類似検索でも
    QSAR でも多くの場合よく効きます。新しい手法を評価するときも、まず比べる相手はたいてい ECFP4
    です。

    ただ、この 2048 bit
    が分子の何を記録していて、何を記録していないのかを説明できる人は多くありません。弱点を知らないまま使っていると、効かないデータに当たったときに理由がわかりません。

    /// admonition | 名前について
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) と、RDKit の **Morgan
    fingerprint** は同じものです。ECFP*n* の数字は原子のまわりを見る範囲の*直径*で、RDKit
    では代わりに半径 (radius) で指定します。ECFP4 は `radius=2` にあたります。
    ///

    この notebook は 2 部構成です。

    **第 1 部 · 中身**: ECFP4 が分子から bit を作る手順を追います。

    * ECFP4 の中身
    * 1 · ECFP4 の中身を見てみる

    **第 2 部 · 弱点**: あえて ECFP4 が効かないデータセット (OpenADMET の PXR 誘導データ)
    を使って、どこで、なぜ崩れるのかを見ます。

    * 2 · ECFP4 が効かないデータ: PXR
    * 3 · ECFP4 が崩れるところ: 類似性原理、activity cliff、同一の fingerprint、分子全体の性質
    * 4 · モデルの中身: ECFP4 で学習した LightGBM が何を覚えたのか
    * 5 · Model lab: count、bit 長、キラリティ、記述子を試す
    * 6 · まとめ

    ---

    ## 第 1 部 · ECFP4 の中身

    まずは 80 秒の動画で、ECFP4 の中身と性質を見てください。
    """)
    return


@app.cell(hide_code=True)
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    動画の内容をまとめると、ECFP4 には次の性質があります。

    * fingerprint が記録するのはどの部分構造があるかだけ
    * 部分構造が何回出てくるか 分子のどこにあるか 分子全体としてどうかは残らない
    * folding によって無関係な部分構造が同じ bit に押し込まれる
    * Tanimoto 類似度は衝突した bit も含めて共通の bit を数える

    /// admonition | 部分構造と環境
    ECFP4 の部分構造は、ある原子を中心に radius 以内の原子と結合をまとめた円形のものです。原子環境
    (atom environment) とも呼ばれ、ウィジェットに出てくる environment や `# envs` はこれを指します。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    下のウィジェットでは、同じ手順を好きな化合物で追えます。SMILES を入力するか例を選び、「次へ」で 1 ステップずつ進めるか、自動再生してください。
    """)
    return


@app.cell(hide_code=True)
def _(ECFPStepper, mo):
    ecfp_stepper = mo.ui.anywidget(ECFPStepper())
    ecfp_stepper
    return (ecfp_stepper,)


@app.cell
def _():
    import altair as alt
    import lightgbm as lgb
    import numpy as np
    import polars as pl
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
    from rdkit.Chem import rdMolDescriptors as rdmd
    from scipy.stats import spearmanr

    alt.data_transformers.disable_max_rows()  # a few charts plot all ~4.6k compounds

    from chemari import (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MorganBitTiles,
        MorganExplorer,
        census_for,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )

    return (
        BitAtlas,
        BitImportance,
        Chem,
        Crippen,
        Descriptors,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MorganBitTiles,
        MorganExplorer,
        alt,
        census_for,
        fingerprint_matrix,
        lgb,
        np,
        pl,
        rdFingerprintGenerator,
        rdmd,
        spearmanr,
        standardize_smiles,
        tanimoto_matrix,
    )


@app.cell
def _(pl):
    from pathlib import Path

    HF = "https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main/"
    FILES = {
        "train": "pxr-challenge_TRAIN.csv",
        "test_p1": "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "test_p2": "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv",
    }
    CI_LO = "pEC50_ci.lower (-log10(molarity))"
    CI_HI = "pEC50_ci.upper (-log10(molarity))"

    def _read(name: str) -> pl.DataFrame:
        local = Path("data") / FILES[name]  # use a local copy when present, else download
        return pl.read_csv(local if local.exists() else HF + FILES[name])

    raw = pl.concat(
        [
            _read(key)
            .select(
                pl.col("Molecule Name").alias("id"),
                pl.col("SMILES").alias("smiles_raw"),
                "pEC50",
                (pl.col(CI_HI) - pl.col(CI_LO)).alias("ci_width"),
            )
            .with_columns(pl.lit("train" if key == "train" else "test").alias("split"))
            for key in FILES
        ]
    )
    return (raw,)


@app.cell
def _(Chem, pl, raw, standardize_smiles):
    # largest fragment, neutralized, canonical; stereo kept
    data = raw.with_columns(
        pl.col("smiles_raw").map_elements(standardize_smiles, return_dtype=pl.Utf8).alias("smiles")
    ).with_row_index("row")

    # "changed" = the standardized parent differs from the input as a structure, not just in notation
    _canon_raw = raw["smiles_raw"].map_elements(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)), return_dtype=pl.Utf8
    )
    hygiene = {
        "invalid": data["smiles"].null_count(),
        "changed": int((_canon_raw != data["smiles"]).sum()),
        "dup_within": data.height - data["smiles"].n_unique(),
        "overlap": int(
            data.filter(pl.col("split") == "test")["smiles"]
            .is_in(data.filter(pl.col("split") == "train")["smiles"].implode())
            .sum()
        ),
    }
    train = data.filter(pl.col("split") == "train")
    test = data.filter(pl.col("split") == "test")
    return data, hygiene, test, train


@app.cell(hide_code=True)
def _(Chem, census_for, mo, np, rdFingerprintGenerator, train):
    _envs = census_for(train["smiles"].to_list(), 2, 2048).n_envs
    # distinct radius-0..2 environments per molecule, before folding
    _gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    _per_mol = np.median(
        [
            len(_gen.GetSparseCountFingerprint(Chem.MolFromSmiles(s)).GetNonzeroElements())
            for s in train["smiles"]
        ]
    )
    mo.md(
        f"""
    ## 1 · ECFP4 の中身を見てみる

    ここでは第 2 部で使う PXR データセットの分子で ECFP4 を見ます。

    このデータセットの化合物には、1 分子あたり **{_per_mol:.0f} 種類** (中央値)
    の部分構造があります。train {train.height:,} 化合物全体では **{int(_envs.sum()):,}
    種類**になり、これを 2048 bit に folding するので、1 bit には平均
    **{_envs[_envs > 0].mean():.0f} 種類**の部分構造が入ります。部分構造が 1 種類だけの bit
    はありません。

    グリッドから化合物を選ぶと、その下に分子 (左) と bit の一覧 (右) が表示されます。

    * **bit にマウスを乗せる**: 分子のどこから来た bit かが光る
    * **赤**: 分子内の別の部分構造と同じ bit に落ちている (衝突)
    * **collisions only**: 衝突している bit だけを表示
    * **グレーのバッジ**: データセット内でこの bit を共有するほかの部分構造の数
    * **行をクリック**: その bit を共有する部分構造の一覧
    * **radius / fold to**: bit を作り直す
    """
    )
    return


@app.cell
def _(MolGrid, data, mo, pl):
    # the benzene → pyridine pair from section 4 first, then everything by potency
    _pair = ["OADMET-0006254", "OADMET-0002810"]
    _rows = data.select("id", "smiles", "split", "pEC50")
    grid = mo.ui.anywidget(
        MolGrid(
            pl.concat(
                [
                    _rows.filter(pl.col("id").is_in(_pair)).sort("pEC50"),
                    _rows.filter(~pl.col("id").is_in(_pair)).sort("pEC50", descending=True),
                ]
            ),
            subset=["split", "pEC50"],
            color_by="pEC50",
            selection_mode="single",
            selection=["OADMET-0006254"],
            page_size=12,
            cell_size=150,
        )
    )
    grid
    return (grid,)


@app.cell
def _(MorganBitTiles, data, grid, mo, pl, train):
    _sel = grid.value.get("selection") or ["OADMET-0006254"]
    _row = data.filter(pl.col("id") == _sel[0]).row(0, named=True)
    mo.ui.anywidget(
        MorganBitTiles(
            _row["smiles"],
            reference=train["smiles"].to_list(),
            ids=train["id"].to_list(),
            label=f"{_row['id']} · {_row['split']} · pEC50 {_row['pEC50']:.2f}",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    見どころ:

    * **OADMET-0006254 は分子内で衝突している**
      * 2048 bit では 4 級炭素 `C(C)(S)(C)C` と芳香族の `c(N)(c)c` が同じ bit 381 に落ちる
      * collisions only で確認できる 8192 bit にすると衝突は消える
    * **radius 0 の bit は多くの分子で立つ**
      * 原子 1 個だけの部分構造なので分子の種類を問わず現れる
      * CH₂ の bit 80 (`[C;D2;H2]`) は train 化合物の 6 割以上で立つ
      * カルボン酸ではカルボニル炭素 `[C;D3;H0]` とヒドロキシ酸素 `[O;D1;H1]` が同じ bit 807
        に落ちる
    * **同じ部分構造は何度出ても 1 行にまとまる**
      * メチル 3 つや芳香族 CH 4 つも 1 行で ×3 や ×4 と表示される
      * bit ベクトルには回数が残らない

    ECFP4 の bit は部分構造があるかどうかだけを記録し、1 つの bit を別々の部分構造が共有していることも珍しくありません。
    """)
    return


@app.cell(hide_code=True)
def _(BitAtlas, mo, train):
    mo.vstack(
        [
            mo.md("train 全体では、各 bit に次のような部分構造が入っています。"),
            mo.ui.anywidget(BitAtlas(train["smiles"].to_list(), ids=train["id"].to_list())),
        ]
    )
    return


@app.cell(hide_code=True)
def _(hygiene, mo, test, train):
    mo.vstack(
        [
            mo.md(r"""
    ---

    ## 第 2 部 · ECFP4 の弱点

    ## 2 · ECFP4 が効かないデータ: PXR

    **Pregnane X receptor (PXR)** は、体に入ってきた異物を感知して CYP3A4 や P-gp
    などの発現を引き上げる核内受容体です。PXR
    を活性化する薬はほかの薬の代謝まで速めてしまうので、PXR 誘導は薬物相互作用 (DDI)
    の代表的なリスクです。OpenADMET はこれを 11,000 以上の化合物で測定し、新しい 513 化合物の
    **pEC50**
    を予測する[ブラインドチャレンジ](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)を
    [Hugging Face 上で](https://huggingface.co/spaces/openadmet/pxr-challenge)開催しました。

    このチャレンジでは、ECFP4 を使ったモデルが一貫して下位に沈みました。私自身の参加 (95 チーム中 4
    位) でも、ECFP4 だけの LightGBM は CV MAE が 0.57
    前後で、記述子や埋め込みを組み合わせたアンサンブルは 0.40
    を切っています。ほかのチームも同じ傾向を報告しています。

    データは [Hugging Face で公開されている OpenADMET の PXR
    データ](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY-4.0)
    です。**train** と、チャレンジ終了後にラベルが公開された **test** 全体を使います。SMILES
    はすべて同じ手順で標準化しています: 最大フラグメントを残して中和し、立体化学は保ったまま
    canonical SMILES にします。
    """),
            mo.hstack(
                [
                    mo.stat(f"{train.height:,}", label="train"),
                    mo.stat(f"{test.height:,}", label="test"),
                    mo.stat(str(hygiene["changed"]), label="中和した化合物"),
                    mo.stat(str(hygiene["dup_within"]), label="標準化後の重複"),
                    mo.stat(str(hygiene["overlap"]), label="train と test の重複"),
                ],
                widths="equal",
                gap=0.5,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    513 個の test 化合物は、ランダムに選ばれたものではありません。OpenADMET
    はスクリーニングで活性が強く (EC50 ≤ 1 µM)、PXR を欠く細胞では活性を示さなかった **63**
    化合物をヒットとし、それらとの **ECFP4 Tanimoto 類似度が 0.4 を超える**類縁体を Enamine
    から購入しました。スクリーニングのヒットから類縁体を集める、創薬でよくあるヒット展開のやり方です。

    この作り方から、次の 2 つが言えます。

    * どの test 化合物にも train の中に似た化合物がある
    * その似た化合物はたいてい活性が強い

    各化合物について、ECFP4 Tanimoto (2048 bit) で train 中の**最近傍 (NN)** を探します (train
    化合物は自分自身を除く)。
    """)
    return


@app.cell
def _(fingerprint_matrix, np, test, train, tanimoto_matrix):
    X_train = fingerprint_matrix(train["smiles"].to_list(), radius=2, n_bits=2048)
    X_test = fingerprint_matrix(test["smiles"].to_list(), radius=2, n_bits=2048)

    S_train = tanimoto_matrix(X_train)
    np.fill_diagonal(S_train, 0.0)  # leave-one-out: a compound is not its own neighbour
    S_test = tanimoto_matrix(X_test, X_train)

    y_train = train["pEC50"].to_numpy()
    y_test = test["pEC50"].to_numpy()
    nn_test = S_test.argmax(1)
    return S_test, S_train, X_train, nn_test, y_test, y_train


@app.cell(hide_code=True)
def _(S_test, S_train, alt, mo, nn_test, np, pl, test, y_test, y_train):
    _dens = pl.concat(
        [
            pl.DataFrame({"NN Tanimoto": S_train.max(1), "set": "train → rest of train"}),
            pl.DataFrame({"NN Tanimoto": S_test.max(1), "set": "test → train"}),
        ]
    )
    _density = (
        alt.Chart(_dens)
        .transform_density(
            "NN Tanimoto", groupby=["set"], as_=["NN Tanimoto", "density"], extent=[0, 1]
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("NN Tanimoto:Q", title="Tanimoto to nearest training neighbour"),
            y=alt.Y("density:Q", stack=None),
            color=alt.Color(
                "set:N",
                scale=alt.Scale(
                    range=["#3b82f6", "#f59e0b"],
                    domain=["train → rest of train", "test → train"],
                ),
                legend=alt.Legend(orient="top-right", title=None, fillColor="white", padding=4),
            ),
        )
        .properties(height=220, width=300)
    )
    _pts = pl.DataFrame(
        {
            "id": test["id"],
            "NN Tanimoto": S_test.max(1),
            "test pEC50": y_test,
            "NN pEC50": y_train[nn_test],
        }
    )
    _scatter = (
        alt.Chart(_pts)
        .mark_circle(size=24, opacity=0.6)
        .encode(
            x=alt.X("NN Tanimoto:Q", scale=alt.Scale(domain=[0.2, 1])),
            y=alt.Y("test pEC50:Q", scale=alt.Scale(domain=[1.5, 7.5])),
            color=alt.Color(
                "NN pEC50:Q",
                scale=alt.Scale(range=["#2563eb", "#dc2626"], interpolate="hcl"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=["id", alt.Tooltip("NN Tanimoto:Q", format=".2f"), "test pEC50", "NN pEC50"],
        )
        .properties(height=220, width=300)
    )
    _nn = y_train[nn_test]
    mo.vstack(
        [
            mo.hstack([_density, _scatter], justify="start", gap=2),
            mo.md(
                f"""
    **左: 最近傍との類似度**

    * test から train への NN 類似度 (中央値 **{np.median(S_test.max(1)):.2f}**) は train どうし
      (**{np.median(S_train.max(1)):.2f}**) より高い
    * test は train の適用範囲の内側にある

    **右: test の pEC50 と最近傍の pEC50**

    * 最近傍はほとんどが強い (平均 pEC50 **{_nn.mean():.2f}** train 全体は {y_train.mean():.2f}
      **{(_nn >= 5.5).mean():.0%}** が 5.5 以上)
    * それでも test 自身の pEC50 は {y_test.min():.1f}〜{y_test.max():.1f} に広がる
    * ヒットのまわりの SAR 探索になっていて 似ていても活性が残るかどうかは ECFP4 Tanimoto
      からはわからない
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(1, 50, value=1, step=1, label="近傍の数 k", show_value=True)
    mo.md(
        f"""
    最も単純な fingerprint モデルとして、各 test 化合物の pEC50 を**最も似ている k 個**の train
    化合物の平均で予測してみます。{k_slider}
    """
    )
    return (k_slider,)


@app.cell(hide_code=True)
def _(S_test, alt, k_slider, mo, np, pl, spearmanr, test, y_test, y_train):
    _k = k_slider.value
    _idx = np.argsort(-S_test, axis=1)[:, :_k]
    _pred = y_train[_idx].mean(1)
    _rand = np.abs(y_test[:, None] - y_train[None, :]).mean()
    _df = pl.DataFrame(
        {
            "true pEC50": y_test,
            "kNN prediction": _pred,
            "NN similarity": S_test.max(1),
            "id": test["id"],
        }
    )
    _lim = [1.5, 7.5]
    _pts = (
        alt.Chart(_df)
        .mark_circle(size=28, opacity=0.6)
        .encode(
            x=alt.X("kNN prediction:Q", scale=alt.Scale(domain=_lim)),
            y=alt.Y("true pEC50:Q", scale=alt.Scale(domain=_lim)),
            color=alt.Color(
                "NN similarity:Q",
                scale=alt.Scale(scheme="viridis"),
                legend=alt.Legend(orient="right"),
            ),
            tooltip=[
                "id",
                alt.Tooltip("true pEC50:Q", format=".2f"),
                alt.Tooltip("kNN prediction:Q", format=".2f"),
                alt.Tooltip("NN similarity:Q", format=".2f"),
            ],
        )
    )
    _diag = (
        alt.Chart(pl.DataFrame({"x": _lim, "y": _lim}))
        .mark_line(color="#9ca3af", strokeDash=[4, 4])
        .encode(x="x", y="y")
    )
    _rho = spearmanr(_pred, y_test)[0]
    _mae = np.abs(_pred - y_test).mean()
    mo.vstack(
        [
            mo.hstack(
                [
                    (_diag + _pts).properties(width=330, height=300),
                    mo.vstack(
                        [
                            mo.stat(f"{_mae:.2f}", label=f"MAE ({_k}-NN)"),
                            mo.stat(f"{_rho:.2f}", label="Spearman ρ"),
                            mo.stat(f"{_rand:.2f}", label="ランダムな train 化合物で予測した MAE"),
                        ]
                    ),
                ],
                widths=[1.1, 1],
                align="center",
            ),
            mo.md("""
    * k = 1 では ρ ≈ 0 で 順位づけの力はほぼない
    * MAE もランダムな train 化合物で予測した場合とほぼ同じ
    * k を増やすと MAE は下がるが 予測が平均値に寄るだけ (点が縦の帯に集まる)

    test は train の ECFP4 近傍から作られているのに、最近傍の活性は test の予測にほとんど役立ちません。
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · ECFP4 が崩れるところ

    ### 3a · 類似性原理を測ってみる

    類似性原理 (similar property principle) によれば、構造が似ている 2
    分子ほど活性の差は小さくなるはずです。train の全ペア約 850 万組を Tanimoto
    類似度で区切り、区間ごとの |Δ pEC50| を見ます。
    """)
    return


@app.cell
def _(S_train, np, y_train):
    _iu = np.triu_indices(len(y_train), k=1)
    _sim = S_train[_iu]
    _dy = np.abs(y_train[_iu[0]] - y_train[_iu[1]])
    _edges = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    _which = np.clip(np.digitize(_sim, _edges) - 1, 0, len(_edges) - 2)
    similarity_curve = [
        {
            "sim_lo": float(_edges[b]),
            "sim_mid": float(_edges[b] + 0.025),
            "pairs": int((_which == b).sum()),
            "mean_dy": float(_dy[_which == b].mean()),
            "q90_dy": float(np.quantile(_dy[_which == b], 0.9)),
            "frac_gt1": float((_dy[_which == b] > 1).mean()),
        }
        for b in range(len(_edges) - 1)
        if (_which == b).sum() >= 5
    ]
    random_pair_dy = float(_dy.mean())

    # Keep only reasonably similar pairs for the cliff browser below.
    _keep = _sim >= 0.4
    pair_i, pair_j = _iu[0][_keep], _iu[1][_keep]
    pair_sim, pair_dy = _sim[_keep], _dy[_keep]
    return pair_dy, pair_i, pair_j, pair_sim, random_pair_dy, similarity_curve


@app.cell(hide_code=True)
def _(alt, mo, pair_dy, pair_sim, pl, random_pair_dy, similarity_curve):
    _df = pl.DataFrame(similarity_curve)
    _base = alt.Chart(_df).encode(
        x=alt.X(
            "sim_mid:Q", title="Tanimoto similarity of the pair", scale=alt.Scale(domain=[0, 1])
        )
    )
    _band = (
        _base.transform_calculate(zero="0")
        .mark_area(opacity=0.18, color="#d6336c")
        .encode(y=alt.Y("q90_dy:Q"), y2="zero:Q")
    )
    _line = _base.mark_line(point=True, color="#d6336c").encode(
        y=alt.Y("mean_dy:Q", title="|Δ pEC50| (mean, shaded to 90th pct.)"),
        tooltip=[
            alt.Tooltip("sim_lo:Q", title="bin from", format=".2f"),
            alt.Tooltip("pairs:Q", format=","),
            alt.Tooltip("mean_dy:Q", format=".2f"),
            alt.Tooltip("frac_gt1:Q", title="share with |Δ| > 1", format=".0%"),
        ],
    )
    _rule = (
        alt.Chart(pl.DataFrame({"y": [random_pair_dy]}))
        .mark_rule(strokeDash=[5, 4], color="#6b7280")
        .encode(y="y:Q")
    )
    _hi = pair_sim >= 0.5
    mo.vstack(
        [
            mo.hstack(
                [
                    (_band + _line + _rule).properties(height=260, width=360),
                    mo.vstack(
                        [
                            mo.stat(f"{random_pair_dy:.2f}", label="ランダムなペアの |Δ pEC50|"),
                            mo.stat(
                                f"{pair_dy[_hi].mean():.2f}", label="Tanimoto ≥ 0.5 の |Δ pEC50|"
                            ),
                            mo.stat(
                                f"{(pair_dy[_hi] > 1).mean():.0%}",
                                label="Tanimoto ≥ 0.5 で 10 倍以上違うペア",
                            ),
                        ]
                    ),
                ],
                widths=[1.1, 1],
                align="center",
            ),
            mo.md(f"""
    **PXR では、似ていても活性が近いとはあまり言えません。** Tanimoto ≥ 0.5 のペアでも |Δ pEC50|
    はランダムなペアの {pair_dy[_hi].mean() / random_pair_dy:.0%}
    までしか縮まらず、{(pair_dy[_hi] > 1).mean():.0%} のペアは活性が 10 倍以上違います。

    * 線は区間ごとの |Δ pEC50| の平均 網掛けは 90 パーセンタイルまで 破線はランダムなペア
    * 点にマウスを乗せると区間のペア数が出る

    似た分子は活性も似ている、という ECFP4 モデルの前提が、PXR ではあまり成り立ちません。
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(MolPair, mo, train):
    # a clean cliff: both compounds well measured (95% CI < 1 log unit), one methyl apart
    cliff_example = ["OADMET-0001944", "OADMET-0002007"]
    _rows = [train.filter(train["id"] == i).row(0, named=True) for i in cliff_example]
    _dy = abs(_rows[1]["pEC50"] - _rows[0]["pEC50"])
    mo.vstack(
        [
            mo.md(f"""
    ### 3b · activity cliff を眺める

    **activity cliff** とは、見た目は似ているのに活性値 (このデータでは pEC50) が大きく違うペアのことです。たとえば次の 2
    つは、ベンゼン環のメチル 1 つしか違いませんが (青は共通部分)、EC50 は約 {10**_dy:.0f} 倍違います。
    """),
            mo.ui.anywidget(
                MolPair(
                    _rows[0],
                    _rows[1],
                    value_cols=["pEC50"],
                    # the dumbbell's axis: every train compound's pEC50
                    value_ranges={"pEC50": (train["pEC50"].min(), train["pEC50"].max())},
                    properties=["cLogP"],
                    show_common=True,
                )
            ),
        ]
    )
    return (cliff_example,)


@app.cell(hide_code=True)
def _(mo, np):
    def _curve_panel(title, ec50, color, note):
        # x: log concentration mapped to pixels; tested range is 40..230
        w, h, x0, x1, top, bot = 350, 190, 40, 230, 30, 160
        xs = np.linspace(20, 340, 140)
        ys = bot - (bot - top) / (1 + np.exp(-(xs - ec50) / 18))
        inside = (xs >= x0) & (xs <= x1)
        solid = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[inside], ys[inside]))
        dashed = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[xs >= x1], ys[xs >= x1]))
        pts = np.linspace(x0 + 10, x1 - 5, 7)
        dots = "".join(
            f"<circle cx='{x:.1f}' cy='{bot - (bot - top) / (1 + np.exp(-(x - ec50) / 18)) + d:.1f}' r='3.2' fill='{color}'/>"
            for x, d in zip(pts, [2, -3, 3, -2, 2, -3, 1])
        )
        mid = (top + bot) / 2
        known = ec50 <= x1
        marker = (
            f"<line x1='{ec50}' y1='{mid}' x2='{ec50}' y2='{bot}' stroke='{color}' stroke-dasharray='3 3'/>"
            f"<circle cx='{ec50}' cy='{mid}' r='4.5' fill='none' stroke='{color}' stroke-width='2'/>"
            f"<text x='{ec50}' y='{bot + 14}' text-anchor='middle' font-size='11' fill='{color}'>EC50</text>"
            if known
            else f"<line x1='{ec50 - 35}' y1='{mid}' x2='{ec50 + 35}' y2='{mid}' stroke='{color}' stroke-width='2'/>"
            f"<line x1='{ec50 - 35}' y1='{mid - 6}' x2='{ec50 - 35}' y2='{mid + 6}' stroke='{color}' stroke-width='2'/>"
            f"<line x1='{ec50 + 35}' y1='{mid - 6}' x2='{ec50 + 35}' y2='{mid + 6}' stroke='{color}' stroke-width='2'/>"
            f"<text x='{ec50 + 4}' y='{mid + 22}' font-size='11' fill='{color}'>EC50 ?</text>"
        )
        return f"""
    <svg viewBox='0 0 {w} {h + 22}' width='{w}' style='max-width:100%;font-family:system-ui,sans-serif'>
      <text x='{w / 2}' y='14' text-anchor='middle' font-size='13' font-weight='600' fill='currentColor'>{title}</text>
      <rect x='{x0}' y='{top - 8}' width='{x1 - x0}' height='{bot - top + 8}' fill='#9ca3af' opacity='0.15'/>
      <text x='{(x0 + x1) / 2}' y='{top + 4}' text-anchor='middle' font-size='10' fill='#6b7280'>tested range</text>
      <line x1='20' y1='{bot}' x2='{w - 8}' y2='{bot}' stroke='currentColor' opacity='0.6'/>
      <line x1='20' y1='{top - 10}' x2='20' y2='{bot}' stroke='currentColor' opacity='0.6'/>
      <text x='{w - 8}' y='{bot + 28}' text-anchor='end' font-size='10' fill='#6b7280'>log concentration →</text>
      <text x='14' y='{top - 14}' font-size='10' fill='#6b7280'>response</text>
      <polyline points='{solid}' fill='none' stroke='{color}' stroke-width='2.2'/>
      <polyline points='{dashed}' fill='none' stroke='{color}' stroke-width='2' stroke-dasharray='5 4' opacity='0.8'/>
      {dots}{marker}
      <text x='{w / 2}' y='{h + 20}' text-anchor='middle' font-size='11' fill='#6b7280'>{note}</text>
    </svg>"""

    mo.vstack(
        [
            mo.md(r"""
    その前に、pEC50 の測り方を押さえておきます。pEC50 は、いくつかの濃度で測った応答に S
    字の用量反応曲線を当てはめて求めた値です。強い化合物は測定範囲の中で応答が頭打ちになるので、曲線の中点
    (EC50) がはっきり決まります。弱い化合物は最高濃度でも応答が上がりきらず、EC50
    は曲線を延ばした外挿になります。
    """),
            mo.hstack(
                [
                    mo.Html(
                        _curve_panel(
                            "strong compound",
                            120,
                            "#1c7ed6",
                            "plateau inside the range → EC50 is pinned down",
                        )
                    ),
                    mo.Html(
                        _curve_panel(
                            "weak compound", 275, "#d6336c", "no plateau → EC50 is extrapolated"
                        )
                    ),
                ],
                justify="center",
                gap=2,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, pl, train):
    min_sim = mo.ui.slider(
        0.4, 0.9, value=0.55, step=0.05, label="Tanimoto の下限", show_value=True
    )
    min_dy = mo.ui.slider(0.5, 3.0, value=1.5, step=0.25, label="|Δ pEC50| の下限", show_value=True)
    _bins = (
        train.with_columns(
            pl.col("pEC50")
            .cut([3, 4, 5, 6], labels=["< 3", "3–4", "4–5", "5–6", "≥ 6"])
            .alias("pEC50 bin")
        )
        .group_by("pEC50 bin")
        .agg(pl.len().alias("n"), pl.col("ci_width").median().alias("median 95% CI width"))
        .sort("pEC50 bin")
    )
    _low = _bins.filter(pl.col("pEC50 bin") == "< 3")
    _high = _bins.filter(pl.col("pEC50 bin") == "≥ 6")
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{_low['median 95% CI width'].item():.1f}",
                        label="pEC50 < 3 の CI 幅 (中央値)",
                    ),
                    mo.stat(
                        f"{_high['median 95% CI width'].item():.1f}",
                        label="pEC50 ≥ 6 の CI 幅 (中央値)",
                    ),
                    mo.stat(
                        f"{_low['n'].item()} ({_low['n'].item() / train.height:.0%})",
                        label="pEC50 < 3 の train 化合物",
                    ),
                ],
                widths="equal",
                gap=0.5,
            ),
            mo.md(r"""
    弱い化合物の pEC50 は幅を持った推定値です。cliff の片方がこの範囲にあると、測定の不確かさが大きいので、pEC50
    の差をそのまま構造の違いとは読めません。

    スライダーで「似ている」と「違う」の基準を決め、表からペアを選ぶと、下で 2 つの fingerprint を
    bit ごとに比べられます。fingerprint モデルがこの差の説明に使えるのは片方にしかない bit
    だけで、たいていありふれた部分構造が数個です。

    表は **SALI** (Structure–Activity Landscape Index, Guha & Van Drie 2008) の大きい順に並んでいます。
    SALI = |Δ pEC50| / (1 − Tanimoto) で、構造の違いが小さいのに活性の差が大きいほど大きくなります。上にあるペアほど「崖」が急です。

    * **only A / only B**: 片方にしかない bit だけ表示
    * **# mols / # envs**: その bit が立つ分子数 / 入っている部分構造の種類数
    * **Δ pEC50**: その bit がある分子とない分子の平均 pEC50 の差

    /// details | Δ pEC50 の算出方法
    train 全体 (4,139 化合物) を、ウィジェットで選んだ radius / bit 数の fingerprint でその bit
    が立っている化合物と立っていない化合物に分け、pEC50
    の平均の差をとったものです。モデルは使っていません。

    * bit どうしは一緒に立つことが多く 同じ骨格のシリーズならまとめて立つ Δ はその bit
      の効果とは限らない
    * 片方の群が小さいと極端な値になる (ほぼ全化合物で立つ bit は「立っていない」側が 100
      個ほどしかない)
    * 衝突した bit では 中に入っている部分構造すべてをまとめた平均になる
    ///
    """),
            mo.hstack([min_sim, min_dy], justify="start", gap=2),
        ]
    )
    return min_dy, min_sim


@app.cell
def _(Chem, Crippen, min_dy, min_sim, np, pair_dy, pair_i, pair_j, pair_sim, pl, train):
    _m = (pair_sim >= min_sim.value) & (pair_dy >= min_dy.value)
    _logp = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in train["smiles"]])
    _ids, _y = train["id"].to_numpy(), train["pEC50"].to_numpy()
    _a, _b = pair_i[_m], pair_j[_m]
    # Order each pair so that A is the more potent compound.
    _swap = _y[_a] < _y[_b]
    _a, _b = np.where(_swap, _b, _a), np.where(_swap, _a, _b)
    cliffs = (
        pl.DataFrame(
            {
                "A": _ids[_a],
                "B": _ids[_b],
                "Tanimoto": pair_sim[_m].astype(float).round(3),
                "pEC50 A": _y[_a],
                "pEC50 B": _y[_b],
                "Δ pEC50": pair_dy[_m].round(2),
                "Δ logP (A−B)": (_logp[_a] - _logp[_b]).round(2),
                "B CI width": train["ci_width"].to_numpy()[_b].round(2),
            }
        )
        .with_columns((pl.col("Δ pEC50") / (1 - pl.col("Tanimoto") + 1e-3)).round(1).alias("SALI"))
        .sort("SALI", descending=True)
        .with_row_index("rank", offset=1)
    )
    cliffs = cliffs.select("rank", pl.exclude("rank", "SALI"), "SALI")
    return (cliffs,)


@app.cell
def _(cliff_example, cliffs, mo):
    # where the example pair from 3b sits in the table (0 when the sliders leave it out)
    _rank = next(
        (
            r
            for r, a, b in cliffs.select("rank", "A", "B").iter_rows()
            if {a, b} == set(cliff_example)
        ),
        0,
    )
    cliff_table = mo.ui.table(
        cliffs,
        selection="single",
        initial_selection=[0] if cliffs.height else None,
        page_size=6,
        freeze_columns_right=["SALI"],
        label=f"{cliffs.height:,} ペア · SALI の大きい順"
        + (f" · 上の例は rank {_rank}" if _rank else ""),
    )
    cliff_table
    return (cliff_table,)


@app.cell
def _(MorganExplorer, cliff_table, mo, train):
    _sel = cliff_table.value
    if _sel is None or len(_sel) == 0:
        cliff_explorer = mo.md("_上の表からペアを選んでください。_")
    else:
        _r = _sel.row(0, named=True)
        _smi = dict(zip(train["id"], train["smiles"]))
        cliff_explorer = mo.ui.anywidget(
            MorganExplorer(
                [
                    {"id": _r["A"], "smiles": _smi[_r["A"]], "label": f"pEC50 {_r['pEC50 A']:.2f}"},
                    {"id": _r["B"], "smiles": _smi[_r["B"]], "label": f"pEC50 {_r['pEC50 B']:.2f}"},
                ],
                reference=train["smiles"].to_list(),
                y=train["pEC50"].to_numpy(),
                y_label="pEC50",
                pair_note=f"Δ pEC50 {_r['Δ pEC50']:.2f} · Δ logP {_r['Δ logP (A−B)']:+.2f}",
            )
        )
    cliff_explorer
    return


@app.cell(hide_code=True)
def _(cliffs, mo, pl):
    _n = cliffs.height
    _noisy = cliffs.filter(pl.col("B CI width") > 1.5).height if _n else 0
    _lip = cliffs.filter(pl.col("Δ logP (A−B)") > 0).height if _n else 0
    mo.md(
        f"""
    この条件の {_n} ペアについて:

    * {_noisy} ペアは弱いほうの CI 幅が 1.5 log 単位を超える 不確かさが大きいので 差をそのまま構造の違いとは読めない
    * {_lip} ペア ({_lip / max(_n, 1):.0%}) は強いほうが計算 logP も高い 個々の bit
      ではなく分子全体の性質が効いている

    cliff の差は、測定の不確かさがあるうえに分子全体の性質も関わっているので、fingerprint だけでは説明できません。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c · 違う分子なのに、fingerprint は同じ

    2 つの異なる分子がまったく同じ bit ベクトルになるなら、その bit
    で作ったモデルは両者に**必ず**同じ値を予測します。train の中のそうしたグループを、fingerprint
    が区別できない理由とあわせてすべて示します。
    """)
    return


@app.cell
def _(Chem, X_train, mo, np, pl, rdFingerprintGenerator, train):
    from collections import defaultdict

    def why_identical(smiles: list[str]) -> str:
        """Classify a group of molecules that share a Morgan bit vector."""
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        flat = {Chem.MolToSmiles(m, isomericSmiles=False) for m in mols}
        if len(flat) > 1:
            # Different constitution, same set of radius-2 environments: only repeat counts differ.
            return "環サイズ / 鎖長"
        return "立体化学"  # same constitution: only stereo differs

    _groups = defaultdict(list)
    for _i, _fp in enumerate(X_train):
        _groups[_fp.tobytes()].append(_i)
    _twins = sorted(
        (g for g in _groups.values() if len(g) > 1),
        key=lambda g: -np.ptp(train["pEC50"].to_numpy()[g]),
    )
    _smiles = train["smiles"].to_list()
    _rows = [
        {**train.row(i, named=True), "group": gi + 1, "why": why_identical([_smiles[k] for k in g])}
        for gi, g in enumerate(_twins)
        for i in g
    ]
    fp_twins = pl.DataFrame(_rows).select("id", "smiles", "group", "why", "pEC50", "ci_width")
    _summary = fp_twins.group_by("group", "why").agg(
        (pl.col("pEC50").max() - pl.col("pEC50").min()).alias("range")
    )
    _n_stereo = _summary.filter(pl.col("why") != "環サイズ / 鎖長").height
    _chiral_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )

    def _splits(g):
        return (
            len(
                {
                    _chiral_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(_smiles[k])).tobytes()
                    for k in g
                }
            )
            > 1
        )

    _n_split = sum(
        _splits(g) for g in _twins if why_identical([_smiles[k] for k in g]) != "環サイズ / 鎖長"
    )

    # one row per group: stereo groups first, then ring size / chain length; largest spread first
    _y = train["pEC50"].to_numpy()
    _ids = train["id"].to_list()
    twin_smiles = dict(zip(_ids, _smiles))
    _table_rows = []
    for _g in sorted(
        _twins,
        key=lambda g: (why_identical([_smiles[k] for k in g]) == "環サイズ / 鎖長", -np.ptp(_y[g])),
    ):
        _why = why_identical([_smiles[k] for k in _g])
        _a, _b = sorted(_g, key=lambda k: -_y[k])[:2]
        _table_rows.append(
            {
                "why": _why,
                "A": _ids[_a],
                "B": _ids[_b],
                "pEC50 A": _y[_a],
                "pEC50 B": _y[_b],
                "Δ pEC50": round(float(_y[_a] - _y[_b]), 2),
                "includeChirality": "—"
                if _why == "環サイズ / 鎖長"
                else ("分かれる" if _splits(_g) else "分かれない"),
            }
        )
    twin_table = mo.ui.table(
        pl.DataFrame(_table_rows),
        selection="single",
        initial_selection=[0],
        page_size=10,
        label="同じ fingerprint のグループ (行を選ぶと下で比べられる)",
    )
    mo.vstack(
        [
            mo.md(
                f"""
    **{len(_twins)} グループ** (train 化合物 {fp_twins.height} 個) が同じ fingerprint
    に重なっていて、グループ内の pEC50 の差は最大 **{_summary["range"].max():.2f}** です。理由は 2
    つあります。

    **立体化学 ({_n_stereo} グループ)**

    * RDKit の ECFP4 は `includeChirality=True` を渡さない限り キラリティも二重結合の E/Z も無視する
    * どのグループも 同じ化合物について立体を指定したレコードとしていないレコードの組
    * 例
        * lansoprazole (OADMET-0003758) と dexlansoprazole (OADMET-0003782) 差 0.94
        * bupivacaine (OADMET-0001982) と levobupivacaine (OADMET-0003731)
        * rifampicin の E/Z 表記あり (OADMET-0002338) となし (OADMET-0003649)
    * `includeChirality=True` にすると {_n_stereo} グループ中 {_n_split} グループが分かれる
      lansoprazole のスルホキシドの立体中心は拾われない

    **環サイズ / 鎖長 ({len(_twins) - _n_stereo} グループ)**

    * radius 2 の範囲ではどの原子も同じ周囲を見るので 部分構造の集合は同じで違うのは出てくる回数だけ
    * bit ベクトルは回数を持たないので区別できない count fingerprint なら区別できる
    * ただし count fingerprint も同じように折りたたむので 別の部分構造が同じ bit に衝突すると
      その回数も足し合わされる bit の値は「その部分構造が何回あるか」とは限らない

    bit ベクトルが同じ分子は、どのモデルでも同じ予測になります。立体 (デフォルトでは) と出現回数は ECFP4 からは読み取れません。
    """
            ),
            twin_table,
        ]
    )
    return twin_smiles, twin_table


@app.cell
def _(MorganExplorer, mo, train, twin_smiles, twin_table):
    _sel = twin_table.value
    if _sel is None or len(_sel) == 0:
        twin_explorer = mo.md("_上の表からグループを選んでください。_")
    else:
        _r = _sel.row(0, named=True)
        twin_explorer = mo.ui.anywidget(
            MorganExplorer(
                [
                    {
                        "id": _r["A"],
                        "smiles": twin_smiles[_r["A"]],
                        "label": f"pEC50 {_r['pEC50 A']:.2f}",
                    },
                    {
                        "id": _r["B"],
                        "smiles": twin_smiles[_r["B"]],
                        "label": f"pEC50 {_r['pEC50 B']:.2f}",
                    },
                ],
                reference=train["smiles"].to_list(),
                y=train["pEC50"].to_numpy(),
                y_label="pEC50",
                pair_note=f"Δ pEC50 {_r['Δ pEC50']:.2f}",
                stereo_labels=True,
            )
        )
    twin_explorer
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d · 部分構造の寄せ集めでは表せないもの

    PXR
    のリガンド結合ポケットは大きく、柔軟で、疎水的で、まったく異なる骨格を受け入れることで知られています。結合を決めているのが特定の部分構造ではなく、分子全体がどれだけ脂溶的で大きいかだとすると、部分構造の有無で分子を表す
    ECFP4 とは相性が悪いことになります。
    """)
    return


@app.cell
def _(Chem, Crippen, Descriptors, np, rdmd, test, train):
    DESCRIPTORS = {
        "MolLogP": Crippen.MolLogP,
        "MolWt": Descriptors.MolWt,
        "TPSA": rdmd.CalcTPSA,
        "HBD": rdmd.CalcNumHBD,
        "HBA": rdmd.CalcNumHBA,
        "RotBonds": rdmd.CalcNumRotatableBonds,
        "Rings": rdmd.CalcNumRings,
        "AromaticRings": rdmd.CalcNumAromaticRings,
        "FractionCSP3": rdmd.CalcFractionCSP3,
        "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
        "Heteroatoms": rdmd.CalcNumHeteroatoms,
        "Halogens": lambda m: sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms()),
        "MolMR": Crippen.MolMR,
        "StereoCenters": lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
    }

    def descriptor_matrix(smiles):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        return np.array([[f(m) for f in DESCRIPTORS.values()] for m in mols], dtype=float)

    D_train = descriptor_matrix(train["smiles"].to_list())
    D_test = descriptor_matrix(test["smiles"].to_list())
    return DESCRIPTORS, D_test, D_train


@app.cell(hide_code=True)
def _(DESCRIPTORS, mo):
    prop_pick = mo.ui.dropdown(list(DESCRIPTORS), value="MolLogP", label="記述子")
    prop_pick
    return (prop_pick,)


@app.cell(hide_code=True)
def _(DESCRIPTORS, D_train, alt, mo, pl, prop_pick, spearmanr, y_train):
    _j = list(DESCRIPTORS).index(prop_pick.value)
    _rhos = sorted(
        ((name, spearmanr(D_train[:, k], y_train)[0]) for k, name in enumerate(DESCRIPTORS)),
        key=lambda t: -abs(t[1]),
    )
    _df = pl.DataFrame({prop_pick.value: D_train[:, _j], "pEC50": y_train})
    _chart = (
        alt.Chart(_df)
        .mark_rect()
        .encode(
            x=alt.X(f"{prop_pick.value}:Q", bin=alt.Bin(maxbins=40)),
            y=alt.Y("pEC50:Q", bin=alt.Bin(maxbins=30)),
            color=alt.Color("count():Q", scale=alt.Scale(scheme="greys"), legend=None),
        )
        .properties(width=320, height=250)
    )
    _bars = (
        alt.Chart(
            pl.DataFrame({"descriptor": [n for n, _ in _rhos], "Spearman ρ": [r for _, r in _rhos]})
        )
        .mark_bar()
        .encode(
            y=alt.Y("descriptor:N", sort=None, title=None),
            x=alt.X("Spearman ρ:Q", scale=alt.Scale(domain=[-0.5, 0.5])),
            color=alt.condition(
                alt.datum.descriptor == prop_pick.value, alt.value("#d6336c"), alt.value("#9ca3af")
            ),
        )
        .properties(width=220, height=250)
    )
    mo.hstack([_chart, _bars], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(
    D_test, D_train, S_test, mo, np, pair_dy, pair_i, pair_j, pair_sim, spearmanr, y_test, y_train
):
    _m = pair_sim >= 0.5
    _rho_pairs = spearmanr(abs(D_train[pair_i[_m], 0] - D_train[pair_j[_m], 0]), pair_dy[_m])[0]
    _rho_logp = spearmanr(D_test[:, 0], y_test)[0]  # column 0 = MolLogP
    _order = np.argsort(-S_test, axis=1)

    def _knn_rho(k):
        return spearmanr(y_train[_order[:, :k]].mean(1), y_test)[0]

    mo.vstack(
        [
            mo.md("**分子全体の数値 1 つでも、ECFP4 の kNN より test をうまく順位づけられます。**"),
            mo.hstack(
                [
                    mo.stat(f"{_rho_logp:.2f}", label="logP だけの Spearman ρ (test)"),
                    mo.stat(f"{_knn_rho(1):.2f}", label="ECFP4 kNN k=1 の ρ (test)"),
                    mo.stat(f"{_knn_rho(50):.2f}", label="ECFP4 kNN k=50 の ρ (test)"),
                ],
                widths="equal",
                gap=0.5,
            ),
            mo.md(f"""
    * 左は選んだ記述子と pEC50 の分布 右は 14 個の記述子それぞれと pEC50 の Spearman ρ (train)
    * Tanimoto ≥ 0.5 のペアの中でも logP の差は活性の差と弱いながら連動する (ρ = {_rho_pairs:.2f})
    * bit ベクトルには「少しだけ脂溶性が高い」という軸がない メチルを 1 つ足しても bit
      が立つか立たないかだけ

    脂溶性のような分子全体の連続的な性質は、部分構造があるかないかの集まりでは表せません。
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, model_scores):
    mo.md(f"""
    ## 4 · モデルの中身

    多くの人が最初に作るモデル、**2048 bit の ECFP4 で学習した LightGBM** が何を学習したのかを、2 つの方法で調べます。

    * **特徴量重要度** (bit ごとの gain の合計): 木がどの bit で多く分岐したか
    * **TreeSHAP** (LightGBM の `pred_contrib=True`): 各 bit
      がこの分子の予測をどれだけ上げたか下げたか 寄与をその bit
      を立てた原子に配分すると原子ごとのマップになる (Riniker & Landrum の similarity map
      と同じ発想)

    このモデルの test での成績は MAE **{model_scores["MAE"]:.2f}**、Spearman ρ
    **{model_scores["rho"]:.2f}** です。5 章の Model lab の基準 (ECFP4 bit) と同じ設定なので、予測と実測の比較はそちらで見られます。
    """)
    return


@app.cell
def _(fingerprint_matrix, lgb, mo, np, spearmanr, test, train, y_test, y_train):
    N_BITS = 2048  # the usual default, as in most first models
    Xm_train = fingerprint_matrix(train["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    Xm_test = fingerprint_matrix(test["smiles"].to_list(), 2, N_BITS).astype(np.float32)
    with mo.status.spinner(f"{N_BITS} bit の ECFP4 で LightGBM を学習中…"):
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            random_state=0,
            verbose=-1,
        ).fit(Xm_train, y_train)
    pred_test = model.predict(Xm_test)
    model_scores = {
        "MAE": float(np.abs(pred_test - y_test).mean()),
        "rho": float(spearmanr(pred_test, y_test)[0]),
    }
    return N_BITS, Xm_test, Xm_train, model, model_scores, pred_test


@app.cell
def _(N_BITS, Xm_train, model, np):
    # per-bit importance of the model: LightGBM gain, and TreeSHAP over the training set
    bit_gain = model.booster_.feature_importance("gain")
    _abs, _on_sum = np.zeros(N_BITS), np.zeros(N_BITS)
    for _s in range(0, len(Xm_train), 500):  # in chunks: at 8192 bits the full matrix is large
        _x = Xm_train[_s : _s + 500]
        _c = model.predict(_x, pred_contrib=True)[:, :-1]  # last column = expected value
        _abs += np.abs(_c).sum(0)
        _on_sum += (_c * _x).sum(0)
    bit_shap = _abs / len(Xm_train)
    _n_on = Xm_train.sum(0)
    bit_effect = np.divide(_on_sum, _n_on, out=np.zeros(N_BITS), where=_n_on > 0)
    return bit_effect, bit_gain, bit_shap


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 重要度の高い bit と、その中身

    下の表は、モデルの bit を重要度の高い順に並べたものです。

    * **gain** / **mean |SHAP|**: 重要度 (全 bit に対する割合) どちらで並べるか選べる
    * **mean SHAP (bit on)**: その bit が立っている分子での平均寄与 赤は予測を上げ 青は下げる
    * **main substructure**: その bit で一番多い部分構造と その bit が立つ分子のうちそれを含む割合
    * 行をクリックすると その bit に入る部分構造と その bit が立っている分子が下に出る
    * 列見出しをもう一度押すと並びが逆になり モデルが使わなかった bit (gain 0) を見られる

    /// details | 各指標の定義
    * **gain**: その bit での分岐が学習中に減らした損失の合計 (LightGBM の
      [`feature_importance(importance_type="gain")`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance))
    * **mean |SHAP|**: train の全分子について その bit の TreeSHAP 寄与 (LightGBM の
      [`predict(pred_contrib=True)`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict)) の絶対値を平均したもの ([SHAP](https://shap.readthedocs.io/en/latest/))
    * **mean SHAP (bit on)**: 同じ寄与を その bit が立っている分子だけで平均したもの (符号つき)
    * gain 0 の bit どうしは 立っている分子の多い順に並ぶ
    ///
    """)
    return


@app.cell
def _(BitImportance, bit_effect, bit_gain, bit_shap, mo, train):
    bit_importance = mo.ui.anywidget(
        BitImportance(
            train["smiles"].to_list(),
            importance={"gain": bit_gain, "mean |SHAP|": bit_shap},
            effect=bit_effect,
            effect_label="mean SHAP (bit on)",
            ids=train["id"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    bit_importance
    return


@app.cell(hide_code=True)
def _(N_BITS, Xm_test, bit_gain, census_for, mo, np, pl, pred_test, spearmanr, train, y_test):
    _census = census_for(train["smiles"].to_list(), 2, N_BITS)
    _gain = bit_gain
    _top = np.argsort(-_gain)[:15]
    _top_bits = pl.DataFrame(
        {
            "rank": np.arange(1, 16),
            "bit": _top.astype(int),
            "gain": (_gain[_top] / _gain.sum()).round(4),
            "# environments in bit": _census.n_envs[_top].astype(int),
            "# training molecules with bit": _census.on[:, _top].sum(0).astype(int),
            # share of the molecules setting the bit that contain its most common substructure
            "top substructure share": [
                round(_census.examples[int(b)][0]["count"] / max(int(_census.on[:, b].sum()), 1), 3)
                for b in _top
            ],
        }
    )
    _envs = np.median(_top_bits["# environments in bit"])
    _n_on = _census.on.sum(0)
    _unused = np.flatnonzero(_gain == 0)
    _n0, _n0_bit = len(_unused), int(_unused[np.argmax(_n_on[_unused])]) if len(_unused) else -1
    _n0_mols = int(_n_on[_n0_bit]) if len(_unused) else 0
    # does a test molecule with many ignored bits get a worse prediction?
    _k = (Xm_test[:, _unused] > 0).sum(1)
    _err = np.abs(pred_test - y_test)
    _rho0 = spearmanr(_k, _err)[0]
    _mae_lo, _mae_hi = _err[_k <= 2].mean(), _err[_k >= 7].mean()
    _share = np.median(_top_bits["top substructure share"])
    _mixed = _top_bits.filter(pl.col("top substructure share") < 0.8)["bit"].to_list()
    _mixed_note = (
        f"* 例外は 2 番目の部分構造も多くの分子に現れる bit ({' '.join(map(str, _mixed))}) で こうした bit の重要度はどの部分構造のものか決められない"
        if _mixed
        else "* 上位 15 bit に 一番多い部分構造が 8 割を切る bit はない"
    )
    mo.vstack(
        [
            mo.md("""
    **重要度上位の bit には何種類もの部分構造が入っていますが、ほとんどは 1 つの部分構造がその bit
    を占めています。**
    """),
            mo.hstack(
                [
                    mo.stat(f"{_envs:.0f}", label="上位 15 bit あたりの部分構造数 (中央値)"),
                    mo.stat(f"{_share:.0%}", label="一番多い部分構造が占める割合 (中央値)"),
                ],
                widths="equal",
                gap=0.5,
            ),
            mo.md(f"""
    * {N_BITS} bit では上位 15 bit に 1 bit あたり約 {_envs:.0f} 種類入るが その bit が立つ分子の中央値
      {_share:.0%} は一番多い部分構造 1 つで説明できる
    {_mixed_note}
    * モデルが一度も使わなかった bit は {_n0:,} 個 一番多く立つものは {_n0_mols:,} 分子 (bit {_n0_bit})
      これらを落として学習し直しても test の成績はほぼ同じ (MAE 0.59)
    * こうした bit が多い test 分子ほど誤差はわずかに大きい (Spearman {_rho0:.2f} 0〜2 個で MAE
      {_mae_lo:.2f} 7 個以上で {_mae_hi:.2f})

    重要度は bit につく値ですが、上位の bit はたいてい 1 つの部分構造が占めているので、その部分構造の重要度として読めます。例外の bit は中身を確かめる必要があります。
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1 つの予測を読み解く

    下のペアは、チャレンジ全体で 2 番目に難しかった test 化合物 **OADMET-0006254** (pEC50
    2.06、セクション 1 で自分自身と衝突していた分子) と、train での最近傍 **OADMET-0002810** (pEC50
    5.95) です。違いはベンゼンの CH → ピリジンの N の 1 か所だけ。OpenADMET
    の解析によれば、関連する共結晶構造ではこの窒素が SER247
    と水素結合しており、ポケット内でのリガンドの収まり方が変わっている可能性が高いそうです。Tier-1
    のチームは、どこもこの化合物を過大に予測しました。

    原子の色はモデルの TreeSHAP 寄与です (赤は予測 pEC50 を上げ、青は下げます)。bit
    をクリックすると、代わりにその部分構造が表示されます。表の **SHAP** 列は、A と B それぞれでの
    bit の寄与です。メニューからは、ほかに予測が大きく外れた test 化合物も選べます。
    """)
    return


@app.cell
def _(S_test, nn_test, pl, pred_test, test, train):
    cliff_pairs = (
        pl.DataFrame(
            {
                "test id": test["id"],
                "test pEC50": test["pEC50"],
                "predicted": pred_test.round(2),
                "NN id": train["id"].to_numpy()[nn_test],
                "NN pEC50": train["pEC50"].to_numpy()[nn_test],
                "Tanimoto": S_test.max(1).astype(float).round(3),
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
    _labels = {
        f"{r['test id']} (実測 {r['test pEC50']:.2f}, 予測 {r['predicted']:.2f}) vs {r['NN id']}": r[
            "test id"
        ]
        for r in _worst.iter_rows(named=True)
    }
    pair_pick = mo.ui.dropdown(
        _labels,
        value=next(k for k in _labels if k.startswith("OADMET-0006254")),
        label="ペア (test 化合物と、train での最近傍)",
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
    pair_X = fingerprint_matrix([_smi[i] for i in _ids], 2, N_BITS).astype(np.float32)
    pair_contrib = model.predict(pair_X, pred_contrib=True)  # last column = expected value
    _pred = pair_contrib.sum(1)
    _maps = [{str(b): float(c[b]) for b in np.flatnonzero(x)} for c, x in zip(pair_contrib, pair_X)]
    _y = dict(zip(data["id"], data["pEC50"]))
    pair_y = [_y[i] for i in _ids]
    pair_smiles = [_smi[i] for i in _ids]
    shap_explorer = mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _ids[0],
                    "smiles": _smi[_ids[0]],
                    "label": f"test · true {_y[_ids[0]]:.2f} · predicted {_pred[0]:.2f}",
                },
                {
                    "id": _ids[1],
                    "smiles": _smi[_ids[1]],
                    "label": f"train · true {_y[_ids[1]]:.2f} · predicted {_pred[1]:.2f}",
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
            pair_note=f"baseline (mean prediction) {pair_contrib[0, -1]:.2f}",
        )
    )
    shap_explorer
    return pair_X, pair_contrib, pair_smiles, pair_y


@app.cell(hide_code=True)
def _(N_BITS, census_for, mo, pair_X, pair_contrib, pair_smiles, pair_y, train):
    from chemari import molecule_bit_tiles

    _a, _b = pair_X.astype(bool)
    _diff = pair_contrib[0, :-1] - pair_contrib[1, :-1]  # per-feature A − B
    _differ, _shared, _neither = _a ^ _b, _a & _b, ~(_a | _b)
    _gap = _diff.sum()
    _measured = pair_y[0] - pair_y[1]
    _lift = pair_contrib[0, :-1][_shared].sum()  # what the shared bits add to A's prediction
    # substructures only A has: how often does each appear in train?
    _census = census_for(train["smiles"].to_list(), 2, N_BITS)
    _only_a = [
        t
        for t in molecule_bit_tiles(pair_smiles[0], 2, N_BITS)
        if _a[t["bit"]] and not _b[t["bit"]]
    ]
    _train_count = [
        next((e["count"] for e in _census.examples.get(t["bit"], []) if e["uid"] == t["uid"]), 0)
        for t in _only_a
    ]
    _rare = sum(c <= 3 for c in _train_count)
    mo.md(
        f"""
    **予測の差はどこから来るのか?** モデルの予測は A − B = **{_gap:+.2f}** (実測は {_measured:+.2f})
    です。TreeSHAP でこの差を bit ごとに分けると、A と B で違う **{int(_differ.sum())}** 個の bit が
    **{_diff[_differ].sum():+.2f}**、共通の **{int(_shared.sum())}** 個の bit が
    **{_diff[_shared].sum():+.2f}**、どちらにもない bit が **{_diff[_neither].sum():+.2f}** です。

    * 構造の小さな違いでも {int(_differ.sum())} 個の bit が変わる 違う原子から radius 2
      以内の部分構造がすべて変わるため
    * それでも予測の差は {abs(_gap):.2f} しかなく 実測の差 {abs(_measured):.2f} には届かない 共通の
      bit は A の予測を {_lift:+.2f} 押し上げている
    * A にしかない部分構造 {len(_only_a)} 種類のうち {_rare} 種類は train に 3
      分子以下しか出てこない その bit の重みは 同じ bit に入る別の部分構造から学習されたもの
    * 原子マップに描けるのは立っている bit だけ bit
      が立っていないことも木にとっては意味があるが原子の上には描けない

    モデルが予測できた差は、実測の差の {abs(_gap) / abs(_measured):.0%} だけでした。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Model lab

    ここまでに見た弱点には、それぞれ対応する手があります。

    * **count fingerprint**: 出現回数を残す (3c)
    * **bit 数を増やす**: 衝突を減らす (1)
    * **キラリティを含める**: 立体を区別する (3c)
    * **分子全体の記述子**: 部分構造では表せない性質を補う (3d)

    LightGBM を train で学習し、test の 513 化合物で評価します。3
    つの基準設定は計算済みで、設定を変えて**学習**を押すとスコアボードに行が追加されます。「参考」の行は
    CheMeleon (後述) を使ったもので、notebook の外で同じ条件で計算した結果です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lab_features = mo.ui.multiselect(
        ["ECFP bit", "ECFP count", "RDKit 記述子"],
        value=["ECFP count"],
        label="特徴量",
    )
    lab_radius = mo.ui.dropdown(
        {"ECFP2 (radius 1)": 1, "ECFP4 (radius 2)": 2, "ECFP6 (radius 3)": 3},
        value="ECFP4 (radius 2)",
        label="radius",
    )
    lab_bits = mo.ui.dropdown(
        {"256": 256, "1024": 1024, "2048": 2048, "8192": 8192}, value="2048", label="bit 数"
    )
    lab_chiral = mo.ui.checkbox(label="キラリティを含める")
    lab_run = mo.ui.run_button(label="学習")
    mo.hstack(
        [lab_features, lab_radius, lab_bits, lab_chiral, lab_run], justify="start", gap=1, wrap=True
    )
    return lab_bits, lab_chiral, lab_features, lab_radius, lab_run


@app.cell
def _(Chem, mo, np, test, train):
    from rdkit.Chem import Descriptors as _Descriptors

    def rdkit_descriptors(smiles):
        """All RDKit 2D descriptors (~217); non-finite values become NaN (LightGBM handles them)."""
        rows = [
            list(_Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s)).values()) for s in smiles
        ]
        out = np.array(rows, dtype=float)
        out[~np.isfinite(out)] = np.nan
        return out

    with mo.status.spinner("RDKit 記述子を計算中…"):
        R_train = rdkit_descriptors(train["smiles"].to_list())
        R_test = rdkit_descriptors(test["smiles"].to_list())
    return R_test, R_train


@app.cell
def _(
    Chem,
    R_test,
    R_train,
    S_test,
    lgb,
    np,
    rdFingerprintGenerator,
    spearmanr,
    test,
    train,
    y_test,
    y_train,
):
    def featurize(smiles, parts, radius, n_bits, chiral):
        gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits, includeChirality=chiral
        )
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        blocks = []
        if "ECFP bit" in parts:
            blocks.append(np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32))
        if "ECFP count" in parts:
            blocks.append(
                np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
            )
        return blocks

    def evaluate(parts, radius=2, n_bits=2048, chiral=False, seed=0):
        """Fit LightGBM on train, return test metrics and predictions."""
        tr = featurize(train["smiles"].to_list(), parts, radius, n_bits, chiral)
        te = featurize(test["smiles"].to_list(), parts, radius, n_bits, chiral)
        if "RDKit 記述子" in parts:
            tr.append(R_train)
            te.append(R_test)
        model = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            verbose=-1,
        ).fit(np.hstack(tr), y_train)
        pred = model.predict(np.hstack(te))
        fp_desc = " + ".join(p.replace("ECFP", f"ECFP{2 * radius}") for p in parts)
        if any(p.startswith("ECFP") for p in parts):
            fp_desc += f" · {n_bits} bit{' · キラリティあり' if chiral else ''}"
        return {
            "features": fp_desc,
            "MAE": round(float(np.abs(pred - y_test).mean()), 3),
            "Spearman ρ": round(float(spearmanr(pred, y_test)[0]), 3),
            "pred. SD": round(float(pred.std()), 2),
            "_pred": pred,
        }

    nn_sim_test = S_test.max(1)
    return evaluate, nn_sim_test


@app.cell
def _(evaluate, mo):
    with mo.status.spinner("3 つの基準モデルを学習中…"):
        baseline_runs = [
            evaluate(["ECFP bit"]),
            evaluate(["RDKit 記述子"]),
            evaluate(["ECFP bit", "RDKit 記述子"]),
        ]
    return (baseline_runs,)


@app.cell
def _(np, pl, spearmanr, test, y_test):
    # CheMeleon (a GNN pretrained to predict Mordred descriptors) needs PyTorch, so its runs were
    # computed outside the notebook with the same split and LightGBM settings (results/)
    from pathlib import Path as _Path

    _file = "chemeleon_test_predictions.csv"
    _local = _Path("results") / _file
    _url = "https://raw.githubusercontent.com/N283T/chemari/main/results/" + _file
    try:
        _preds = test.select("id").join(pl.read_csv(_local if _local.exists() else _url), on="id")
    except (OSError, pl.exceptions.PolarsError):
        _preds = None  # offline and no local copy: the scoreboard shows only the notebook's runs
    reference_runs = []
    for _col in [] if _preds is None else _preds.columns[1:]:
        _pred = _preds[_col].to_numpy()
        reference_runs.append(
            {
                "features": _col,
                "MAE": round(float(np.abs(_pred - y_test).mean()), 3),
                "Spearman ρ": round(float(spearmanr(_pred, y_test)[0]), 3),
                "pred. SD": round(float(_pred.std()), 2),
                "_pred": _pred,
            }
        )
    return (reference_runs,)


@app.cell
def _(mo):
    get_runs, set_runs = mo.state([])
    return get_runs, set_runs


@app.cell
def _(evaluate, lab_bits, lab_chiral, lab_features, lab_radius, lab_run, mo, set_runs):
    mo.stop(not lab_run.value)
    mo.stop(not lab_features.value, mo.md("特徴量を 1 つ以上選んでください。").callout(kind="warn"))
    with mo.status.spinner("学習中…"):
        _res = evaluate(lab_features.value, lab_radius.value, lab_bits.value, lab_chiral.value)
    set_runs(lambda runs: [*runs, _res])
    return


@app.cell(hide_code=True)
def _(baseline_runs, get_runs, mo, pl, reference_runs, y_test):
    all_runs = baseline_runs + reference_runs + get_runs()
    _nb, _nr = len(baseline_runs), len(reference_runs)
    _board = pl.DataFrame(
        [
            {k: v for k, v in r.items() if not k.startswith("_")}
            | {"source": "基準" if i < _nb else "参考" if i < _nb + _nr else "追加"}
            for i, r in enumerate(all_runs)
        ]
    )
    run_pick = mo.ui.table(
        _board,
        selection="single",
        # the newest run of your own, else the best baseline
        initial_selection=[len(all_runs) - 1 if get_runs() else _nb - 1],
        label=f"スコアボード (test の実測 pEC50 の SD = {y_test.std():.2f}。行を選ぶと詳しく見られます)",
        page_size=8,
    )
    run_pick
    return all_runs, run_pick


@app.cell(hide_code=True)
def _(all_runs, alt, mo, nn_sim_test, pl, run_pick, test, y_test):
    _i = (
        run_pick.value.select(pl.col("features")).to_series().to_list()[0]
        if run_pick.value is not None and len(run_pick.value)
        else None
    )
    _run = next((r for r in reversed(all_runs) if r["features"] == _i), all_runs[-1])
    _pred = _run["_pred"]
    _df = pl.DataFrame(
        {
            "id": test["id"],
            "true pEC50": y_test,
            "predicted": _pred,
            "NN similarity": nn_sim_test,
            "error": _pred - y_test,
        }
    ).with_columns(
        pl.col("true pEC50").cut([4, 5, 6], labels=["< 4", "4–5", "5–6", "≥ 6"]).alias("true bin")
    )
    _lim = [1.5, 7.5]
    _scatter = (
        alt.Chart(_df)
        .mark_circle(size=26, opacity=0.6)
        .encode(
            x=alt.X("predicted:Q", scale=alt.Scale(domain=_lim)),
            y=alt.Y("true pEC50:Q", scale=alt.Scale(domain=_lim)),
            color=alt.Color("NN similarity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[
                "id",
                alt.Tooltip("true pEC50:Q", format=".2f"),
                alt.Tooltip("predicted:Q", format=".2f"),
            ],
        )
        .properties(width=280, height=260, title=_run["features"])
    ) + alt.Chart(pl.DataFrame({"x": _lim, "y": _lim})).mark_line(
        color="#9ca3af", strokeDash=[4, 4]
    ).encode(x="x", y="y")
    _bias = (
        alt.Chart(_df)
        .mark_boxplot(extent="min-max", size=26, color="#d6336c")
        .encode(
            x=alt.X("true bin:N", sort=["< 4", "4–5", "5–6", "≥ 6"], title="true pEC50"),
            y=alt.Y("error:Q", title="prediction − truth"),
        )
        .properties(width=220, height=260, title="error by true potency")
    )
    _zero = alt.Chart(pl.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
    mo.vstack(
        [
            mo.hstack([_scatter, _bias + _zero], justify="start", gap=2),
            mo.md(f"""
    **どのモデルも予測が平均に寄ります。**

    * 予測の SD は {_pred.std():.2f} で 実測の {y_test.std():.2f} より小さい
    * 弱い化合物は強めに 強い化合物は弱めに予測される (右の箱ひげ図)
    * 基準の 3 つでは ECFP4 だけのモデルがいちばん幅が狭い
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    試してみる価値があるもの (数字は test の MAE):

    * **bit → count**: 出現回数が加わり 3c の環サイズ違いも区別できる 0.59 → 0.55
    * **bit 数**: 256 では明らかに悪い (0.64) が 2048 と 8192 の差は小さい (0.59 → 0.58)
    * **キラリティを含める**: 変わる化合物が少なくスコアはほぼ同じ 3c
      の立体のグループは区別できるようになる
    * **RDKit 記述子**: 記述子だけ (217 種) で 0.56 と ECFP4 (0.59) を上回り ECFP4 と組み合わせると
      0.53
    * **事前学習済み GNN の埋め込み** (参考): Mordred 記述子の予測で事前学習した GNN [CheMeleon](https://github.com/JacksonBurns/chemeleon)
      の出力を特徴量にすると [0.54](https://github.com/N283T/chemari/blob/main/results/chemeleon_lightgbm.csv)

    部分構造 (fingerprint) と分子全体の性質 (記述子) を組み合わせるのが一番効きます。記述子を予測するように事前学習した
    CheMeleon もある意味その組み合わせで、[チャレンジでの私のレポート](https://n283t.github.io/openadmet-pxr-model-report/) でも GNN
    の事前学習モデルが効きました。ただ、どのモデルでも予測が平均に寄る傾向は残ります。

    ## 6 · まとめ

    **ECFP4 について**

    * **ECFP4 は局所的な部分構造があるかどうかの集まり**: 出現回数 立体 (デフォルト)
      分子全体の性質は残らない
    * **folding で無関係な部分構造が同じ bit に入る**: ただしよく立つ bit はたいてい 1
      つの部分構造が占めている
    * **Tanimoto が高くても活性が近いとは限らない**

    **PXR で起きたこと**

    * **test は train に近いのに 最近傍の活性はほとんど当てにならない**: test は ECFP4
      近傍として作られたが kNN (k=1) の順位相関はほぼ 0
    * **分子全体の性質も効いている**: logP 1 つでも ECFP4 の kNN より test をよく順位づけ
      RDKit 記述子だけのモデルは ECFP4 だけのモデルを上回った
    * **cliff の差には測定の不確かさが混ざる**: 弱い化合物の pEC50 は外挿で幅を持つので 差をそのまま構造の違いとは読めない

    **ECFP4 を使うときは**

    * 回帰では count も試す (PXR では bit より良かった) ただし衝突した bit
      には別々の部分構造の回数が足し合わされる
    * 立体が効くならキラリティをオンにする
    * 分子全体の記述子と組み合わせる
    * bit を解釈するときは その bit に入っている部分構造まで確かめる

    ECFP4
    は今でも手堅い出発点です。中身と弱点を知っていれば、効かないデータに当たったときに、何を疑い何を足せばいいかがわかります。

    ---

    ### この notebook について

    * **データ:**
      [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0)。train と、フェーズ 1・フェーズ 2 で公開された test のラベルを使っています。test
      の設計と最難関化合物の解析は、参考文献 [4]–[6] によります。
    * **ウィジェット:** `ECFPMovie`、`ECFPStepper`、`MolGrid`、`MorganBitTiles`、`BitAtlas`、`MorganExplorer`
      は、この notebook のために作った anywidget コンポーネントです
      ([ソース](https://github.com/N283T/chemari))。
    * **AI の利用:** ウィジェット、動画、notebook の骨組みのコーディングには、Claude (Anthropic)
      をアシスタントとして使いました。問いの立て方、解析の選び方、解釈は私自身の PXR
      チャレンジでの取り組みに基づくもので、表示している数値はすべてこの notebook
      の中でその場で計算しています。

    参加者がどんな手法を使ったかは、[チャレンジ後の解析](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)に Tier 1 の 28 チーム分の表としてまとまっています。各チームのモデルレポートは[結果発表](https://openadmet.ghost.io/its-the-end-of-the-pxr-challenge-as-we-know-it-and-i-feel-fine/)から読めます。私のレポート (Activity トラック 4 位) は[こちら](https://n283t.github.io/openadmet-pxr-model-report/)です。お察しのとおり、fingerprint を使ったモデルは最終的なアンサンブルに採用しませんでした。

    ### 参考文献

    1. Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742–754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
    2. Morgan, H. L. The Generation of a Unique Machine Description for Chemical Structures. *J. Chem. Doc.* **1965**, 5, 107–113. [doi:10.1021/c160017a018](https://doi.org/10.1021/c160017a018)
    3. Riniker, S.; Landrum, G. A. Similarity maps – a visualization strategy for molecular fingerprints and machine-learning methods. *J. Cheminform.* **2013**, 5, 43. [doi:10.1186/1758-2946-5-43](https://doi.org/10.1186/1758-2946-5-43)
    4. OpenADMET. [Announcing the next OpenADMET Blind Challenge: Predicting PXR Induction](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/) (2026-03-17)
    5. OpenADMET. [Predicting PXR Induction - We have liftoff](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/) (2026-04-01)
    6. OpenADMET. [Don't Look Back in Error: What we learned predicting PXR induction (Part I)](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/) (2026-08-27)
    """)
    return


if __name__ == "__main__":
    app.run()
