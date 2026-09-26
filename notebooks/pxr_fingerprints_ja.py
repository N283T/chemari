# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "molwidgets @ git+https://github.com/N283T/openadmet-marimo",
#     "polars>=1.30",
#     "numpy>=2",
#     "altair>=5.5",
#     "scikit-learn>=1.7",
#     "scipy>=1.14",
#     "rdkit>=2025.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="似ているのに、同じじゃない: PXR と Morgan fingerprint")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 似ているのに、同じじゃない
    ### PXR induction で Morgan fingerprint がつまずく理由

    **Pregnane X receptor (PXR)** は、体内に入ってきた異物を感知して CYP3A4 や P-gp などの発現を
    オンにする核内受容体です。PXR を活性化する薬は *別の* 薬の代謝を速めてしまうため、PXR induction は
    代表的な drug–drug interaction (DDI) リスクとして知られています。OpenADMET は 11,000 以上の化合物で
    これを測定し、新規 513 化合物の **pEC50** を予測する blind challenge を開催しました。

    このチャレンジでは、QSAR の定番である **Morgan (ECFP 系) fingerprint** を使ったモデルが、
    一貫して最も弱い入力のひとつでした。私自身の参加 (95 チーム中 4 位) でも、Morgan だけの LightGBM は
    CV MAE 0.57 前後にとどまり、記述子や embedding を組み合わせた ensemble は 0.40 を切りました。
    他のチームからも同じ傾向が報告されています。この notebook のテーマは **なぜか** です。

    Fingerprint モデルの本質は nearest-neighbour です。つまり *部分構造を共有する分子は活性も似ている*
    という前提に立っています。この前提を PXR のデータで直接検証し、どこで崩れるのかを bit 単位で見ていきます。

    /// details | この notebook で得られる直感
    1. Morgan の bit が実際に何を表しているのか (そして何を捨てているのか)
    2. Test set は training set とどのくらい似ているのか
    3. 構造の類似度は PXR 活性をどのくらい予測できるのか — *similarity principle* を実測する
    4. 3 つの盲点: 立体化学、「あるかないか」と「いくつあるか」、分子全体の物性
    5. 自分で仮説を試せる小さな model lab
    ///

    インタラクティブな部品 (分子グリッドと fingerprint explorer) は、この notebook のために作った
    [anywidget](https://anywidget.dev) ベースの自作パッケージ `molwidgets` です。
    """)
    return


@app.cell
def _():
    import altair as alt
    import numpy as np
    import polars as pl
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
    from rdkit.Chem import rdMolDescriptors as rdmd
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor

    alt.data_transformers.disable_max_rows()  # a few charts plot all ~4.6k compounds

    from molwidgets import (
        MolGrid,
        MorganExplorer,
        census_for,
        fingerprint_matrix,
        standardize_smiles,
        tanimoto_matrix,
    )

    return (
        Chem,
        Crippen,
        Descriptors,
        MolGrid,
        MorganExplorer,
        RandomForestRegressor,
        alt,
        census_for,
        fingerprint_matrix,
        np,
        pl,
        rdFingerprintGenerator,
        rdmd,
        spearmanr,
        standardize_smiles,
        tanimoto_matrix,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · データと、ケモインフォマティクス的な下ごしらえ

    Hugging Face で公開されている OpenADMET PXR データ (CC-BY-4.0) の **training set** と、チャレンジ後に
    2 段階で正解が公開された **test set 全体** を使います。SMILES はすべて同じ手順で標準化します:
    最大フラグメントを残す、電荷を中和する、立体化学は保持する、canonical SMILES で書き出す。
    """)
    return


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
        "stereo": int(data["smiles"].str.contains("@").sum()),
    }
    train = data.filter(pl.col("split") == "train")
    test = data.filter(pl.col("split") == "test")
    return data, hygiene, test, train


@app.cell(hide_code=True)
def _(hygiene, mo, test, train):
    mo.hstack(
        [
            mo.stat(f"{train.height:,}", label="training 化合物"),
            mo.stat(f"{test.height:,}", label="test 化合物 (正解公開済み)"),
            mo.stat(str(hygiene["invalid"]), label="パースできない SMILES"),
            mo.stat(str(hygiene["changed"]), label="中和した荷電体 / 塩"),
            mo.stat(str(hygiene["dup_within"]), label="標準化後の重複"),
            mo.stat(str(hygiene["overlap"]), label="train にも含まれる test 化合物"),
        ],
        widths="equal",
        gap=0.5,
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, pl, train):
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
    _chart = (
        alt.Chart(_bins)
        .mark_bar(color="#6b7280")
        .encode(
            x=alt.X("pEC50 bin:N", sort=None),
            y=alt.Y("median 95% CI width:Q", title="median 95% CI width (log units)"),
            tooltip=["pEC50 bin", "n", alt.Tooltip("median 95% CI width:Q", format=".2f")],
        )
        .properties(height=180, width=300)
    )
    mo.hstack(
        [
            _chart,
            mo.md(
                f"""
    **ラベルの信頼度は一様ではありません。** pEC50 は dose–response カーブのフィッティングから得られます。
    弱い化合物はプラトーに届かないので、EC50 は外挿値になります。pEC50 3 未満では 95% CI の中央値が
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["median 95% CI width"].item():.1f} log unit** もあり、
    強い化合物の 0.2〜0.3 とは桁違いです。該当するのは
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item()} 化合物** (training の {_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item() / train.height:.0%})
    で、その正確な値はほぼノイズです。後で activity cliff を見るときに思い出してください。
    """
            ),
        ],
        widths=[1, 1.4],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Morgan fingerprint の解剖

    Morgan fingerprint は分子を *円形の原子環境の集まり (bag)* として表現します。各原子は元素・次数・
    水素数・電荷・環に含まれるかどうか、から作った識別子から出発します (**radius 0**)。反復のたびに
    隣接原子の識別子を取り込むので、radius 1 は原子とその隣、radius 2 はさらにその隣まで見ることになります。
    各環境は 32-bit 整数に hash され、`hash % n_bits` で固定長の bit vector に **fold** されます。

    ここから先で効いてくる帰結は 2 つです:

    * Fingerprint が記録するのは環境が **あるかないか** だけで、どこにあるか、(bit vector なら) 何回あるか、
      分子全体がどんな形か、は記録されません。
    * Fold するので、無関係な環境が **同じ bit** に落ちることがあります。これが *collision* です。

    グリッドから好きな化合物を選び (id で絞り込む、pEC50 でソートする、`c1ccncc1` のような SMARTS を
    入れる、など)、explorer の行をクリックすると、その bit を立てている原子がハイライトされます。
    表には各 bit について、それを持つ training 化合物の数、その bit に落ちる *異なる* 部分構造の数
    (**# envs**)、bit がある場合とない場合の平均 pEC50 の差も出ています。Bit を選ぶと、衝突している
    部分構造が並んで表示されます。
    """)
    return


@app.cell
def _(MolGrid, data, mo, pl):
    grid = mo.ui.anywidget(
        MolGrid(
            data.select("id", "smiles", "split", "pEC50").sort("pEC50", descending=True),
            subset=["split", "pEC50"],
            color_by="pEC50",
            selection_mode="single",
            selection=[data.sort("pEC50", descending=True)["id"][0]],
            page_size=12,
            cell_size=165,
        )
    )
    grid
    return (grid,)


@app.cell
def _(MorganExplorer, data, grid, mo, pl, train):
    _sel = grid.value.get("selection") or [train.sort("pEC50", descending=True)["id"][0]]
    _row = data.filter(pl.col("id") == _sel[0]).row(0, named=True)
    explorer = mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _row["id"],
                    "smiles": _row["smiles"],
                    "label": f"{_row['split']} · pEC50 {_row['pEC50']:.2f}",
                }
            ],
            reference=train["smiles"].to_list(),
            y=train["pEC50"].to_numpy(),
            y_label="pEC50",
        )
    )
    explorer
    return (explorer,)


@app.cell(hide_code=True)
def _(census_for, explorer, mo, np, train):
    _r, _n = explorer.value.get("radius", 2), explorer.value.get("n_bits", 2048)
    _census = census_for(train["smiles"].to_list(), _r, _n)
    _envs = _census.n_envs
    _bit = explorer.value.get("selected_bit", -1)
    _sel = (
        f"選択中の bit **{_bit}** は、training set の中で **{len(explorer.value.get('bit_examples', []))}** 種類の"
        "異なる部分構造によって立てられています (上のギャラリー)。"
        if _bit >= 0
        else "Bit の表の行をクリックしてみてください。選んだ bit が Python 側に戻り、この文章が更新されます。"
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(
                        f"{int(_envs.sum()):,}",
                        label=f"training set に現れる radius ≤ {_r} の環境の種類",
                    ),
                    mo.stat(f"{int((_envs > 0).sum()):,} / {_n:,}", label="使われている bit"),
                    mo.stat(f"{_envs[_envs > 0].mean():.1f}", label="1 bit あたりの環境数"),
                    mo.stat(f"{(_envs > 1).mean():.0%}", label="2 種類以上の環境が同居する bit"),
                ],
                widths="equal",
            ),
            mo.md(
                f"{_n:,} bit しかないので、training set の {int(_envs.sum()):,} 種類の環境は重なり合うしかありません。"
                "Explorer を 4096 bit や radius 1 に切り替えると、この数字がどう動くか見られます。"
                + _sel
            ),
        ]
    ).callout(kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · パズル: test set は「見覚えがある」

    モデルが弱いときによく聞く言い訳は *「test 化合物が applicability domain の外だから」* です。
    確かめてみましょう。各化合物について、Morgan fingerprint (radius 2, 2048 bits) の Tanimoto
    類似度で training set の **nearest neighbour (NN)** を探します。Training 化合物については自分自身を除きます。
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
    nn_train = S_train.argmax(1)
    nn_test = S_test.argmax(1)
    return S_test, S_train, X_test, X_train, nn_test, nn_train, y_test, y_train


@app.cell(hide_code=True)
def _(S_test, S_train, alt, mo, np, pl):
    _df = pl.concat(
        [
            pl.DataFrame({"NN Tanimoto": S_train.max(1), "set": "train → 他の train"}),
            pl.DataFrame({"NN Tanimoto": S_test.max(1), "set": "test → train"}),
        ]
    )
    _chart = (
        alt.Chart(_df)
        .transform_density(
            "NN Tanimoto", groupby=["set"], as_=["NN Tanimoto", "density"], extent=[0, 1]
        )
        .mark_area(opacity=0.55)
        .encode(
            x=alt.X("NN Tanimoto:Q", title="最も近い training 化合物との Tanimoto 類似度"),
            y=alt.Y("density:Q", stack=None),
            color=alt.Color(
                "set:N",
                scale=alt.Scale(
                    range=["#3b82f6", "#f59e0b"], domain=["train → 他の train", "test → train"]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(height=200, width=640)
    )
    mo.vstack(
        [
            _chart,
            mo.hstack(
                [
                    mo.stat(
                        f"{np.median(S_train.max(1)):.2f}",
                        label="NN 類似度の中央値 (train → train)",
                    ),
                    mo.stat(
                        f"{np.median(S_test.max(1)):.2f}",
                        label="NN 類似度の中央値 (test → train)",
                    ),
                    mo.stat(
                        f"{(S_test.max(1) >= 0.5).mean():.0%}",
                        label="NN 類似度 ≥ 0.5 の test 化合物",
                    ),
                ],
                widths="equal",
            ),
            mo.md(
                "Test set は、training 化合物同士よりも **training set に近い** のです。Test 化合物は、"
                "training set に *実際にある* ケミストリーを中心に設計されています。構造の類似度が活性の情報を"
                "運んでいるなら、これは簡単なテストのはずです。"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    k_slider = mo.ui.slider(1, 50, value=1, step=1, label="近傍数 k", show_value=True)
    mo.md(
        f"""
    そこで、いちばん素朴な fingerprint モデルを使ってみます。各 test 化合物の pEC50 を、
    **最も似ている k 個** の training 化合物の平均で予測します。{k_slider}
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
    mo.hstack(
        [
            (_diag + _pts).properties(width=330, height=300),
            mo.vstack(
                [
                    mo.stat(f"{_mae:.2f}", label=f"MAE, {_k}-NN"),
                    mo.stat(f"{_rho:.2f}", label="Spearman ρ"),
                    mo.stat(f"{_rand:.2f}", label="ランダムな training 化合物で当てた場合の MAE"),
                    mo.md(
                        "**k = 1** では、nearest neighbour に順位付けの力はほぼありません (ρ ≈ 0)。"
                        "その pEC50 のずれは、ランダムに選んだ training 化合物と同じくらいです。"
                        "近傍を増やして平均しても、予測が全部平均値に寄っていくだけです。"
                        "スライダーを動かすと、点の雲が縦の帯に潰れていくのが見えます。"
                    ),
                ]
            ),
        ],
        widths=[1.1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, nn_test, pl, y_test, y_train):
    _df = pl.concat(
        [
            pl.DataFrame({"pEC50": y_train, "set": "training 化合物全体"}),
            pl.DataFrame(
                {
                    "pEC50": y_train[nn_test],
                    "set": "各 test 化合物の NN (training)",
                }
            ),
            pl.DataFrame({"pEC50": y_test, "set": "test 化合物 (正解)"}),
        ]
    )
    _chart = (
        alt.Chart(_df)
        .transform_density("pEC50", groupby=["set"], as_=["pEC50", "density"], extent=[1.5, 8])
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("pEC50:Q"),
            y=alt.Y("density:Q"),
            color=alt.Color(
                "set:N",
                scale=alt.Scale(
                    domain=[
                        "training 化合物全体",
                        "各 test 化合物の NN (training)",
                        "test 化合物 (正解)",
                    ],
                    range=["#9ca3af", "#3b82f6", "#f59e0b"],
                ),
                legend=alt.Legend(orient="top", title=None, direction="vertical"),
            ),
        )
        .properties(height=220, width=380)
    )
    _nn = y_train[nn_test]
    mo.vstack(
        [
            mo.md("### Neighbour は誰なのか?"),
            mo.hstack(
                [
                    _chart,
                    mo.md(
                        f"""
    上の散布図の縦の帯は偶然ではありません。Test 化合物の nearest neighbour は、圧倒的に **強活性** の
    training 化合物です。その平均 pEC50 は **{_nn.mean():.2f}** で、training 全体の
    **{y_train.mean():.2f}** を大きく上回り、**{(_nn >= 5.5).mean():.0%}** が pEC50 ≥ 5.5 です。

    ところが test 化合物自身はそうなっていません。Test 化合物は最も活性の高い chemotype の周りで作られた
    analogue で、活性は ~2 から ~7 まで広がっています (平均 **{y_test.mean():.2f}**)。つまり test set は
    **ヒット化合物周りの SAR 探索** です。どの化合物にも近くに強活性の「親戚」がいて、問われているのは
    「どの小さな変化なら活性が残るのか」。これこそ、類似度ベースのモデルが答えられない問いです。
    """
                    ),
                ],
                widths=[1, 1.1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Similarity principle を実測する

    *Similar property principle* によれば、2 つの分子の構造類似度が上がるほど、活性の差は小さくなるはずです。
    Training set の全ペア (約 850 万) で直接測ってみます。ペアを Tanimoto 類似度で bin に分け、
    各 bin の典型的な |Δ pEC50| を見ます。
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
def _(alt, mo, pl, random_pair_dy, similarity_curve):
    _df = pl.DataFrame(similarity_curve)
    _base = alt.Chart(_df).encode(
        x=alt.X("sim_mid:Q", title="ペアの Tanimoto 類似度", scale=alt.Scale(domain=[0, 1]))
    )
    _band = (
        _base.transform_calculate(zero="0")
        .mark_area(opacity=0.18, color="#d6336c")
        .encode(y=alt.Y("q90_dy:Q"), y2="zero:Q")
    )
    _line = _base.mark_line(point=True, color="#d6336c").encode(
        y=alt.Y("mean_dy:Q", title="|Δ pEC50| (平均、帯は 90 パーセンタイルまで)"),
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
    _hi = _df.filter(pl.col("sim_lo") >= 0.5)
    mo.hstack(
        [
            (_band + _line + _rule).properties(height=260, width=360),
            mo.md(
                f"""
    点線はランダムなペアの値 (**{random_pair_dy:.2f}** log unit) です。カーブは確かに下がっており、
    類似度が無意味なわけではありません。ただ、その下がり方がゆるやかです。Tanimoto ≥ 0.5 のペアでも
    典型的な差は **{(_hi["mean_dy"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.2f}** log unit あり、
    **{(_hi["frac_gt1"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.0%}** のペアは活性が 10 倍以上違います。

    点にカーソルを乗せるとペア数が見られます。本当に近いペアはごくわずかで、データの大半は
    fingerprint が「そこそこ似ている」と言い、PXR が「何でもありうる」と言う領域にあります。
    """
            ),
        ],
        widths=[1.1, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    min_sim = mo.ui.slider(
        0.4, 0.9, value=0.55, step=0.05, label="Tanimoto の下限", show_value=True
    )
    min_dy = mo.ui.slider(0.5, 3.0, value=1.5, step=0.25, label="|Δ pEC50| の下限", show_value=True)
    mo.md(
        f"""
    ### Activity cliff を眺める

    **Activity cliff** とは、見た目は似ているのに活性が大きく違うペアのことです。「似ている」と「違う」の
    基準を決め、表からペアを選んで、2 つの fingerprint を bit 単位で比べてみてください。片方にしかない bit こそが、
    fingerprint モデルが差を説明するために *使える* 唯一の手がかりです。そしてそれは多くの場合、
    ありふれた環境がいくつかあるだけです。

    {mo.hstack([min_sim, min_dy], justify="start", gap=2)}
    """
    )
    return min_dy, min_sim


@app.cell
def _(Crippen, Chem, min_dy, min_sim, np, pair_dy, pair_i, pair_j, pair_sim, pl, train):
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
    )
    train_logp = _logp
    return cliffs, train_logp


@app.cell
def _(cliffs, mo):
    cliff_table = mo.ui.table(
        cliffs,
        selection="single",
        initial_selection=[0] if cliffs.height else None,
        page_size=6,
        label=f"{cliffs.height:,} ペア · SALI = |Δ pEC50| / (1 − similarity)",
    )
    cliff_table
    return (cliff_table,)


@app.cell
def _(MorganExplorer, cliff_table, mo, pl, train):
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
    眺めるときに注目したい点は 2 つです:

    * **弱い側の値は本当に測れているか?** この {_n} ペアのうち {_noisy} ペアでは、弱い側の化合物の CI が
      1.5 log unit を超えています。「cliff」の一部は、スケールの底にあるアッセイノイズです。
    * **どちらが脂溶性が高いか?** {_n} ペア中 {_lip} ペア ({_lip / max(_n, 1):.0%}) で、強い側の化合物のほうが
      計算 logP も高くなっています。*分子全体* の性質が、どの単独の bit でも表せない役割を果たしています。
      これが次のセクションのテーマです。
    """
    ).callout(kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Bit には見えないもの

    ### 5a · 違う分子なのに、fingerprint が完全に一致する

    2 つの異なる分子がまったく同じ bit vector を持つなら、その bit で作ったモデルは両者に **必ず**
    同じ値を予測します。Training set からそういうグループをすべて探し、fingerprint が区別できない
    *理由* を見てみます。
    """)
    return


@app.cell
def _(Chem, MolGrid, X_train, mo, np, pl, rdFingerprintGenerator, train):
    from collections import defaultdict

    def why_identical(smiles: list[str]) -> str:
        """Classify a group of molecules that share a Morgan bit vector."""
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        flat = {Chem.MolToSmiles(m, isomericSmiles=False) for m in mols}
        if len(flat) > 1:
            # Different constitution, same set of radius-2 environments: only repeat counts differ.
            return "ring size / chain length"
        # Count specified stereo elements (tetrahedral centres and double-bond geometry).
        n_specified = {
            sum(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms())
            + sum(b.GetStereo() != Chem.BondStereo.STEREONONE for b in m.GetBonds())
            for m in mols
        }
        return "stereo specified vs unspecified" if len(n_specified) > 1 else "stereoisomers"

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
    twin_grid = mo.ui.anywidget(
        MolGrid(
            fp_twins,
            subset=["group", "why", "pEC50", "ci_width"],
            color_by="pEC50",
            page_size=14,
            cell_size=150,
        )
    )
    _n_stereo = _summary.filter(pl.col("why") != "ring size / chain length").height
    _chiral_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    _n_split = sum(
        len(
            {_chiral_gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(_smiles[k])).tobytes() for k in g}
        )
        > 1
        for g in _twins
        if why_identical([_smiles[k] for k in g]) != "ring size / chain length"
    )
    mo.vstack(
        [
            mo.md(
                f"""
    **{len(_twins)} グループ** ({fp_twins.height} 化合物) が同じ fingerprint に潰れており、グループ内の活性差は
    最大 **{_summary["range"].max():.2f}** log unit です。原因は 2 種類あります:

    * **立体化学** ({_n_stereo} グループ)。RDKit の Morgan fingerprint は、`includeChirality=True` を指定しない限り
      キラリティも二重結合の幾何異性も無視します。よく見ると、これらのグループでは片方の記録だけ立体中心や
      二重結合が *指定されて* います。単一エナンチオマーと、おそらくラセミ体の組 (lansoprazole / dexlansoprazole、
      bupivacaine / levobupivacaine) や、同じ薬が E/Z 指定あり・なしで 2 回登録されたもの (PXR agonist の
      代表格である rifampicin) です。Lansoprazole の組は ~1 log unit 違い、化学者にとっては本物の問いですが、
      fingerprint には見えません。Chirality をオンにすると {_n_stereo} グループ中 {_n_split} グループが分離しますが、
      sulfoxide の立体中心はどちらにしても拾われません。
    * **環サイズ / 鎖長** ({len(_twins) - _n_stereo} グループ)。Cyclohexyl と cycloheptyl のアミン、azepane と azocane、
      nonanoic acid と palmitic acid。どの原子から見ても radius 2 以内の環境は同じなので、環境の *集合* は一致し、
      違うのは各環境が *何回* 出てくるかだけです。Bit vector は有無しか記録しないので区別できません。
      Count fingerprint なら区別できます (5c 参照)。
    """
            ),
            twin_grid,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b · 分子全体の物性

    PXR の ligand-binding pocket は大きく、柔軟で、疎水的です。まったく違う scaffold を受け入れることで
    知られています。結合を決めているのが特定の pharmacophore ではなく *どのくらい脂溶性で、どのくらい大きいか*
    なのだとしたら、局所的な部分構造の集まりという表現は、そもそも言語として合っていません。
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
def _(DESCRIPTORS, D_train, mo):
    prop_pick = mo.ui.dropdown(list(DESCRIPTORS), value="MolLogP", label="記述子")
    prop_pick
    return (prop_pick,)


@app.cell(hide_code=True)
def _(DESCRIPTORS, D_train, alt, mo, np, pl, prop_pick, spearmanr, y_train):
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
def _(D_train, mo, pair_dy, pair_i, pair_j, pair_sim, spearmanr):
    _m = pair_sim >= 0.5
    _dlogp = abs(D_train[pair_i[_m], 0] - D_train[pair_j[_m], 0])
    _rho = spearmanr(_dlogp, pair_dy[_m])[0]
    mo.md(
        f"""
    Crippen logP というたった 1 つの数字が、fingerprint kNN が test set を順位付けしたのと同じくらいの精度で
    training 化合物を順位付けします。さらに似ているペア (Tanimoto ≥ 0.5) の *中でも*、logP の差が活性の差と
    連動しています (Spearman ρ = **{_rho:.2f}**)。Fingerprint から見ると cliff に見えるものの一部は、
    なめらかな脂溶性のトレンドなのです。Fold された bit vector には「少しだけ脂溶性が高い」という軸がありません。
    メチル基を 1 つ足しても、bit が 1 つ立つか立たないかでしかありません。
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5c · 「あるかないか」と「いくつあるか」

    Binary の bit が伝えるのは環境があるかどうかだけです。**Count** fingerprint は何回出てくるかを記録し、
    分子全体で合計すれば、サイズや脂溶性のおおまかな指標になります。上の仮説が正しければ、count で失われた
    情報の一部が戻ってくるはずです。下の lab で試してみてください。

    ## 6 · Model lab

    Training set で random forest を学習し、正解公開済みの test 513 化合物でスコアを出します。
    デフォルトの 3 設定は計算済みです。設定を変えて **Train** を押すと、スコアボードに自分の行が追加されます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lab_features = mo.ui.multiselect(
        ["Morgan bits", "Morgan counts", "14 descriptors"],
        value=["Morgan counts"],
        label="特徴量",
    )
    lab_radius = mo.ui.dropdown({"1": 1, "2": 2, "3": 3}, value="2", label="radius")
    lab_bits = mo.ui.dropdown(
        {"256": 256, "1024": 1024, "2048": 2048, "8192": 8192}, value="2048", label="bits"
    )
    lab_chiral = mo.ui.checkbox(label="chirality を含める")
    lab_run = mo.ui.run_button(label="Train")
    mo.hstack(
        [lab_features, lab_radius, lab_bits, lab_chiral, lab_run], justify="start", gap=1, wrap=True
    )
    return lab_bits, lab_chiral, lab_features, lab_radius, lab_run


@app.cell
def _(
    Chem,
    D_test,
    D_train,
    RandomForestRegressor,
    S_test,
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
        if "Morgan bits" in parts:
            blocks.append(np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.float32))
        if "Morgan counts" in parts:
            blocks.append(
                np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32)
            )
        return blocks

    def evaluate(parts, radius=2, n_bits=2048, chiral=False, seed=0):
        """Fit a random forest on train, return test metrics and predictions."""
        tr = featurize(train["smiles"].to_list(), parts, radius, n_bits, chiral)
        te = featurize(test["smiles"].to_list(), parts, radius, n_bits, chiral)
        if "14 descriptors" in parts:
            tr.append(D_train)
            te.append(D_test)
        model = RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, max_features=0.3, n_jobs=-1, random_state=seed
        ).fit(np.hstack(tr), y_train)
        pred = model.predict(np.hstack(te))
        fp_desc = " + ".join(parts)
        if any(p.startswith("Morgan") for p in parts):
            fp_desc += f" (r{radius}, {n_bits}{', chiral' if chiral else ''})"
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
    with mo.status.spinner("基準モデル 3 つを学習中…"):
        baseline_runs = [
            evaluate(["Morgan bits"]),
            evaluate(["14 descriptors"]),
            evaluate(["Morgan bits", "14 descriptors"]),
        ]
    return (baseline_runs,)


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
def _(baseline_runs, get_runs, mo, pl, y_test):
    all_runs = baseline_runs + get_runs()
    _board = pl.DataFrame(
        [
            {k: v for k, v in r.items() if not k.startswith("_")}
            | {"source": "reference" if i < len(baseline_runs) else "yours"}
            for i, r in enumerate(all_runs)
        ]
    )
    run_pick = mo.ui.table(
        _board,
        selection="single",
        initial_selection=[len(all_runs) - 1],
        label=f"スコアボード (test の正解 pEC50 の SD = {y_test.std():.2f}、行を選ぶと下に詳細が出ます)",
        page_size=8,
    )
    run_pick
    return all_runs, run_pick


@app.cell(hide_code=True)
def _(all_runs, alt, mo, nn_sim_test, np, pl, run_pick, test, y_test):
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
        .properties(width=220, height=260, title="正解の活性帯ごとの誤差")
    )
    _zero = alt.Chart(pl.DataFrame({"y": [0]})).mark_rule(color="#9ca3af").encode(y="y:Q")
    mo.vstack(
        [
            mo.hstack([_scatter, _bias + _zero], justify="start", gap=2),
            mo.md(
                f"予測の SD は **{_pred.std():.2f}** で、正解の **{y_test.std():.2f}** より小さくなっています。"
                "どのモデルも予測が中央に寄りますが、問題はその程度です。弱い化合物は強めに、強い化合物は弱めに"
                "予測され、fingerprint だけのモデルが最も強く縮みます。"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lab で試してみる価値があること:

    * **Bits → counts** (radius と長さはそのまま)。Count にするだけで「いくつあるか」の情報がタダで加わります。
    * **256 → 8192 bits.** Collision が減ると少し良くなります。Collision は実在しますが (explorer 参照)、
      主役ではありません。
    * **Chirality を含める。** 変わる化合物はわずかなのでスコアはほとんど動きません。ただし 5a の立体異性体の
      組にとっては、「原理的に不可能」が「可能」に変わる違いです。
    * **Morgan + 14 記述子.** 分子全体を表すシンプルな 14 個の数字が、2048 bit の取りこぼしの多くを補います。

    ## 7 · まとめ

    1. **PXR の test set は domain の外ではない。** Test 化合物は平均すると、training 化合物同士よりも
       training set に *近い*。
    2. **ここでは類似度が活性を運ばない。** 最も近い training 化合物の pEC50 は、ランダムな training 化合物と
       同じくらいの情報しか持たない。類似度–活性のカーブは下がるが、ゆるやか。
    3. **シグナルの多くは分子全体にある。** 脂溶性とサイズは、局所環境の集まりでは表せない分散の一部を説明し、
       cliff に見えるものの一部も説明する。
    4. **「cliff」の一部はアッセイの底。** 弱い化合物の信頼区間はとても広い。スケールの底での差は疑ってかかる。
    5. **Fingerprint はそれでも役に立つ — 材料のひとつとして。** Count、大きな fold、chirality、少数の記述子が
       それぞれ差の一部を取り戻す。チャレンジ上位のモデルは、学習済み embedding や構造ベースの特徴量でさらに先へ進んだ。

    ---

    ### この notebook について

    * **データ:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0)。Training set と、phase 1・phase 2 で公開された test の正解ラベル。
    * **ウィジェット:** `MolGrid` と `MorganExplorer` は、この notebook のために作った anywidget コンポーネントです
      ([ソース](https://github.com/N283T/openadmet-marimo))。分子の描画はブラウザ上の RDKit.js、fingerprint・
      bit 環境・collision の計算は Python の RDKit で行っています。
    * **関連:** Pat Walters の [marimo-chem-utils](https://github.com/PatWalters/marimo_chem_utils) と
      [practical cheminformatics tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials)、
      グリッドの着想元である [mols2grid](https://github.com/cbouy/mols2grid)。
    * **AI の利用:** ウィジェットと notebook の骨組みは、コーディングアシスタントとして Claude (Anthropic) と
      一緒に作りました。問いの設定、分析の選択、解釈は私自身の PXR チャレンジでの経験に基づいています。
      表示している数値はすべて、この notebook の中でその場で計算しています。
    """)
    return


if __name__ == "__main__":
    app.run()
