# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "molwidgets @ git+https://github.com/N283T/openadmet-marimo",
#     "polars>=1.30",
#     "numpy>=2",
#     "altair>=5.5",
#     "duckdb>=1.1",
#     "pyarrow>=18",
#     "sqlglot>=26",
#     "lightgbm>=4.5",
#     "scikit-learn>=1.7",  # required by lightgbm's sklearn API
#     "scipy>=1.14",
#     "rdkit>=2025.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="ECFP4 を中身から見る")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ECFP4 を中身から見る
    ### OpenADMET の 3 つのデータセットで

    分子を機械学習にかけるとき、とりあえずこう書いていないでしょうか。

    ```python
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    ```

    これが **ECFP4** です。計算が速く、調整するパラメータもほとんどなく、多くのデータで最初に試す表現です。ただ、同じ ECFP4 でも、よく効くデータとそうでないデータがあります。どこで効いて、どこで効かないのか、その理由は何なのかは、中身を知らないとわかりません。

    この notebook では、ECFP4 の中身を追ったあと (第 1 部)、OpenADMET が公開している 3 つのデータセット・16 のエンドポイントで同じ見方を繰り返します (第 2 部)。データセットを切り替えると、第 2 部のすべてのセルが切り替わります。最後に、データセットをまたいで見えたことをまとめます (第 3 部)。
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import duckdb
    import numpy as np
    import polars as pl
    from scipy.stats import spearmanr

    from molwidgets import (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MorganBitTiles,
        MorganExplorer,
        bench,
        census_for,
    )

    _ = alt.data_transformers.disable_max_rows()
    return (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MorganBitTiles,
        MorganExplorer,
        Path,
        alt,
        bench,
        census_for,
        duckdb,
        np,
        pl,
        spearmanr,
    )


@app.cell
def _(Path, bench, mo, pl):
    # precomputed tables: local copy, else GitHub; whatever is missing is computed here
    _where = bench.open_tables(Path("results/precomputed"))
    _missing = [t for t in bench.TABLES if t not in _where]
    if _missing:
        with mo.status.spinner(f"事前計算がないので計算中 ({', '.join(_missing)})… 数分かかります"):
            _tables = {t: [] for t in bench.TABLES}
            _raw = {}
            for _task in bench.TASKS:
                _raw.setdefault(_task.dataset, bench.load_raw(_task.dataset))
                _mol = bench.load_task(_task, raw=_raw[_task.dataset])
                _tables["molecules"].append(_mol)
                for _k, _v in bench.compute_task(_mol).items():
                    _tables[_k].append(_v)
            _computed = {k: pl.concat(v, how="diagonal_relaxed") for k, v in _tables.items()}
    else:
        _computed = {}
    molecules, neighbours, predictions, metrics, shap = (
        _computed[t] if t in _computed else pl.read_parquet(_where[t]) for t in bench.TABLES
    )
    tasks = pl.DataFrame(
        [
            {
                "task": t.key,
                "dataset": t.dataset,
                "endpoint": t.endpoint,
                "label": t.label,
                "what": t.note,
            }
            for t in bench.TASKS
        ]
    )
    _sources = "\n".join(
        f"* `{k}`: {'この場で計算' if k in _computed else _where[k]}" for k in bench.TABLES
    )
    mo.accordion(
        {
            "データの読み込みについて": mo.md(
                "計算に時間のかかる部分 (記述子、モデルの学習、TreeSHAP) は事前に計算して parquet "
                "ファイルにしてあり、リポジトリの `results/precomputed/` から読み込みます。"
                "ファイルが見つからないときは、同じコード (`molwidgets.bench`) でこの場で計算します。"
                "読み込んだ表は DuckDB で SQL を使って集計します。\n\n読み込み元:\n\n" + _sources
            )
        }
    )
    return metrics, molecules, neighbours, predictions, shap, tasks


@app.cell
def _(duckdb, metrics, neighbours, tasks):
    # the notebook's own DuckDB connection for its SQL cells (the default connection is also
    # used by marimo's server to parse SQL; sharing it can deadlock the two)
    db = duckdb.connect()
    db.register("metrics", metrics)
    db.register("neighbours", neighbours)
    _ = db.register("tasks", tasks)
    return (db,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 第 1 部 · ECFP4 の中身

    まずは 80 秒の動画で、ECFP4 が分子から bit を作る手順を見てください。
    """)
    return


@app.cell(hide_code=True)
def _(ECFPMovie, mo):
    mo.ui.anywidget(ECFPMovie())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    同じ手順を好きな化合物で追えます。SMILES を入力するか例を選び、1 ステップずつ進めるか自動再生してください。
    """)
    return


@app.cell(hide_code=True)
def _(ECFPStepper, mo):
    mo.ui.anywidget(ECFPStepper())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 実際に ECFP を見てみる

    下のウィジェットは、実際に生成される ECFP を可視化したものです。右の bit を選ぶと該当する部分構造が分子の上でハイライトされるほか、collision (別の部分構造が同じ bit に入ること) があるかどうかも確認できます。好きな分子に変えたり、radius や folding (bit 数) を変えたりして、いろいろ試してみてください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    tiles_smiles = mo.ui.text("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", label="SMILES", full_width=True)
    tiles_smiles
    return (tiles_smiles,)


@app.cell(hide_code=True)
def _(MorganBitTiles, mo, tiles_smiles):
    mo.ui.anywidget(MorganBitTiles(tiles_smiles.value, label=""))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## ECFP4 の注意点

    動画で見た collision のほかにも、ECFP4 には注意するポイントがあります。ここでは 2 分子の fingerprint を比較するウィジェットを使って、それらを見ていきます。

    ### 違う分子なのに fingerprint が一致する

    類似検索で、違う分子なのに類似度 1.0 で複数ヒットすることがたまにあります。これは次のようなときに起こります。

    * 立体化学が違う
    * 環のサイズや鎖の長さが違う
    """)
    return


@app.cell
def _():
    # names checked against PubChem (structure, and stereo for the enantiomers)
    SAME_FP = [
        {
            "label": "cyclohexylamine / cycloheptylamine (環サイズ)",
            "a": "NC1CCCCC1",
            "b": "NC1CCCCCC1",
            "name_a": "cyclohexylamine",
            "name_b": "cycloheptylamine",
        },
        {
            "label": "azepane / azocane (環サイズ)",
            "a": "C1CCCNCC1",
            "b": "C1CCCNCCC1",
            "name_a": "azepane",
            "name_b": "azocane",
        },
        {
            "label": "nonanoic acid / palmitic acid (鎖長)",
            "a": "CCCCCCCCC(=O)O",
            "b": "CCCCCCCCCCCCCCCC(=O)O",
            "name_a": "nonanoic acid",
            "name_b": "palmitic acid",
        },
        {
            "label": "(R)- / (S)-thalidomide (立体)",
            "a": "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "b": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "name_a": "(R)-thalidomide",
            "name_b": "(S)-thalidomide",
        },
        {
            "label": "lansoprazole / dexlansoprazole (立体)",
            "a": "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1",
            "b": "Cc1c(OCC(F)(F)F)ccnc1C[S@@](=O)c1nc2ccccc2[nH]1",
            "name_a": "lansoprazole",
            "name_b": "dexlansoprazole",
        },
    ]
    CLOSE_BUT_FAR = [
        {
            "label": "N-methylacetamide / N-ethylacetamide (動画の 2 分子、CH₃ → C₂H₅)",
            "a": "CC(=O)NC",
            "b": "CC(=O)NCC",
            "name_a": "N-methylacetamide",
            "name_b": "N-ethylacetamide",
        },
        {
            "label": "phenethylamine / 4-pyridylethylamine (CH → N)",
            "a": "NCCc1ccccc1",
            "b": "NCCc1ccncc1",
            "name_a": "phenethylamine",
            "name_b": "4-pyridylethylamine",
        },
        {
            "label": "diazepam / nordazepam (N-CH₃ → N-H)",
            "a": "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
            "b": "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1",
            "name_a": "diazepam",
            "name_b": "nordazepam",
        },
        {
            "label": "paracetamol / phenacetin (OH → OEt)",
            "a": "CC(=O)Nc1ccc(O)cc1",
            "b": "CCOc1ccc(NC(C)=O)cc1",
            "name_a": "paracetamol",
            "name_b": "phenacetin",
        },
        {
            "label": "OADMET-0001944 / OADMET-0002007 (PXR, CH₃ 1 つ)",
            "a": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
            "b": "CCN(CC)CC(=O)Nc1c(C)cc(C)cc1C",
            "name_a": "OADMET-0001944",
            "name_b": "OADMET-0002007",
        },
    ]
    return CLOSE_BUT_FAR, SAME_FP


@app.cell(hide_code=True)
def _(SAME_FP, mo):
    # one example per cause, for the MolPair right below
    _by_label = {e["label"]: e for e in SAME_FP}
    same_kind = mo.ui.dropdown(
        {
            "立体化学": _by_label["(R)- / (S)-thalidomide (立体)"],
            "環サイズ": _by_label["cyclohexylamine / cycloheptylamine (環サイズ)"],
            "鎖長": _by_label["nonanoic acid / palmitic acid (鎖長)"],
        },
        value="立体化学",
        label="例",
    )
    same_kind
    return (same_kind,)


@app.cell(hide_code=True)
def _(MolPair, mo, same_kind):
    _e = same_kind.value
    mo.ui.anywidget(
        MolPair(
            {"id": _e["name_a"], "smiles": _e["a"]},
            {"id": _e["name_b"], "smiles": _e["b"]},
            show_smiles=False,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    特に後者は分子の組成そのものが違うのに、同じものとして扱われます。類似検索なら目で見て気づけますが、機械学習では違う分子に対してモデルがまったく同じ予測を返すので、リスクが大きくなります。

    理由は単純で、デフォルトの ECFP は部分構造があるかないかしか表現しないためです。

    * 立体化学は考慮されない → `includeChirality=True` で区別できる
    * 同じ部分構造が何回出てきても 1 になる (環や鎖が長くなっても増えない) → 回数も記録する count fingerprint で区別できる

    下の表からペアを選び、ウィジェットで count に切り替えたり chirality を ON/OFF したりして、左下の類似度や右の bit のリストがどう変わるか確認してみてください。
    """)
    return


@app.cell(hide_code=True)
def _(SAME_FP, bench, mo, np, pl):
    def _separates(e):
        # which setting tells the pair apart (checked here with RDKit)
        out = [
            name
            for name, kw in [("count", {"count": True}), ("chirality", {"chirality": True})]
            if not np.array_equal(*bench.fingerprints([e["a"], e["b"]], **kw))
        ]
        return " / ".join(out) or "どちらでも区別できない"

    same_table = mo.ui.table(
        pl.DataFrame(
            [
                {
                    "A": e["name_a"],
                    "B": e["name_b"],
                    "違い": e["label"].rsplit("(", 1)[1].rstrip(")"),
                    "区別できる設定": _separates(e),
                }
                for e in SAME_FP
            ]
        ),
        selection="single",
        initial_selection=[0],
        label="同じ fingerprint になるペア",
    )
    same_table
    return (same_table,)


@app.cell(hide_code=True)
def _(MorganExplorer, SAME_FP, mo, same_table):
    _sel = same_table.value
    _name = _sel["A"][0] if _sel is not None and len(_sel) else SAME_FP[0]["name_a"]
    _e = next(e for e in SAME_FP if e["name_a"] == _name)
    mo.ui.anywidget(
        MorganExplorer(
            [{"id": _e["name_a"], "smiles": _e["a"]}, {"id": _e["name_b"], "smiles": _e["b"]}],
            stereo_labels=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ただし、どちらも万能ではありません。

    * chirality を入れても区別できない立体がある (lansoprazole と dexlansoprazole の違いはスルホキシドの硫黄の立体で、ECFP はこれを拾わない)
    * count では、collision が起きている bit で別の部分構造の回数もまとめて数えられるので、collision のデメリットが通常より大きくなることもある

    collision は bit 数を増やせば減らせますが、bit 長が長くなるわりに得られるものが少ないこともあります。データセットに合わせて、使う設定を見極めましょう。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 似ているのに類似度が思ったより低い

    一方で、見た目はよく似ているのに、Tanimoto が 0.4〜0.7 くらいと直感より低くなるペアもあります。ECFP4 は各原子のまわり半径 2 までを見るので、原子が 1 つ変わるだけでそのまわりの部分構造がまとめて変わり、10 個以上の bit が入れ替わるためです。

    前の節の鎖長や環サイズの例では、炭素が増えても fingerprint は変わりませんでした。違いは変わった場所にあります。長い鎖や大きい環の途中に同じ単位を足しても、半径 2 の中に見える部分構造はすでにあるものと同じなので、新しい bit は立ちません。一方、原子を置き換えたり、置換基や鎖の端を変えたりすると、そこから半径 2 以内の部分構造がすべて新しくなります。
    """)
    return


@app.cell(hide_code=True)
def _(CLOSE_BUT_FAR, mo):
    far_kind = mo.ui.dropdown(
        {e["label"]: e for e in CLOSE_BUT_FAR}, value=CLOSE_BUT_FAR[0]["label"], label="例"
    )
    far_kind
    return (far_kind,)


@app.cell(hide_code=True)
def _(MolPair, far_kind, mo):
    _e = far_kind.value
    mo.ui.anywidget(
        MolPair(
            {"id": _e["name_a"], "smiles": _e["a"]},
            {"id": _e["name_b"], "smiles": _e["b"]},
            show_smiles=False,
            show_common=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    先ほどと同じように、ウィジェットで fingerprint を比べてみましょう。表からペアを選ぶと、右側に片方にしかない bit (differ) が並び、変わった原子のまわりの部分構造がまとめて入れ替わっているのがわかります。
    """)
    return


@app.cell(hide_code=True)
def _(CLOSE_BUT_FAR, SAME_FP, bench, mo, pl):
    def _bits(e):
        a, b = bench.fingerprints([e["a"], e["b"]]).astype(bool)
        return round(float((a & b).sum() / (a | b).sum()), 2), int((a ^ b).sum())

    # the chain-length pair from the previous section, for comparison (no bit changes)
    far_pairs = CLOSE_BUT_FAR + [e for e in SAME_FP if "鎖長" in e["label"]]

    far_table = mo.ui.table(
        pl.DataFrame(
            [
                {
                    "A": e["name_a"],
                    "B": e["name_b"],
                    "違い": e["label"].rsplit("(", 1)[1].rstrip(")"),
                    "Tanimoto": t,
                    "入れ替わる bit": n,
                }
                for e in far_pairs
                for t, n in [_bits(e)]
            ]
        ),
        selection="single",
        initial_selection=[0],
        label="見た目は近いのに Tanimoto が低いペア",
    )
    far_table
    return far_pairs, far_table


@app.cell(hide_code=True)
def _(MorganExplorer, far_pairs, far_table, mo):
    _sel = far_table.value
    _name = _sel["A"][0] if _sel is not None and len(_sel) else far_pairs[0]["name_a"]
    _e = next(e for e in far_pairs if e["name_a"] == _name)
    mo.ui.anywidget(
        MorganExplorer(
            [{"id": _e["name_a"], "smiles": _e["a"]}, {"id": _e["name_b"], "smiles": _e["b"]}],
            row_filter="differ",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この問題は ECFP の設定を変えるだけでは避けにくいです。次のような方法があります。

    * 別の fingerprint を試す
    * スキャフォールドや MCES のような部分グラフに基づく比較を使う
    * 記述子を加える

    下のウィジェットの similarity タブでは、上の表で選んだペアをいくつかの fingerprint、MCES、物性で比べられます。係数は Tanimoto から Dice や cosine に切り替えられます。手法によって値の出方が違う (MACCS は無関係なペアでも 0.5 前後になる) ので、手法どうしで数値を比べるのではなく、同じ手法でペアを変えたときの値の動きを見てください。
    """)
    return


@app.cell(hide_code=True)
def _(MolPair, far_pairs, far_table, mo):
    _sel = far_table.value
    _name = _sel["A"][0] if _sel is not None and len(_sel) else far_pairs[0]["name_a"]
    _e = next(e for e in far_pairs if e["name_a"] == _name)
    mo.ui.anywidget(
        MolPair(
            {"id": _e["name_a"], "smiles": _e["a"]},
            {"id": _e["name_b"], "smiles": _e["b"]},
            show_smiles=False,
            view="similarity",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ただし、どれも決定的な解決策ではなく、どの程度を「似ている」とするかは結局は感覚に近いものです。スクリーニング、クラスタリング、交差検証の分割などでは類似度のしきい値で機械的に区切ることが多いので、いくつか試しながら、目的に合う方法と基準を選んでください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 第 2 部 · データセットで見る
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 つのデータセット

    どれも実際の創薬プロジェクトで測られた値で、チャレンジで使われた train / test の分け方のまま使います。test と同じ構造が train にある化合物は test から外し、同じ split の中で重複する構造は平均しています。比の尺度で測る値 (溶解度、クリアランス、透過性、非結合率など) は log10(x + 1) に変換しています。
    """)
    return


@app.cell
def _(db, mo):
    _overview = mo.sql(
        """
        SELECT t.dataset, t.endpoint, t.what AS measured, m.n_train, m.n_test
        FROM tasks t JOIN metrics m USING (task)
        ORDER BY t.dataset DESC, m.n_train DESC
        """,
        engine=db,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **PXR**: PXR (薬物代謝酵素の発現を制御する核内受容体) の活性化 ([blog](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/))
    * **ASAP**: 抗ウイルス薬の探索 (ASAP Discovery)。MERS-CoV / SARS-CoV-2 のメインプロテアーゼ阻害と ADMET ([blog](https://polarishub.io/blog/antiviral-competition))
    * **ExpansionRx**: RNA を標的にした創薬プログラム (Expansion Therapeutics) の ADMET ([blog](https://openadmet.ghost.io/expansionrx-openadmet-blind-challenge/))
    """)
    return


@app.cell(hide_code=True)
def _(mo, tasks):
    _options = {f"{r['dataset']} · {r['endpoint']}": r["task"] for r in tasks.iter_rows(named=True)}
    task_pick = mo.ui.dropdown(_options, value="PXR · pEC50", label="データセット · エンドポイント")
    mo.vstack(
        [
            mo.md(r"""
    下から好きなデータセットとエンドポイントを選んでください。選んだデータの中で、同じ手順で ECFP を見ていきます。データセットの中身にはあえて詳しく触れません。いろいろなデータセットに切り替えて、違いを楽しんでみてください。
    """),
            task_pick,
        ]
    )
    return (task_pick,)


@app.cell
def _(bench, molecules, pl, task_pick):
    task = bench.TASK[task_pick.value]
    mols = molecules.filter(pl.col("task") == task.key)
    train = mols.filter(pl.col("split") == "train")
    test = mols.filter(pl.col("split") == "test")
    return mols, task, test, train


@app.cell(hide_code=True)
def _(alt, mo, mols, task, test, train):
    _hist = (
        alt.Chart(mols.select("y", "split"))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("y:Q", bin=alt.Bin(maxbins=40), title=task.label),
            y=alt.Y("count():Q", stack=None, title="化合物数"),
            color=alt.Color(
                "split:N",
                scale=alt.Scale(domain=["train", "test"], range=["#1c7ed6", "#f08c00"]),
                title=None,
            ),
        )
        .properties(height=200, width=420)
    )
    mo.vstack(
        [
            mo.md(f"""
    ### 2.1 · データセットの中身

    **{task.dataset} · {task.endpoint}**: {task.note}。train {train.height:,} 化合物、test {test.height:,} 化合物です。
    """),
            mo.hstack(
                [
                    _hist,
                    mo.vstack(
                        [
                            mo.stat(
                                f"{train['y'].std():.2f}", label=f"train の標準偏差 ({task.label})"
                            ),
                            mo.stat(f"{test['y'].std():.2f}", label="test の標準偏差"),
                        ]
                    ),
                ],
                widths=[1.4, 1],
                align="center",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(MolGrid, mo, mols, pl):
    mol_grid = mo.ui.anywidget(
        MolGrid(
            mols.sort("y", descending=True).select("id", "smiles", pl.col("y").round(2), "split"),
            color_by="y",
            show_legend=False,
            group_by="split",
            page_size=12,
            selection_mode="pair",
        )
    )
    mo.vstack(
        [
            mo.md(
                "データセットの中の化合物を眺めてみましょう。下のグリッドでは、train / test だけに絞ったり、"
                "部分構造や類似度で検索したりできます。化合物を選ぶと分子量などの情報が表示され、"
                "2 つ選ぶと化合物どうしを比較できます。"
            ),
            mol_grid,
        ]
    )
    return (mol_grid,)


@app.cell(hide_code=True)
def _(MolPair, mo, mol_grid, mols, pl, task):
    _picked = [
        {**r, task.label: r["y"]}
        for i in mol_grid.value.get("selection", [])
        for r in mols.filter(pl.col("id") == i).iter_rows(named=True)
    ]
    if _picked:
        _out = mo.ui.anywidget(
            MolPair(
                *[{**r, "id": f"{r['id']} ({r['split']})"} for r in _picked[:2]],
                value_cols=[task.label],
                value_ranges={task.label: (float(mols["y"].min()), float(mols["y"].max()))},
                properties=["MW", "cLogP", "TPSA", "HBD", "HBA"],
            )
        )
    else:
        _out = mo.callout(
            mo.md("上のグリッドで化合物を選ぶと、ここに表示されます。2 つ選ぶと比較になります。"),
            kind="info",
        )
    _out
    return


@app.cell(hide_code=True)
def _(census_for, mo, np, task, train):
    from rdkit import Chem as _Chem
    from rdkit.Chem import rdFingerprintGenerator as _rfg

    _census = census_for(train["smiles"].to_list(), 2, 2048)
    _envs = _census.n_envs
    # per molecule: distinct radius 0–2 environments before folding, and bits set after folding
    _gen = _rfg.GetMorganGenerator(radius=2, fpSize=2048)
    _mols = [_Chem.MolFromSmiles(s) for s in train["smiles"]]
    _n_envs = np.array([len(_gen.GetSparseCountFingerprint(m).GetNonzeroElements()) for m in _mols])
    _n_bits = np.array([_gen.GetFingerprint(m).GetNumOnBits() for m in _mols])
    mo.md(f"""
    ### 2.2 · {task.dataset} · {task.endpoint} での ECFP4

    第 1 部の動画などで ECFP の collision に少し触れましたが、違う部分構造が同じ bit に入る collision には、2 つの場合があります。

    * **分子内**: 同じ分子の違う部分構造が同じ bit に入る。その分子で立つ bit が 1 つ減る
    * **データセット全体**: 別々の分子の違う部分構造が同じ bit に入る。bit が立っていても、どの部分構造によるものかを区別できない[^effect]

    {task.dataset} · {task.endpoint} のデータセットでは、次のようになっています。

    * 1 分子あたりの部分構造は **{np.median(_n_envs):.0f} 種類** (中央値)
    * 分子内の collision がある化合物は train の **{(_n_bits < _n_envs).mean():.0%}**
    * train {train.height:,} 化合物全体の部分構造は **{int(_envs.sum()):,} 種類**。2048 bit に折りたたむので、1 bit に平均 **{_envs[_envs > 0].mean():.0f} 種類**が入る[^load]
    * bit が立っているとき、それがその bit でいちばん多い部分構造によるものである割合 (purity) は平均 **{_census.purity()[1]:.0%}**

    [^effect]: collision が類似度を高めに見せることと、予測への影響は Virany & Tripp (2025) が調べている
    [^load]: bit-load とも呼ばれる (Gütlein & Kramer, 2016)
    """)
    return


@app.cell(hide_code=True)
def _(BitAtlas, mo, train):
    mo.vstack(
        [
            mo.md(
                "下は train の全 bit です。行を選ぶと、その bit に入っている部分構造が表示されます。"
            ),
            mo.ui.anywidget(BitAtlas(train["smiles"].to_list(), ids=train["id"].to_list())),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1 つの分子でも、両方の collision を確認できます。グリッドから化合物を選んでください。

    * 赤: 分子内の collision
    * グレーのバッジ: データセット全体で、同じ bit に入るほかの部分構造の数。bit を選ぶと、その部分構造が下に表示される
    """)
    return


@app.cell(hide_code=True)
def _(MolGrid, mo, mols, pl, test):
    bits_grid = mo.ui.anywidget(
        MolGrid(
            mols.sort("y", descending=True).select("id", "smiles", pl.col("y").round(2), "split"),
            color_by="y",
            show_legend=False,
            group_by="split",
            page_size=6,  # one row: the bit view below is the point here
            selection_mode="single",
            selection=[test.sort("y", descending=True)["id"][0]],
        )
    )
    bits_grid
    return (bits_grid,)


@app.cell(hide_code=True)
def _(MorganBitTiles, bits_grid, mo, mols, pl, task, train):
    _sel = bits_grid.value.get("selection") or []
    if _sel:
        _row = mols.filter(pl.col("id") == _sel[0]).row(0, named=True)
        _out = mo.ui.anywidget(
            MorganBitTiles(
                _row["smiles"],
                reference=train["smiles"].to_list(),
                ids=train["id"].to_list(),
                label=f"{_row['id']} · {_row['split']} · {task.label} {_row['y']:.2f}",
            )
        )
    else:
        _out = mo.callout(mo.md("上のグリッドで化合物を選ぶと、ここに表示されます。"), kind="info")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 · 違う分子なのに fingerprint が同じ

    train の中で、bit ベクトルがまったく同じになる分子のグループです。bit ベクトルが同じなら、fingerprint だけのモデルはどれにも同じ値を予測します。count fingerprint やキラリティ付きの fingerprint で分かれるかも示します。
    """)
    return


@app.cell(hide_code=True)
def _(bench, mo, np, pl, train):
    _smi = train["smiles"].to_list()
    _y = train["y"].to_numpy()
    _keys = {
        name: [row.tobytes() for row in bench.fingerprints(_smi, **kw).astype(np.uint16)]
        for name, kw in [("bit", {}), ("count", {"count": True}), ("chiral", {"chirality": True})]
    }
    _groups: dict[bytes, list[int]] = {}
    for _i, _k in enumerate(_keys["bit"]):
        _groups.setdefault(_k, []).append(_i)
    _rows = []
    for _g in (g for g in _groups.values() if len(g) > 1):
        _rows.append(
            {
                "分子数": len(_g),
                "ids": " / ".join(train["id"][i] for i in _g[:4]),
                "値の幅": round(float(np.ptp(_y[_g])), 2),
                "count で分かれる": len({_keys["count"][i] for i in _g}) > 1,
                "キラリティで分かれる": len({_keys["chiral"][i] for i in _g}) > 1,
            }
        )
    twins = pl.DataFrame(_rows).sort("値の幅", descending=True) if _rows else pl.DataFrame()
    mo.vstack(
        [
            mo.md(
                f"train {train.height:,} 化合物のうち **{twins.height} グループ** "
                f"({int(twins['分子数'].sum()) if twins.height else 0} 化合物) が同じ bit ベクトルです。"
            ),
            mo.ui.table(twins, page_size=8) if twins.height else mo.md(""),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 · 最近傍の類似度

    ここからは、この fingerprint で測った類似度の話です。test の各化合物について、ECFP4 の Tanimoto 類似度で train の**最近傍**を探します。最近傍が近いほど、「似た化合物の値を借りる」予測が使えそうに思えます。下の図は全エンドポイントの最近傍類似度の分布で、選んだエンドポイントを色で示します。
    """)
    return


@app.cell
def _(db, mo):
    nn_by_task = mo.sql(
        """
        SELECT
            t.dataset || ' · ' || t.endpoint AS name, n.task,
            min(n.tanimoto) AS lo, quantile_cont(n.tanimoto, 0.25) AS q1,
            median(n.tanimoto) AS med, quantile_cont(n.tanimoto, 0.75) AS q3, max(n.tanimoto) AS hi
        FROM neighbours n JOIN tasks t USING (task)
        GROUP BY ALL
        ORDER BY med
        """,
        output=False,
        engine=db,
    )
    return (nn_by_task,)


@app.cell(hide_code=True)
def _(alt, metrics, mo, nn_by_task, pl, task):
    # one box per endpoint, drawn from the quantiles computed in SQL (not from every row)
    _d = nn_by_task.with_columns((pl.col("task") == task.key).alias("selected"))
    _y = alt.Y("name:N", sort=_d["name"].to_list(), title=None)
    _color = alt.Color(
        "selected:N",
        scale=alt.Scale(domain=[True, False], range=["#d6336c", "#adb5bd"]),
        legend=None,
    )
    _x = alt.X("lo:Q", title="最近傍の Tanimoto (ECFP4)", scale=alt.Scale(domain=[0, 1]))
    _base = alt.Chart(_d).encode(y=_y, color=_color)
    _box = (
        _base.mark_rule().encode(x=_x, x2="hi:Q")
        + _base.mark_bar(size=12).encode(
            x="q1:Q",
            x2="q3:Q",
            tooltip=[
                "name",
                alt.Tooltip("med:Q", title="中央値", format=".2f"),
                alt.Tooltip("q1:Q", format=".2f"),
                alt.Tooltip("q3:Q", format=".2f"),
            ],
        )
        + _base.mark_tick(color="white", size=12, thickness=2).encode(x="med:Q")
    ).properties(height=360, width=460)
    _m = metrics.filter(pl.col("task") == task.key).row(0, named=True)
    mo.hstack(
        [
            _box,
            mo.vstack(
                [
                    mo.stat(f"{_m['nn_tanimoto_median']:.2f}", label="最近傍 Tanimoto (中央値)"),
                    mo.stat(f"{_m['share_nn_ge_06']:.0%}", label="最近傍が 0.6 以上の test 化合物"),
                    mo.stat(
                        f"{_m['n_identical_fp']}",
                        label="train と fingerprint が同じ test 化合物 (立体違いなど)",
                    ),
                ]
            ),
        ],
        widths=[1.6, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.5 · 似ていれば値も近いか

    **類似性原理** (似た分子は性質も似ている) が成り立つなら、最近傍が近いほど、test 化合物と最近傍の値の差は小さくなるはずです。下の線は最近傍の類似度の区間ごとの |Δ| (test − 最近傍) の平均で、破線はランダムな train–test ペアの |Δ| です。
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, neighbours, np, pl, spearmanr, task):
    _nb = neighbours.filter(pl.col("task") == task.key).with_columns(
        (pl.col("y_test") - pl.col("y_nn")).abs().alias("dy"),
        (pl.col("tanimoto") * 10).floor().clip(2, 9).alias("bin"),
    )
    _curve = (
        _nb.group_by("bin")
        .agg(pl.len().alias("n"), pl.col("dy").mean().alias("mean_dy"))
        .filter(pl.col("n") >= 5)
        .with_columns(((pl.col("bin") + 0.5) / 10).alias("sim"))
        .sort("bin")
    )
    _rng = np.random.default_rng(0)
    _yt, _yn = _nb["y_test"].to_numpy(), _nb["y_nn"].to_numpy()
    _random = float(np.abs(_rng.permutation(_yn) - _yt).mean())
    _line = (
        alt.Chart(_curve)
        .mark_line(point=True, color="#d6336c")
        .encode(
            x=alt.X("sim:Q", title="最近傍の Tanimoto", scale=alt.Scale(domain=[0.2, 1])),
            y=alt.Y("mean_dy:Q", title=f"|Δ {task.label}| の平均"),
            tooltip=["n:Q", alt.Tooltip("mean_dy:Q", format=".2f")],
        )
    )
    _rule = (
        alt.Chart(pl.DataFrame({"y": [_random]}))
        .mark_rule(strokeDash=[5, 4], color="#6b7280")
        .encode(y="y:Q")
    )
    _rho = spearmanr(_yn, _yt)[0]
    _close = _nb["tanimoto"].to_numpy() >= 0.6
    _rho_close = spearmanr(_yn[_close], _yt[_close])[0] if _close.sum() >= 10 else float("nan")
    mo.hstack(
        [
            (_line + _rule).properties(height=240, width=400),
            mo.vstack(
                [
                    mo.stat(f"{_rho:.2f}", label="最近傍の値で予測したときの Spearman ρ"),
                    mo.stat(f"{_rho_close:.2f}", label="同じ、最近傍が 0.6 以上のものだけ"),
                    mo.stat(f"{_random:.2f}", label="ランダムなペアの |Δ|"),
                ]
            ),
        ],
        widths=[1.4, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.6 · 似ているのに値が違う

    test 化合物と最近傍のペアのうち、似ている (Tanimoto ≥ 0.6) のに値が大きく違うものです。行を選ぶと下で比べられます。「common part」で共通部分構造と違う部分を見られます。
    """)
    return


@app.cell(hide_code=True)
def _(mo, neighbours, pl, task):
    _pairs = (
        neighbours.filter((pl.col("task") == task.key) & (pl.col("tanimoto") >= 0.6))
        .with_columns((pl.col("y_test") - pl.col("y_nn")).alias("Δ"))
        .sort(pl.col("Δ").abs(), descending=True)
        .select(
            pl.col("test_id").alias("test"),
            pl.col("nn_id").alias("最近傍 (train)"),
            pl.col("tanimoto").round(2),
            pl.col("y_test").round(2).alias("test の値"),
            pl.col("y_nn").round(2).alias("最近傍の値"),
            pl.col("Δ").round(2),
        )
    )
    cliff_table = mo.ui.table(
        _pairs, selection="single", initial_selection=[0] if _pairs.height else [], page_size=8
    )
    cliff_table
    return (cliff_table,)


@app.cell(hide_code=True)
def _(MolPair, cliff_table, mo, mols, pl, task):
    _sel = cliff_table.value
    if _sel is None or len(_sel) == 0:
        _out = mo.md("_このエンドポイントには Tanimoto ≥ 0.6 のペアがありません。_")
    else:
        _r = _sel.row(0, named=True)
        _rows = [
            mols.filter(pl.col("id") == i).row(0, named=True)
            for i in (_r["test"], _r["最近傍 (train)"])
        ]
        _out = mo.ui.anywidget(
            MolPair(
                *[{**r, task.label: r["y"]} for r in _rows],
                value_cols=[task.label],
                properties=["MW", "cLogP", "TPSA", "HBD", "HBA"],
            )
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.7 · モデル

    同じ LightGBM (OpenADMET のチャレンジ tutorial と同じ設定) を、4 種類の特徴量で学習した test での成績です。

    * **ECFP4 bit**: 2048 bit
    * **ECFP4 count**: 同じ 2048 次元で、出現回数を残す
    * **RDKit desc**: RDKit の 2D 記述子 217 種 (分子量、logP、TPSA など分子全体の性質)
    * **bit + desc**: 両方を並べたもの
    """)
    return


@app.cell(hide_code=True)
def _(alt, bench, metrics, mo, pl, predictions, task):
    _m = metrics.filter(pl.col("task") == task.key).row(0, named=True)
    _bars = pl.DataFrame(
        {
            "features": bench.FEATURES,
            "Spearman ρ": [_m[f"rho {f}"] for f in bench.FEATURES],
            "R²": [_m[f"r2 {f}"] for f in bench.FEATURES],
        }
    )
    _bar = (
        alt.Chart(_bars)
        .mark_bar()
        .encode(
            y=alt.Y("features:N", sort=bench.FEATURES, title=None),
            x=alt.X("Spearman ρ:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("features:N", legend=None, scale=alt.Scale(scheme="tableau10")),
            tooltip=[
                "features",
                alt.Tooltip("Spearman ρ:Q", format=".2f"),
                alt.Tooltip("R²:Q", format=".2f"),
            ],
        )
        .properties(height=150, width=360)
    )
    _p = predictions.filter(
        (pl.col("task") == task.key)
        & (pl.col("split") == "test")
        & (pl.col("features") == "ECFP4 bit")
    ).select("id", "y", "pred")
    _lo, _hi = float(_p["y"].min()), float(_p["y"].max())
    _scatter = (
        alt.Chart(_p)
        .mark_circle(size=18, opacity=0.5)
        .encode(
            x=alt.X("y:Q", title=f"測定値 ({task.label})", scale=alt.Scale(domain=[_lo, _hi])),
            y=alt.Y("pred:Q", title="予測値 (ECFP4 bit)", scale=alt.Scale(domain=[_lo, _hi])),
            tooltip=["id", alt.Tooltip("y:Q", format=".2f"), alt.Tooltip("pred:Q", format=".2f")],
        )
        .properties(height=260, width=260)
    )
    _diag = (
        alt.Chart(pl.DataFrame({"a": [_lo, _hi]}))
        .mark_line(strokeDash=[4, 4], color="#adb5bd")
        .encode(x="a:Q", y="a:Q")
    )
    mo.hstack([_bar, _scatter + _diag], widths=[1.3, 1], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.8 · モデルが見ているもの

    ECFP4 bit のモデルについて、TreeSHAP (各 bit がその分子の予測をどれだけ上げたか下げたか) を train の全化合物で平均したものが **mean |SHAP|** です。行をクリックすると、その bit に入る部分構造と、その bit が立っている分子が出ます。
    """)
    return


@app.cell(hide_code=True)
def _(BitImportance, mo, np, pl, shap, task, train):
    _s = shap.filter(
        (pl.col("task") == task.key)
        & (pl.col("bit") >= 0)
        & pl.col("id").is_in(train["id"].implode())
    )
    _agg = _s.group_by("bit").agg(
        pl.col("shap").abs().sum().alias("abs"), pl.col("shap").mean().alias("on")
    )
    _imp, _eff = np.zeros(2048), np.zeros(2048)
    _imp[_agg["bit"].to_numpy()] = _agg["abs"].to_numpy() / train.height
    _eff[_agg["bit"].to_numpy()] = _agg["on"].to_numpy()
    mo.ui.anywidget(
        BitImportance(
            train["smiles"].to_list(),
            importance={"mean |SHAP|": _imp},
            effect=_eff,
            effect_label="mean SHAP (bit on)",
            ids=train["id"].to_list(),
            y=train["y"].to_numpy(),
            y_label=task.label,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, neighbours, pl, task):
    _nb = neighbours.filter(pl.col("task") == task.key).sort(
        (pl.col("y_test") - pl.col("y_nn")).abs(), descending=True
    )
    shap_pick = mo.ui.dropdown(
        {
            f"{r['test_id']} (Tanimoto {r['tanimoto']:.2f})": r["test_id"]
            for r in _nb.head(30).iter_rows(named=True)
        },
        value=None
        if _nb.height == 0
        else f"{_nb['test_id'][0]} (Tanimoto {_nb['tanimoto'][0]:.2f})",
        label="test 化合物 (最近傍との差が大きい順)",
    )
    mo.vstack(
        [
            mo.md(r"""
    1 つの予測を分解します。test 化合物と train の最近傍を並べ、原子を TreeSHAP 寄与で塗っています (赤は予測を上げ、青は下げます)。
    """),
            shap_pick,
        ]
    )
    return (shap_pick,)


@app.cell(hide_code=True)
def _(MorganExplorer, mo, mols, neighbours, pl, predictions, shap, shap_pick, task, train):
    _r = neighbours.filter(
        (pl.col("task") == task.key) & (pl.col("test_id") == shap_pick.value)
    ).row(0, named=True)
    _ids = [_r["test_id"], _r["nn_id"]]
    _smi = dict(zip(mols["id"], mols["smiles"]))
    _y = dict(zip(mols["id"], mols["y"]))
    _pred = dict(
        predictions.filter((pl.col("task") == task.key) & (pl.col("features") == "ECFP4 bit"))
        .select("id", "pred")
        .iter_rows()
    )
    _maps = []
    for _i in _ids:
        _c = shap.filter((pl.col("task") == task.key) & (pl.col("id") == _i) & (pl.col("bit") >= 0))
        _maps.append({str(b): float(v) for b, v in _c.select("bit", "shap").iter_rows()})
    mo.ui.anywidget(
        MorganExplorer(
            [
                {
                    "id": _ids[0],
                    "smiles": _smi[_ids[0]],
                    "label": f"test · 測定 {_y[_ids[0]]:.2f} · 予測 {_pred[_ids[0]]:.2f}",
                },
                {
                    "id": _ids[1],
                    "smiles": _smi[_ids[1]],
                    "label": f"train · 測定 {_y[_ids[1]]:.2f} · 予測 (out of fold) {_pred[_ids[1]]:.2f}",
                },
            ],
            reference=train["smiles"].to_list(),
            y=train["y"].to_numpy(),
            y_label=task.label,
            contributions=_maps,
            contrib_label="SHAP",
            contrib_radius=2,
            contrib_n_bits=2048,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 第 3 部 · データセットをまたいで見えたこと

    全エンドポイントの数字を 1 つの表にまとめます (test での Spearman ρ)。
    """)
    return


@app.cell
def _(db, mo):
    _summary = mo.sql(
        """
        SELECT
            t.dataset, t.endpoint,
            round(m.nn_tanimoto_median, 2) AS nn_tanimoto,
            round(m.rho_1nn, 2) AS one_nn,
            round(m."rho ECFP4 bit", 2) AS bit,
            round(m."rho ECFP4 count", 2) AS count,
            round(m."rho RDKit desc", 2) AS "desc",
            round(m."rho bit + desc", 2) AS bit_desc,
            round(m."pair_slope ECFP4 bit", 2) AS pair_slope_bit
        FROM metrics m JOIN tasks t USING (task)
        ORDER BY m."rho ECFP4 bit" DESC
        """,
        engine=db,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * **ペア差の再現**: test 化合物と最近傍の差を、モデルがどれだけの大きさで予測できたか (予測の差を測定の差に回帰した傾き。1 なら大きさまで再現、0 なら差を予測できていない)。最近傍の予測は、その化合物を学習に使っていないモデル (5-fold の out of fold) で出しています

    _(ここに結論の文章を書く: 近いシリーズでは ECFP4 がよく効く / count はほぼどこでも効く / 物性では記述子 / PXR のような設定では FP だけの単純なモデルは難しいが原因は 1 つではない / ごく近いペアの差はどのデータでも予測できない / ECFP4 は起点)_
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### この notebook について

    * **データ**: [PXR challenge](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY-4.0)、[ASAP-Polaris-OpenADMET antiviral challenge](https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge) (MIT)、[OpenADMET-ExpansionRx challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) (CC-BY-4.0)
    * **事前計算**: `dev/precompute.py` → `results/precomputed/`。計算のコードは `molwidgets.bench`
    * **ウィジェット**: `ECFPMovie`、`ECFPStepper`、`MorganBitTiles`、`MolGrid`、`MolPair`、`BitAtlas`、`BitImportance`、`MorganExplorer` は、この notebook のために作った anywidget コンポーネントです ([ソース](https://github.com/N283T/openadmet-marimo))
    * **AI の利用**: ウィジェット、動画、notebook の骨組みのコーディングには Claude (Anthropic) をアシスタントとして使いました。問いの立て方、解析の選び方、解釈は私自身のものです

    ### 参考文献

    * Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742–754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
    * Gütlein, M.; Kramer, S. Filtered circular fingerprints improve either prediction or runtime performance while retaining interpretability. *J. Cheminform.* **2016**, 8, 60. [doi:10.1186/s13321-016-0173-z](https://doi.org/10.1186/s13321-016-0173-z)
    * Virany, W.; Tripp, A. Hash Collisions in Molecular Fingerprints: Effects on Property Prediction and Bayesian Optimization. AI for Science workshop, NeurIPS 2025. [arXiv:2511.17078](https://arxiv.org/abs/2511.17078)
    """)
    return


if __name__ == "__main__":
    app.run()
