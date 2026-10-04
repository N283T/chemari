# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.25",
#     "chemari @ git+https://github.com/N283T/chemari@v0.1.3",
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
app = marimo.App(width="medium", app_title="その ECFP4、理解して使っていますか？")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # その ECFP4、理解して使っていますか？

    分子を機械学習にかけるとき、とりあえずこう書いていないでしょうか。

    ```python
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    ```

    これが **ECFP4**<sup><a href="#ref-1">1</a></sup> で、ケモインフォマティクスで最もよく使われる分子表現です。計算が速く、調整するパラメータもほとんどありません。類似検索でも QSAR でも、多くの場合に良い結果が出ます。新しい手法を評価するときも、まず比べる相手はたいてい ECFP4 です。

    ただ、この 2048 bit が分子の何を記録していて、何を記録していないのかを説明できる人は多くありません。同じ設定で作った fingerprint でも、その中身はデータセットによって大きく違います。この notebook では、ウィジェットで ECFP4 の中身を見ながら理解を深めていきます。

    /// admonition | 名前について
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) と RDKit の **Morgan fingerprint** は同じものです。ECFP*n* の数字は原子のまわりを見る範囲の*直径*で、RDKit では代わりに半径 (radius) で指定します。ECFP4 は `radius=2` にあたります。
    ///

    この notebook は 3 部構成です。

    * **[第 1 部](#part-1) · ECFP4 の中身**: ECFP4 のアルゴリズムとその特性
    * **[第 2 部](#part-2) · データセットでの ECFP4**: データセットの中で ECFP4 がどうなっているか
    * **[第 3 部](#part-3) · おまけ**: データセットどうしを比べて全体を俯瞰する

    おすすめの読み方は次の順番です。

    1. [第 1 部](#part-1)と[第 2 部](#part-2)で ECFP4 の仕組みとウィジェットの見方を確かめる (第 2 部はまず 1 つのエンドポイントで)
    2. [第 3 部](#part-3)で 16 のエンドポイントを眺める
    3. 気になったエンドポイントを[選び直して](#picker)、第 2 部で中身を見る
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import duckdb
    import numpy as np
    import polars as pl

    from chemari import (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MolScatter,
        MorganBitTiles,
        MorganExplorer,
        census_for,
    )
    from chemari.examples import openadmet as bench

    _ = alt.data_transformers.disable_max_rows()
    return (
        BitAtlas,
        BitImportance,
        ECFPMovie,
        ECFPStepper,
        MolGrid,
        MolPair,
        MolScatter,
        MorganBitTiles,
        MorganExplorer,
        Path,
        alt,
        bench,
        census_for,
        duckdb,
        np,
        pl,
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
    molecules, neighbours, predictions, metrics, shap, gain, bitlen = (
        _computed[t] if t in _computed else pl.read_parquet(_where[t]) for t in bench.TABLES
    )
    tasks = pl.DataFrame(
        [
            {
                "task": t.key,
                "dataset": t.dataset,
                "endpoint": t.endpoint,
                "label": t.label,
                "what": bench.NOTES_JA[t.key],
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
                "ファイルが見つからないときは、同じコード (`chemari.examples.openadmet`) でこの場で計算します。"
                "読み込んだ表は DuckDB で SQL を使って集計します。\n\n読み込み元:\n\n" + _sources
            )
        }
    )
    return bitlen, gain, metrics, molecules, neighbours, predictions, shap, tasks


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

    ## <span id="part-1"></span>第 1 部 · ECFP4 の中身

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
    同じ手順を好きな分子で追えます。SMILES を入力するか例を選び、1 ステップずつ進めるか自動再生してください。
    """)
    return


@app.cell(hide_code=True)
def _(ECFPStepper, mo):
    mo.ui.anywidget(ECFPStepper())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    動画と上のウィジェットに出てきた用語を整理しておきます。

    * **radius**: 原子のまわりを結合何本分まで見るか。0 から 1 つずつ広げ、ECFP4 では 2 まで見る
    * **部分構造 (environment)**: ある原子を中心に、radius の範囲に入る原子と結合
    * **folding**: fingerprint を固定長 (ECFP4 では多くの場合 2048 bit) に折りたたむ操作
    * **bit**: fingerprint の 1 つの要素。その番号に入る部分構造が分子にあれば 1 になる
    * **collision**: 違う部分構造が同じ bit に入ること

    ### 実際に ECFP4 を見てみる

    下のウィジェットは実際に生成される ECFP4 を可視化したものです。右の bit を選ぶと該当する部分構造が分子の上でハイライトされるほか、collision があるかどうかも確認できます。好きな分子に変えたり、radius や folding (bit 数) を変えたりして、いろいろ試してみてください。
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

    ### ECFP4 の注意点

    動画で見た collision のほかにも、ECFP4 には注意点があります。ここでは 2 つの分子の fingerprint を比較するウィジェットを使って、それらを見ていきます。

    ### 注意点 1 · 違う分子なのに fingerprint が一致する

    類似検索で、違う分子が類似度 1.0 で複数ヒットすることがたまにあります。これは次のようなときに起こります。

    * 立体化学が違う
    * 環のサイズや鎖の長さが違う
    """)
    return


@app.cell
def _():
    # names checked against PubChem (structure, and stereo for the enantiomers)
    SAME_FP = [
        {
            "label": "cyclohexylamine / cycloheptylamine (ring size)",
            "a": "NC1CCCCC1",
            "b": "NC1CCCCCC1",
            "name_a": "cyclohexylamine",
            "name_b": "cycloheptylamine",
        },
        {
            "label": "azepane / azocane (ring size)",
            "a": "C1CCCNCC1",
            "b": "C1CCCNCCC1",
            "name_a": "azepane",
            "name_b": "azocane",
        },
        {
            "label": "nonanoic acid / palmitic acid (chain length)",
            "a": "CCCCCCCCC(=O)O",
            "b": "CCCCCCCCCCCCCCCC(=O)O",
            "name_a": "nonanoic acid",
            "name_b": "palmitic acid",
        },
        {
            "label": "(R)- / (S)-thalidomide (stereo)",
            "a": "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "b": "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1",
            "name_a": "(R)-thalidomide",
            "name_b": "(S)-thalidomide",
        },
        {
            "label": "lansoprazole / dexlansoprazole (stereo)",
            "a": "Cc1c(OCC(F)(F)F)ccnc1CS(=O)c1nc2ccccc2[nH]1",
            "b": "Cc1c(OCC(F)(F)F)ccnc1C[S@@](=O)c1nc2ccccc2[nH]1",
            "name_a": "lansoprazole",
            "name_b": "dexlansoprazole",
        },
    ]
    CLOSE_BUT_FAR = [
        {
            "label": "N-methylacetamide / N-ethylacetamide (from the movie, CH₃ → C₂H₅)",
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
            "label": "OADMET-0001944 / OADMET-0002007 (PXR, one CH₃)",
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
            "stereo": _by_label["(R)- / (S)-thalidomide (stereo)"],
            "ring size": _by_label["cyclohexylamine / cycloheptylamine (ring size)"],
            "chain length": _by_label["nonanoic acid / palmitic acid (chain length)"],
        },
        value="stereo",
        label="example",
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
    特に環のサイズや鎖の長さが違う場合は、分子の組成そのものが違うのに同じものとして扱われます。類似検索なら目で見て気づけますが、機械学習では違う分子に対してモデルがまったく同じ予測を返すので、リスクが大きくなります。

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
        return " / ".join(out) or "neither"

    same_table = mo.ui.table(
        pl.DataFrame(
            [
                {
                    "A": e["name_a"],
                    "B": e["name_b"],
                    "difference": e["label"].rsplit("(", 1)[1].rstrip(")"),
                    "separated by": _separates(e),
                }
                for e in SAME_FP
            ]
        ),
        selection="single",
        initial_selection=[0],
        label="pairs with the same fingerprint",
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

    * chirality を入れても区別できない立体がある (lansoprazole と dexlansoprazole の違いはスルホキシドの硫黄の立体で、ECFP はこの立体を区別しない)
    * count では collision が起きている bit で別の部分構造の回数もまとめて数えられるので、collision のデメリットが通常より大きくなることもある

    collision は bit 数を増やせば減らせますが、bit 数が増えるわりに得られるものが少ないこともあります。データセットに合わせて使う設定を見極めましょう。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 注意点 2 · 似ているのに類似度が思ったより低い

    一方で、見た目はよく似ているのに Tanimoto が 0.4〜0.7 くらいと直感より低くなるペアもあります。ECFP4 は各原子のまわり半径 2 までを見るので、原子が 1 つ変わるだけでそのまわりの部分構造がまとめて変わり、10 個以上の bit が入れ替わるためです。

    前の節の鎖長や環サイズの例では、炭素が増えても fingerprint は変わりませんでした。この 2 つの違いは、構造のどこが変わったかにあります。

    * 長い鎖や大きい環の途中に同じ単位を足す → 半径 2 の中に見える部分構造はすでにあるものと同じ → 新しい bit は立たない
    * 原子を置き換える、置換基や鎖の端を変える → そこから半径 2 以内の部分構造がすべて新しくなる
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <span id="far-pair"></span>下の表からペアを選ぶと、その 2 つの構造と fingerprint の比較が順に表示されます。fingerprint の比較では、右側に片方にしかない bit (differ) が並びます。変わった原子のまわりの部分構造がまとめて入れ替わっているのがわかります。
    """)
    return


@app.cell(hide_code=True)
def _(CLOSE_BUT_FAR, SAME_FP, bench, mo, pl):
    def _bits(e):
        a, b = bench.fingerprints([e["a"], e["b"]]).astype(bool)
        return round(float((a & b).sum() / (a | b).sum()), 2), int((a ^ b).sum())

    # the chain-length pair from the previous section, for comparison (no bit changes)
    far_pairs = CLOSE_BUT_FAR + [e for e in SAME_FP if "chain length" in e["label"]]

    far_table = mo.ui.table(
        pl.DataFrame(
            [
                {
                    "A": e["name_a"],
                    "B": e["name_b"],
                    "difference": e["label"].rsplit("(", 1)[1].rstrip(")"),
                    "Tanimoto": t,
                    "bits that differ": n,
                }
                for e in far_pairs
                for t, n in [_bits(e)]
            ]
        ),
        selection="single",
        initial_selection=[0],
        label="pairs that look alike but have a low Tanimoto",
    )
    far_table
    return far_pairs, far_table


@app.cell(hide_code=True)
def _(far_pairs, far_table):
    # the pair picked in the table, for the two widgets below
    _sel = far_table.value
    _name = _sel["A"][0] if _sel is not None and len(_sel) else far_pairs[0]["name_a"]
    far_pair = next(e for e in far_pairs if e["name_a"] == _name)
    return (far_pair,)


@app.cell(hide_code=True)
def _(MolPair, far_pair, mo):
    mo.ui.anywidget(
        MolPair(
            {"id": far_pair["name_a"], "smiles": far_pair["a"]},
            {"id": far_pair["name_b"], "smiles": far_pair["b"]},
            show_smiles=False,
            show_common=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(MorganExplorer, far_pair, mo):
    _e = far_pair
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

    [構造を並べたウィジェット](#far-pair)を similarity タブに切り替えると、選んだペアをいくつかの手法と類似度 (係数) で比較できます。手法によって値の出方が違うので、手法どうしで数値を比べるのではなく、同じ手法でペアを変えたときの値の動きを見てください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ただし、どれも決定的な解決策ではなく、どの程度を「似ている」とするかは結局は感覚に近いものです。スクリーニング、クラスタリング、交差検証の分割などでは類似度のしきい値で機械的に区切ることが多いので、いくつか試しながら目的に合う方法と基準を選んでください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## <span id="part-2"></span>第 2 部 · データセットでの ECFP4
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3 つのデータセット

    3 つのデータセットの値は、どれも実際の創薬プロジェクトで測られたものです。train / test は、チャレンジで使われた分け方のまま使います。test と同じ構造が train にある化合物は test から外し、同じ split の中で重複する構造は値を平均しています。比の尺度で測る値 (溶解度、クリアランス、透過性、非結合率など) は log10(x + 1) に変換しています。
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
    task_pick = mo.ui.dropdown(_options, value="PXR · pEC50", label="dataset · endpoint")
    mo.vstack(
        [
            mo.md(r"""
    <span id="picker"></span>下から好きなデータセットとエンドポイントを選んでください。どれを選んでも、同じ手順で ECFP4 を見ていきます。データセットの背景は、上の blog を見てください。いろいろなデータセットに切り替えて、違いを楽しんでみてください。
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
def _(alt, bench, mo, mols, task, test, train):
    _hist = (
        alt.Chart(mols.select("y", "split"))
        .mark_bar(opacity=0.6)
        .encode(
            x=alt.X("y:Q", bin=alt.Bin(maxbins=40), title=task.label),
            y=alt.Y("count():Q", stack=None, title="compounds"),
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
    ### <span id="sec-2-1"></span>2.1 · データセットの中身

    **{task.dataset} · {task.endpoint}**: {bench.NOTES_JA[task.key]}。train {train.height:,} 化合物、test {test.height:,} 化合物です。
    """),
            mo.hstack(
                [
                    _hist,
                    mo.vstack(
                        [
                            mo.stat(f"{train['y'].std():.2f}", label=f"SD of {task.label} (train)"),
                            mo.stat(f"{test['y'].std():.2f}", label="SD (test)"),
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
    ### <span id="sec-2-2"></span>2.2 · {task.dataset} · {task.endpoint} での ECFP4

    [第 1 部](#part-1)で見た collision には 2 つの場合があります。

    * **分子内**: 同じ分子の違う部分構造が同じ bit に入る。その分子で立つ bit が 1 つ減る
    * **データセット全体**: 別々の分子の違う部分構造が同じ bit に入る。bit が立っていても、どの部分構造によるものかを区別できない<sup><a href="#ref-2">2</a></sup>

    {task.dataset} · {task.endpoint} では、次のようになっています。

    * 1 化合物あたりの部分構造は **{np.median(_n_envs):.0f} 種類** (中央値)
    * 分子内の collision がある化合物は train の **{(_n_bits < _n_envs).mean():.0%}**
    * train {train.height:,} 化合物全体の部分構造は **{int(_envs.sum()):,} 種類**。2048 bit に折りたたむので、1 bit に平均 **{_envs[_envs > 0].mean():.0f} 種類**が入る<sup><a href="#ref-3">3</a></sup>
    * bit が立っている化合物のうち、その bit でいちばん多い部分構造を持つものの割合 (purity) は平均 **{_census.purity()[1]:.0%}**

    部分構造の区別には RDKit が折りたたむ前に付ける識別子を使っています。別の部分構造が同じ識別子になる場合は、ここでは数えられません。

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
    1 つの化合物でも両方の collision を確認できます。グリッドから化合物を選んでください。

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
def _(bench, mo, mols, np, pl, task):
    # groups of molecules (train and test together) whose bit vectors are identical, and whether
    # counts or chirality tell them apart
    _smi = mols["smiles"].to_list()
    _keys = {
        name: [row.tobytes() for row in bench.fingerprints(_smi, **kw).astype(np.uint16)]
        for name, kw in [("bit", {}), ("count", {"count": True}), ("chiral", {"chirality": True})]
    }
    _groups: dict[bytes, list[int]] = {}
    for _i, _k in enumerate(_keys["bit"]):
        _groups.setdefault(_k, []).append(_i)
    _y = mols["y"].to_numpy()
    _rows = []
    for _g in (g for g in _groups.values() if len(g) > 1):
        _by = [k for k in ("count", "chiral") if len({_keys[k][i] for i in _g}) > 1]
        _lo, _hi = min(_g, key=lambda i: _y[i]), max(_g, key=lambda i: _y[i])
        _rows.append(
            {
                "compounds": " / ".join(mols["id"][i] for i in _g[:3])
                + (" …" if len(_g) > 3 else ""),
                "n": len(_g),
                f"Δ {task.label}": round(float(_y[_hi] - _y[_lo]), 2),
                "separated by": " / ".join(
                    {"count": "count", "chiral": "chirality"}[k] for k in _by
                )
                or "neither",
                # the two members furthest apart in value, for the comparison below
                "_a": mols["id"][_lo],
                "_b": mols["id"][_hi],
            }
        )
    twins = (
        pl.DataFrame(_rows).sort(f"Δ {task.label}", descending=True) if _rows else pl.DataFrame()
    )
    _n = {
        k: int((twins["separated by"].str.contains(k)).sum()) if twins.height else 0
        for k in ("count", "chirality", "neither")
    }
    _numbers = (
        f"""
    * 同じ fingerprint になる化合物は **{twins.height} 組**、あわせて **{int(twins["n"].sum())} 化合物**
    * count で区別できるのは **{_n["count"]} 組**、chirality で区別できるのは **{_n["chirality"]} 組**、どちらでも区別できないのは **{_n["neither"]} 組**
    * 同じ fingerprint の中での {task.label} の差は最大で **{twins[f"Δ {task.label}"].max():.2f}**
    """
        if twins.height
        else """
    * 同じ fingerprint になる化合物の組はない
    """
    )
    mo.md(f"""
    ### <span id="sec-2-3"></span>2.3 · 違う分子なのに fingerprint が一致する

    [第 1 部](#part-1)の注意点 1 です。fingerprint が同じ化合物には、fingerprint だけを使うモデルは同じ値を予測します。

    {task.dataset} · {task.endpoint} では、次のようになっています。
    {_numbers}""")
    return (twins,)


@app.cell(hide_code=True)
def _(mo, twins):
    twins_table = (
        mo.ui.table(
            twins.drop("_a", "_b"),
            selection="single",
            initial_selection=[0],
            page_size=6,
            label="groups of compounds with the same fingerprint",
        )
        if twins.height
        else None
    )
    twins_table
    return (twins_table,)


@app.cell(hide_code=True)
def _(MorganExplorer, mo, mols, pl, task, twins, twins_table):
    if twins_table is None:
        _out = None
    else:
        _sel = twins_table.value
        _key = _sel["compounds"][0] if _sel is not None and len(_sel) else twins["compounds"][0]
        _row = twins.filter(pl.col("compounds") == _key).row(0, named=True)
        _pair = [mols.filter(pl.col("id") == _row[k]).row(0, named=True) for k in ("_a", "_b")]
        _out = mo.ui.anywidget(
            MorganExplorer(
                [
                    {
                        "id": r["id"],
                        "smiles": r["smiles"],
                        "label": f"{r['split']} · {task.label} {r['y']:.2f}",
                    }
                    for r in _pair
                ],
                stereo_labels=True,
            )
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-2-4"></span>2.4 · Nearest neighbor (NN)

    ここからは類似度を見ていきましょう。test の各化合物について、train の中で ECFP4 の Tanimoto 類似度がいちばん高い化合物を **nearest neighbor (NN)** と呼びます。

    NN との類似度を見ると、test の化合物に似た構造が train にあるかどうかがわかります。下の分布が右に寄っていれば、test は train にある構造の近くにあります。左に寄っていれば、test には train にない構造が多いことになります。
    """)
    return


@app.cell(hide_code=True)
def _(alt, metrics, mo, neighbours, pl, task):
    _nn = neighbours.filter(pl.col("task") == task.key).select("tanimoto")
    _hist = (
        alt.Chart(_nn)
        .mark_bar(color="#1c7ed6", opacity=0.8)
        .encode(
            x=alt.X(
                "tanimoto:Q",
                bin=alt.Bin(extent=[0, 1], step=0.05),
                title="Tanimoto to the NN (ECFP4)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y("count():Q", title="test compounds"),
        )
        .properties(height=220, width=420)
    )
    _m = metrics.filter(pl.col("task") == task.key).row(0, named=True)
    mo.hstack(
        [
            _hist,
            mo.vstack(
                [
                    mo.stat(f"{_m['nn_tanimoto_median']:.2f}", label="median Tanimoto to the NN"),
                    mo.stat(f"{_m['share_nn_ge_06']:.0%}", label="test compounds with NN ≥ 0.6"),
                    mo.stat(
                        f"{_m['n_identical_fp']}",
                        label="test compounds sharing a train fingerprint",
                    ),
                ]
            ),
        ],
        widths=[1.4, 1],
        align="center",
    )
    return


@app.cell
def _(neighbours, np, pl, task):
    # every test compound with its NN: the similarity, and how far apart their values are
    nn_pairs = neighbours.filter(pl.col("task") == task.key).with_columns(
        (pl.col("y_test") - pl.col("y_nn")).alias("delta"),
        (pl.col("y_test") - pl.col("y_nn")).abs().alias("dy"),
    )
    # the same difference for random train–test pairs (NN values shuffled)
    random_dy = float(
        np.abs(
            np.random.default_rng(0).permutation(nn_pairs["y_nn"].to_numpy())
            - nn_pairs["y_test"].to_numpy()
        ).mean()
    )
    # activity cliffs: similar by ECFP4, yet as far apart in value as unrelated compounds
    cliffs = nn_pairs.filter((pl.col("tanimoto") >= 0.6) & (pl.col("dy") >= random_dy))
    return cliffs, nn_pairs, random_dy


@app.cell(hide_code=True)
def _(mo, nn_pairs, task):
    mo.md(f"""
    ### <span id="sec-2-5"></span>2.5 · 類似性原理と activity cliff

    創薬などの化合物の解析では、類似性原理と、その例外にあたる activity cliff が対になる重要な概念です。

    * **類似性原理** (similarity principle): 構造が似た分子は性質も似ているという考え方<sup><a href="#ref-4">4</a></sup>。類似検索や似た化合物の値を使った予測の前提になっている
    * **activity cliff**: 構造がよく似ているのに活性が大きく違う化合物のペア<sup><a href="#ref-5">5</a></sup>。メチル基を 1 つ足すだけで活性が 100 倍以上変わることがある「magic methyl」が有名な例<sup><a href="#ref-6">6</a></sup>

    下の図で、{task.dataset} · {task.endpoint} ではどうなっているかを見ます。

    * 点: test 化合物 ({nn_pairs.height:,} 個)
    * 横軸: NN との類似度
    * 縦軸: NN との {task.label} の差 |Δ|
    * 赤い線: 類似度の区間ごとの |Δ| の平均
    * 破線: ランダムな train–test ペアの |Δ| の平均

    類似性原理が成り立っていれば、右に行くほど点は下に集まります。activity cliff は、ここでは類似度が 0.6 以上で |Δ| がランダムなペアの平均以上あるペアとしました (図の右上の色を付けた領域)。
    """)
    return


@app.cell(hide_code=True)
def _(alt, cliffs, mo, nn_pairs, pl, random_dy, task):
    _curve = (
        nn_pairs.with_columns((pl.col("tanimoto") * 10).floor().clip(0, 9).alias("bin"))
        .group_by("bin")
        .agg(pl.len().alias("n"), pl.col("dy").mean().alias("mean_dy"))
        .filter(pl.col("n") >= 5)
        .with_columns(((pl.col("bin") + 0.5) / 10).alias("sim"))
        .sort("bin")
    )
    _top = float(nn_pairs["dy"].max()) * 1.05
    _x = alt.X("tanimoto:Q", title="Tanimoto to the NN (ECFP4)", scale=alt.Scale(domain=[0, 1]))
    _y = alt.Y("dy:Q", title=f"|Δ {task.label}| (test − NN)", scale=alt.Scale(domain=[0, _top]))
    _region = (
        alt.Chart(pl.DataFrame({"x": [0.6], "x2": [1.0], "y": [random_dy], "y2": [_top]}))
        .mark_rect(color="#d6336c", opacity=0.08)
        .encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    )
    _label = (
        alt.Chart(pl.DataFrame({"x": [0.99], "y": [_top * 0.97], "t": ["activity cliff"]}))
        .mark_text(align="right", baseline="top", color="#d6336c", fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="t:N")
    )
    _points = (
        alt.Chart(nn_pairs.select("test_id", "nn_id", "tanimoto", "dy"))
        .mark_circle(size=18, opacity=0.35, color="#1c7ed6")
        .encode(
            x=_x,
            y=_y,
            tooltip=[
                alt.Tooltip("test_id:N", title="test"),
                alt.Tooltip("nn_id:N", title="NN"),
                alt.Tooltip("tanimoto:Q", format=".2f"),
                alt.Tooltip("dy:Q", title="|Δ|", format=".2f"),
            ],
        )
    )
    _line = (
        alt.Chart(_curve)
        .mark_line(point=True, color="#d6336c")
        .encode(
            x="sim:Q",
            y="mean_dy:Q",
            tooltip=[
                alt.Tooltip("n:Q", title="compounds"),
                alt.Tooltip("mean_dy:Q", title="mean |Δ|", format=".2f"),
            ],
        )
    )
    _rule = (
        alt.Chart(pl.DataFrame({"y": [random_dy]}))
        .mark_rule(strokeDash=[5, 4], color="#6b7280")
        .encode(y="y:Q")
    )
    mo.hstack(
        [
            (_region + _points + _rule + _line + _label).properties(height=280, width=430),
            mo.vstack(
                [
                    mo.stat(f"{nn_pairs['dy'].mean():.2f}", label="mean |Δ| to the NN"),
                    mo.stat(f"{random_dy:.2f}", label="mean |Δ| of random pairs"),
                    mo.stat(f"{cliffs.height}", label="activity cliffs"),
                ]
            ),
        ],
        widths=[1.5, 1],
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo, task):
    mo.md(rf"""
    下の表は activity cliff のペアです。行を選ぶと、その 2 つを 2 通りの見方で比べられます。

    * **Molecules**: 構造と測定値、物性を並べる
    * **Fingerprints**: 2 つの間で違う bit を並べる。`Δ {task.label}` の列は、train の中でその bit が立っている化合物の平均 {task.label} から立っていない化合物の平均を引いた値。プラスなら、その bit を持つ化合物は平均より値が高い傾向にある
    """)
    return


@app.cell(hide_code=True)
def _(cliffs, mo, pl):
    _pairs = cliffs.sort("dy", descending=True).select(
        pl.col("test_id").alias("test"),
        pl.col("nn_id").alias("NN (train)"),
        pl.col("tanimoto").round(2).alias("Tanimoto"),
        pl.col("y_test").round(2).alias("value (test)"),
        pl.col("y_nn").round(2).alias("value (NN)"),
        pl.col("delta").round(2).alias("Δ"),
    )
    cliff_table = mo.ui.table(
        _pairs,
        selection="single",
        initial_selection=[0] if _pairs.height else [],
        page_size=6,
        label="activity cliff pairs",
    )
    cliff_table
    return (cliff_table,)


@app.cell(hide_code=True)
def _(MolPair, MorganExplorer, cliff_table, mo, mols, pl, task, train):
    _sel = cliff_table.value
    if _sel is None or len(_sel) == 0:
        _out = mo.callout(
            mo.md("このエンドポイントには activity cliff にあたるペアがありません。"), kind="info"
        )
    else:
        _r = _sel.row(0, named=True)
        _rows = [
            mols.filter(pl.col("id") == i).row(0, named=True)
            for i in (_r["test"], _r["NN (train)"])
        ]
        _out = mo.ui.tabs(
            {
                "Molecules": mo.ui.anywidget(
                    MolPair(
                        *[{**r, task.label: r["y"]} for r in _rows],
                        value_cols=[task.label],
                        properties=["MW", "cLogP", "TPSA", "HBD", "HBA"],
                    )
                ),
                "Fingerprints": mo.ui.anywidget(
                    MorganExplorer(
                        [
                            {
                                "id": r["id"],
                                "smiles": r["smiles"],
                                "label": f"{r['split']} · {task.label} {r['y']:.2f}",
                            }
                            for r in _rows
                        ],
                        reference=train["smiles"].to_list(),
                        y=train["y"].to_numpy(),
                        y_label=task.label,
                        row_filter="differ",
                    )
                ),
            }
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-2-6"></span>2.6 · モデル

    ECFP4 を特徴量にして測定値を予測するモデルを作ります。モデルは LightGBM です。比較のために、特徴量だけを変えたモデルも学習させています。

    * **ECFP4 bit**: 2048 bit
    * **ECFP4 count**: 同じ 2048 次元で、出現回数を残す
    * **RDKit desc**: RDKit の 2D 記述子 217 種 (分子量、logP、TPSA など分子全体の性質)
    * **bit + desc**: 両方を並べたもの

    どのモデルも train で学習し、test を予測しています。基準として、[2.4](#sec-2-4) の NN の測定値をそのまま予測値にした場合 (**NN value**) も並べました。類似性原理だけでどこまで予測できるかの目安です。

    棒グラフの下の散布図は、選んだモデルの test での予測です。青い点は NN との類似度が 0.6 以上 ([2.5](#sec-2-5) で「似ている」とした範囲) の化合物、三角は [2.5](#sec-2-5) の activity cliff の test 化合物です。点にカーソルを合わせると、その化合物が右に表示されます。クリックすると固定できます。
    """)
    return


@app.cell(hide_code=True)
def _(alt, bench, metrics, mo, nn_pairs, pl, task):
    _m = metrics.filter(pl.col("task") == task.key).row(0, named=True)
    _order = ["NN value", *bench.FEATURES]
    _yt, _yn = nn_pairs["y_test"].to_numpy(), nn_pairs["y_nn"].to_numpy()
    _nn = {
        "rho": _m["rho_1nn"],
        "r2": float(1 - ((_yn - _yt) ** 2).sum() / ((_yt - _yt.mean()) ** 2).sum()),
        "mae": float(abs(_yn - _yt).mean()),
    }
    _colours = alt.Scale(
        domain=_order, range=["#adb5bd", "#4c78a8", "#f58518", "#e45756", "#72b7b2"]
    )

    def _room(scores):
        # the data's range (with zero) plus room on the right for the value labels
        _lo, _hi = min(0.0, scores.min()), max(0.0, scores.max())
        return [_lo, _hi + 0.2 * (_hi - _lo)]

    def _panel(key, title, first, domain=None):
        _bars = pl.DataFrame(
            {
                "features": _order,
                "score": [_nn[key]] + [_m[f"{key} {f}"] for f in bench.FEATURES],
            }
        )
        _bar = (
            alt.Chart(_bars, title=alt.Title(title, fontSize=12, anchor="start"))
            .mark_bar()
            .encode(
                y=alt.Y(
                    "features:N",
                    sort=_order,
                    title=None,
                    axis=alt.Axis(labels=first, ticks=first, domain=first),
                ),
                x=alt.X(
                    "score:Q",
                    title=None,
                    scale=alt.Scale(domain=domain or _room(_bars["score"]), nice=False),
                ),
                color=alt.Color("features:N", legend=None, scale=_colours),
                tooltip=["features", alt.Tooltip("score:Q", title=title, format=".2f")],
            )
            .properties(height=170, width=265)
        )
        # labels of negative bars (R² of the NN value) sit right of zero
        _text = (
            _bar.mark_text(align="left", dx=3, fontSize=11)
            .transform_calculate(at="max(datum.score, 0)")
            .encode(x="at:Q", text=alt.Text("score:Q", format=".2f"), color=alt.value("#495057"))
        )
        return _bar + _text

    mo.hstack(
        [
            alt.hconcat(
                _panel("rho", "Spearman ρ", True, [0, 1]),
                _panel("r2", "R²", False),
                _panel("mae", "MAE (lower is better)", False),
                spacing=28,
            ).resolve_scale(color="shared")
        ],
        justify="center",
    )
    return


@app.cell(hide_code=True)
def _(bench, mo):
    model_pick = mo.ui.radio(
        ["NN value", *bench.FEATURES], value="ECFP4 bit", inline=True, label="model"
    )
    return (model_pick,)


@app.cell(hide_code=True)
def _(MolScatter, cliffs, mo, model_pick, mols, nn_pairs, pl, predictions, task):
    _smi = dict(zip(mols["id"], mols["smiles"]))
    # "NN value" predicts each test compound with its NN's measured value
    _pred = (
        nn_pairs.select(
            pl.col("test_id").alias("id"), pl.col("y_test").alias("y"), pl.col("y_nn").alias("pred")
        )
        if model_pick.value == "NN value"
        else predictions.filter(
            (pl.col("task") == task.key)
            & (pl.col("split") == "test")
            & (pl.col("features") == model_pick.value)
        ).select("id", "y", "pred")
    )
    _p = (
        _pred.join(nn_pairs.select(pl.col("test_id").alias("id"), "nn_id", "tanimoto"), on="id")
        .with_columns(
            pl.col("id").replace_strict(_smi).alias("smiles"),
            pl.col("nn_id").replace_strict(_smi).alias("nn_smiles"),
            pl.col("id").is_in(cliffs["test_id"].implode()).alias("cliff"),
            pl.when(pl.col("tanimoto") >= 0.6)
            .then(pl.lit("≥ 0.6"))
            .otherwise(pl.lit("< 0.6"))
            .alias("NN"),
        )
        .sort("tanimoto")
    )
    _scatter = mo.ui.anywidget(
        MolScatter(
            _p.select(
                "id",
                "smiles",
                pl.col("y").alias("measured"),
                pl.col("pred").alias("predicted"),
                pl.col("tanimoto").alias("Tanimoto to the NN"),
                "cliff",
                "NN",
                "nn_id",
                "nn_smiles",
            ),
            x="measured",
            y="predicted",
            x_label=f"measured {task.label}",
            y_label=f"predicted ({model_pick.value})",
            color_by="NN",
            color_label="Tanimoto to the NN",
            color_map={"≥ 0.6": "#1c7ed6", "< 0.6": "#b8c2cc"},
            mark_by="cliff",
            mark_label="activity cliff",
            diagonal=True,
            same_axes=True,
            card_title="test compound",
            info_title=f"prediction ({model_pick.value})",
            axis_fields=["measured", "predicted"],
            partner_id_col="nn_id",
            partner_smiles_col="nn_smiles",
            partner_label="NN (train)",
            partner_fields=["Tanimoto to the NN"],
        )
    )
    mo.vstack([model_pick, _scatter])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-2-7"></span>2.7 · 特徴量重要度

    ECFP4 bit のモデルがどの bit を使っているかを 2 つの指標で並べます。

    * **gain**: LightGBM の木がその bit で分岐したときに減った誤差の合計 ([`feature_importance(importance_type="gain")`](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster.feature_importance))
    * **mean |SHAP|**: TreeSHAP (各 bit がその化合物の予測をどれだけ上げたか下げたか) の絶対値を train の全化合物で平均したもの。bit が立っていない化合物の分も含む

    行をクリックすると、その bit に入る部分構造とその bit が立っている化合物が下に表示されます。
    """)
    return


@app.cell(hide_code=True)
def _(BitImportance, gain, mo, np, pl, shap, task, train):
    _s = shap.filter(
        (pl.col("task") == task.key)
        & (pl.col("bit") >= 0)
        & pl.col("id").is_in(train["id"].implode())
    )
    # the mean SHAP of each bit over the train compounds that set it
    _agg = _s.group_by("bit").agg(pl.col("shap").mean().alias("on"))
    _eff = np.zeros(2048)
    _eff[_agg["bit"].to_numpy()] = _agg["on"].to_numpy()
    _gain = gain.filter(pl.col("task") == task.key).sort("bit")
    mo.ui.anywidget(
        BitImportance(
            train["smiles"].to_list(),
            importance={
                "gain": _gain["gain"].to_numpy(),
                "mean |SHAP|": _gain["shap_abs"].to_numpy(),
            },
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
        label="test compound (largest difference from the NN first)",
    )
    mo.vstack(
        [
            mo.md(r"""
    1 つの予測を分解します。test 化合物と train の NN を並べています。原子の色は、bit の TreeSHAP 寄与をその bit に入る部分構造の原子に均等に割り振ったものです (赤は予測を上げ、青は下げます)。
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
                    "label": f"test · measured {_y[_ids[0]]:.2f} · predicted {_pred[_ids[0]]:.2f}",
                },
                {
                    "id": _ids[1],
                    "smiles": _smi[_ids[1]],
                    # no prediction here: the stored one is out of fold, the SHAP is not
                    "label": f"train · measured {_y[_ids[1]]:.2f}",
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
def _(mo, task):
    mo.md(f"""
    ---

    ## <span id="part-3"></span>第 3 部 · おまけ: データセットどうしの比較

    [第 2 部](#part-2)ではエンドポイントを 1 つ選んで中身を見てきました。最後におまけとして 16 のエンドポイントを並べて比べます。気になるエンドポイントがあれば、[上の選択](#picker)で切り替えて第 2 部のウィジェットで中身を見てください。図の中では、いま選んでいる {task.dataset} · {task.endpoint} を太字にしています。

    ### <span id="sec-3-1"></span>3.1 · 1 bit に入る部分構造

    [2.2](#sec-2-2) で見たデータセット全体の collision をデータセットごとに比べます。`bits` で bit 数を変えられます。`endpoint` に切り替えると、エンドポイントごとの分布を見られます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    load_unit = mo.ui.radio(["dataset", "endpoint"], value="dataset", inline=True, label="per")
    load_bits = mo.ui.radio(
        {"1024": 1024, "2048": 2048, "4096": 4096, "8192": 8192},
        value="2048",
        inline=True,
        label="bits",
    )
    return load_bits, load_unit


@app.cell(hide_code=True)
def _(bench, census_for, molecules, np, pl):
    # each dataset's whole train set, and every endpoint's train set
    train_sets = {}
    for _t in bench.TASKS:
        _smi = molecules.filter((pl.col("task") == _t.key) & (pl.col("split") == "train"))[
            "smiles"
        ].to_list()
        train_sets[("endpoint", f"{_t.dataset} · {_t.endpoint}")] = _smi
        train_sets.setdefault(("dataset", _t.dataset), []).extend(_smi)
    # a compound measured for several endpoints counts once
    train_sets = {k: list(dict.fromkeys(v)) for k, v in train_sets.items()}

    def bit_load_of(smiles, n_bits):
        """The 2.2 census of one train set: its numbers, and substructures on a bit → bits."""
        _c = census_for(smiles, 2, n_bits)
        _e = _c.n_envs
        _k, _n = np.unique(_e, return_counts=True)
        return {
            "n_train": len(smiles),
            "substructures": int(_e.sum()),
            "per_bit": float(_e[_e > 0].mean()),
            "empty_bits": int((_e == 0).sum()),
            "few": float((_e <= 5).mean()),  # share of bits with at most 5 substructures
            "purity": _c.purity()[1],
        }, pl.DataFrame({"envs": _k, "bits": _n})

    return bit_load_of, train_sets


@app.cell(hide_code=True)
def _(bit_load_of, load_bits, load_unit, mo, pl, train_sets):
    _rows, _hist = [], []
    with mo.status.spinner("部分構造を数えています…"):
        for (_unit, _name), _smi in train_sets.items():
            if _unit == load_unit.value:
                _row, _h = bit_load_of(_smi, load_bits.value)
                _rows.append({"name": _name, **_row})
                _hist.append(_h.with_columns(pl.lit(_name).alias("name")))
    bit_load = pl.DataFrame(_rows).sort("per_bit")
    bit_load_hist = pl.concat(_hist)
    return bit_load, bit_load_hist


@app.cell(hide_code=True)
def _(alt, bit_load, bit_load_hist, dataset_colours, load_bits, load_unit, mo, pl, task):
    _by_dataset = load_unit.value == "dataset"
    _sel = task.dataset if _by_dataset else f"{task.dataset} · {task.endpoint}"
    _load = bit_load
    _d = bit_load_hist
    _w, _h = (290, 220) if _by_dataset else (190, 130)
    _cols = 3 if _by_dataset else 4
    _top = int(_d["bits"].max())
    _n = int(_load["n_train"].max())
    _most = int(_d["envs"].max())

    def _panel(i, r):
        _first = i % _cols == 0
        # endpoint names are long: the dataset goes on the line below
        _ds_name, _, _head = r["name"].rpartition(" · ")
        _bars = (
            alt.Chart(
                _d.filter(pl.col("name") == r["name"]),
                title=alt.Title(
                    _head,
                    subtitle=_ds_name or alt.Undefined,
                    fontSize=14 if _by_dataset else 13,
                    fontWeight="bold" if r["name"] == _sel else "normal",
                    subtitleFontSize=11,
                    subtitleColor="#495057",
                    anchor="start",
                ),
            )
            .mark_bar(width={"band": 0.85})
            .encode(
                x=alt.X(
                    "envs:O",
                    title="substructures on a bit",
                    axis=alt.Axis(
                        values=list(range(0, _most + 1, 5 if _most <= 30 else 10)), labelAngle=0
                    ),
                    scale=alt.Scale(domain=list(range(_most + 1))),
                ),
                y=alt.Y(
                    "bits:Q",
                    title="bits" if _first else None,
                    scale=alt.Scale(domain=[0, _top]),
                    axis=alt.Axis(labels=_first, ticks=_first),
                ),
                # the dataset colours of 3.2
                color=alt.value(dataset_colours[_ds_name or _head]),
                tooltip=[
                    alt.Tooltip("envs:O", title="substructures on a bit"),
                    alt.Tooltip("bits:Q", title="bits"),
                ],
            )
        )
        # the numbers in a box in the empty top-right corner, clear of the grid lines
        _big, _small = (17, 12.5) if _by_dataset else (12.5, 11)
        _pad, _right = 7, _w - 6
        # wide enough for the longest line (about 0.6 em per character), the same in every panel
        _box_w = 2 * _pad + max(len(f"{_n:,} compounds") * _small, len("10.6 per bit") * _big) * 0.6
        _lines = [
            (f"{r['per_bit']:.1f} per bit", _big, "bold", 6 + _pad),
            (f"purity {r['purity']:.0%}", _small, "normal", 6 + _pad + _big + 4),
            (f"{r['n_train']:,} compounds", _small, "normal", 6 + _pad + _big + _small + 8),
        ]
        _box = (
            alt.Chart(pl.DataFrame({"x": [0]}))
            .mark_rect(fill="white", stroke="#ced4da", strokeWidth=1, cornerRadius=4, opacity=0.95)
            .encode(
                x=alt.value(_right - _box_w),
                x2=alt.value(_right),
                y=alt.value(6),
                y2=alt.value(6 + 2 * _pad + _big + 2 * _small + 8),
            )
        )
        _labels = [
            alt.Chart(pl.DataFrame({"t": [t]}))
            .mark_text(
                align="right", baseline="top", fontSize=size, fontWeight=weight, color="#212529"
            )
            .encode(x=alt.value(_right - _pad), y=alt.value(y), text="t:N")
            for t, size, weight, y in _lines
        ]
        return alt.layer(_bars, _box, *_labels).properties(width=_w, height=_h)

    _chart = alt.concat(
        *[_panel(i, r) for i, r in enumerate(_load.iter_rows(named=True))],
        columns=_cols,
        spacing=24 if _by_dataset else 18,
    )
    mo.vstack(
        [
            mo.hstack([load_unit, load_bits], justify="start", gap=2),
            mo.hstack([_chart], justify="center"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(bit_load_of, mo, train_sets):
    # the text is about 2048 bits, whatever the switch above shows
    _at = {
        d: bit_load_of(train_sets[("dataset", d)], 2048)[0] for d in ("PXR", "ASAP", "ExpansionRx")
    }
    _pxr, _asap, _exp = _at["PXR"], _at["ASAP"], _at["ExpansionRx"]
    _pxr_long = bit_load_of(train_sets[("dataset", "PXR")], 8192)[0]
    mo.md(f"""
    bit 数が 2048 の場合、次のような傾向が見られます。

    * **PXR**
        * 空の bit がない
        * 1 bit に入る部分構造がいちばん多い (平均 {_pxr["per_bit"]:.1f} 種類)
        * purity がいちばん低い ({_pxr["purity"]:.0%})
    * **ASAP と ExpansionRx**
        * ほとんどの bit は部分構造が 5 種類以下 (ASAP {_asap["few"]:.0%}、ExpansionRx {_exp["few"]:.0%})。PXR では {_pxr["few"]:.0%} だけ
        * ExpansionRx は PXR より化合物が多いのに、部分構造の種類は PXR の {_exp["substructures"] / _pxr["substructures"]:.0%} ほど

    PXR の化合物は構造がかなり多様だとわかります。bit 数を増やすと分布は左 (0 の側) に寄りますが、8192 bit にしても PXR は 1 bit に平均 {_pxr_long["per_bit"]:.1f} 種類の部分構造が入ります。

    1 つの bit にどんな部分構造が入っているかは、[2.2](#sec-2-2) の表で見られます。[上の選択](#picker)で PXR と ASAP を切り替えると違いがわかります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-2"></span>3.2 · collision と精度

    では、bit 数を増やして collision を減らすと、モデルの精度は良くなるでしょうか。ECFP4 bit のモデルを bit 数ごとに学習し直しました。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # checks or unchecks every endpoint below
    bitlen_all = mo.ui.checkbox(value=True, label="all")
    return (bitlen_all,)


@app.cell(hide_code=True)
def _(bitlen_all, mo):
    dataset_colours = {"ASAP": "#0ca678", "ExpansionRx": "#7048e8", "PXR": "#e8590c"}
    # the same per dataset (rebuilt when the overall "all" flips)
    bitlen_all_of = mo.ui.dictionary(
        {d: mo.ui.checkbox(value=bitlen_all.value, label="all") for d in dataset_colours}
    )
    return bitlen_all_of, dataset_colours


@app.cell(hide_code=True)
def _(bench, bitlen_all_of, mo):
    # one checkbox per endpoint: which lines the charts below show (rebuilt when an "all" flips)
    bitlen_show = mo.ui.dictionary(
        {
            t.key: mo.ui.checkbox(value=bitlen_all_of.value[t.dataset], label=t.endpoint)
            for t in bench.TASKS
        }
    )
    # what the two charts show: one fingerprint measure, one test score
    bitlen_fp = mo.ui.radio(
        ["substructures per bit", "purity", "empty bits"],
        value="substructures per bit",
        inline=True,
        label="fingerprint",
    )
    bitlen_score = mo.ui.radio(
        ["Spearman ρ", "R²", "MAE"], value="Spearman ρ", inline=True, label="test score"
    )
    return bitlen_fp, bitlen_score, bitlen_show


@app.cell(hide_code=True)
def _(
    alt,
    bench,
    bitlen,
    bitlen_all,
    bitlen_all_of,
    bitlen_fp,
    bitlen_score,
    bitlen_show,
    dataset_colours,
    mo,
    pl,
    task,
    tasks,
):
    # the ECFP4 bit model refitted at each length (precomputed), one line per endpoint
    _d = (
        bitlen.join(tasks.select("task", "dataset", "endpoint"), on="task")
        .with_columns(
            (pl.col("dataset") + " · " + pl.col("endpoint")).alias("name"),
            (pl.col("task") == task.key).alias("selected"),
            (pl.col("empty_bits") / pl.col("n_bits")).alias("empty"),
        )
        .filter(pl.col("task").is_in([k for k, on in bitlen_show.value.items() if on]))
    )
    _colour = alt.Color(
        "dataset:N",
        scale=alt.Scale(domain=list(dataset_colours), range=list(dataset_colours.values())),
        legend=None,
    )

    def _lines(field, title, fmt, domain=None):
        _base = alt.Chart(_d, title=alt.Title(title, fontSize=13, anchor="start")).encode(
            x=alt.X("n_bits:O", title="bits", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                f"{field}:Q",
                title=None,
                scale=alt.Scale(domain=domain) if domain else alt.Scale(zero=False),
                axis=alt.Axis(format=fmt),
            ),
            color=_colour,
            detail="name:N",
            tooltip=[
                alt.Tooltip("name:N", title="endpoint"),
                alt.Tooltip("n_bits:O", title="bits"),
                alt.Tooltip(f"{field}:Q", title=title, format=fmt),
            ],
        )
        # the selected endpoint thicker, on top
        return (
            _base.mark_line(point=alt.OverlayMarkDef(size=25), strokeWidth=1.5).transform_filter(
                "!datum.selected"
            )
            + _base.mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=3.5).transform_filter(
                "datum.selected"
            )
        ).properties(width=400, height=260)

    # the checkboxes as a two-column grid: the dataset (in the colour of its lines), then its
    # endpoints, which wrap inside their own column
    _rows = "".join(
        f'<div style="font-weight:600;white-space:nowrap"><span style="color:{c}">●</span> {d}</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:4px 18px">'
        + f'<span style="margin-right:10px">{bitlen_all_of[d]}</span>'
        + "".join(f"{bitlen_show[t.key]}" for t in bench.TASKS if t.dataset == d)
        + "</div>"
        for d, c in dataset_colours.items()
    )
    _boxes = mo.Html(
        '<div style="display:grid;grid-template-columns:max-content 1fr;gap:8px 20px;'
        "align-items:start;padding:10px 14px;border:1px solid var(--slate-4, #e5e7eb);"
        f'border-radius:8px;font-size:0.9rem"><div></div><div>{bitlen_all}</div>{_rows}</div>'
    )
    _fp = {
        "substructures per bit": ("per_bit", "substructures per bit", ".1f"),
        "purity": ("purity", "purity", ".0%"),
        "empty bits": ("empty", "empty bits", ".0%", [0, 1]),
    }
    _score = {
        "Spearman ρ": ("rho", "Spearman ρ (test)", ".2f", [0, 1]),
        "R²": ("r2", "R² (test)", ".2f"),
        "MAE": ("mae", "MAE (test, lower is better)", ".2f"),
    }
    # share of empty bits at the longest length, median over the endpoints
    _empty_long = (
        bitlen.filter(pl.col("n_bits") == 8192)
        .select((pl.col("empty_bits") / pl.col("n_bits")).median())
        .item()
    )
    _pxr = {
        r["n_bits"]: r for r in bitlen.filter(pl.col("task") == "pxr/pEC50").iter_rows(named=True)
    }
    mo.vstack(
        [
            mo.md(
                f"線は 1 本が 1 つのエンドポイントで、太い線はいま選んでいる {task.dataset} · {task.endpoint} です。"
            ),
            _boxes,
            # one fingerprint measure beside one test score, each under its own switch
            mo.hstack(
                [
                    mo.vstack([bitlen_fp, _lines(*_fp[bitlen_fp.value])], align="center"),
                    mo.vstack([bitlen_score, _lines(*_score[bitlen_score.value])], align="center"),
                ],
                justify="center",
                gap=2,
            ),
            mo.md(
                f"""
    * 1 bit に入る部分構造は減少し、purity は上昇する
    * 空の bit も増加する。bit 数を増やしても、その分だけ情報が増えるわけではない (8192 bit では 16 のエンドポイントの中央値で {_empty_long:.0%} が空)
    * モデルの精度はある程度の bit 数で頭打ちになる
        * PXR と、ExpansionRx の化合物が多いエンドポイント: 2048 bit あたり
        * ASAP: 1024 bit から変わらない

    bit 数を増やすと、重なっていた部分構造が別々の bit に分かれ、fingerprint の情報は増えます。それでも精度が頭打ちになる理由はいろいろ考えられますが、増えた情報が予測の精度にあまりつながっていないのだと思います。

    * 分かれるのはほとんどが珍しい部分構造。PXR では {_pxr[2048]["substructures"]:,} 種類のうち 20 化合物以上に出るのは {_pxr[2048]["frequent_substructures"]:,} 種類で、数個の化合物からはその部分構造が値に与える影響を学習しにくい (化合物の数は [2.2](#sec-2-2) の表で bit を選ぶと確認できる)
    * 予測が外れる主な原因は別にある。test が train から遠いこと ([2.4](#sec-2-4)) や activity cliff ([2.5](#sec-2-5)) は、bit 数を増やしても解消しない
    """
            ),
            mo.callout(
                mo.md(
                    "今回の設定 (`min_child_samples=20`) では、LightGBM は両側に 20 化合物以上が残る分岐しか"
                    "作りません。珍しい部分構造は、別の bit に分かれても予測に使われないままです。"
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-3"></span>3.3 · train との近さと精度

    test 化合物に似た化合物が train にあれば、その測定値は予測の手がかりになります。[2.4](#sec-2-4) の NN との類似度と [2.6](#sec-2-6) の予測の精度を、16 のエンドポイントで並べます。
    """)
    return


@app.cell
def _(db, mo):
    nn_by_task = mo.sql(
        """
        SELECT
            t.dataset || ' · ' || t.endpoint AS name, n.task,
            min(n.tanimoto) AS lo, quantile_cont(n.tanimoto, 0.25) AS q1,
            median(n.tanimoto) AS med, quantile_cont(n.tanimoto, 0.75) AS q3, max(n.tanimoto) AS hi,
            avg((n.tanimoto >= 0.6)::INT) AS share_close
        FROM neighbours n JOIN tasks t USING (task)
        GROUP BY ALL
        ORDER BY med DESC
        """,
        output=False,
        engine=db,
    )
    return (nn_by_task,)


@app.cell
def _(db, mo):
    # every score of every model, one row each; best marks the top model per endpoint
    # (the lowest for MAE). NN value's R² and MAE come from the neighbours table
    scores = mo.sql(
        """
        WITH nn AS (
            SELECT
                n.task,
                1 - sum(power(n.y_nn - n.y_test, 2)) / sum(power(n.y_test - a.mean, 2)) AS r2,
                avg(abs(n.y_nn - n.y_test)) AS mae
            FROM neighbours n
            JOIN (SELECT task, avg(y_test) AS mean FROM neighbours GROUP BY task) a USING (task)
            GROUP BY n.task
        ),
        long AS (
            UNPIVOT (
                SELECT
                    m.task, m.rho_1nn AS "rho NN value", nn.r2 AS "r2 NN value",
                    nn.mae AS "mae NN value", COLUMNS('^(rho|r2|mae|pair_slope) ')
                FROM metrics m JOIN nn USING (task)
            )
            ON COLUMNS(* EXCLUDE task) INTO NAME metric VALUE score
        )
        SELECT
            t.dataset || ' · ' || t.endpoint AS name, l.task,
            split_part(l.metric, ' ', 1) AS stat,
            substr(l.metric, strpos(l.metric, ' ') + 1) AS model,
            l.score,
            CASE WHEN stat = 'mae'
                THEN l.score = min(l.score) OVER (PARTITION BY l.task, stat)
                ELSE l.score = max(l.score) OVER (PARTITION BY l.task, stat)
            END AS best
        FROM long l JOIN tasks t USING (task)
        """,
        output=False,
        engine=db,
    )
    return (scores,)


@app.cell(hide_code=True)
def _(alt, nn_by_task, task):
    # rows in the same order in every chart of Part 3; the selected endpoint in bold
    rows = nn_by_task["name"].to_list()
    _sel = f"{task.dataset} · {task.endpoint}"
    row_axis = alt.Axis(
        labelFontWeight=alt.expr(f"datum.value == '{_sel}' ? 'bold' : 'normal'"),
        labelLimit=260,
    )
    return row_axis, rows


@app.cell(hide_code=True)
def _(alt, mo, neighbours, nn_by_task, pl, predictions, row_axis, rows, scores):
    _y = alt.Y("name:N", sort=rows, title=None, axis=row_axis)
    _x = alt.X("lo:Q", title="Tanimoto to the NN (ECFP4)", scale=alt.Scale(domain=[0, 1]))
    _base = alt.Chart(nn_by_task).encode(y=_y)
    _box = (
        _base.mark_rule(color="#868e96").encode(x=_x, x2="hi:Q")
        + _base.mark_bar(size=12, color="#adb5bd").encode(
            x="q1:Q",
            x2="q3:Q",
            tooltip=[
                "name",
                alt.Tooltip("med:Q", title="median", format=".2f"),
                alt.Tooltip("q1:Q", format=".2f"),
                alt.Tooltip("q3:Q", format=".2f"),
                alt.Tooltip("share_close:Q", title="NN ≥ 0.6", format=".0%"),
            ],
        )
        + _base.mark_tick(color="#212529", size=12, thickness=2).encode(x="med:Q")
    ).properties(
        width=300, height=380, title=alt.Title("similarity to the NN", fontSize=12, anchor="start")
    )
    # NN value → ECFP4 bit, per endpoint
    _two = scores.filter(
        (pl.col("stat") == "rho") & pl.col("model").is_in(["NN value", "ECFP4 bit"])
    )
    _dy = alt.Y(
        "name:N", sort=rows, title=None, axis=alt.Axis(labels=False, ticks=False, domain=False)
    )
    _db = alt.Chart(_two).encode(y=_dy)
    _dumbbell = (
        _db.mark_line(color="#ced4da", strokeWidth=2).encode(x="score:Q", detail="name:N")
        + _db.mark_circle(size=80, opacity=1).encode(
            x=alt.X("score:Q", title="Spearman ρ (test)", scale=alt.Scale(domain=[-0.1, 1])),
            color=alt.Color(
                "model:N",
                # the colours of 2.6
                scale=alt.Scale(domain=["NN value", "ECFP4 bit"], range=["#adb5bd", "#4c78a8"]),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["name", "model", alt.Tooltip("score:Q", title="ρ", format=".2f")],
        )
    ).properties(
        width=300, height=380, title=alt.Title("NN value → ECFP4 bit", fontSize=12, anchor="start")
    )
    # how far the model's rho is above NN value, per endpoint
    _rho = scores.filter(pl.col("stat") == "rho").pivot(on="model", index="task", values="score")
    _gain = dict(_rho.select("task", pl.col("ECFP4 bit") - pl.col("NN value")).iter_rows())
    # the model's test error for compounds with and without a close NN, per endpoint
    _err = (
        neighbours.join(
            predictions.filter(
                (pl.col("features") == "ECFP4 bit") & (pl.col("split") == "test")
            ).select("task", pl.col("id").alias("test_id"), "pred"),
            on=["task", "test_id"],
        )
        .group_by("task", (pl.col("tanimoto") >= 0.6).alias("close"))
        .agg((pl.col("pred") - pl.col("y_test")).abs().mean().alias("mae"))
        .pivot(on="close", index="task", values="mae")
    )
    _closer_better = int((_err["true"] < _err["false"]).sum())
    _pxr = _err.filter(pl.col("task") == "pxr/pEC50").row(0, named=True)
    _pxr_mae = {True: _pxr["true"], False: _pxr["false"]}
    mo.vstack(
        [
            mo.md(
                """
    * 左: test 化合物と NN の類似度
    * 右: 精度の比較 (Spearman ρ)。NN の測定値をそのまま予測値にした場合 (NN value) と、ECFP4 bit のモデル
    """
            ),
            mo.hstack([alt.hconcat(_box, _dumbbell, spacing=24)], justify="center"),
            mo.md(
                f"""
    * ASAP の pIC50: test のほぼすべてに似た NN がある。NN value だけで ρ は 0.61 と 0.75
    * PXR: 似た NN がほとんどない。NN value の ρ は 0.05 だがモデルは 0.61
    * ASAP の KSOL: モデルが NN value を下回る。測定値が上限付近に集まっていて順位がつきにくい

    エンドポイントどうしで比べると、NN との類似度が高いほど NN value の ρ は高くなります。モデルが NN value を上回る幅は、類似度が低いエンドポイントほど大きくなります (PXR で {_gain["pxr/pEC50"]:+.2f}、ASAP の pIC50 で {_gain["asap/mers"]:+.2f} と {_gain["asap/sars2"]:+.2f})。

    同じエンドポイントの中でも、似た NN がある test 化合物のほうがモデルの誤差は小さく、16 のうち {_closer_better} のエンドポイントでそうなっています。PXR では類似度が 0.6 以上の化合物の MAE は {_pxr_mae[True]:.2f}、0.6 未満は {_pxr_mae[False]:.2f} です。

    test 化合物それぞれの NN とその構造は [2.4](#sec-2-4) で見られます。
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-4"></span>3.4 · 特徴量と精度

    [2.6](#sec-2-6) の 5 つの予測を 16 のエンドポイントで並べます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    score_pick = mo.ui.radio(
        {"Spearman ρ": "rho", "R²": "r2", "MAE": "mae"},
        value="Spearman ρ",
        inline=True,
        label="test score",
    )
    score_view = mo.ui.radio(
        {"score": "score", "Δ from ECFP4 bit": "delta"}, value="score", inline=True, label="show"
    )
    return score_pick, score_view


@app.cell(hide_code=True)
def _(alt, pl, row_axis, rows, scores):
    def score_heat(stat, models, title, scheme="viridis", best=None, baseline=None):
        """Endpoints × models, each cell coloured from the worst to the best score of the table
        (the lowest is best for MAE); ``best`` adds a column naming the best model of each row.

        ``baseline``: a model name; the cells then show each score minus that model's score in
        the same row, on a diverging scale (blue = better than the baseline, red = worse)."""
        _d = scores.filter((pl.col("stat") == stat) & pl.col("model").is_in(models))
        _flip = -1 if stat == "mae" else 1
        if baseline is None:
            _lo, _hi = _d["score"].min(), _d["score"].max()
            _shade = (pl.col("score") - _lo) / (_hi - _lo)
            _d = _d.with_columns(
                pl.col("score").alias("shown"),
                (1 - _shade if stat == "mae" else _shade).alias("shade"),
            )
            _scale, _fmt = alt.Scale(domain=[0, 1], scheme=scheme), ".2f"
            # dark cells get white numbers; viridis is dark at the low end, the others at the high
            _dark = "datum.shade < 0.62" if scheme == "viridis" else "datum.shade > 0.6"
        else:
            _base_score = _d.filter(pl.col("model") == baseline).select(
                "task", pl.col("score").alias("base")
            )
            _d = _d.join(_base_score, on="task").with_columns(
                (pl.col("score") - pl.col("base")).alias("shown")
            )
            _d = _d.with_columns((_flip * pl.col("shown")).alias("shade"))
            _most = float(_d["shade"].abs().max())
            _scale, _fmt = alt.Scale(domain=[-_most, 0, _most], scheme="redblue"), "+.2f"
            _dark = f"abs(datum.shade) > {0.6 * _most}"
        _base = alt.Chart(_d).encode(
            x=alt.X("model:N", sort=models, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
            y=alt.Y("name:N", sort=rows, title=None, axis=row_axis),
        )
        _rect = _base.mark_rect().encode(
            color=alt.Color("shade:Q", scale=_scale, legend=None),
            tooltip=["name", "model", alt.Tooltip("score:Q", title=title, format=".2f")],
        )
        _ink = alt.condition(_dark, alt.value("white"), alt.value("#212529"))
        # the best model of each row in bold
        _text = [
            _base.mark_text(fontSize=11, fontWeight=w)
            .transform_filter(f"{'' if b else '!'}datum.best")
            .encode(text=alt.Text("shown:Q", format=_fmt), color=_ink)
            for b, w in ((True, "bold"), (False, "normal"))
        ]
        _heat = alt.layer(_rect, *_text).properties(width=82 * len(models), height=380)
        if best is None:
            return _heat
        _name = (
            alt.Chart(_d.filter(pl.col("best")), title=alt.Title("best", fontSize=11))
            .mark_text(align="left", fontSize=12, fontWeight="bold")
            .encode(
                x=alt.value(6),
                y=alt.Y("name:N", sort=rows, title=None, axis=None),
                text="model:N",
                color=alt.Color("model:N", scale=best, legend=None),
            )
            .properties(width=90, height=380)
        )
        return alt.hconcat(_heat, _name, spacing=4).resolve_scale(color="independent")

    return (score_heat,)


@app.cell(hide_code=True)
def _(alt, mo, score_heat, score_pick, score_view):
    _models = ["NN value", "ECFP4 bit", "ECFP4 count", "RDKit desc", "bit + desc"]
    # the model colours of 2.6
    _colours = alt.Scale(
        domain=_models, range=["#adb5bd", "#4c78a8", "#f58518", "#e45756", "#72b7b2"]
    )
    _title = {"rho": "Spearman ρ", "r2": "R²", "mae": "MAE"}[score_pick.value]
    mo.vstack(
        [
            mo.hstack([score_pick, score_view], justify="start", gap=2),
            mo.hstack(
                [
                    score_heat(
                        score_pick.value,
                        _models,
                        _title,
                        best=_colours,
                        baseline="ECFP4 bit" if score_view.value == "delta" else None,
                    )
                ],
                justify="center",
            ),
            mo.md(
                """
    ECFP4 bit だけのモデルがいちばん良いエンドポイントはほとんどありませんでした。多くのエンドポイントで、記述子を足すと精度が上がります。

    ただし、上がり幅はエンドポイントごとに違います。同じ LogD でも、ECFP4 bit から bit + desc にしたときの ρ の上がり幅は ASAP で +0.42、ExpansionRx で +0.03 です。どの特徴量が合うかは、測る値の種類だけでなくデータセットによっても変わります。特徴量や設定を変えて比べると、そのデータセットの傾向が見えてきます。

    これは ECFP4 の精度が悪いという話ではありません。単独では精度が低いモデルでも、ほかのモデルと違う外し方をしているなら、アンサンブルで精度が上がる見込みがあります。選んだエンドポイントでどの化合物の予測が外れているかは、[2.6](#sec-2-6) の散布図で見られます。
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### <span id="sec-3-5"></span>3.5 · activity cliff と精度

    [2.5](#sec-2-5) の activity cliff は、NN と構造が似ているのに値が大きく違う test 化合物でした。その化合物でモデルの誤差がどうなるかを、16 のエンドポイントで比べます。比べる相手は、同じく NN との類似度が 0.6 以上で activity cliff ではない化合物です。
    """)
    return


@app.cell
def _(bench, neighbours, np, pl, predictions):
    # test compounds with a close NN, split as in 2.5 into activity cliffs and the rest, and
    # each model's test error (MAE) on the two groups
    _parts = []
    for _t in bench.TASKS:
        _nb = neighbours.filter(pl.col("task") == _t.key)
        _dy = (_nb["y_test"] - _nb["y_nn"]).abs().to_numpy()
        _random = np.abs(
            np.random.default_rng(0).permutation(_nb["y_nn"].to_numpy()) - _nb["y_test"].to_numpy()
        ).mean()
        _parts.append(
            _nb.with_columns(pl.Series("cliff", _dy >= _random)).filter(pl.col("tanimoto") >= 0.6)
        )
    cliff_error = (
        pl.concat(_parts)
        .join(
            predictions.filter(pl.col("split") == "test").select(
                "task", pl.col("id").alias("test_id"), "features", "pred"
            ),
            on=["task", "test_id"],
        )
        .group_by("task", "features", "cliff")
        .agg(pl.len().alias("n"), (pl.col("pred") - pl.col("y_test")).abs().mean().alias("mae"))
    )
    return (cliff_error,)


@app.cell(hide_code=True)
def _(bench, mo):
    cliff_model = mo.ui.radio(bench.FEATURES, value="ECFP4 bit", inline=True, label="model")
    return (cliff_model,)


@app.cell(hide_code=True)
def _(alt, cliff_error, cliff_model, mo, pl, row_axis, rows, tasks):
    _d = (
        cliff_error.filter(pl.col("features") == cliff_model.value)
        .join(tasks.select("task", "dataset", "endpoint"), on="task")
        .with_columns(
            (pl.col("dataset") + " · " + pl.col("endpoint")).alias("name"),
            pl.when(pl.col("cliff"))
            .then(pl.lit("activity cliff"))
            .otherwise(pl.lit("other compounds with a close NN"))
            .alias("group"),
        )
    )
    _base = alt.Chart(_d).encode(y=alt.Y("name:N", sort=rows, title=None, axis=row_axis))
    _chart = (
        _base.mark_line(color="#ced4da", strokeWidth=2).encode(x="mae:Q", detail="name:N")
        + _base.mark_circle(size=90, opacity=1).encode(
            x=alt.X("mae:Q", title=f"MAE (test, {cliff_model.value})"),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(
                    domain=["other compounds with a close NN", "activity cliff"],
                    range=["#adb5bd", "#d6336c"],
                ),
                title=None,
                legend=alt.Legend(orient="top", labelLimit=300),
            ),
            tooltip=[
                "name",
                "group",
                alt.Tooltip("n:Q", title="compounds"),
                alt.Tooltip("mae:Q", title="MAE", format=".2f"),
            ],
        )
    ).properties(width=460, height=380)
    _wide = _d.pivot(on="cliff", index="task", values="mae")
    _ratio = _wide["true"] / _wide["false"]
    _larger = int((_ratio > 1).sum())
    mo.vstack(
        [
            cliff_model,
            mo.hstack([_chart], justify="center"),
            mo.md(
                f"""
    activity cliff の化合物では、16 のうち {_larger} のエンドポイントで誤差が大きくなります。誤差の大きさは、それ以外の化合物の {_ratio.median():.1f} 倍です (中央値)。特徴量を変えてもこの傾向は変わりません。

    activity cliff のペアの構造と、2 つの間で違う bit は [2.5](#sec-2-5) で見られます。
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## <span id="summary"></span>まとめ

    この notebook では、ECFP4 を数字だけでなく中身から見てきました。なんとなく ECFP4 を使っていた方には、初めて知ることも多かったのではないでしょうか。構造を実際に目で見ると、仕組みがつかみやすくなります。

    [第 2 部](#part-2)と[第 3 部](#part-3)では、データセットの中での ECFP4 を見ました。興味深いのは、collision の数などがデータセットごとに大きく違うことです。これは ECFP4 を使うだけでは見えてきません。特に PXR の中身はほかの 2 つとかなり違っていました。私も PXR のチャレンジに参加しましたが、そのときはこの違いに気づいていませんでした。

    モデルを学習したあとに特徴量重要度を確かめるのはよくある手順です。ただ fingerprint では bit が何を表すのかがわかりにくく、確かめづらい面がありました。[2.7](#sec-2-7) のウィジェットでは、重要な bit の部分構造とその bit が立っている化合物をその場で見られます。見てみると、重要な bit が意味のありそうな構造だとは限らず、むしろよくある部分構造であることも多いとわかります (データセットによります)。複数のデータセットで見比べると違いがよくわかります。

    この notebook の結果は、今回のデータセットでのものです。一般に成り立つとは言い切れません。ぜひこの notebook とウィジェット ([`chemari`](https://github.com/N283T/chemari)) を使って、ECFP4 への理解を深めたり、ご自身のデータセットで ECFP4 を調べたりしてみてください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## この notebook について

    * **データ**: [PXR challenge](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test) (CC-BY-4.0)、[ASAP-Polaris-OpenADMET antiviral challenge](https://huggingface.co/datasets/openadmet/ASAP_Polaris_OpenADMET_challenge) (MIT)、[OpenADMET-ExpansionRx challenge](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) (CC-BY-4.0)
    * **事前計算**: `dev/precompute.py` → `results/precomputed/`。計算のコードは `chemari.examples.openadmet`
    * **ウィジェット**: この notebook のウィジェット (`ECFPMovie`、`ECFPStepper`、`MorganBitTiles`、`MolGrid`、`MolPair`、`BitAtlas`、`BitImportance`、`MolScatter`、`MorganExplorer`) は、[CheMari](https://github.com/N283T/chemari) というパッケージにまとめて公開しています。今後は PyPI での公開や、ほかのウィジェットの開発も検討しています
    * **AI の利用**: ウィジェット、動画、notebook の骨組みのコーディングには Claude (Anthropic) をアシスタントとして使いました。問いの立て方、解析の選び方、解釈は私自身のものです

    ## 参考文献

    1. <span id="ref-1"></span>Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742–754. [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
    2. <span id="ref-2"></span>Virany, W.; Tripp, A. Hash Collisions in Molecular Fingerprints: Effects on Property Prediction and Bayesian Optimization. AI for Science workshop, NeurIPS 2025. [arXiv:2511.17078](https://arxiv.org/abs/2511.17078) (collision が類似度を高めに見せることと、予測への影響)
    3. <span id="ref-3"></span>Gütlein, M.; Kramer, S. Filtered circular fingerprints improve either prediction or runtime performance while retaining interpretability. *J. Cheminform.* **2016**, 8, 60. [doi:10.1186/s13321-016-0173-z](https://doi.org/10.1186/s13321-016-0173-z) (1 bit あたりの部分構造の数を bit-load と呼んでいる)
    4. <span id="ref-4"></span>Johnson, M. A.; Maggiora, G. M. (eds.) *Concepts and Applications of Molecular Similarity*. Wiley, **1990**
    5. <span id="ref-5"></span>Maggiora, G. M. On Outliers and Activity Cliffs — Why QSAR Often Disappoints. *J. Chem. Inf. Model.* **2006**, 46, 1535. [doi:10.1021/ci060117s](https://doi.org/10.1021/ci060117s)
    6. <span id="ref-6"></span>Schönherr, H.; Cernak, T. Profound Methyl Effects in Drug Discovery and a Call for New C–H Methylation Reactions. *Angew. Chem. Int. Ed.* **2013**, 52, 12256–12267. [doi:10.1002/anie.201303207](https://doi.org/10.1002/anie.201303207)
    """)
    return


if __name__ == "__main__":
    app.run()
