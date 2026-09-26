"""Generate notebooks/ecfp_inside_ja.py from the English notebook.

Every English passage is replaced by a Japanese one; the run fails if any passage
is missing, so edit the English notebook first and then update the pairs here.

    uv run python scripts/translate_ecfp_inside_ja.py && uv run ruff format notebooks
"""

from pathlib import Path

src = Path("notebooks/ecfp_inside.py").read_text()

R = []  # (old, new)


def r(old, new):
    R.append((old, new))


r(
    '''app_title="Inside ECFP4: a hands-on guide to Morgan fingerprints"''',
    '''app_title="ECFP4 の中身: Morgan fingerprint ハンズオンガイド"''',
)
r(
    """    # Inside ECFP4
    ### A hands-on guide to Morgan fingerprints — what they encode, what they lose, and what that means for your model

    If you have trained a QSAR model, you have probably typed something like""",
    """    # ECFP4 の中身
    ### Morgan fingerprint ハンズオンガイド — 何を表し、何を失い、それがモデルにとって何を意味するか

    QSAR モデルを作ったことがあれば、きっと一度はこう書いたはずです""",
)
r(
    """    and moved on. ECFP4 is the default molecular representation in cheminformatics: it powers
    similarity search, clustering, library design and countless ML baselines. It even defined the
    test set of the OpenADMET PXR challenge: the 513 test compounds were bought from Enamine
    because their **ECFP4 Tanimoto similarity to a potent, selective PXR hit was above 0.4**
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)).

    Yet few people could say what bit 1380 of that vector *means*, or how many different
    substructures share it. This notebook opens the box:

    1. **The algorithm** — from atoms to bits, one step at a time
    2. **Collisions** — what folding does, drawn bit by bit
    3. **Blind spots** — counts, stereochemistry, ring size
    4. **Similarity** — what "Tanimoto > 0.4" does and does not promise
    5. **ECFP inside a model** — feature importance and per-atom attributions
    6. **A cheat sheet** of strengths, weaknesses and sensible defaults

    Examples come from the OpenADMET PXR induction dataset. The interactive pieces are custom
    [anywidget](https://anywidget.dev) components from the `molwidgets` package written for this notebook.

    /// admonition | Names you will see
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) and the **Morgan fingerprint**
    (RDKit's name, after Morgan's 1965 canonicalization algorithm) are the same idea. The number in
    ECFP*n* is the *diameter*: **ECFP4 = Morgan radius 2**, ECFP6 = radius 3. **FCFP** uses
    pharmacophoric atom features (donor, acceptor, aromatic, …) instead of element-level invariants.
    ///""",
    """    そして、そのまま先に進んだのではないでしょうか。ECFP4 はケモインフォマティクスの標準的な分子表現で、
    類似検索、クラスタリング、ライブラリ設計、数えきれないほどの ML ベースラインを支えています。
    OpenADMET の PXR チャレンジでは test set の定義にまで使われました。Test の 513 化合物は、
    **強活性かつ選択的な PXR ヒットとの ECFP4 Tanimoto 類似度が 0.4 を超える** という基準で Enamine から
    購入されたものです
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/))。

    それなのに、そのベクトルの bit 1380 が何を *意味するか*、そこに何種類の部分構造が同居しているかを
    答えられる人はほとんどいません。この notebook では、その箱を開けてみます:

    1. **アルゴリズム** — 原子から bit まで 1 ステップずつ
    2. **Collision** — fold が何をするかを bit ごとに描いて見る
    3. **盲点** — 数 (count)、立体化学、環サイズ
    4. **類似度** — 「Tanimoto > 0.4」が約束すること、しないこと
    5. **モデルの中の ECFP** — feature importance と原子ごとの寄与
    6. 長所・短所・おすすめ設定の **チートシート**

    例には OpenADMET の PXR induction データセットを使います。インタラクティブな部品は、この notebook の
    ために作った [anywidget](https://anywidget.dev) ベースの自作パッケージ `molwidgets` です。

    /// admonition | 出てくる名前
    **ECFP** (Extended-Connectivity FingerPrint, Rogers & Hahn 2010) と **Morgan fingerprint**
    (RDKit での呼び名。Morgan が 1965 年に発表した canonicalization アルゴリズムに由来) は同じものです。
    ECFP*n* の数字は *直径* を表し、**ECFP4 = Morgan radius 2**、ECFP6 = radius 3 です。**FCFP** は元素
    レベルの invariant の代わりに、薬理作用団的な原子の特徴 (donor, acceptor, aromatic, …) を使います。
    ///""",
)
r(
    """        "isobutane (start here)": "CC(C)C",
        "paracetamol (small, symmetric ring)": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · potent PXR agonist (pEC50 5.95)": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · its pyridine analogue (pEC50 2.06)": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",""",
    """        "isobutane (ここから)": "CC(C)C",
        "paracetamol (小さく、対称な環)": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · 強活性の PXR agonist (pEC50 5.95)": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · その pyridine analogue (pEC50 2.06)": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",""",
)
r(
    '''EXAMPLES, value="isobutane (start here)", label="molecule"''',
    '''EXAMPLES, value="isobutane (ここから)", label="分子"''',
)
r(
    '''placeholder="…or paste a SMILES"''',
    '''placeholder="…または SMILES を貼り付け"''',
)
r(
    """mo.md(f"`{_smi}` is not a valid SMILES.")""",
    """mo.md(f"`{_smi}` は正しい SMILES ではありません。")""",
)
r(
    """share of molecules""",
    """分子の割合""",
)
r(
    """(_with_collision(n), "PXR training set"), (_theory(n), "birthday formula")""",
    """(_with_collision(n), "PXR training set"), (_theory(n), "誕生日問題の式")""",
)
r(
    '''title="molecules with ≥ 1 internal collision"''',
    '''title="分子内 collision が 1 回以上ある分子"''',
)
r(
    '''label="median distinct ECFP4 features per molecule"''',
    '''label="1 分子あたりの distinct な ECFP4 特徴 (中央値)"''',
)
r(
    '''label=f"molecules with an internal collision at {fold_pick.value} bits"''',
    '''label=f"{fold_pick.value} bit で分子内 collision がある分子"''',
)
r(
    '''                        "The simple formula tracks the real data closely. Doubling the length roughly "
                        "halves the collision rate; you need ~16k bits before collisions become rare."''',
    '''                        "単純な式が実データをよく再現しています。長さを 2 倍にすると collision 率は"
                        "おおよそ半分になり、collision がまれになるには ~16k bit が必要です。"''',
)
r(
    """        "ibuprofen (start here)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · potent PXR agonist": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · its pyridine analogue": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",""",
    """        "ibuprofen (ここから)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "paracetamol": "CC(=O)Nc1ccc(O)cc1",
        "OADMET-0002810 · 強活性の PXR agonist": "CC(C)(C)NS(=O)(=O)C1(CNc2cc(Br)ccc2C#N)CCC1",
        "OADMET-0006254 · その pyridine analogue": "CC(C)(C)NS(=O)(=O)C1(CNc2c(Br)cncc2C#N)CCC1",""",
)
r(
    '''TILE_EXAMPLES, value="ibuprofen (start here)", label="molecule"''',
    '''TILE_EXAMPLES, value="ibuprofen (ここから)", label="分子"''',
)
r(
    """    ## 3 · Blind spots

    Folding loses information by accident. Some information is never collected in the first place.
    Each row below is a pair of *different* molecules; compare their Tanimoto similarity under four
    common settings.""",
    """    ## 3 · 盲点

    Folding は偶然によって情報を失います。一方で、そもそも最初から集められない情報もあります。
    下の各行は *異なる* 分子のペアです。よく使われる 4 つの設定での Tanimoto 類似度を比べてみてください。""",
)
r(
    '''"cyclohexyl- vs cycloheptylamine"''',
    '''"cyclohexyl- と cycloheptylamine"''',
)
r(
    '''"nonanoic vs palmitic acid"''',
    '''"nonanoic acid と palmitic acid"''',
)
r(
    '''"(R)- vs (S)-ibuprofen"''',
    '''"(R)- と (S)-ibuprofen"''',
)
r(
    '''"dexlansoprazole vs lansoprazole (stereo unspecified)"''',
    '''"dexlansoprazole と lansoprazole (立体未指定)"''',
)
r(
    '''"biphenyl vs terphenyl"''',
    '''"biphenyl と terphenyl"''',
)
r(
    '''"benzene → pyridine (PXR pair from §5)"''',
    '''"benzene → pyridine (§5 の PXR ペア)"''',
)
r(
    """"pair": name,""",
    """"ペア": name,""",
)
r(
    """    * **"How many" is invisible to bits.** Ring size and chain length beyond the radius produce the
      *same set* of environments, only repeated more often: Tanimoto 1.0 on bits, clearly below 1 on
      counts. Size is a strong driver of many ADMET endpoints (PXR's pocket is large and greasy), so
      this is not a corner case.
    * **Stereochemistry is off by default.** Enantiomers are identical unless you pass
      `includeChirality=True` — and even then some stereocentres (the lansoprazole sulfoxide) are
      not captured. The PXR training set contains exactly such pairs, with potency differences near a
      log unit.
    * **FCFP trades detail for pharmacophores.** It merges atoms with the same role (e.g. any
      aromatic carbon), which helps scaffold hopping but can hide changes that matter. Check the
      benzene → pyridine pair: one aromatic CH becomes an N.

    ## 4 · Similarity: what "Tanimoto > 0.4" promises

    Tanimoto similarity on ECFP4 is the share of set bits two molecules have in common. It is the
    workhorse of analogue searching, and it is how the PXR test set was built. Below, every test
    compound is placed by its similarity to its nearest **potent** training compound (pEC50 ≥ 6)
    against its own measured potency.""",
    """    * **「いくつあるか」は bit には見えない.** Radius より外側の環サイズや鎖長の違いは、*同じ集合* の環境を
      生むだけで、違うのは繰り返しの回数です。Bit では Tanimoto 1.0、count では明らかに 1 未満になります。
      サイズは多くの ADMET エンドポイントを強く左右する要因 (PXR のポケットは大きく脂溶性) なので、
      これは例外的なケースではありません。
    * **立体化学はデフォルトでオフ.** `includeChirality=True` を指定しないとエナンチオマーは同一になります。
      指定しても拾われない立体中心 (lansoprazole の sulfoxide) もあります。PXR の training set にはまさに
      このようなペアがあり、活性が ~1 log unit 違います。
    * **FCFP は細部と引き換えに pharmacophore を取る.** 同じ役割の原子 (たとえば芳香族炭素すべて) を
      まとめるので scaffold hopping には役立ちますが、大事な変化を隠すこともあります。芳香族 CH が 1 つ N に
      変わる benzene → pyridine のペアを見てください。

    ## 4 · 類似度: 「Tanimoto > 0.4」が約束すること

    ECFP4 の Tanimoto 類似度は、2 つの分子が共通して立てている bit の割合です。アナログ探索の主力であり、
    PXR の test set もこれで作られました。下の図では、各 test 化合物を、最も近い **強活性** の training 化合物
    (pEC50 ≥ 6) との類似度と、自身の実測活性でプロットしています。""",
)
r(
    """Tanimoto to nearest potent hit""",
    """最も近い強活性ヒットとの Tanimoto""",
)
r(
    """    The test compounds were chosen as ECFP4 neighbours (> 0.4) of **63** hits that were both potent
    and selective in the counter-screen; after re-measurement the launch post counts 46 such hits.
    Here we simply use all {_potent.height} training compounds with pEC50 ≥ 6, and RDKit's ECFP4 is not
    bit-identical to every vendor's implementation, so **{(_df["最も近い強活性ヒットとの Tanimoto"] > 0.4).mean():.0%}**
    of test compounds clear 0.4 in this plot. Either way, their own potency spans the whole assay range (dashed line:
    pEC50 6). Among the {_hi.height} compounds with Tanimoto ≥ 0.5 to a current hit, only
    **{(_hi["test pEC50"] >= 6).mean():.0%}** are hits themselves and
    **{(_hi["test pEC50"] < 4).mean():.0%}** are essentially inactive (pEC50 < 4).

    That is not a failure of ECFP. Similarity search is *supposed* to return the neighbourhood of a hit,
    and medicinal chemists buy analogues precisely to find out which ones keep the activity. What ECFP4
    Tanimoto cannot tell you is **which** of those small changes matter — and a model built on the same
    bits inherits the same blindness. The OpenADMET
    [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
    found that the hardest test compounds for all top teams were exactly these activity cliffs.""",
    """    Test 化合物は、強活性かつ counter-screen で選択的だった **63** ヒットの ECFP4 近傍 (> 0.4) として
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
    でも、上位チーム全員にとって最も難しかった test 化合物は、まさにこうした activity cliff でした。""",
)
r(
    """    ## 5 · ECFP inside a model

    Let's train the model most people would train first — **LightGBM on 2048-bit ECFP4**, the same
    baseline OpenADMET's challenge tutorial uses — and then ask it what it learned. Two standard tools:

    * **Global feature importance** (total gain per bit): which bits the trees split on most.
    * **Local attributions** (TreeSHAP, via LightGBM's `pred_contrib=True`): how much each bit pushed
      *this* molecule's prediction up or down. Spreading each bit's contribution over the atoms that set
      it gives a per-atom map, in the spirit of Riniker & Landrum's similarity maps.

    Choose the fold size and train. Watch what happens to the "top bits" as you change it.""",
    """    ## 5 · モデルの中の ECFP

    たいていの人が最初に作るモデル — OpenADMET のチャレンジ tutorial と同じ **2048-bit ECFP4 の LightGBM** —
    を学習させて、何を学んだのかを聞いてみます。定番の道具は 2 つです:

    * **Global feature importance** (bit ごとの gain の合計): 木がどの bit で最もよく分岐しているか。
    * **Local attribution** (LightGBM の `pred_contrib=True` による TreeSHAP): *この* 分子の予測を各 bit が
      どれだけ押し上げたか・押し下げたか。各 bit の寄与をそれを立てた原子に配分すると、Riniker & Landrum の
      similarity map のような原子ごとのマップになります。

    Fold サイズを選んで学習させてください。切り替えたとき「上位の bit」がどうなるかに注目です。""",
)
r(
    """        {"2048 bits (the usual default)": 2048, "8192 bits": 8192},
        value="2048 bits (the usual default)",
        label="fold size for the model",""",
    """        {"2048 bits (よくあるデフォルト)": 2048, "8192 bits": 8192},
        value="2048 bits (よくあるデフォルト)",
        label="モデルの fold サイズ",""",
)
r(
    """mo.status.spinner(f"Training LightGBM on {N_BITS}-bit ECFP4…")""",
    """mo.status.spinner(f"{N_BITS}-bit ECFP4 で LightGBM を学習中…")""",
)
r(
    '''title=f"top bits ({N_BITS})"''',
    '''title=f"上位の bit ({N_BITS})"''',
)
r(
    '''title="distinct substructures sharing the bit"''',
    '''title="その bit を共有する異なる部分構造の数"''',
)
r(
    """legend=alt.Legend(title="share of gain")""",
    """legend=alt.Legend(title="gain の割合")""",
)
r(
    '''label="median substructures per top-15 bit"''',
    '''label="上位 15 bit の部分構造数 (中央値)"''',
)
r(
    """    Each bar is one of the 15 most important bits; its length is the number of **different**
    substructures from the training set that were folded into it. At {N_BITS} bits the model's
    favourite features are each a mixture of about
    **{np.median(top_bits["# environments in bit"]):.0f}** unrelated environments. "Bit {top_bits["bit"][0]}
    is the most important feature" is not an explanation: it is a pointer to a bag of substructures.
    Switch to 8192 bits above and the bags shrink to a few members each — while the accuracy barely
    moves. Pick a bit below to see what is inside it.""",
    """    各バーは重要度上位 15 の bit の 1 つで、長さはその bit に fold された training set の **異なる**
    部分構造の数です。{N_BITS} bit では、モデルのお気に入りの特徴はそれぞれ約
    **{np.median(top_bits["# environments in bit"]):.0f}** 種類の無関係な環境の混合物です。
    「bit {top_bits["bit"][0]} が最も重要な特徴」は説明になっていません。部分構造の詰め合わせを指さしているだけです。
    上で 8192 bit に切り替えると、詰め合わせはそれぞれ数種類まで減ります — 精度はほとんど変わらないのに。
    下で bit を選ぶと、その中身が見られます。""",
)
r(
    """    ### Reading one prediction

    Now zoom into single molecules. The pair below is the second-hardest test compound of the whole
    challenge, **OADMET-0006254** (pEC50 2.06), next to its nearest training neighbour
    **OADMET-0002810** (pEC50 5.95). The only change is a benzene CH → pyridine N. According to the
    OpenADMET analysis, a related co-crystal structure shows that nitrogen H-bonding to SER247, which
    likely changes how the ligand sits in the pocket. Every Tier-1 team over-predicted this compound.

    Atom colours show the model's TreeSHAP attribution (red raises the predicted pEC50, blue lowers it).
    Click a bit to see its environments instead; the table's **SHAP** columns show each bit's
    contribution for A and B. The menu offers the other worst-predicted test compounds.""",
    """    ### 1 つの予測を読む

    ここからは分子単位で見ていきます。下のペアは、チャレンジ全体で 2 番目に難しかった test 化合物
    **OADMET-0006254** (pEC50 2.06) と、その training の nearest neighbour **OADMET-0002810** (pEC50 5.95) です。
    違いは benzene の CH → pyridine の N だけ。OpenADMET の分析によれば、関連する共結晶構造ではその窒素が
    SER247 と水素結合しており、ポケット内でのリガンドの収まり方が変わると考えられます。Tier-1 の全チームが
    この化合物を過大に予測しました。

    原子の色はモデルの TreeSHAP attribution です (赤は予測 pEC50 を上げ、青は下げる)。Bit をクリックすると
    代わりにその環境が表示され、表の **SHAP** 列には A と B それぞれの bit ごとの寄与が出ます。リストから
    予測の外れが大きい他の test 化合物も選べます。""",
)
r(
    """"label": f"test · true {_y[_ids[0]]:.2f} · predicted {_pred[0]:.2f}",""",
    """"label": f"test · 実測 {_y[_ids[0]]:.2f} · 予測 {_pred[0]:.2f}",""",
)
r(
    """"label": f"train · true {_y[_ids[1]]:.2f} · predicted {_pred[1]:.2f}",""",
    """"label": f"train · 実測 {_y[_ids[1]]:.2f} · 予測 {_pred[1]:.2f}",""",
)
r(
    """pair_note=f"baseline (mean prediction) {_contrib[0, -1]:.2f}",""",
    """pair_note=f"baseline (予測の平均) {_contrib[0, -1]:.2f}",""",
)
r(
    """    **Where does the prediction gap come from?** The model predicts A − B = **{_gap:+.2f}**
    (measured: {pair_y[0] - pair_y[1]:+.2f}). TreeSHAP splits that gap exactly over the features:
    the **{int(_differ.sum())}** bits that differ between A and B account for **{_diff[_differ].sum():+.2f}**,
    the **{int(_shared.sum())}** bits they share for **{_diff[_shared].sum():+.2f}** (trees split on combinations,
    so a shared bit can matter more in one context than the other), and bits absent from both for
    **{_diff[_neither].sum():+.2f}**. The atom map shows only bits that are *present*; absences matter to trees
    too, but they cannot be drawn on atoms.

    What the attribution map can and cannot tell you:

    * **The change is visible only as a few bits.** A single atom swap flips a handful of environments
      (the *only A* / *only B* rows). The model sees them — but in the training data each of those bits
      also stands for many other substructures (**# envs**), so its learned effect is an average over
      everything folded into it. In the benzene → pyridine pair, the new nitrogen barely lights up.
    * **Similar fingerprints give similar predictions.** Most attribution sits on shared bits. The model is
      doing exactly what ECFP asks it to do; the activity cliff is simply not in the representation.
    * **An attribution on a colliding bit is ambiguous.** A red atom means "the model likes this bit", not
      "the model likes this group", when the bit holds a dozen substructures. Retrain at 8192 bits and compare.""",
    """    **予測の差はどこから来るのか?** モデルの予測は A − B = **{_gap:+.2f}** です (実測: {pair_y[0] - pair_y[1]:+.2f})。
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
      比べてみてください。""",
)
r(
    """    ## 6 · Cheat sheet

    **Strengths**

    * Fast, deterministic, needs no training data.
    * Describes local substructure well; excellent for similarity search, clustering and analogue retrieval.
    * A strong baseline for SAR within a congeneric series.

    **Weaknesses**

    * Folding collisions: roughly half of drug-like molecules collide with themselves at 2048 bits, and
      every bit is shared across a dataset.
    * Bit vectors record presence, not counts: ring size, chain length and repeated groups disappear.
    * Stereochemistry is ignored by default.
    * No global shape, size or physicochemistry; every environment counts the same.
    * Hashed bits are hard to interpret.

    **Defaults worth changing**

    * Prefer **count** fingerprints for regression.
    * Use ≥ 4096 bits, or unfolded/sparse features, if you want to *interpret* individual bits.
    * Pass `includeChirality=True` when stereo matters.
    * Add a few whole-molecule descriptors (logP, size, TPSA) next to the bits.

    **When interpreting a model**

    * Map each bit back to *all* the environments it contains before telling a story about it.
    * Read per-atom attributions across a matched pair, not for a single molecule in isolation.

    ---

    ### About this notebook

    * **Data:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0). Test-set design and the analysis of the hardest compounds are from OpenADMET's
      [challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/),
      [launch post](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/) and
      [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/).
    * **References:** Rogers & Hahn, *J. Chem. Inf. Model.* 2010, 50, 742 (ECFP); Morgan, *J. Chem. Doc.* 1965, 5, 107;
      Riniker & Landrum, *J. Cheminform.* 2013, 5, 43 (similarity maps).
    * **Widgets:** `ECFPStepper`, `MorganBitTiles` and `MorganExplorer` are custom anywidget components written for
      this notebook ([source](https://github.com/N283T/openadmet-marimo)); `molwidgets.ecfp` is the readable
      ECFP implementation shown above.
    * **Companion notebook:** *Similar, but not the same* digs into why fingerprint models struggle on this dataset.
    * **AI use:** Built together with Claude (Anthropic) as a coding assistant for the widgets and notebook
      scaffolding. The questions, analysis choices and interpretation come from my own PXR challenge work;
      every number is computed live in this notebook.""",
    """    ## 6 · チートシート

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

    ### この notebook について

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
      表示している数値はすべて、この notebook の中でその場で計算しています。""",
)
r(
    """    ## 1 · The algorithm: from atoms to bits

    ECFP builds a molecule's description from the inside out, in four moves:

    1. **Atom invariants (iteration 0).** Each heavy atom is hashed from six numbers: atomic number,
       degree (including Hs), hydrogen count, formal charge, isotope and ring membership.
    2. **Grow (iteration *r*).** Each atom's new identifier is the hash of its previous identifier and
       the sorted *(bond order, neighbour identifier)* pairs. Radius 1 sees the neighbours, radius 2
       their neighbours, and so on.
    3. **Collect.** Every identifier is a feature, unless it covers exactly the same bonds as one
       already collected (**duplicate**) or its environment stopped growing (**no growth**). The result
       is a *set* of integers: the unfolded ECFP.
    4. **Fold.** Each identifier switches on bit `identifier % n_bits` — a hash table with no collision
       handling.

    Start with **isobutane** and press **next ▶**. Identifiers are shown as short labels (*a, b* for
    iteration 0, *A, B* for 1, *A', B'* for 2); the bit vector under the feature set is what a model sees.""",
    """    ## 1 · アルゴリズム: 原子から bit まで

    ECFP は分子の記述を内側から外側へ、4 つの手順で組み立てます:

    1. **Atom invariants (iteration 0).** 各重原子を 6 つの数字から hash します: 原子番号、次数 (H を含む)、
       水素の数、形式電荷、同位体、環に含まれるかどうか。
    2. **広げる (iteration *r*).** 各原子の新しい identifier は、1 つ前の identifier と、ソートした
       *(結合次数, 隣接原子の identifier)* のペアをまとめて hash したものです。Radius 1 は隣を、radius 2 は
       隣の隣までを見ます。
    3. **集める.** Identifier はすべて特徴になります。ただし、すでに集めたものとまったく同じ結合を覆うもの
       (**duplicate**) と、環境がそれ以上広がらなかったもの (**no growth**) は除きます。結果は整数の *集合*、
       つまり unfold された ECFP です。
    4. **Fold する.** 各 identifier は bit `identifier % n_bits` を立てます — collision の処理を持たない
       hash table です。

    まずは **isobutane** で **next ▶** を押してみてください。Identifier は短いラベルで表示します (iteration 0 は
    *a, b*、1 は *A, B*、2 は *A', B'*)。特徴の集合の下にある bit vector が、モデルが実際に見るものです。""",
)
r(
    """    **In isobutane**, steps 1–4 give the three methyls label **a** and the centre **b**: two features,
    because the fingerprint does not record that *a* occurs three times. Steps 5–8 add **A** = hash(*a* |
    single→*b*), "a methyl on a CH", and **B**, "a CH with three methyls", which is already the whole
    molecule. Steps 9–12 add nothing: each methyl's radius-2 environment covers the same bonds as *B*
    (**duplicate**) and the centre cannot grow (**no growth**). Four features, four bits.

    Then pick **paracetamol** and switch to **explore**. The four aromatic CH carbons share one label at
    iteration 0 — ECFP starts from chemistry-free labels — and one bit is already framed in red at 64 bits:
    two different features the model can no longer tell apart.""",
    """    **Isobutane では**、ステップ 1–4 で 3 つのメチルがラベル **a**、中心が **b** になります。特徴は 2 つです。
    Fingerprint は *a* が 3 回出てくることを記録しないからです。ステップ 5–8 で **A** = hash(*a* | single→*b*)、
    つまり「CH についたメチル」と、**B**「メチルを 3 つ持つ CH」(これでもう分子全体) が加わります。
    ステップ 9–12 では何も増えません。各メチルの radius 2 の環境は *B* と同じ結合を覆うので **duplicate**、
    中心はもう広がれないので **no growth** です。特徴 4 つ、bit 4 つ。

    次は **paracetamol** を選んで **explore** に切り替えてください。4 つの芳香族 CH 炭素は iteration 0 で同じ
    ラベルを共有します — ECFP の出発点は化学的な意味を持たないラベルです。そして 64 bit でもすでに赤枠の bit が
    1 つあります。モデルがもう区別できない 2 つの特徴です。""",
)
r(
    '''"Under the hood: the whole algorithm in ~40 lines, checked against RDKit"''',
    '''"舞台裏: アルゴリズム全体を ~40 行で、RDKit と照合済み"''',
)
r(
    """    The stepper runs on a plain-Python ECFP (`molwidgets.ecfp`). It uses `blake2b` instead of RDKit's
    internal hash, so identifiers are different numbers, but the invariants, neighbour ordering and
    duplicate removal follow RDKit: for **{validation[1]:,} / {train.height:,}** training molecules at
    radius 1 and **{validation[2]:,} / {train.height:,}** at radius 2 it yields exactly as many distinct
    features as RDKit's `GetSparseCountFingerprint`.""",
    """    Stepper は素の Python で書いた ECFP (`molwidgets.ecfp`) で動いています。RDKit 内部の hash の代わりに
    `blake2b` を使っているので identifier の数値は異なりますが、invariant、隣接原子の並べ方、重複の除去は
    RDKit に従っています: training 分子のうち radius 1 で **{validation[1]:,} / {train.height:,}**、radius 2 で
    **{validation[2]:,} / {train.height:,}** について、RDKit の `GetSparseCountFingerprint` とまったく同じ数の
    distinct な特徴を出します。""",
)
r(
    """    ## 2 · Collisions

    Folding is where information is lost. For a single molecule it is the birthday problem: with *k*
    distinct features in *n* bits, the chance of no collision is""",
    """    ## 2 · Collision

    情報が失われるのは fold の段階です。1 つの分子について言えば、これは誕生日問題です: *k* 個の distinct な
    特徴を *n* bit に入れたとき、collision が 1 つも起きない確率は""",
)
r(
    """    A typical molecule has ~50 ECFP4 features, so at 2048 bits roughly **half of all molecules collide
    with themselves**. The formula and the PXR training set agree closely:""",
    """    典型的な分子の ECFP4 特徴は ~50 個なので、2048 bit ではおよそ **半数の分子が自分自身の中で衝突します**。
    式と PXR の training set はよく一致しています:""",
)
r(
    """    Across a dataset, collisions are universal: the {train.height:,} training molecules contain
    **{int(_envs.sum()):,}** distinct ECFP4 environments for 2048 bits, about **{_envs[_envs > 0].mean():.0f}
    per bit**, and every bit is shared.

    Below, a molecule's fingerprint is drawn bit by bit with RDKit's `DrawMorganEnv` (blue: centre atom,
    yellow: aromatic, grey: aliphatic ring, light grey: where the environment attaches). **Red frames**
    are collisions inside the molecule; the **badge** counts other substructures that share the bit
    across the dataset — click a tile to see them.""",
    """    データセット全体では collision は避けられません: {train.height:,} 個の training 分子には distinct な ECFP4
    環境が **{int(_envs.sum()):,}** 種類あり、2048 bit に対して **1 bit あたり約 {_envs[_envs > 0].mean():.0f} 種類**、
    すべての bit が共有されています。

    下では、1 つの分子の fingerprint を RDKit の `DrawMorganEnv` で bit ごとに描いています (青: 中心原子、
    黄: 芳香族、灰: 脂肪族環、薄い灰: 環境がつながる先)。**赤枠** は分子の中での collision、**バッジ** は
    データセット全体でその bit を共有している他の部分構造の数です — タイルをクリックすると、それらが見られます。""",
)
r(
    """    Things to look for:

    * **Bit 807 appears twice in ibuprofen**: the acid's carbonyl carbon (`[C;D3;H0]`) and its hydroxyl
      oxygen (`[O;D1;H1]`). Radius-0 identifiers do not depend on the rest of the molecule, so at 2048
      bits *every* hydroxyl oxygen shares a bit with *every* carbon that has three connections and no H.
      Click it to see the other substructures in that bit; switch to 8192 bits and the red frames vanish.
    * **Repeats collapse.** Three methyls and four aromatic CH carbons give one tile each (×3, ×4).
    * **Radius 0 ignores bond order.** In paracetamol, the carbonyl oxygen `[O;D1;H0]` is the same
      identifier as a sulfonyl oxygen; only the radius-1 tile `O=C` tells them apart.""",
    """    注目してほしい点:

    * **Ibuprofen では bit 807 が 2 回出てくる**: カルボン酸のカルボニル炭素 (`[C;D3;H0]`) とヒドロキシ酸素
      (`[O;D1;H1]`) です。Radius 0 の identifier は分子の残りに依存しないので、2048 bit では *すべての*
      ヒドロキシ酸素が、結合相手が 3 つで H を持たない *すべての* 炭素と bit を共有しています。クリックすると
      その bit の他の部分構造が見られます。8192 bit にすると赤枠は消えます。
    * **繰り返しはまとめられる.** 3 つのメチルと 4 つの芳香族 CH 炭素は、それぞれ 1 枚のタイル (×3、×4) に
      なります。
    * **Radius 0 は結合次数を見ない.** Paracetamol のカルボニル酸素 `[O;D1;H0]` は、スルホニルの酸素と同じ
      identifier です。区別できるのは radius 1 のタイル `O=C` だけです。""",
)
r(
    '''label="look inside a top bit"''',
    '''label="上位の bit の中身を見る"''',
)
r(
    '''label="pair (test compound vs its nearest training neighbour)"''',
    '''label="ペア (test 化合物とその training の nearest neighbour)"''',
)
r(
    """(true {r['test pEC50']:.2f}, predicted {r['predicted']:.2f})""",
    """(実測 {r['test pEC50']:.2f}, 予測 {r['predicted']:.2f})""",
)

# Replacements run in order, so later passages already contain translated column names.
out = src
missing = []
for o, n in R:
    if o not in out:
        missing.append(o[:70])
        continue
    out = out.replace(o, n)
if missing:
    raise SystemExit("not found:\n" + "\n".join(missing))
Path("notebooks/ecfp_inside_ja.py").write_text(out)
print("ok", len(R), "replacements")
