"""Generate notebooks/pxr_fingerprints_ja.py from the English notebook.

Every English passage is replaced by a Japanese one; the run fails if any passage
is missing, so edit the English notebook first and then update the pairs here.

    uv run python scripts/translate_pxr_fingerprints_ja.py && uv run ruff format notebooks
"""

from pathlib import Path

src = Path("notebooks/pxr_fingerprints.py").read_text()

R = []  # (old, new)


def r(old, new):
    R.append((old, new))


r(
    'app_title="Similar, but not the same: Morgan fingerprints on PXR"',
    'app_title="似ているのに、同じじゃない: PXR と Morgan fingerprint"',
)

r(
    """    # Similar, but not the same
    ### Why Morgan fingerprints stumble on PXR induction

    The **pregnane X receptor (PXR)** is a nuclear receptor that senses foreign molecules and switches on
    CYP3A4, P-gp and friends. A drug that activates PXR can speed up the metabolism of *other* drugs, so
    PXR induction is a classic drug–drug-interaction liability. OpenADMET measured it for >11,000
    compounds and ran a blind challenge asking for **pEC50** predictions on 513 new ones.

    In that challenge, models built on **Morgan (ECFP-like) fingerprints** — the workhorse of QSAR —
    were consistently among the weakest inputs. In my own entry (4th of 95), a Morgan-only
    LightGBM landed around 0.57 CV MAE while descriptor/embedding ensembles got below 0.40, and
    other teams reported the same ordering. This notebook asks **why**.

    Fingerprint models are nearest-neighbour machines at heart: they assume that *molecules which
    share substructures share activity*. We will test that assumption directly on the PXR data and
    look, bit by bit, at where it breaks.

    /// details | What you will build intuition for
    1. What a Morgan bit actually encodes (and what it throws away)
    2. How similar the test set really is to the training set
    3. How well structural similarity predicts PXR activity — the *similarity principle*, measured
    4. Three blind spots: stereochemistry, "how much" vs "whether", and whole-molecule properties
    5. A small model lab to test the ideas yourself
    ///

    The interactive pieces (molecule grid and fingerprint explorer) are custom
    [anywidget](https://anywidget.dev) components from the `molwidgets` package that ships with this
    notebook.""",
    """    # 似ているのに、同じじゃない
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
    [anywidget](https://anywidget.dev) ベースの自作パッケージ `molwidgets` です。""",
)

r(
    """    ## 1 · The data, and some chemical hygiene

    We use the public OpenADMET PXR release on Hugging Face (CC-BY-4.0): the **training set** and
    the **full test set**, whose labels were unblinded after the challenge in two phases. Every
    SMILES is standardized the same way: keep the largest fragment, neutralize charges, keep
    stereochemistry, write canonical SMILES.""",
    """    ## 1 · データと、ケモインフォマティクス的な下ごしらえ

    Hugging Face で公開されている OpenADMET PXR データ (CC-BY-4.0) の **training set** と、チャレンジ後に
    2 段階で正解が公開された **test set 全体** を使います。SMILES はすべて同じ手順で標準化します:
    最大フラグメントを残す、電荷を中和する、立体化学は保持する、canonical SMILES で書き出す。""",
)

for old, new in [
    ('label="training compounds"', 'label="training 化合物"'),
    ('label="test compounds (unblinded)"', 'label="test 化合物 (正解公開済み)"'),
    ('label="unparsable SMILES"', 'label="パースできない SMILES"'),
    ('label="charged / salt forms neutralized"', 'label="中和した荷電体 / 塩"'),
    ('label="duplicates after standardization"', 'label="標準化後の重複"'),
    ('label="test compounds also in train"', 'label="train にも含まれる test 化合物"'),
]:
    r(old, new)

r(
    """    **Not all labels are equal.** pEC50 comes from a dose–response fit. Weak compounds never reach
    a plateau, so their EC50 is an extrapolation: below pEC50 3 the median confidence interval is
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["median 95% CI width"].item():.1f} log units wide**,
    versus ~0.2–0.3 for potent compounds. That is
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item()} training compounds** ({_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item() / train.height:.0%})
    whose exact value is mostly noise — keep this in mind when we look at "activity cliffs" later.""",
    """    **ラベルの信頼度は一様ではありません。** pEC50 は dose–response カーブのフィッティングから得られます。
    弱い化合物はプラトーに届かないので、EC50 は外挿値になります。pEC50 3 未満では 95% CI の中央値が
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["median 95% CI width"].item():.1f} log unit** もあり、
    強い化合物の 0.2〜0.3 とは桁違いです。該当するのは
    **{_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item()} 化合物** (training の {_bins.filter(pl.col("pEC50 bin") == "< 3")["n"].item() / train.height:.0%})
    で、その正確な値はほぼノイズです。後で activity cliff を見るときに思い出してください。""",
)

r(
    """    ## 2 · Anatomy of a Morgan fingerprint

    A Morgan fingerprint describes a molecule as a *bag of circular atom environments*. Every atom
    starts with an identifier built from its element, degree, hydrogen count, charge and ring
    membership (**radius 0**). Each iteration merges in the identifiers of the neighbours, so
    radius 1 sees an atom plus its bonded neighbours, radius 2 the neighbours of those, and so on.
    Every environment is hashed to a 32-bit integer, then **folded** into a fixed-length bit vector
    with `hash % n_bits`.

    Two consequences matter for everything that follows:

    * The fingerprint records **whether** an environment occurs, not where, how often (for bit
      vectors), or what the whole molecule looks like.
    * Folding means unrelated environments can land on the **same bit** — a *collision*.

    Pick any compound in the grid (filter by id, sort by pEC50, or type a SMARTS such as
    `c1ccncc1`), then click rows in the explorer to highlight the atoms that set each bit. For every
    bit the table also shows how many training compounds carry it, how many *different*
    substructures land on it (**# envs**) and how the mean pEC50 changes when it is on. Select a bit
    to see those colliding substructures side by side.""",
    """    ## 2 · Morgan fingerprint の解剖

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
    部分構造が並んで表示されます。""",
)

r(
    '''        f"Selected bit **{_bit}** is set by **{len(explorer.value.get('bit_examples', []))}** different "
        "substructures in the training set (gallery above)."
        if _bit >= 0
        else "Click a row in the bit table: the selected bit flows back into Python and updates this text."''',
    '''        f"選択中の bit **{_bit}** は、training set の中で **{len(explorer.value.get('bit_examples', []))}** 種類の"
        "異なる部分構造によって立てられています (上のギャラリー)。"
        if _bit >= 0
        else "Bit の表の行をクリックしてみてください。選んだ bit が Python 側に戻り、この文章が更新されます。"''',
)
for old, new in [
    (
        'label=f"distinct radius ≤ {_r} environments in training set"',
        'label=f"training set に現れる radius ≤ {_r} の環境の種類"',
    ),
    ('label="bits used"', 'label="使われている bit"'),
    ('label="environments per used bit"', 'label="1 bit あたりの環境数"'),
    ('label="bits shared by ≥ 2 environments"', 'label="2 種類以上の環境が同居する bit"'),
]:
    r(old, new)
r(
    """                f"With {_n:,} bits, the training set's {int(_envs.sum()):,} environments have nowhere to go "
                "but on top of each other. Switch the explorer to 4096 bits or radius 1 and watch these "
                "numbers move. " + _sel""",
    """                f"{_n:,} bit しかないので、training set の {int(_envs.sum()):,} 種類の環境は重なり合うしかありません。"
                "Explorer を 4096 bit や radius 1 に切り替えると、この数字がどう動くか見られます。" + _sel""",
)

r(
    """    ## 3 · How the test set was built

    The 513 test compounds are not a random sample. OpenADMET took the **63** compounds that were potent
    (EC50 ≤ 1 µM) *and* selective in the PXR-null counter-screen, and bought Enamine analogues with an
    **ECFP4 Tanimoto similarity > 0.4** to them
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/)).
    It is a hit-expansion set, exactly how a medicinal chemist would follow up a screen.

    So two things are true by construction: every test compound has a close relative in the training
    data, and that relative is usually potent. Let's confirm it, and then ask the question that actually
    matters: **given a close, potent neighbour, can structural similarity tell which analogues keep the
    activity?**

    For every compound we find its **nearest neighbour (NN)** in the training set by Tanimoto
    similarity on Morgan fingerprints (radius 2, 2048 bits). For training compounds we exclude the
    compound itself.""",
    """    ## 3 · Test set はどう作られたか

    Test の 513 化合物はランダムに選ばれたものではありません。OpenADMET は、強活性 (EC50 ≤ 1 µM) *かつ*
    PXR-null の counter-screen で選択的だった **63** 化合物を選び、それらとの **ECFP4 Tanimoto 類似度が
    0.4 を超える** analogue を Enamine から購入しました
    ([challenge announcement](https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/))。
    メディシナルケミストがスクリーニングのヒットを追いかけるときと同じ、hit expansion のセットです。

    つまり、次の 2 つは設計上当然成り立ちます。どの test 化合物にも training データに近い「親戚」がいること、
    そしてその親戚はたいてい強活性であること。まずそれを確認し、そのうえで本当に大事な問いに進みます:
    **近くに強活性の neighbour がいるとき、構造の類似度で「どの analogue が活性を保つか」を見分けられるか?**

    各化合物について、Morgan fingerprint (radius 2, 2048 bits) の Tanimoto 類似度で training set の
    **nearest neighbour (NN)** を探します。Training 化合物については自分自身を除きます。""",
)

# legend keys for NN density
r('"train → rest of train"', '"train → 他の train"')
r(
    'title="Tanimoto similarity to nearest training neighbour"',
    'title="最も近い training 化合物との Tanimoto 類似度"',
)
r('label="median NN similarity, train → train"', 'label="NN 類似度の中央値 (train → train)"')
r('label="median NN similarity, test → train"', 'label="NN 類似度の中央値 (test → train)"')
r('label="test compounds with a NN ≥ 0.5"', 'label="NN 類似度 ≥ 0.5 の test 化合物"')
r(
    '''                "As designed, the test set is **closer** to the training data than the training compounds "
                "are to each other. Being out of domain is not the problem here. The question is whether "
                "similarity carries the activity, and the most direct way to find out is to predict with it."''',
    '''                "設計どおり、test set は training 化合物同士よりも **training set に近い** です。"
                "Applicability domain の外にあることが問題なのではありません。問題は類似度が活性を運ぶかどうかで、"
                "それを確かめるいちばん直接的な方法は、類似度で予測してみることです。"''',
)

r('label="neighbours k"', 'label="近傍数 k"')
r(
    """    So let's use the most direct fingerprint model there is: predict each test compound's pEC50 as the
    mean of its **k most similar** training compounds. {k_slider}""",
    """    そこで、いちばん素朴な fingerprint モデルを使ってみます。各 test 化合物の pEC50 を、
    **最も似ている k 個** の training 化合物の平均で予測します。{k_slider}""",
)

r(
    'label="MAE if you guess a random training compound"',
    'label="ランダムな training 化合物で当てた場合の MAE"',
)
r(
    '''                        "With **k = 1**, the nearest neighbour has essentially no ranking power "
                        "(ρ ≈ 0), and its pEC50 is off by as much as a randomly chosen training compound. "
                        "Averaging more neighbours helps only by shrinking every prediction toward the "
                        "mean. Drag the slider and watch the cloud collapse into a vertical band."''',
    '''                        "**k = 1** では、nearest neighbour に順位付けの力はほぼありません (ρ ≈ 0)。"
                        "その pEC50 のずれは、ランダムに選んだ training 化合物と同じくらいです。"
                        "近傍を増やして平均しても、予測が全部平均値に寄っていくだけです。"
                        "スライダーを動かすと、点の雲が縦の帯に潰れていくのが見えます。"''',
)

r('"all training compounds"', '"training 化合物全体"')
r('"nearest training neighbour of each test compound"', '"各 test 化合物の NN (training)"')
r('"test compounds (truth)"', '"test 化合物 (正解)"')
r('mo.md("### Who are the neighbours?")', 'mo.md("### Neighbour は誰なのか?")')
r(
    """    The vertical band in the scatter above is the test-set design showing through. The nearest
    neighbours of test compounds are overwhelmingly **potent** training compounds: their mean pEC50 is
    **{_nn.mean():.2f}**, against **{y_train.mean():.2f}** for the training set as a whole, and
    **{(_nn >= 5.5).mean():.0%}** of them have pEC50 ≥ 5.5. That is expected, since the analogues were
    chosen around the hits.

    What matters is that the test compounds themselves do not follow. Their potency spreads from ~2 to ~7
    (mean **{y_test.mean():.2f}**): an **SAR exploration around the hits**, where every compound has a
    close, potent relative *by construction* and the real question is which small changes keep the
    activity. That is exactly the question a similarity-based model cannot answer, and the one the
    OpenADMET [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
    identifies as the shared failure point of every top team.""",
    """    上の散布図の縦の帯は、test set の設計がそのまま見えているものです。Test 化合物の nearest neighbour は
    圧倒的に **強活性** の training 化合物で、その平均 pEC50 は **{_nn.mean():.2f}** (training 全体では
    **{y_train.mean():.2f}**)、**{(_nn >= 5.5).mean():.0%}** が pEC50 ≥ 5.5 です。Analogue はヒットの周りで
    選ばれたのだから、これは当然です。

    大事なのは、test 化合物自身の活性はそれに従わないことです。活性は ~2 から ~7 まで広がっています
    (平均 **{y_test.mean():.2f}**)。つまり test set は **ヒット化合物周りの SAR 探索** で、どの化合物にも
    *設計上* 近くに強活性の「親戚」がいて、本当に問われているのは「どの小さな変化なら活性が残るのか」です。
    これこそ類似度ベースのモデルが答えられない問いであり、OpenADMET の
    [post-challenge analysis](https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/)
    が上位チーム共通の失敗点として挙げているものです。""",
)

r(
    """    ## 4 · The similarity principle, measured

    The *similar property principle* says that the activity difference between two molecules should
    shrink as their structural similarity grows. We can measure it directly on all
    ~8.5 million training pairs: bin the pairs by Tanimoto similarity and look at the typical
    |Δ pEC50| in each bin.""",
    """    ## 4 · Similarity principle を実測する

    *Similar property principle* によれば、2 つの分子の構造類似度が上がるほど、活性の差は小さくなるはずです。
    Training set の全ペア (約 850 万) で直接測ってみます。ペアを Tanimoto 類似度で bin に分け、
    各 bin の典型的な |Δ pEC50| を見ます。""",
)
r('title="Tanimoto similarity of the pair"', 'title="ペアの Tanimoto 類似度"')
r(
    'title="|Δ pEC50| (mean, shaded to 90th pct.)"',
    'title="|Δ pEC50| (平均、帯は 90 パーセンタイルまで)"',
)
r(
    """    The dashed line is a random pair (**{random_pair_dy:.2f}** log units). The curve does go down —
    similarity is not useless — but slowly. Even among pairs with Tanimoto ≥ 0.5, the typical
    difference is **{(_hi["mean_dy"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.2f}** log units and
    **{(_hi["frac_gt1"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.0%}** of pairs differ by more than
    10-fold in potency.

    Hover the points for the counts: there are very few truly close pairs, so most of the data lives in
    the region where the fingerprint says "somewhat similar" and PXR says "could be anything".""",
    """    点線はランダムなペアの値 (**{random_pair_dy:.2f}** log unit) です。カーブは確かに下がっており、
    類似度が無意味なわけではありません。ただ、その下がり方がゆるやかです。Tanimoto ≥ 0.5 のペアでも
    典型的な差は **{(_hi["mean_dy"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.2f}** log unit あり、
    **{(_hi["frac_gt1"] * _hi["pairs"]).sum() / _hi["pairs"].sum():.0%}** のペアは活性が 10 倍以上違います。

    点にカーソルを乗せるとペア数が見られます。本当に近いペアはごくわずかで、データの大半は
    fingerprint が「そこそこ似ている」と言い、PXR が「何でもありうる」と言う領域にあります。""",
)

r('label="min Tanimoto"', 'label="Tanimoto の下限"')
r('label="min |Δ pEC50|"', 'label="|Δ pEC50| の下限"')
r(
    """    ### Browse the cliffs

    An **activity cliff** is a pair that looks alike but behaves differently. Choose what "alike" and
    "differently" mean, pick a pair from the table, and compare the two fingerprints bit by bit.
    Bits present in only one molecule are exactly what a fingerprint model *can* use to explain the
    difference — often they are a handful of generic environments.""",
    """    ### Activity cliff を眺める

    **Activity cliff** とは、見た目は似ているのに活性が大きく違うペアのことです。「似ている」と「違う」の
    基準を決め、表からペアを選んで、2 つの fingerprint を bit 単位で比べてみてください。片方にしかない bit こそが、
    fingerprint モデルが差を説明するために *使える* 唯一の手がかりです。そしてそれは多くの場合、
    ありふれた環境がいくつかあるだけです。""",
)
r(
    'label=f"{cliffs.height:,} pairs · SALI = |Δ pEC50| / (1 − similarity)"',
    'label=f"{cliffs.height:,} ペア · SALI = |Δ pEC50| / (1 − similarity)"',
)
r('mo.md("_Select a pair in the table above._")', 'mo.md("_上の表からペアを選んでください。_")')
r(
    """    Two things to look for while browsing:

    * **Is the weak partner a real measurement?** In {_noisy} of these {_n} pairs, the weaker compound
      has a CI wider than 1.5 log units — part of the "cliff" is assay noise at the bottom of the scale.
    * **Which one is greasier?** In {_lip} of {_n} pairs ({_lip / max(_n, 1):.0%}) the more potent
      compound also has the higher calculated logP. A property of the *whole molecule* is doing work that
      no single bit can express. That is the next section.""",
    """    眺めるときに注目したい点は 2 つです:

    * **弱い側の値は本当に測れているか?** この {_n} ペアのうち {_noisy} ペアでは、弱い側の化合物の CI が
      1.5 log unit を超えています。「cliff」の一部は、スケールの底にあるアッセイノイズです。
    * **どちらが脂溶性が高いか?** {_n} ペア中 {_lip} ペア ({_lip / max(_n, 1):.0%}) で、強い側の化合物のほうが
      計算 logP も高くなっています。*分子全体* の性質が、どの単独の bit でも表せない役割を果たしています。
      これが次のセクションのテーマです。""",
)

r(
    """    ## 5 · What the bits cannot see

    ### 5a · Different molecules, identical fingerprints

    If two different molecules produce exactly the same bit vector, any model built on those bits
    **must** predict the same value for both. Let's find every such group in the training set and ask
    *why* the fingerprint cannot tell them apart.""",
    """    ## 5 · Bit には見えないもの

    ### 5a · 違う分子なのに、fingerprint が完全に一致する

    2 つの異なる分子がまったく同じ bit vector を持つなら、その bit で作ったモデルは両者に **必ず**
    同じ値を予測します。Training set からそういうグループをすべて探し、fingerprint が区別できない
    *理由* を見てみます。""",
)
r(
    """    **{len(_twins)} groups** ({fp_twins.height} training compounds) collapse onto a shared fingerprint;
    potency inside a group differs by up to **{_summary["range"].max():.2f}** log units. There are two
    different reasons:

    * **Stereochemistry** ({_n_stereo} groups). RDKit Morgan fingerprints ignore chirality and
      double-bond geometry unless you pass `includeChirality=True`. Look closely: in these groups one
      record has its stereocentre or double bond *specified* and the other does not — a single
      enantiomer next to what is most likely the racemate (lansoprazole / dexlansoprazole,
      bupivacaine / levobupivacaine), or the same drug registered twice with and without E/Z labels
      (rifampicin, the textbook PXR agonist). The lansoprazole pair differs by ~1 log unit, which is a
      real question for a chemist and invisible to the fingerprint. Switching on chirality separates
      {_n_split} of these {_n_stereo} groups; the sulfoxide stereocentre is not picked up either way.
    * **Ring size / chain length** ({len(_twins) - _n_stereo} groups). Cyclohexyl vs cycloheptyl amine,
      azepane vs azocane, nonanoic vs palmitic acid: every atom sees the same neighbourhood within
      radius 2, so the *set* of environments is identical — only *how many times* each occurs differs.
      A bit vector
      stores presence, not counts; a count fingerprint separates these (see 5c).""",
    """    **{len(_twins)} グループ** ({fp_twins.height} 化合物) が同じ fingerprint に潰れており、グループ内の活性差は
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
      Count fingerprint なら区別できます (5c 参照)。""",
)

r(
    """    ### 5b · Whole-molecule properties

    PXR's ligand-binding pocket is large, flexible and hydrophobic; it is famous for accepting
    very different scaffolds. If binding is driven by *how greasy and how big* a molecule is rather
    than by a specific pharmacophore, then a bag of local substructures is the wrong language.""",
    """    ### 5b · 分子全体の物性

    PXR の ligand-binding pocket は大きく、柔軟で、疎水的です。まったく違う scaffold を受け入れることで
    知られています。結合を決めているのが特定の pharmacophore ではなく *どのくらい脂溶性で、どのくらい大きいか*
    なのだとしたら、局所的な部分構造の集まりという表現は、そもそも言語として合っていません。""",
)
r('label="descriptor"', 'label="記述子"')
r(
    """    A single number — Crippen logP — ranks training compounds about as well as the fingerprint kNN
    ranked the test set. And *within* the similar pairs (Tanimoto ≥ 0.5), the logP difference
    tracks the potency difference (Spearman ρ = **{_rho:.2f}**): part of what looks like a cliff to
    the fingerprint is a smooth lipophilicity trend. A folded bit vector has no axis for "a bit more
    lipophilic"; adding a methyl either flips a bit or it doesn't.""",
    """    Crippen logP というたった 1 つの数字が、fingerprint kNN が test set を順位付けしたのと同じくらいの精度で
    training 化合物を順位付けします。さらに似ているペア (Tanimoto ≥ 0.5) の *中でも*、logP の差が活性の差と
    連動しています (Spearman ρ = **{_rho:.2f}**)。Fingerprint から見ると cliff に見えるものの一部は、
    なめらかな脂溶性のトレンドなのです。Fold された bit vector には「少しだけ脂溶性が高い」という軸がありません。
    メチル基を 1 つ足しても、bit が 1 つ立つか立たないかでしかありません。""",
)

r(
    """    ### 5c · "Whether" versus "how much"

    A binary bit says an environment is present. A **count** fingerprint records how many times — and
    counts summed over a molecule are a crude measure of size and lipophilicity. If the hypothesis
    above is right, counts should recover some of the lost signal. Test it in the lab below.

    ## 6 · Model lab

    Train a LightGBM model (the same kind of baseline as OpenADMET's challenge tutorial) on the training
    set and score it on the 513 unblinded test compounds. The three default configurations are
    pre-computed; change the settings and press **Train** to add your own row to the scoreboard.""",
    """    ### 5c · 「あるかないか」と「いくつあるか」

    Binary の bit が伝えるのは環境があるかどうかだけです。**Count** fingerprint は何回出てくるかを記録し、
    分子全体で合計すれば、サイズや脂溶性のおおまかな指標になります。上の仮説が正しければ、count で失われた
    情報の一部が戻ってくるはずです。下の lab で試してみてください。

    ## 6 · Model lab

    Training set で LightGBM (OpenADMET のチャレンジ tutorial と同じ種類のベースライン) を学習し、
    正解公開済みの test 513 化合物でスコアを出します。
    デフォルトの 3 設定は計算済みです。設定を変えて **Train** を押すと、スコアボードに自分の行が追加されます。""",
)
r('label="features"', 'label="特徴量"')
r('label="include chirality"', 'label="chirality を含める"')
r(
    'mo.status.spinner("Training the three reference models…")',
    'mo.status.spinner("基準モデル 3 つを学習中…")',
)
r('mo.md("Pick at least one feature family.")', 'mo.md("特徴量を 1 つ以上選んでください。")')
r('mo.status.spinner("Training…")', 'mo.status.spinner("学習中…")')
r(
    'label=f"Scoreboard (true pEC50 SD on test = {y_test.std():.2f}; pick a row to inspect it)"',
    'label=f"スコアボード (test の正解 pEC50 の SD = {y_test.std():.2f}、行を選ぶと下に詳細が出ます)"',
)
r('title="error by true potency"', 'title="正解の活性帯ごとの誤差"')
r(
    '''                f"Predictions span an SD of **{_pred.std():.2f}** against **{y_test.std():.2f}** for the truth. "
                "Every model here regresses toward the middle; the question is how much. Weak compounds are "
                "predicted too potent, potent compounds too weak — and fingerprint-only models compress the most."''',
    '''                f"予測の SD は **{_pred.std():.2f}** で、正解の **{y_test.std():.2f}** より小さくなっています。"
                "どのモデルも予測が中央に寄りますが、問題はその程度です。弱い化合物は強めに、強い化合物は弱めに"
                "予測され、fingerprint だけのモデルが最も強く縮みます。"''',
)

r(
    """    Things worth trying in the lab:

    * **Bits → counts** at the same radius and length. Counts add "how much" information for free.
    * **256 → 8192 bits.** Fewer collisions help a little. Collisions are real (see the explorer),
      but they are not the main story.
    * **Include chirality.** Only a handful of compounds change, so the score barely moves — but for
      the stereoisomer pairs in 5a it is the difference between "impossible" and "possible".
    * **Morgan + 14 descriptors.** Fourteen simple whole-molecule numbers repair much of what 2048
      bits miss.

    ## 7 · Take-aways

    1. **The PXR test set is in domain by design.** It is a hit-expansion set: every test compound was
       picked as an ECFP4 neighbour of a potent hit, so it is closer to the training data than training
       compounds are to each other, and its nearest neighbour is usually potent.
    2. **Similarity does not transfer activity here.** The nearest training neighbour's pEC50 is about
       as informative as a random training compound's. The similarity–activity curve falls, but slowly.
    3. **Much of the signal is global.** Lipophilicity and size explain a share of the variance that a
       bag of local environments cannot express, and they account for part of what looks like cliffs.
    4. **Some "cliffs" are assay floor.** Weak compounds carry very wide confidence intervals; treat
       differences at the bottom of the scale with suspicion.
    5. **Fingerprints are still useful — as one ingredient.** Counts, larger folds, chirality and a few
       descriptors each recover part of the gap; the challenge's best models went further with learned
       embeddings and structure-based features.

    ---

    ### About this notebook

    * **Data:** [openadmet/pxr-challenge-train-test](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test)
      (CC-BY-4.0), training set plus the phase 1 and phase 2 unblinded test labels.
    * **Widgets:** `MolGrid` and `MorganExplorer` are custom anywidget components written for this
      notebook ([source](https://github.com/N283T/openadmet-marimo)). Molecules are drawn in the browser
      with RDKit.js; fingerprints, bit environments and collisions are computed with RDKit in Python.
    * **Related:** Pat Walters' [marimo-chem-utils](https://github.com/PatWalters/marimo_chem_utils) and
      [practical cheminformatics tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials);
      [mols2grid](https://github.com/cbouy/mols2grid), which inspired the grid.
    * **AI use:** Built together with Claude (Anthropic) as a coding assistant for the widgets and
      notebook scaffolding. The question, analysis choices and interpretation come from my own
      PXR challenge work, and every number shown is computed live in this notebook.""",
    """    Lab で試してみる価値があること:

    * **Bits → counts** (radius と長さはそのまま)。Count にするだけで「いくつあるか」の情報がタダで加わります。
    * **256 → 8192 bits.** Collision が減ると少し良くなります。Collision は実在しますが (explorer 参照)、
      主役ではありません。
    * **Chirality を含める。** 変わる化合物はわずかなのでスコアはほとんど動きません。ただし 5a の立体異性体の
      組にとっては、「原理的に不可能」が「可能」に変わる違いです。
    * **Morgan + 14 記述子.** 分子全体を表すシンプルな 14 個の数字が、2048 bit の取りこぼしの多くを補います。

    ## 7 · まとめ

    1. **PXR の test set は設計上 domain の内側にある。** Hit expansion のセットで、どの test 化合物も
       強活性ヒットの ECFP4 近傍として選ばれている。そのため training 化合物同士よりも training set に近く、
       nearest neighbour はたいてい強活性。
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
      表示している数値はすべて、この notebook の中でその場で計算しています。""",
)

missing = [o[:70] for o, _ in R if o not in src]
if missing:
    raise SystemExit("not found:\n" + "\n".join(missing))
out = src
for o, n in R:
    out = out.replace(o, n)
Path("notebooks/pxr_fingerprints_ja.py").write_text(out)
print("ok", len(R), "replacements")
