import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(r"""
    # Pair comparison: design candidates

    Four ways to show two molecules against each other, all built on the same data
    (`pair_alignment`: MCS mapping, B laid out on A's coordinates, both drawn in one shared frame).
    Pick an example pair or type two SMILES. Blue = only in A, amber = only in B,
    violet = matched but element / charge / stereo differs.
    """)
    return (mo,)


@app.cell
def _():
    from pathlib import Path

    import anywidget
    import numpy as np
    import polars as pl
    import traitlets
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry import Point2D

    from molwidgets import fingerprint_matrix, pair_alignment, tanimoto_matrix

    return (
        Chem,
        Crippen,
        Descriptors,
        Lipinski,
        Path,
        Point2D,
        anywidget,
        fingerprint_matrix,
        np,
        pair_alignment,
        pl,
        rdMolDescriptors,
        rdMolDraw2D,
        tanimoto_matrix,
        traitlets,
    )


@app.cell
def _(Path, pl):
    # PXR pairs use real measured pEC50 values; the drug pairs carry no activity values.
    _data = Path(__file__).parent.parent / "data"
    _pec50 = {}
    _smiles = {}
    for _f in ["pxr-challenge_TRAIN.csv", "pxr-challenge_TEST_PHASE_2_UNBLINDED.csv"]:
        if (_data / _f).exists():
            _df = pl.read_csv(_data / _f).select("Molecule Name", "SMILES", "pEC50")
            _smiles.update(dict(zip(_df["Molecule Name"], _df["SMILES"])))
            _pec50.update(dict(zip(_df["Molecule Name"], _df["pEC50"])))

    def _pxr(i):
        return {"id": i, "smiles": _smiles[i], "values": {"pEC50": _pec50[i]}}

    def _drug(name, smi):
        return {"id": name, "smiles": smi, "values": {}}

    EXAMPLES = {}
    if _smiles:
        EXAMPLES["PXR · one methyl, ~50-fold"] = (
            _pxr("OADMET-0001944"),
            _pxr("OADMET-0002007"),
        )
        EXAMPLES["PXR · CH → N and Br moved"] = (
            _pxr("OADMET-0002810"),
            _pxr("OADMET-0006254"),
        )
    EXAMPLES |= {
        "paracetamol → phenacetin": (
            _drug("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
            _drug("phenacetin", "CCOc1ccc(NC(C)=O)cc1"),
        ),
        "haloperidol → bromperidol": (
            _drug("haloperidol", "C1CN(CCC1(C2=CC=C(C=C2)Cl)O)CCCC(=O)C3=CC=C(C=C3)F"),
            _drug("bromperidol", "C1CN(CCC1(C2=CC=C(C=C2)Br)O)CCCC(=O)C3=CC=C(C=C3)F"),
        ),
        "morphine → codeine": (
            _drug("morphine", "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O"),
            _drug("codeine", "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)OC)O[C@H]3[C@H](C=C4)O"),
        ),
        "sildenafil → vardenafil": (
            _drug("sildenafil", "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"),
            _drug(
                "vardenafil", "CCCC1=NC(=C2N1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)CC)OCC)C"
            ),
        ),
        "aspirin vs caffeine (unrelated)": (
            _drug("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
            _drug("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ),
        "ibuprofen vs atorvastatin (unrelated)": (
            _drug("ibuprofen", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
            _drug(
                "atorvastatin",
                "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
            ),
        ),
        "(R)- vs (S)-thalidomide": (
            _drug("(R)-thalidomide", "O=C1CC[C@@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"),
            _drug("(S)-thalidomide", "O=C1CC[C@H](N2C(=O)c3ccccc3C2=O)C(=O)N1"),
        ),
    }
    return (EXAMPLES,)


@app.cell
def _(EXAMPLES, mo):
    pick = mo.ui.dropdown(list(EXAMPLES), value=next(iter(EXAMPLES)), label="example pair")
    pick
    return (pick,)


@app.cell
def _(EXAMPLES, mo, pick):
    _a, _b = EXAMPLES[pick.value]
    smi_a = mo.ui.text(_a["smiles"], label="A", full_width=True)
    smi_b = mo.ui.text(_b["smiles"], label="B", full_width=True)
    mo.vstack([smi_a, smi_b])
    return smi_a, smi_b


@app.cell
def _(
    Chem,
    Crippen,
    Descriptors,
    Lipinski,
    Point2D,
    fingerprint_matrix,
    np,
    pair_alignment,
    rdMolDescriptors,
    rdMolDraw2D,
    tanimoto_matrix,
):
    ONLY_A = (0.66, 0.8, 1.0)
    ONLY_B = (1.0, 0.8, 0.45)
    CHANGED = (0.84, 0.62, 1.0)
    CLEAR = (1.0, 1.0, 1.0, 0.0)
    # typical ranges, so bars and dumbbells mean the same thing for every pair
    PROPS = {
        "MW": (Descriptors.MolWt, (0, 600)),  # ty: ignore[unresolved-attribute]
        "cLogP": (Crippen.MolLogP, (-2, 7)),  # ty: ignore[unresolved-attribute]
        "TPSA": (rdMolDescriptors.CalcTPSA, (0, 160)),
        "HBD": (Lipinski.NumHDonors, (0, 6)),  # ty: ignore[unresolved-attribute]
        "HBA": (Lipinski.NumHAcceptors, (0, 12)),  # ty: ignore[unresolved-attribute]
        "RotB": (rdMolDescriptors.CalcNumRotatableBonds, (0, 12)),
        "heavy atoms": (lambda m: m.GetNumHeavyAtoms(), (0, 45)),
    }
    VALUE_RANGES = {"pEC50": (2, 8)}
    # name shown, unit, and the difference that counts as large (the Δ bar is full there)
    PROP_INFO = {
        "MW": ("molecular weight", "g/mol", 100),
        "cLogP": ("calculated logP (Crippen)", "", 1.0),
        "TPSA": ("topological polar surface area", "Å²", 20),
        "HBD": ("H-bond donors", "", 2),
        "HBA": ("H-bond acceptors", "", 2),
        "RotB": ("rotatable bonds", "", 3),
        "heavy atoms": ("heavy atoms", "", 6),
    }

    def draw(mol, frame, size, atom_cols=None, bond_cols=None):
        """Draw in a fixed frame (shared by both molecules), with every atom highlighted
        (most invisibly) so the layout never depends on what is marked."""
        w, h = size
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        o = d.drawOptions()
        o.clearBackground = False
        o.addStereoAnnotation = True
        o.padding = 0.02
        d.SetScale(w, h, Point2D(*frame[0]), Point2D(*frame[1]))
        cols = {i: CLEAR for i in range(mol.GetNumAtoms())}
        cols.update(atom_cols or {})
        bond_cols = bond_cols or {}
        d.DrawMolecule(
            rdMolDraw2D.PrepareMolForDrawing(mol),
            highlightAtoms=list(cols),
            highlightAtomColors=cols,
            highlightBonds=list(bond_cols),
            highlightBondColors=bond_cols,
        )
        coords = [
            [round(p.x, 1), round(p.y, 1)]
            for p in (d.GetDrawCoords(i) for i in range(mol.GetNumAtoms()))
        ]
        d.FinishDrawing()
        return {"svg": d.GetDrawingText(), "coords": coords}

    def small(smiles, size=(150, 90)):
        mol = Chem.MolFromSmiles(smiles)
        d = rdMolDraw2D.MolDraw2DSVG(*size)
        d.drawOptions().clearBackground = False
        d.drawOptions().fixedBondLength = 34
        d.drawOptions().explicitMethyl = True  # a lone methyl is otherwise just a line
        d.drawOptions().dummiesAreAttachments = True  # * drawn as a wavy attachment bond
        if mol is None:  # e.g. a piece of an aromatic ring on its own
            return None
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
        d.FinishDrawing()
        return d.GetDrawingText()

    def _components(mol, only):
        only, seen, out = set(only), set(), []
        for s in sorted(only):
            if s in seen:
                continue
            comp, stack = set(), [s]
            while stack:
                x = stack.pop()
                if x not in comp:
                    comp.add(x)
                    stack += [
                        n.GetIdx()
                        for n in mol.GetAtomWithIdx(x).GetNeighbors()
                        if n.GetIdx() in only
                    ]
            seen |= comp
            out.append(sorted(comp))
        return out

    def _fragment(mol, comp):
        """The component as SMILES, with * where it hangs on the common part."""
        cut, site = [], set()
        for x in comp:
            for n in mol.GetAtomWithIdx(x).GetNeighbors():
                if n.GetIdx() not in comp:
                    cut.append(mol.GetBondBetweenAtoms(x, n.GetIdx()).GetIdx())
                    site.add(n.GetIdx())
        if not cut:
            return Chem.MolFragmentToSmiles(mol, atomsToUse=comp), site
        fm = Chem.FragmentOnBonds(
            mol, sorted(set(cut)), addDummies=True, dummyLabels=[(0, 0)] * len(set(cut))
        )
        frag = next(f for f in Chem.GetMolFrags(fm) if comp[0] in f)
        return Chem.MolFragmentToSmiles(fm, atomsToUse=list(frag)), site

    def edits(al):
        """What turns A into B: substituents added / removed / replaced, atoms swapped, stereo flipped."""
        ma, mb = al["mol_a"], al["mol_b"]
        a2b = dict(al["mapping"])
        b2a = {j: i for i, j in al["mapping"]}
        sites: dict = {}
        for side, mol, only, to_a in (
            ("a", ma, al["only_a"], lambda i: i),
            ("b", mb, al["only_b"], lambda j: b2a.get(j, -1)),
        ):
            for comp in _components(mol, only):
                smi, site = _fragment(mol, comp)
                key = frozenset(to_a(x) for x in site) or frozenset([f"{side}{comp[0]}"])
                e = sites.setdefault(key, {"a": [], "b": [], "atoms_a": [], "atoms_b": []})
                e[side].append(smi)
                e[f"atoms_{side}"] += comp
                # the attachment atoms, so hovering shows where it sits
                if side == "a":
                    e["atoms_b"] += [a2b[x] for x in site if x in a2b]
                else:
                    e["atoms_a"] += [b2a[x] for x in site if x in b2a]
        out = []
        for e in sites.values():
            a = ".".join(e["a"]) or "[H]*"
            b = ".".join(e["b"]) or "[H]*"
            kind = "added" if not e["a"] else "removed" if not e["b"] else "replaced"
            out.append(
                {
                    "kind": kind,
                    "a": a,
                    "b": b,
                    "svg_a": None if a == "[H]*" else small(a),
                    "svg_b": None if b == "[H]*" else small(b),
                    "atoms_a": sorted(set(e["atoms_a"])),
                    "atoms_b": sorted(set(e["atoms_b"])),
                }
            )
        cip = lambda m: dict(
            Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
        )
        cip_a, cip_b = cip(ma), cip(mb)
        for i, j in zip(al["changed_a"], al["changed_b"]):
            x, y = ma.GetAtomWithIdx(i), mb.GetAtomWithIdx(j)
            if x.GetSymbol() != y.GetSymbol():
                kind, a, b = "atom swap", x.GetSymbol(), y.GetSymbol()
            elif x.GetFormalCharge() != y.GetFormalCharge():
                kind, a, b = "charge", f"{x.GetFormalCharge():+d}", f"{y.GetFormalCharge():+d}"
            else:
                kind, a, b = "stereo", cip_a.get(i, "–"), cip_b.get(j, "–")
            out.append({"kind": kind, "a": a, "b": b, "atoms_a": [i], "atoms_b": [j]})
        for bi, bj in zip(al["changed_bonds_a"], al["changed_bonds_b"]):
            ba, bb = ma.GetBondWithIdx(bi), mb.GetBondWithIdx(bj)
            name = lambda s: str(s).replace("STEREO", "")
            out.append(
                {
                    "kind": "E/Z",
                    "a": name(ba.GetStereo()),
                    "b": name(bb.GetStereo()),
                    "atoms_a": [ba.GetBeginAtomIdx(), ba.GetEndAtomIdx()],
                    "atoms_b": [bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()],
                }
            )
        return out

    def pair_data(a, b, width=360):
        al = pair_alignment(a["smiles"], b["smiles"])
        ma, mb = al["mol_a"], al["mol_b"]
        pts = np.vstack([m.GetConformer().GetPositions()[:, :2] for m in (ma, mb)])
        frame = (pts.min(0) - 0.7, pts.max(0) + 0.7)
        # the canvas takes the molecules' aspect ratio, so no side is mostly empty
        span = frame[1] - frame[0]
        size = (width, int(width * min(0.85, max(0.5, span[1] / span[0]))))
        col_a = {i: ONLY_A for i in al["only_a"]} | {i: CHANGED for i in al["changed_a"]}
        col_b = {i: ONLY_B for i in al["only_b"]} | {i: CHANGED for i in al["changed_b"]}
        bonds_a = {i: ONLY_A for i in al["only_bonds_a"]} | {
            i: CHANGED for i in al["changed_bonds_a"]
        }
        bonds_b = {i: ONLY_B for i in al["only_bonds_b"]} | {
            i: CHANGED for i in al["changed_bonds_b"]
        }
        props = []
        for name in a["values"]:
            va, vb = a["values"][name], b["values"].get(name)
            if vb is None:
                continue
            lo, hi = VALUE_RANGES.get(name, (min(va, vb) - 1, max(va, vb) + 1))
            props.append({"name": name, "a": va, "b": vb, "range": [lo, hi], "value": True})
        for name, (fn, rng) in PROPS.items():
            props.append({"name": name, "a": float(fn(ma)), "b": float(fn(mb)), "range": list(rng)})
        fps = fingerprint_matrix([a["smiles"], b["smiles"]])
        return {
            "ids": [a["id"], b["id"]],
            "panels": [
                {
                    **draw(ma, frame, size, col_a, bonds_a),
                    "symbols": [x.GetSymbol() for x in ma.GetAtoms()],
                },
                {
                    **draw(mb, frame, size, col_b, bonds_b),
                    "symbols": [x.GetSymbol() for x in mb.GetAtoms()],
                },
            ],
            "mapping": [list(p) for p in al["mapping"]],
            "tanimoto": float(tanimoto_matrix(fps)[0, 1]),
            "props": props,
            "edits": edits(al),
            "counts": {
                "shared": len(al["mapping"]),
                "only_a": len(al["only_a"]),
                "only_b": len(al["only_b"]),
                "changed": len(al["changed_a"]) + len(al["changed_bonds_a"]),
            },
        }

    def _props(a, b, ma, mb):
        props = []
        for name in a["values"]:
            va, vb = a["values"][name], b["values"].get(name)
            if vb is None:
                continue
            lo, hi = VALUE_RANGES.get(name, (min(va, vb) - 1, max(va, vb) + 1))
            props.append(
                {"name": name, "a": va, "b": vb, "range": [lo, hi], "value": True, "scale": 1.0}
            )
        for name, (fn, rng) in PROPS.items():
            label, unit, scale = PROP_INFO[name]
            props.append(
                {
                    "name": name,
                    "label": label,
                    "unit": unit,
                    "scale": scale,
                    "a": float(fn(ma)),
                    "b": float(fn(mb)),
                    "range": list(rng),
                }
            )
        return props

    CORE = (0.8, 0.93, 0.8)  # the common part, when both are shown

    def scaffold(al):
        """The common part as laid out in A, with * wherever either molecule carries
        something the other lacks; element changes are labelled like 'C/N'."""
        from rdkit.Geometry import Point3D

        ma, mb = al["mol_a"], al["mol_b"]
        a2b = dict(al["mapping"])
        b2a = {j: i for i, j in al["mapping"]}
        core = set(a2b)
        m = Chem.RWMol(ma)
        Chem.Kekulize(m, clearAromaticFlags=True)
        conf = m.GetConformer()
        xy = lambda k: np.array(conf.GetAtomPosition(k))[:2]
        sites = []
        for i in core:
            for n in ma.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() not in core:  # A has something here
                    sites.append((i, xy(n.GetIdx())))
            for n in mb.GetAtomWithIdx(a2b[i]).GetNeighbors():
                if n.GetIdx() not in b2a:  # B has something here: point away from the ring/chain
                    nb = [
                        xy(x.GetIdx())
                        for x in ma.GetAtomWithIdx(i).GetNeighbors()
                        if x.GetIdx() in core
                    ]
                    away = xy(i) - np.mean(nb, axis=0) if nb else np.array([1.0, 0.0])
                    sites.append((i, xy(i) + 1.3 * away / (np.linalg.norm(away) or 1)))
        changed = []
        for i in al["changed_a"]:
            at, other = m.GetAtomWithIdx(i), mb.GetAtomWithIdx(a2b[i])
            if at.GetSymbol() != other.GetSymbol():
                at.SetProp("atomLabel", f"{at.GetSymbol()}/{other.GetSymbol()}")
        for i in sorted(set(range(ma.GetNumAtoms())) - core, reverse=True):
            m.RemoveAtom(i)
        new = {old: k for k, old in enumerate(sorted(core))}
        for i, pos in sites:
            d = m.AddAtom(Chem.Atom(0))
            m.AddBond(new[i], d, Chem.BondType.SINGLE)
            m.GetConformer().SetAtomPosition(d, Point3D(float(pos[0]), float(pos[1]), 0.0))
        changed = [new[i] for i in al["changed_a"]]
        out = m.GetMol()
        out.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(out)
        return out, changed

    def draw_scaffold(al, frame, size):
        mol, changed = scaffold(al)
        w, h = size
        d = rdMolDraw2D.MolDraw2DSVG(w, h)
        o = d.drawOptions()
        o.clearBackground = False
        o.padding = 0.02
        o.dummiesAreAttachments = True
        d.SetScale(w, h, Point2D(*frame[0]), Point2D(*frame[1]))
        cols = {i: CLEAR for i in range(mol.GetNumAtoms())} | {i: CHANGED for i in changed}
        d.DrawMolecule(
            rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False),
            highlightAtoms=list(cols),
            highlightAtomColors=cols,
            highlightBonds=[],
        )
        d.FinishDrawing()
        return d.GetDrawingText()

    def compare_data(a, b, analysed=False, width=340):
        """Any two molecules side by side. Only when `analysed` is the MCS searched: then B is
        laid out in A's orientation and the differences are marked and listed."""
        from rdkit.Chem import rdDepictor

        plain = [Chem.MolFromSmiles(x["smiles"]) for x in (a, b)]
        if None in plain:
            raise ValueError("cannot parse " + ("A" if plain[0] is None else "B"))
        for m in plain:
            rdDepictor.Compute2DCoords(m)  # same call as pair_alignment: A is identical
        al = pair_alignment(a["smiles"], b["smiles"]) if analysed else None
        ma, mb = (al["mol_a"], al["mol_b"]) if al else plain
        # each molecule centred in its own panel, both at one scale. The frame also covers
        # B's un-analysed layout and is rounded up to whole Å, so pressing the button
        # re-lays out B without moving or rescaling A (unless the new B is larger)
        pos = [m.GetConformer().GetPositions()[:, :2] for m in (ma, mb)]
        centre = [(p.min(0) + p.max(0)) / 2 for p in pos]
        extent = pos + [plain[1].GetConformer().GetPositions()[:, :2]]
        half = np.ceil(np.max([(p.max(0) - p.min(0)) / 2 for p in extent], axis=0) + 0.7)
        size = (width, int(width * min(0.85, max(0.5, half[1] / half[0]))))
        cols = [{}, {}], [{}, {}]
        if al:
            # common part pale green, differences in their colours, both at once
            core_bonds = [
                {
                    bd.GetIdx()
                    for bd in m.GetBonds()
                    if bd.GetBeginAtomIdx() in mine and bd.GetEndAtomIdx() in mine
                }
                for m, mine in (
                    (ma, {p[0] for p in al["mapping"]}),
                    (mb, {p[1] for p in al["mapping"]}),
                )
            ]
            cols = (
                [
                    {p[0]: CORE for p in al["mapping"]}
                    | {i: ONLY_A for i in al["only_a"]}
                    | {i: CHANGED for i in al["changed_a"]},
                    {p[1]: CORE for p in al["mapping"]}
                    | {i: ONLY_B for i in al["only_b"]}
                    | {i: CHANGED for i in al["changed_b"]},
                ],
                [
                    {i: CORE for i in core_bonds[0] - set(al["only_bonds_a"])}
                    | {i: ONLY_A for i in al["only_bonds_a"]}
                    | {i: CHANGED for i in al["changed_bonds_a"]},
                    {i: CORE for i in core_bonds[1] - set(al["only_bonds_b"])}
                    | {i: ONLY_B for i in al["only_bonds_b"]}
                    | {i: CHANGED for i in al["changed_bonds_b"]},
                ],
            )
        panels = [
            {
                **draw(m, (c - half, c + half), size, cols[0][k], cols[1][k]),
                "symbols": [x.GetSymbol() for x in m.GetAtoms()],
            }
            for k, (m, c) in enumerate(zip((ma, mb), centre))
        ]
        fps = fingerprint_matrix([a["smiles"], b["smiles"]])
        return {
            "ids": [a["id"], b["id"]],
            "panels": panels,
            "tanimoto": float(tanimoto_matrix(fps)[0, 1]),
            "props": _props(a, b, ma, mb),
            "analysed": bool(al),
            "mapping": [list(p) for p in al["mapping"]] if al else [],
            "edits": edits(al) if al else [],
            "scaffold": draw_scaffold(al, (centre[0] - half, centre[0] + half), size)
            if al and al["mapping"]
            else "",
            "counts": {
                "shared": len(al["mapping"]),
                "n": [ma.GetNumAtoms(), mb.GetNumAtoms()],
                "only_a": len(al["only_a"]),
                "only_b": len(al["only_b"]),
                "changed": len(al["changed_a"]) + len(al["changed_bonds_a"]),
                "timed_out": al["timed_out"],
            }
            if al
            else None,
        }

    return compare_data, pair_data


@app.cell
def _(EXAMPLES, mo, pair_data, pick, smi_a, smi_b):
    # typed SMILES replace the example (and its values)
    _a, _b = EXAMPLES[pick.value]
    if smi_a.value != _a["smiles"]:
        _a = {"id": "A", "smiles": smi_a.value, "values": {}}
    if smi_b.value != _b["smiles"]:
        _b = {"id": "B", "smiles": smi_b.value, "values": {}}
    pair = (_a, _b)
    try:
        data = pair_data(_a, _b)
    except ValueError as err:
        data = None
        mo.output.replace(mo.callout(str(err), kind="danger"))
    return data, pair


@app.cell
def _():
    # shared by every candidate: colours, tokens, and molecule boxes whose hover rings are
    # placed (invisible) on every atom up front, so showing one never moves anything
    PRELUDE = r"""
    const SIDE = { a: "#3b82f6", b: "#f59e0b" };
    const NS = "http://www.w3.org/2000/svg";
    function el(tag, cls = "", html = "") { const e = document.createElement(tag); if (cls) e.className = cls; if (html) e.innerHTML = html; return e; }
    function svgEl(tag, attrs) { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; }
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
    function isDark() {
      for (const n of [document.documentElement, document.body])
        if (n.classList.contains("dark") || n.classList.contains("dark-theme")) return true;
      return false;
    }
    function num(v, name) {
      if (v === null || v === undefined) return "–";
      const d = name === "MW" || name === "TPSA" ? 1 : Number.isInteger(v) ? 0 : 2;
      return v.toFixed(d);
    }
    const H = (s) => (s === "[H]*" ? "H" : s);
    function signed(v, name) { const s = num(Math.abs(v), name); return v > 0 ? "+" + s : v < 0 ? "−" + s : s; }
    const BASE_CSS = `
    .c-root { font: 13px/1.45 system-ui, sans-serif; color: var(--fg); --fg:#1f2328; --muted:#6b7280; --border:#d0d7de; --soft:#f6f8fa; --card:#fff; }
    .c-root.dark { --fg:#e6e6e6; --muted:#9aa4b2; --border:#3a3f47; --soft:#24282e; --card:#1c1f24; }
    .c-mol { background:#fff; color:#1f2328; border:1px solid var(--border); border-radius:10px; overflow:hidden; }
    .c-mol svg { width:100%; height:auto; display:block; }
    .c-ring { fill:none; stroke-width:2.6; pointer-events:none; opacity:0; }
    .c-ring.on { opacity:1; }
    .c-hit { fill:transparent; cursor:pointer; }
    .c-id { display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:600; }
    .c-id i { width:10px; height:10px; border-radius:50%; display:inline-block; }
    .c-muted { color:var(--muted); }
    .c-info { height:1.5em; font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; }
    .c-info b { color:var(--fg); }
    `;
    function root(host, css) {
      const r = el("div", "c-root" + (isDark() ? " dark" : ""));
      r.appendChild(el("style", "", BASE_CSS + css));
      host.appendChild(r);
      return r;
    }
    function molBox(panel, side) {
      const box = el("div", "c-mol", panel.svg);
      const svg = box.querySelector("svg");
      const g = svgEl("g", {});
      svg.appendChild(g);
      const rings = panel.coords.map(([x, y]) => { const c = svgEl("circle", { cx: x, cy: y, r: 13, class: "c-ring", stroke: SIDE[side] }); g.appendChild(c); return c; });
      const hits = panel.coords.map(([x, y]) => { const c = svgEl("circle", { cx: x, cy: y, r: 11, class: "c-hit" }); svg.appendChild(c); return c; });
      return { box, svg, hits, show(list) { rings.forEach((r, i) => r.classList.toggle("on", list.includes(i))); } };
    }
    // hovering an atom in either box rings it and its partner; `info` gets a one-line description
    function linkBoxes(A, B, data, info) {
      const m = data.mapping, P = data.panels;
      const partner = (side, i) => { const h = m.find((p) => p[side === "a" ? 0 : 1] === i); return h ? h[side === "a" ? 1 : 0] : undefined; };
      const clear = () => { A.show([]); B.show([]); info && (info.innerHTML = "hover an atom to find its partner"); };
      for (const [side, box, other] of [["a", A, B], ["b", B, A]]) {
        box.hits.forEach((h, i) => {
          h.addEventListener("mouseenter", () => {
            const j = partner(side, i);
            box.show([i]); other.show(j === undefined ? [] : [j]);
            if (!info) return;
            const o = side === "a" ? "b" : "a";
            const lab = (s, k) => `<b style="color:${SIDE[s]}">${s.toUpperCase()}</b> ${P[s === "a" ? 0 : 1].symbols[k]}${k}`;
            info.innerHTML = j === undefined ? `${lab(side, i)} · no partner in ${o.toUpperCase()}` : `${lab("a", side === "a" ? i : j)} ↔ ${lab("b", side === "a" ? j : i)}`;
          });
          h.addEventListener("mouseleave", clear);
        });
      }
      clear();
      return clear;
    }
    """
    return (PRELUDE,)


@app.cell
def _(PRELUDE, anywidget, traitlets):
    def candidate(js: str):
        class _Candidate(anywidget.AnyWidget):
            _esm = PRELUDE + js
            data = traitlets.Dict().tag(sync=True)

        return _Candidate

    return (candidate,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 0 · Compare (any two molecules; structure analysis on demand)

    Two molecules, similar or not, with their properties on dumbbells. **Compare
    structures** runs the MCS only when asked: B is then turned to A's orientation, the
    differences are coloured and listed, and hovering an atom finds its partner.
    """)
    return


@app.cell
def _(PRELUDE, anywidget, compare_data, traitlets):
    COMPARE_JS = r"""
    const CSS = `
    .c-root { --track:#eef1f4; } .c-root.dark { --track:#2c3139; }
    .mc-bar { display:flex; flex-wrap:wrap; align-items:center; gap:6px 14px; margin-bottom:8px; font-size:12.5px; }
    .mc-btn { font:inherit; font-size:12.5px; border:1px solid var(--border); background:var(--soft); color:var(--fg);
      border-radius:6px; padding:3px 12px; cursor:pointer; }
    .mc-btn:disabled { opacity:.6; cursor:default; }
    .mc-grp i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
    .mc-cols { display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr) minmax(260px, 300px); gap:12px; align-items:start; }
    @media (max-width: 860px) { .mc-cols { grid-template-columns: 1fr 1fr; } .mc-cols > .pp { grid-column: 1 / -1; } }
    .mc-card .c-id { margin-bottom:3px; }

    /* property panel */
    .pp { border:1px solid var(--border); border-radius:8px; background:var(--card); overflow:hidden; font-variant-numeric:tabular-nums; }
    .pp-sec { padding:9px 12px 10px; }
    .pp-sec + .pp-sec { border-top:1px solid var(--border); }
    .pp-h { font-size:10.5px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
    .pp-sim { display:flex; align-items:center; gap:12px; }
    .pp-sim b { font-size:24px; font-weight:700; line-height:1; min-width:52px; }
    .pp-sim .meter { flex:1; }
    .pp-sim .track { height:8px; border-radius:4px; background:var(--track); overflow:hidden; }
    .pp-sim .track i { display:block; height:100%; background:var(--fg); opacity:.55; border-radius:4px; }
    .pp-sim .ticks { display:flex; justify-content:space-between; font-size:10px; color:var(--muted); margin-top:2px; }
    .pp-act + .pp-act { margin-top:8px; }
    .pp-act .name { font-size:12px; font-weight:600; margin-bottom:2px; }
    .pp-act .vals { display:flex; align-items:baseline; gap:10px; }
    .pp-act .vals b { font-size:22px; font-weight:700; line-height:1.1; }
    .pp-act .vals .to { color:var(--muted); font-size:16px; }
    .pp-act .chip { display:inline-block; margin-top:4px; font-size:12px; padding:1px 8px; border-radius:999px; background:var(--soft);
      border:1px solid var(--border); }
    .pp-t { width:100%; border-collapse:collapse; font-size:12.5px; }
    .pp-t th { font-size:11px; font-weight:600; color:var(--muted); text-align:right; padding:0 6px 4px; }
    .pp-t th:first-child { text-align:left; padding-left:0; }
    .pp-t td { padding:4px 5px; text-align:right; border-top:1px solid var(--border); white-space:nowrap; }
    .pp-t td:first-child { text-align:left; padding-left:0; }
    .pp-t td:last-child { padding-right:0; }
    .pp-t .unit { color:var(--muted); font-size:11px; margin-left:3px; }
    .pp-t td.hi { font-weight:700; }
    .pp-t tr.same td { color:var(--muted); }
    .pp-t tr.same td.hi { font-weight:400; }
    .pp-t tr:hover td { background:var(--soft); }
    .dlt { display:inline-flex; align-items:center; gap:6px; justify-content:flex-end; }
    .dlt .v { min-width:36px; text-align:right; }
    .dlt .v.big { font-weight:700; }
    /* diverging bar: centre = no change, full half = the difference that counts as large */
    .dbar { position:relative; width:40px; height:10px; }
    .dbar::before { content:""; position:absolute; left:50%; top:0; bottom:0; width:1px; background:var(--border); }
    .dbar::after { content:""; position:absolute; left:0; right:0; top:4px; height:2px; background:var(--track); z-index:-1; }
    .dbar i { position:absolute; top:2px; height:6px; border-radius:2px; background:var(--fg); opacity:.55; }
    .dbar i.big { opacity:.85; }

    /* radar: A and B as two polygons on the typical ranges */
    .pp-hrow { display:flex; align-items:center; justify-content:space-between; margin-bottom:4px; }
    .pp-hrow .pp-h { margin:0; }
    .pp-sw { display:inline-flex; border:1px solid var(--border); border-radius:5px; overflow:hidden; }
    .pp-sw button { font:inherit; font-size:11px; color:var(--muted); background:var(--card); border:0; padding:0 7px; cursor:pointer; line-height:18px; }
    .pp-sw button + button { border-left:1px solid var(--border); }
    .pp-sw button.on { background:var(--soft); color:var(--fg); font-weight:600; }
    .rd svg { width:100%; height:auto; display:block; overflow:visible; }
    .rd .grid { fill:none; stroke:var(--border); }
    .rd .spoke { stroke:var(--border); }
    .rd .spoke.on { stroke:var(--fg); stroke-width:1.5; }
    .rd .lab { font-size:11px; fill:var(--muted); }
    .rd .lab.on { fill:var(--fg); font-weight:700; }
    .rd .lab.diff { fill:var(--fg); }
    .rd .poly { stroke-width:2; stroke-linejoin:round; }
    .rd .pt { stroke:var(--card); stroke-width:1.5; }
    .rd .pt.on { r:5; }
    .rd .hit { fill:transparent; cursor:default; }
    .rd-read { height:34px; border-top:1px solid var(--border); padding-top:5px; font-size:12px; font-variant-numeric:tabular-nums; }
    .rd-read .t { font-weight:600; }
    .rd-read .u { color:var(--muted); font-size:11px; }
    .rd-read .row { display:flex; gap:12px; align-items:baseline; }

    /* structure comparison */
    .mc-cmp { display:grid; grid-template-columns: minmax(0, 320px) minmax(0,1fr); gap:12px; margin-top:10px; align-items:start; }
    @media (max-width: 760px) { .mc-cmp { grid-template-columns: 1fr; } }
    .mc-cmp h4 { margin:0 0 4px; font-size:10.5px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; color:var(--muted); }
    .mc-edits { display:flex; flex-direction:column; gap:4px; }
    .mc-edit { display:grid; grid-template-columns: 76px 84px 16px 84px 1fr; align-items:center; gap:8px;
      border:1px solid var(--border); border-radius:8px; padding:2px 10px; font-size:12px; }
    .mc-edit:hover { background:var(--soft); }
    .mc-edit .k { font-size:10.5px; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); font-weight:600; }
    .mc-edit .f { background:#fff; border-radius:6px; height:44px; display:flex; align-items:center; justify-content:center;
      font:600 15px ui-monospace, monospace; color:#1f2328; }
    .mc-edit .f svg { height:44px; width:auto; }
    .mc-edit .to { text-align:center; color:var(--muted); }
    .mc-edit .s { font:11px ui-monospace, monospace; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    `;
    const GROUP = { core: "rgb(204,237,204)", a: "rgb(168,204,255)", b: "rgb(255,204,115)", c: "rgb(214,158,255)" };
    let view = "radar"; // kept across redraws (e.g. after compare structures)
    const unit = (p) => (p.unit ? `<span class="unit">${esc(p.unit)}</span>` : "");

    function table(d) {
      const rows = d.props.filter((p) => !p.value).map((p) => {
        const dv = p.b - p.a, same = Math.abs(dv) < 1e-9, t = Math.min(1, Math.abs(dv) / p.scale), big = t >= 0.5;
        const fill = same ? "" : `<i class="${big ? "big" : ""}" style="${dv > 0 ? "left:50%" : `left:${50 - 50 * t}%`};width:${Math.max(2, 50 * t)}%"></i>`;
        return `<tr class="${same ? "same" : ""}" title="${esc(p.label)}${p.unit ? ` (${esc(p.unit)})` : ""}"><td>${esc(p.name)}${unit(p)}</td>` +
          `<td class="${p.a > p.b ? "hi" : ""}">${num(p.a, p.name)}</td><td class="${p.b > p.a ? "hi" : ""}">${num(p.b, p.name)}</td>` +
          `<td><span class="dlt"><span class="dbar">${fill}</span><span class="v${big ? " big" : ""}">${same ? "–" : signed(dv, p.name)}</span></span></td></tr>`;
      }).join("");
      return el("table", "pp-t", `<thead><tr><th></th><th style="color:${SIDE.a}">A</th><th style="color:${SIDE.b}">B</th><th>B − A</th></tr></thead><tbody>${rows}</tbody>`);
    }

    // one spoke per property, each on its typical range; hovering a spoke reads out A, B and Δ.
    // Every highlight element exists from the start and is only switched on, so nothing moves.
    function radar(d) {
      const ps = d.props.filter((p) => !p.value);
      const W = 280, H = 234, cx = W / 2, cy = 117, R = 92, n = ps.length;
      const ang = (k) => -Math.PI / 2 + (2 * Math.PI * k) / n;
      const at = (k, f) => [cx + R * f * Math.cos(ang(k)), cy + R * f * Math.sin(ang(k))];
      const frac = (p, v) => Math.max(0.02, Math.min(1, (v - p.range[0]) / (p.range[1] - p.range[0])));
      const ring = (f) => ps.map((_, k) => at(k, f).join(",")).join(" ");
      let svg = `<svg viewBox="0 0 ${W} ${H}">`;
      for (const f of [0.25, 0.5, 0.75, 1]) svg += `<polygon class="grid" points="${ring(f)}"/>`;
      ps.forEach((p, k) => {
        const [x, y] = at(k, 1);
        svg += `<line class="spoke" data-k="${k}" x1="${cx}" y1="${cy}" x2="${x}" y2="${y}"/>`;
        const [lx, ly] = at(k, 1.13);
        const anchor = Math.abs(lx - cx) < 4 ? "middle" : lx > cx ? "start" : "end";
        const diff = Math.abs(p.b - p.a) >= 0.5 * p.scale ? " diff" : "";
        svg += `<text class="lab${diff}" data-k="${k}" x="${lx}" y="${ly + 3.5}" text-anchor="${anchor}">${esc(p.name)}</text>`;
      });
      for (const side of ["a", "b"]) {
        const pts = ps.map((p, k) => at(k, frac(p, p[side])).join(",")).join(" ");
        svg += `<polygon class="poly" points="${pts}" fill="${SIDE[side]}" fill-opacity=".14" stroke="${SIDE[side]}"/>`;
      }
      for (const side of ["a", "b"])
        ps.forEach((p, k) => {
          const [x, y] = at(k, frac(p, p[side]));
          svg += `<circle class="pt" data-k="${k}" cx="${x}" cy="${y}" r="3.2" fill="${SIDE[side]}"/>`;
        });
      // wide invisible wedges, so a spoke is easy to hit
      ps.forEach((_, k) => {
        const [x1, y1] = at(k - 0.5, 1.35), [x2, y2] = at(k + 0.5, 1.35);
        svg += `<path class="hit" data-k="${k}" d="M${cx},${cy} L${x1},${y1} A${R * 1.35},${R * 1.35} 0 0 1 ${x2},${y2} Z"/>`;
      });
      svg += `</svg>`;
      const box = el("div", "rd", svg);
      const read = el("div", "rd-read");
      const idle = () => (read.innerHTML = `<span class="c-muted">hover a spoke for A, B and Δ · bold labels differ most</span>`);
      box.querySelectorAll(".hit").forEach((h) => {
        const k = Number(h.dataset.k), p = ps[k];
        h.addEventListener("mouseenter", () => {
          box.querySelectorAll("[data-k]").forEach((e) => e.classList.toggle("on", Number(e.dataset.k) === k && !e.classList.contains("hit")));
          const dv = p.b - p.a;
          read.innerHTML =
            `<div class="row"><span class="t">${esc(p.label)}</span>${p.unit ? `<span class="u">${esc(p.unit)}</span>` : ""}</div>` +
            `<div class="row"><span><b style="color:${SIDE.a}">A</b> ${num(p.a, p.name)}</span><span><b style="color:${SIDE.b}">B</b> ${num(p.b, p.name)}</span>` +
            `<span>Δ <b>${Math.abs(dv) < 1e-9 ? "0" : signed(dv, p.name)}</b></span></div>`;
        });
        h.addEventListener("mouseleave", () => {
          box.querySelectorAll(".on").forEach((e) => e.classList.remove("on"));
          idle();
        });
      });
      idle();
      const wrap = el("div");
      wrap.append(box, read);
      return wrap;
    }

    function panel(d) {
      const pp = el("div", "pp");
      const sim = el("div", "pp-sec");
      sim.innerHTML =
        `<div class="pp-h">Similarity · Tanimoto on ECFP4</div><div class="pp-sim"><b>${d.tanimoto.toFixed(2)}</b>` +
        `<div class="meter"><div class="track"><i style="width:${100 * d.tanimoto}%"></i></div>` +
        `<div class="ticks"><span>0</span><span>1</span></div></div></div>`;
      pp.append(sim);

      const acts = d.props.filter((p) => p.value);
      if (acts.length) {
        const sec = el("div", "pp-sec");
        sec.innerHTML = `<div class="pp-h">Activity</div>` + acts.map((p) => {
          const dv = p.b - p.a;
          return `<div class="pp-act"><div class="name">${esc(p.name)}</div><div class="vals">` +
            `<b style="color:${SIDE.a}">${num(p.a, p.name)}</b><span class="to">→</span><b style="color:${SIDE.b}">${num(p.b, p.name)}</b></div>` +
            `<span class="chip">Δ (B − A) <b>${signed(dv, p.name)}</b></span></div>`;
        }).join("");
        pp.append(sec);
      }

      const sec = el("div", "pp-sec");
      const head = el("div", "pp-hrow", `<div class="pp-h">Properties</div>`);
      const sw = el("div", "pp-sw");
      const body = el("div");
      const show = (v) => {
        view = v;
        sw.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === v));
        body.replaceChildren(v === "radar" ? radar(d) : table(d));
      };
      for (const [v, label] of [["radar", "chart"], ["table", "table"]]) {
        const b = el("button", "", label);
        b.dataset.v = v;
        b.addEventListener("click", () => show(v));
        sw.append(b);
      }
      head.append(sw);
      sec.append(head, body);
      show(view);
      pp.append(sec);
      return pp;
    }

    function render({ model, el: host }) {
      const r = root(host, CSS);
      function draw() {
        r.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
        const d = model.get("data");
        if (!d.panels) return;
        const A = molBox(d.panels[0], "a"), B = molBox(d.panels[1], "b");
        const card = (side, k, box) => {
          const c = el("div", "mc-card");
          c.append(el("div", "c-id", `<i style="background:${SIDE[side]}"></i>${side.toUpperCase()} · ${esc(d.ids[k])}`), box.box);
          return c;
        };
        const g = el("div", "mc-cols");
        g.append(card("a", 0, A), card("b", 1, B), panel(d));

        const bar = el("div", "mc-bar");
        const btn = el("button", "mc-btn", d.analysed ? "hide structure comparison" : "compare structures");
        btn.addEventListener("click", () => {
          btn.disabled = true;
          btn.textContent = d.analysed ? "…" : "searching the common substructure…";
          model.set("analysed", !d.analysed);
          model.save_changes();
        });
        bar.append(btn);
        r.append(g, bar);
        if (!d.analysed) return;

        const c = d.counts;
        const grp = (colour, text) => bar.append(el("span", "mc-grp", `<i style="background:${colour}"></i>${text}`));
        grp(GROUP.core, `common part ${c.shared} of ${c.n[0]} / ${c.n[1]} atoms`);
        if (c.only_a) grp(GROUP.a, `${c.only_a} only in A`);
        if (c.only_b) grp(GROUP.b, `${c.only_b} only in B`);
        if (c.changed) grp(GROUP.c, `${c.changed} changed`);
        if (c.timed_out) bar.append(el("span", "c-muted", "(search timed out)"));
        const info = el("div", "c-info");
        const clear = linkBoxes(A, B, d, info);
        bar.append(info);

        const cmp = el("div", "mc-cmp");
        const left = el("div");
        left.append(el("h4", "", "Common scaffold"));
        if (d.scaffold) left.append(el("div", "c-mol", d.scaffold));
        const right = el("div");
        right.append(el("h4", "", "A → B"));
        const small = c.shared < 0.5 * Math.max(...c.n);
        if (!d.edits.length) right.append(el("div", "c-muted", "No difference in the graph."));
        else if (small || d.edits.length > 6)
          right.append(el("div", "c-muted", small
            ? "The common part covers less than half of the larger molecule, so the two are not read as one edit of each other."
            : `${d.edits.length} separate differences: too many to read as a single edit.`));
        else {
          const list = el("div", "mc-edits");
          for (const e of d.edits) {
            const frag = (svg, txt) => (svg ? svg : esc(txt === "[H]*" ? "H" : txt));
            const row = el("div", "mc-edit",
              `<span class="k">${esc(e.kind)}</span><div class="f">${frag(e.svg_a, e.a)}</div><span class="to">→</span>` +
              `<div class="f">${frag(e.svg_b, e.b)}</div><span class="s">${esc(H(e.a))} → ${esc(H(e.b))}</span>`);
            row.addEventListener("mouseenter", () => { A.show(e.atoms_a); B.show(e.atoms_b); });
            row.addEventListener("mouseleave", clear);
            list.append(row);
          }
          right.append(list);
        }
        cmp.append(left, right);
        r.append(cmp);
      }
      model.on("change:data", draw);
      draw();
    }
    export default { render };
    """

    class MolCompare(anywidget.AnyWidget):
        _esm = PRELUDE + COMPARE_JS
        data = traitlets.Dict().tag(sync=True)
        analysed = traitlets.Bool(False).tag(sync=True)

        def __init__(self, a, b, **kwargs):
            self._pair = (a, b)
            super().__init__(**kwargs)
            self.observe(self._update, names=["analysed"])
            self._update()

        def _update(self, _change=None):
            self.data = compare_data(*self._pair, analysed=self.analysed)

    return (MolCompare,)


@app.cell
def _(MolCompare, data, mo, pair):
    mo.ui.anywidget(MolCompare(*pair)) if data else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · Overlay

    One drawing: B sits on A's coordinates, so the shared part coincides and only the
    difference moves. Drag the slider from A to B, or let it blink. The pair's values
    change along with it.
    """)
    return


@app.cell
def _(candidate):
    Overlay = candidate(r"""
    const CSS = `
    .ov-wrap { display:grid; grid-template-columns: minmax(0, 520px) minmax(160px, 1fr); gap:16px; align-items:start; }
    @media (max-width: 700px) { .ov-wrap { grid-template-columns: 1fr; } }
    .ov-stage { position:relative; }
    .ov-stage .c-mol + .c-mol { position:absolute; inset:0; background:transparent; border-color:transparent; }
    .ov-stage .c-mol { transition: opacity .25s; }
    .ov-ctl { display:flex; align-items:center; gap:8px; margin-top:8px; }
    .ov-ctl input { flex:1; accent-color:#6b7280; }
    .ov-btn { font:inherit; font-size:12px; border:1px solid var(--border); background:var(--soft); color:var(--fg); border-radius:6px; padding:2px 10px; cursor:pointer; }
    .ov-btn.on { background:var(--fg); color:var(--card); }
    .ov-side { display:flex; flex-direction:column; gap:10px; font-variant-numeric:tabular-nums; }
    .ov-val { display:grid; grid-template-columns: auto 1fr; gap:2px 10px; font-size:12.5px; }
    .ov-val .k { color:var(--muted); }
    .ov-edit { font-size:12.5px; }
    .ov-edit code { font:12px ui-monospace, monospace; background:var(--soft); padding:0 4px; border-radius:4px; }
    `;
    function render({ model, el: host }) {
      const r = root(host, CSS);
      function draw() {
        r.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
        const d = model.get("data");
        if (!d.panels) return;
        const wrap = el("div", "ov-wrap");
        const left = el("div");
        const stage = el("div", "ov-stage");
        const A = molBox(d.panels[0], "a"), B = molBox(d.panels[1], "b");
        stage.append(A.box, B.box);
        const slider = Object.assign(el("input"), { type: "range", min: 0, max: 100, value: 0 });
        const blink = el("button", "ov-btn", "blink");
        const ctl = el("div", "ov-ctl");
        ctl.append(el("span", "c-id", `<i style="background:${SIDE.a}"></i>${esc(d.ids[0])}`), slider,
                   el("span", "c-id", `<i style="background:${SIDE.b}"></i>${esc(d.ids[1])}`), blink);
        left.append(stage, ctl);
        const side = el("div", "ov-side");
        const vals = el("div", "ov-val");
        side.append(el("div", "", `Tanimoto <b>${d.tanimoto.toFixed(2)}</b>`), vals);
        const list = el("div", "ov-edit");
        list.innerHTML = d.edits.length
          ? d.edits.map((e) => `<div>${esc(e.kind)}: <code>${esc(H(e.a))}</code> → <code>${esc(H(e.b))}</code></div>`).join("")
          : "<div class='c-muted'>same graph</div>";
        side.append(list);
        wrap.append(left, side);
        r.appendChild(wrap);
        // t = 0 → A only, 0.5 → both, 1 → B only
        function set(t) {
          A.box.style.opacity = Math.min(1, 2 * (1 - t));
          B.box.style.opacity = Math.min(1, 2 * t);
          vals.innerHTML = d.props.filter((p) => p.value || p.a !== p.b).map((p) => {
            const v = p.a + (p.b - p.a) * t;
            return `<span class="k">${esc(p.name)}</span><span>${t === 0 || t === 1 ? `<b>${num(v, p.name)}</b>` : `${num(p.a, p.name)} → ${num(p.b, p.name)}`}</span>`;
          }).join("");
        }
        slider.addEventListener("input", () => { stopBlink(); set(slider.value / 100); });
        let timer = null;
        const stopBlink = () => { clearInterval(timer); timer = null; blink.classList.remove("on"); };
        blink.addEventListener("click", () => {
          if (timer) return stopBlink();
          blink.classList.add("on");
          timer = setInterval(() => { slider.value = slider.value > 50 ? 0 : 100; set(slider.value / 100); }, 800);
        });
        set(0);
      }
      model.on("change:data", draw);
      draw();
    }
    export default { render };
    """)
    return (Overlay,)


@app.cell
def _(Overlay, data, mo):
    mo.ui.anywidget(Overlay(data=data)) if data else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Edit card

    Reads like a matched molecular pair: the two molecules small, and the edits that turn
    A into B large (substituents added / removed / replaced with `*` where they attach,
    atom swaps, stereo flips). Hover an edit to find it in the molecules.
    """)
    return


@app.cell
def _(candidate):
    EditCard = candidate(r"""
    const CSS = `
    .ec-top { display:grid; grid-template-columns: minmax(0,1fr) auto minmax(0,1fr); gap:10px; align-items:center; }
    @media (max-width: 640px) { .ec-top { grid-template-columns: 1fr; } }
    .ec-arrow { display:flex; flex-direction:column; align-items:center; gap:4px; font-variant-numeric:tabular-nums; min-width:110px; }
    .ec-arrow .line { font-size:26px; color:var(--muted); line-height:1; }
    .ec-pill { font-size:12px; border:1px solid var(--border); border-radius:999px; padding:0 9px; white-space:nowrap; }
    .ec-edits { display:flex; flex-direction:column; gap:6px; margin-top:12px; }
    .ec-edit { display:grid; grid-template-columns: 92px 120px 24px 120px 1fr; align-items:center; gap:8px;
      border:1px solid var(--border); border-radius:10px; padding:4px 10px; cursor:default; }
    .ec-edit:hover { background:var(--soft); }
    .ec-edit .k { font-size:11px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); font-weight:600; }
    .ec-edit .f { background:#fff; border-radius:6px; height:64px; display:flex; align-items:center; justify-content:center;
      font:600 18px ui-monospace, monospace; color:#1f2328; }
    .ec-edit .f svg { height:64px; width:auto; }
    .ec-edit .to { text-align:center; color:var(--muted); font-size:18px; }
    .ec-edit .s { font:11.5px ui-monospace, monospace; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    `;
    function render({ model, el: host }) {
      const r = root(host, CSS);
      function draw() {
        r.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
        const d = model.get("data");
        if (!d.panels) return;
        const top = el("div", "ec-top");
        const A = molBox(d.panels[0], "a"), B = molBox(d.panels[1], "b");
        const ca = el("div"), cb = el("div");
        ca.append(el("div", "c-id", `<i style="background:${SIDE.a}"></i>${esc(d.ids[0])}`), A.box);
        cb.append(el("div", "c-id", `<i style="background:${SIDE.b}"></i>${esc(d.ids[1])}`), B.box);
        const arrow = el("div", "ec-arrow");
        arrow.append(el("span", "ec-pill", `Tanimoto <b>${d.tanimoto.toFixed(2)}</b>`), el("div", "line", "⟶"));
        for (const p of d.props.filter((p) => p.value)) {
          const dv = p.b - p.a;
          arrow.append(el("span", "ec-pill", `Δ ${esc(p.name)} <b>${signed(dv, p.name)}</b>${p.name.startsWith("p") ? ` · ${Math.pow(10, Math.abs(dv)).toFixed(0)}×` : ""}`));
        }
        top.append(ca, arrow, cb);
        const info = el("div", "c-info");
        const clear = linkBoxes(A, B, d, info);
        const list = el("div", "ec-edits");
        if (!d.edits.length) list.append(el("div", "c-muted", "No edit: the two graphs are the same."));
        for (const e of d.edits) {
          const frag = (svg, txt) => (svg ? svg : esc(txt === "[H]*" ? "H" : txt));
          const row = el("div", "ec-edit",
            `<span class="k">${esc(e.kind)}</span><div class="f">${frag(e.svg_a, e.a)}</div><span class="to">→</span>` +
            `<div class="f">${frag(e.svg_b, e.b)}</div><span class="s">${esc(H(e.a))} → ${esc(H(e.b))}</span>`);
          row.addEventListener("mouseenter", () => { A.show(e.atoms_a); B.show(e.atoms_b); });
          row.addEventListener("mouseleave", clear);
          list.append(row);
        }
        r.append(top, info, list);
      }
      model.on("change:data", draw);
      draw();
    }
    export default { render };
    """)
    return (EditCard,)


@app.cell
def _(EditCard, data, mo):
    mo.ui.anywidget(EditCard(data=data)) if data else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Mirror

    The molecules face each other and their properties meet in the middle as a butterfly
    chart: A's bar grows to the left, B's to the right, each on a fixed typical range, so
    the longer side is the larger value at a glance.
    """)
    return


@app.cell
def _(candidate):
    Mirror = candidate(r"""
    const CSS = `
    .mi-wrap { display:grid; grid-template-columns: minmax(0,1fr) minmax(220px, 280px) minmax(0,1fr); gap:10px; align-items:center; }
    @media (max-width: 760px) { .mi-wrap { grid-template-columns: 1fr; } }
    .mi-fly { display:flex; flex-direction:column; gap:3px; font-variant-numeric:tabular-nums; font-size:11.5px; }
    .mi-head { text-align:center; margin-bottom:4px; }
    .mi-row { display:grid; grid-template-columns: 1fr 74px 1fr; align-items:center; }
    .mi-row .n { text-align:center; color:var(--muted); }
    .mi-row.value .n { color:var(--fg); font-weight:600; }
    .mi-bar { position:relative; height:16px; }
    .mi-bar i { position:absolute; top:2px; bottom:2px; border-radius:3px; opacity:.35; }
    .mi-bar.win i { opacity:.9; }
    .mi-bar span { position:absolute; top:0; line-height:16px; font-size:11px; }
    .mi-l i { right:0; } .mi-l span { left:2px; }
    .mi-r i { left:0; } .mi-r span { right:2px; }
    `;
    function render({ model, el: host }) {
      const r = root(host, CSS);
      function draw() {
        r.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
        const d = model.get("data");
        if (!d.panels) return;
        const A = molBox(d.panels[0], "a"), B = molBox(d.panels[1], "b");
        const ca = el("div"), cb = el("div");
        ca.append(el("div", "c-id", `<i style="background:${SIDE.a}"></i>${esc(d.ids[0])}`), A.box);
        cb.append(el("div", "c-id", `<i style="background:${SIDE.b}"></i>${esc(d.ids[1])}`), B.box);
        const fly = el("div", "mi-fly");
        fly.append(el("div", "mi-head", `Tanimoto <b>${d.tanimoto.toFixed(2)}</b>`));
        for (const p of d.props) {
          const [lo, hi] = p.range;
          const w = (v) => Math.max(2, Math.min(100, (100 * (v - lo)) / (hi - lo)));
          const row = el("div", "mi-row" + (p.value ? " value" : ""));
          const bar = (side, v, win) =>
            `<div class="mi-bar mi-${side === "a" ? "l" : "r"}${win ? " win" : ""}"><i style="width:${w(v)}%;background:${SIDE[side]}"></i><span>${num(v, p.name)}</span></div>`;
          row.innerHTML = bar("a", p.a, p.a > p.b) + `<span class="n">${esc(p.name)}</span>` + bar("b", p.b, p.b > p.a);
          fly.append(row);
        }
        const wrap = el("div", "mi-wrap");
        wrap.append(ca, fly, cb);
        const info = el("div", "c-info");
        linkBoxes(A, B, d, info);
        r.append(wrap, info);
      }
      model.on("change:data", draw);
      draw();
    }
    export default { render };
    """)
    return (Mirror,)


@app.cell
def _(Mirror, data, mo):
    mo.ui.anywidget(Mirror(data=data)) if data else None
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Side by side with dumbbells

    The two drawings as they are, and under them one line per property: where A and B sit
    on a typical range, joined by a line whose length is the difference.
    """)
    return


@app.cell
def _(candidate):
    Dumbbell = candidate(r"""
    const CSS = `
    .db-mols { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
    .db-list { display:grid; grid-template-columns: 90px 1fr 80px; gap:2px 12px; align-items:center; margin-top:8px;
      font-variant-numeric:tabular-nums; font-size:12px; }
    .db-list .n { color:var(--muted); text-align:right; }
    .db-list .n.value { color:var(--fg); font-weight:600; }
    .db-list .dv { white-space:nowrap; }
    .db-list svg { width:100%; height:22px; display:block; overflow:visible; }
    .db-axis { stroke:var(--border); stroke-width:2; }
    .db-tick { font-size:9px; fill:var(--muted); }
    `;
    function render({ model, el: host }) {
      const r = root(host, CSS);
      function draw() {
        r.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
        const d = model.get("data");
        if (!d.panels) return;
        const A = molBox(d.panels[0], "a"), B = molBox(d.panels[1], "b");
        const mols = el("div", "db-mols");
        const ca = el("div"), cb = el("div");
        ca.append(el("div", "c-id", `<i style="background:${SIDE.a}"></i>${esc(d.ids[0])}`), A.box);
        cb.append(el("div", "c-id", `<i style="background:${SIDE.b}"></i>${esc(d.ids[1])}`), B.box);
        mols.append(ca, cb);
        const info = el("div", "c-info");
        linkBoxes(A, B, d, info);
        const list = el("div", "db-list");
        list.append(el("span", "n", "Tanimoto"), el("span", "", `<b>${d.tanimoto.toFixed(2)}</b>`), el("span"));
        for (const p of d.props) {
          const [lo, hi] = p.range;
          const x = (v) => 4 + 92 * Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
          const [xa, xb] = [x(p.a), x(p.b)];
          const svg =
            `<svg viewBox="0 0 100 22" preserveAspectRatio="none">` +
            `<line class="db-axis" x1="4" x2="96" y1="11" y2="11" vector-effect="non-scaling-stroke"/>` +
            `<line x1="${xa}" x2="${xb}" y1="11" y2="11" stroke="#9ca3af" stroke-width="4" vector-effect="non-scaling-stroke"/>` +
            `</svg>`;
          const cell = el("div", "", svg);
          cell.style.position = "relative";
          // dots as HTML so they stay round whatever the width
          for (const [xx, s] of [[xa, "a"], [xb, "b"]])
            cell.append(Object.assign(el("i"), { style: `position:absolute;left:calc(${xx}% - 6px);top:5px;width:12px;height:12px;border-radius:50%;background:${SIDE[s]};border:2px solid var(--card)` }));
          const dv = p.b - p.a;
          list.append(
            el("span", "n" + (p.value ? " value" : ""), esc(p.name)),
            cell,
            el("span", "dv", Math.abs(dv) < 1e-9 ? `<span class="c-muted">same</span>` : `<b>${signed(dv, p.name)}</b>${p.value && p.name.startsWith("p") ? ` <span class="c-muted">${Math.pow(10, Math.abs(dv)).toFixed(0)}×</span>` : ""}`)
          );
        }
        r.append(mols, info, list);
      }
      model.on("change:data", draw);
      draw();
    }
    export default { render };
    """)
    return (Dumbbell,)


@app.cell
def _(Dumbbell, data, mo):
    mo.ui.anywidget(Dumbbell(data=data)) if data else None
    return


if __name__ == "__main__":
    app.run()
