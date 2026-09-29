// ECFPStepper: walk through the Morgan/ECFP algorithm one atom and one iteration at a time,
// following the ECFPMovie story (radius 0 → R, the feature set, folding, collisions).
// The molecule SVG is rendered in Python (RDKit) because it carries per-atom labels.

const CSS = `
.es-root { font: 13px/1.45 system-ui, sans-serif; color: var(--es-fg); --es-fg:#1f2328; --es-muted:#6b7280;
  --es-border:#d0d7de; --es-card:#fff; --es-soft:#f6f8fa; --es-new:#2f9e44; --es-dup:#e8590c; --es-hit:#e03131;
  --es-accent:#1c7ed6; }
.es-root.dark { --es-fg:#e6e6e6; --es-muted:#9aa4b2; --es-border:#3a3f47; --es-card:#1c1f24; --es-soft:#24282e; }
.es-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }
.es-btn { font:inherit; color:var(--es-fg); background:var(--es-soft); border:1px solid var(--es-border);
  border-radius:6px; padding:3px 10px; cursor:pointer; }
.es-btn.primary { background:var(--es-accent); border-color:var(--es-accent); color:#fff; font-weight:600; }
.es-btn:disabled { opacity:.4; cursor:default; }
.es-seg { display:inline-flex; border:1px solid var(--es-border); border-radius:6px; overflow:hidden; }
.es-seg button { font:inherit; color:var(--es-fg); background:var(--es-soft); border:0; padding:3px 10px; cursor:pointer; }
.es-seg button + button { border-left:1px solid var(--es-border); }
.es-seg button.on { background:var(--es-fg); color:var(--es-card); }
.es-lbl { color:var(--es-muted); }
.es-pos { color:var(--es-muted); font-variant-numeric:tabular-nums; }
.es-input { font:12.5px ui-monospace,monospace; flex:1 1 220px; min-width:0; padding:4px 8px; border-radius:6px;
  border:1px solid var(--es-border); background:var(--es-card); color:var(--es-fg); }
.es-ex { font:inherit; font-size:12px; padding:1px 9px; border-radius:999px; border:1px solid var(--es-border);
  background:none; color:var(--es-muted); cursor:pointer; }
.es-ex.on { border-color:var(--es-accent); color:var(--es-accent); }
.es-err { color:var(--es-hit); width:100%; }
.es-err:empty { display:none; }
.es-chapters { display:grid; gap:6px; margin:2px 0 8px; }
.es-chap { font-size:11.5px; color:var(--es-muted); cursor:pointer; user-select:none; }
.es-chap .track { height:4px; border-radius:2px; background:var(--es-border); overflow:hidden; margin-bottom:3px; }
.es-chap .track i { display:block; height:100%; background:var(--es-accent); transition:width .3s; }
.es-chap.on { color:var(--es-fg); font-weight:600; }
.es-caption { margin:0 0 10px; min-height:3em; }
.es-caption b.t { font-size:16px; display:block; }
.es-caption span.s { color:var(--es-muted); }
.es-main { display:grid; grid-template-columns: minmax(260px, 1fr) minmax(300px, 1.1fr); gap:12px; }
@media (max-width: 760px) { .es-main { grid-template-columns: 1fr; } }
.es-mol { background:#fff; border:1px solid var(--es-border); border-radius:8px; padding:4px; }
.es-mol svg { width:100%; height:auto; display:block; }
.es-panel { display:flex; flex-direction:column; gap:8px; }
.es-box { background:var(--es-soft); border-radius:8px; padding:8px 10px; }
.es-box h4 { margin:0 0 6px; font-size:13px; }
.es-env { font-family:ui-monospace,monospace; }
.es-tag { display:inline-block; border-radius:4px; padding:0 6px; color:#fff; font-size:11px; font-weight:600; margin-right:6px; }
.es-k { display:inline-flex; align-items:center; justify-content:center; min-width:26px; height:26px; padding:0 6px;
  border-radius:7px; color:#fff; font-weight:800; font-size:14px; box-shadow: inset 0 -2px 0 rgba(0,0,0,.18); box-sizing:border-box; }
.es-k.sm { min-width:20px; height:20px; font-size:11.5px; border-radius:5px; padding:0 4px; }
.es-k.dropped { text-decoration:line-through; }
.es-bench { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:6px 0; }
.es-inv { display:grid; grid-template-columns:repeat(3, 1fr); gap:4px; }
.es-inv > div { background:var(--es-card); border:1px solid var(--es-border); border-radius:6px; padding:2px 7px; font-size:11.5px; color:var(--es-muted); }
.es-inv b { color:var(--es-fg); float:right; font-family:ui-monospace,monospace; }
.es-pair { display:inline-flex; align-items:center; gap:4px; background:var(--es-card); border:1px solid var(--es-border);
  border-radius:8px; padding:2px 6px 2px 8px; font:700 15px ui-monospace,monospace; color:var(--es-dup); }
.es-hash { display:inline-flex; align-items:center; gap:4px; color:var(--es-muted); font-size:12px; }
.es-hash svg { width:20px; height:20px; animation: es-spin 1.4s linear infinite; }
.es-root .pop { animation: es-pop .45s cubic-bezier(.3,1.6,.5,1) both; }
@keyframes es-pop { 0% { transform:scale(.2); opacity:0; } 100% { transform:scale(1); opacity:1; } }
@keyframes es-spin { to { transform:rotate(360deg); } }
.es-num { font:12px ui-monospace,monospace; color:var(--es-muted); }
.es-chips { display:flex; flex-wrap:wrap; gap:6px; }
.es-chip { display:inline-flex; align-items:center; gap:6px; background:var(--es-card); border:1px solid var(--es-border);
  border-radius:999px; padding:2px 10px 2px 3px; font-size:12px; cursor:default; }
.es-chip.fresh { border-color:var(--es-new); box-shadow:0 0 0 2px rgba(47,158,68,.3); }
.es-chip.hit { border-color:var(--es-hit); box-shadow:0 0 0 2px rgba(224,49,49,.25); }
.es-vec-head { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:10px 0 6px; }
.es-arrow { font-family:ui-monospace,monospace; color:var(--es-muted); }
.es-grid { display:grid; gap:3px; }
.es-cell { aspect-ratio:1; border-radius:3px; background:var(--es-card); border:1px solid var(--es-border);
  display:flex; align-items:center; justify-content:center; font:700 10px/1 ui-monospace,monospace; color:#fff;
  overflow:hidden; transition:transform .15s; }
.es-grid.tiny .es-cell { border-radius:1px; font-size:0; }
.es-cell.hit { box-shadow:0 0 0 2px var(--es-hit) inset; }
.es-cell.fresh { outline:2px solid var(--es-new); outline-offset:1px; transform:scale(1.15); }
.es-cell.hl { outline:2px solid var(--es-fg); outline-offset:1px; }
.es-muted { color:var(--es-muted); }
.es-click { cursor:pointer; }
.es-progress { height:2px; margin:-4px 0 6px; border-radius:1px; overflow:hidden; }
.es-progress.on { background:var(--es-border); }
.es-progress i { display:block; height:100%; width:0; background:var(--es-accent); }
.es-bitrow { cursor:pointer; border-radius:6px; padding:2px 4px; margin:2px 0; }
.es-bitrow:hover, .es-bitrow.on { background:var(--es-card); }
.es-hint { color:var(--es-muted); font-size:11.5px; margin-top:4px; }
`;

const S = {
  go: "Show", guided: "guided", explore: "explore", back: "◀ back", next: "next ▶", nextCh: "next chapter ⏭",
  restart: "↺ restart", play: "▶ play", pause: "❚❚ pause", step: (i, n) => `step ${i} / ${n}`,
  radius: "radius", clickAtom: "click an atom",
  fpView: "fingerprint", radiusHint: "click an atom or a feature to see how its identifier was made",
  fpHint: "click a feature, a bit or an atom to light up its substructure",
  atRadius: (r, n) => `Radius ${r}: ${n} new feature${n === 1 ? "" : "s"}`,
  inv: ["Atomic number", "Degree (incl. H)", "Hydrogens", "Charge", "Isotope", "In a ring"], yes: "yes", no: "no",
  own: "own", nbrs: "+ neighbours, sorted by (bond, identifier)",
  cap: {
    atom0first: ["Radius 0 · six numbers per atom", "…hashed into one integer: the atom's identifier."],
    atom0: ["Radius 0 · every atom gets one", "Different properties, different identifiers; the same six numbers give the same identifier."],
    set: ["Collect them in a set", "Each identifier is stored once, however many atoms share it."],
    atomNfirst: (r) => (r === 1
      ? [`Radius ${r} · look one bond further`, "Own identifier + sorted (bond, neighbour) pairs → hash."]
      : [`Radius ${r} · neighbours of neighbours`, "The same recipe on last round's identifiers: each now covers a bigger substructure."]),
    atomN: (r) => [`Radius ${r} · every atom does the same`, "Using the old identifiers. Atoms not reached yet still show last round's letter."],
    update: (r) => [`Radius ${r} · update, then collect`, "All atoms switch to their new identifiers at once. An environment that covers the same bonds as an earlier one is a duplicate and is dropped."],
    fold: (n) => [`Folding into ${n} bits`, `bit = identifier mod ${n}` + (n === 2048 ? "." : " (2048 in practice). Try other lengths below.")],
    coll: ["Collisions", "Different substructures in the same bit: the model cannot tell them apart. Click a bit to see its substructures."],
    noColl: ["No collisions", "Every feature has a bit of its own at this length. Try fewer bits."],
  },
  kept: "kept", dup: (k) => `dropped: same bonds as ${k}`, grow: "dropped: no growth",
  whyNew: "A new environment: its identifier joins the fingerprint.",
  whyDup: "Covers exactly the same bonds as an environment already in the fingerprint, so it adds nothing.",
  whyGrow: "The environment did not grow (it already covers everything it can reach), so it is dropped.",
  describes: "This identifier describes", atoms: (n) => `${n} atom${n === 1 ? "" : "s"}`, bonds: (n) => `${n} bond${n === 1 ? "" : "s"}`,
  sameAs: (a) => `same identifier as atom ${a}`,
  setTitle: (n, d) => `The unfolded ECFP so far: ${n} feature${n === 1 ? "" : "s"}` + (d ? ` <span class="es-muted" style="font-weight:400">· ${d} environment${d === 1 ? "" : "s"} dropped</span>` : ""),
  xAtoms: (n) => `×${n} atoms`,
  foldLater: "Folded into a bit vector at the end.",
  bitsOn: (on, n) => `<b>${on}</b> of ${n} bits on`,
  collided: (k) => `${k} bit${k > 1 ? "s hold" : " holds"} two or more features`,
  dropped: (k) => `${k} dropped`,
  vecHint: "Each identifier switches on bit (identifier mod n_bits); a red-framed bit holds two different features, which the model can no longer tell apart.",
  molHint: "Labels are identifiers (RDKit's own values, lettered as in the movie). Same letter, same identifier; grey letters are dropped environments.",
  chapters: (R) => ["Radius 0", ...Array.from({ length: R }, (_, i) => `Radius ${i + 1}`), "Folding"],
};

const BOND = { 1: "–", 1.5: ":", 2: "=", 3: "≡" };
const GEAR = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>';
const fmt = (n) => n.toLocaleString("en-US");
const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

function seg(options, current, onPick) {
  const wrap = el("div", { className: "es-seg" });
  for (const [val, label] of options) {
    const b = el("button", { textContent: label });
    if (val === current) b.classList.add("on");
    b.addEventListener("click", () => onPick(val));
    wrap.appendChild(b);
  }
  return wrap;
}

function render({ model, el: host }) {
  const root = el("div", { className: "es-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);

  const get = (k) => model.get(k);
  const set = (obj) => { for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };
  const chip = (l, cls = "") => `<span class="es-k ${cls}" style="background:${get("colours")[l] || "#94a3b8"}">${esc(l)}</span>`;

  // Guided tour, in the movie's order: each radius atom by atom, then a collect step; then folding.
  let tour = [], pos = 0, guided = get("guided"), timer = null, molEl = null, collBit = -1;
  // explore: a radius (0 … R) or "fp", the finished fingerprint; `pick` = labels lit up in "fp"
  let view = 0, pick = [];
  const R = () => get("steps").length - 1;
  function buildTour() {
    tour = [];
    get("steps").forEach((layer, r) => {
      layer.forEach((s) => tour.push({ kind: "atom", r, atom: s.atom }));
      tour.push({ kind: r === 0 ? "set" : "update", r });
    });
    tour.push({ kind: "fold" }, { kind: "coll" });
  }
  const chapterOf = (st) => (st.kind === "fold" || st.kind === "coll" ? R() + 1 : st.r);
  function goto(p) {
    pos = Math.max(0, Math.min(tour.length - 1, p));
    const st = tour[pos];
    if (st.kind === "atom") set({ radius: st.r, atom: st.atom, envs: [] });
    else if (st.kind === "coll") showBit(collidedBits()[0] ?? -1, false);
    else set({ radius: st.kind === "fold" ? R() : st.r, atom: -1, envs: [] });
    draw();
  }

  // Autoplay, paced like the movie: the first two atoms of each radius slowly, the rest quickly.
  function stopPlay() { clearTimeout(timer); timer = null; }
  function duration(st) {
    const nth = st.kind === "atom" ? get("steps")[st.r].findIndex((x) => x.atom === st.atom) : -1;
    return st.kind !== "atom" ? 2000 : nth === 0 ? 2600 : nth === 1 ? 1800 : 800;
  }
  // while playing, a thin bar under the controls fills up until the next step
  let tickStart = 0, tickLen = 0;
  function play() {
    stopPlay();
    const schedule = () => { tickStart = performance.now(); tickLen = duration(tour[pos]); timer = setTimeout(tick, tickLen); };
    const tick = () => {
      if (pos >= tour.length - 1) { stopPlay(); draw(); return; }
      timer = -1; // still playing while the next step draws
      goto(pos + 1);
      schedule();
      progress();
    };
    schedule();
    draw();
  }
  function progress() {
    const bar = body.querySelector(".es-progress i");
    if (!bar || !timer) return;
    const done = Math.min(1, (performance.now() - tickStart) / tickLen);
    bar.animate([{ width: `${done * 100}%` }, { width: "100%" }], { duration: Math.max(0, tickLen * (1 - done)), fill: "forwards" });
  }

  // The fingerprint up to a tour position: kept identifiers, each once, with how many atoms have it.
  function features(upto) {
    const steps = get("steps"), kept = new Map();
    let dropped = 0;
    const visit = (s, r) => {
      if (s.status !== "new") { dropped++; return; }
      if (!kept.has(s.label)) kept.set(s.label, { ...s, r, count: 0 });
      kept.get(s.label).count++;
    };
    if (!guided) steps.slice(0, get("radius") + 1).forEach((L, r) => L.forEach((s) => visit(s, r)));
    else for (const st of tour.slice(0, upto + 1)) if (st.kind === "atom") visit(steps[st.r][st.atom], st.r);
    return { kept, dropped };
  }
  function folded(kept) {
    const nBits = get("n_bits"), m = new Map();
    for (const f of kept.values()) { const b = f.id % nBits; if (!m.has(b)) m.set(b, []); m.get(b).push(f); }
    return m;
  }
  function allKept() {
    const kept = new Map();
    get("steps").forEach((L, r) => L.forEach((s) => {
      if (s.status !== "new") return;
      if (!kept.has(s.label)) kept.set(s.label, { ...s, r, count: 0 });
      kept.get(s.label).count++;
    }));
    return kept;
  }
  // explore, fingerprint view: light up the substructures of some features
  function pickLabels(labels) {
    pick = labels;
    const all = allKept();
    set({ radius: R(), atom: -1, envs: labels.map((l) => [all.get(l).atom, all.get(l).r]) });
    draw();
  }
  function setView(v) {
    view = v;
    pick = [];
    if (v === "fp") set({ radius: R(), atom: -1, envs: [] });
    else set({ radius: v, atom: Math.max(0, get("atom")), envs: [] });
    draw();
  }
  const collidedBits = () => [...folded(allKept()).entries()].filter(([, v]) => v.length > 1).map(([b]) => b).sort((a, b) => a - b);
  // light up the substructures that share one bit
  function showBit(b, redraw = true) {
    collBit = b;
    const fs = folded(allKept()).get(b) || [];
    set({ radius: R(), atom: -1, envs: fs.map((f) => [f.atom, f.r]) });
    if (redraw) draw();
  }

  // --- SMILES input and examples
  const top = el("div", { className: "es-bar" });
  const input = el("input", { className: "es-input", type: "text", spellcheck: false, placeholder: "SMILES" });
  const go = el("button", { className: "es-btn primary", textContent: S.go });
  const exs = el("span", { style: "display:contents" });
  const err = el("div", { className: "es-err" });
  top.append(input, go, exs, err);
  root.appendChild(top);
  const submit = (v) => { v = v.trim(); if (v && v !== get("smiles")) set({ smiles: v }); };
  go.onclick = () => submit(input.value);
  input.addEventListener("keydown", (ev) => ev.key === "Enter" && submit(input.value));
  function drawTop() {
    input.value = get("smiles");
    exs.innerHTML = "";
    for (const [name, smi] of get("examples") || []) {
      const b = el("button", { className: "es-ex" + (smi === get("smiles") ? " on" : ""), textContent: name, title: smi });
      b.onclick = () => submit(smi);
      exs.appendChild(b);
    }
    err.textContent = get("error");
  }
  const body = el("div");
  root.appendChild(body);

  function draw() {
    body.innerHTML = "";
    const steps = get("steps");
    if (!steps.length) return;
    const r = get("radius"), atom = get("atom"), maxR = steps.length - 1, nBits = get("n_bits");
    const st = guided ? tour[pos] : { kind: atom >= 0 ? "atom" : "none", r, atom };

    // --- controls
    const bar = el("div", { className: "es-bar" });
    bar.appendChild(seg([[true, S.guided], [false, S.explore]], guided, (v) => {
      stopPlay();
      guided = v;
      set({ guided: v });
      if (guided) goto(pos);
      else setView(chapterOf(tour[pos]) > maxR ? "fp" : tour[pos].r);
    }));
    if (guided) {
      const btn = (text, cls, disabled, fn) => { const b = el("button", { className: "es-btn " + cls, textContent: text }); b.disabled = disabled; b.onclick = fn; return b; };
      const chap = chapterOf(st);
      bar.append(
        btn(S.back, "", pos === 0, () => { stopPlay(); goto(pos - 1); }),
        btn(S.next, "primary", pos === tour.length - 1, () => { stopPlay(); goto(pos + 1); }),
        btn(timer ? S.pause : S.play, "", false, () => {
          if (timer) { stopPlay(); draw(); return; }
          if (pos >= tour.length - 1) goto(0);
          play();
        }),
        btn(S.nextCh, "", chap > maxR, () => { stopPlay(); goto(tour.findIndex((x) => chapterOf(x) === chap + 1)); }),
        btn(S.restart, "", false, () => { stopPlay(); goto(0); }),
        el("span", { className: "es-pos", textContent: S.step(pos + 1, tour.length) }),
      );
    } else {
      bar.append(
        seg([...[...Array(maxR + 1).keys()].map((i) => [i, `radius ${i}`]), ["fp", S.fpView]], view, setView),
        el("span", { className: "es-lbl", textContent: view === "fp" ? S.fpHint : S.radiusHint }),
      );
    }
    body.appendChild(bar);
    if (guided) {
      body.appendChild(el("div", { className: "es-progress" + (timer ? " on" : "") }, "<i></i>"));
      requestAnimationFrame(progress);
    }

    // --- chapter progress and narration
    if (guided) {
      const names = S.chapters(maxR), chaps = el("div", { className: "es-chapters" });
      chaps.style.gridTemplateColumns = `repeat(${names.length}, 1fr)`;
      names.forEach((name, c) => {
        const idx = tour.map((x, i) => [chapterOf(x), i]).filter(([cc]) => cc === c).map(([, i]) => i);
        const done = idx.filter((i) => i <= pos).length / idx.length;
        const d = el("div", { className: "es-chap" + (chapterOf(st) === c ? " on" : "") }, `<div class="track"><i style="width:${done * 100}%"></i></div>${name}`);
        d.onclick = () => { stopPlay(); goto(idx[0]); };
        chaps.appendChild(d);
      });
      body.appendChild(chaps);
      let c;
      if (st.kind === "atom") {
        const first = steps[st.r][0].atom === st.atom;
        c = st.r === 0 ? (first ? S.cap.atom0first : S.cap.atom0) : first ? S.cap.atomNfirst(st.r) : S.cap.atomN(st.r);
      } else if (st.kind === "set") c = S.cap.set;
      else if (st.kind === "update") c = S.cap.update(st.r);
      else if (st.kind === "fold") c = S.cap.fold(nBits);
      else c = [...folded(allKept()).values()].some((v) => v.length > 1) ? S.cap.coll : S.cap.noColl;
      body.appendChild(el("div", { className: "es-caption" }, `<b class="t">${c[0]}</b><span class="s">${c[1]}</span>`));
    }

    // --- molecule
    const main = el("div", { className: "es-main" });
    const left = el("div");
    molEl = el("div", { className: "es-mol" }, get("svg"));
    molEl.addEventListener("click", (ev) => {
      const target = ev.target.closest(".es-hit");
      if (!target) return;
      const a = Number(target.getAttribute("data-atom"));
      stopPlay();
      if (guided) {
        const rr = st.kind === "fold" || st.kind === "coll" ? maxR : st.r;
        goto(tour.findIndex((x) => x.kind === "atom" && x.r === rr && x.atom === a));
      } else if (view === "fp") {
        const l = [...allKept().values()].find((f) => f.atom === a && f.r === maxR)?.label
          ?? [...allKept().values()].filter((f) => f.atom === a).pop()?.label;
        if (l) pickLabels([l]);
      } else set({ atom: a === atom ? -1 : a });
    });
    left.appendChild(molEl);
    left.appendChild(el("div", { className: "es-hint" }, S.molHint));
    main.appendChild(left);
    const panel = el("div", { className: "es-panel" });

    if (!guided) {
      if (view === "fp") exploreFingerprint(panel);
      else exploreRadius(panel, view, atom);
      main.appendChild(panel);
      body.appendChild(main);
      return;
    }

    // --- what happens at this step
    const box = el("div", { className: "es-box" });
    const inn = (html, tag = "span") => `<${tag}>${html}</${tag}>`;
    const popChip = (l, cls) => chip(l, cls);
    const hash = () => inn(`<span class="es-hash">${GEAR} hash</span>`) + inn(`<span class="es-arrow">→</span>`);
    if (st.kind === "atom") {
      const s = steps[st.r][st.atom];
      const [tag, colour] = s.status === "new" ? [S.kept, "var(--es-new)"] : s.status === "duplicate" ? [S.dup(s.dup_of), "var(--es-dup)"] : [S.grow, "var(--es-muted)"];
      let bench;
      if (st.r === 0) {
        const v = [...s.inv.slice(0, 5), s.inv[5] ? S.yes : S.no];
        bench = `<div class="es-inv">${S.inv.map((k, i) => inn(`${k}<b>${v[i]}</b>`, "div")).join("")}</div>`;
      } else {
        const prev = steps[st.r - 1];
        bench = `<div class="es-bench">${inn(`<span class="es-muted">${S.own}</span>`)}${popChip(prev[st.atom].label)}${inn(`<span class="es-muted">${S.nbrs}</span>`)}</div>` +
          `<div class="es-bench">${s.nbrs.map(([o, j]) => inn(`<span class="es-pair">${BOND[o] || o}${chip(prev[j].label, "sm")}</span>`)).join("")}</div>`;
      }
      bench += `<div class="es-bench">${hash()}${popChip(s.label, s.status === "new" ? "" : "dropped")}${inn(`<span class="es-num">${fmt(s.id)}</span>`)}</div>`;
      const same = steps[st.r].filter((x) => x.label === s.label && x.atom !== s.atom).map((x) => x.atom);
      box.innerHTML = `<h4>radius ${st.r} · atom ${st.atom}</h4>${bench}` +
        `<div>${S.describes} <span class="es-env">${esc(s.env)}</span> <span class="es-muted">(${S.atoms(s.atoms.length)}, ${S.bonds(s.bonds.length)})</span>` +
        (same.length ? ` <span class="es-muted">· ${S.sameAs(same.join(", "))}</span>` : "") + `</div>` +
        `<div style="margin-top:6px"><span class="es-tag" style="background:${colour}">${esc(tag)}</span>${s.status === "new" ? S.whyNew : s.status === "duplicate" ? S.whyDup : S.whyGrow}</div>`;
    } else if (st.kind === "set" || st.kind === "update") {
      const L = steps[st.r], seen = new Set(), news = [], drops = [];
      for (const s of L) {
        if (seen.has(s.label)) continue;
        seen.add(s.label);
        (s.status === "new" ? news : drops).push(s);
      }
      const count = (l) => L.filter((x) => x.label === l && x.status === "new").length;
      box.innerHTML = `<h4>radius ${st.r}</h4><div class="es-bench">` +
        news.map((s) => inn(`<span class="es-chip">${chip(s.label, "sm")}<span class="es-env">${esc(s.env)}</span>${count(s.label) > 1 ? `<span class="es-muted">${S.xAtoms(count(s.label))}</span>` : ""}</span>`)).join("") + "</div>" +
        (drops.length ? `<div class="es-muted" style="margin-top:4px">${S.dropped(drops.length)}</div><div class="es-bench">` +
          drops.map((s) => inn(`<span class="es-chip">${chip(s.label, "sm dropped")}<span class="es-muted">${s.status === "duplicate" ? S.dup(s.dup_of) : S.grow}</span></span>`)).join("") + "</div>" : "");
    } else if (st.kind === "fold") {
      const f = allKept().values().next().value;
      box.innerHTML = `<div class="es-bench">${popChip(f.label)}${inn(`<span class="es-env">= ${fmt(f.id)}</span>`)}</div>` +
        `<div class="es-bench">${inn(`<span class="es-env">${fmt(f.id)} mod ${nBits} = <b>${f.id % nBits}</b></span>`)}${inn(`<span class="es-arrow">→ bit ${f.id % nBits}</span>`)}</div>`;
    } else if (st.kind === "coll") {
      const bitsNow = folded(allKept()), hits = collidedBits();
      box.innerHTML = hits.length
        ? hits.map((b) => `<div class="es-bench es-bitrow${b === collBit ? " on" : ""}" data-bit="${b}"><b style="min-width:54px">bit ${b}</b>` +
            bitsNow.get(b).map((f) => `<span class="es-chip${b === collBit ? " hit" : ""}">${chip(f.label, "sm")}<span class="es-env">${esc(f.env)}</span></span>`).join("") + "</div>").join("")
        : `<div class="es-muted">${S.cap.noColl[1]}</div>`;
      box.querySelectorAll(".es-bitrow").forEach((row) => (row.onclick = () => { stopPlay(); showBit(Number(row.dataset.bit)); }));
    } else {
      box.innerHTML = `<h4>radius ${r}</h4><div class="es-muted">${S.clickAtom}</div>`;
    }
    panel.appendChild(box);

    // --- the fingerprint collected so far
    const { kept, dropped } = features(pos);
    const current = st.kind === "atom" ? steps[st.r][st.atom] : null;
    const bits = folded(kept);
    const fp = el("div", { className: "es-box" });
    fp.appendChild(el("h4", {}, S.setTitle(kept.size, dropped)));
    const chips = el("div", { className: "es-chips" });
    for (const f of kept.values()) {
      const fresh = current && current.status === "new" && f.label === current.label;
      const hit = st.kind === "coll" && (folded(allKept()).get(collBit) || []).some((x) => x.label === f.label);
      const c = el("span", { className: "es-chip" + (fresh ? " fresh" : "") + (hit ? " hit" : ""), title: `${fmt(f.id)} · radius ${f.r}` });
      c.innerHTML = chip(f.label, "sm" + (fresh && f.count === 1 ? " pop" : "")) + `<span class="es-env">${esc(f.env)}</span>` + (f.count > 1 ? `<span class="es-muted">${S.xAtoms(f.count)}</span>` : "");
      const bit = f.id % nBits;
      c.addEventListener("mouseenter", () => fp.querySelector(`.es-cell[data-bit="${bit}"]`)?.classList.add("hl"));
      c.addEventListener("mouseleave", () => fp.querySelector(`.es-cell[data-bit="${bit}"]`)?.classList.remove("hl"));
      chips.appendChild(c);
    }
    fp.appendChild(chips);

    // --- the same features as a folded bit vector (from the folding chapter on in guided mode)
    if (guided && st.kind !== "fold" && st.kind !== "coll") fp.appendChild(el("div", { className: "es-hint" }, S.foldLater));
    else {
      const collided = [...bits.values()].filter((v) => v.length > 1).length;
      const head = el("div", { className: "es-vec-head" });
      head.append(
        el("span", { className: "es-arrow", textContent: "↓ identifier mod" }),
        seg([[16, "16"], [64, "64"], [256, "256"], [2048, "2048"]], nBits, (v) => {
          set({ n_bits: v });
          if (guided && st.kind === "coll") showBit(collidedBits()[0] ?? -1);
        }),
        el("span", {}, S.bitsOn(bits.size, nBits) +
          (collided ? ` · <span style="color:var(--es-hit);font-weight:600">${S.collided(collided)}</span>` : "")),
      );
      fp.appendChild(head);
      const tiny = nBits > 256;
      const grid = el("div", { className: "es-grid" + (tiny ? " tiny" : "") });
      grid.style.gridTemplateColumns = `repeat(${nBits <= 64 ? 16 : nBits <= 256 ? 32 : 64}, 1fr)`;
      const freshBit = current && current.status === "new" ? current.id % nBits : -1;
      let k = 0;
      for (let b = 0; b < nBits; b++) {
        const fs = bits.get(b);
        const cell = el("div", { className: "es-cell" });
        cell.dataset.bit = b;
        cell.title = `bit ${b}` + (fs ? ": " + fs.map((f) => `${f.label} ${f.env}`).join(" | ") : " (0)");
        if (fs) {
          cell.style.background = get("colours")[fs[0].label];
          if (!tiny) cell.textContent = fs.map((f) => f.label).join("");
          if (fs.length > 1) cell.classList.add("hit");
          // features drop into their bits one after another
          if (guided && st.kind === "fold") { cell.classList.add("pop"); cell.style.animationDelay = `${(0.6 + k++ * 0.05).toFixed(2)}s`; }
        }
        if (b === freshBit || (st.kind === "coll" && b === collBit)) cell.classList.add("fresh");
        if (guided && st.kind === "coll" && fs && fs.length > 1) { cell.style.cursor = "pointer"; cell.onclick = () => showBit(b); }
        grid.appendChild(cell);
      }
      fp.appendChild(grid);
      fp.appendChild(el("div", { className: "es-hint" }, S.vecHint));
    }
    panel.appendChild(fp);
    main.appendChild(panel);
    body.appendChild(main);
  }

  // --- explore, one radius: the clicked atom's recipe, then everything that happened at this radius
  function exploreRadius(panel, r, atom) {
    const steps = get("steps");
    if (atom >= 0) {
      const s = steps[r][atom], box = el("div", { className: "es-box" });
      const [tag, colour] = s.status === "new" ? [S.kept, "var(--es-new)"] : s.status === "duplicate" ? [S.dup(s.dup_of), "var(--es-dup)"] : [S.grow, "var(--es-muted)"];
      let bench;
      if (r === 0) {
        const v = [...s.inv.slice(0, 5), s.inv[5] ? S.yes : S.no];
        bench = `<div class="es-inv">${S.inv.map((k, i) => `<div>${k}<b>${v[i]}</b></div>`).join("")}</div>`;
      } else {
        const prev = steps[r - 1];
        bench = `<div class="es-bench"><span class="es-muted">${S.own}</span>${chip(prev[atom].label)}<span class="es-muted">${S.nbrs}</span></div>` +
          `<div class="es-bench">${s.nbrs.map(([o, j]) => `<span class="es-pair">${BOND[o] || o}${chip(prev[j].label, "sm")}</span>`).join("")}</div>`;
      }
      bench += `<div class="es-bench"><span class="es-hash">${GEAR} hash</span><span class="es-arrow">→</span>${chip(s.label, s.status === "new" ? "" : "dropped")}<span class="es-num">${fmt(s.id)}</span></div>`;
      box.innerHTML = `<h4>radius ${r} · atom ${atom}</h4>${bench}` +
        `<div>${S.describes} <span class="es-env">${esc(s.env)}</span> <span class="es-muted">(${S.atoms(s.atoms.length)}, ${S.bonds(s.bonds.length)})</span></div>` +
        `<div style="margin-top:6px"><span class="es-tag" style="background:${colour}">${esc(tag)}</span>${s.status === "new" ? S.whyNew : s.status === "duplicate" ? S.whyDup : S.whyGrow}</div>`;
      panel.appendChild(box);
    }
    const L = steps[r], seen = new Set(), news = [], drops = [];
    for (const s of L) {
      if (s.status === "new" ? seen.has(s.label) : false) continue;
      if (s.status === "new") { seen.add(s.label); news.push(s); } else drops.push(s);
    }
    const count = (l) => L.filter((x) => x.label === l && x.status === "new").length;
    const box = el("div", { className: "es-box" });
    box.innerHTML = `<h4>${S.atRadius(r, news.length)}</h4><div class="es-bench">` +
      news.map((s) => `<span class="es-chip es-click${s.atom === atom ? " fresh" : ""}" data-atom="${s.atom}">${chip(s.label, "sm")}<span class="es-env">${esc(s.env)}</span>${count(s.label) > 1 ? `<span class="es-muted">${S.xAtoms(count(s.label))}</span>` : ""}</span>`).join("") + "</div>" +
      (drops.length ? `<div class="es-muted" style="margin-top:4px">${S.dropped(drops.length)}</div><div class="es-bench">` +
        drops.map((s) => `<span class="es-chip es-click${s.atom === atom ? " fresh" : ""}" data-atom="${s.atom}">${chip(s.label, "sm dropped")}<span class="es-muted">atom ${s.atom} · ${s.status === "duplicate" ? S.dup(s.dup_of) : S.grow}</span></span>`).join("") + "</div>" : "");
    box.querySelectorAll(".es-click").forEach((c) => (c.onclick = () => set({ atom: Number(c.dataset.atom) })));
    panel.appendChild(box);
  }

  // --- explore, the finished fingerprint: every feature, the bit vector and its collisions
  function exploreFingerprint(panel) {
    const nBits = get("n_bits"), kept = allKept(), bits = folded(kept), colours = get("colours");
    const lit = new Set(pick);
    const fp = el("div", { className: "es-box" });
    fp.appendChild(el("h4", {}, S.setTitle(kept.size, 0)));
    const chips = el("div", { className: "es-chips" });
    for (const f of kept.values()) {
      const c = el("span", { className: "es-chip es-click" + (lit.has(f.label) ? " fresh" : ""), title: `${fmt(f.id)} · radius ${f.r}` });
      c.innerHTML = chip(f.label, "sm") + `<span class="es-env">${esc(f.env)}</span>` + (f.count > 1 ? `<span class="es-muted">${S.xAtoms(f.count)}</span>` : "");
      c.onclick = () => pickLabels([f.label]);
      chips.appendChild(c);
    }
    fp.appendChild(chips);
    panel.appendChild(fp);

    const vec = el("div", { className: "es-box" });
    const hits = [...bits.entries()].filter(([, v]) => v.length > 1).sort((a, b) => a[0] - b[0]);
    const head = el("div", { className: "es-vec-head", style: "margin-top:0" });
    head.append(
      el("span", { className: "es-arrow", textContent: "identifier mod" }),
      seg([[16, "16"], [64, "64"], [256, "256"], [2048, "2048"]], nBits, (v) => { pick = []; set({ n_bits: v, envs: [] }); }),
      el("span", {}, S.bitsOn(bits.size, nBits) + (hits.length ? ` · <span style="color:var(--es-hit);font-weight:600">${S.collided(hits.length)}</span>` : "")),
    );
    vec.appendChild(head);
    const tiny = nBits > 256;
    const grid = el("div", { className: "es-grid" + (tiny ? " tiny" : "") });
    grid.style.gridTemplateColumns = `repeat(${nBits <= 64 ? 16 : nBits <= 256 ? 32 : 64}, 1fr)`;
    for (let b = 0; b < nBits; b++) {
      const fs = bits.get(b), cell = el("div", { className: "es-cell" });
      cell.title = `bit ${b}` + (fs ? ": " + fs.map((f) => `${f.label} ${f.env}`).join(" | ") : " (0)");
      if (fs) {
        cell.style.background = colours[fs[0].label];
        cell.style.cursor = "pointer";
        if (!tiny) cell.textContent = fs.map((f) => f.label).join("");
        if (fs.length > 1) cell.classList.add("hit");
        if (fs.some((f) => lit.has(f.label))) cell.classList.add("fresh");
        cell.onclick = () => pickLabels(fs.map((f) => f.label));
      }
      grid.appendChild(cell);
    }
    vec.appendChild(grid);
    if (hits.length) {
      const list = el("div", { style: "margin-top:8px" });
      list.innerHTML = hits.map(([b, fs]) => `<div class="es-bench es-bitrow${fs.every((f) => lit.has(f.label)) ? " on" : ""}" data-bit="${b}"><b style="min-width:54px">bit ${b}</b>` +
        fs.map((f) => `<span class="es-chip hit">${chip(f.label, "sm")}<span class="es-env">${esc(f.env)}</span></span>`).join("") + "</div>").join("");
      list.querySelectorAll(".es-bitrow").forEach((row) => (row.onclick = () => pickLabels(bits.get(Number(row.dataset.bit)).map((f) => f.label))));
      vec.appendChild(list);
    }
    vec.appendChild(el("div", { className: "es-hint" }, S.vecHint));
    panel.appendChild(vec);
  }

  function reset() {
    stopPlay();
    buildTour();
    pos = 0;
    view = 0;
    pick = [];
    drawTop();
    draw();
  }
  reset();
  model.on("change:steps", reset);
  model.on("change:error", drawTop);
  model.on("change:smiles", drawTop);
  model.on("change:svg", () => molEl && (molEl.innerHTML = get("svg")));
  model.on("change:n_bits", draw);
  for (const k of ["radius", "atom"]) model.on(`change:${k}`, () => !guided && draw());
  return () => stopPlay();
}

export default { render };
