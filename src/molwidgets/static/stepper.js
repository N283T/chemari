// ECFPStepper: walk through the Morgan/ECFP algorithm one iteration at a time.
// The molecule SVG is rendered in Python (RDKit) because it carries per-atom notes.

const CSS = `
.es-root { font: 13px/1.45 system-ui, sans-serif; color: var(--es-fg); --es-fg:#1f2328; --es-muted:#6b7280;
  --es-border:#d0d7de; --es-card:#fff; --es-soft:#f6f8fa; --es-new:#2f9e44; --es-dup:#e8590c; --es-hit:#d6336c; }
.es-root.dark { --es-fg:#e6e6e6; --es-muted:#9aa4b2; --es-border:#3a3f47; --es-card:#1c1f24; --es-soft:#24282e; }
.es-bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:8px; }
.es-seg { display:inline-flex; border:1px solid var(--es-border); border-radius:6px; overflow:hidden; }
.es-seg button { font:inherit; color:var(--es-fg); background:var(--es-soft); border:0; padding:3px 10px; cursor:pointer; }
.es-seg button + button { border-left:1px solid var(--es-border); }
.es-seg button.on { background:var(--es-fg); color:var(--es-card); }
.es-lbl { color:var(--es-muted); }
.es-main { display:grid; grid-template-columns: minmax(280px, 1.1fr) minmax(280px, 1fr); gap:12px; }
@media (max-width: 760px) { .es-main { grid-template-columns: 1fr; } }
.es-mol { background:#fff; border:1px solid var(--es-border); border-radius:8px; padding:4px; }
.es-mol svg { width:100%; height:auto; display:block; cursor:pointer; }
.es-panel { display:flex; flex-direction:column; gap:8px; }
.es-box { background:var(--es-soft); border-radius:8px; padding:8px 10px; }
.es-box h4 { margin:0 0 4px; font-size:13px; }
.es-box code { font-family:ui-monospace,monospace; font-size:12px; }
.es-recipe { font-family:ui-monospace,monospace; font-size:12px; white-space:pre-wrap; word-break:break-all; }
.es-tag { display:inline-block; border-radius:4px; padding:0 6px; color:#fff; font-size:11px; font-weight:600; }
.es-table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; font-size:12px; }
.es-table th, .es-table td { padding:2px 6px; text-align:right; border-bottom:1px solid var(--es-border); }
.es-table th:first-child, .es-table td:first-child { text-align:left; }
.es-table tr.cur td { font-weight:700; }
.es-fold { margin-top:10px; }
.es-fold-grid { display:grid; gap:2px; }
.es-cell { aspect-ratio:1; border-radius:2px; background:var(--es-soft); border:1px solid var(--es-border); cursor:pointer; }
.es-cell.on { background:#74c0fc; border-color:#4dabf7; }
.es-cell.hit { background:var(--es-hit); border-color:var(--es-hit); }
.es-cell.sel { outline:2px solid var(--es-fg); }
.es-muted { color:var(--es-muted); }
`;

const STATUS = {
  new: ["new feature", "var(--es-new)"],
  duplicate: ["dropped: duplicate environment", "var(--es-dup)"],
  "no growth": ["dropped: environment stopped growing", "var(--es-muted)"],
};

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

const hex = (n) => (n >>> 0).toString(16).padStart(8, "0");

function render({ model, el: host }) {
  const root = el("div", { className: "es-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  let foldPick = null; // selected bit in the fold grid

  const get = (k) => model.get(k);
  const set = (obj) => { for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };

  function features(uptoRadius) {
    const out = [];
    get("steps").slice(0, uptoRadius + 1).forEach((layer) => layer.forEach((s) => { if (s.status === "new") out.push(s); }));
    return out;
  }

  function draw() {
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const steps = get("steps");
    const r = get("radius");
    const atom = get("atom");
    const nBits = get("n_bits");
    const maxR = steps.length - 1;

    const bar = el("div", { className: "es-bar" });
    bar.append(
      el("span", { className: "es-lbl", textContent: "iteration (radius)" }),
      seg([...Array(maxR + 1).keys()].map((i) => [i, String(i)]), r, (v) => { foldPick = null; set({ radius: v }); }),
      el("span", { className: "es-lbl", textContent: "fold to" }),
      seg([[16, "16"], [64, "64"], [256, "256"], [1024, "1024"], [2048, "2048"]], nBits, (v) => { foldPick = null; set({ n_bits: v }); }),
      el("span", { className: "es-lbl", textContent: "bits" }),
    );
    root.appendChild(bar);

    const main = el("div", { className: "es-main" });
    const left = el("div");
    const mol = el("div", { className: "es-mol" }, get("svg"));
    mol.title = "click an atom";
    mol.addEventListener("click", (ev) => {
      const target = ev.target.closest(".es-hit");
      if (!target) return;
      const m = String(target.getAttribute("data-atom")).match(/(\d+)/);
      if (m) { foldPick = null; set({ atom: Number(m[1]) === atom ? -1 : Number(m[1]) }); }
    });
    left.appendChild(mol);
    left.appendChild(el("div", { className: "es-muted", style: "font-size:11.5px;margin-top:4px" },
      "Atoms with the same colour carry the same identifier at this iteration. Numbers are the first hex digits of each identifier. Click an atom to see how its identifier was built."));
    main.appendChild(left);

    const panel = el("div", { className: "es-panel" });
    // --- selected atom recipe
    const box = el("div", { className: "es-box" });
    if (atom >= 0) {
      const s = steps[r][atom];
      const [label, color] = STATUS[s.status];
      let recipe;
      if (r === 0) {
        recipe = get("invariant_names").map((n, k) => `${n.padEnd(14)} ${s.recipe[k]}`).join("\n");
      } else {
        const prev = steps[r - 1][atom].identifier;
        const nb = s.recipe[2].map(([bo, id]) => `  (bond ${bo / 2}, ${hex(id)})`).join("\n");
        recipe = `iteration      ${r}\nown id (r=${r - 1})  ${hex(prev)}\nneighbours\n${nb}`;
      }
      box.innerHTML = `<h4>atom ${atom} · radius ${r} → id <code>${hex(s.identifier)}</code></h4>` +
        `<div class="es-recipe">${recipe}</div>` +
        `<div style="margin-top:6px"><span class="es-tag" style="background:${color}">${label}</span>` +
        (s.duplicate_of != null ? ` <span class="es-muted">same bonds as atom ${s.duplicate_of}'s environment</span>` : "") +
        `</div><div class="es-muted" style="margin-top:4px">${s.atoms.length} atom${s.atoms.length === 1 ? "" : "s"}, ${s.bonds.length} bond${s.bonds.length === 1 ? "" : "s"} in this environment · bit ${s.identifier % nBits} of ${nBits}</div>`;
    } else {
      box.innerHTML = r === 0
        ? `<h4>Iteration 0: atom invariants</h4>Every atom is hashed from six numbers: ${get("invariant_names").join(", ")}. Click an atom.`
        : `<h4>Iteration ${r}</h4>Each atom's new id = hash(its id from iteration ${r - 1}, its neighbours' ids and bond orders). Click an atom.`;
    }
    panel.appendChild(box);

    // --- per-iteration summary
    const t = el("table", { className: "es-table" });
    t.innerHTML = "<tr><th>radius</th><th>new identifiers</th><th>dropped: duplicate</th><th>dropped: no growth</th><th>distinct so far</th></tr>";
    const seenIds = new Set();
    steps.forEach((layer, i) => {
      const c = { duplicate: 0, "no growth": 0 };
      let fresh = 0;
      layer.forEach((s) => {
        if (s.status === "new") {
          if (!seenIds.has(s.identifier)) fresh++;
          seenIds.add(s.identifier);
        } else c[s.status]++;
      });
      const tr = el("tr", { className: i === r ? "cur" : "" });
      tr.innerHTML = `<td>${i}</td><td>${fresh}</td><td>${c.duplicate}</td><td>${c["no growth"]}</td><td>${seenIds.size}</td>`;
      t.appendChild(tr);
    });
    const tb = el("div", { className: "es-box" });
    tb.appendChild(el("h4", {}, "What each iteration adds (symmetric atoms share an identifier)"));
    tb.appendChild(t);
    panel.appendChild(tb);
    main.appendChild(panel);
    root.appendChild(main);

    // --- fold grid
    const feats = features(r);
    const byBit = new Map();
    for (const f of feats) {
      const b = f.identifier % nBits;
      if (!byBit.has(b)) byBit.set(b, []);
      byBit.get(b).push(f);
    }
    const distinct = new Set(feats.map((f) => f.identifier)).size;
    const collided = [...byBit.values()].filter((v) => new Set(v.map((f) => f.identifier)).size > 1).length;
    const fold = el("div", { className: "es-fold" });
    fold.appendChild(el("div", {},
      `<b>Folding.</b> ${distinct} distinct identifiers (radius ≤ ${r}) → ${byBit.size} bits set out of ${nBits}. ` +
      (collided ? `<span style="color:var(--es-hit);font-weight:600">${collided} bit${collided > 1 ? "s hold" : " holds"} more than one identifier</span> (red): the fingerprint can no longer tell those environments apart.` : "No collisions in this molecule at this size.")));
    const cols = nBits <= 64 ? 16 : nBits <= 256 ? 32 : 64;
    const grid = el("div", { className: "es-fold-grid" });
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    grid.style.maxWidth = nBits <= 64 ? "420px" : "100%";
    for (let b = 0; b < nBits; b++) {
      const fs = byBit.get(b);
      const cell = el("div", { className: "es-cell" });
      if (fs) {
        const hit = new Set(fs.map((f) => f.identifier)).size > 1;
        cell.classList.add(hit ? "hit" : "on");
        cell.title = `bit ${b}: ` + fs.map((f) => `atom ${f.atom} r${f.radius} (${hex(f.identifier)})`).join(", ");
        if (foldPick === b) cell.classList.add("sel");
        cell.addEventListener("click", () => { foldPick = b; set({ atom: fs[0].atom, radius: fs[0].radius }); });
      }
      grid.appendChild(cell);
    }
    fold.appendChild(grid);
    if (nBits > 256) fold.appendChild(el("div", { className: "es-muted", style: "font-size:11.5px" }, "Hover a cell for its environments; click to jump to one."));
    root.appendChild(fold);
  }

  draw();
  for (const k of ["svg", "steps", "radius", "atom", "n_bits"]) model.on(`change:${k}`, draw);
}

export default { render };
