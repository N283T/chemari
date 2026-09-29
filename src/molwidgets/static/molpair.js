// MolPair: two molecules side by side in the same orientation, with what differs marked.
// The drawings come from Python (RDKit, aligned on the common substructure); this side adds
// the controls, the atom-to-atom hover and the comparison of values and descriptors.
// Depends on isDark / busyIndicator (prepended by the Python side).

const SIDE = { a: "#3b82f6", b: "#f59e0b" };
const RGB = (c) => `rgb(${c.map((x) => Math.round(x * 255)).join(",")})`;
const CSS = `
.mp-root { font: 13px/1.4 system-ui, sans-serif; color: var(--mp-fg); --mp-fg:#1f2328; --mp-muted:#6b7280;
  --mp-border:#d0d7de; --mp-card:#fff; --mp-soft:#f6f8fa; --mp-up:#2f9e44; --mp-down:#e03131; }
.mp-root.dark { --mp-fg:#e6e6e6; --mp-muted:#9aa4b2; --mp-border:#3a3f47; --mp-card:#1c1f24; --mp-soft:#24282e; }
.mp-bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:8px; }
.mp-seg { display:inline-flex; border:1px solid var(--mp-border); border-radius:6px; overflow:hidden; }
.mp-seg button { font:inherit; color:var(--mp-fg); background:var(--mp-soft); border:0; padding:3px 9px; cursor:pointer; }
.mp-seg button + button { border-left:1px solid var(--mp-border); }
.mp-seg button.on { background:var(--mp-fg); color:var(--mp-card); }
.mp-chk { display:inline-flex; gap:4px; align-items:center; color:var(--mp-muted); cursor:pointer; }
.mp-main { display:grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap:10px; }
@media (max-width: 720px) { .mp-main { grid-template-columns: 1fr; } }
.mp-card { background:#fff; color:#1f2328; border:1px solid var(--mp-border); border-radius:8px; padding:4px 8px 6px;
  display:flex; flex-direction:column; }
.mp-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; font-size:12px; }
.mp-head b { display:inline-flex; align-items:center; gap:5px; }
.mp-head b i { width:10px; height:10px; border-radius:50%; display:inline-block; }
.mp-card .mp-svg { position:relative; }
.mp-card svg { width:100%; height:auto; display:block; }
.mp-hit { fill:transparent; cursor:pointer; }
.mp-ring { fill:none; stroke-width:2.5; pointer-events:none; }
.mp-sum { display:flex; flex-wrap:wrap; align-items:center; gap:4px 14px; margin-top:8px; font-size:12.5px;
  font-variant-numeric:tabular-nums; }
.mp-sum .grp { cursor:default; padding:0 4px; border-radius:4px; }
.mp-sum .grp[data-hover]:hover { background:var(--mp-soft); }
.mp-sum .grp i { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:5px; }
/* fixed height, so the text changing on hover never moves anything */
.mp-hover { margin-left:auto; color:var(--mp-muted); height:1.4em; overflow:hidden; white-space:nowrap; }
.mp-hover b { color:var(--mp-fg); }
.mp-table .sep { border-left:1px solid var(--mp-border); }
.mp-sub { color:var(--mp-muted); font-weight:400; }
.mp-table-wrap { overflow-x:auto; margin-top:6px; }
.mp-table { border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; width:100%; }
.mp-table th, .mp-table td { padding:3px 8px; text-align:right; border-bottom:1px solid var(--mp-border); white-space:nowrap; }
.mp-table th { color:var(--mp-muted); font-weight:600; background:var(--mp-soft); }
.mp-table th:first-child, .mp-table td:first-child { text-align:left; }
.mp-table td.same { color:var(--mp-muted); }
.mp-note { color:var(--mp-muted); font-size:12px; margin-top:6px; }
.mp-note:empty { display:none; }
`;

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function fmt(v, digits) {
  if (v === null || v === undefined) return "–";
  if (typeof v !== "number") return esc(v);
  if (digits === undefined) digits = Number.isInteger(v) ? 0 : 2;
  return v.toFixed(digits);
}

function signed(v, digits) {
  const s = fmt(Math.abs(v), digits);
  return v > 0 ? `+${s}` : v < 0 ? `−${s}` : s;
}

function fold(f) {
  return f >= 10 ? f.toFixed(0) : f.toFixed(1);
}

async function render({ model, el: host }) {
  const root = el("div", { className: "mp-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  const busy = busyIndicator(model, host, ["molecules", "highlight", "stereo_labels", "radius", "n_bits"]);
  const get = (k) => model.get(k);
  const set = (changes) => {
    busy(changes);
    for (const [k, v] of Object.entries(changes)) model.set(k, v);
    model.save_changes();
  };

  const bar = el("div", { className: "mp-bar" });
  const main = el("div", { className: "mp-main" });
  const sumWrap = el("div");
  const tableWrap = el("div", { className: "mp-table-wrap" });
  const note = el("div", { className: "mp-note" });
  root.append(bar, main, sumWrap, tableWrap, note);
  let hoverBox = null;
  const rings = { a: null, b: null };

  function drawBar() {
    bar.innerHTML = "";
    const seg = el("div", { className: "mp-seg" });
    for (const [val, label] of [["diff", "differences"], ["core", "common part"], ["none", "plain"]]) {
      const btn = el("button", { textContent: label });
      if (get("highlight") === val) btn.classList.add("on");
      btn.addEventListener("click", () => set({ highlight: val }));
      seg.appendChild(btn);
    }
    const chk = el("label", { className: "mp-chk" });
    const box = el("input", { type: "checkbox", checked: get("stereo_labels") });
    box.addEventListener("change", () => set({ stereo_labels: box.checked }));
    chk.append(box, document.createTextNode("R/S, E/Z labels"));
    bar.append(seg, chk);
  }

  // partner of atom i on `side`, or undefined
  function partner(side, i) {
    const m = get("mapping") || [];
    const hit = side === "a" ? m.find((p) => p[0] === i) : m.find((p) => p[1] === i);
    return hit ? hit[side === "a" ? 1 : 0] : undefined;
  }

  function ringAt(side, i, colour) {
    const g = rings[side];
    if (!g) return;
    const xy = get("view").panels[side === "a" ? 0 : 1].coords[i];
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    Object.entries({ cx: xy[0], cy: xy[1], r: 12, class: "mp-ring", stroke: colour }).forEach(([k, v]) => c.setAttribute(k, v));
    g.appendChild(c);
  }

  function hover(side, i) {
    for (const g of Object.values(rings)) if (g) g.innerHTML = "";
    if (!hoverBox) return;
    if (side === null) {
      hoverBox.innerHTML = "hover an atom to find its partner";
      return;
    }
    const other = side === "a" ? "b" : "a";
    const j = partner(side, i);
    const panels = get("view").panels;
    const sym = (s, k) => panels[s === "a" ? 0 : 1].symbols[k];
    ringAt(side, i, SIDE[side]);
    const here = `<b style="color:${SIDE[side]}">${side.toUpperCase()}</b> ${sym(side, i)}${i}`;
    if (j === undefined) {
      hoverBox.innerHTML = `${here} · no partner in ${other.toUpperCase()}`;
      return;
    }
    ringAt(other, j, SIDE[other]);
    const there = `<b style="color:${SIDE[other]}">${other.toUpperCase()}</b> ${sym(other, j)}${j}`;
    hoverBox.innerHTML = side === "a" ? `${here} ↔ ${there}` : `${there} ↔ ${here}`;
  }

  function card(side, mol, panel) {
    const c = el("div", { className: "mp-card" });
    c.appendChild(
      el("div", { className: "mp-head" }, `<b><i style="background:${SIDE[side]}"></i>${side.toUpperCase()} · ${esc(mol.id)}</b>`)
    );
    const holder = el("div", { className: "mp-svg" }, panel.svg);
    const svg = holder.querySelector("svg");
    if (svg) {
      const NS = "http://www.w3.org/2000/svg";
      rings[side] = document.createElementNS(NS, "g");
      svg.appendChild(rings[side]);
      panel.coords.forEach(([x, y], i) => {
        const hit = document.createElementNS(NS, "circle");
        Object.entries({ cx: x, cy: y, r: 11, class: "mp-hit" }).forEach(([k, v]) => hit.setAttribute(k, v));
        hit.addEventListener("mouseenter", () => hover(side, i));
        hit.addEventListener("mouseleave", () => hover(null));
        svg.appendChild(hit);
      });
    }
    c.appendChild(holder);
    return c;
  }

  // Tanimoto, then one entry per atom group in its highlight colour (hover: ring those atoms)
  function summary(s) {
    const line = el("div", { className: "mp-sum" });
    line.appendChild(el("span", {}, `Tanimoto (${esc(s.fp)}) <b>${s.tanimoto.toFixed(2)}</b>`));
    const grp = (text, colour, group) => {
      const g = el("span", { className: "grp" }, `<i style="background:${colour}"></i>${text}`);
      if (group) {
        g.dataset.hover = "";
        g.addEventListener("mouseenter", () => ringGroup(group));
        g.addEventListener("mouseleave", () => hover(null));
      }
      line.appendChild(g);
    };
    const [core, only, changed] = [[0.62, 0.8, 1.0], [1.0, 0.72, 0.3], [0.84, 0.6, 1.0]].map(RGB);
    grp(`${s.core} atoms shared`, core, s.groups.core);
    if (s.only[0]) grp(`${s.only[0]} only in A`, only, [s.groups.only[0], []]);
    if (s.only[1]) grp(`${s.only[1]} only in B`, only, [[], s.groups.only[1]]);
    if (s.changed) grp(`${s.changed} changed (element / charge / R/S)`, changed, s.groups.changed);
    if (s.changed_bonds) grp(`${s.changed_bonds} E/Z changed`, changed);
    if (s.timed_out) grp("MCS search timed out", "#adb5bd");
    hoverBox = el("span", { className: "mp-hover" });
    line.appendChild(hoverBox);
    return line;
  }

  function ringGroup([atomsA, atomsB]) {
    hover(null);
    atomsA.forEach((i) => ringAt("a", i, SIDE.a));
    atomsB.forEach((i) => ringAt("b", i, SIDE.b));
  }

  // rows A, B and Δ; columns: the given values, then the descriptors
  function table(s, mols, formats) {
    const deltas = Object.fromEntries(s.deltas.map((d) => [d.name, d]));
    const digits = (name, v) =>
      name in formats ? formats[name] : name === "MW" || name === "TPSA" ? 1 : Number.isInteger(v) ? 0 : 2;
    const cols = [
      ...Object.keys(mols[0].values || {}).map((name) => ({
        name,
        a: mols[0].values[name],
        b: mols[1].values[name],
        delta: deltas[name],
      })),
      ...s.descriptors.map((d, k) => ({ ...d, first: k === 0, desc: true })),
    ];
    if (!cols.length) return "";
    const cls = (c, extra = "") => ` class="${c.first && cols[0] !== c ? "sep" : ""} ${extra}"`;
    const head = cols.map((c) => `<th${cls(c)}>${esc(c.name)}</th>`).join("");
    const row = (label, key) =>
      `<tr><td>${label}</td>${cols.map((c) => `<td${cls(c)}>${fmt(c[key], digits(c.name, c[key]))}</td>`).join("")}</tr>`;
    const delta = cols
      .map((c) => {
        if (!c.desc && !c.delta) return `<td${cls(c)}></td>`;
        const v = c.desc ? c.b - c.a : c.delta.delta;
        const dg = digits(c.name, c.desc ? c.a : v);
        if (Math.abs(v) < 1e-9) return `<td${cls(c, "same")}>0</td>`;
        const f = c.delta?.fold ? ` <span class="mp-sub">(${fold(c.delta.fold)}-fold)</span>` : "";
        return `<td${cls(c)}><b>${signed(v, dg)}</b>${f}</td>`;
      })
      .join("");
    return (
      `<table class="mp-table"><thead><tr><th></th>${head}</tr></thead><tbody>` +
      row(`<b style="color:${SIDE.a}">A</b>`, "a") +
      row(`<b style="color:${SIDE.b}">B</b>`, "b") +
      `<tr><td>Δ (B − A)</td>${delta}</tr></tbody></table>`
    );
  }

  function draw() {
    drawBar();
    const view = get("view");
    const mols = get("molecules");
    main.innerHTML = "";
    if (!view?.panels) return;
    const formats = view.formats || {};
    main.append(card("a", mols[0], view.panels[0]), card("b", mols[1], view.panels[1]));
    sumWrap.replaceChildren(summary(view.summary));
    tableWrap.innerHTML = table(view.summary, mols, formats);
    note.textContent = get("note");
    hover(null);
  }

  model.on("change:view", draw);
  model.on("change:highlight", drawBar);
  model.on("change:note", () => (note.textContent = get("note")));
  draw();
}

export default { render };
