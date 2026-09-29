// MolPair: two molecules side by side in the same orientation, with what differs marked.
// The drawings come from Python (RDKit, aligned on the common substructure); this side adds
// the controls, the atom-to-atom hover and the comparison of values and descriptors.
// Depends on isDark / busyIndicator (prepended by the Python side).

const SIDE = { a: "#3b82f6", b: "#f59e0b" };
const RGB = (c) => `rgb(${c.map((x) => Math.round(x * 255)).join(",")})`;
const KEY = {
  diff: [[[1.0, 0.72, 0.3], "in one molecule only"], [[0.84, 0.6, 1.0], "matched, but element / charge / stereo differs"]],
  core: [[[0.62, 0.8, 1.0], "common substructure"]],
  none: [],
};

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
.mp-key { color:var(--mp-muted); font-size:12px; }
.mp-key i { display:inline-block; width:11px; height:11px; border-radius:50%; vertical-align:-1px; margin:0 4px 0 8px; }
.mp-main { display:grid; grid-template-columns: minmax(0, 1fr) minmax(150px, auto) minmax(0, 1fr); gap:10px; align-items:stretch; }
@media (max-width: 720px) { .mp-main { grid-template-columns: 1fr; } }
.mp-card { background:#fff; color:#1f2328; border:1px solid var(--mp-border); border-radius:8px; padding:4px 8px 6px;
  display:flex; flex-direction:column; }
.mp-head { display:flex; justify-content:space-between; align-items:baseline; gap:8px; font-size:12px; }
.mp-head b { display:inline-flex; align-items:center; gap:5px; }
.mp-head b i { width:10px; height:10px; border-radius:50%; display:inline-block; }
.mp-vals { color:#57606a; font-variant-numeric:tabular-nums; text-align:right; }
.mp-vals span + span { margin-left:8px; }
.mp-card .mp-svg { position:relative; }
.mp-card svg { width:100%; height:auto; display:block; }
.mp-hit { fill:transparent; cursor:pointer; }
.mp-ring { fill:none; stroke-width:2.5; pointer-events:none; }
.mp-mid { display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;
  text-align:center; font-variant-numeric:tabular-nums; padding:4px 2px; min-width:170px; }
.mp-gauge { position:relative; width:108px; height:108px; }
.mp-gauge svg { width:100%; height:100%; transform:rotate(-90deg); }
.mp-gauge .track { stroke:var(--mp-soft); }
.mp-gauge .arc { transition: stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1); }
.mp-gauge .val { position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.mp-gauge .val b { font-size:26px; font-weight:800; letter-spacing:-.5px; line-height:1; }
.mp-gauge .val span { font-size:10px; color:var(--mp-muted); text-transform:uppercase; letter-spacing:.8px; margin-top:3px; }
.mp-cap { font-size:10.5px; color:var(--mp-muted); margin-top:-6px; }
.mp-delta { width:100%; display:flex; flex-direction:column; align-items:center; gap:3px; }
.mp-flow { position:relative; width:100%; height:30px; display:flex; align-items:center; }
.mp-flow .band { position:absolute; left:14px; right:14px; top:50%; height:6px; margin-top:-3px; border-radius:3px;
  background:linear-gradient(90deg, ${SIDE.a}, ${SIDE.b}); opacity:.85; }
.mp-flow .band::after { content:""; position:absolute; right:-9px; top:50%; margin-top:-7px; border:7px solid transparent;
  border-left:10px solid ${SIDE.b}; border-right:0; }
.mp-flow .end { position:relative; z-index:1; width:22px; height:22px; border-radius:50%; color:#fff; font:800 11px/22px system-ui;
  box-shadow:0 0 0 3px var(--mp-card, #fff); }
.mp-flow .num { position:relative; z-index:1; margin:0 auto; padding:1px 10px; border-radius:999px; background:var(--mp-bg, #fff);
  border:1.5px solid var(--mp-border); font-size:17px; font-weight:800; }
.mp-root.dark .mp-flow .num { background:#1c1f24; }
.mp-dname { font-size:10.5px; color:var(--mp-muted); }
.mp-fold { display:inline-flex; align-items:baseline; gap:4px; padding:3px 12px; border-radius:999px; color:#fff;
  background:linear-gradient(135deg, #e8590c, #d6336c); box-shadow:0 3px 10px rgba(214,51,108,.35); }
.mp-fold b { font-size:20px; font-weight:900; letter-spacing:-.5px; }
.mp-fold span { font-size:11px; font-weight:600; opacity:.9; }
.mp-chips { display:flex; flex-wrap:wrap; justify-content:center; gap:4px; }
.mp-chip { font-size:11px; padding:1px 8px; border-radius:999px; color:#1f2328; font-weight:600; cursor:default;
  transition:transform .12s, box-shadow .12s; }
.mp-chip:hover { transform:translateY(-1px); box-shadow:0 2px 6px rgba(0,0,0,.18); }
.mp-hover { min-height:2.6em; font-size:12px; color:var(--mp-muted); }
.mp-hover b { color:var(--mp-fg); }
.mp-root .pop { animation: mp-pop .5s cubic-bezier(.3,1.6,.5,1) both; }
@keyframes mp-pop { 0% { transform:scale(.3); opacity:0; } 100% { transform:scale(1); opacity:1; } }
.mp-table-wrap { overflow-x:auto; margin-top:8px; }
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
  const tableWrap = el("div", { className: "mp-table-wrap" });
  const note = el("div", { className: "mp-note" });
  root.append(bar, main, tableWrap, note);
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
    const key = el(
      "span",
      { className: "mp-key" },
      (KEY[get("highlight")] || []).map(([c, t]) => `<i style="background:${RGB(c)}"></i>${t}`).join("")
    );
    bar.append(seg, chk, key);
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
      hoverBox.innerHTML = `${here}<br>no partner in ${other.toUpperCase()}`;
      return;
    }
    ringAt(other, j, SIDE[other]);
    const there = `<b style="color:${SIDE[other]}">${other.toUpperCase()}</b> ${sym(other, j)}${j}`;
    hoverBox.innerHTML = side === "a" ? `${here} ↔ ${there}` : `${there} ↔ ${here}`;
  }

  function card(side, mol, panel, formats) {
    const c = el("div", { className: "mp-card" });
    const vals = Object.entries(mol.values || {})
      .map(([k, v]) => `<span>${esc(k)} <b>${fmt(v, formats[k])}</b></span>`)
      .join("");
    c.appendChild(
      el(
        "div",
        { className: "mp-head" },
        `<b><i style="background:${SIDE[side]}"></i>${esc(mol.id)}</b><span class="mp-vals">${vals}</span>`
      )
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

  // count a number up from 0 (only when a new pair arrives, not on every redraw)
  function countUp(node, to, render, animate) {
    if (!animate) return void (node.textContent = render(to));
    const t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 900), e = 1 - Math.pow(1 - k, 3);
      node.textContent = render(to * e);
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function gauge(value, label, animate) {
    const R = 44, C = 2 * Math.PI * R;
    const g = el(
      "div",
      { className: "mp-gauge" },
      `<svg viewBox="0 0 108 108"><defs><linearGradient id="mp-grad" x1="0" y1="0" x2="1" y2="1">` +
        `<stop offset="0" stop-color="${SIDE.a}"/><stop offset="1" stop-color="${SIDE.b}"/></linearGradient></defs>` +
        `<circle class="track" cx="54" cy="54" r="${R}" fill="none" stroke-width="10"/>` +
        `<circle class="arc" cx="54" cy="54" r="${R}" fill="none" stroke="url(#mp-grad)" stroke-width="10" stroke-linecap="round"` +
        ` stroke-dasharray="${C}" stroke-dashoffset="${animate ? C : C * (1 - value)}"/></svg>` +
        `<div class="val"><b></b><span>${esc(label)}</span></div>`
    );
    countUp(g.querySelector(".val b"), value, (v) => v.toFixed(2), animate);
    if (animate) requestAnimationFrame(() => requestAnimationFrame(() => g.querySelector(".arc").setAttribute("stroke-dashoffset", C * (1 - value))));
    return g;
  }

  function middle(s, formats, animate) {
    const mid = el("div", { className: "mp-mid" });
    mid.appendChild(gauge(s.tanimoto, "Tanimoto", animate));
    mid.appendChild(el("div", { className: "mp-cap" }, esc(s.fp)));
    for (const d of s.deltas) {
      const box = el("div", { className: "mp-delta" });
      const flow = el(
        "div",
        { className: "mp-flow" },
        `<div class="band"></div><span class="end" style="background:${SIDE.a}">A</span>` +
          `<span class="num"></span><span class="end" style="background:${SIDE.b}">B</span>`
      );
      countUp(flow.querySelector(".num"), d.delta, (v) => signed(v, formats[d.name] ?? 2), animate);
      box.append(flow, el("div", { className: "mp-dname" }, `Δ ${esc(d.name)} (B − A)`));
      if (d.fold) {
        const f = el("div", { className: "mp-fold" + (animate ? " pop" : "") }, `<b>×${fold(d.fold)}</b><span>fold difference</span>`);
        if (animate) f.style.animationDelay = ".6s";
        box.appendChild(f);
      }
      mid.appendChild(box);
    }
    // chips in the highlight colours double as a legend; hovering one rings its atoms
    const chips = el("div", { className: "mp-chips" });
    const chip = (text, colour, group) => {
      const c = el("span", { className: "mp-chip", textContent: text });
      c.style.background = colour;
      if (group) {
        c.addEventListener("mouseenter", () => ringGroup(group));
        c.addEventListener("mouseleave", () => hover(null));
      }
      chips.appendChild(c);
    };
    chip(`${s.core} shared`, RGB([0.62, 0.8, 1.0]), s.groups.core);
    if (s.only[0]) chip(`+${s.only[0]} only A`, RGB([1.0, 0.72, 0.3]), [s.groups.only[0], []]);
    if (s.only[1]) chip(`+${s.only[1]} only B`, RGB([1.0, 0.72, 0.3]), [[], s.groups.only[1]]);
    if (s.changed) chip(`${s.changed} changed`, RGB([0.84, 0.6, 1.0]), s.groups.changed);
    if (s.changed_bonds) chip(`${s.changed_bonds} E/Z`, RGB([0.84, 0.6, 1.0]));
    if (!s.only[0] && !s.only[1] && !s.changed && !s.changed_bonds) chip("same graph", "#d0d7de");
    if (s.timed_out) chip("MCS timed out", "#ffd8a8");
    mid.appendChild(chips);
    hoverBox = el("div", { className: "mp-hover" });
    mid.appendChild(hoverBox);
    return mid;
  }

  function ringGroup([atomsA, atomsB]) {
    hover(null);
    atomsA.forEach((i) => ringAt("a", i, SIDE.a));
    atomsB.forEach((i) => ringAt("b", i, SIDE.b));
  }

  function table(s) {
    if (!s.descriptors.length) return "";
    const digits = (name, v) => (name === "MW" || name === "TPSA" ? 1 : Number.isInteger(v) ? 0 : 2);
    const head = s.descriptors.map((d) => `<th>${esc(d.name)}</th>`).join("");
    const row = (label, key) =>
      `<tr><td>${label}</td>${s.descriptors.map((d) => `<td>${fmt(d[key], digits(d.name, d[key]))}</td>`).join("")}</tr>`;
    const delta = s.descriptors
      .map((d) => {
        const v = d.b - d.a;
        const dg = digits(d.name, d.a) || digits(d.name, d.b);
        return Math.abs(v) < 1e-9 ? `<td class="same">0</td>` : `<td><b>${signed(v, dg)}</b></td>`;
      })
      .join("");
    return (
      `<table class="mp-table"><thead><tr><th></th>${head}</tr></thead><tbody>` +
      row(`<b style="color:${SIDE.a}">A</b>`, "a") +
      row(`<b style="color:${SIDE.b}">B</b>`, "b") +
      `<tr><td>Δ (B − A)</td>${delta}</tr></tbody></table>`
    );
  }

  let lastPair = "";
  function draw() {
    drawBar();
    const view = get("view");
    const mols = get("molecules");
    main.innerHTML = "";
    if (!view?.panels) return;
    const formats = view.formats || {};
    const pairKey = mols.map((m) => m.smiles).join(">");
    const animate = pairKey !== lastPair;
    lastPair = pairKey;
    main.append(
      card("a", mols[0], view.panels[0], formats),
      middle(view.summary, formats, animate),
      card("b", mols[1], view.panels[1], formats)
    );
    tableWrap.innerHTML = table(view.summary);
    note.textContent = get("note");
    hover(null);
  }

  model.on("change:view", draw);
  model.on("change:highlight", drawBar);
  model.on("change:note", () => (note.textContent = get("note")));
  draw();
}

export default { render };
