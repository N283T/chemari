// SubstructureBits: type a SMARTS, see which Morgan bits it sets and what else shares them.
// Depends on loadRDKit / drawSvg / isDark (prepended by the Python side).

const CSS = `
.sb-root { font: 13px/1.45 system-ui, sans-serif; color: var(--sb-fg); --sb-fg:#1f2328; --sb-muted:#6b7280;
  --sb-border:#d0d7de; --sb-card:#fff; --sb-soft:#f6f8fa; --sb-in:#2f9e44; --sb-ctx:#868e96; --sb-sel:#fff4e0; }
.sb-root.dark { --sb-fg:#e6e6e6; --sb-muted:#9aa4b2; --sb-border:#3a3f47; --sb-card:#1c1f24; --sb-soft:#24282e; --sb-sel:#3a2f1c; }
.sb-bar { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-bottom:8px; }
.sb-bar input { font:13px ui-monospace,monospace; color:var(--sb-fg); background:var(--sb-soft); border:1px solid var(--sb-border);
  border-radius:6px; padding:4px 8px; width:16em; }
.sb-bar input.bad { border-color:#e03131; }
.sb-preset { font:inherit; font-size:12px; color:var(--sb-fg); background:var(--sb-card); border:1px solid var(--sb-border);
  border-radius:999px; padding:1px 10px; cursor:pointer; }
.sb-preset.on { background:var(--sb-fg); color:var(--sb-card); }
.sb-seg { display:inline-flex; border:1px solid var(--sb-border); border-radius:6px; overflow:hidden; }
.sb-seg button { font:inherit; color:var(--sb-fg); background:var(--sb-soft); border:0; padding:2px 9px; cursor:pointer; }
.sb-seg button + button { border-left:1px solid var(--sb-border); }
.sb-seg button.on { background:var(--sb-fg); color:var(--sb-card); }
.sb-lbl { color:var(--sb-muted); }
.sb-top { display:grid; grid-template-columns: 180px 1fr; gap:12px; align-items:start; }
@media (max-width: 700px) { .sb-top { grid-template-columns: 1fr; } }
.sb-query { background:var(--sb-card); border:1px solid var(--sb-border); border-radius:8px; padding:4px; text-align:center; }
.sb-query svg { width:100%; height:auto; display:block; }
.sb-summary { color:var(--sb-muted); font-size:12px; margin-top:4px; }
.sb-ex { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:6px; }
.sb-card { background:var(--sb-card); border:1px solid var(--sb-border); border-radius:8px; padding:3px; font-size:11px; }
.sb-card svg { width:100%; height:auto; display:block; }
.sb-table-wrap { max-height:340px; overflow:auto; border:1px solid var(--sb-border); border-radius:8px; margin-top:10px; }
.sb-table { border-collapse:collapse; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }
.sb-table th { position:sticky; top:0; z-index:2; background:var(--sb-soft); text-align:left; padding:4px 6px;
  border-bottom:1px solid var(--sb-border); white-space:nowrap; }
.sb-table td { padding:3px 6px; border-bottom:1px solid var(--sb-border); white-space:nowrap; }
.sb-table td.env { font-family:ui-monospace,monospace; }
.sb-table tr { cursor:pointer; }
.sb-table tr:hover td { background:var(--sb-soft); }
.sb-table tr.sel td { background:var(--sb-sel); }
.sb-kind { display:inline-block; border-radius:4px; padding:0 6px; color:#fff; font-size:10.5px; font-weight:600; }
.sb-purity { position:relative; display:inline-block; width:60px; height:8px; background:var(--sb-soft); border-radius:4px;
  vertical-align:middle; margin-right:4px; overflow:hidden; }
.sb-purity i { position:absolute; left:0; top:0; bottom:0; background:#1c7ed6; }
.sb-muted { color:var(--sb-muted); }
.sb-hint { color:var(--sb-muted); font-size:11.5px; margin-top:4px; }
`;

const MATCH = [0.55, 0.78, 1.0];
const ENV = [1.0, 0.72, 0.3];
const CENTER = [0.93, 0.27, 0.42];

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");

function drawQuery(RDKit, smarts, dark) {
  const q = RDKit.get_qmol(smarts);
  if (!q) return "";
  try {
    let svg = q.get_svg_with_highlights(JSON.stringify({ width: 170, height: 130, clearBackground: false }));
    if (dark) svg = svg.replace(/stroke:#000000/g, "stroke:#e6e6e6").replace(/fill:#000000/g, "fill:#e6e6e6");
    return svg;
  } catch {
    return "";
  } finally {
    q.delete();
  }
}

async function render({ model, el: host }) {
  const root = el("div", { className: "sb-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  root.appendChild(el("div", { className: "sb-muted" }, "loading RDKit.js…"));
  const RDKit = await loadRDKit();
  let selected = -1;
  let hover = -1;
  const get = (k) => model.get(k);
  const set = (obj) => { for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };

  function exampleHl(ex, bit) {
    if (bit >= 0 && ex.bits[bit]) {
      const atoms = new Set(), bonds = new Set(), atomColors = {}, bondColors = {};
      for (const e of ex.bits[bit]) {
        e.atoms.forEach((a) => { atoms.add(a); atomColors[a] = ENV; });
        e.bonds.forEach((b) => { bonds.add(b); bondColors[b] = ENV; });
      }
      for (const e of ex.bits[bit]) atomColors[e.center] = CENTER;
      return { atoms: [...atoms], bonds: [...bonds], atomColors, bondColors };
    }
    return { atoms: ex.match, bonds: [], atomColors: Object.fromEntries(ex.match.map((a) => [a, MATCH])), bondColors: {} };
  }

  function drawExamples() {
    const res = get("result");
    const dark = root.classList.contains("dark");
    const bit = hover >= 0 ? hover : selected;
    const box = root.querySelector(".sb-ex");
    if (!box) return;
    box.innerHTML = "";
    for (const ex of res.examples || []) {
      const card = el("div", { className: "sb-card" });
      const has = bit < 0 || ex.bits[bit];
      card.innerHTML = drawSvg(RDKit, ex.smiles, 200, 140, exampleHl(ex, bit), dark) +
        `<div class="sb-muted">${esc(ex.id)}${bit >= 0 ? (has ? ` · bit ${bit} ✓` : ` · bit ${bit} not from this match`) : ""}</div>`;
      box.appendChild(card);
    }
  }

  function draw() {
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const res = get("result");
    const dark = root.classList.contains("dark");

    const bar = el("div", { className: "sb-bar" });
    const input = el("input", { value: get("smarts"), placeholder: "SMARTS, e.g. [nX2]", spellcheck: false });
    if (!res.valid) input.classList.add("bad");
    input.addEventListener("change", () => { selected = -1; set({ smarts: input.value.trim() }); });
    bar.append(el("span", { className: "sb-lbl", textContent: "SMARTS" }), input);
    for (const [label, sm] of get("presets")) {
      const b = el("button", { className: "sb-preset" + (sm === get("smarts") ? " on" : ""), textContent: label, title: sm });
      b.onclick = () => { selected = -1; set({ smarts: sm }); };
      bar.appendChild(b);
    }
    root.appendChild(bar);
    const bar2 = el("div", { className: "sb-bar" });
    const seg = el("div", { className: "sb-seg" });
    for (const n of [1024, 2048, 8192]) {
      const b = el("button", { textContent: String(n), className: n === get("n_bits") ? "on" : "" });
      b.onclick = () => { selected = -1; set({ n_bits: n }); };
      seg.appendChild(b);
    }
    bar2.append(el("span", { className: "sb-lbl", textContent: `ECFP${2 * get("radius")} folded to` }), seg, el("span", { className: "sb-lbl", textContent: "bits" }));
    root.appendChild(bar2);

    if (!res.valid) {
      root.appendChild(el("div", { className: "sb-muted" }, "Not a valid SMARTS pattern."));
      return;
    }

    const top = el("div", { className: "sb-top" });
    const q = el("div");
    q.appendChild(el("div", { className: "sb-query" }, drawQuery(RDKit, get("smarts"), dark)));
    q.appendChild(el("div", { className: "sb-summary" },
      `matches <b>${res.n_match}</b> of ${res.n_total} molecules · sets <b>${res.bits.filter((b) => b.kind === "inside").length}</b> bits of its own`));
    top.appendChild(q);
    top.appendChild(el("div", { className: "sb-ex" }));
    root.appendChild(top);

    const wrap = el("div", { className: "sb-table-wrap" });
    const t = el("table", { className: "sb-table" });
    t.innerHTML = `<tr><th>bit</th><th>kind</th><th>r</th><th>environment</th><th>set by pattern in</th>` +
      `<th>on in dataset</th><th>share from pattern</th><th># envs in bit</th><th>Δ ${esc(get("y_label"))}</th></tr>`;
    for (const b of res.bits) {
      const tr = el("tr", { className: b.bit === selected ? "sel" : "" });
      const colour = b.kind === "inside" ? "var(--sb-in)" : "var(--sb-ctx)";
      const purity = b.n_on ? b.n_with / b.n_on : 0;
      tr.innerHTML = `<td>${b.bit}</td><td><span class="sb-kind" style="background:${colour}">${b.kind}</span></td>` +
        `<td>${b.radius}</td><td class="env">${esc(b.env)}</td><td>${b.n_with}</td><td>${b.n_on}</td>` +
        `<td><span class="sb-purity"><i style="width:${(purity * 100).toFixed(0)}%"></i></span>${(purity * 100).toFixed(0)}%</td>` +
        `<td>${b.n_envs}</td><td>${b.delta == null ? "–" : (b.delta > 0 ? "+" : "") + b.delta.toFixed(2)}</td>`;
      tr.addEventListener("mouseenter", () => { hover = b.bit; drawExamples(); });
      tr.addEventListener("mouseleave", () => { hover = -1; drawExamples(); });
      tr.addEventListener("click", () => { selected = selected === b.bit ? -1 : b.bit; draw(); });
      t.appendChild(tr);
    }
    wrap.appendChild(t);
    root.appendChild(wrap);
    root.appendChild(el("div", { className: "sb-hint" },
      "<b>inside</b>: the environment lies entirely within the matched atoms (the pattern's own bits); " +
      "<b>context</b>: centred on a matched atom but reaching its neighbours. " +
      "<b>share from pattern</b>: of the molecules with this bit on, how many owe it to this pattern; " +
      "the rest is other substructures folded into the same bit. Hover or click a row to see the environment in the examples."));
    drawExamples();
  }

  draw();
  model.on("change:result", draw);
}

export default { render };
