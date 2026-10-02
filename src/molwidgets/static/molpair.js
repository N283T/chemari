// MolPair: two compounds side by side, the Tanimoto similarity, and a property table. Each row
// has A and B as a dumbbell on the property's typical range and B − A as a bar on the same scale.
// Two switches (both off by default) highlight the common substructure and align B to A. Below
// the drawings, the table can switch to the difference: the common part with R1, R2, … and per
// site A's piece → B's piece. The common part's SMILES has a copy button, and each drawing a
// copy icon for its SMILES. Drawings and numbers come from Python (RDKit). Styled like MorganExplorer / BitImportance.
// Depends on isDark / busyIndicator (prepended by the Python side).

const CSS = `
.mp-root { font: 13px/1.45 system-ui, sans-serif; color: var(--mp-fg); --mp-fg:#1f2328; --mp-muted:#6b7280;
  --mp-border:#d0d7de; --mp-card:#fff; --mp-soft:#f6f8fa; --mp-track:#eef1f4; --mp-up:#e03131; --mp-down:#1c7ed6; }
.mp-root.dark { --mp-fg:#e6e6e6; --mp-muted:#9aa4b2; --mp-border:#3a3f47; --mp-card:#1c1f24; --mp-soft:#24282e;
  --mp-track:#2c3139; --mp-up:#ff6b6b; --mp-down:#4dabf7; }
.mp-bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:8px; }
.mp-sim { display:flex; align-items:center; gap:8px; flex:1; min-width:220px; }
.mp-sim .lbl { color:var(--mp-muted); }
.mp-sim b { font-variant-numeric:tabular-nums; }
.mp-sim .track { flex:1; max-width:220px; height:6px; border-radius:3px; background:var(--mp-track); overflow:hidden; }
.mp-sim .track i { display:block; height:100%; background:var(--mp-fg); opacity:.6; }
.mp-sim .band { font-weight:600; font-size:12px; white-space:nowrap; }
.mp-sw { font:inherit; color:var(--mp-fg); background:var(--mp-soft); border:1px solid var(--mp-border);
  border-radius:6px; padding:2px 9px; cursor:pointer; }
.mp-sw.on { background:var(--mp-fg); color:var(--mp-card); border-color:var(--mp-fg); }
.mp-mols { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
@media (max-width: 560px) { .mp-mols { grid-template-columns:1fr; } }
.mp-mol { background:#fff; color:#1f2328; border:1px solid var(--mp-border); border-radius:8px; padding:4px 8px 6px; }
.mp-mol h4 { margin:0 0 2px; font-size:13px; display:flex; gap:6px; align-items:center; }
.mp-mol h4 .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.mp-mol h4 span { color:#6b7280; font-weight:400; }
.mp-mol svg { width:100%; height:auto; max-height:240px; display:block; }
.mp-mol .cap { color:#6b7280; font-size:11.5px; text-align:center; }
.mp-mol .smi { font:11px ui-monospace, monospace; color:#6b7280; word-break:break-all; text-align:center; }
.mp-mol .bad { color:#e03131; font-size:12px; padding:40px 0; text-align:center; }
.mp-mcs { color:var(--mp-muted); font-size:12px; margin:6px 0 0; min-height:1.45em; display:flex;
  flex-wrap:wrap; align-items:center; gap:6px; }
.mp-mcs .txt { flex:1; min-width:0; word-break:break-all; }
.mp-seg { display:inline-flex; }
.mp-seg .mp-sw { border-radius:0; font-size:12px; }
.mp-seg .mp-sw:first-child { border-radius:6px 0 0 6px; }
.mp-seg .mp-sw:last-child { border-radius:0 6px 6px 0; }
.mp-seg .mp-sw + .mp-sw { border-left:0; }
.mp-copy { font:12px system-ui, sans-serif; color:var(--mp-fg); background:var(--mp-soft); border:1px solid var(--mp-border);
  border-radius:6px; padding:2px 10px; margin-left:6px; cursor:pointer; vertical-align:middle; flex:none; white-space:nowrap; }
.mp-copy:hover { border-color:var(--mp-muted); }
.mp-mcs code { font:11px ui-monospace, monospace; color:var(--mp-fg); }
.mp-mcs i { display:inline-block; width:10px; height:10px; border-radius:50%; background:rgb(140,199,255); vertical-align:-1px; margin-right:5px; }
.mp-wrap { border:1px solid var(--mp-border); border-radius:8px; overflow:hidden; margin-top:8px; }
.mp-table { border-collapse:collapse; width:100%; font-size:12.5px; }
.mp-table th { background:var(--mp-soft); text-align:right; padding:5px 8px; font-weight:600; white-space:nowrap;
  border-bottom:1px solid var(--mp-border); }
.mp-table th:first-child { text-align:left; }
.mp-table td { padding:3px 8px; text-align:right; border-bottom:1px solid var(--mp-border); font-variant-numeric:tabular-nums; white-space:nowrap; }
.mp-table td:first-child { text-align:left; }
.mp-table tr:last-child td { border-bottom:0; }
.mp-table tbody tr:hover td { background:var(--mp-soft); }
.mp-table tr.value td { font-weight:600; }
.mp-table tr.value + tr:not(.value) td { border-top:1px solid var(--mp-border); }
.mp-table td.delta { color:var(--mp-muted); font-weight:400; }
.mp-table td.delta.up { color:var(--mp-up); }
.mp-table td.delta.down { color:var(--mp-down); }
.mp-table td.a { padding-right:10px; }
.mp-table td.b { text-align:left; padding-left:10px; }
.mp-table td.bell { width:40%; }
.mp-table td.dbar { width:18%; }
.mp-track { position:relative; height:14px; }
.mp-track i { position:absolute; display:block; }
.mp-track .rail { top:6px; left:0; right:0; height:2px; background:var(--mp-track); }
.mp-track .span { top:6px; height:2px; background:var(--mp-muted); opacity:.6; }
.mp-track .pt { top:2px; width:10px; height:10px; margin-left:-5px; border-radius:50%; }
.mp-track .zero { top:0; bottom:0; left:50%; width:1px; background:var(--mp-border); }
.mp-track .bar { top:3px; height:8px; border-radius:2px; opacity:.75; }
.mp-track .bar.up { background:var(--mp-up); }
.mp-track .bar.down { background:var(--mp-down); }
.mp-diff { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.2fr); gap:12px; padding:8px; align-items:start; }
@media (max-width: 640px) { .mp-diff { grid-template-columns:1fr; } }
.mp-diff .lbl { color:var(--mp-muted); font-size:12px; margin-bottom:2px; }
.mp-core { background:#fff; border:1px solid var(--mp-border); border-radius:8px; padding:4px; }
.mp-core svg { width:100%; height:auto; max-height:220px; display:block; }
.mp-edits { display:grid; grid-template-columns:auto minmax(0,1fr) auto minmax(0,1fr); gap:6px 8px; align-items:center; }
.mp-edits .h { font-weight:600; font-size:12px; text-align:center; }
.mp-edits .site { font-weight:600; }
.mp-edits .to { color:var(--mp-muted); font-size:16px; }
.mp-piece { background:#fff; color:#1f2328; border:1px solid var(--mp-border); border-radius:8px; height:100px;
  display:flex; align-items:center; justify-content:center; font:600 16px ui-monospace, monospace; overflow:hidden; }
.mp-piece svg { max-width:100%; max-height:100%; width:auto; height:auto; display:block; }
.mp-none { color:var(--mp-muted); padding:12px; }
.mp-simhead { display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; padding:6px 8px; border-bottom:1px solid var(--mp-border);
  background:var(--mp-soft); font-size:12px; color:var(--mp-muted); }
.mp-table td.st { width:46%; }
.mp-strack { position:relative; height:16px; }
/* the axis labels sit over the track's 0, 0.5 and 1 (the outer two aligned to its ends) */
.mp-saxis { position:relative; height:1.4em; font-weight:400; color:var(--mp-muted); }
.mp-saxis span { position:absolute; top:0; transform:translateX(-50%); }
.mp-saxis span:first-child { transform:none; }
.mp-saxis span:last-child { transform:translateX(-100%); }
.mp-strack i { position:absolute; display:block; }
.mp-strack .rail { top:7px; left:0; right:0; height:2px; background:var(--mp-track); }
.mp-strack .fill { top:4px; left:0; height:8px; border-radius:2px; background:var(--mp-fg); opacity:.55; }
.mp-strack .tick { top:13px; height:3px; width:1px; background:var(--mp-border); }
.mp-table td.own { color:var(--mp-muted); font-size:11.5px; text-align:left; }
.mp-simfoot { padding:6px 8px; color:var(--mp-muted); font-size:12px; border-top:1px solid var(--mp-border); }
.mp-simhead .mp-warn { margin-left:auto; color:var(--mp-up); white-space:nowrap; }
`;

const SIDE = ["#3b82f6", "#f59e0b"]; // A, B: the same colours as MorganExplorer

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function fmt(v, digits) {
  if (v === null || v === undefined) return "–";
  if (digits === undefined) digits = Number.isInteger(v) ? 0 : 2;
  return Number(v).toFixed(digits);
}

function deltaCell(a, b, digits) {
  if (a === null || a === undefined || b === null || b === undefined) return `<td class="delta"></td>`;
  if (digits === undefined) digits = 2;
  const d = b - a;
  if (Math.abs(d) < Math.pow(10, -digits) / 2) return `<td class="delta">0</td>`;
  return `<td class="delta ${d > 0 ? "up" : "down"}">${d > 0 ? "▲" : "▼"} ${fmt(Math.abs(d), digits)}</td>`;
}

const has = (v) => v !== null && v !== undefined;

const copyButton = (text) => `<button class="mp-copy" data-copy="${esc(text)}">copy</button>`;

// The "common part" view: the common substructure with R1, R2, … where A and B differ, and per
// site what A has there → what B has (H when nothing; a changed atom shows its element or CIP
// label). Hover a piece for its SMILES.
function diffView(data) {
  const e = data.edits || { sites: [] };
  const cell = (site, k) => {
    if (site.change) return `<div class="mp-piece" title="${esc(site.kind)}">${esc(site.change[k])}</div>`;
    const smi = site.smiles[k];
    return `<div class="mp-piece" title="${esc(smi || "H")}">${site.svg[k] || "H"}</div>`;
  };
  const rows = e.sites.map((site) =>
    `<span class="site">${esc(site.label)}</span>${cell(site, 0)}<span class="to">→</span>${cell(site, 1)}`).join("");
  const edits = e.sites.length
    ? `<div class="mp-edits"><span></span><span class="h" style="color:${SIDE[0]}">A</span><span></span>` +
      `<span class="h" style="color:${SIDE[1]}">B</span>${rows}</div>`
    : `<div class="mp-none">no difference: same graph and stereocentres</div>`;
  return `<div class="mp-diff"><div><div class="lbl">common substructure</div><div class="mp-core mw-copyable">${e.core_svg}${smilesCopyHtml(e.core_smiles)}</div></div>` +
    `<div><div class="lbl">what differs</div>${edits}</div></div>`;
}

// The "similarity" view: one row per method with its value on 0–1. The methods' scales differ,
// which the note above the table says.
function simView(sim, metric, bands) {
  if (!sim || !sim.rows) return `<div class="mp-none">computing similarities…</div>`;
  const x = (v) => Math.max(0, Math.min(100, v * 100));
  const rows = sim.rows.map((r) => {
    const own = !(metric in r.values); // MCES and properties have one measure of their own
    const v = r.values[own ? "" : metric];
    const track = `<i class="rail"></i>` + [0.25, 0.5, 0.75].map((t) => `<i class="tick" style="left:${x(t)}%"></i>`).join("") +
      `<i class="fill" style="width:${x(v)}%;${bandStyle(bands, v)}"></i>`;
    return `<tr><td>${esc(r.label)}</td><td><b>${v.toFixed(2)}</b></td>` +
      `<td class="st"><div class="mp-strack">${track}</div></td><td class="own">${own ? "own measure" : ""}</td></tr>`;
  }).join("");
  const sc = sim.scaffold || {};
  const word = (v) => (v ? "same" : "different");
  const scaffold = sc.murcko === null
    ? "Murcko scaffold: none (a molecule without rings)"
    : `Murcko scaffold: <b>${word(sc.murcko)}</b> · generic scaffold (atoms and bonds ignored): <b>${word(sc.generic)}</b>`;
  return `<div class="mp-simhead"><span class="mp-seg" data-metric-seg></span><span>coefficient for the fingerprints</span><span class="mp-warn">Each method has its own scale.</span></div>` +
    `<table class="mp-table"><thead><tr><th>Method</th><th>value</th><th class="st"><div class="mp-saxis"><span style="left:0">0</span><span style="left:50%">0.5</span><span style="left:100%">1</span></div></th><th></th></tr></thead>` +
    `<tbody>${rows}</tbody></table><div class="mp-simfoot">${scaffold}</div>`;
}

// the colour of the band a similarity falls in (the last one whose lower bound it reaches)
function bandStyle(bands, v) {
  const band = (bands || []).filter((b) => v >= b.from).pop();
  return band ? `background:${band.color};opacity:1` : "";
}

// One table row: label | A | dumbbell | B | B − A bar | B − A. The dumbbell's axis is `range`
// (for values without one, the two values ± 1); the bar is centred on 0 and half its track is
// the whole range, so bars compare across rows.
function row(cls, label, a, b, digits, range) {
  let [lo, hi] = range || [Math.min(a ?? b, b ?? a) - 1, Math.max(a ?? b, b ?? a) + 1];
  if (!(hi > lo)) [lo, hi] = [0, 1];
  const x = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  let bell = `<i class="rail"></i>`;
  if (has(a) && has(b) && a !== b)
    bell += `<i class="span" style="left:${Math.min(x(a), x(b))}%;width:${Math.abs(x(b) - x(a))}%"></i>`;
  if (has(a)) bell += `<i class="pt" style="left:${x(a)}%;background:${SIDE[0]}"></i>`;
  if (has(b)) bell += `<i class="pt" style="left:${x(b)}%;background:${SIDE[1]};opacity:.85"></i>`;
  let bar = `<i class="zero"></i>`;
  if (has(a) && has(b) && Math.abs(b - a) >= Math.pow(10, -(digits ?? 2)) / 2) {
    const w = Math.min(50, (Math.abs(b - a) / (hi - lo)) * 50);
    bar += `<i class="bar ${b > a ? "up" : "down"}" style="left:${b > a ? 50 : 50 - w}%;width:${w}%"></i>`;
  }
  return `<tr class="${cls}"><td>${esc(label)}</td><td class="a">${fmt(a, digits)}</td>` +
    `<td class="bell"><div class="mp-track">${bell}</div></td><td class="b">${fmt(b, digits)}</td>` +
    `<td class="dbar"><div class="mp-track">${bar}</div></td>${deltaCell(a, b, digits)}</tr>`;
}

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "mp-root" + (isDark(el) ? " dark" : "");
  root.innerHTML = `<style>${CSS}</style>`;
  const body = document.createElement("div");
  root.appendChild(body);
  el.appendChild(root);
  const busy = busyIndicator(model, el, ["molecules", "show_common", "align", "view"]);
  enableSmilesCopy(root, model);
  root.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    await copyText(b.dataset.copy);
    b.textContent = "copied";
    setTimeout(() => (b.textContent = "copy"), 1200);
  });

  // set several traits at once; the "computing" indicator only when Python has to search
  function change(next) {
    if (!(model.get("data") || {}).searched || "align" in next) busy(next);
    for (const [k, v] of Object.entries(next)) model.set(k, v);
    model.save_changes();
  }

  function toggle(name, label) {
    const b = document.createElement("button");
    b.className = "mp-sw" + (model.get(name) ? " on" : "");
    b.textContent = label;
    // once the common part is known both drawings are here, so that switch needs no Python
    b.addEventListener("click", () => change({ [name]: !model.get(name) }));
    return b;
  }

  function viewSwitch() {
    const seg = document.createElement("span");
    seg.className = "mp-seg";
    for (const [v, label] of [["properties", "properties"], ["common", "common part"], ["similarity", "similarity"]]) {
      const b = document.createElement("button");
      b.className = "mp-sw" + (model.get("view") === v ? " on" : "");
      b.textContent = label;
      // the common-part view also highlights it in the drawings
      b.addEventListener("click", () => change(v === "common" ? { view: v, show_common: true } : { view: v }));
      seg.appendChild(b);
    }
    return seg;
  }

  function molBox(side, k) {
    const name = k === 0 ? "A" : "B";
    const head = `<h4><i class="dot" style="background:${SIDE[k]}"></i>${name}${side.id && side.id !== name ? ` <span>${esc(side.id)}</span>` : ""}</h4>`;
    if (!side.valid)
      return `<div class="mp-mol">${head}<div class="bad">Could not parse SMILES</div><div class="smi">${esc(side.smiles)}</div></div>`;
    const cap = (model.get("show_formula") ? `<div class="cap">${esc(side.formula)}</div>` : "") +
      (model.get("show_smiles") ? `<div class="smi">${esc(side.smiles)}</div>` : "");
    const svg = model.get("show_common") && side.svg_common ? side.svg_common : side.svg;
    return `<div class="mp-mol mw-copyable">${head}${svg}${smilesCopyHtml(side.smiles)}${cap}</div>`;
  }

  function draw() {
    const data = model.get("data") || {};
    const sides = data.sides || [];
    const both = sides.length === 2 && sides[0].valid && sides[1].valid;
    body.innerHTML = "";

    const bar = document.createElement("div");
    bar.className = "mp-bar";
    if (both && data.similarity !== null && data.similarity !== undefined) {
      const sim = document.createElement("div");
      sim.className = "mp-sim";
      // the band the value falls in (the last one whose lower bound it reaches) sets colour and label
      const band = (model.get("similarity_bands") || []).filter((b) => data.similarity >= b.from).pop();
      const fill = band ? `background:${band.color};opacity:1` : "";
      sim.innerHTML = `<span class="lbl">Tanimoto (ECFP4)</span><b>${data.similarity.toFixed(2)}</b>` +
        `<span class="track"><i style="width:${Math.round(data.similarity * 100)}%;${fill}"></i></span>` +
        (band ? `<span class="band" style="color:${band.color}">${esc(band.label)}</span>` : "");
      bar.appendChild(sim);
    }
    if (both) bar.append(toggle("show_common", "common part"), toggle("align", "align B to A"));
    body.appendChild(bar);

    const mols = document.createElement("div");
    mols.className = "mp-mols";
    mols.innerHTML = sides.map(molBox).join("");
    body.appendChild(mols);

    // one line, always present (it holds the table's switch), so the table never moves
    const mcs = document.createElement("div");
    mcs.className = "mp-mcs";
    const txt = document.createElement("span");
    txt.className = "txt";
    const common = model.get("view") === "common";
    if (both && data.searched && (model.get("show_common") || common))
      txt.innerHTML = data.mcs_atoms > 0
        ? `<i></i>common substructure: ${data.mcs_atoms} atoms · <code>${esc(data.mcs_smiles)}</code>${copyButton(data.mcs_smiles)}`
        : "no common substructure found";
    mcs.appendChild(txt);
    if (both) mcs.appendChild(viewSwitch());
    body.appendChild(mcs);

    if (!both) return;
    if (model.get("view") === "similarity") {
      const wrap = document.createElement("div");
      wrap.className = "mp-wrap";
      wrap.innerHTML = simView(model.get("similarities"), model.get("similarity_metric"), model.get("similarity_bands"));
      const seg = wrap.querySelector("[data-metric-seg]");
      for (const m of (model.get("similarities") || {}).metrics || []) {
        const b = document.createElement("button");
        b.className = "mp-sw" + (model.get("similarity_metric") === m ? " on" : "");
        b.textContent = m;
        // every coefficient is already computed: switching needs no Python
        b.addEventListener("click", () => { model.set("similarity_metric", m); model.save_changes(); });
        seg?.appendChild(b);
      }
      body.appendChild(wrap);
      return;
    }
    if (common) {
      const wrap = document.createElement("div");
      wrap.className = "mp-wrap";
      wrap.innerHTML = !data.searched
        ? `<div style="padding:12px;color:var(--mp-muted)">searching for the common substructure…</div>`
        : data.mcs_atoms > 0 ? diffView(data)
        : `<div style="padding:12px;color:var(--mp-muted)">no common substructure found</div>`;
      body.appendChild(wrap);
      return;
    }
    const ranges = data.value_ranges || {};
    const valueRows = Object.keys(sides[0].values || {}).map((key) => {
      const a = sides[0].values[key], b = sides[1].values[key];
      return row("value", key, a, b, undefined, ranges[key]);
    });
    const propRows = (data.property_meta || []).map(({ key, label, digits, range }) =>
      row("", label, sides[0].props[key], sides[1].props[key], digits, range));
    const wrap = document.createElement("div");
    wrap.className = "mp-wrap";
    wrap.innerHTML = `<table class="mp-table"><thead><tr><th>Property</th><th style="color:${SIDE[0]}">A</th><th></th>` +
      `<th style="color:${SIDE[1]};text-align:left">B</th><th colspan="2" style="text-align:center">B − A</th></tr></thead>` +
      `<tbody>${[...valueRows, ...propRows].join("")}</tbody></table>`;
    body.appendChild(wrap);
  }

  draw();
  model.on("change:data", draw);
  model.on("change:show_common", draw);
  model.on("change:view", draw);
  model.on("change:similarities", draw);
  model.on("change:similarity_metric", draw);
  model.on("change:show_formula", draw);
  model.on("change:show_smiles", draw);
}

export default { render };
