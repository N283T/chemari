// MolPair: two compounds side by side, the Tanimoto similarity, and a property table with B − A.
// Two switches (both off by default) highlight the common substructure and align B to A.
// Drawings and numbers come from Python (RDKit). Styled like MorganExplorer / BitImportance.
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
.mp-mcs { color:var(--mp-muted); font-size:12px; margin:6px 0 0; min-height:1.45em; }
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

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "mp-root" + (isDark(el) ? " dark" : "");
  root.innerHTML = `<style>${CSS}</style>`;
  const body = document.createElement("div");
  root.appendChild(body);
  el.appendChild(root);
  const busy = busyIndicator(model, el, ["molecules", "show_common", "align"]);

  function toggle(name, label) {
    const b = document.createElement("button");
    b.className = "mp-sw" + (model.get(name) ? " on" : "");
    b.textContent = label;
    b.addEventListener("click", () => {
      const next = { [name]: !model.get(name) };
      busy(next);
      model.set(name, next[name]);
      model.save_changes();
    });
    return b;
  }

  function molBox(side, k) {
    const name = k === 0 ? "A" : "B";
    const head = `<h4><i class="dot" style="background:${SIDE[k]}"></i>${name}${side.id && side.id !== name ? ` <span>${esc(side.id)}</span>` : ""}</h4>`;
    if (!side.valid)
      return `<div class="mp-mol">${head}<div class="bad">Could not parse SMILES</div><div class="smi">${esc(side.smiles)}</div></div>`;
    return `<div class="mp-mol">${head}${side.svg}<div class="cap">${esc(side.formula)}</div><div class="smi">${esc(side.smiles)}</div></div>`;
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
      sim.innerHTML = `<span class="lbl">Tanimoto (ECFP4)</span><b>${data.similarity.toFixed(2)}</b>` +
        `<span class="track"><i style="width:${Math.round(data.similarity * 100)}%"></i></span>`;
      bar.appendChild(sim);
    }
    if (both) bar.append(toggle("show_common", "common part"), toggle("align", "align B to A"));
    body.appendChild(bar);

    const mols = document.createElement("div");
    mols.className = "mp-mols";
    mols.innerHTML = sides.map(molBox).join("");
    body.appendChild(mols);

    // one line, always present, so switching the common part on does not move the table
    const mcs = document.createElement("div");
    mcs.className = "mp-mcs";
    if (both && model.get("show_common"))
      mcs.innerHTML = data.mcs_atoms > 0
        ? `<i></i>common substructure: ${data.mcs_atoms} atoms · <code>${esc(data.mcs_smiles)}</code>`
        : "no common substructure found";
    body.appendChild(mcs);

    if (!both) return;
    const valueRows = Object.keys(sides[0].values || {}).map((key) => {
      const a = sides[0].values[key], b = sides[1].values[key];
      return `<tr class="value"><td>${esc(key)}</td><td>${fmt(a)}</td><td>${fmt(b)}</td>${deltaCell(a, b)}</tr>`;
    });
    const propRows = (data.property_meta || []).map(({ key, label, digits }) => {
      const a = sides[0].props[key], b = sides[1].props[key];
      return `<tr><td>${esc(label)}</td><td>${fmt(a, digits)}</td><td>${fmt(b, digits)}</td>${deltaCell(a, b, digits)}</tr>`;
    });
    const wrap = document.createElement("div");
    wrap.className = "mp-wrap";
    wrap.innerHTML = `<table class="mp-table"><thead><tr><th>Property</th><th style="color:${SIDE[0]}">A</th>` +
      `<th style="color:${SIDE[1]}">B</th><th>B − A</th></tr></thead><tbody>${[...valueRows, ...propRows].join("")}</tbody></table>`;
    body.appendChild(wrap);
  }

  draw();
  model.on("change:data", draw);
}

export default { render };
