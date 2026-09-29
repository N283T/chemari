// MolPair: two compounds side by side with their common substructure highlighted, the Tanimoto
// similarity, and a property table with B − A. Drawings and numbers come from Python (RDKit).
// Depends on isDark / busyIndicator (prepended by the Python side).

const CSS = `
.mp-root { font: 13px/1.45 system-ui, -apple-system, sans-serif; color: var(--mp-fg);
  --mp-fg:#1f2937; --mp-strong:#111827; --mp-muted:#6b7280; --mp-soft:#4b5563; --mp-border:#e5e7eb; --mp-line:#f3f4f6;
  --mp-card:#fff; --mp-hover:#f8fafc; --mp-up:#b91c1c; --mp-down:#1d4ed8; }
.mp-root.dark { --mp-fg:#e5e7eb; --mp-strong:#f9fafb; --mp-muted:#9ca3af; --mp-soft:#cbd5e1; --mp-border:#374151; --mp-line:#2b3240;
  --mp-card:#1c1f24; --mp-hover:#252a31; --mp-up:#f87171; --mp-down:#60a5fa; }
.mp-card { border:1px solid var(--mp-border); border-radius:10px; padding:14px 16px; background:var(--mp-card); }
.mp-head { display:flex; align-items:center; gap:14px; margin-bottom:8px; flex-wrap:wrap; }
.mp-head .title { font-weight:600; font-size:15px; }
.mp-head .sim { font-size:13px; color:var(--mp-soft); }
.mp-head .sim b { color:var(--mp-strong); }
.mp-bar { flex:1; min-width:120px; height:8px; background:var(--mp-line); border-radius:4px; overflow:hidden; }
.mp-bar div { height:100%; background:linear-gradient(90deg, #93c5fd, #2563eb); }
.mp-align { font:inherit; font-size:12px; color:var(--mp-fg); background:none; border:1px solid var(--mp-border);
  border-radius:999px; padding:1px 10px 1px 8px; cursor:pointer; display:inline-flex; align-items:center; gap:6px; }
.mp-align i { width:24px; height:14px; border-radius:7px; background:var(--mp-line); position:relative; transition:background .15s; }
.mp-align i::after { content:""; position:absolute; top:2px; left:2px; width:10px; height:10px; border-radius:50%;
  background:#fff; box-shadow:0 0 0 1px rgba(0,0,0,.15); transition:left .15s; }
.mp-align.on i { background:#2563eb; }
.mp-align.on i::after { left:12px; }
.mp-mols { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media (max-width: 560px) { .mp-mols { grid-template-columns:1fr; } }
.mp-mol { border:1px solid #f1f5f9; border-radius:8px; padding:6px; text-align:center; background:#fafafa; color:#374151; }
.mp-mol svg { width:100%; height:auto; max-height:240px; }
.mp-mol .label { font-size:12px; font-weight:600; margin-bottom:2px; }
.mp-mol .label span { color:#6b7280; font-weight:400; margin-left:4px; }
.mp-mol .smi { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; font-size:11px; color:#6b7280;
  word-break:break-all; margin-top:2px; }
.mp-mol .formula { font-size:12px; }
.mp-mol .bad { color:#b91c1c; font-size:12px; padding:40px 0; }
.mp-mcs { font-size:12px; color:var(--mp-muted); margin:8px 0 4px; }
.mp-mcs code { font-family:ui-monospace, Menlo, monospace; font-size:11px; color:var(--mp-fg); }
table.mp-props { width:100%; border-collapse:collapse; font-size:13px; margin-top:6px; }
table.mp-props th { text-align:right; font-weight:500; color:var(--mp-muted); padding:4px 8px; border-bottom:1px solid var(--mp-border); }
table.mp-props th:first-child { text-align:left; }
table.mp-props td { text-align:right; padding:4px 8px; border-bottom:1px solid var(--mp-line); font-variant-numeric:tabular-nums; }
table.mp-props td:first-child { text-align:left; }
table.mp-props tr.value td:first-child { font-weight:600; }
table.mp-props tr.value + tr:not(.value) td { border-top:1px solid var(--mp-border); }
table.mp-props td.delta { color:var(--mp-muted); }
table.mp-props td.delta.up { color:var(--mp-up); }
table.mp-props td.delta.down { color:var(--mp-down); }
table.mp-props tr:hover td { background:var(--mp-hover); }
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
  const style = document.createElement("style");
  style.textContent = CSS;
  root.appendChild(style);
  const card = document.createElement("div");
  card.className = "mp-card";
  root.appendChild(card);
  el.appendChild(root);
  const busy = busyIndicator(model, el, ["molecules", "align"]);

  function molBox(side, k) {
    const name = k === 0 ? "A" : "B";
    const label = `<div class="label" style="color:${SIDE[k]}">${name}${side.id && side.id !== name ? `<span>${esc(side.id)}</span>` : ""}</div>`;
    if (!side.valid)
      return `<div class="mp-mol">${label}<div class="bad">Could not parse SMILES</div><div class="smi">${esc(side.smiles)}</div></div>`;
    return `<div class="mp-mol">${label}${side.svg}<div class="formula">${esc(side.formula)}</div><div class="smi">${esc(side.smiles)}</div></div>`;
  }

  function draw() {
    const data = model.get("data") || {};
    const sides = data.sides || [];
    const both = sides.length === 2 && sides[0].valid && sides[1].valid;

    let head = `<div class="mp-head"><span class="title">A vs B</span>`;
    if (both && data.similarity !== null && data.similarity !== undefined)
      head += `<span class="sim">Tanimoto (ECFP4): <b>${data.similarity.toFixed(2)}</b></span>` +
        `<div class="mp-bar"><div style="width:${Math.round(data.similarity * 100)}%"></div></div>`;
    head += `</div>`;

    let mcs = "";
    if (both)
      mcs = data.mcs_atoms > 0
        ? `<div class="mp-mcs">Common substructure (highlighted): ${data.mcs_atoms} atoms · <code>${esc(data.mcs_smiles)}</code></div>`
        : `<div class="mp-mcs">No common substructure found.</div>`;

    let table = "";
    if (both) {
      const valueRows = Object.keys(sides[0].values || {}).map((key) => {
        const a = sides[0].values[key], b = sides[1].values[key];
        return `<tr class="value"><td>${esc(key)}</td><td>${fmt(a)}</td><td>${fmt(b)}</td>${deltaCell(a, b)}</tr>`;
      });
      const propRows = (data.property_meta || []).map(({ key, label, digits }) => {
        const a = sides[0].props[key], b = sides[1].props[key];
        return `<tr><td>${esc(label)}</td><td>${fmt(a, digits)}</td><td>${fmt(b, digits)}</td>${deltaCell(a, b, digits)}</tr>`;
      });
      table = `<table class="mp-props"><thead><tr><th>Property</th><th style="color:${SIDE[0]}">A</th>` +
        `<th style="color:${SIDE[1]}">B</th><th>B − A</th></tr></thead><tbody>${[...valueRows, ...propRows].join("")}</tbody></table>`;
    }
    card.innerHTML = head + `<div class="mp-mols">${sides.map(molBox).join("")}</div>` + mcs + table;

    // "align B to A": off by default; redraws B along the common part in A's orientation
    if (both && data.mcs_atoms >= 3) {
      const btn = document.createElement("button");
      btn.className = "mp-align" + (model.get("align") ? " on" : "");
      btn.innerHTML = "<i></i>align B to A";
      btn.addEventListener("click", () => {
        const next = { align: !model.get("align") };
        busy(next);
        model.set("align", next.align);
        model.save_changes();
      });
      card.querySelector(".mp-head").appendChild(btn);
    }
  }

  draw();
  model.on("change:data", draw);
}

export default { render };
