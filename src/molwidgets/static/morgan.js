// MorganExplorer: see which atom environments set each Morgan bit, compare two molecules
// bit by bit, and inspect bit collisions across a reference dataset.
// Depends on loadRDKit / drawSvg / isDark (prepended by the Python side).

const CENTER = [0.93, 0.27, 0.42];
const ENV = [1.0, 0.72, 0.3];
const COLORS = { A: "#3b82f6", B: "#f59e0b" };

const CSS = `
.me-root { font: 13px/1.4 system-ui, sans-serif; color: var(--me-fg); --me-fg:#1f2328; --me-muted:#6b7280;
  --me-border:#d0d7de; --me-card:#fff; --me-soft:#f6f8fa; --me-sel:#fff4e0; }
.me-root.dark { --me-fg:#e6e6e6; --me-muted:#9aa4b2; --me-border:#3a3f47; --me-card:#1c1f24; --me-soft:#24282e; --me-sel:#3a2f1c; }
.me-bar { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:8px; }
.me-seg { display:inline-flex; border:1px solid var(--me-border); border-radius:6px; overflow:hidden; }
.me-seg button { font:inherit; color:var(--me-fg); background:var(--me-soft); border:0; padding:3px 9px; cursor:pointer; }
.me-seg button + button { border-left:1px solid var(--me-border); }
.me-seg button.on { background:var(--me-fg); color:var(--me-card); }
.me-lbl { color:var(--me-muted); }
.me-main { display:grid; grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.25fr); gap:12px; }
@media (max-width: 760px) { .me-main { grid-template-columns: 1fr; } }
.me-mols { display:grid; gap:8px; }
.me-mol { background:var(--me-card); border:1px solid var(--me-border); border-radius:8px; padding:6px 8px; }
.me-mol h4 { margin:0 0 2px; font-size:13px; display:flex; gap:6px; align-items:center; }
.me-mol h4 .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.me-mol svg { width:100%; height:auto; display:block; }
.me-mol .me-cap { color:var(--me-muted); font-size:11.5px; min-height:1.4em; }
.me-summary { background:var(--me-soft); border-radius:8px; padding:6px 10px; font-variant-numeric:tabular-nums; }
.me-summary b { font-size:15px; }
.me-table-wrap { max-height: 430px; overflow:auto; border:1px solid var(--me-border); border-radius:8px; }
.me-table { border-collapse:collapse; width:100%; font-size:12px; font-variant-numeric:tabular-nums; }
.me-table th { position:sticky; top:0; z-index:2; background:var(--me-soft); text-align:left; padding:4px 6px; cursor:pointer;
  border-bottom:1px solid var(--me-border); white-space:nowrap; user-select:none; }
.me-table td { padding:3px 6px; border-bottom:1px solid var(--me-border); white-space:nowrap; }
.me-table td.env { font-family:ui-monospace,monospace; max-width:220px; overflow:hidden; text-overflow:ellipsis; }
.me-table tr { cursor:pointer; }
.me-table tr:hover td { background:var(--me-soft); }
.me-table tr.sel td { background:var(--me-sel); }
.me-chip { display:inline-block; min-width:14px; text-align:center; border-radius:4px; color:#fff; font-size:10.5px;
  padding:0 4px; margin-right:2px; font-weight:600; }
.me-warn { color:#e8590c; font-weight:600; }
.me-bar2 { position:relative; display:inline-block; width:54px; height:8px; background:var(--me-soft); border-radius:4px; vertical-align:middle; }
.me-bar2 i { position:absolute; top:0; bottom:0; border-radius:4px; }
.me-coll { margin-top:10px; }
.me-coll h4 { margin:4px 0; font-size:13px; }
.me-coll-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap:6px; }
.me-coll-card { background:var(--me-card); border:1px solid var(--me-border); border-radius:8px; padding:4px; font-size:11px; }
.me-coll-card svg { width:100%; height:auto; display:block; }
.me-coll-card code { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.me-muted { color:var(--me-muted); }
`;

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

function seg(options, current, onPick) {
  const wrap = el("div", { className: "me-seg" });
  for (const [val, label] of options) {
    const b = el("button", { textContent: label });
    if (val === current) b.classList.add("on");
    b.addEventListener("click", () => onPick(val));
    wrap.appendChild(b);
  }
  return wrap;
}

function hlFor(bitEntry) {
  if (!bitEntry) return null;
  const atoms = new Set(), bonds = new Set(), atomColors = {}, bondColors = {};
  for (const env of bitEntry.envs) {
    for (const a of env.atoms) { atoms.add(a); if (!(a in atomColors)) atomColors[a] = ENV; }
    for (const b of env.bonds) { bonds.add(b); bondColors[b] = ENV; }
  }
  for (const env of bitEntry.envs) atomColors[env.center] = CENTER;
  return { atoms: [...atoms], bonds: [...bonds], atomColors, bondColors };
}

async function render({ model, el: host }) {
  const root = el("div", { className: "me-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  root.appendChild(el("div", { className: "me-muted" }, "loading RDKit.js…"));
  const RDKit = await loadRDKit();

  let filter = "all";
  let sortKey = "status";
  let sortDesc = false;
  let hover = null;
  const get = (k) => model.get(k);

  function rows() {
    const mols = get("payload");
    const stats = get("bit_stats") || {};
    const byBit = new Map();
    mols.forEach((m, i) => {
      const tag = i === 0 ? "A" : "B";
      for (const b of m.bits) {
        if (!byBit.has(b.bit)) byBit.set(b.bit, { bit: b.bit, in: {}, envs: {} });
        const r = byBit.get(b.bit);
        r.in[tag] = b;
      }
    });
    const two = mols.length === 2;
    return [...byBit.values()].map((r) => {
      const s = stats[r.bit] || {};
      const first = (r.in.A || r.in.B).envs;
      const status = two ? (r.in.A && r.in.B ? "shared" : r.in.A ? "only A" : "only B") : "on";
      const delta = s.mean_on != null && s.mean_off != null ? s.mean_on - s.mean_off : null;
      const envSet = new Set([...(r.in.A?.envs || []), ...(r.in.B?.envs || [])].map((e) => e.smiles));
      return {
        ...r,
        status,
        radius: Math.min(...first.map((e) => e.radius)),
        env: [...envSet].join("  |  "),
        localCollision: envSet.size > 1,
        n_on: s.n_on ?? null,
        n_envs: s.n_envs ?? null,
        delta,
      };
    });
  }

  function sorted(list) {
    const order = { shared: 0, "only A": 1, "only B": 2, on: 0 };
    const key = (r) => (sortKey === "status" ? order[r.status] * 10 + r.radius : r[sortKey]);
    return [...list].sort((a, b) => {
      const x = key(a), y = key(b);
      if (x == null) return 1;
      if (y == null) return -1;
      const c = x < y ? -1 : x > y ? 1 : a.bit - b.bit;
      return sortDesc ? -c : c;
    });
  }

  function draw() {
    const mols = get("payload");
    const dark = root.classList.contains("dark");
    const sel = get("selected_bit");
    const active = hover ?? (sel >= 0 ? sel : null);
    const all = rows();
    const deltaMax = Math.max(0.5, ...all.map((r) => Math.abs(r.delta ?? 0)));
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());

    // --- controls
    const bar = el("div", { className: "me-bar" });
    bar.append(
      el("span", { className: "me-lbl", textContent: "radius" }),
      seg([[0, "0"], [1, "1"], [2, "2"], [3, "3"]], get("radius"), (v) => { model.set("radius", v); model.set("selected_bit", -1); model.save_changes(); }),
      el("span", { className: "me-lbl", textContent: "bits" }),
      seg([[64, "64"], [256, "256"], [1024, "1024"], [2048, "2048"], [4096, "4096"]], get("n_bits"), (v) => { model.set("n_bits", v); model.set("selected_bit", -1); model.save_changes(); }),
    );
    if (mols.length === 2)
      bar.append(
        el("span", { className: "me-lbl", textContent: "show" }),
        seg([["all", "all"], ["shared", "shared"], ["only A", "only A"], ["only B", "only B"], ["collide", "⚠ in-molecule collisions"]], filter, (v) => { filter = v; draw(); }),
      );
    else bar.append(el("span", { className: "me-lbl", textContent: "show" }), seg([["all", "all"], ["collide", "⚠ in-molecule collisions"]], filter, (v) => { filter = v; draw(); }));
    root.appendChild(bar);

    if (!mols.length) {
      root.appendChild(el("div", { className: "me-muted" }, "Select one or two molecules to explore their Morgan bits."));
      return;
    }

    const main = el("div", { className: "me-main" });
    const left = el("div", { className: "me-mols" });
    mols.forEach((m, i) => {
      const tag = i === 0 ? "A" : "B";
      const entry = active != null ? m.bits.find((b) => b.bit === active) : null;
      const card = el("div", { className: "me-mol" });
      const dot = mols.length === 2 ? `<span class="dot" style="background:${COLORS[tag]}"></span>${tag} · ` : "";
      const extra = m.label ? ` <span class="me-muted" style="font-weight:400">${m.label}</span>` : "";
      card.innerHTML = `<h4>${dot}${m.id}${extra}</h4>` + drawSvg(RDKit, m.smiles, 320, 200, hlFor(entry), dark);
      const cap = el("div", { className: "me-cap" });
      if (active != null)
        cap.innerHTML = entry
          ? `bit ${active}: ${entry.envs.length} environment${entry.envs.length > 1 ? "s" : ""} — <code>${entry.envs.map((e) => e.smiles).join(" | ")}</code>`
          : `bit ${active} is <b>off</b> in this molecule`;
      else cap.textContent = `${m.bits.length} bits on — click a row to highlight`;
      card.appendChild(cap);
      left.appendChild(card);
    });
    if (mols.length === 2) {
      const a = new Set(mols[0].bits.map((b) => b.bit)), b = new Set(mols[1].bits.map((x) => x.bit));
      const inter = [...a].filter((x) => b.has(x)).length;
      const tani = inter / (a.size + b.size - inter);
      const extra = get("pair_note") ? ` · ${get("pair_note")}` : "";
      left.appendChild(el("div", { className: "me-summary" },
        `Tanimoto <b>${tani.toFixed(3)}</b> · shared ${inter} · only A ${a.size - inter} · only B ${b.size - inter}${extra}`));
    }
    main.appendChild(left);

    // --- bit table
    const right = el("div");
    const list = sorted(all.filter((r) => filter === "all" || (filter === "collide" ? r.localCollision : r.status === filter)));
    const hasStats = all.some((r) => r.n_on != null);
    const cols = [["bit", "bit"], ["status", mols.length === 2 ? "in" : "r"], ["env", "environment(s)"]];
    if (hasStats) cols.push(["n_on", "# mols"], ["n_envs", "# envs"], ["delta", `Δ ${get("y_label")}`]);
    const wrap = el("div", { className: "me-table-wrap" });
    const table = el("table", { className: "me-table" });
    const thead = el("tr");
    for (const [k, label] of cols) {
      const th = el("th", { textContent: label + (sortKey === k ? (sortDesc ? " ↓" : " ↑") : "") });
      th.addEventListener("click", () => { sortDesc = sortKey === k ? !sortDesc : k !== "bit" && k !== "status"; sortKey = k; draw(); });
      thead.appendChild(th);
    }
    table.appendChild(thead);
    for (const r of list) {
      const tr = el("tr", { className: r.bit === sel ? "sel" : "" });
      const chips = mols.length === 2
        ? ["A", "B"].filter((t) => r.in[t]).map((t) => `<span class="me-chip" style="background:${COLORS[t]}">${t}</span>`).join("")
        : "";
      const warn = r.localCollision;
      let html = `<td>${r.bit}</td><td>${chips}${mols.length === 2 ? "" : ""}<span class="me-muted"> r${r.radius}</span></td>`;
      html += `<td class="env" title="${r.env}">${r.localCollision ? '<span class="me-warn">⚠ </span>' : ""}${r.env}</td>`;
      if (hasStats) {
        const d = r.delta;
        const w = d == null ? 0 : (Math.abs(d) / deltaMax) * 27;
        const bar = d == null ? "" : `<span class="me-bar2"><i style="left:${d < 0 ? 27 - w : 27}px;width:${w}px;background:${d < 0 ? "#3b82f6" : "#e8590c"}"></i></span>`;
        html += `<td>${r.n_on ?? "–"}</td><td>${r.n_envs ?? "–"}</td><td>${bar} ${d == null ? "–" : (d > 0 ? "+" : "") + d.toFixed(2)}</td>`;
      }
      tr.innerHTML = html;
      if (warn) tr.title = "Several different substructures in this molecule set this bit";
      tr.addEventListener("mouseenter", () => { hover = r.bit; redrawMols(); });
      tr.addEventListener("mouseleave", () => { hover = null; redrawMols(); });
      tr.addEventListener("click", () => { hover = null; model.set("selected_bit", r.bit === sel ? -1 : r.bit); model.save_changes(); });
      table.appendChild(tr);
    }
    wrap.appendChild(table);
    right.appendChild(wrap);
    right.appendChild(el("div", { className: "me-muted", style: "margin-top:4px;font-size:11.5px" },
      hasStats
        ? `# mols: molecules in the reference set with this bit on · # envs: distinct substructures hashed into this bit (&gt;1 = collision) · Δ: mean ${get("y_label")} with bit on minus off`
        : "Provide a reference set to see dataset statistics and collisions."));
    main.appendChild(right);
    root.appendChild(main);

    // --- collision gallery for the selected bit
    const ex = get("bit_examples");
    if (sel >= 0 && ex.length) {
      const box = el("div", { className: "me-coll" });
      box.appendChild(el("h4", {}, `Bit ${sel} in the reference set: ${ex.length} distinct environment${ex.length > 1 ? "s" : ""}${ex.length > 1 ? ' <span class="me-warn">— the model cannot tell these apart</span>' : ""}`));
      const g = el("div", { className: "me-coll-grid" });
      for (const e of ex.slice(0, 12)) {
        const hl = { atoms: e.atoms, bonds: e.bonds, atomColors: Object.fromEntries(e.atoms.map((a) => [a, a === e.center ? CENTER : ENV])), bondColors: Object.fromEntries(e.bonds.map((b) => [b, ENV])) };
        const c = el("div", { className: "me-coll-card" });
        c.innerHTML = drawSvg(RDKit, e.parent_smiles, 200, 140, hl, dark) + `<code title="${e.smiles}">${e.smiles}</code><span class="me-muted">r${e.radius} · in ${e.count} mols</span>`;
        g.appendChild(c);
      }
      box.appendChild(g);
      root.appendChild(box);
    }
  }

  function redrawMols() {
    // Cheap partial redraw: only molecule SVGs depend on hover.
    const mols = get("payload");
    const dark = root.classList.contains("dark");
    const sel = get("selected_bit");
    const active = hover ?? (sel >= 0 ? sel : null);
    root.querySelectorAll(".me-mol").forEach((card, i) => {
      const m = mols[i];
      const entry = active != null ? m.bits.find((b) => b.bit === active) : null;
      const svg = card.querySelector("svg");
      if (svg) svg.outerHTML = drawSvg(RDKit, m.smiles, 320, 200, hlFor(entry), dark);
    });
  }

  draw();
  for (const k of ["payload", "bit_stats", "selected_bit", "bit_examples", "radius", "n_bits"]) model.on(`change:${k}`, draw);
}

export default { render };
