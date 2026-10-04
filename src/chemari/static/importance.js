// BitImportance: a fingerprint model's bits ranked by importance (table), and below it (layout
// "stacked") or beside it (layout "side") the selected bit's substructures and the molecules that
// set it, with the responsible atoms highlighted.
// Substructure pictures are RDKit DrawMorganEnv SVGs from Python; molecules are drawn with RDKit.js.
// Depends on loadRDKit / drawSvg / legendHtml / MORGAN_ENV_KEY / isDark / busyIndicator (prepended).

const CSS = `
.bi-root { font: 13px/1.45 system-ui, sans-serif; color: var(--bi-fg); --bi-fg:#1f2328; --bi-muted:#6b7280;
  --bi-border:#d0d7de; --bi-card:#fff; --bi-soft:#f6f8fa; --bi-sel:#fff4e0; --bi-bar:#e8590c; --bi-track:#eef1f4; }
.bi-root.dark { --bi-fg:#e6e6e6; --bi-muted:#9aa4b2; --bi-border:#3a3f47; --bi-card:#1c1f24; --bi-soft:#24282e;
  --bi-sel:#3a2f1c; --bi-track:#2c3139; }
.bi-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:6px; }
.bi-lbl { color:var(--bi-muted); }
.bi-bar .bi-btn { font:inherit; color:var(--bi-fg); background:var(--bi-soft); border:1px solid var(--bi-border);
  border-radius:6px; padding:2px 9px; cursor:pointer; }
.bi-bar .bi-btn:disabled { opacity:.4; cursor:default; }
.bi-bar input { font:inherit; width:70px; padding:2px 6px; border:1px solid var(--bi-border); border-radius:6px;
  background:var(--bi-card); color:var(--bi-fg); }
.bi-legend { color:var(--bi-muted); font-size:11px; line-height:1.7; }
.bi-right { display:flex; gap:6px; align-items:center; margin-left:auto; }
.bi-bar select { font:inherit; color:var(--bi-fg); background:var(--bi-soft); border:1px solid var(--bi-border);
  border-radius:6px; padding:3px 7px; }
.bi-wrap { border:1px solid var(--bi-border); border-radius:8px; overflow:auto; max-height:420px; }
/* every column has a width for its content; the table spreads the room left over across all of
   them in proportion, so the gaps between columns grow evenly instead of in one place */
.bi-table { border-collapse:collapse; width:100%; font-size:12.5px; table-layout:fixed; }
.bi-table th { position:sticky; top:0; z-index:1; background:var(--bi-soft); text-align:left; padding:5px 6px;
  font-weight:600; line-height:1.2; vertical-align:bottom; white-space:nowrap; border-bottom:1px solid var(--bi-border); }
.bi-table th.sortable { cursor:pointer; }
.bi-table th.on { color:var(--bi-bar); }
.bi-table td { padding:3px 6px; border-bottom:1px solid var(--bi-border); vertical-align:middle; white-space:nowrap; }
.bi-table tr:last-child td { border-bottom:0; }
.bi-table tbody tr { cursor:pointer; }
.bi-table tbody tr:hover td { background:var(--bi-soft); }
.bi-table tbody tr.sel td { background:var(--bi-sel); }
.bi-num { text-align:right; font-variant-numeric:tabular-nums; }
.bi-table th.bi-num { text-align:right; }
/* the model's columns (importance scores, effect) and the dataset's (substructure, counts) are
   two groups, split by a rule */
.bi-table .bi-eff { font-variant-numeric:tabular-nums; padding-left:16px; white-space:nowrap; }
.bi-table .bi-gstart { border-left:1px solid var(--bi-border); padding-left:14px; }
.bi-rank { color:var(--bi-muted); }
.bi-score { display:flex; align-items:center; gap:6px; }
.bi-score .track { flex:none; width:110px; height:6px; border-radius:3px; background:var(--bi-track); overflow:hidden; }
.bi-score .track i { display:block; height:100%; background:var(--bi-bar); }
.bi-score span { font-variant-numeric:tabular-nums; width:40px; flex:none; text-align:right; }
.bi-score.dim .track i { opacity:.35; }
.bi-score.dim span { color:var(--bi-muted); }
.bi-thumb { display:flex; align-items:center; gap:6px; }
.bi-thumb .pic { width:56px; height:42px; flex:none; background:#fff; border:1px solid var(--bi-border); border-radius:4px; overflow:hidden; }
.bi-thumb .pic svg { width:100% !important; height:100% !important; display:block; }
.bi-mixed { color:#e8590c; font-weight:600; }
.bi-detail { margin-top:12px; border:1px solid var(--bi-border); border-radius:8px; padding:8px 10px; }
.bi-detail h4 { margin:0; font-size:14px; }
.bi-dhead { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
.bi-sec { display:flex; flex-wrap:wrap; gap:4px 8px; align-items:center; margin:8px 0 4px; color:var(--bi-muted); }
.bi-sec b { color:var(--bi-fg); }
.bi-strip { display:flex; gap:6px; overflow-x:auto; padding-bottom:4px; }
.bi-tile { flex:0 0 118px; background:#fff; color:#1f2328; border:1px solid var(--bi-border); border-radius:6px;
  padding:2px 4px 3px; min-width:0; cursor:pointer; }
.bi-tile.on { outline:2px solid var(--bi-bar); outline-offset:1px; }
.bi-tile.off { opacity:.45; }
.bi-tile svg { width:100% !important; height:auto !important; display:block; }
.bi-thead { display:flex; justify-content:space-between; align-items:center; gap:4px; font-size:10.5px; color:#6b7280; }
.bi-sw { display:inline-block; width:10px; height:10px; border-radius:3px; flex:none; }
.bi-tenv { font:10.5px ui-monospace,monospace; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bi-mols { display:flex; gap:6px; overflow-x:auto; padding-bottom:4px; }
.bi-mol { flex:0 0 160px; background:#fff; color:#1f2328; border:1px solid var(--bi-border); border-radius:6px; padding:2px 4px 3px; min-width:0; }
.bi-mol svg { width:100% !important; height:auto !important; display:block; }
/* layout "side": table and detail side by side (2 : 1), one height; the table scrolls down, the
   detail's strips scroll sideways and its molecules fill the height left */
.bi-root.side .bi-split { display:grid; grid-template-columns:minmax(0, 2fr) minmax(0, 1fr); gap:12px; height:var(--bi-h); }
.bi-root.side .bi-wrap { max-height:none; min-height:0; }
.bi-root.side .bi-detail { margin-top:0; overflow:hidden; min-height:0; display:flex; flex-direction:column; }
.bi-root.side .bi-detail > * { flex:none; }
.bi-root.side .bi-detail > .bi-mols { flex:1 1 0; min-height:0; display:grid; grid-auto-flow:column;
  grid-template-rows:minmax(0, 1fr); grid-auto-columns:150px; overflow-y:hidden; }
.bi-root.side .bi-mol { display:flex; flex-direction:column; min-height:0; }
.bi-root.side .bi-mol svg { height:100% !important; min-height:0; flex:1; }
.bi-mhead { padding-right:26px; display:flex; justify-content:space-between; font-size:10.5px; color:#6b7280; gap:4px; }
.bi-mhead span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.bi-chip { font:inherit; font-size:12px; padding:1px 9px; border-radius:999px; border:1px solid var(--bi-border);
  background:var(--bi-card); color:var(--bi-fg); cursor:pointer; }
.bi-empty { color:var(--bi-muted); }
`;

// one colour per substructure of the selected bit, in the order they are listed
const PALETTE = ["#f59f00", "#1c7ed6", "#2f9e44", "#ae3ec9", "#e8590c", "#0c8599", "#d6336c", "#5c940d"];
const OTHER = "#adb5bd";
const colourOf = (k) => (k >= 0 && k < PALETTE.length ? PALETTE[k] : OTHER);
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
const pct = (v) => `${(100 * v).toFixed(1)}%`;
const share = (v) => (v > 0 && v < 0.005 ? "<1%" : `${Math.round(100 * v)}%`);
const signed = (v) => `<b style="color:${v >= 0 ? "#d63333" : "#3366e6"}">${v >= 0 ? "+" : ""}${v.toFixed(3)}</b>`;

async function render({ model, el: host }) {
  const root = el("div", { className: "bi-root" });
  enableSmilesCopy(root, model);
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  const get = (k) => model.get(k);
  const busy = busyIndicator(model, host, ["sort", "descending", "page", "page_size", "selected", "mol_filter", "mol_page"]);
  const set = (obj) => { busy(obj); for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };
  const RDKit = await loadRDKit();
  let tableScroll = 0;

  // pick a column to rank by; picking it again flips the direction (low → high finds unused bits)
  const sortBy = (key) => (key === get("sort") ? set({ descending: !get("descending") }) : set({ sort: key, descending: true }));
  const arrow = (key) => (key === get("sort") ? (get("descending") ? " ↓" : " ↑") : "");

  function tableSection() {
    const frag = document.createDocumentFragment();
    const names = get("score_names"), rows = get("rows"), effLabel = get("effect_label");
    const side = get("layout") === "side";
    const size = get("page_size"), page = get("page"), order = get("order");
    const pages = Math.max(1, Math.ceil(order.length / size));

    // one row: the pager and go to bit | the ranking
    const right = (...nodes) => { const d = el("div", { className: "bi-right" }); d.append(...nodes); return d; };
    const sortSel = el("select");
    for (const [v, label] of [...names.map((n) => [n, n]), ["mols", "# mols"], ["envs", "# envs"]])
      sortSel.append(new Option(`rank by ${label}`, v, false, v === get("sort")));
    sortSel.addEventListener("change", () => set({ sort: sortSel.value, descending: true }));
    const dirBtn = el("button", { className: "bi-btn", textContent: get("descending") ? "↓" : "↑",
      title: "high → low or low → high (low → high finds the bits the model does not use)" });
    dirBtn.onclick = () => set({ descending: !get("descending") });
    const jump = el("input", { type: "number", min: 0, max: get("n_bits") - 1, placeholder: "bit" });
    jump.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      const where = order.indexOf(parseInt(jump.value, 10));
      if (where >= 0) set({ page: Math.floor(where / size), selected: order[where] });
    });
    const btn = (text, disabled, p) => { const b = el("button", { className: "bi-btn", textContent: text, disabled }); b.onclick = () => set({ page: p }); return b; };
    const bar = el("div", { className: "bi-bar" });
    bar.append(
      btn("« first", page <= 0, 0), btn("‹ prev", page <= 0, page - 1),
      el("span", { className: "bi-lbl", textContent: `ranks ${page * size + 1}–${Math.min(order.length, (page + 1) * size)} of ${order.length.toLocaleString()}` }),
      btn("next ›", page >= pages - 1, page + 1), btn("last »", page >= pages - 1, pages - 1),
      el("span", { className: "bi-lbl", style: "margin-left:12px", textContent: "go to bit" }), jump,
      right(sortSel, dirBtn),
    );
    frag.appendChild(bar);

    const wrap = el("div", { className: "bi-wrap" });
    const table = el("table", { className: "bi-table" });
    const max = Object.fromEntries(names.map((n) => [n, Math.max(1e-12, get("max_score")[n] || 0)]));
    const cols = table.appendChild(el("colgroup"));
    const col = (w) => cols.appendChild(el("col", w ? { style: `width:${w}px` } : {}));
    col(40); col(60);
    for (const _ of names) col(180);
    if (effLabel) col(140);
    if (!side) col(150);
    col(70); col(80);
    const head = el("tr");
    const th = (text, key, cls = "", title = "") => {
      const h = el("th", { className: cls + (key ? " sortable" : "") + (key && key === get("sort") ? " on" : ""), textContent: text + (key ? arrow(key) : ""), title });
      if (key) h.onclick = () => sortBy(key);
      head.appendChild(h);
    };
    th("#"); th("bit");
    for (const n of names) th(n, n, "", `rank by ${n}; click again to flip`);
    if (effLabel) th(effLabel, null, "bi-eff", "red raises the prediction, blue lowers it");
    if (!side) th("main substructure", null, "bi-gstart", "the bit's most common substructure and the share of the bit's molecules that contain it");
    th("# envs", "envs", side ? "bi-num bi-gstart" : "bi-num", "distinct substructures that fold onto the bit (orange: the most common one is in under 80% of its molecules)");
    th("# mols", "mols", "bi-num", "molecules that set the bit");
    table.appendChild(el("thead")).appendChild(head);
    const body = table.appendChild(el("tbody"));
    for (const r of rows) {
      const tr = el("tr", { className: r.bit === get("selected") ? "sel" : "" });
      let html = `<td class="bi-rank">${r.rank}</td><td><b>${r.bit}</b></td>`;
      for (const n of names)
        html += `<td><div class="bi-score${n === get("sort") ? "" : " dim"}"><div class="track"><i style="width:${(100 * r.scores[n]) / max[n]}%"></i></div><span>${pct(r.scores[n])}</span></div></td>`;
      if (effLabel) html += `<td class="bi-eff">${r.effect == null ? "–" : signed(r.effect)}</td>`;
      if (!side)
        html += r.main_svg
          ? `<td class="bi-gstart"><div class="bi-thumb" title="${esc(r.main_env)}"><div class="pic">${r.main_svg}</div><span class="${r.main_share < 0.8 ? "bi-mixed" : "bi-lbl"}">${Math.round(100 * r.main_share)}%</span></div></td>`
          : `<td class="bi-lbl bi-gstart">not in this dataset</td>`;
      html += `<td class="bi-num${side ? " bi-gstart" : ""}${r.main_share < 0.8 && r.n_envs ? " bi-mixed" : ""}">${r.n_envs}</td><td class="bi-num">${r.n_mols.toLocaleString()}</td>`;
      tr.innerHTML = html;
      tr.onclick = () => set({ selected: r.bit });
      body.appendChild(tr);
    }
    wrap.appendChild(table);
    return [frag, wrap];
  }

  function detailSection() {
    const d = get("detail"), box = el("div", { className: "bi-detail" });
    if (!d || d.bit == null) { box.appendChild(el("div", { className: "bi-empty", textContent: "Click a row to see the bit's substructures and molecules." })); return box; }
    const filter = get("mol_filter");
    box.appendChild(el("div", { className: "bi-dhead" }, `<h4>bit ${d.bit}</h4><span class="bi-lbl">${d.envs.length} substructure${d.envs.length === 1 ? "" : "s"} · ${d.n_mols.toLocaleString()} molecules</span>`));

    // substructures: the colour key for the molecules below; click one to filter
    const sec = el("div", { className: "bi-sec" }, `<b>Substructures</b> click one to keep only the molecules that contain it`);
    if (filter >= 0) { const all = el("button", { className: "bi-chip", textContent: "show all molecules" }); all.onclick = () => set({ mol_filter: -1 }); sec.appendChild(all); }
    sec.appendChild(el("div", { className: "bi-legend bi-right" }, legendHtml("Pictures", MORGAN_ENV_KEY)));
    box.appendChild(sec);
    const strip = el("div", { className: "bi-strip" });
    d.envs.forEach((e, k) => {
      const t = el("div", { className: "bi-tile" + (filter === e.uid ? " on" : filter >= 0 ? " off" : ""), title: `${e.env}\ne.g. ${e.id}` },
        `<div class="bi-thead"><span class="bi-sw" style="background:${colourOf(k)}"></span><span>r${e.radius} · ${e.count.toLocaleString()} mols · <b>${share(e.count / Math.max(d.n_mols, 1))}</b></span></div>` +
        `${e.svg}<div class="bi-tenv">${esc(e.env)}</div>`);
      t.onclick = () => set({ mol_filter: filter === e.uid ? -1 : e.uid });
      strip.appendChild(t);
    });
    box.appendChild(strip);

    // the molecules that set the bit, with each substructure's atoms in its colour
    const m = get("mols") || {}, items = m.items || [], total = m.total || 0;
    const size = get("mol_page_size"), page = get("mol_page"), pages = Math.max(1, Math.ceil(total / size));
    const msec = el("div", { className: "bi-sec" },
      `<b>Molecules</b> ${total.toLocaleString()} with this bit` + (filter >= 0 ? " and the chosen substructure" : "") + (get("y_label") ? " · most active first" : ""));
    const nav = el("span", { className: "bi-bar", style: "margin:0" });
    const btn = (text, disabled, p) => { const b = el("button", { className: "bi-btn", textContent: text, disabled }); b.onclick = () => set({ mol_page: p }); return b; };
    nav.append(btn("‹ prev", page <= 0, page - 1),
      el("span", { className: "bi-lbl", textContent: total ? `${page * size + 1}–${Math.min(total, (page + 1) * size)} of ${total.toLocaleString()}` : "none" }),
      btn("next ›", page >= pages - 1, page + 1));
    msec.appendChild(nav);
    box.appendChild(msec);
    const grid = el("div", { className: "bi-mols" });
    for (const it of items) {
      const atoms = [], bonds = [], atomColors = {}, bondColors = {};
      for (const h of it.hits) {
        const c = rgb(colourOf(h.k));
        for (const a of h.atoms) { atoms.push(a); atomColors[a] = c; }
        for (const b of h.bonds) { bonds.push(b); bondColors[b] = c; }
      }
      const svg = drawSvg(RDKit, it.smiles, 220, 160, { atoms, bonds, atomColors, bondColors }, false);
      const y = it.y == null ? "" : `${esc(get("y_label"))} ${it.y.toFixed(2)}`;
      grid.appendChild(el("div", { className: "bi-mol mw-copyable", title: it.smiles }, `<div class="bi-mhead"><span>${esc(it.id)}</span><span>${y}</span></div>${svg}${smilesCopyHtml(it.smiles)}`));
    }
    box.appendChild(grid);
    return box;
  }

  function draw() {
    const oldWrap = root.querySelector(".bi-wrap");
    if (oldWrap) tableScroll = oldWrap.scrollTop;
    const oldDetail = root.querySelector(".bi-detail");
    const detailScroll = oldDetail ? oldDetail.scrollTop : 0;
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const [frag, wrap] = tableSection();
    const detail = detailSection();
    const side = get("layout") === "side";
    root.classList.toggle("side", side);
    if (side) {
      const split = el("div", { className: "bi-split" });
      split.append(wrap, detail);
      root.append(frag, split);
      root.style.setProperty("--bi-h", `${get("height")}px`);
    } else root.append(frag, wrap, detail);
    wrap.scrollTop = tableScroll;
    detail.scrollTop = detailScroll;
  }

  draw();
  // a new ranking or page starts at the top of the table; picking a bit keeps the scroll position
  for (const k of ["sort", "descending", "page"]) model.on(`change:${k}`, () => { tableScroll = 0; draw(); if (root.querySelector(".bi-wrap")) root.querySelector(".bi-wrap").scrollTop = 0; });
  for (const k of ["rows", "detail", "mols", "selected", "mol_filter", "score_names", "height", "layout"]) model.on(`change:${k}`, draw);
}

export default { render };
