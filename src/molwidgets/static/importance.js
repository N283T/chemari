// BitImportance: a fingerprint model's bits ranked by importance (table), and below it the selected
// bit's substructures and the molecules that set it, with the responsible atoms highlighted.
// Substructure pictures are RDKit DrawMorganEnv SVGs from Python; molecules are drawn with RDKit.js.
// Depends on loadRDKit / drawSvg / legendHtml / MORGAN_ENV_KEY / isDark / busyIndicator (prepended).

const CSS = `
.bi-root { font: 13px/1.45 system-ui, sans-serif; color: var(--bi-fg); --bi-fg:#1f2328; --bi-muted:#6b7280;
  --bi-border:#d0d7de; --bi-card:#fff; --bi-soft:#f6f8fa; --bi-sel:#fff4e0; --bi-bar:#e8590c; --bi-track:#eef1f4; }
.bi-root.dark { --bi-fg:#e6e6e6; --bi-muted:#9aa4b2; --bi-border:#3a3f47; --bi-card:#1c1f24; --bi-soft:#24282e;
  --bi-sel:#3a2f1c; --bi-track:#2c3139; }
.bi-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:6px; }
.bi-seg { display:inline-flex; border:1px solid var(--bi-border); border-radius:6px; overflow:hidden; }
.bi-seg button { font:inherit; color:var(--bi-fg); background:var(--bi-soft); border:0; padding:2px 9px; cursor:pointer; }
.bi-seg button + button { border-left:1px solid var(--bi-border); }
.bi-seg button.on { background:var(--bi-fg); color:var(--bi-card); }
.bi-lbl { color:var(--bi-muted); }
.bi-bar .bi-btn { font:inherit; color:var(--bi-fg); background:var(--bi-soft); border:1px solid var(--bi-border);
  border-radius:6px; padding:2px 9px; cursor:pointer; }
.bi-bar .bi-btn:disabled { opacity:.4; cursor:default; }
.bi-bar input { font:inherit; width:70px; padding:2px 6px; border:1px solid var(--bi-border); border-radius:6px;
  background:var(--bi-card); color:var(--bi-fg); }
.bi-legend { color:var(--bi-muted); font-size:11.5px; margin:0 0 8px; line-height:1.9; }
.bi-wrap { border:1px solid var(--bi-border); border-radius:8px; max-height:420px; overflow:auto; }
.bi-table { border-collapse:collapse; width:100%; font-size:12.5px; }
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
.bi-rank { color:var(--bi-muted); }
.bi-score { display:flex; align-items:center; gap:6px; }
.bi-score .track { width:56px; height:6px; border-radius:3px; background:var(--bi-track); overflow:hidden; flex:none; }
.bi-score .track i { display:block; height:100%; background:var(--bi-bar); }
.bi-score span { font-variant-numeric:tabular-nums; min-width:38px; }
.bi-score.dim .track i { opacity:.35; }
.bi-score.dim span { color:var(--bi-muted); }
.bi-thumb { display:flex; align-items:center; gap:6px; }
.bi-thumb .pic { width:56px; height:42px; flex:none; background:#fff; border:1px solid var(--bi-border); border-radius:4px; overflow:hidden; }
.bi-thumb .pic svg { width:100% !important; height:100% !important; display:block; }
.bi-mixed { color:#e8590c; font-weight:600; }
.bi-detail { margin-top:12px; border:1px solid var(--bi-border); border-radius:8px; padding:8px 10px; }
.bi-detail h4 { margin:0 0 2px; font-size:14px; }
.bi-sec { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:10px 0 6px; color:var(--bi-muted); }
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
.bi-mhead { display:flex; justify-content:space-between; font-size:10.5px; color:#6b7280; gap:4px; }
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

function seg(options, current, onPick) {
  const wrap = el("div", { className: "bi-seg" });
  for (const [val, label] of options) {
    const b = el("button", { textContent: label });
    if (val === current) b.classList.add("on");
    b.addEventListener("click", () => onPick(val));
    wrap.appendChild(b);
  }
  return wrap;
}

const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
const pct = (v) => `${(100 * v).toFixed(1)}%`;
const share = (v) => (v > 0 && v < 0.005 ? "<1%" : `${Math.round(100 * v)}%`);
const signed = (v) => `<b style="color:${v >= 0 ? "#d63333" : "#3366e6"}">${v >= 0 ? "+" : ""}${v.toFixed(3)}</b>`;

async function render({ model, el: host }) {
  const root = el("div", { className: "bi-root" });
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
    const size = get("page_size"), page = get("page"), order = get("order");
    const pages = Math.max(1, Math.ceil(order.length / size));

    const bar = el("div", { className: "bi-bar" });
    bar.append(
      el("span", { className: "bi-lbl", textContent: "rank by" }),
      seg([...names.map((n) => [n, n]), ["mols", "# mols"], ["envs", "# envs"]], get("sort"), (v) => sortBy(v)),
      seg([[true, "high → low"], [false, "low → high"]], get("descending"), (v) => set({ descending: v })),
      el("span", { className: "bi-lbl", textContent: `radius ${get("radius")} · ${get("n_bits").toLocaleString()} bits` }),
    );
    frag.appendChild(bar);
    const nav = el("div", { className: "bi-bar" });
    const btn = (text, disabled, p) => { const b = el("button", { className: "bi-btn", textContent: text, disabled }); b.onclick = () => set({ page: p }); return b; };
    const jump = el("input", { type: "number", min: 0, max: get("n_bits") - 1, placeholder: "bit" });
    jump.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;
      const where = order.indexOf(parseInt(jump.value, 10));
      if (where >= 0) set({ page: Math.floor(where / size), selected: order[where] });
    });
    nav.append(
      btn("« first", page <= 0, 0), btn("‹ prev", page <= 0, page - 1),
      el("span", { className: "bi-lbl", textContent: `ranks ${page * size + 1}–${Math.min(order.length, (page + 1) * size)} of ${order.length.toLocaleString()}` }),
      btn("next ›", page >= pages - 1, page + 1), btn("last »", page >= pages - 1, pages - 1),
      el("span", { className: "bi-lbl", style: "margin-left:12px", textContent: "go to bit" }), jump,
    );
    frag.appendChild(nav);
    frag.appendChild(el("div", { className: "bi-legend" }, legendHtml("Pictures", MORGAN_ENV_KEY)));

    const wrap = el("div", { className: "bi-wrap" });
    const table = el("table", { className: "bi-table" });
    const max = Object.fromEntries(names.map((n) => [n, Math.max(1e-12, get("max_score")[n] || 0)]));
    const head = el("tr");
    const th = (text, key, cls = "", title = "") => {
      const h = el("th", { className: cls + (key ? " sortable" : "") + (key && key === get("sort") ? " on" : ""), textContent: text + (key ? arrow(key) : ""), title });
      if (key) h.onclick = () => sortBy(key);
      head.appendChild(h);
    };
    th("#"); th("bit");
    for (const n of names) th(n, n, "", `rank by ${n}; click again to flip`);
    if (effLabel) th(effLabel, null, "bi-num", "red raises the prediction, blue lowers it");
    th("main substructure", null, "", "the bit's most common substructure and the share of the bit's molecules that contain it");
    th("# envs", "envs", "bi-num", "distinct substructures that fold onto the bit");
    th("# mols", "mols", "bi-num", "molecules that set the bit");
    table.appendChild(el("thead")).appendChild(head);
    const body = table.appendChild(el("tbody"));
    for (const r of rows) {
      const tr = el("tr", { className: r.bit === get("selected") ? "sel" : "" });
      let html = `<td class="bi-rank">${r.rank}</td><td><b>${r.bit}</b></td>`;
      for (const n of names)
        html += `<td><div class="bi-score${n === get("sort") ? "" : " dim"}"><div class="track"><i style="width:${(100 * r.scores[n]) / max[n]}%"></i></div><span>${pct(r.scores[n])}</span></div></td>`;
      if (effLabel) html += `<td class="bi-num">${r.effect == null ? "–" : signed(r.effect)}</td>`;
      html += r.main_svg
        ? `<td><div class="bi-thumb" title="${esc(r.main_env)}"><div class="pic">${r.main_svg}</div><span class="${r.main_share < 0.8 ? "bi-mixed" : "bi-lbl"}">${Math.round(100 * r.main_share)}%</span></div></td>`
        : `<td class="bi-lbl">not in this dataset</td>`;
      html += `<td class="bi-num${r.main_share < 0.8 && r.n_envs ? " bi-mixed" : ""}">${r.n_envs}</td><td class="bi-num">${r.n_mols.toLocaleString()}</td>`;
      tr.innerHTML = html;
      tr.onclick = () => set({ selected: r.bit });
      body.appendChild(tr);
    }
    wrap.appendChild(table);
    frag.appendChild(wrap);
    return [frag, wrap];
  }

  function detailSection() {
    const d = get("detail"), box = el("div", { className: "bi-detail" });
    if (!d || d.bit == null) { box.appendChild(el("div", { className: "bi-empty", textContent: "Click a row to see the bit's substructures and molecules." })); return box; }
    const filter = get("mol_filter");
    box.appendChild(el("h4", {}, `bit ${d.bit}`));
    box.appendChild(el("div", { className: "bi-lbl" }, `${d.envs.length} substructure${d.envs.length === 1 ? "" : "s"} · ${d.n_mols.toLocaleString()} molecules`));

    // substructures: the colour key for the molecules below; click one to filter
    const sec = el("div", { className: "bi-sec" }, `<b>Substructures</b> click one to keep only the molecules that contain it`);
    if (filter >= 0) { const all = el("button", { className: "bi-chip", textContent: "show all molecules" }); all.onclick = () => set({ mol_filter: -1 }); sec.appendChild(all); }
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
      grid.appendChild(el("div", { className: "bi-mol", title: it.smiles }, `<div class="bi-mhead"><span>${esc(it.id)}</span><span>${y}</span></div>${svg}`));
    }
    box.appendChild(grid);
    return box;
  }

  function draw() {
    const oldWrap = root.querySelector(".bi-wrap");
    if (oldWrap) tableScroll = oldWrap.scrollTop;
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const [frag, wrap] = tableSection();
    root.appendChild(frag);
    root.appendChild(detailSection());
    wrap.scrollTop = tableScroll;
  }

  draw();
  // a new ranking or page starts at the top of the table; picking a bit keeps the scroll position
  for (const k of ["sort", "descending", "page"]) model.on(`change:${k}`, () => { tableScroll = 0; draw(); if (root.querySelector(".bi-wrap")) root.querySelector(".bi-wrap").scrollTop = 0; });
  for (const k of ["rows", "detail", "mols", "selected", "mol_filter", "score_names"]) model.on(`change:${k}`, draw);
}

export default { render };
