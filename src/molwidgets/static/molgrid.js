// MolGrid: a paged, sortable, SMARTS-searchable molecule grid with two-way selection; `group_by`
// adds buttons to show one group of a column (e.g. train / test) and colours each card's edge by
// it (the colour scale then becomes a dot before the value's number). One
// search box with a mode: text filter, substructure filter (SMARTS), or similarity (ranks by ECFP4
// Tanimoto to a SMILES, or to the first selected molecule; computed in Python).
// Depends on loadRDKit / drawSvg / isDark (prepended by the Python side).

const PALETTE = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]; // viridis stops
const GROUP_COLORS = ["#adb5bd", "#1c7ed6", "#f08c00", "#2f9e44", "#ae3ec9"]; // card edge per group

function lerpColor(t) {
  t = Math.max(0, Math.min(1, t));
  const pos = t * (PALETTE.length - 1);
  const i = Math.min(Math.floor(pos), PALETTE.length - 2);
  const f = pos - i;
  const a = PALETTE[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = PALETTE[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
}

const CSS = `
.mg-root { font: 13px/1.35 system-ui, sans-serif; color: var(--mg-fg); --mg-fg: #1f2328; --mg-muted: #6b7280;
  --mg-border: #d0d7de; --mg-card: #ffffff; --mg-accent: #1c7ed6; --mg-soft: #f6f8fa; }
.mg-root.dark { --mg-fg: #e6e6e6; --mg-muted: #9aa4b2; --mg-border: #3a3f47; --mg-card: #1c1f24;
  --mg-accent: #4dabf7; --mg-soft: #24282e; }
.mg-bars { display: flex; flex-direction: column; gap: 6px; margin-bottom: 8px; }
.mg-bar { display: flex; flex-wrap: wrap; gap: 6px 10px; align-items: center; }
.mg-right { display: flex; gap: 6px; align-items: center; margin-left: auto; }
/* the search box with its mode switch attached to its right end */
.mg-find { display: flex; flex: 1; min-width: 16em; max-width: 34em; }
.mg-bar .mg-find input.mg-box { flex: 1; min-width: 0; max-width: none; border-radius: 6px 0 0 6px; border-right: 0; }
.mg-bar .mg-find .mg-modes { border-radius: 0 6px 6px 0; }
.mg-bar input, .mg-bar select, .mg-bar button { font: inherit; color: var(--mg-fg); background: var(--mg-soft);
  border: 1px solid var(--mg-border); border-radius: 6px; padding: 3px 7px; }
.mg-bar input.mg-box { flex: 1; min-width: 9em; max-width: 22em; }
.mg-bar input.mg-box.mono { font-family: ui-monospace, monospace; }
.mg-bar button.mg-clear { padding: 0 6px; line-height: 1.5; color: var(--mg-muted); }
.mg-bar input.bad { border-color: #e03131; }
.mg-bar button { cursor: pointer; }
.mg-bar .mg-info { color: var(--mg-muted); }
.mg-grid { display: grid; gap: 8px; grid-template-columns: repeat(auto-fill, minmax(var(--mg-cell), 1fr)); }
.mg-card { position: relative; background: var(--mg-card); border: 1px solid var(--mg-border); border-radius: 8px;
  padding: 4px 6px 6px; cursor: pointer; overflow: hidden; transition: box-shadow .12s, border-color .12s; }
.mg-card:hover { box-shadow: 0 1px 6px rgba(0,0,0,.15); }
.mg-card.sel { border-color: var(--mg-accent); box-shadow: 0 0 0 2px var(--mg-accent) inset; }
.mg-card .mg-strip { position: absolute; left: 0; top: 0; bottom: 0; width: 5px; }
.mg-card svg { display: block; width: 100%; height: auto; }
.mg-card .mg-id { padding-right: 26px; font-weight: 600; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.mg-card .mg-props { display: grid; grid-template-columns: auto 1fr; column-gap: 6px; font-size: 11.5px; }
.mg-card .mg-props span:nth-child(odd) { color: var(--mg-muted); }
.mg-card .mg-props span:nth-child(even) { text-align: right; font-variant-numeric: tabular-nums; }
.mg-groups { display: inline-flex; border: 1px solid var(--mg-border); border-radius: 6px; overflow: hidden; }
.mg-bar .mg-groups button { border: 0; border-radius: 0; padding: 3px 9px; }
.mg-bar .mg-groups button + button { border-left: 1px solid var(--mg-border); }
.mg-bar .mg-groups button.on { background: var(--mg-fg); color: var(--mg-card); }
/* the group tag sits in the drawing's top-left corner, under the id */
/* with groups the strip shows the group and the value's colour becomes a dot before its number */
.mg-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: -0.5px;
  border: 1px solid rgba(0,0,0,.18); }
.mg-groups button i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 5px; }
.mg-legend { display: flex; align-items: center; gap: 6px; color: var(--mg-muted); font-size: 11.5px; }
.mg-legend .mg-ramp { width: 90px; height: 8px; border-radius: 4px;
  background: linear-gradient(90deg, ${PALETTE.join(",")}); }
.mg-pager { display: flex; justify-content: center; gap: 8px; align-items: center; margin-top: 8px; }
.mg-empty { color: var(--mg-muted); padding: 24px; text-align: center; }
`;

async function render({ model, el }) {
  const root = document.createElement("div");
  enableSmilesCopy(root, model);
  root.className = "mg-root";
  if (isDark(el)) root.classList.add("dark");
  const style = document.createElement("style");
  style.textContent = CSS;
  root.appendChild(style);

  // two rows: what to look at (groups, sort) and how to find things (search, colour key, counts)
  const bar = document.createElement("div");
  bar.className = "mg-bars";
  const row1 = Object.assign(document.createElement("div"), { className: "mg-bar" });
  const row2 = Object.assign(document.createElement("div"), { className: "mg-bar" });
  bar.append(row1, row2);
  // one search box; the select before it says what it searches
  const box = Object.assign(document.createElement("input"), { className: "mg-box", type: "search" });
  // "all" plus one button per value of the `group_by` column (e.g. train / test)
  const groups = document.createElement("div");
  groups.className = "mg-groups";
  let group = null; // null = all
  // text: any field contains it; substructure: SMARTS match; similarity: rank by ECFP4 Tanimoto
  // to a SMILES (computed in Python)
  const modeSel = Object.assign(document.createElement("div"), { className: "mg-groups mg-modes" });
  const MODES = ["text", "substructure", "similarity"];
  const sortSel = document.createElement("select");
  const dirBtn = Object.assign(document.createElement("button"), { title: "sort direction" });
  const clearBtn = Object.assign(document.createElement("button"), { textContent: "×", title: "clear selection", className: "mg-clear" });
  const legend = document.createElement("div");
  legend.className = "mg-legend";
  const info = document.createElement("span");
  info.className = "mg-info";
  const find = Object.assign(document.createElement("div"), { className: "mg-find" });
  find.append(box, modeSel);
  const sortWrap = Object.assign(document.createElement("div"), { className: "mg-right" });
  sortWrap.append(sortSel, dirBtn);
  const counts = Object.assign(document.createElement("div"), { className: "mg-right" });
  counts.append(legend, info, clearBtn);
  row1.append(groups, sortWrap);
  row2.append(find, counts);

  const grid = document.createElement("div");
  grid.className = "mg-grid";
  const pager = document.createElement("div");
  pager.className = "mg-pager";
  const first = Object.assign(document.createElement("button"), { textContent: "« first" });
  const prev = Object.assign(document.createElement("button"), { textContent: "‹ prev" });
  const next = Object.assign(document.createElement("button"), { textContent: "next ›" });
  const last = Object.assign(document.createElement("button"), { textContent: "last »" });
  const pageLbl = document.createElement("span");
  pager.append(first, prev, pageLbl, next, last);
  for (const b of [first, prev, next, last]) b.style.cssText = "font:inherit;padding:2px 10px;border-radius:6px;cursor:pointer";
  root.append(bar, grid, pager);
  el.appendChild(root);

  grid.innerHTML = `<div class="mg-empty">loading RDKit.js…</div>`;
  const RDKit = await loadRDKit();

  let page = 0;
  let desc = true;
  const svgCache = new Map();
  let query = null; // compiled RDKit qmol
  let matchCache = new Map();

  const get = (k) => model.get(k);
  const idOf = (r) => String(r[get("id_col")]);

  const groupValues = () => {
    const key = get("group_by");
    return key ? [...new Set(get("data").map((r) => String(r[key])))] : [];
  };

  const groupColor = (v) => GROUP_COLORS[groupValues().indexOf(String(v)) % GROUP_COLORS.length];

  function setupGroups() {
    const values = groupValues();
    if (group !== null && !values.includes(group)) group = null;
    groups.innerHTML = "";
    groups.style.display = values.length > 1 ? "inline-flex" : "none";
    const count = (v) => get("data").filter((r) => String(r[get("group_by")]) === v).length;
    for (const [v, label] of [[null, `all ${get("data").length}`], ...values.map((v) => [v, `${v} ${count(v)}`])]) {
      const b = Object.assign(document.createElement("button"), { className: v === group ? "on" : "" });
      // the group's colour, as on the cards' left edge
      b.innerHTML = (v === null ? "" : `<i style="background:${groupColor(v)}"></i>`) + label;
      b.addEventListener("click", () => { group = v; page = 0; setupGroups(); draw(); });
      groups.appendChild(b);
    }
  }

  function setupSort() {
    const keys = get("data").length ? Object.keys(get("data")[0]) : [];
    const current = get("sort_by");
    sortSel.innerHTML = "";
    sortSel.append(new Option("(original order)", ""));
    for (const k of keys) if (k !== get("smiles_col")) sortSel.append(new Option(`sort: ${k}`, k));
    sortSel.value = keys.includes(current) ? current : "";
    dirBtn.textContent = desc ? "↓" : "↑";
  }

  const mode = () => get("search_mode");
  const similar = () => mode() === "similarity";
  // the similarity query: the SMILES typed in the box, else the first selected molecule
  function sendSimilarityQuery() {
    let q = "";
    if (similar()) {
      q = box.value.trim();
      if (!q) {
        const first = get("selection")[0];
        const row = first === undefined ? null : get("data").find((r) => idOf(r) === String(first));
        q = row ? row[get("smiles_col")] : "";
      }
    }
    if (q !== get("similarity_query")) { model.set("similarity_query", q); model.save_changes(); }
  }

  function setupMode() {
    modeSel.innerHTML = "";
    for (const m of MODES) {
      const b = Object.assign(document.createElement("button"), { textContent: m, className: m === mode() ? "on" : "" });
      b.addEventListener("click", () => setMode(m));
      modeSel.appendChild(b);
    }
    box.placeholder = { text: "filter text…", substructure: "SMARTS / SMILES", similarity: "SMILES (or select a molecule)" }[mode()] || "";
    box.classList.toggle("mono", mode() !== "text");
  }

  function compileQuery() {
    matchCache = new Map();
    if (query) {
      query.delete();
      query = null;
    }
    const q = mode() === "substructure" ? box.value.trim() : "";
    if (q) query = RDKit.get_qmol(q);
  }

  function matchOf(smi) {
    if (!query) return null;
    if (matchCache.has(smi)) return matchCache.get(smi);
    const mol = RDKit.get_mol(smi);
    let m = null;
    if (mol) {
      const res = JSON.parse(mol.get_substruct_match(query));
      if (res.atoms) m = res;
      mol.delete();
    }
    matchCache.set(smi, m);
    return m;
  }

  function filtered() {
    const text = mode() === "text" ? box.value.trim().toLowerCase() : "";
    let rows = get("data");
    if (group !== null) rows = rows.filter((r) => String(r[get("group_by")]) === group);
    if (text) rows = rows.filter((r) => Object.values(r).some((v) => String(v).toLowerCase().includes(text)));
    if (query) rows = rows.filter((r) => matchOf(r[get("smiles_col")]));
    const sims = similar() ? get("similarity")?.values : null;
    if (sims) return [...rows].sort((a, b) => (sims[idOf(b)] ?? -1) - (sims[idOf(a)] ?? -1));
    const key = sortSel.value;
    if (key) {
      rows = [...rows].sort((a, b) => {
        const x = a[key], y = b[key];
        const bad = (v) => v === null || v === undefined || Number.isNaN(v);
        if (bad(x)) return 1;
        if (bad(y)) return -1;
        const c = x < y ? -1 : x > y ? 1 : 0;
        return desc ? -c : c;
      });
    }
    return rows;
  }

  function colorScale() {
    const key = get("color_by");
    if (!key) return null;
    const vals = get("data").map((r) => r[key]).filter((v) => typeof v === "number" && !Number.isNaN(v));
    if (!vals.length) return null;
    const lo = get("color_range")?.[0] ?? Math.min(...vals);
    const hi = get("color_range")?.[1] ?? Math.max(...vals);
    return { key, lo, hi, f: (v) => (typeof v === "number" ? lerpColor((v - lo) / (hi - lo || 1)) : "transparent") };
  }

  function draw() {
    const rows = filtered();
    const size = get("cell_size");
    // round the page up to whole rows, so the last row is never half empty
    const gap = 8;
    const cols = Math.max(1, Math.floor((grid.clientWidth + gap) / (size + gap)));
    const per = cols * Math.max(1, Math.round(get("page_size") / cols));
    const pages = Math.max(1, Math.ceil(rows.length / per));
    page = Math.min(page, pages - 1);
    const sel = new Set(get("selection").map(String));
    const hls = get("highlights") || {};
    const subset = get("subset");
    const scale = colorScale();
    const dark = root.classList.contains("dark");
    root.style.setProperty("--mg-cell", `${size}px`);

    legend.innerHTML = scale && get("show_legend")
      ? `<span>${scale.key}</span><span>${fmt(scale.lo)}</span><div class="mg-ramp"></div><span>${fmt(scale.hi)}</span>`
      : "";
    const sims = similar() ? get("similarity")?.values : null;
    box.classList.toggle("bad", similar() ? !!get("similarity")?.error : mode() === "substructure" && !!box.value.trim() && !query);
    clearBtn.style.display = sel.size ? "" : "none";
    sortSel.disabled = dirBtn.disabled = !!sims; // ranked by similarity instead
    info.textContent = `${rows.length} / ${get("data").length} shown · ${sel.size} selected` + (sims ? " · by similarity" : "");
    pageLbl.textContent = `page ${page + 1} / ${pages}`;
    first.disabled = prev.disabled = page === 0;
    last.disabled = next.disabled = page >= pages - 1;
    pager.style.display = pages > 1 ? "flex" : "none";

    grid.innerHTML = "";
    if (!rows.length) {
      grid.innerHTML = `<div class="mg-empty">no molecules match</div>`;
      return;
    }
    for (const r of rows.slice(page * per, (page + 1) * per)) {
      const id = idOf(r);
      const smi = r[get("smiles_col")];
      const card = document.createElement("div");
      card.className = "mg-card mw-copyable" + (sel.has(id) ? " sel" : "");
      card.title = Object.entries(r)
        .map(([k, v]) => `${k}: ${fmt(v)}`)
        .join("\n");
      const hl = matchOf(smi) || hls[id] || null;
      const ck = `${smi}|${size}|${dark}|${hl ? JSON.stringify(hl) : ""}`;
      if (!svgCache.has(ck)) svgCache.set(ck, drawSvg(RDKit, smi, size, Math.round(size * 0.8), hl, dark));
      // the left edge: the group when there are groups, else the colour scale; with groups the
      // scale colours the value itself
      const gkey = get("group_by");
      const grouped = gkey && gkey in r;
      const stripColor = grouped ? groupColor(r[gkey]) : scale ? scale.f(r[scale.key]) : null;
      const strip = stripColor ? `<div class="mg-strip" style="background:${stripColor}"></div>` : "";
      const value = (k) => grouped && scale && k === scale.key && typeof r[k] === "number"
        ? `<i class="mg-dot" style="background:${scale.f(r[k])}"></i>${fmt(r[k])}`
        : fmt(r[k]);
      const props = (sims ? `<span>Tanimoto</span><span><b>${fmt(sims[id])}</b></span>` : "") + subset
        .filter((k) => k in r && k !== gkey)
        .map((k) => `<span>${k}</span><span>${value(k)}</span>`)
        .join("");
      card.innerHTML = `${strip}<div class="mg-id">${id}</div>${svgCache.get(ck)}${smilesCopyHtml(smi)}<div class="mg-props">${props}</div>`;
      card.addEventListener("click", (ev) => toggle(id, ev.shiftKey || ev.metaKey || ev.ctrlKey));
      grid.appendChild(card);
    }
  }

  function toggle(id, additive) {
    const mode = get("selection_mode");
    let sel = get("selection").map(String);
    if (sel.includes(id)) sel = sel.filter((s) => s !== id);
    else if (mode === "single") sel = [id];
    else if (mode === "pair") sel = additive || sel.length === 1 ? [...sel, id].slice(-2) : [id];
    else sel = [...sel, id];
    model.set("selection", sel);
    model.save_changes();
  }

  // text filters as you type; the structure searches run on Enter (or leaving the box)
  box.addEventListener("input", () => { if (mode() === "text") { page = 0; draw(); } });
  box.addEventListener("change", () => {
    if (mode() === "substructure") { model.set("smarts", box.value.trim()); model.save_changes(); compileQuery(); }
    if (similar()) sendSimilarityQuery();
    page = 0;
    draw();
  });
  function setMode(m) {
    if (m === mode()) return;
    box.value = ""; // what was typed belongs to the other kind of search
    model.set("search_mode", m);
    if (get("smarts")) model.set("smarts", "");
    model.save_changes();
    setupMode(); compileQuery(); sendSimilarityQuery(); page = 0; draw();
  }
  sortSel.addEventListener("change", () => { page = 0; model.set("sort_by", sortSel.value); model.save_changes(); draw(); });
  dirBtn.addEventListener("click", () => { desc = !desc; dirBtn.textContent = desc ? "↓" : "↑"; draw(); });
  clearBtn.addEventListener("click", () => { model.set("selection", []); model.save_changes(); });
  prev.addEventListener("click", () => { page--; draw(); });
  next.addEventListener("click", () => { page++; draw(); });
  first.addEventListener("click", () => { page = 0; draw(); });
  last.addEventListener("click", () => { page = Infinity; draw(); }); // draw() clamps to the last page

  if (mode() === "substructure") box.value = get("smarts") || "";
  setupMode();
  compileQuery();
  sendSimilarityQuery();
  setupGroups();
  setupSort();
  draw();

  const onData = () => { svgCache.clear(); matchCache = new Map(); page = 0; setupGroups(); setupSort(); draw(); };
  model.on("change:data", onData);
  for (const k of ["highlights", "subset", "color_by", "color_range", "show_legend", "page_size", "cell_size"])
    model.on(`change:${k}`, draw);
  // with an empty box the first selected molecule is the similarity query
  model.on("change:selection", () => { sendSimilarityQuery(); draw(); });
  model.on("change:similarity", () => { page = 0; draw(); });
  model.on("change:search_mode", () => { setupMode(); compileQuery(); sendSimilarityQuery(); draw(); });
  let lastCols = 0;
  new ResizeObserver(() => {
    const cols = Math.max(1, Math.floor((grid.clientWidth + 8) / (get("cell_size") + 8)));
    if (cols !== lastCols) { lastCols = cols; draw(); }
  }).observe(grid);
  model.on("change:smarts", () => {
    if (mode() !== "substructure") return;
    box.value = get("smarts") || ""; compileQuery(); page = 0; draw();
  });
  model.on("change:sort_by", () => { sortSel.value = get("sort_by") || ""; draw(); });

  return () => { if (query) query.delete(); };
}

export default { render };
