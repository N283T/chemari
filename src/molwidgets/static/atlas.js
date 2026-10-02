// BitAtlas: every folded bit of a reference set, one row per bit, with the distinct substructures
// that fold onto it (RDKit DrawMorganEnv pictures rendered in Python for the visible page). To the
// right, how crowded the bits are: the number of bits holding each number of substructures (a
// click on a bar lists only those bits) and the dataset's totals.
// Depends on legendHtml / MORGAN_ENV_KEY / isDark (prepended by the Python side).

const CSS = `
.ba-root { font: 13px/1.45 system-ui, sans-serif; color: var(--ba-fg); --ba-fg:#1f2328; --ba-muted:#6b7280;
  --ba-border:#d0d7de; --ba-card:#fff; --ba-soft:#f6f8fa; --ba-accent:#1c7ed6; --ba-bar:#adb5bd; --ba-focus:#e7f5ff; }
.ba-root.dark { --ba-fg:#e6e6e6; --ba-muted:#9aa4b2; --ba-border:#3a3f47; --ba-card:#1c1f24; --ba-soft:#24282e;
  --ba-bar:#5c636e; --ba-focus:#1b2a3a; }
.ba-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:6px; }
.ba-seg { display:inline-flex; border:1px solid var(--ba-border); border-radius:6px; overflow:hidden; }
.ba-seg button { font:inherit; color:var(--ba-fg); background:var(--ba-soft); border:0; padding:2px 9px; cursor:pointer; }
.ba-seg button + button { border-left:1px solid var(--ba-border); }
.ba-seg button.on { background:var(--ba-fg); color:var(--ba-card); }
.ba-lbl { color:var(--ba-muted); }
.ba-right { display:flex; gap:6px; align-items:center; margin-left:auto; }
.ba-pager { justify-content:center; margin:8px 0 2px; }
.ba-legend { color:var(--ba-muted); font-size:11.5px; line-height:1.9; }
.ba-bar select { font:inherit; color:var(--ba-fg); background:var(--ba-soft); border:1px solid var(--ba-border);
  border-radius:6px; padding:3px 7px; }
.ba-bar input { font:inherit; width:70px; padding:2px 6px; border:1px solid var(--ba-border); border-radius:6px;
  background:var(--ba-card); color:var(--ba-fg); }
.ba-bar .ba-btn { font:inherit; color:var(--ba-fg); background:var(--ba-soft); border:1px solid var(--ba-border);
  border-radius:6px; padding:2px 9px; cursor:pointer; }
.ba-bar .ba-btn:disabled { opacity:.4; cursor:default; }
/* a fixed height, so the widget does not jump when a filter leaves only a few rows */
.ba-list { border:1px solid var(--ba-border); border-radius:8px; height:520px; overflow-y:auto; position:relative; }
.ba-body { display:flex; gap:8px; align-items:stretch; }
.ba-body .ba-list { flex:1; min-width:0; }
.ba-hist { flex:none; width:210px; border:1px solid var(--ba-border); border-radius:8px; padding:8px 10px; font-size:12px;
  display:flex; flex-direction:column; gap:2px; }
.ba-hist .h { font-weight:600; font-size:13px; margin-bottom:2px; }
.ba-hist .r { display:grid; grid-template-columns: 44px 1fr 34px; gap:8px; align-items:center; height:20px; padding:0 4px; margin:0 -4px;
  border-radius:4px; cursor:pointer; }
.ba-hist .r.head { height:auto; cursor:default; color:var(--ba-muted); font-size:11px; }
.ba-hist .r.none { cursor:default; color:var(--ba-muted); }
.ba-hist .r:not(.none):not(.head):hover { background:var(--ba-soft); }
.ba-hist .r.on { background:var(--ba-focus); }
.ba-hist .r.on .t i { background:var(--ba-accent); }
.ba-hist .k { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.ba-hist .t { height:10px; background:var(--ba-soft); border-radius:2px; overflow:hidden; }
.ba-hist .t i { display:block; height:100%; background:var(--ba-bar); }
.ba-hist .n { color:var(--ba-muted); font-variant-numeric:tabular-nums; text-align:right; }
/* the dataset's totals: label left, value right, one per line */
.ba-hist .s { padding-bottom:8px; margin-bottom:6px; border-bottom:1px solid var(--ba-border); display:grid; grid-template-columns:1fr auto;
  gap:3px 8px; color:var(--ba-muted); white-space:nowrap; }
.ba-hist .s b { color:var(--ba-fg); font-variant-numeric:tabular-nums; text-align:right; }
.ba-hist .hint { color:var(--ba-muted); font-size:11px; margin-top:auto; padding-top:6px; }
@media (max-width: 640px) { .ba-body { flex-direction:column; } .ba-hist { width:auto; } }
.ba-row { display:grid; grid-template-columns: 128px 1fr; gap:10px; align-items:center; padding:6px 8px;
  border-bottom:1px solid var(--ba-border); }
.ba-row:last-child { border-bottom:0; }
.ba-row.focus { background:var(--ba-focus); }
.ba-meta b { font-size:14px; }
.ba-num { color:var(--ba-muted); font-size:11.5px; }
.ba-meter { height:4px; border-radius:2px; background:var(--ba-soft); margin-top:3px; overflow:hidden; }
.ba-meter i { display:block; height:100%; background:var(--ba-bar); }
.ba-strip { display:flex; gap:6px; overflow-x:auto; padding-bottom:2px; }
.ba-tile { flex:0 0 104px; width:104px; min-width:0; overflow:hidden; background:#fff; color:#1f2328; border:1px solid var(--ba-border); border-radius:6px;
  padding:2px 4px 3px; }
.ba-tile svg { width:100% !important; height:auto !important; max-width:100%; display:block; }
.ba-thead { display:flex; justify-content:space-between; font-size:10.5px; color:#6b7280; }
.ba-tenv { font:10.5px ui-monospace,monospace; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ba-more { flex:0 0 auto; align-self:center; color:var(--ba-muted); font-size:12px; padding:0 6px; white-space:nowrap; }
.ba-empty { color:var(--ba-muted); font-size:12px; }
`;

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

function seg(options, current, onPick) {
  const wrap = el("div", { className: "ba-seg" });
  for (const [val, label] of options) {
    const b = el("button", { textContent: label });
    if (val === current) b.classList.add("on");
    b.addEventListener("click", () => onPick(val));
    wrap.appendChild(b);
  }
  return wrap;
}

const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");

function render({ model, el: host }) {
  const root = el("div", { className: "ba-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  const get = (k) => model.get(k);
  const busy = busyIndicator(model, host, ["radius", "n_bits", "sort", "reverse", "count_filter", "page", "page_size", "per_row"]);
  const set = (obj) => { busy(obj); for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };

  function draw() {
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const nBits = get("n_bits"), size = get("page_size"), page = get("page");
    const pages = Math.max(1, Math.ceil(get("order").length / size)); // fewer when a bar filters the list
    const sum = get("summary");

    const bar = el("div", { className: "ba-bar" });
    const jump = el("input", { type: "number", min: 0, max: nBits - 1, placeholder: "bit" });
    const go = () => {
      const v = parseInt(jump.value, 10);
      const where = get("order").indexOf(v);
      if (where >= 0) set({ focus: v, page: Math.floor(where / size) });
    };
    jump.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
    const prev = el("button", { className: "ba-btn", textContent: "‹ prev", disabled: page <= 0 });
    const next = el("button", { className: "ba-btn", textContent: "next ›", disabled: page >= pages - 1 });
    const first = el("button", { className: "ba-btn", textContent: "« first", disabled: page <= 0 });
    const last = el("button", { className: "ba-btn", textContent: "last »", disabled: page >= pages - 1 });
    first.addEventListener("click", () => set({ page: 0 }));
    last.addEventListener("click", () => set({ page: pages - 1 }));
    prev.addEventListener("click", () => set({ page: page - 1 }));
    next.addEventListener("click", () => set({ page: page + 1 }));
    // two rows, like MolGrid's toolbar: the fingerprint | the sort; go to bit | the picture key.
    // The pager sits under the list.
    const sortSel = el("select");
    for (const [v, label] of [["bit", "sort: bit index"], ["envs", "sort: most substructures"], ["mols", "sort: most molecules"], ["purity", "sort: purity"]])
      sortSel.append(new Option(label, v, false, v === get("sort")));
    sortSel.addEventListener("change", () => set({ sort: sortSel.value }));
    // the arrow is the direction of the sorted value: bit index starts ascending, the counts
    // descending; the button reverses it
    const up = (get("sort") === "bit") !== get("reverse");
    const dirBtn = el("button", { className: "ba-btn", textContent: up ? "↑" : "↓", title: "sort direction" });
    dirBtn.addEventListener("click", () => set({ reverse: !get("reverse") }));
    const right = (...nodes) => { const d = el("div", { className: "ba-right" }); d.append(...nodes); return d; };
    bar.append(
      el("span", { className: "ba-lbl", textContent: "radius" }),
      seg([[1, "1"], [2, "2"], [3, "3"]], get("radius"), (v) => set({ radius: v, count_filter: [] })),
      el("span", { className: "ba-lbl", textContent: "fold to" }),
      seg([[256, "256"], [1024, "1024"], [2048, "2048"], [8192, "8192"]], nBits, (v) => set({ n_bits: v, count_filter: [] })),
      right(sortSel, dirBtn),
    );
    root.appendChild(bar);
    const second = el("div", { className: "ba-bar" });
    second.append(
      el("span", { className: "ba-lbl", textContent: "go to bit" }), jump,
      right(el("div", { className: "ba-legend" }, legendHtml("Pictures", MORGAN_ENV_KEY))),
    );
    root.appendChild(second);

    const list = el("div", { className: "ba-list" });
    for (const r of get("rows")) {
      const row = el("div", { className: "ba-row" + (r.bit === get("focus") ? " focus" : "") });
      const meta = el("div", { className: "ba-meta" },
        `<b>bit ${r.bit}</b>` +
        `<div class="ba-num">${r.n_envs} substructure${r.n_envs === 1 ? "" : "s"}</div>` +
        `<div class="ba-meter"><i style="width:${(100 * r.n_envs) / Math.max(sum.max_envs || 1, 1)}%"></i></div>` +
        `<div class="ba-num">${r.n_mols.toLocaleString()} mols</div>` +
        `<div class="ba-meter"><i style="width:${(100 * r.n_mols) / Math.max(sum.max_mols || 1, 1)}%"></i></div>` +
        (r.purity != null
          ? `<div class="ba-num" title="share of the bit's molecules that set it through its most common substructure">purity ${Math.round(r.purity * 100)}%</div>` +
            `<div class="ba-meter"><i style="width:${100 * r.purity}%"></i></div>`
          : ""));
      const strip = el("div", { className: "ba-strip" });
      if (!r.envs.length) strip.appendChild(el("div", { className: "ba-empty", textContent: "no substructure in this dataset" }));
      for (const e of r.envs)
        strip.appendChild(el("div", { className: "ba-tile", title: `${e.env}\ne.g. ${e.id}` },
          `<div class="ba-thead"><span>r${e.radius}</span><span>${e.count} mols</span></div>${e.svg}` +
          `<div class="ba-tenv">${esc(e.env)}</div>`));
      if (r.n_envs > r.envs.length) strip.appendChild(el("div", { className: "ba-more", textContent: `+${r.n_envs - r.envs.length} more` }));
      row.append(meta, strip);
      list.appendChild(row);
    }
    // the list with the distribution of substructures per bit to its right
    const body = el("div", { className: "ba-body" });
    body.append(list, histogram());
    root.appendChild(body);
    const nav = el("div", { className: "ba-bar ba-pager" });
    nav.append(first, prev, el("span", { className: "ba-lbl", textContent: `page ${page + 1} / ${pages}` }), next, last);
    root.appendChild(nav);
    // after a jump, bring the focused bit into view inside the scrolling list
    const hit = list.querySelector(".ba-row.focus");
    if (hit) list.scrollTop = hit.offsetTop;
  }

  // How crowded the bits are: for each number of substructures on a bit (binned when the range
  // is wide), how many bits hold that many. A sparse fingerprint piles up at 0–1; collisions move
  // the pile down.
  function histogram() {
    const per = get("summary").envs_per_bit || [];
    const max = Math.max(0, ...per);
    const step = Math.max(1, Math.ceil((max + 1) / 16));
    const bins = [];
    for (let lo = 0; lo <= max; lo += step) bins.push({ lo, hi: Math.min(max, lo + step - 1), n: 0 });
    for (const v of per) bins[Math.floor(v / step)].n++;
    const top = Math.max(1, ...bins.map((b) => b.n));
    const sum = get("summary"), cf = get("count_filter") || [];
    // the dataset's totals first, then how they spread over the bits
    const box = el("div", { className: "ba-hist" });
    if (sum.n_mols != null) box.appendChild(el("div", { className: "h", textContent: "Summary" }));
    if (sum.n_mols != null)
      box.appendChild(el("div", { className: "s" },
        `<span>molecules</span><b>${sum.n_mols.toLocaleString()}</b>` +
        `<span>substructures</span><b>${sum.n_envs.toLocaleString()}</b>` +
        `<span>bits used</span><b>${sum.used_bits.toLocaleString()} / ${per.length.toLocaleString()}</b>` +
        `<span>mean per bit</span><b>${(sum.n_envs / Math.max(sum.used_bits, 1)).toFixed(1)}</b>` +
        (sum.mean_purity != null
          ? `<span title="how often a set bit comes from its bit's most common substructure">mean purity</span><b>${Math.round(sum.mean_purity * 100)}%</b>`
          : "")));
    box.insertAdjacentHTML("beforeend", `<div class="h">Substructures per bit</div>` +
      `<div class="r head"><span class="k">per bit</span><span></span><span class="n">bits</span></div>`);
    for (const b of bins) {
      const name = b.lo === b.hi ? `${b.lo}` : `${b.lo}–${b.hi}`;
      const on = cf.length === 2 && cf[0] === b.lo && cf[1] === b.hi;
      const row = el("div", { className: "r" + (on ? " on" : ""), title: `${b.n.toLocaleString()} bits hold ${name} substructures` },
        `<span class="k">${name}</span><span class="t"><i style="width:${(100 * b.n) / top}%"></i></span>` +
        `<span class="n">${b.n.toLocaleString()}</span>`);
      // a click lists only these bits; a second click lists all again
      if (b.n) row.addEventListener("click", () => set({ count_filter: on ? [] : [b.lo, b.hi] }));
      else row.classList.add("none");
      box.appendChild(row);
    }
    box.appendChild(el("div", { className: "hint", textContent: cf.length === 2 ? "click the bar again to list every bit" : "click a bar to list only those bits" }));
    return box;
  }

  draw();
  for (const k of ["rows", "summary", "order", "page", "radius", "n_bits", "sort", "reverse", "count_filter", "focus"]) model.on(`change:${k}`, draw);
}

export default { render };
