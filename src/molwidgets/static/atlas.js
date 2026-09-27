// BitAtlas: every folded bit of a reference set, one row per bit, with the distinct substructures
// that fold onto it (RDKit DrawMorganEnv pictures rendered in Python for the visible page).
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
.ba-stats { color:var(--ba-muted); margin:-2px 0 4px; }
.ba-legend { color:var(--ba-muted); font-size:11.5px; margin:0 0 8px; line-height:1.9; }
.ba-bar input { font:inherit; width:70px; padding:2px 6px; border:1px solid var(--ba-border); border-radius:6px;
  background:var(--ba-card); color:var(--ba-fg); }
.ba-bar .ba-btn { font:inherit; color:var(--ba-fg); background:var(--ba-soft); border:1px solid var(--ba-border);
  border-radius:6px; padding:2px 9px; cursor:pointer; }
.ba-bar .ba-btn:disabled { opacity:.4; cursor:default; }
.ba-list { border:1px solid var(--ba-border); border-radius:8px; max-height:520px; overflow-y:auto; position:relative; }
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
  const set = (obj) => { for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };

  function draw() {
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const nBits = get("n_bits"), size = get("page_size"), page = get("page");
    const pages = Math.ceil(nBits / size);
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
    bar.append(
      el("span", { className: "ba-lbl", textContent: "radius" }),
      seg([[1, "1"], [2, "2"], [3, "3"]], get("radius"), (v) => set({ radius: v })),
      el("span", { className: "ba-lbl", textContent: "fold to" }),
      seg([[256, "256"], [1024, "1024"], [2048, "2048"], [8192, "8192"]], nBits, (v) => set({ n_bits: v })),
      el("span", { className: "ba-lbl", textContent: "sort" }),
      seg([["bit", "bit index"], ["envs", "most substructures"], ["mols", "most molecules"]], get("sort"),
        (v) => set({ sort: v })),
    );
    root.appendChild(bar);
    const nav = el("div", { className: "ba-bar" });
    nav.append(
      first, prev, el("span", { className: "ba-lbl", textContent: `page ${page + 1} / ${pages}` }), next, last,
      el("span", { className: "ba-lbl", style: "margin-left:12px", textContent: "go to bit" }), jump,
    );
    root.appendChild(nav);
    if (sum.n_mols != null)
      root.appendChild(el("div", { className: "ba-stats" },
        `${sum.n_mols.toLocaleString()} molecules · ${sum.n_envs.toLocaleString()} distinct substructures → ` +
        `${sum.used_bits.toLocaleString()} / ${nBits.toLocaleString()} bits used · ` +
        `${(sum.n_envs / Math.max(sum.used_bits, 1)).toFixed(1)} substructures per bit on average`));
    root.appendChild(el("div", { className: "ba-legend" }, legendHtml("Pictures", MORGAN_ENV_KEY)));

    const list = el("div", { className: "ba-list" });
    for (const r of get("rows")) {
      const row = el("div", { className: "ba-row" + (r.bit === get("focus") ? " focus" : "") });
      const meta = el("div", { className: "ba-meta" },
        `<b>bit ${r.bit}</b>` +
        `<div class="ba-num">${r.n_envs} substructure${r.n_envs === 1 ? "" : "s"}</div>` +
        `<div class="ba-meter"><i style="width:${(100 * r.n_envs) / Math.max(sum.max_envs || 1, 1)}%"></i></div>` +
        `<div class="ba-num">${r.n_mols.toLocaleString()} mols</div>` +
        `<div class="ba-meter"><i style="width:${(100 * r.n_mols) / Math.max(sum.max_mols || 1, 1)}%"></i></div>`);
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
    root.appendChild(list);
    // after a jump, bring the focused bit into view inside the scrolling list
    const hit = list.querySelector(".ba-row.focus");
    if (hit) list.scrollTop = hit.offsetTop;
  }

  draw();
  for (const k of ["rows", "summary", "page", "radius", "n_bits", "sort", "focus"]) model.on(`change:${k}`, draw);
}

export default { render };
