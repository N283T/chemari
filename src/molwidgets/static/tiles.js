// MorganBitTiles: a molecule's Morgan fingerprint as a grid of bit pictures (RDKit DrawMorganEnv).
// Pictures are rendered in Python; this module only lays them out and handles selection.

const CSS = `
.bt-root { font: 13px/1.45 system-ui, sans-serif; color: var(--bt-fg); --bt-fg:#1f2328; --bt-muted:#6b7280;
  --bt-border:#d0d7de; --bt-card:#fff; --bt-soft:#f6f8fa; --bt-hit:#d6336c; --bt-accent:#1c7ed6; }
.bt-root.dark { --bt-fg:#e6e6e6; --bt-muted:#9aa4b2; --bt-border:#3a3f47; --bt-card:#1c1f24; --bt-soft:#24282e; }
.bt-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }
.bt-seg { display:inline-flex; border:1px solid var(--bt-border); border-radius:6px; overflow:hidden; }
.bt-seg button { font:inherit; color:var(--bt-fg); background:var(--bt-soft); border:0; padding:2px 9px; cursor:pointer; }
.bt-seg button + button { border-left:1px solid var(--bt-border); }
.bt-seg button.on { background:var(--bt-fg); color:var(--bt-card); }
.bt-lbl { color:var(--bt-muted); }
.bt-summary { margin-bottom:8px; }
.bt-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap:8px; }
.bt-tile { background:#fff; color:#1f2328; border:1px solid var(--bt-border); border-radius:8px; padding:4px 6px 6px;
  cursor:pointer; position:relative; }
.bt-tile:hover { box-shadow:0 1px 6px rgba(0,0,0,.15); }
.bt-tile.hit { border:2px solid var(--bt-hit); }
.bt-tile.sel { box-shadow:0 0 0 3px var(--bt-accent); }
.bt-tile svg { width:100%; height:auto; display:block; }
.bt-head { display:flex; justify-content:space-between; font-size:12px; font-weight:700; }
.bt-head span:last-child { font-weight:400; color:#6b7280; }
.bt-env { font:11px ui-monospace,monospace; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bt-badge { display:inline-block; margin-top:3px; border-radius:4px; padding:0 5px; font-size:10.5px; font-weight:600; }
.bt-badge.warn { background:#fff0f6; color:#c2255c; }
.bt-badge.ok { background:#ebfbee; color:#2b8a3e; }
.bt-flag { font-size:10.5px; color:var(--bt-hit); font-weight:600; }
.bt-gallery { margin-top:14px; }
.bt-gallery h4 { margin:0 0 6px; font-size:13px; }
.bt-tile.mine { border:2px solid var(--bt-accent); }
.bt-muted { color:var(--bt-muted); }
.bt-hint { color:var(--bt-muted); font-size:11.5px; margin-top:6px; }
`;

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

function seg(options, current, onPick) {
  const wrap = el("div", { className: "bt-seg" });
  for (const [val, label] of options) {
    const b = el("button", { textContent: label });
    if (val === current) b.classList.add("on");
    b.addEventListener("click", () => onPick(val));
    wrap.appendChild(b);
  }
  return wrap;
}

const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;");

function render({ model, el: host }) {
  const root = el("div", { className: "bt-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  const get = (k) => model.get(k);
  const set = (obj) => { for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };

  function draw() {
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const tiles = get("tiles");
    const sel = get("selected");
    const hasRef = tiles.some((t) => t.n_envs != null);

    const bar = el("div", { className: "bt-bar" });
    if (get("label")) bar.appendChild(el("b", { textContent: get("label") }));
    bar.append(
      el("span", { className: "bt-lbl", textContent: "radius" }),
      seg([[1, "1"], [2, "2"], [3, "3"]], get("radius"), (v) => set({ radius: v })),
      el("span", { className: "bt-lbl", textContent: "fold to" }),
      seg([[256, "256"], [1024, "1024"], [2048, "2048"], [8192, "8192"]], get("n_bits"), (v) => set({ n_bits: v })),
      el("span", { className: "bt-lbl", textContent: "bits" }),
    );
    root.appendChild(bar);

    const bits = new Set(tiles.map((t) => t.bit));
    const hits = new Set(tiles.filter((t) => t.collides).map((t) => t.bit));
    root.appendChild(el("div", { className: "bt-summary" },
      `<b>${tiles.length}</b> distinct environments → <b>${bits.size}</b> bits on` +
      (hits.size ? ` · <span style="color:var(--bt-hit);font-weight:600">${hits.size} bit${hits.size > 1 ? "s hold" : " holds"} two environments of this molecule (red frames)</span>` : " · no collisions inside this molecule")));

    const grid = el("div", { className: "bt-grid" });
    for (const t of tiles) {
      const card = el("div", { className: "bt-tile" + (t.collides ? " hit" : "") + (t.bit === sel ? " sel" : "") });
      let badge = "";
      if (t.n_envs != null) {
        const others = t.n_envs - 1;
        badge = others > 0
          ? `<span class="bt-badge warn">shared with ${others} other substructure${others > 1 ? "s" : ""}</span>`
          : `<span class="bt-badge ok">unique in dataset</span>`;
      }
      card.innerHTML =
        `<div class="bt-head"><span>bit ${t.bit}</span><span>r${t.radius}${t.count > 1 ? ` · ×${t.count}` : ""}</span></div>` +
        t.svg +
        `<div class="bt-env" title="${esc(t.env)}">${esc(t.env)}</div>` +
        (t.collides ? `<div class="bt-flag">⚠ same bit as another tile</div>` : "") + badge;
      if (hasRef) card.addEventListener("click", () => set({ selected: t.bit === sel ? -1 : t.bit }));
      grid.appendChild(card);
    }
    root.appendChild(grid);
    root.appendChild(el("div", { className: "bt-hint" },
      "Drawn with RDKit's DrawMorganEnv: blue = centre atom, yellow = aromatic atoms, grey = aliphatic ring atoms, " +
      "light grey = neighbours outside the environment (shown only to mark where it attaches). " +
      "×n = the same identifier from n atoms with identical environments." + (hasRef ? " Click a tile to see everything else that sets its bit." : "")));

    const gal = get("gallery");
    if (sel >= 0 && gal.length) {
      const box = el("div", { className: "bt-gallery" });
      const n = tiles.find((t) => t.bit === sel)?.n_envs ?? gal.length;
      box.appendChild(el("h4", {},
        `Bit ${sel} across the reference set: ${n} different environment${n > 1 ? "s" : ""}` +
        (gal.length < n ? ` <span class="bt-muted" style="font-weight:400">(first ${gal.length} shown)</span>` : "") +
        ` <span class="bt-muted" style="font-weight:400">· blue frame = from this molecule</span>`));
      const g = el("div", { className: "bt-grid" });
      for (const e of gal) {
        const c = el("div", { className: "bt-tile" + (e.mine ? " mine" : "") });
        c.style.cursor = "default";
        c.innerHTML = `<div class="bt-head"><span>r${e.radius}</span><span>in ${e.count} mols</span></div>` + e.svg +
          `<div class="bt-env" title="${esc(e.env)}">${esc(e.env)}</div><div class="bt-muted" style="font-size:10.5px">e.g. ${esc(e.id)}</div>`;
        g.appendChild(c);
      }
      box.appendChild(g);
      root.appendChild(box);
    }
  }

  draw();
  for (const k of ["tiles", "selected", "gallery", "radius", "n_bits", "label"]) model.on(`change:${k}`, draw);
}

export default { render };
