// MorganBitTiles: the molecule on the left, its Morgan bits on the right. Each bit row carries
// RDKit's DrawMorganEnv picture (rendered in Python); hovering or clicking a row highlights that
// environment in the full molecule (drawn here with RDKit.js).
// Depends on loadRDKit / drawSvg / isDark (prepended by the Python side).

const CSS = `
.bt-root { font: 13px/1.45 system-ui, sans-serif; color: var(--bt-fg); --bt-fg:#1f2328; --bt-muted:#6b7280;
  --bt-border:#d0d7de; --bt-card:#fff; --bt-soft:#f6f8fa; --bt-hit:#d6336c; --bt-accent:#1c7ed6; --bt-sel:#e7f5ff; }
.bt-root.dark { --bt-fg:#e6e6e6; --bt-muted:#9aa4b2; --bt-border:#3a3f47; --bt-card:#1c1f24; --bt-soft:#24282e; --bt-sel:#1b2a3a; }
.bt-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }
.bt-seg { display:inline-flex; border:1px solid var(--bt-border); border-radius:6px; overflow:hidden; }
.bt-seg button { font:inherit; color:var(--bt-fg); background:var(--bt-soft); border:0; padding:2px 9px; cursor:pointer; }
.bt-seg button + button { border-left:1px solid var(--bt-border); }
.bt-seg button.on { background:var(--bt-fg); color:var(--bt-card); }
.bt-lbl { color:var(--bt-muted); }
.bt-stats { color:var(--bt-muted); margin:-2px 0 6px; }
/* both columns share the row height: the list is capped, the molecule box stretches to match */
.bt-main { display:grid; grid-template-columns: minmax(260px, 1fr) minmax(300px, 1.15fr); gap:12px; align-items:stretch; }
@media (max-width: 760px) { .bt-main { grid-template-columns: 1fr; } }
.bt-mol { display:flex; flex-direction:column; justify-content:space-between; background:var(--bt-card); border:1px solid var(--bt-border); border-radius:8px; padding:6px 8px; min-height:0; }
.bt-mol svg { width:100%; height:auto; max-height:460px; display:block; margin:auto 0; }
.bt-cap { color:var(--bt-muted); font-size:12px; min-height:2.6em; }
.bt-cap code { font-family:ui-monospace,monospace; color:var(--bt-fg); }
.bt-list { max-height:520px; overflow:auto; border:1px solid var(--bt-border); border-radius:8px; }
.bt-row { display:grid; grid-template-columns: 96px 1fr; gap:8px; align-items:center; padding:4px 8px;
  border-bottom:1px solid var(--bt-border); cursor:pointer; }
.bt-row:last-child { border-bottom:0; }
.bt-row:hover { background:var(--bt-soft); }
.bt-row.sel { background:var(--bt-sel); }
.bt-row.hit { box-shadow: inset 4px 0 0 var(--bt-hit); }
.bt-thumb { background:#fff; border:1px solid var(--bt-border); border-radius:6px; overflow:hidden; }
.bt-thumb svg { width:100%; height:auto; display:block; }
.bt-info b { font-size:13px; }
.bt-env { font:11.5px ui-monospace,monospace; color:var(--bt-muted); word-break:break-all; }
.bt-badge { display:inline-block; margin:2px 4px 0 0; border-radius:4px; padding:0 5px; font-size:10.5px; font-weight:600; }
.bt-badge.warn { background:#f1f3f5; color:#495057; }
.bt-root.dark .bt-badge.warn { background:#2c3036; color:#c1c7cf; }
.bt-badge.ok { background:#ebfbee; color:#2b8a3e; }
.bt-badge.hit { background:var(--bt-hit); color:#fff; }
.bt-val { float:right; font:600 12px ui-monospace,monospace; border:1px solid var(--bt-border); border-radius:4px; padding:0 6px; }
.bt-val.many { background:var(--bt-fg); color:var(--bt-card); border-color:var(--bt-fg); }
.bt-gallery { margin-top:12px; }
.bt-gallery h4 { margin:0 0 6px; font-size:13px; }
.bt-empty { color:var(--bt-muted); border:1px dashed var(--bt-border); border-radius:8px; padding:18px; text-align:center; }
.bt-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap:8px; }
.bt-tile { background:#fff; color:#1f2328; border:1px solid var(--bt-border); border-radius:8px; padding:4px 6px 6px; }
.bt-tile.mine { border:2px solid var(--bt-accent); }
.bt-tile svg { width:100%; height:auto; display:block; }
.bt-head { display:flex; justify-content:space-between; font-size:11px; color:#6b7280; }
.bt-tenv { font:11px ui-monospace,monospace; color:#6b7280; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bt-muted { color:var(--bt-muted); }
.bt-hint { color:var(--bt-muted); font-size:11.5px; margin-top:6px; }
.bt-legend { color:var(--bt-muted); font-size:11.5px; margin:-2px 0 8px; line-height:1.9; }
`;

const CENTER = [0.55, 0.62, 0.95];
const ENV = [1.0, 0.78, 0.35];

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

function highlight(tiles) {
  if (!tiles.length) return { atoms: [], bonds: [], stable: true };
  const atoms = new Set(), bonds = new Set(), atomColors = {}, bondColors = {};
  for (const t of tiles)
    for (const w of t.where) {
      w.atoms.forEach((a) => { atoms.add(a); if (!(a in atomColors)) atomColors[a] = ENV; });
      w.bonds.forEach((b) => { bonds.add(b); bondColors[b] = ENV; });
    }
  for (const t of tiles) for (const w of t.where) atomColors[w.center] = CENTER;
  return { atoms: [...atoms], bonds: [...bonds], atomColors, bondColors, stable: true };
}

async function render({ model, el: host }) {
  const root = el("div", { className: "bt-root" });
  enableSmilesCopy(root, model);
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);
  root.appendChild(el("div", { className: "bt-muted" }, "loading RDKit.js…"));
  const RDKit = await loadRDKit();
  const get = (k) => model.get(k);
  const busy = busyIndicator(model, host, ["radius", "n_bits", "chirality", "selected"]);
  const set = (obj) => { busy(obj); for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };
  let hover = null; // uid under the mouse
  let onlyHits = false; // list filter: only rows that collide inside the molecule

  function active() {
    const tiles = get("tiles");
    if (hover != null) return tiles.filter((t) => t.uid === hover);
    const sel = get("selected");
    return sel >= 0 ? tiles.filter((t) => t.bit === sel) : [];
  }

  function drawMol() {
    const box = root.querySelector(".bt-mol");
    if (!box) return;
    const dark = root.classList.contains("dark");
    const act = active();
    box.innerHTML = drawSvg(RDKit, get("smiles"), 400, 340, highlight(act), dark) + smilesCopyHtml(get("smiles"));
    const cap = el("div", { className: "bt-cap" });
    if (!act.length) cap.innerHTML = "Hover or click a bit on the right to see where it comes from.";
    else {
      const envs = act.map((t) => `<code>${esc(t.env)}</code>${t.count > 1 ? ` ×${t.count}` : ""}`).join(" and ");
      cap.innerHTML = `<b>bit ${act[0].bit}</b> · ${envs}` +
        (act.length > 1 ? ` <span style="color:var(--bt-hit);font-weight:600">— two different environments, one bit</span>` : "");
    }
    box.appendChild(cap);
  }

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
    const chiral = el("label", { className: "mw-chiral" + (get("chirality") ? " on" : "") });
    const box = el("input", { type: "checkbox", checked: get("chirality") });
    box.addEventListener("change", () => set({ chirality: box.checked, selected: -1 }));
    chiral.append(box, "chirality");
    bar.append(chiral);
    // the value each bit holds: 1 in a bit fingerprint; in a count fingerprint the number of
    // times its environments occur, summed over every environment folded onto it
    const count = get("mode") === "count";
    const value = {};
    for (const t of tiles) value[t.bit] = (value[t.bit] || 0) + t.count;
    bar.append(
      el("span", { className: "bt-lbl", textContent: "fingerprint" }),
      seg([["bit", "bit"], ["count", "count"]], get("mode"), (v) => set({ mode: v })),
    );
    const bits = new Set(tiles.map((t) => t.bit));
    const hits = new Set(tiles.filter((t) => t.collides).map((t) => t.bit));
    bar.append(
      el("span", { className: "bt-lbl", textContent: "show" }),
      seg([[false, "all bits"], [true, "collisions only"]], onlyHits, (v) => {
        onlyHits = v;
        // keep the selection inside the visible list
        if (v && hits.size && !hits.has(sel)) set({ selected: [...hits][0] });
        else draw();
      }),
    );
    root.appendChild(bar);
    root.appendChild(el("div", { className: "bt-stats" },
      `${tiles.length} environments → ${bits.size} bits` +
      (count ? ` · values add up to ${Object.values(value).reduce((a, b) => a + b, 0)} (bit fingerprint: ${bits.size} ones)` : "") +
      (hits.size ? ` · <span style="color:var(--bt-hit);font-weight:600">${hits.size} collision${hits.size > 1 ? "s" : ""} inside this molecule</span>` : " · no collisions inside this molecule")));
    root.appendChild(el("div", { className: "bt-legend" },
      legendHtml("Molecule", [[CENTER, "centre atom"], [ENV, "rest of the environment"]]) + "<br>" +
      legendHtml("Bit cards", MORGAN_ENV_KEY) +
      `<span style="white-space:nowrap"><i style="display:inline-block;width:4px;height:12px;background:var(--bt-hit);vertical-align:-2px;margin-right:4px"></i>red edge: shares a bit with another row</span>`));

    const main = el("div", { className: "bt-main" });
    main.appendChild(el("div", { className: "bt-mol mw-copyable" }));
    const list = el("div", { className: "bt-list" });
    const shown = onlyHits ? tiles.filter((t) => t.collides) : tiles;
    if (!shown.length)
      list.appendChild(el("div", { className: "bt-empty", style: "border:0" },
        `No collisions inside this molecule at ${get("n_bits")} bits.`));
    for (const t of shown) {
      const row = el("div", { className: "bt-row" + (t.bit === sel ? " sel" : "") + (t.collides ? " hit" : "") });
      let badges = "";
      if (t.collides) badges += `<span class="bt-badge hit">same bit as another row</span>`;
      if (t.n_envs != null) {
        const others = t.n_envs - 1;
        badges += others > 0
          ? `<span class="bt-badge warn">shared with ${others} other substructure${others > 1 ? "s" : ""}</span>`
          : `<span class="bt-badge ok">unique in dataset</span>`;
      }
      const v = count ? value[t.bit] : 1;
      const why = count && t.collides ? ` title="${t.count} here + ${v - t.count} from the other row(s) on this bit"` : "";
      row.innerHTML = `<div class="bt-thumb">${t.svg}</div>` +
        `<div class="bt-info"><span class="bt-val${v > 1 ? " many" : ""}"${why}>${v}</span><b>bit ${t.bit}</b> <span class="bt-muted">r${t.radius}${t.count > 1 ? ` · ×${t.count} atoms` : ""}</span>` +
        `<div class="bt-env">${esc(t.env)}</div>${badges}</div>`;
      row.addEventListener("mouseenter", () => { hover = t.uid; drawMol(); });
      row.addEventListener("mouseleave", () => { hover = null; drawMol(); });
      row.addEventListener("click", () => { hover = null; set({ selected: t.bit === sel ? -1 : t.bit }); });
      list.appendChild(row);
    }
    main.appendChild(list);
    root.appendChild(main);
    root.appendChild(el("div", { className: "bt-hint" },
      "Each row is one distinct environment, drawn with RDKit's DrawMorganEnv. Hover a row to light it up in the molecule." +
      (count ? " The box on the right is the bit's value in a count fingerprint: how often its environments occur, collisions added together." : " The box on the right is the bit's value: 1 however often the environment occurs.") +
      (hasRef ? " Click it to see every other substructure in the dataset that sets the same bit." : "")));
    drawMol();

    const gal = get("gallery");
    if (sel >= 0 && gal.length) {
      const box = el("div", { className: "bt-gallery" });
      const n = tiles.find((t) => t.bit === sel)?.n_envs ?? gal.length;
      box.appendChild(el("h4", {},
        `Bit ${sel} across the dataset: ${n} different environment${n > 1 ? "s" : ""}` +
        (gal.length < n ? ` <span class="bt-muted" style="font-weight:400">(first ${gal.length} shown)</span>` : "") +
        ` <span class="bt-muted" style="font-weight:400">· blue frame = from this molecule</span>`));
      const g = el("div", { className: "bt-grid" });
      for (const e of gal) {
        const c = el("div", { className: "bt-tile" + (e.mine ? " mine" : "") });
        c.innerHTML = `<div class="bt-head"><span>r${e.radius}</span><span>in ${e.count} mols</span></div>` + e.svg +
          `<div class="bt-tenv" title="${esc(e.env)}">${esc(e.env)}</div><div class="bt-head"><span>e.g. ${esc(e.id)}</span></div>`;
        g.appendChild(c);
      }
      box.appendChild(g);
      root.appendChild(box);
    } else if (hasRef) {
      root.appendChild(el("div", { className: "bt-gallery bt-empty" },
        "Click a bit on the right to see every substructure in the dataset that sets it."));
    }
  }

  draw();
  for (const k of ["tiles", "selected", "gallery", "radius", "n_bits", "label", "smiles", "mode", "chirality"]) model.on(`change:${k}`, draw);
}

export default { render };
