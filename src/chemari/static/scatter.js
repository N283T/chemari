// MolScatter: a scatter plot of molecules with a card panel beside it. Hovering a point shows
// that molecule and its values in the panel (and, when the rows carry one, the id of a partner
// molecule such as its nearest neighbour, whose drawing pops up on hover); clicking pins it, and
// the panel then stays on it until it is unpinned: the point again, an empty spot in the plot, or
// the panel's button. Points can be coloured by a column (a ramp for numbers, one colour per
// value otherwise); a boolean column draws its points as triangles. Plot and panel share one
// height. `selected` (an id) syncs both ways.
// Depends on loadRDKit / drawSvg / isDark / smilesCopyHtml / enableSmilesCopy (prepended).

const RAMP = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]; // viridis stops
const CATS = ["#1c7ed6", "#f08c00", "#2f9e44", "#ae3ec9", "#e03131", "#868e96"]; // categorical colours
const NS = "http://www.w3.org/2000/svg";

function lerpColor(t) {
  t = Math.max(0, Math.min(1, t));
  const pos = t * (RAMP.length - 1);
  const i = Math.min(Math.floor(pos), RAMP.length - 2);
  const f = pos - i;
  const a = RAMP[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = RAMP[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function fmt(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(2);
  return String(v);
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

// about eight round tick values covering [lo, hi]
function ticks(lo, hi) {
  const raw = (hi - lo) / 8;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function el(tag, attrs = {}) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

// a point: a circle, or an upward triangle of about the same area for flagged rows
function mark(cx, cy, r, triangle, attrs) {
  if (!triangle) return el("circle", { cx, cy, r, ...attrs });
  const s = r * 1.55;
  const d = `M${cx},${cy - s} L${cx + s * 0.95},${cy + s * 0.65} L${cx - s * 0.95},${cy + s * 0.65} Z`;
  return el("path", { d, ...attrs });
}

const CSS = `
.ms-root { font: 13px/1.35 system-ui, sans-serif; color: var(--ms-fg); --ms-fg: #1f2328; --ms-muted: #6b7280;
  --ms-border: #d0d7de; --ms-card: #ffffff; --ms-soft: #f6f8fa; --ms-grid: #eef0f2;
  display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 14px; }
.ms-root.dark { --ms-fg: #e6e6e6; --ms-muted: #9aa4b2; --ms-border: #3a3f47; --ms-card: #1c1f24; --ms-soft: #24282e;
  --ms-grid: #2b3038; }
.ms-plot { position: relative; display: flex; flex-direction: column; min-width: 0; }
.ms-legend { display: flex; flex-wrap: wrap; gap: 4px 16px; align-items: center; color: var(--ms-muted);
  font-size: 11.5px; height: 20px; }
.ms-legend .ms-ramp { display: inline-block; width: 90px; height: 8px; border-radius: 4px; vertical-align: -0.5px;
  margin: 0 5px; background: linear-gradient(90deg, ${RAMP.join(",")}); }
.ms-legend span { display: inline-flex; align-items: center; white-space: nowrap; }
.ms-legend .ms-key { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; }
.ms-legend svg { display: inline-block; margin-right: 4px; }
.ms-canvas { flex: 1; min-height: 0; }
.ms-canvas svg { display: block; }
.ms-canvas svg text { fill: var(--ms-muted); font-size: 11px; }
.ms-canvas svg .ms-title { fill: var(--ms-fg); font-weight: 600; }
.ms-canvas svg .ms-pt { cursor: pointer; }
/* the panel: one card in two sections divided by a rule — the hovered / pinned molecule (the
   height left over) and its values (the height they need) */
.ms-panel { position: relative; display: flex; flex-direction: column; min-height: 0; overflow: hidden;
  background: var(--ms-card); border: 1px solid var(--ms-border); border-radius: 10px; }
.ms-sec { padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; min-height: 0; }
.ms-sec + .ms-sec { border-top: 1px solid var(--ms-border); }
.ms-sec.mol { flex: 1 1 0; }
.ms-sec.info { flex: 0 0 auto; gap: 12px; padding-bottom: 14px; }
/* section heading: a small upper-case title, with a chip on the right */
.ms-sh { display: flex; align-items: center; justify-content: space-between; gap: 8px; min-height: 20px; }
.ms-sh .t { font-size: 10.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--ms-muted); }
.ms-chip { font: inherit; font-size: 11px; font-weight: 600; line-height: 18px; padding: 0 8px; border-radius: 999px;
  white-space: nowrap; border: 1px solid transparent; }
.ms-chip.pin { color: #1864ab; background: rgba(28,126,214,.1); border-color: rgba(28,126,214,.3); cursor: pointer; }
.ms-chip.pin:hover { background: rgba(28,126,214,.18); }
.ms-chip.delta { color: #c92a2a; background: rgba(224,49,49,.08); border-color: rgba(224,49,49,.3);
  font-variant-numeric: tabular-nums; }
.ms-root.dark .ms-chip.pin { color: #74c0fc; }
.ms-root.dark .ms-chip.delta { color: #ff8787; }
.ms-idrow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.ms-id { font-weight: 700; font-size: 15px; letter-spacing: .01em; }
.ms-flag { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; font-weight: 600; line-height: 18px;
  padding: 0 8px; border-radius: 999px; color: #c92a2a; background: rgba(224,49,49,.08); border: 1px solid rgba(224,49,49,.3); }
.ms-root.dark .ms-flag { color: #ff8787; }
/* the drawing sits on a tinted plate and is rendered at the plate's pixel size (see pic()) */
.ms-pic { position: relative; flex: 1 1 0; min-height: 60px; min-width: 0; background: var(--ms-soft); border-radius: 8px; }
.ms-pic > svg { position: absolute; inset: 0; }
/* values on one shared axis: a dot per value with its label beside it, ticks below */
.ms-line { position: relative; height: 50px; margin: 0 4px; }
.ms-line .track { position: absolute; left: 0; right: 0; top: 26px; height: 4px; border-radius: 2px; background: var(--ms-grid); }
.ms-line .gap { position: absolute; top: 26px; height: 4px; background: #e03131; opacity: .45; }
.ms-line .dot { position: absolute; top: 28px; width: 12px; height: 12px; transform: translate(-50%, -50%);
  border-radius: 50%; border: 2px solid var(--ms-card); box-shadow: 0 0 0 1px rgba(0,0,0,.18); }
.ms-line .lab { position: absolute; top: 2px; font-size: 11.5px; white-space: nowrap; color: var(--ms-muted); }
.ms-line .lab b { color: var(--ms-fg); font-size: 14px; font-variant-numeric: tabular-nums; margin-left: 3px; }
.ms-line .lab.r { transform: translateX(-100%); }
.ms-line .lab .sep { margin: 0 6px; }
.ms-line .tick { position: absolute; top: 36px; transform: translateX(-50%); font-size: 10px; color: var(--ms-muted); }
/* the rest as label / value pairs in two columns */
.ms-facts { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; padding-top: 10px; border-top: 1px dashed var(--ms-border); }
.ms-facts > div { min-width: 0; }
.ms-facts span { display: block; }
.ms-facts .k { color: var(--ms-muted); font-size: 11px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.ms-facts .v { font-weight: 700; font-size: 14px; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
/* the partner's id: hovering it pops up its drawing, clicking copies its SMILES */
.ms-facts .v.pid { cursor: copy; text-decoration: underline dotted; text-underline-offset: 3px; }
.ms-facts .v.pid.done { color: #1a7f37; text-decoration: none; }
.ms-pop { position: absolute; z-index: 4; display: none; padding: 6px 8px 4px; background: var(--ms-card);
  border: 1px solid var(--ms-border); border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,.18); pointer-events: none; }
.ms-pop .t { display: block; font-size: 11px; color: var(--ms-muted); }
.ms-pop svg { display: block; }
.ms-panel.ms-empty { align-items: center; justify-content: center; gap: 6px; padding: 16px; color: var(--ms-muted);
  text-align: center; background: var(--ms-soft); border-style: dashed; }
@media (max-width: 640px) { .ms-root { grid-template-columns: 1fr; } }
`;

async function render({ model, el: host }) {
  const get = (k) => model.get(k);
  const root = document.createElement("div");
  root.className = "ms-root";
  enableSmilesCopy(root, model);
  const dark = isDark(host);
  if (dark) root.classList.add("dark");
  const style = document.createElement("style");
  style.textContent = CSS;
  const plot = Object.assign(document.createElement("div"), { className: "ms-plot" });
  const legend = Object.assign(document.createElement("div"), { className: "ms-legend" });
  const canvas = Object.assign(document.createElement("div"), { className: "ms-canvas" });
  const panel = Object.assign(document.createElement("div"), { className: "ms-panel" });
  plot.append(legend, canvas);
  root.append(style, plot, panel);
  host.appendChild(root);
  const setHeight = () => { root.style.height = `${get("height")}px`; };
  setHeight();

  const RDKit = await loadRDKit();
  const svgCache = new Map();
  // fill each .ms-pic plate with its molecule drawn at the plate's pixel size; `data-bond` is the
  // bond length in pixels (smaller if the molecule would not fit)
  function fillPics() {
    for (const box of panel.querySelectorAll(".ms-pic")) {
      const w = Math.round(box.clientWidth), h = Math.round(box.clientHeight);
      if (w < 20 || h < 20) continue;
      const bond = Number(box.dataset.bond);
      const k = `${box.dataset.smiles}|${w}|${h}|${bond}`;
      if (!svgCache.has(k))
        svgCache.set(k, drawSvg(RDKit, box.dataset.smiles, w, h, null, dark, { fixedBondLength: bond }));
      box.innerHTML = svgCache.get(k) + smilesCopyHtml(box.dataset.smiles);
    }
  }

  const pointById = new Map();
  let hovered = null;
  let focusLayer = null;
  let fill = () => "#4c78a8";

  function rows() {
    const x = get("x"), y = get("y");
    return get("data").filter((r) => typeof r[x] === "number" && typeof r[y] === "number");
  }

  function drawPlot() {
    const data = rows();
    const xk = get("x"), yk = get("y"), ck = get("color_by"), mk = get("mark_by");
    const W = Math.max(240, canvas.clientWidth), H = Math.max(160, canvas.clientHeight);
    const L = 46, R = 10, T = 6, B = 38;
    const ext = (k) => [Math.min(...data.map((r) => r[k])), Math.max(...data.map((r) => r[k]))];
    let [x0, x1] = ext(xk), [y0, y1] = ext(yk);
    if (get("same_axes")) { x0 = y0 = Math.min(x0, y0); x1 = y1 = Math.max(x1, y1); }
    const pad = (a, b) => { const p = (b - a || 1) * 0.04; return [a - p, b + p]; };
    [x0, x1] = pad(x0, x1); [y0, y1] = pad(y0, y1);
    const sx = (v) => L + ((v - x0) / (x1 - x0)) * (W - L - R);
    const sy = (v) => H - B - ((v - y0) / (y1 - y0)) * (H - T - B);
    // a numeric column is a viridis ramp; any other column gets one colour per value, from
    // `color_map` or the palette in order of first appearance
    let c0 = 0, c1 = 1;
    const numeric = ck && data.some((r) => typeof r[ck] === "number");
    const cats = new Map();
    if (ck && numeric) {
      const cr = get("color_range");
      [c0, c1] = cr && cr.length === 2 ? cr : ext(ck);
    } else if (ck) {
      const given = get("color_map") || {};
      for (const [k, v] of Object.entries(given)) cats.set(k, v);
      for (const r of data) {
        const k = String(r[ck]);
        if (!cats.has(k)) cats.set(k, CATS[cats.size % CATS.length]);
      }
    }
    fill = (r) => {
      if (!ck) return "#4c78a8";
      if (numeric) return typeof r[ck] === "number" ? lerpColor((r[ck] - c0) / (c1 - c0 || 1)) : "#adb5bd";
      return cats.get(String(r[ck])) || "#adb5bd";
    };

    const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const gridCol = getComputedStyle(root).getPropertyValue("--ms-grid").trim() || "#eee";
    for (const v of ticks(x0, x1)) {
      svg.append(el("line", { x1: sx(v), x2: sx(v), y1: T, y2: H - B, stroke: gridCol }));
      const t = el("text", { x: sx(v), y: H - B + 14, "text-anchor": "middle" }); t.textContent = fmt(v); svg.append(t);
    }
    for (const v of ticks(y0, y1)) {
      svg.append(el("line", { x1: L, x2: W - R, y1: sy(v), y2: sy(v), stroke: gridCol }));
      const t = el("text", { x: L - 6, y: sy(v) + 4, "text-anchor": "end" }); t.textContent = fmt(v); svg.append(t);
    }
    const xt = el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "ms-title" });
    xt.textContent = get("x_label") || xk;
    const yt = el("text", { x: 0, y: 0, "text-anchor": "middle", class: "ms-title",
      transform: `translate(12 ${(T + H - B) / 2}) rotate(-90)` });
    yt.textContent = get("y_label") || yk;
    svg.append(xt, yt);
    if (get("diagonal")) {
      const a = Math.max(x0, y0), b = Math.min(x1, y1);
      svg.append(el("line", { x1: sx(a), y1: sy(a), x2: sx(b), y2: sy(b), stroke: "#adb5bd", "stroke-dasharray": "4 4" }));
    }

    pointById.clear();
    const idk = get("id_col");
    const plain = el("g"), flagged = el("g");
    for (const r of data) {
      const id = String(r[idk]);
      const cx = sx(r[xk]), cy = sy(r[yk]), tri = Boolean(mk && r[mk]);
      const p = mark(cx, cy, tri ? 4.2 : 3.6, tri, {
        fill: fill(r), "fill-opacity": tri ? 1 : 0.8, class: "ms-pt",
        ...(tri ? { stroke: dark ? "#e6e6e6" : "#1f2328", "stroke-width": 0.8 } : {}),
      });
      p.dataset.id = id;
      (tri ? flagged : plain).append(p);
      pointById.set(id, { r, cx, cy, tri });
    }
    focusLayer = el("g", { "pointer-events": "none" });
    svg.append(plain, flagged, focusLayer);
    svg.addEventListener("mousemove", (ev) => {
      const t = ev.target.closest(".ms-pt");
      const id = t ? t.dataset.id : null;
      if (id === hovered) return;
      hovered = id;
      if (id && !get("selected")) show();
    });
    svg.addEventListener("mouseleave", () => { hovered = null; if (!get("selected")) show(); });
    // a point pins it (again: unpins); anywhere else in the plot unpins
    svg.addEventListener("click", (ev) => {
      const t = ev.target.closest(".ms-pt");
      const next = t && get("selected") !== t.dataset.id ? t.dataset.id : "";
      if (next === get("selected")) return;
      model.set("selected", next);
      model.save_changes();
    });
    canvas.replaceChildren(svg);

    const parts = [];
    if (ck && numeric) parts.push(`<span>${esc(get("color_label") || ck)} ${fmt(c0)}<i class="ms-ramp"></i>${fmt(c1)}</span>`);
    else if (ck) {
      if (get("color_label")) parts.push(`<span>${esc(get("color_label"))}</span>`);
      for (const [k, c] of cats) parts.push(`<span><i class="ms-key" style="background:${c}"></i>${esc(k)}</span>`);
    }
    if (mk) {
      const tri = `<svg width="11" height="11" viewBox="0 0 11 11"><path d="M5.5,1 L10,9.5 L1,9.5 Z" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>`;
      parts.push(`<span>${tri}${esc(get("mark_label") || mk)}</span>`);
    }
    legend.innerHTML = parts.join("");
  }

  const DOTS = ["#1f2328", "#1c7ed6", "#f08c00", "#2f9e44", "#ae3ec9"];

  // `axis_fields` on a number line spanning all rows' values: a dot per value with its label
  // beside it (pointing away from the others, so labels never cross), the first two joined
  function axisLine(r) {
    const keys = get("axis_fields").filter((k) => typeof r[k] === "number");
    if (!keys.length) return "";
    const all = rows().flatMap((d) => keys.map((k) => d[k])).filter((v) => typeof v === "number");
    const lo = Math.min(...all), hi = Math.max(...all);
    const frac = (v) => (v - lo) / (hi - lo || 1);
    const pct = (v) => `${frac(v) * 100}%`;
    const colour = (i) => DOTS[i % DOTS.length];
    let html = `<div class="ms-line"><div class="track"></div>`;
    for (const t of ticks(lo, hi)) html += `<span class="tick" style="left:${pct(t)}">${fmt(t)}</span>`;
    if (keys.length > 1) {
      const [a, b] = [r[keys[0]], r[keys[1]]].sort((x, y) => x - y);
      html += `<div class="gap" style="left:${pct(a)};width:calc(${pct(b)} - ${pct(a)})"></div>`;
    }
    const order = keys.map((k, i) => [k, i]).sort((p, q) => r[p[0]] - r[q[0]]);
    order.forEach(([k, i], n) => {
      const f = frac(r[k]);
      // the leftmost label reads leftwards from its dot, the others rightwards; near an edge, inwards
      let right = n === 0 && order.length > 1;
      if (f < 0.22) right = false;
      if (f > 0.78) right = true;
      const off = right ? "calc(" + pct(r[k]) + " + 6px)" : "calc(" + pct(r[k]) + " - 6px)";
      html += `<span class="lab${right ? " r" : ""}" style="left:${off}">${esc(k)}<b>${fmt(r[k])}</b></span>`;
    });
    keys.forEach((k, i) => { html += `<span class="dot" title="${esc(k)}" style="left:${pct(r[k])};background:${colour(i)}"></span>`; });
    return html + `</div>`;
  }

  // the pinned point, else the hovered one: enlarged in the plot and drawn in the panel. A pin
  // stays until the same point is clicked again (or another point is clicked).
  function show() {
    const id = get("selected") || hovered;
    const p = pointById.get(id);
    if (focusLayer) {
      focusLayer.replaceChildren();
      if (p) {
        focusLayer.append(mark(p.cx, p.cy, p.tri ? 7 : 6.5, p.tri, {
          fill: fill(p.r), stroke: dark ? "#ffffff" : "#1f2328", "stroke-width": 2,
        }));
      }
    }
    if (!p) {
      panel.className = "ms-panel ms-empty";
      panel.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="6" cy="17" r="2"/><circle cx="12" cy="10" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="17" cy="15" r="2"/></svg><div>${esc(get("empty_text"))}</div>`;
      return;
    }
    panel.className = "ms-panel";
    const r = p.r;
    const rid = String(r[get("id_col")]);
    const smi = r[get("smiles_col")];
    const mk = get("mark_by");
    const tri = `<svg width="9" height="9" viewBox="0 0 10 10"><path d="M5,0.8 L9.4,9 L0.6,9 Z" fill="currentColor"/></svg>`;
    const flag = mk && r[mk] ? `<span class="ms-flag">${tri}${esc(get("mark_label") || mk)}</span>` : "";
    const pin = get("selected") === rid
      ? `<button type="button" class="ms-chip pin ms-unpin" title="Unpin (or click an empty spot in the plot)">Pinned ×</button>`
      : "";
    let html = `<div class="ms-sec mol"><div class="ms-sh"><span class="t">${esc(get("card_title"))}</span>${pin}</div>` +
      `<div class="ms-idrow"><span class="ms-id">${esc(rid)}</span>${flag}</div>` +
      `<div class="ms-pic mw-copyable" data-smiles="${esc(smi)}" data-bond="28"></div></div>`;

    const keys = get("axis_fields").filter((k) => typeof r[k] === "number");
    const delta = keys.length > 1 ? r[keys[1]] - r[keys[0]] : null;
    const deltaChip = delta === null ? ""
      : `<span class="ms-chip delta" title="${esc(keys[1])} − ${esc(keys[0])}">Δ ${delta >= 0 ? "+" : "−"}${fmt(Math.abs(delta))}</span>`;
    const skip = new Set([get("id_col"), get("smiles_col"), get("partner_id_col"), get("partner_smiles_col"), mk, ...get("axis_fields"), ...get("partner_fields")]);
    if (typeof r[get("color_by")] !== "number") skip.add(get("color_by")); // a category: the colour says it
    const fields = get("fields").length ? get("fields") : Object.keys(r).filter((k) => !skip.has(k));
    const facts = fields.map((k) => [k, fmt(r[k]), ""]);
    // the partner: its id (hover for its drawing) and its values
    const ps = get("partner_smiles_col");
    if (ps && r[ps]) {
      const pid = get("partner_id_col") ? String(r[get("partner_id_col")]) : "structure";
      facts.push([get("partner_label") || "partner", pid,
        ` class="v pid" data-smiles="${esc(r[ps])}" data-label="${esc(get("partner_label"))} · ${esc(pid)}"`]);
      for (const k of get("partner_fields")) facts.push([k, fmt(r[k]), ""]);
    }
    const factsHtml = facts.length
      ? `<div class="ms-facts">${facts.map(([k, v, attrs]) => `<div><span class="k">${esc(k)}</span><span${attrs || ' class="v"'}>${esc(v)}</span></div>`).join("")}</div>`
      : "";
    const head = get("info_title") || deltaChip
      ? `<div class="ms-sh"><span class="t">${esc(get("info_title"))}</span>${deltaChip}</div>` : "";
    html += `<div class="ms-sec info">${head}${axisLine(r)}${factsHtml}</div><div class="ms-pop"></div>`;
    panel.innerHTML = html;
    fillPics();
    mergeLabels();
  }

  // when the values' labels on the number line overlap (close values, or both near an edge), show
  // them as one label centred over the dots instead, kept inside the line
  function mergeLabels() {
    const line = panel.querySelector(".ms-line");
    if (!line) return;
    const labs = [...line.querySelectorAll(".lab")];
    const boxes = labs.map((l) => l.getBoundingClientRect());
    const clash = boxes.some((a, i) => boxes.some((b, j) => j > i && a.left < b.right + 4 && b.left < a.right + 4));
    if (!clash) return;
    const lb = line.getBoundingClientRect();
    const centre = boxes.reduce((t, b) => t + (b.left + b.right) / 2, 0) / boxes.length - lb.left;
    const one = document.createElement("span");
    one.className = "lab";
    one.innerHTML = labs.map((l) => l.innerHTML).join('<span class="sep">·</span>');
    labs.forEach((l) => l.remove());
    line.append(one);
    const w = one.getBoundingClientRect().width;
    one.style.left = `${Math.max(0, Math.min(centre - w / 2, lb.width - w))}px`;
  }

  panel.addEventListener("click", async (ev) => {
    if (ev.target.closest(".ms-unpin")) {
      model.set("selected", "");
      model.save_changes();
      return;
    }
    // the partner's id copies its SMILES, like the copy icons on the drawings
    const t = ev.target.closest(".pid");
    if (!t || t.classList.contains("done")) return;
    await copyText(t.dataset.smiles);
    const text = t.textContent;
    t.classList.add("done");
    t.textContent = "SMILES copied ✓";
    setTimeout(() => { t.classList.remove("done"); t.textContent = text; }, 1200);
  });
  panel.addEventListener("mouseover", (ev) => {
    const t = ev.target.closest(".pid");
    const pop = panel.querySelector(".ms-pop");
    if (!t || !pop) return;
    const k = `pop|${t.dataset.smiles}`;
    if (!svgCache.has(k)) svgCache.set(k, drawSvg(RDKit, t.dataset.smiles, 240, 160, null, dark, { fixedBondLength: 24 }));
    pop.innerHTML = `<span class="t">${esc(t.dataset.label)} · click to copy SMILES</span>${svgCache.get(k)}`;
    pop.style.display = "block";
    // above the id, kept inside the panel
    const pb = panel.getBoundingClientRect(), tb = t.getBoundingClientRect();
    const left = Math.max(4, Math.min(tb.left - pb.left, pb.width - pop.offsetWidth - 4));
    pop.style.left = `${left}px`;
    pop.style.top = `${tb.top - pb.top - pop.offsetHeight - 6}px`;
  });
  panel.addEventListener("mouseout", (ev) => {
    if (!ev.target.closest(".pid")) return;
    const pop = panel.querySelector(".ms-pop");
    if (pop) pop.style.display = "none";
  });

  const redraw = () => { drawPlot(); show(); };
  redraw();
  let lastW = canvas.clientWidth, lastH = canvas.clientHeight;
  const ro = new ResizeObserver(() => {
    if (canvas.clientWidth === lastW && canvas.clientHeight === lastH) return;
    lastW = canvas.clientWidth; lastH = canvas.clientHeight;
    redraw();
  });
  ro.observe(canvas);
  for (const k of ["data", "x", "y", "color_by", "color_range", "color_map", "mark_by", "diagonal", "same_axes"])
    model.on(`change:${k}`, redraw);
  model.on("change:height", () => { setHeight(); redraw(); });
  model.on("change:selected", show);
  return () => ro.disconnect();
}

export default { render };
