// ECFPStepper: walk through the Morgan/ECFP algorithm one atom and one iteration at a time.
// The molecule SVG is rendered in Python (RDKit) because it carries per-atom labels.

const CSS = `
.es-root { font: 13px/1.45 system-ui, sans-serif; color: var(--es-fg); --es-fg:#1f2328; --es-muted:#6b7280;
  --es-border:#d0d7de; --es-card:#fff; --es-soft:#f6f8fa; --es-new:#2f9e44; --es-dup:#e8590c; --es-hit:#d6336c;
  --es-accent:#1c7ed6; }
.es-root.dark { --es-fg:#e6e6e6; --es-muted:#9aa4b2; --es-border:#3a3f47; --es-card:#1c1f24; --es-soft:#24282e; }
.es-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:8px; }
.es-btn { font:inherit; color:var(--es-fg); background:var(--es-soft); border:1px solid var(--es-border);
  border-radius:6px; padding:3px 10px; cursor:pointer; }
.es-btn.primary { background:var(--es-accent); border-color:var(--es-accent); color:#fff; font-weight:600; }
.es-btn:disabled { opacity:.4; cursor:default; }
.es-seg { display:inline-flex; border:1px solid var(--es-border); border-radius:6px; overflow:hidden; }
.es-seg button { font:inherit; color:var(--es-fg); background:var(--es-soft); border:0; padding:3px 10px; cursor:pointer; }
.es-seg button + button { border-left:1px solid var(--es-border); }
.es-seg button.on { background:var(--es-fg); color:var(--es-card); }
.es-lbl { color:var(--es-muted); }
.es-pos { color:var(--es-muted); font-variant-numeric:tabular-nums; }
.es-main { display:grid; grid-template-columns: minmax(260px, 1fr) minmax(300px, 1.1fr); gap:12px; }
@media (max-width: 760px) { .es-main { grid-template-columns: 1fr; } }
.es-mol { background:#fff; border:1px solid var(--es-border); border-radius:8px; padding:4px; }
.es-mol svg { width:100%; height:auto; display:block; }
.es-panel { display:flex; flex-direction:column; gap:8px; }
.es-box { background:var(--es-soft); border-radius:8px; padding:8px 10px; }
.es-box h4 { margin:0 0 6px; font-size:13px; }
.es-recipe { font-family:ui-monospace,monospace; font-size:14px; background:var(--es-card); border:1px solid var(--es-border);
  border-radius:6px; padding:6px 8px; margin:4px 0; word-break:break-word; }
.es-env { font-family:ui-monospace,monospace; }
.es-tag { display:inline-block; border-radius:4px; padding:0 6px; color:#fff; font-size:11px; font-weight:600; margin-right:6px; }
.es-chips { display:flex; flex-wrap:wrap; gap:6px; }
.es-chip { display:inline-flex; align-items:center; gap:6px; background:var(--es-card); border:1px solid var(--es-border);
  border-radius:999px; padding:2px 10px 2px 3px; font-size:12px; }
.es-chip .dot { min-width:20px; height:20px; border-radius:999px; display:inline-flex; align-items:center; justify-content:center;
  font-weight:700; font-size:11px; color:#1f2328; padding:0 4px; }
.es-chip.fresh { border-color:var(--es-new); box-shadow:0 0 0 2px rgba(47,158,68,.3); }
.es-vec-head { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:10px 0 6px; }
.es-arrow { font-family:ui-monospace,monospace; color:var(--es-muted); }
.es-grid { display:grid; gap:3px; }
.es-cell { aspect-ratio:1; border-radius:3px; background:var(--es-card); border:1px solid var(--es-border);
  display:flex; align-items:center; justify-content:center; font:700 10px/1 ui-monospace,monospace; color:#1f2328;
  overflow:hidden; transition:transform .15s; }
.es-grid.tiny .es-cell { border-radius:1px; font-size:0; }
.es-cell.hit { box-shadow:0 0 0 2px var(--es-hit) inset; }
.es-cell.fresh { outline:2px solid var(--es-new); outline-offset:1px; transform:scale(1.15); }
.es-cell.hl { outline:2px solid var(--es-fg); outline-offset:1px; }
.es-fold-line { font-family:ui-monospace,monospace; font-size:12px; margin-top:6px; }
.es-muted { color:var(--es-muted); }
.es-hint { color:var(--es-muted); font-size:11.5px; margin-top:4px; }
`;

const STATUS = {
  new: ["kept", "var(--es-new)"],
  duplicate: ["dropped: duplicate", "var(--es-dup)"],
  "no growth": ["dropped: no growth", "var(--es-muted)"],
};

function el(tag, attrs = {}, html = "") {
  const e = Object.assign(document.createElement(tag), attrs);
  if (html) e.innerHTML = html;
  return e;
}

function seg(options, current, onPick) {
  const wrap = el("div", { className: "es-seg" });
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
  const root = el("div", { className: "es-root" });
  if (isDark(host)) root.classList.add("dark");
  root.appendChild(el("style", {}, CSS));
  host.appendChild(root);

  const get = (k) => model.get(k);
  const set = (obj) => { for (const [k, v] of Object.entries(obj)) model.set(k, v); model.save_changes(); };

  // Guided tour: every (iteration, atom) in order.
  const tour = [];
  get("steps").forEach((layer, r) => layer.forEach((s) => tour.push([r, s.atom])));
  let guided = get("guided");
  let pos = Math.max(0, tour.findIndex(([r, a]) => r === get("radius") && a === get("atom")));

  const goto = (p) => {
    pos = Math.max(0, Math.min(tour.length - 1, p));
    set({ radius: tour[pos][0], atom: tour[pos][1] });
  };

  // Steps visited so far: the tour prefix in guided mode, whole iterations in explore mode.
  function visited() {
    const steps = get("steps");
    if (guided) return tour.slice(0, pos + 1).map(([r, a]) => steps[r][a]);
    return steps.slice(0, get("radius") + 1).flat();
  }

  function draw() {
    root.querySelectorAll(":scope > :not(style)").forEach((n) => n.remove());
    const steps = get("steps");
    const r = get("radius");
    const atom = get("atom");
    const maxR = steps.length - 1;

    // --- controls
    const bar = el("div", { className: "es-bar" });
    bar.appendChild(seg([[true, "guided"], [false, "explore"]], guided, (v) => {
      guided = v;
      if (guided) goto(pos); else draw();
    }));
    if (guided) {
      const prev = el("button", { className: "es-btn", textContent: "◀ back" });
      const next = el("button", { className: "es-btn primary", textContent: "next ▶" });
      const nextIt = el("button", { className: "es-btn", textContent: "next iteration ⏭" });
      const restart = el("button", { className: "es-btn", textContent: "↺ restart", title: "back to the first step" });
      prev.disabled = pos === 0;
      next.disabled = pos === tour.length - 1;
      nextIt.disabled = r === maxR;
      prev.onclick = () => goto(pos - 1);
      next.onclick = () => goto(pos + 1);
      nextIt.onclick = () => goto(tour.findIndex(([rr]) => rr === r + 1));
      restart.onclick = () => goto(0);
      bar.append(prev, next, nextIt, restart, el("span", { className: "es-pos", textContent: `step ${pos + 1} / ${tour.length}` }));
    } else {
      bar.append(
        el("span", { className: "es-lbl", textContent: "iteration" }),
        seg([...Array(maxR + 1).keys()].map((i) => [i, String(i)]), r, (v) => set({ radius: v })),
        el("span", { className: "es-lbl", textContent: "· click an atom" }),
      );
    }
    root.appendChild(bar);

    // --- molecule
    const main = el("div", { className: "es-main" });
    const left = el("div");
    const mol = el("div", { className: "es-mol" }, get("svg"));
    mol.addEventListener("click", (ev) => {
      const target = ev.target.closest(".es-hit");
      if (!target) return;
      const a = Number(target.getAttribute("data-atom"));
      if (guided) goto(tour.findIndex(([rr, aa]) => rr === r && aa === a));
      else set({ atom: a === atom ? -1 : a });
    });
    left.appendChild(mol);
    left.appendChild(el("div", { className: "es-hint" },
      `Labels show each atom's identifier at iteration ${r}; the same label (and colour) means the same identifier. ` +
      "Lower-case letters are iteration 0, capitals iteration 1, primes (A', A'') iterations 2 and 3."));
    main.appendChild(left);

    // --- explanation of the current atom
    const panel = el("div", { className: "es-panel" });
    const box = el("div", { className: "es-box" });
    if (atom >= 0) {
      const s = steps[r][atom];
      const [tag, colour] = STATUS[s.status];
      const intro = r === 0
        ? "Iteration 0 describes the atom on its own, from six numbers:"
        : `Iteration ${r} combines this atom's label from iteration ${r - 1} with its neighbours' labels and bond types:`;
      box.innerHTML =
        `<h4>Iteration ${r} · atom ${atom}</h4>` +
        `<div class="es-muted">${intro}</div>` +
        `<div class="es-recipe">${esc(s.recipe_text)}</div>` +
        `<div>This identifier describes <span class="es-env">${esc(s.env)}</span> ` +
        `<span class="es-muted">(${s.atoms.length} atom${s.atoms.length === 1 ? "" : "s"}, ${s.bonds.length} bond${s.bonds.length === 1 ? "" : "s"})</span></div>` +
        `<div style="margin-top:6px"><span class="es-tag" style="background:${colour}">${tag}</span>${esc(s.why)}</div>` +
        (s.status === "new"
          ? `<div class="es-fold-line">fold: 0x${(s.identifier >>> 0).toString(16).padStart(8, "0")} mod ${get("n_bits")} → bit ${s.identifier % get("n_bits")}</div>`
          : `<div class="es-fold-line es-muted">not added, so no bit changes</div>`);
    } else {
      box.innerHTML = `<h4>Iteration ${r}</h4><div class="es-muted">Click an atom to see how its identifier was built.</div>`;
    }
    panel.appendChild(box);

    // --- the fingerprint collected so far
    const kept = new Map();
    let dropped = 0;
    for (const s of visited()) {
      if (s.status !== "new") { dropped++; continue; }
      if (!kept.has(s.identifier)) kept.set(s.identifier, { ...s, atomsWith: 0 });
      kept.get(s.identifier).atomsWith++;
    }
    const current = atom >= 0 ? steps[r][atom] : null;
    const colours = get("colours");
    const fp = el("div", { className: "es-box" });
    fp.appendChild(el("h4", {}, `The fingerprint so far: ${kept.size} feature${kept.size === 1 ? "" : "s"}` +
      (dropped ? ` <span class="es-muted" style="font-weight:400">· ${dropped} environment${dropped === 1 ? "" : "s"} dropped</span>` : "")));
    const chips = el("div", { className: "es-chips" });
    const nBits = get("n_bits");
    for (const f of kept.values()) {
      const fresh = current && current.status === "new" && f.identifier === current.identifier;
      const c = el("span", { className: "es-chip" + (fresh ? " fresh" : "") });
      c.innerHTML = `<span class="dot" style="background:${colours[f.identifier] || "#ddd"}">${esc(f.label)}</span>` +
        `<span class="es-env">${esc(f.env)}</span>` + (f.atomsWith > 1 ? `<span class="es-muted">×${f.atomsWith} atoms</span>` : "");
      const bit = f.identifier % nBits;
      c.addEventListener("mouseenter", () => fp.querySelector(`.es-cell[data-bit="${bit}"]`)?.classList.add("hl"));
      c.addEventListener("mouseleave", () => fp.querySelector(`.es-cell[data-bit="${bit}"]`)?.classList.remove("hl"));
      chips.appendChild(c);
    }
    fp.appendChild(chips);

    // --- the same features as a folded bit vector
    const byBit = new Map();
    for (const f of kept.values()) {
      const b = f.identifier % nBits;
      if (!byBit.has(b)) byBit.set(b, []);
      byBit.get(b).push(f);
    }
    const collided = [...byBit.values()].filter((v) => v.length > 1).length;
    const head = el("div", { className: "es-vec-head" });
    head.append(
      el("span", { className: "es-arrow", textContent: "↓ identifier mod" }),
      seg([[16, "16"], [64, "64"], [256, "256"], [2048, "2048"]], nBits, (v) => set({ n_bits: v })),
      el("span", {}, `<b>${byBit.size}</b> of ${nBits} bits on` +
        (collided ? ` · <span style="color:var(--es-hit);font-weight:600">${collided} bit${collided > 1 ? "s hold" : " holds"} two or more features</span>` : "")),
    );
    fp.appendChild(head);
    const tiny = nBits > 256;
    const grid = el("div", { className: "es-grid" + (tiny ? " tiny" : "") });
    grid.style.gridTemplateColumns = `repeat(${nBits <= 64 ? 16 : nBits <= 256 ? 32 : 64}, 1fr)`;
    const freshBit = current && current.status === "new" ? current.identifier % nBits : -1;
    for (let b = 0; b < nBits; b++) {
      const fs = byBit.get(b);
      const cell = el("div", { className: "es-cell" });
      cell.dataset.bit = b;
      cell.title = `bit ${b}` + (fs ? ": " + fs.map((f) => `${f.label} ${f.env}`).join(" | ") : " (0)");
      if (fs) {
        cell.style.background = colours[fs[0].identifier] || "#74c0fc";
        if (!tiny) cell.textContent = fs.map((f) => f.label).join("");
        if (fs.length > 1) cell.classList.add("hit");
      }
      if (b === freshBit) cell.classList.add("fresh");
      grid.appendChild(cell);
    }
    fp.appendChild(grid);
    fp.appendChild(el("div", { className: "es-hint" },
      "Top: the unfolded ECFP, a set of identifiers (each counted once, however many atoms share it). " +
      "Bottom: the bit vector a model sees. Each identifier switches on bit (identifier mod n_bits); " +
      "a red-framed bit holds two different features, which the model can no longer tell apart."));
    panel.appendChild(fp);
    main.appendChild(panel);
    root.appendChild(main);
  }

  draw();
  for (const k of ["svg", "steps", "radius", "atom", "n_bits"]) model.on(`change:${k}`, draw);
}

export default { render };
