// Shared RDKit.js (MinimalLib) loader. One WASM instance per page, shared by all widgets.
const RDKIT_VERSION = "2026.3.6";
const RDKIT_BASE = `https://cdn.jsdelivr.net/npm/@rdkit/rdkit@${RDKIT_VERSION}/dist/`;

export function loadRDKit() {
  if (!globalThis.__chemariRDKit) {
    globalThis.__chemariRDKit = new Promise((resolve, reject) => {
      const init = () =>
        globalThis
          .initRDKitModule({ locateFile: (f) => RDKIT_BASE + f })
          .then((RDKit) => {
            RDKit.prefer_coordgen(true); // CoordGen: cleaner rings and macrocycles
            resolve(RDKit);
          }, reject);
      if (globalThis.initRDKitModule) return init();
      const script = document.createElement("script");
      script.src = RDKIT_BASE + "RDKit_minimal.js";
      script.onload = init;
      script.onerror = () => reject(new Error("Failed to load RDKit.js"));
      document.head.appendChild(script);
    });
  }
  return globalThis.__chemariRDKit;
}

// Draw a SMILES to an SVG string. `hl` = {atoms, bonds, atomColors, bondColors}.
// `extra` = further RDKit.js drawing options, e.g. {addStereoAnnotation: true} for R/S and E/Z labels.
export function drawSvg(RDKit, smiles, width, height, hl = null, dark = false, extra = {}) {
  const mol = RDKit.get_mol(smiles);
  if (!mol) return `<svg width="${width}" height="${height}"><text x="8" y="20" fill="#c33">invalid</text></svg>`;
  try {
    const opts = {
      width,
      height,
      clearBackground: false,
      bondLineWidth: 1.2,
      fixedBondLength: 28,
      padding: 0.06,
      ...extra,
    };
    if (dark) opts.backgroundColour = [0, 0, 0, 0];
    if (dark) opts.symbolColour = [0.9, 0.9, 0.9];
    if (hl) {
      // copy: the invisible highlights pushed below must not leak into the caller's object
      opts.atoms = [...(hl.atoms || [])];
      opts.bonds = [...(hl.bonds || [])];
      if (hl.atomColors) opts.highlightAtomColors = hl.atomColors;
      if (hl.bondColors) opts.highlightBondColors = hl.bondColors;
      opts.highlightRadius = 0.35;
      if (hl.stable) {
        // Highlight circles count towards the drawing's bounds, so lighting up an edge atom would
        // shift/rescale the molecule. Give every atom an invisible highlight: bounds never change.
        const n = mol.get_num_atoms ? mol.get_num_atoms() : JSON.parse(mol.get_json()).molecules[0].atoms.length;
        const on = new Set(opts.atoms);
        opts.highlightAtomColors = { ...(opts.highlightAtomColors || {}) };
        for (let i = 0; i < n; i++)
          if (!on.has(i)) { opts.atoms.push(i); opts.highlightAtomColors[i] = [1, 1, 1, 0]; }
      }
    }
    let svg = mol.get_svg_with_highlights(JSON.stringify(opts));
    if (dark) svg = darkenSvg(svg);
    return svg;
  } finally {
    mol.delete();
  }
}

// RDKit draws bonds/atoms in black; flip neutral strokes for dark backgrounds.
function darkenSvg(svg) {
  return svg
    .replace(/stroke:#000000/g, "stroke:#e6e6e6")
    .replace(/fill:#000000/g, "fill:#e6e6e6");
}

// Colour key shared by the widgets. `items` = [[rgb 0..1, label, extra CSS], ...].
export function legendHtml(title, items) {
  const dot = ([r, g, b], css = "") =>
    `<i style="display:inline-block;width:11px;height:11px;border-radius:50%;vertical-align:-1px;margin-right:4px;` +
    `background:rgb(${r * 255},${g * 255},${b * 255});${css}"></i>`;
  return `<span style="white-space:nowrap;margin-right:12px"><b style="font-weight:600">${title}</b></span>` +
    items.map(([c, label, css]) => `<span style="white-space:nowrap;margin-right:12px">${dot(c, css)}${label}</span>`).join("");
}

// Default colours of RDKit's Draw.DrawMorganEnv, used for every bit picture rendered in Python.
export const MORGAN_ENV_KEY = [
  [[0.6, 0.6, 0.9], "centre atom"],
  [[0.9, 0.9, 0.2], "aromatic atom"],
  [[0.8, 0.8, 0.8], "aliphatic ring atom"],
  [[0.9, 0.9, 0.9], "* where it attaches", "border:1px dashed #9ca3af"],
];

export function isDark(el) {
  // marimo toggles a `dark` / `dark-theme` class; otherwise use the first opaque background.
  for (const n of [document.documentElement, document.body]) {
    if (n.classList.contains("dark") || n.classList.contains("dark-theme")) return true;
    if (n.classList.contains("light") || n.classList.contains("light-theme")) return false;
  }
  for (let n = el; n && n.nodeType === 1; n = n.parentElement || n.getRootNode()?.host) {
    const css = getComputedStyle(n).backgroundColor;
    const m = css.match(/[\d.]+/g);
    if (m && m.length >= 3 && (m.length < 4 || Number(m[3]) > 0.5)) {
      const scale = css.startsWith("color(") ? 255 : 1; // color(srgb r g b) uses 0..1
      const [r, g, b] = m.slice(0, 3).map((v) => Number(v) * scale);
      return 0.2126 * r + 0.7152 * g + 0.0722 * b < 110;
    }
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

// "computing…" while Python recomputes after a change made in the browser. Python bumps `rev`
// when it is done (see _Computing in widgets.py). Returns mark(changes): call it with the
// {name: value} object you are about to set; only real changes to `triggers` count. Nothing
// shows for fast answers, so quick updates do not flicker.
const BUSY_CSS = `
.mw-host { position: relative; }
.mw-host.mw-busy > :not(style) { opacity: .55; transition: opacity .15s; }
.mw-host.mw-busy::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 3px; z-index: 5;
  background: linear-gradient(90deg, transparent, #1c7ed6, transparent) no-repeat; background-size: 35% 100%;
  animation: mw-slide 1s ease-in-out infinite; border-radius: 2px; }
.mw-host.mw-busy::after { content: "computing…"; position: absolute; top: 8px; right: 8px; z-index: 5;
  font: 600 11px/1.6 system-ui, sans-serif; color: #fff; background: rgba(28, 126, 214, .92);
  padding: 0 9px; border-radius: 999px; }
@keyframes mw-slide { from { background-position: -50% 0; } to { background-position: 150% 0; } }
`;

export function busyIndicator(model, host, triggers) {
  host.classList.add("mw-host");
  const style = document.createElement("style");
  style.textContent = BUSY_CSS;
  host.appendChild(style);
  let show = null, giveUp = null;
  const stop = () => { clearTimeout(show); clearTimeout(giveUp); host.classList.remove("mw-busy"); };
  model.on("change:rev", stop);
  return (changes) => {
    const same = (k) => JSON.stringify(model.get(k)) === JSON.stringify(changes[k]);
    if (!Object.keys(changes).some((k) => triggers.includes(k) && !same(k))) return;
    clearTimeout(show); clearTimeout(giveUp);
    show = setTimeout(() => host.classList.add("mw-busy"), 150);
    giveUp = setTimeout(stop, 30000); // never leave it spinning if an answer is lost
  };
}

// "Copy SMILES" icon in the top-right corner of a molecule drawing. Put smilesCopyHtml(smiles)
// inside the drawing's box and give the box the class `mw-copyable`; enableSmilesCopy(root,
// model) adds the style and one click handler for everything under `root`, and hides the icons
// while the widget's `copy_smiles` is false. The handler runs in the
// capture phase and stops the click, so it never also selects a card or an atom.
const COPY_ICON =
  '<svg class="mw-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
  '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V6a2 2 0 0 1 2-2h9"/></svg>';
const CHECK_ICON =
  '<svg class="mw-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">' +
  '<path d="M5 12.5l4.5 4.5L19 7.5"/></svg>';
const COPY_CSS = `
.mw-copyable { position: relative; }
.mw-copy { position: absolute; top: 4px; right: 4px; z-index: 2; width: 24px; height: 24px; padding: 0; margin: 0;
  display: flex; align-items: center; justify-content: center; border: 1px solid #d0d7de; border-radius: 6px;
  background: rgba(255, 255, 255, .92); color: #57606a; cursor: pointer; opacity: .75; }
.mw-copyable:hover .mw-copy, .mw-copy:focus-visible { opacity: 1; }
.mw-copy:hover { color: #1f2328; border-color: #8c959f; }
.mw-copy.done { color: #1a7f37; border-color: #1a7f37; opacity: 1; }
.mw-nocopy .mw-copy { display: none; }
/* the chirality switch of MorganExplorer / MorganBitTiles: a checkbox styled like the segmented
   buttons next to it (dark when on) */
.mw-chiral { display: inline-flex; align-items: center; gap: 6px; margin-left: 4px; padding: 2px 9px;
  border: 1px solid #d0d7de; border-radius: 6px; color: inherit; background: #f6f8fa;
  font: inherit; cursor: pointer; user-select: none; }
.mw-chiral input { width: 13px; height: 13px; margin: 0; accent-color: #1f2328; cursor: pointer; }
.mw-chiral.on { background: #1f2328; border-color: #1f2328; color: #fff; }
.mw-chiral.on input { accent-color: #fff; }
.dark .mw-chiral { border-color: #3a3f47; background: #24282e; }
.dark .mw-chiral.on { background: #e6e6e6; border-color: #e6e6e6; color: #1c1f24; }
.dark .mw-chiral.on input { accent-color: #1c1f24; }
.mw-copy svg.mw-ico { width: 14px; height: 14px; max-width: none; max-height: none; margin: 0; display: block; }
`;

export function smilesCopyHtml(smiles) {
  const attr = String(smiles).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  return `<button type="button" class="mw-copy" data-smiles="${attr}" title="Copy SMILES" aria-label="Copy SMILES">${COPY_ICON}</button>`;
}

export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // clipboard API unavailable (e.g. an insecure context): the old way
    const t = document.createElement("textarea");
    t.value = text;
    document.body.appendChild(t);
    t.select();
    document.execCommand("copy");
    t.remove();
  }
}

export function enableSmilesCopy(root, model) {
  const style = document.createElement("style");
  style.textContent = COPY_CSS;
  root.appendChild(style);
  // the widget's `copy_smiles` option hides the icons
  const show = () => root.classList.toggle("mw-nocopy", model.get("copy_smiles") === false);
  show();
  model.on("change:copy_smiles", show);
  root.addEventListener("click", async (ev) => {
    const b = ev.target.closest(".mw-copy");
    if (!b) return;
    ev.stopPropagation();
    ev.preventDefault();
    await copyText(b.dataset.smiles);
    b.classList.add("done");
    b.innerHTML = CHECK_ICON;
    setTimeout(() => { b.classList.remove("done"); b.innerHTML = COPY_ICON; }, 1200);
  }, true);
}
