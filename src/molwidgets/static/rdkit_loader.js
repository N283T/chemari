// Shared RDKit.js (MinimalLib) loader. One WASM instance per page, shared by all widgets.
const RDKIT_VERSION = "2026.3.6";
const RDKIT_BASE = `https://cdn.jsdelivr.net/npm/@rdkit/rdkit@${RDKIT_VERSION}/dist/`;

export function loadRDKit() {
  if (!globalThis.__molwidgetsRDKit) {
    globalThis.__molwidgetsRDKit = new Promise((resolve, reject) => {
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
  return globalThis.__molwidgetsRDKit;
}

// Draw a SMILES to an SVG string. `hl` = {atoms, bonds, atomColors, bondColors}.
export function drawSvg(RDKit, smiles, width, height, hl = null, dark = false) {
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
    };
    if (dark) opts.backgroundColour = [0, 0, 0, 0];
    if (dark) opts.symbolColour = [0.9, 0.9, 0.9];
    if (hl) {
      opts.atoms = hl.atoms || [];
      opts.bonds = hl.bonds || [];
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
