// Shared RDKit.js (MinimalLib) loader. One WASM instance per page, shared by all widgets.
const RDKIT_VERSION = "2026.3.6";
const RDKIT_BASE = `https://cdn.jsdelivr.net/npm/@rdkit/rdkit@${RDKIT_VERSION}/dist/`;

export function loadRDKit() {
  if (!globalThis.__molwidgetsRDKit) {
    globalThis.__molwidgetsRDKit = new Promise((resolve, reject) => {
      const init = () =>
        globalThis
          .initRDKitModule({ locateFile: (f) => RDKIT_BASE + f })
          .then(resolve, reject);
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
