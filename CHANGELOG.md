# Changelog

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- A page of every widget with the code that makes it, on GitHub Pages
  ([n283t.github.io/chemari](https://n283t.github.io/chemari/)); the notebook behind it is
  `docs/widgets.py`.
- `LICENSE` (MIT) and this changelog.

### Changed

- On a static export of a marimo notebook, a control that needs Python shows "needs Python: not
  available on a static page" instead of "computing…".
- The README is shorter; the widgets are a table.

### Removed

- The notebooks on the PXR data alone (`examples/ecfp_pxr.py`, `ecfp_pxr_ja.py`).

## [0.1.4] - 2026-10-04

### Changed

- `ecfp_openadmet`: the "AI use" line says that the narration of the demo video was generated
  with Gemini TTS.

## [0.1.3] - 2026-10-04

### Fixed

- `ECFPMovie` drew a frame on every animation frame while paused and kept a CPU core busy.
- `MorganBitTiles` and `MorganExplorer` lost their scroll position when a row was selected.

## [0.1.2] - 2026-10-04

### Fixed

- The English `ecfp_openadmet` notebook showed the endpoint notes in Japanese.

## [0.1.1] - 2026-10-04

### Fixed

- `ECFPMovie`: a click on the picture pauses the movie.

## [0.1.0] - 2026-10-04

### Added

- The widgets as the package `chemari`: `MolGrid`, `MolPair`, `MolScatter`, `ECFPMovie`,
  `ECFPStepper`, `MorganBitTiles`, `MorganExplorer`, `BitAtlas`, `BitImportance`.
- The notebook `examples/ecfp_openadmet.py` ("Do you really know your ECFP4?") in English and
  Japanese, with its precomputed tables.

[Unreleased]: https://github.com/N283T/chemari/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/N283T/chemari/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/N283T/chemari/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/N283T/chemari/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/N283T/chemari/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/N283T/chemari/releases/tag/v0.1.0
