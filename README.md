# PlanetFlow

**Planetary imaging post-processing pipeline with GUI**

PlanetFlow is a desktop application for processing planetary observation data captured as SER video files. It automates the full post-processing workflow — from raw frame sorting through wavelet sharpening, de-rotation stacking, multi-channel compositing, and animated GIF export — all from a single PySide6 GUI.

Supports both **monochrome cameras** (filter wheel, multi-filter SER) and **color cameras** (single Bayer sensor, continuous capture).

> Korean version: [README_ko.md](README_ko.md)

---

## Features

- **10-step configurable pipeline** with per-step enable/disable controls
- **Dual camera mode**: monochrome filter-wheel workflow and single color-camera workflow
- **Frame quality assessment** using Laplacian sharpness scoring
- **Planetary de-rotation** via JPL Horizons ephemeris (astroquery), with warp-scale auto-tune
- **Wavelet sharpening** (à trous algorithm, WaveSharp-compatible 0–500 scale) with limb feather control
- **Flexible multi-channel compositing**: user-defined RGB/LRGB specs (RGB, IR-RGB, CH4-G-IR, and custom)
- **Independent time-series compositing**: Step 08 has its own composite specs, separate from Step 06
- **Auto white balance + chromatic aberration correction** for color camera mode (Steps 06 & 08)
- **Time-series animation**: sliding-window stacking with quality weighting + animated GIF export
- **Summary contact sheet**: two output formats — composites-only grid always generated; two-zone grid (composites + filter images) for mono sessions with Step 05 data; optional per-window analytic views with filter stats and channel alignment tables
- **Live preview widgets**: wavelet (Steps 05 & 07), RGB composite (Step 06), levels (Step 10), color correction (Step 06 color), AP grid (Step 02)
- **Bilingual UI**: Korean / English (switchable at runtime)
- **Standalone executable**: ships as a single binary via PyInstaller (no Python required)
- **Lucky Stacking (Step 02)**: Fourier-domain quality-weighted stacking (Mackay 2013) — per-frequency weighting by `|FFT|^power` + Gaussian rolloff; AS!4-compatible PDS AP grid; multi-level parallelism (SER-level + frame-level ThreadPool)
- **Multi-format filename support**: Automatically recognises FireCapture/WinJUPOS (`YYYY-MM-DD-HHMM_D-CAM-FILTER-TARGET`), ASIAIR (`Target_YYYYMMDD-HHMMSS`), and SharpCap (`HH_MM_SS` under date/target folders) conventions
- **Profile management**: Named session profiles — save, load, and delete telescope/session configurations from the Settings panel
- **16-bit SER support**: Full 16-bit pipeline with correct centering and color debayer
- **System info on welcome screen**: Shows active profile, CPU cores, RAM, and GPU
- **Graceful pipeline stop**: Stop button on every step panel — confirms when all threads have truly halted

---

## Pipeline Overview

| Step | Name | Description |
|------|------|-------------|
| 01 | PIPP Preprocessing | Reject clipped/deformed frames, center-align, crop to square ROI (Optional) |
| 02 | Lucky Stacking | Per-AP independent patch stacking with NCC alignment and wide Gaussian blending, AS!4-compatible AP grid, σ-clip (Optional) |
| 03 | Quality Assessment | Score each TIF; find optimal time windows across all filters |
| 04 | De-rotation Stack | Spherical-warp de-rotation + quality-weighted mean stack; warp-scale auto-tune |
| 05 | Wavelet Master | Final wavelet sharpening on de-rotated master stacks with limb feathering |
| 06 | RGB Composite | User-defined multi-channel composites per window; auto WB+CA for color mode |
| 07 | Wavelet Preview | Apply wavelet sharpening to individual TIF stacks; export per-filter PNGs (Optional) |
| 08 | Time-Series Composite | Sliding-window stacks with independent composite specs; global filter normalisation (Optional) |
| 09 | Animated GIF | Assemble time-series frames into animated GIFs (Optional) |
| 10 | Summary Grid | Composites-only grid (always) + optional two-zone grid with filter images (mono); per-window analytic views; black-point + gamma levels (Optional) |

---

## Requirements

- Python 3.10 or later
- The following packages (install via `pip install -r requirements.txt`):

```
numpy
scipy
opencv-python
tifffile
Pillow
imageio[ffmpeg]
astropy
astroquery
scikit-image
PySide6
```

---

## Installation

```bash
git clone https://github.com/<your-username>/PlanetFlow.git
cd PlanetFlow
pip install -r requirements.txt
```

---

## Running (from source)

### GUI (recommended)

```bash
python gui/main.py
```

### CLI

Edit `PipelineConfig` in `main.py` to set your paths and parameters, then:

```bash
python main.py
```

---

## Building a Standalone Executable

No Python installation required on the target machine.

### Linux

```bash
./build_linux.sh
# Output: dist/PlanetFlow
```

### Windows

```bat
build_windows.bat
:: Output: dist\PlanetFlow.exe
```

Both scripts use a shared PyInstaller spec (`astro_pipeline.spec`) that collects all scientific library dependencies automatically.

> **Note:** PyInstaller cannot cross-compile. Build on the target OS.
> First launch extracts files to `/tmp` (Linux) or `%TEMP%` (Windows) — takes 5–15 s. Subsequent launches are fast.

---

## Output Structure

```
<output_dir>/
├── step03_quality/           # Quality CSV, window JSON, rankings
├── step04_derotated/         # De-rotated 16-bit TIFs per window
├── step05_wavelet_master/    # Master-sharpened PNGs per window
├── step06_rgb_composite/     # RGB/IR-RGB/CH4-G-IR composites per window
├── step07_wavelet_preview/   # Per-filter PNG previews (IR/R/G/B/CH4)
├── step08_series/            # Time-series composite frames
├── step09_gif/               # Animated GIFs
└── step10_summary_grid/      # Final summary grid PNGs
    ├── summary_grid_simple.png   # Composites only (always generated)
    ├── summary_grid.png          # Composites + filters (mono camera, when Step 05 data exists)
    └── analytic/                 # Per-window analytic views (optional, mono only)
```

> Steps 01 (PIPP) and 02 (Lucky Stacking) use their own user-configured output folders, separate from `<output_dir>`.

---

## Workflow

```
SER files
  └─► Step 01 (PIPP crop, optional)
        └─► Step 02 (Lucky Stacking, optional)
              └─► Step 03 (quality assessment)
                    └─► Step 04 (de-rotation stack)
                          └─► Step 05 (wavelet master)
                                └─► Step 06 (RGB composite)
                                      └─► Step 10 (summary grid, optional)
              └─► Step 07 (wavelet preview, optional)
                    └─► Step 08 (time-series composite, optional)
                          └─► Step 09 (animated GIF, optional)
```

---

## Typical Usage

1. Capture SER files with your planetary camera (e.g., Firecapture)
2. *(Optional)* Run **Step 01** to reject bad frames and crop to planet ROI
3. *(Optional)* Run **Step 02** (Lucky Stacking) to select the best frames and stack to TIF
4. Run **Steps 03–06** for quality assessment, de-rotation stacking, wavelet sharpening, and RGB compositing
5. *(Optional)* Run **Step 07** (Wavelet Preview) then **Steps 08–09** for time-series animation
6. *(Optional)* Run **Step 10** for a summary contact sheet

Alternatively, use the **▶ Run All** button which automatically executes all enabled steps in sequence from the configured start point (Step 1, 2, or 3), with input validation and a confirmation dialog before starting.
