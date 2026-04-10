# AstroPipeline

**Planetary imaging post-processing pipeline with GUI**

AstroPipeline is a desktop application for processing planetary observation data captured as SER video files. It automates the full post-processing workflow — from raw frame sorting through wavelet sharpening, de-rotation stacking, multi-channel compositing, and animated GIF export — all from a single PySide6 GUI.

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
- **Independent time-series compositing**: Step 08 has its own composite specs, separate from Step 07
- **Auto white balance + chromatic aberration correction** for color camera mode (Steps 07 & 08)
- **Time-series animation**: sliding-window stacking with quality weighting + animated GIF export
- **Summary contact sheet**: all windows × composites in a single image
- **Live preview widgets**: wavelet (Step 06), RGB composite (Step 07), levels (Step 10), color correction (Step 07 color)
- **Bilingual UI**: Korean / English (switchable at runtime)
- **Standalone executable**: ships as a single binary via PyInstaller (no Python required)

---

## Pipeline Overview

| Step | Name | Description |
|------|------|-------------|
| 01 | PIPP Preprocessing | Reject clipped/deformed frames, center-align, crop to square ROI |
| 02 | AutoStakkert! 4 | *(External)* Manual stacking — run AS!4 on Step 01 output |
| 03 | Wavelet Preview | Apply wavelet sharpening to all TIF stacks; export filter PNGs |
| 04 | Quality Assessment | Score each TIF; find optimal time windows across all filters |
| 05 | De-rotation Stack | Spherical-warp de-rotation + quality-weighted mean stack; warp-scale auto-tune |
| 06 | Wavelet Master | Final wavelet sharpening on de-rotated master stacks with limb feathering |
| 07 | RGB Composite | User-defined multi-channel composites per window; auto WB+CA for color mode |
| 08 | Time-Series Composite | Sliding-window stacks with independent composite specs; global filter normalisation |
| 09 | Animated GIF | Assemble time-series frames into animated GIFs |
| 10 | Summary Grid | Contact sheet with black-point + gamma levels adjustment |

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
git clone https://github.com/<your-username>/AstroPipeline.git
cd AstroPipeline
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
# Output: dist/AstroPipeline
```

### Windows

```bat
build_windows.bat
:: Output: dist\AstroPipeline.exe
```

Both scripts use a shared PyInstaller spec (`astro_pipeline.spec`) that collects all scientific library dependencies automatically.

> **Note:** PyInstaller cannot cross-compile. Build on the target OS.
> First launch extracts files to `/tmp` (Linux) or `%TEMP%` (Windows) — takes 5–15 s. Subsequent launches are fast.

---

## Output Structure

```
<output_dir>/
├── step01_pipp/              # Cropped SER + rejection stats
├── step03_wavelet_preview/   # Per-filter PNG previews (IR/R/G/B/CH4)
├── step04_quality/           # Quality CSV, window JSON, rankings
├── step05_derotated/         # De-rotated 16-bit TIFs per window
├── step06_wavelet_master/    # Master-sharpened PNGs per window
├── step07_rgb_composite/     # RGB/IR-RGB/CH4-G-IR composites
├── step08_series/            # Time-series composite frames
├── step09_gif/               # Animated GIFs
└── step10_summary_grid/      # Final contact sheet PNG
```

---

## Workflow

```
SER files
  └─► Step 01 (PIPP crop)
        └─► [AS!4 external stacking]
              └─► Step 03 (wavelet preview)
                    └─► Step 04 (quality score)
                          ├─► Step 05 (de-rotation stack)
                          │     └─► Step 06 (wavelet master)
                          │           └─► Step 07 (RGB composite)
                          │                 └─► Step 10 (summary grid)
                          └─► Step 08 (time-series)
                                └─► Step 09 (animated GIF)
```

---

## Typical Usage

1. Capture SER files with your planetary camera (e.g., Firecapture)
2. Run **Step 01** to reject bad frames and crop to planet ROI
3. Stack with **AutoStakkert! 4** externally
4. Run **Steps 03–04** to preview and score stacks
5. Run **Steps 05–07** for final de-rotated composites
6. *(Optional)* Run **Steps 08–10** for time-series animation and summary

---

## License

MIT
