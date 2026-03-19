# Spectra-Spin

Batch spectral image reconstruction from **spinning-disk spectral microscopy** data.

This project provides tools for extracting per-pixel spectral information from
raw camera frames captured through a spinning-disk spectral module.  A series of
2-D monochrome images is analysed to detect periodic interference structures,
isolate individual spectral bands, and assemble them into a single 3-D
hyperspectral data cube `(height, width, spectral_channels)`.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
  - [Edge Filtering](#1-edge-filtering)
  - [Periodic Structure Detection](#2-periodic-structure-detection-dynamic-programming)
  - [Smoothing and Regularization](#3-smoothing-and-regularization)
  - [Spectral Band Extraction](#4-spectral-band-extraction)
  - [Spectral Pixel Allocation](#5-spectral-pixel-allocation)
  - [Batch Accumulation](#6-batch-accumulation)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Class `SpectralAnalyzer`](#class-spectralanayzer)
  - [Function `reconstruct_spectral_image`](#function-reconstruct_spectral_image)
- [CLI Usage](#cli-usage)
- [Project Structure](#project-structure)
- [AI Disclaimer](#ai-disclaimer)
- [License](#license)

---

## Overview

A spinning-disk spectral microscopy system disperses the emission spectrum of
each point in the sample across a series of spatially separated bands on the
camera sensor.  Each raw frame therefore contains alternating **light** (signal)
and **dark** (gap) horizontal stripes whose positions encode the spectral axis.

The reconstruction pipeline implemented here:

1. Detects the positions of these periodic stripes automatically using a
   dynamic-programming (DP) algorithm on the edge-filtered image.
2. Smooths and regularizes the detected positions with B-spline fitting.
3. Extracts the intensity data between adjacent dark boundaries to form
   per-band spectral slices.
4. Places those slices at their correct spatial coordinates to produce a 3-D
   hyperspectral image.
5. Optionally averages (or sums / takes the median of) multiple frames to
   improve the signal-to-noise ratio.

The interactive prototype of the algorithm lives in `demo.ipynb`; the
production-ready, class-based implementation is in **`batch.py`**.

---

## Algorithm

### 1. Edge Filtering

A **Prewitt filter** along the vertical axis (`axis=0`) is applied to the raw
image to enhance horizontal intensity transitions — the boundaries between
spectral bands:

```
edge_image = prewitt(image, axis=0)
```

### 2. Periodic Structure Detection (Dynamic Programming)

The detected edge image is processed with a custom DP algorithm that traces
globally optimal paths of maximum (for light lines) or minimum (for dark lines)
cumulative intensity.

#### Light Line Detection

$$S_{\max}(i, j) = I(i, j) + \max_{k \in \{-1, 0, 1\}} S_{\max}(i+k,\; j-1)$$

The algorithm iteratively computes the maximum cumulative intensity matrix
$S_{\max}$.  The value of each node $(i, j)$ is defined as the sum of the local
pixel intensity $I(i, j)$ and the maximum accumulated value from three adjacent
nodes in the preceding column $j - 1$.  This formulation guarantees the
identification of the globally optimal path with the highest overall brightness.

#### Dark Line Detection

$$S_{\min}(i, j) = I(i, j) + \min_{k \in \{-1, 0, 1\}} S_{\min}(i+k,\; j-1)$$

The algorithm constructs the minimum cumulative cost matrix $S_{\min}$.  The
value at point $(i, j)$ is calculated by adding the local intensity $I(i, j)$ to
the minimum accumulated value from the local neighbourhood in the preceding
column $j - 1$.  This minimises the total intensity along the graph, ensuring
the localisation of the darkest optimal path.

In both equations, $i \in [0, M-1]$ represents the spatial row coordinate,
$j \in [1, N-1]$ denotes the column coordinate, and $k$ constrains the spatial
derivative of the path, restricting transitions exclusively to adjacent pixels.

#### Backtracking

$$P(i, j) = \arg\max_{k \in \{-1, 0, 1\}} S(i+k,\; j-1)$$

$$y(N-1) = \arg\max_{i \in [0, M-1]} S(i,\; N-1)$$

$$y(j-1) = y(j) + P(y(j),\; j), \quad \forall\, j \in \{N-1, \dots, 1\}$$

The backtracking phase reconstructs the spatial coordinates of the optimal
continuous structure, denoted as $y(j)$, iterating from the last column $N - 1$
back to the first.  During the forward pass, a pointer matrix $P(i, j)$ is
constructed to record the optimal spatial transition $k \in \{-1, 0, 1\}$ chosen
to reach each node $(i, j)$.  The sequence is initialised by finding the global
extremum (maximum for light structures, minimum for dark structures) in the
final column of the cumulative cost matrix $S$.  The optimal path is then
iteratively traced backward using the stored spatial offsets.

#### Spatial Period Estimation

$$P(y) = \mathrm{median}_{x \in [0, N-1]}\; I(y, x)$$

$$\bar{P}(y) = P(y) - \frac{1}{M}\sum_{k=0}^{M-1} P(k)$$

$$R(\tau) = \sum_{y=0}^{M-\tau-1} \bar{P}(y)\;\bar{P}(y+\tau)$$

$$T = \arg\max_{\tau > 0} R(\tau)$$

$$N \approx \lfloor M / T \rfloor$$

The automatic estimation of the spatial period is based on the 1-D
autocorrelation of the image's vertical intensity profile.  First, a robust
spatial profile $P(y)$ is extracted by calculating the median intensity along
the horizontal axis $x$ to suppress localised noise.  The DC component is
subsequently removed to produce a zero-mean signal $\bar{P}(y)$.  The discrete
autocorrelation function $R(\tau)$ is then computed for spatial lags $\tau$.  The
spatial period $T$ (the distance between adjacent ridges or valleys) is
identified by locating the first prominent global maximum of $R(\tau)$ at
$\tau > 0$.  The expected number of continuous structures $N$ is approximated by
dividing the total image height $M$ by the estimated period $T$.

### 3. Smoothing and Regularization

Detected light lines are approximated with **B-splines**
(`scipy.interpolate.UnivariateSpline`) to suppress pixel-level jitter.  Dark
boundaries are then re-generated at fixed, user-configurable distances
(`dist_up` / `dist_down`) above and below each smoothed light line.  When these
distances are not provided explicitly, they are estimated automatically from the
median offset between detected light and dark lines.

### 4. Spectral Band Extraction

For each column of the image, the intensity values between the upper and lower
dark boundaries of each band are extracted into a 3-D cube of shape
`(num_bands, image_width, spectral_width)`.

### 5. Spectral Pixel Allocation

Each extracted spectral band is placed at the row position of its corresponding
light line in the output array, producing the final reconstructed spectral image
of shape `(image_height, image_width, spectral_width)`.

### 6. Batch Accumulation

When multiple input images are processed, the resulting spectral images are
stacked and combined pixel-wise using one of three strategies:

| Method   | Description |
|----------|-------------|
| `mean`   | Arithmetic mean across frames (default) |
| `sum`    | Sum of all frames |
| `median` | Pixel-wise median across frames |

---

## Installation

The project requires Python ≥ 3.9 and the following packages:

- `numpy`
- `scipy`
- `scikit-image`

Install them via conda or pip:

```bash
conda install numpy scipy scikit-image
# or
pip install numpy scipy scikit-image
```

No additional installation step is needed — `batch.py` is a standalone module.

---

## Quick Start

### Library usage

```python
from batch import SpectralAnalyzer, reconstruct_spectral_image

# --- Single image ---
analyzer = SpectralAnalyzer(
    crop=(500, 2500, 1000, 2500),
    dist_up=30,
    dist_down=50,
)
spectral_img = analyzer.process_single("data/SD_QD_mix/image_0023.tiff")
print(spectral_img.shape)  # (2000, 1500, 80)

# --- Batch (multiple images) ---
import glob

paths = sorted(glob.glob("data/SD_QD_mix/*.tiff"))
spectral_img = reconstruct_spectral_image(
    paths,
    crop=(500, 2500, 1000, 2500),
    dist_up=30,
    dist_down=50,
    method="mean",
)
```

### Command-line usage

```bash
python batch.py data/SD_QD_mix/*.tiff \
    --crop 500 2500 1000 2500 \
    --dist-up 30 --dist-down 50 \
    --method mean \
    -o spectral_image.npy
```

---

## API Reference

### Class `SpectralAnalyzer`

```python
class SpectralAnalyzer(
    crop: tuple[int, int, int, int] | None = None,
    mask_width: int = 20,
    smooth_factor: float = 1e5,
    dist_up: float | None = None,
    dist_down: float | None = None,
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crop` | `tuple(row_start, row_end, col_start, col_end)` or `None` | `None` | ROI crop applied to every loaded image. `None` disables cropping. |
| `mask_width` | `int` | `20` | Half-width (in pixels) of the depletion mask applied after each DP line detection to prevent re-detection of the same structure. |
| `smooth_factor` | `float` | `1e5` | Smoothing parameter `s` passed to `scipy.interpolate.UnivariateSpline` when fitting B-splines to detected light lines. |
| `dist_up` | `float` or `None` | `None` | Fixed distance (pixels) from each light line upward to its dark boundary. `None` enables automatic estimation. |
| `dist_down` | `float` or `None` | `None` | Fixed distance (pixels) from each light line downward to its dark boundary. `None` enables automatic estimation. |

#### Methods

##### `load_image(path: str) -> np.ndarray`

Load an image from disk and apply the optional ROI crop.

- **path** — filesystem path to any image format supported by `skimage.io.imread`.
- **Returns** — 2-D `ndarray`.

##### `compute_edges(image: np.ndarray) -> np.ndarray`

Apply a Prewitt filter along axis 0 (vertical edge enhancement).

- **image** — 2-D grayscale image.
- **Returns** — filtered image of the same shape.

##### `detect_structures(edge_image: np.ndarray) -> dict`

Run DP-based periodic structure detection.

- **edge_image** — edge-filtered image (output of `compute_edges`).
- **Returns** — `{'light': ndarray, 'dark': ndarray}` where each value has shape `(num_lines, cols)`.

##### `regularize_structures(detected_lines: dict) -> dict`

Smooth light lines with B-splines and generate regularized dark boundaries.

- **detected_lines** — output of `detect_structures`.
- **Returns** — `{'light': ndarray, 'dark_up': ndarray, 'dark_down': ndarray, 'params': {'dist_up': float, 'dist_down': float}}`.

##### `extract_spectral_bands(image, regularized_lines) -> np.ndarray`

Extract spectral bands from the raw image using regularized boundaries.

- **image** — original 2-D image (before edge filtering).
- **regularized_lines** — output of `regularize_structures`.
- **Returns** — 3-D array `(num_bands, image_width, spectral_width)`.

##### `allocate_spectral_pixels(image, regularized_lines, spectral_bands) -> np.ndarray`

Place spectral bands at correct spatial rows to form the final spectral image.

- **image** — original 2-D image (used only for shape).
- **regularized_lines** — output of `regularize_structures`.
- **spectral_bands** — output of `extract_spectral_bands`.
- **Returns** — 3-D array `(image_height, image_width, spectral_width)`.

##### `process_single(image_path: str) -> np.ndarray`

Run the full pipeline on a single image.

- **image_path** — path to the input image.
- **Returns** — reconstructed spectral image `(H, W, S)`.

##### `process_batch(image_paths: list[str], method: str = 'mean') -> np.ndarray`

Process multiple images and accumulate them into a single spectral image.

- **image_paths** — list of file paths.
- **method** — `'mean'`, `'sum'`, or `'median'`.
- **Returns** — accumulated spectral image `(H, W, S)`.

---

### Function `reconstruct_spectral_image`

```python
def reconstruct_spectral_image(
    image_paths: list[str],
    crop: tuple[int,int,int,int] | None = None,
    mask_width: int = 20,
    smooth_factor: float = 1e5,
    dist_up: float | None = None,
    dist_down: float | None = None,
    method: str = 'mean',
) -> np.ndarray
```

Convenience wrapper that creates a `SpectralAnalyzer` with the given parameters
and calls `process_batch`.  All parameters mirror the class constructor plus the
`method` parameter from `process_batch`.

**Returns** — accumulated spectral image of shape `(H, W, S)`.

---

## CLI Usage

```
usage: batch.py [-h] [--crop ROW0 ROW1 COL0 COL1]
                [--mask-width MASK_WIDTH] [--smooth-factor SMOOTH_FACTOR]
                [--dist-up DIST_UP] [--dist-down DIST_DOWN]
                [--method {mean,sum,median}] [-o OUTPUT]
                images [images ...]

Batch spectral image reconstruction from spinning-disk spectral microscopy data.

positional arguments:
  images                Input image file paths (shell globs are expanded).

options:
  -h, --help            show this help message and exit
  --crop ROW0 ROW1 COL0 COL1
                        ROI crop: row_start row_end col_start col_end
  --mask-width MASK_WIDTH
                        DP depletion mask half-width (default: 20)
  --smooth-factor SMOOTH_FACTOR
                        B-spline smoothing factor (default: 1e5)
  --dist-up DIST_UP     Fixed upward boundary distance (default: auto)
  --dist-down DIST_DOWN
                        Fixed downward boundary distance (default: auto)
  --method {mean,sum,median}
                        Accumulation method across images (default: mean)
  -o, --output OUTPUT   Output .npy file (default: spectral_image.npy)
```

### Examples

Process all images in a directory, average them, and save the result:

```bash
python batch.py data/SD_QD_mix/*.tiff \
    --crop 500 2500 1000 2500 \
    --dist-up 30 --dist-down 50 \
    --method mean \
    -o result.npy
```

Process a specific subset of images:

```bash
python batch.py data/SD_QD_mix/image_0001.tiff data/SD_QD_mix/image_0002.tiff \
    --crop 500 2500 1000 2500 \
    --method median \
    -o median_result.npy
```

---

## Project Structure

```
spectra-spin/
├── batch.py          # Production script — SpectralAnalyzer class & CLI
├── demo.ipynb        # Interactive prototype / exploratory notebook
├── data/
│
└── README.md         # This file
```

---

## AI Disclaimer

The initial algorithm prototype was developed interactively in the Jupyter
notebook `demo.ipynb`.  The refactoring of this exploratory code into the
structured, class-based production script `batch.py` — including the design of
the `SpectralAnalyzer` class API, documentation, type annotations, CLI interface,
and batch accumulation logic — was performed with the assistance of **AI-based
coding tools**.  All algorithmic logic faithfully reproduces the original notebook
implementation; no modifications to the core numerical procedures were introduced
during the refactoring process.

