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
  - [`SpectralAnalyzer` (batch.py)](#class-spectralanalyzer-batchpy)
  - [`SpectralAnalyzerOptim` (batch_optim.py)](#class-spectralanalyzeroptim-batch_optimpy)
  - [`SingleResult`](#dataclass-singleresult)
  - [`BatchResult`](#dataclass-batchresult)
  - [`reconstruct_spectral_image`](#function-reconstruct_spectral_image)
- [Project Structure](#project-structure)
- [AI Disclaimer](#ai-disclaimer)

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
5. Optionally combines multiple frames using **sum** or **max** accumulation
   to improve the signal-to-noise ratio.

Two implementations are provided:

| Module | Description |
|--------|-------------|
| `batch.py` | Original implementation. Loads all spectral images into memory before combining. |
| `batch_optim.py` | **Memory-optimized** implementation. Streaming accumulation (one image at a time), compact sparse representation, row-index metadata, and optional per-channel interpolation. |

The interactive prototype of the algorithm lives in `demo.ipynb`; the original
class-based implementation is in `batch.py`; the memory-optimized version is in
`batch_optim.py`.

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
combined pixel-wise using one of the following strategies:

| Method | Description |
|--------|-------------|
| `sum`  | Sum of all frames. |
| `max`  | Element-wise maximum across frames — prevents intensity doubling when different images share the same row index. |

> **Note:** The `max` method is recommended when row indices from different
> images overlap, as it avoids accumulation artefacts from duplicate summation.

---

## Installation

The project requires Python ≥ 3.9 and the following packages:

| Package | Required by |
|---------|-------------|
| `numpy` | `batch.py`, `batch_optim.py` |
| `scipy` | `batch.py`, `batch_optim.py` |
| `scikit-image` | `batch.py`, `batch_optim.py` |
| `pyyaml` | `batch_optim.py` (optional, for YAML metadata) |

Install them via conda or pip:

```bash
conda install numpy scipy scikit-image pyyaml
# or
pip install numpy scipy scikit-image pyyaml
```

No additional installation step is needed — both modules are standalone.

---

## Quick Start

### Using `batch.py` (original)

```python
from batch import SpectralAnalyzer, reconstruct_spectral_image
import glob

analyzer = SpectralAnalyzer(
    crop=(500, 2500, 1000, 2500),
    dist_up=30, dist_down=50,
)
spectral_img = analyzer.process_single("image_0001.tiff")

paths = sorted(glob.glob("data/SD_QD_mix/*.tiff"))
spectral_img = reconstruct_spectral_image(paths, crop=(500, 2500, 1000, 2500))
```

### Using `batch_optim.py` (memory-optimized)

```python
from batch_optim import SpectralAnalyzerOptim
import glob

analyzer = SpectralAnalyzerOptim(
    crop=(500, 2500, 1000, 2500),
    spectral_band_width=100,
    custom_lines_num=True,
    lines_num=22,
    dist_up=30, dist_down=50,
    interpolate_output=True,
)

# Single image → compact SingleResult
result = analyzer.process_single("image_0001.tiff")
print(result.spectral_bands.shape)  # (22, 1500, 100) — compact
print(result.row_indices)           # array of row positions

# Expand to full (H, W, S) if needed
full = analyzer.expand_to_full(
    result.image_shape, result.row_indices, result.spectral_bands,
)

# Batch processing with streaming accumulation
paths = sorted(glob.glob("data/QD_mix_40_phase/*.tiff"))
batch = analyzer.process_batch(paths, method="max")

batch.spectral_img                 # (H, W, S) accumulated image
batch.spectral_img_interpolated    # (H, W, S) interpolated image (or None)
batch.row_indices                  # {path: np.ndarray} metadata

# Save / load metadata
batch.save_metadata("row_indices.yaml")
batch.save_metadata("row_indices.json")
batch.save_metadata("row_indices.npz")
```

---

## API Reference

### Class `SpectralAnalyzer` (`batch.py`)

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
| `mask_width` | `int` | `20` | Half-width (in pixels) of the depletion mask applied after each DP line detection. |
| `smooth_factor` | `float` | `1e5` | Smoothing parameter `s` for `UnivariateSpline`. |
| `dist_up` | `float` or `None` | `None` | Fixed upward distance from light line to dark boundary. `None` = auto. |
| `dist_down` | `float` or `None` | `None` | Fixed downward distance from light line to dark boundary. `None` = auto. |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load_image(path)` | `ndarray` | Load image and apply crop. |
| `compute_edges(image)` | `ndarray` | Prewitt filter (axis 0). |
| `detect_structures(edge_image)` | `dict` | DP-based light/dark line detection. Returns `{'light': ..., 'dark': ...}`. |
| `regularize_structures(detected_lines)` | `dict` | B-spline smoothing + dark boundary generation. |
| `extract_spectral_bands(image, reg_lines)` | `ndarray` | Extract `(num_bands, W, S)` spectral cube. |
| `allocate_spectral_pixels(image, reg_lines, bands)` | `ndarray` | Place bands → full `(H, W, S)` array. |
| `process_single(image_path)` | `ndarray` | Full single-image pipeline → `(H, W, S)`. |
| `process_batch(image_paths, method)` | `ndarray` | Batch processing → accumulated `(H, W, S)`. |

---

### Class `SpectralAnalyzerOptim` (`batch_optim.py`)

Memory-optimized version with streaming accumulation, compact intermediate
representation, row-index metadata, and optional interpolation.

```python
class SpectralAnalyzerOptim(
    crop: tuple[int, int, int, int] | None = None,
    spectral_band_width: int = 100,
    custom_lines_num: bool = False,
    lines_num: int = 25,
    mask_width: int = 60,
    smooth_factor: float = 1e5,
    custom_dist: bool = True,
    dist_up: float | None = None,
    dist_down: float | None = None,
    interpolate_output: bool = False,
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `crop` | `tuple` or `None` | `None` | ROI crop `(row_start, row_end, col_start, col_end)`. |
| `spectral_band_width` | `int` | `100` | Target number of spectral channels per band. |
| `custom_lines_num` | `bool` | `False` | Use fixed `lines_num` instead of auto-estimation. |
| `lines_num` | `int` | `25` | Number of periodic structures (when `custom_lines_num=True`). |
| `mask_width` | `int` | `60` | Half-width of the DP depletion mask. |
| `smooth_factor` | `float` | `1e5` | B-spline smoothing parameter `s`. |
| `custom_dist` | `bool` | `True` | Use `dist_up`/`dist_down` values directly. |
| `dist_up` | `float` or `None` | `None` | Fixed upward distance from light line to dark boundary. |
| `dist_down` | `float` or `None` | `None` | Fixed downward distance from light line to dark boundary. |
| `interpolate_output` | `bool` | `False` | If `True`, produce interpolated spectral image (fills zero-valued pixels channel-by-channel). |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `load_image(path)` | `ndarray` | Load image and apply crop. |
| `detect_structures(edge_image)` | `dict` | DP-based light/dark line detection. |
| `regularize_structures(detected_lines)` | `dict` | B-spline smoothing + boundary generation. |
| `extract_spectral_bands(image, reg_lines, spectral_width)` | `ndarray` | Extract `(num_bands, W, S)` spectral cube. |
| `allocate_spectral_pixels(image, reg_lines, bands)` | `(row_indices, used_bands)` | **Compact** allocation — returns row indices + bands, no full array. |
| `expand_to_full(image_shape, row_indices, bands)` | `ndarray` | Expand compact representation to full `(H, W, S)`. |
| `interpolate_missing_zeros(image, method)` | `ndarray` | Fill zero pixels in a 2-D array via `griddata`. |
| `process_single(image_path)` | `SingleResult` | Full pipeline → compact result (no `(H, W, S)` allocation). |
| `process_batch(image_paths, method)` | `BatchResult` | Streaming batch → accumulated image + metadata. |

---

### Dataclass `SingleResult`

Returned by `SpectralAnalyzerOptim.process_single()`.

| Field | Type | Description |
|-------|------|-------------|
| `spectral_bands` | `ndarray` | Compact spectral data, shape `(num_bands, W, S)`. |
| `row_indices` | `ndarray` | Row positions in the image for each band, shape `(num_bands,)`. |
| `image_shape` | `tuple[int, int]` | `(height, width)` of the source image (after cropping). |
| `spectral_width` | `int` | Number of spectral channels. |
| `regularized_lines` | `dict` | Full regularized-line dictionary. |

---

### Dataclass `BatchResult`

Returned by `SpectralAnalyzerOptim.process_batch()`.

| Field | Type | Description |
|-------|------|-------------|
| `spectral_img` | `ndarray` | Accumulated spectral image `(H, W, S)`, float32. |
| `spectral_img_interpolated` | `ndarray` or `None` | Interpolated version (zero-pixels filled); `None` if `interpolate_output=False`. |
| `row_indices` | `dict[str, ndarray]` | Mapping `image_path → row-index array` for every image. |
| `num_images` | `int` | Number of processed images. |
| `spectral_width` | `int` | Spectral band width used. |
| `accumulation_method` | `str` | `'sum'` or `'max'`. |

#### Methods

| Method | Description |
|--------|-------------|
| `save_metadata(path)` | Save row-index metadata to `.npz`, `.json`, or `.yaml`/`.yml`. |
| `BatchResult.load_metadata(path)` | **Static.** Load metadata back into `dict[str, ndarray]`. |

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

Convenience wrapper from `batch.py` that creates a `SpectralAnalyzer` and calls
`process_batch`.  **Returns** — accumulated spectral image `(H, W, S)`.

---

## Project Structure

```
spectra-spin/
├── batch.py           # Original module — SpectralAnalyzer class & API
├── batch_optim.py     # Memory-optimized module — SpectralAnalyzerOptim
├── demo.ipynb         # Interactive algorithm prototype
├── batch_demo.ipynb   # Batch processing demo & benchmarks (batch.py)
├── optim_demo.ipynb   # Optimized module demo & benchmarks (batch_optim.py)
├── data/
│   ├── QD_mix_40_phase/   # Sample dataset (40 TIFF images)
│   └── SD_QD_mix/         # Additional sample dataset
└── README.md          # This file
```

---

## AI Disclaimer

The initial algorithm prototype was developed interactively in the Jupyter
notebook `demo.ipynb`.  The refactoring of this exploratory code into the
structured, class-based production module `batch.py` — including the design of
the `SpectralAnalyzer` class API, documentation, type annotations,
and batch accumulation logic — was performed with the assistance of **AI-based
coding tools**.  The subsequent memory-optimized implementation
(`batch_optim.py`), including the `SpectralAnalyzerOptim` class, streaming
accumulation, compact sparse representation, metadata serialization, and
interpolation support, was also developed with AI assistance.  All algorithmic
logic faithfully reproduces the original notebook implementation; no
modifications to the core numerical procedures were introduced during the
refactoring process.
