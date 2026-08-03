"""
High-performance batch spectral image reconstruction using Numba.

This module provides ``SpectralReconNumba``, which is a drop-in replacement
for ``SpectralRecon`` from ``batch_recon.py``. It reimplements the heavy
computational bottlenecks (Dynamic Programming path extraction, spectral band
extraction, and sparse-to-dense expansion) using Numba JIT compilation
and multi-threading.
"""

import numpy as np
from numba import njit, prange
import os
import gc
from dataclasses import dataclass

from batch_recon import SpectralRecon, SingleResult, BatchResult



@njit
def _extract_paths_numba(img_work: np.ndarray, num_lines: int, is_dark: bool, mask_width: int):
    rows, cols = img_work.shape
    lines = np.zeros((num_lines, cols), dtype=np.int32)
    
    mask_val = np.max(img_work) + 1000.0 if is_dark else np.min(img_work) - 1000.0
    pad_val = np.inf if is_dark else -np.inf
    
    for line_idx in range(num_lines):
        acc = np.zeros_like(img_work)
        for i in range(rows):
            acc[i, 0] = img_work[i, 0]
            
        ptr = np.zeros((rows, cols), dtype=np.int32)
        
        for j in range(1, cols):
            for i in range(rows):
                v_up = acc[i-1, j-1] if i > 0 else pad_val
                v_st = acc[i, j-1]
                v_dn = acc[i+1, j-1] if i < rows - 1 else pad_val
                
                if is_dark:
                    best = v_up
                    best_ptr = -1
                    if v_st < best:
                        best = v_st
                        best_ptr = 0
                    if v_dn < best:
                        best = v_dn
                        best_ptr = 1
                else:
                    best = v_up
                    best_ptr = -1
                    if v_st > best:
                        best = v_st
                        best_ptr = 0
                    if v_dn > best:
                        best = v_dn
                        best_ptr = 1
                        
                acc[i, j] = img_work[i, j] + best
                ptr[i, j] = best_ptr
                
        path = np.zeros(cols, dtype=np.int32)
        
        best_idx = 0
        best_val = acc[0, cols-1]
        if is_dark:
            for i in range(1, rows):
                if acc[i, cols-1] < best_val:
                    best_val = acc[i, cols-1]
                    best_idx = i
        else:
            for i in range(1, rows):
                if acc[i, cols-1] > best_val:
                    best_val = acc[i, cols-1]
                    best_idx = i
        path[cols-1] = best_idx
        
        for j in range(cols-1, 0, -1):
            cur = path[j]
            nxt = cur + ptr[cur, j]
            if nxt < 0: nxt = 0
            if nxt >= rows: nxt = rows - 1
            path[j-1] = nxt
            
        lines[line_idx] = path
        
        # Mask found line
        for j in range(cols):
            r = path[j]
            r_lo = max(0, r - mask_width)
            r_hi = min(rows, r + mask_width + 1)
            for i in range(r_lo, r_hi):
                img_work[i, j] = mask_val

    return lines


@njit(parallel=True)
def _extract_bands_numba(image: np.ndarray, up_lim: np.ndarray, down_lim: np.ndarray, spectral_width: int):
    num_cols, num_bands = up_lim.shape
    lambda_img = np.zeros((num_bands, num_cols, spectral_width), dtype=np.float32)

    for band in prange(num_bands):
        for col in range(num_cols):
            d = down_lim[col, band]
            u = up_lim[col, band]
            h = u - d
            if h <= 0:
                continue
                
            limit = min(h, spectral_width)
            for k in range(limit):
                lambda_img[band, col, k] = image[d + k, col]
                
    return lambda_img


@njit(parallel=True)
def _expand_full_1d_numba(h: int, w: int, s: int, row_indices: np.ndarray, spectral_bands: np.ndarray):
    out = np.zeros((h, w, s), dtype=np.float32)
    num_bands = row_indices.shape[0]
    for i in prange(num_bands):
        r = row_indices[i]
        if 0 <= r < h:
            for col in range(w):
                for k in range(s):
                    out[r, col, k] = spectral_bands[i, col, k]
    return out


@njit(parallel=True)
def _expand_full_2d_numba(h: int, w: int, s: int, row_indices: np.ndarray, spectral_bands: np.ndarray):
    out = np.zeros((h, w, s), dtype=np.float32)
    num_bands, cols = row_indices.shape
    for i in prange(num_bands):
        for col in range(min(cols, w)):
            r = row_indices[i, col]
            if 0 <= r < h:
                for k in range(s):
                    out[r, col, k] = spectral_bands[i, col, k]
    return out


@njit(parallel=True)
def _interpolate_missing_zeros_numba(img3d: np.ndarray) -> np.ndarray:
    rows, cols, channels = img3d.shape
    out = np.zeros_like(img3d)
    x = np.arange(rows).astype(np.float64)
    
    for ch in prange(channels):
        for c in range(cols):
            profile = img3d[:, c, ch]
            nnz = 0
            for r in range(rows):
                if profile[r] != 0:
                    nnz += 1
                    
            if nnz == 0:
                continue
            elif nnz == 1:
                val = 0.0
                for r in range(rows):
                    if profile[r] != 0:
                        val = profile[r]
                        break
                for r in range(rows):
                    out[r, c, ch] = val
            else:
                xp = np.zeros(nnz, dtype=np.float64)
                fp = np.zeros(nnz, dtype=img3d.dtype)
                idx = 0
                for r in range(rows):
                    if profile[r] != 0:
                        xp[idx] = r
                        fp[idx] = profile[r]
                        idx += 1
                        
                interp = np.interp(x, xp, fp)
                for r in range(rows):
                    out[r, c, ch] = interp[r]
                    
    return out


@njit
def _interp_1d_fill(profile: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation over non-zero values in a 1D profile.
    Returns the interpolated profile (same length), or zeros if no data."""
    n = profile.shape[0]
    nnz = 0
    for i in range(n):
        if profile[i] != 0:
            nnz += 1
    if nnz == 0:
        return np.zeros(n, dtype=np.float64)
    if nnz == 1:
        val = 0.0
        for i in range(n):
            if profile[i] != 0:
                val = profile[i]
                break
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = val
        return out

    xp = np.zeros(nnz, dtype=np.float64)
    fp = np.zeros(nnz, dtype=np.float64)
    idx = 0
    for i in range(n):
        if profile[i] != 0:
            xp[idx] = i
            fp[idx] = profile[i]
            idx += 1
    return np.interp(x, xp, fp)


@njit(parallel=True)
def _interpolate_missing_zeros_2d_numba(img3d: np.ndarray) -> np.ndarray:
    """Two-pass linear 2D interpolation for missing (zero-valued) pixels.

    Pass 1 — interpolate along rows (axis 1) for each (row, channel).
    Pass 2 — interpolate along columns (axis 0) for each (col, channel).
    The final value is the average of the two passes where both provide
    data; otherwise the available pass is used directly.
    Zero pixels that cannot be interpolated by either pass stay zero.
    """
    rows, cols, channels = img3d.shape
    out = img3d.copy()
    x_row = np.arange(cols).astype(np.float64)
    x_col = np.arange(rows).astype(np.float64)

    for ch in prange(channels):
        # --- Pass 1: interpolate along rows (horizontal) ---
        row_interp = np.zeros((rows, cols), dtype=np.float64)
        row_valid = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            profile = np.zeros(cols, dtype=np.float64)
            nnz = 0
            for c in range(cols):
                profile[c] = img3d[r, c, ch]
                if profile[c] != 0:
                    nnz += 1
            if nnz > 0:
                filled = _interp_1d_fill(profile, x_row)
                for c in range(cols):
                    row_interp[r, c] = filled[c]
                    row_valid[r, c] = 1.0

        # --- Pass 2: interpolate along columns (vertical) ---
        col_interp = np.zeros((rows, cols), dtype=np.float64)
        col_valid = np.zeros((rows, cols), dtype=np.float64)
        for c in range(cols):
            profile = np.zeros(rows, dtype=np.float64)
            nnz = 0
            for r in range(rows):
                profile[r] = img3d[r, c, ch]
                if profile[r] != 0:
                    nnz += 1
            if nnz > 0:
                filled = _interp_1d_fill(profile, x_col)
                for r in range(rows):
                    col_interp[r, c] = filled[r]
                    col_valid[r, c] = 1.0

        # --- Combine: average when both passes valid ---
        for r in range(rows):
            for c in range(cols):
                if img3d[r, c, ch] != 0:
                    continue  # original data — already in out
                rv = row_valid[r, c]
                cv = col_valid[r, c]
                if rv > 0 and cv > 0:
                    out[r, c, ch] = (row_interp[r, c] + col_interp[r, c]) / 2.0
                elif rv > 0:
                    out[r, c, ch] = row_interp[r, c]
                elif cv > 0:
                    out[r, c, ch] = col_interp[r, c]
                # else: stays 0

    return out


@njit(parallel=True)
def _binning_2d_numba(img3d: np.ndarray, bin_size: int) -> np.ndarray:
    """Down-sample a 3D image (H, W, S) by averaging non-zero pixels
    in each (bin_size x bin_size) spatial block.

    Parameters
    ----------
    img3d : np.ndarray
        Input image of shape (rows, cols, channels), dtype float32/64.
    bin_size : int
        Side length of the square binning block (must be >= 1).

    Returns
    -------
    np.ndarray
        Down-sampled image of shape (rows // bin_size, cols // bin_size, channels).
    """
    rows, cols, channels = img3d.shape
    out_rows = rows // bin_size
    out_cols = cols // bin_size
    out = np.zeros((out_rows, out_cols, channels), dtype=img3d.dtype)

    for ch in prange(channels):
        for br in range(out_rows):
            r_start = br * bin_size
            r_end = r_start + bin_size
            for bc in range(out_cols):
                c_start = bc * bin_size
                c_end = c_start + bin_size
                total = 0.0
                count = 0
                for r in range(r_start, r_end):
                    for c in range(c_start, c_end):
                        val = img3d[r, c, ch]
                        if val != 0.0:
                            total += val
                            count += 1
                if count > 0:
                    out[br, bc, ch] = total / count
    return out

@dataclass
class BatchResultNumba(BatchResult):
    def interpolate_missing_zeros(self, method: str = "2d"):
        if self.spectral_img is None:
            print("No spectral image to interpolate.")
            return

        if method == "1d":
            print("Starting parallel Numba 1D linear interpolation...")
            self.spectral_img_interpolated = _interpolate_missing_zeros_numba(self.spectral_img)
            print("Interpolation complete.")
        elif method == "2d":
            print("Starting parallel Numba 2D linear interpolation...")
            self.spectral_img_interpolated = _interpolate_missing_zeros_2d_numba(self.spectral_img)
            print("Interpolation complete.")
        else:
            raise ValueError("Unknown method. Use '1d' or '2d'.")

    def apply_output_binning(self, bin_size: int) -> None:
        """Apply spatial binning (down-sampling) to spectral images.

        Averages non-zero pixel intensities within each
        (bin_size × bin_size) block.  Updates ``spectral_img`` and,
        if present, ``spectral_img_interpolated`` in-place.

        Parameters
        ----------
        bin_size : int
            Side length of the binning block.  A value ≤ 1 is a no-op.
        """
        if bin_size <= 1:
            return
        if self.spectral_img is not None:
            print(f"Applying {bin_size}×{bin_size} output binning (ignoring zeros)...")
            self.spectral_img = _binning_2d_numba(self.spectral_img, bin_size)
        if self.spectral_img_interpolated is not None:
            print(f"Applying {bin_size}×{bin_size} binning to interpolated image...")
            self.spectral_img_interpolated = _binning_2d_numba(
                self.spectral_img_interpolated, bin_size)
        print("Binning complete.")


class SpectralReconNumba(SpectralRecon):
    """
    Numba-accelerated version of SpectralRecon.
    Inherits all business logic, only overriding the nested loops for speed.

    Parameters
    ----------
    output_binning : int
        Spatial down-sampling factor applied to the final batch result.
        A value of 0 or 1 means no binning.  For example,
        ``output_binning=2`` averages each 2×2 block of pixels (ignoring
        zeros) and produces an image half the original size in each spatial
        dimension.
    **kwargs
        All other keyword arguments are forwarded to ``SpectralRecon``.
    """

    def __init__(self, *, output_binning: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_binning = output_binning

    @staticmethod
    def _extract_paths(img_work: np.ndarray, num_lines: int, is_dark: bool, mask_width: int = 20) -> np.ndarray:
        lines = _extract_paths_numba(img_work, num_lines, is_dark, mask_width)
        return np.sort(lines, axis=0)

    @staticmethod
    def extract_spectral_bands(image: np.ndarray, regularized_lines: dict, spectral_width: int) -> np.ndarray:
        up_lim = regularized_lines["dark_up"].T
        down_lim = regularized_lines["dark_down"].T
        return _extract_bands_numba(image.astype(np.float32), up_lim, down_lim, spectral_width)

    @staticmethod
    def expand_to_full(image_shape: tuple[int, int], row_indices: np.ndarray, spectral_bands: np.ndarray) -> np.ndarray:
        h, w = image_shape
        s = spectral_bands.shape[2]
        if row_indices.ndim == 1:
            return _expand_full_1d_numba(h, w, s, row_indices, spectral_bands)
        else:
            return _expand_full_2d_numba(h, w, s, row_indices, spectral_bands)


    def process_single(self, image_path: str) -> "SingleResult":
        result = super().process_single(image_path)
        if self.output_binning > 1:
            print(f"Applying {self.output_binning}×{self.output_binning} output binning to single result...")
            result.spectral_img = _binning_2d_numba(result.spectral_img, self.output_binning)
            print("Binning complete.")
        return result

    def process_batch(self, image_paths: list[str], method: str = "max") -> BatchResult:
        if not image_paths:
            raise ValueError("image_paths must be a non-empty list")

        accumulator = None
        all_row_indices = {}
        ref_shape = None
        sw_values = []

        for idx, path in enumerate(image_paths, 1):
            print(f"[{idx}/{len(image_paths)}] Processing {os.path.basename(path)} ...")

            single = self.process_single(path)
            sw_values.append(single.spectral_width)
            all_row_indices[path] = single.row_indices.copy()

            if ref_shape is None:
                ref_shape = single.image_shape

            full_img = self.expand_to_full(single.image_shape, single.row_indices, single.spectral_bands)

            if accumulator is None:
                accumulator = full_img
            else:
                if method == "max":
                    np.maximum(accumulator, full_img, out=accumulator)
                else:
                    accumulator += full_img

            del single, full_img
            gc.collect()

        sw_arr = np.asarray(sw_values, dtype=int)
        print(f"Batch spectral width: min {np.min(sw_arr)}, max {np.max(sw_arr)}")

        result = BatchResultNumba(spectral_img=accumulator,
                                  spectral_img_interpolated=None,
                                  row_indices=all_row_indices,
                                  num_images=len(image_paths),
                                  spectral_width=int(sw_arr[0]),
                                  accumulation_method=method)

        # Apply output binning if configured
        if self.output_binning > 1:
            result.apply_output_binning(self.output_binning)

        return result

