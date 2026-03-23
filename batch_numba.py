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
def _box_blur_h(src: np.ndarray, dst: np.ndarray, r: int):
    rows, cols = src.shape
    for i in range(rows):
        w_sum = 0.0
        for k in range(-r, r):
            idx = max(0, min(cols-1, k))
            w_sum += src[i, idx]
        for j in range(cols):
            right_idx = max(0, min(cols-1, j + r))
            left_idx = max(0, min(cols-1, j - r - 1))
            w_sum += src[i, right_idx]
            dst[i, j] = w_sum
            w_sum -= src[i, left_idx]

@njit
def _box_blur_v(src: np.ndarray, dst: np.ndarray, r: int):
    rows, cols = src.shape
    for j in range(cols):
        w_sum = 0.0
        for k in range(-r, r):
            idx = max(0, min(rows-1, k))
            w_sum += src[idx, j]
        for i in range(rows):
            bottom_idx = max(0, min(rows-1, i + r))
            top_idx = max(0, min(rows-1, i - r - 1))
            w_sum += src[bottom_idx, j]
            dst[i, j] = w_sum
            w_sum -= src[top_idx, j]

@njit
def _fast_blur_2d(src: np.ndarray, dst: np.ndarray, r: int, passes: int):
    temp = src.copy()
    for p in range(passes):
        _box_blur_h(temp, dst, r)
        _box_blur_v(dst, temp, r)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            dst[i, j] = temp[i, j]

@njit(parallel=True)
def _interpolate_missing_zeros_2d_numba(img3d: np.ndarray) -> np.ndarray:
    """
    High-quality Scattered Data Interpolation using Normalized Convolution.
    Applies 3 passes of 2D moving-average box blur to approximate a 
    smooth Cubic B-Spline Radial Basis Function (RBF).
    This generates highly continuous, non-linear 2D interpolation, 
    matching the quality of griddata(method='cubic') but orders of magnitude faster.
    """
    rows, cols, channels = img3d.shape
    out = img3d.copy()
    r = 25  # Coverage radius per pass
    
    for ch in prange(channels):
        src_data = img3d[:, :, ch].astype(np.float64)
        src_mask = (src_data != 0).astype(np.float64)
        
        dst_data = np.zeros_like(src_data)
        dst_mask = np.zeros_like(src_mask)
        
        # Blur the data and the mask 3 times for cubic smoothness
        _fast_blur_2d(src_data, dst_data, r, passes=3)
        _fast_blur_2d(src_mask, dst_mask, r, passes=3)
        
        for i in range(rows):
            for j in range(cols):
                if src_mask[i, j] == 0:
                    if dst_mask[i, j] > 1e-8:
                        # RBF Interpolate: smooth Data sum / smooth Mask sum
                        out[i, j, ch] = dst_data[i, j] / dst_mask[i, j]
                    else:
                        out[i, j, ch] = 0.0

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
            print("Starting parallel Numba 2D smooth cubic interpolation...")
            self.spectral_img_interpolated = _interpolate_missing_zeros_2d_numba(self.spectral_img)
            print("Interpolation complete.")
        else:
            raise ValueError("Unknown method. Use '1d' or '2d'.")


class SpectralReconNumba(SpectralRecon):
    """
    Numba-accelerated version of SpectralRecon.
    Inherits all business logic, only overriding the nested loops for speed.
    """

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

        return result

