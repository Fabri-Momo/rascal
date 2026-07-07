# -*- coding: utf-8 -*-
# Rascal – ZCA in linear space (sRGB -> Linear -> ZCA/Rotation -> Linear -> sRGB)
#
# The ZCA transform (mu_lin, W_lin, Winv_lin) is defined from a reference image:
# - initially from the loaded image,
# - optionally updated via ROI-based reprocessing.
#
# During interactive exploration (rotation, radial push), these parameters remain fixed,
# ensuring a consistent color space.
#
# They are recomputed only when the reference image is explicitly updated
# (e.g. ROI reprocessing or image promotion).
#
# The shader always applies transformations relative to this fixed reference.

import os
os.environ.setdefault("VISPY_GL_VERSION", "gl+es2")
os.environ.setdefault("VISPY_LOG_LEVEL", "warning")

import sys
import json
import base64
import pathlib
import ctypes
import tempfile
import platform
import gc
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

import imageio.v2 as imageio

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL import GL
from vispy import app, gloo, scene
from vispy.scene import visuals

IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

def _resource_path(name):
    """Return the absolute path of a bundled resource.

    Works both in a regular Python run and in a frozen PyInstaller bundle
    (where data files live under ``sys._MEIPASS``)."""
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(os.path.join(meipass, name))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, name))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return candidates[-1]

def _is_ctrl_qt(modifiers):
    """Returns True if the platform-appropriate 'Ctrl' modifier is active.
    On macOS, Cmd (MetaModifier) acts as Ctrl; on other platforms, ControlModifier."""
    if IS_MACOS:
        return bool(modifiers & QtCore.Qt.MetaModifier)
    return bool(modifiers & QtCore.Qt.ControlModifier)

def _is_ctrl_vispy(mods):
    """Same logic for vispy modifier sets (strings)."""
    if IS_MACOS:
        return 'Meta' in mods
    return 'Control' in mods

# ------------ Constants ------------
WIN_SIZE_2D = (720, 520)
COMPACT_SIZE_2D = (280, 180)
N_SAMPLES_3D = 100000
LUT_SIZE = 256
FIB_K = 256  # Number of Fibonacci generators
FIB_Q = 95  # Percentile for the radial envelope
FIB_M = 6    # Local neighbours for interpolation
FIB_SIGMA = 0.25  # Gaussian kernel width

# ------------ RAW image support (NEF, CR2, ARW, etc.) ------------
try:
    import rawpy
    _HAS_RAWPY = True
except ImportError:
    _HAS_RAWPY = False

_RAW_EXTENSIONS = frozenset((
    '.nef', '.cr2', '.cr3', '.arw', '.orf', '.raf', '.rw2', '.dng',
    '.pef', '.srw', '.x3f', '.3fr', '.mef', '.mos', '.mrw', '.nrw',
    '.raw', '.rwl', '.sr2', '.srf', '.kdc', '.dcr', '.erf',
))

def _is_raw_file(path):
    """Return True if *path* has a known camera-RAW extension."""
    return os.path.splitext(str(path))[1].lower() in _RAW_EXTENSIONS

def _read_raw_image(path):
    """Read a camera-RAW file via rawpy and return a normalised float32 RGB array [0,1].
    Orientation is handled by rawpy/LibRaw internally (flip parameter)."""
    with rawpy.imread(str(path)) as raw:
        # LibRaw applies the camera orientation automatically during postprocess
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=False,
            output_bps=16,
            user_flip=-1,  # -1 = use camera EXIF orientation (default LibRaw)
        )
    # rawpy output is always full-range uint16 (0-65535); convert directly
    return (rgb16.astype(np.float32) / 65535.0)

def _imread_any(path):
    """Read an image file, using rawpy for RAW formats and imageio otherwise."""
    if _HAS_RAWPY and _is_raw_file(path):
        return _read_raw_image(path)
    return imageio.imread(pathlib.Path(path))

# ------------ Memory monitoring ------------
_MPIX_WARN_THRESHOLD = 30  # Warn when image exceeds this many megapixels
_MEM_HIGH_PERCENT = 80     # Warn when process memory exceeds this % of total RAM

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

def _get_process_memory_mb():
    """Return current process RSS in MB. Returns None if psutil is unavailable."""
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None

def _get_total_ram_mb():
    """Return total system RAM in MB. Returns None if psutil is unavailable."""
    if not _HAS_PSUTIL:
        return None
    try:
        return psutil.virtual_memory().total / (1024 * 1024)
    except Exception:
        return None

def _get_memory_percent():
    """Return process memory as a percentage of total RAM. Returns None if unavailable."""
    proc = _get_process_memory_mb()
    total = _get_total_ram_mb()
    if proc is not None and total is not None and total > 0:
        return (proc / total) * 100.0
    return None

def _estimate_image_memory_mb(h, w, n_presets=0):
    """Estimate the memory footprint in MB for an image of size (h, w).
    Base: ~6 float32 copies + brush_alpha + samples.
    Each preset adds ~8 full-image copies."""
    px = h * w
    bytes_per_copy = px * 3 * 4  # float32 RGB
    base_copies = 7  # img_file_orig, _img_file_initial, _img_pre_roi, img_original_view, img_roi_base, img_srgb_orig, img_srgb_view
    brush_bytes = px * 1 * 4
    preset_copies_per = 8  # img_srgb_orig, img_file_orig, etc.
    total = (base_copies * bytes_per_copy + brush_bytes +
             n_presets * preset_copies_per * bytes_per_copy)
    return total / (1024 * 1024)

def _format_memory_mb(mb):
    """Format MB into a human-readable string."""
    if mb is None:
        return "N/A"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"

# ------------ Fibonacci radial normalisation ------------
def fibonacci_sphere(K):
    """Generates K uniformly distributed points on a unit sphere via Fibonacci."""
    golden = (1 + 5**0.5) / 2
    k = np.arange(K)
    z = 1 - 2*(k + 0.5)/K
    phi = 2*np.pi*k/golden
    r = np.sqrt(1 - z*z)
    G = np.stack([r*np.cos(phi), r*np.sin(phi), z], axis=1)
    return G.astype(np.float32)

def _fib_process_batch(args):
    """Worker function to process a batch of points in parallel.
    Fully vectorized for optimal performance."""
    batch_U, batch_norms, G, r_env, m, sigma = args

    dots_batch = batch_U @ G.T
    idx = np.argpartition(-dots_batch, m - 1, axis=1)[:, :m]

    d_vals = np.take_along_axis(dots_batch, idx, axis=1)
    w = np.exp((d_vals - 1.0) / (sigma * sigma))

    r_vals = r_env[idx]
    r_local = np.sum(w * r_vals, axis=1) / (np.sum(w, axis=1) + 1e-8)

    norm_new = np.minimum(batch_norms / (r_local + 1e-8), 1.0)
    return (norm_new[:, None] * batch_U).astype(np.float32)

def _interpolate_empty_sectors(r_env, G):
    """Vectorized interpolation for empty sectors."""
    empty = r_env == 0
    if not empty.any() or not (~empty).any():
        return r_env

    filled_idx = np.where(~empty)[0]
    empty_idx = np.where(empty)[0]

    # Compute dot products between empty sectors and filled sectors
    dots = G[empty_idx] @ G[filled_idx].T  # shape: (n_empty, n_filled)
    nearest = filled_idx[np.argmax(dots, axis=1)]
    r_env[empty_idx] = r_env[nearest]
    return r_env

def radial_normalisation_fibonacci(Zs, K=FIB_K, q=FIB_Q, m=FIB_M, sigma=FIB_SIGMA,
                                    progress_cb=None, paint_mask=None, cancel_event=None,
                                    n_workers=None):
    """Radially normalises points according to Fibonacci Voronoi sectors.
    If paint_mask (bool 1-D, same length as Zs) is provided, the Q-percentile
    of each sector is computed only on those pixels; normalisation
    is nevertheless applied to all points.

    Uses ThreadPoolExecutor instead of multiprocessing to avoid serialization
    overhead and reduce memory usage.
    """
    def _cb(val, msg):
        if progress_cb:
            progress_cb(val, msg)

    _cb(5, "Computing norms and directions…")
    G = fibonacci_sphere(K)

    # Compute norms and directions in-place to save memory
    norms = np.linalg.norm(Zs, axis=1)
    U = Zs.copy()
    np.divide(Zs, norms[:, None] + 1e-8, out=U)

    _cb(15, "Assigning to Voronoi sectors…")
    # Limit chunk size for large images to keep per-chunk dot-product allocation small
    if U.shape[0] > 10_000_000:
        CHUNK = 25_000
    elif U.shape[0] > 5_000_000:
        CHUNK = 50_000
    else:
        CHUNK = 100_000
    labels = np.empty(U.shape[0], dtype=np.int32)
    for start in range(0, U.shape[0], CHUNK):
        end = min(start + CHUNK, U.shape[0])
        # Use np.dot for potentially better cache utilization
        dots_chunk = np.dot(U[start:end], G.T)
        labels[start:end] = np.argmax(dots_chunk, axis=1)
        del dots_chunk  # Free memory immediately

    # Pixels used to estimate the radial envelope
    if paint_mask is not None and paint_mask.sum() >= 10:
        ref_norms = norms[paint_mask]
        ref_labels = labels[paint_mask]
    else:
        ref_norms = norms
        ref_labels = labels

    _cb(25, "Computing radial envelope…")
    r_env = np.zeros(K, dtype=np.float32)

    # Vectorized percentile computation using np.bincount for speed
    unique_labels = np.unique(ref_labels)
    for k in unique_labels:
        mask = ref_labels == k
        r_env[k] = np.percentile(ref_norms[mask], q, method='linear')

    # Vectorized interpolation for empty sectors
    r_env = _interpolate_empty_sectors(r_env, G)
    r_env[r_env == 0] = 1e-6

    _cb(35, "Local interpolation (parallel computation)…")
    N = Zs.shape[0]

    # Scale down workers and batch sizes for large images to avoid OOM crash.
    # Each batch allocates (batch_size × K × 4) bytes for dots_batch inside
    # _fib_process_batch, so keeping batches small is critical on big images.
    if n_workers is None:
        if N > 10_000_000:
            n_workers = 1
            batch_size = 25_000
        elif N > 5_000_000:
            n_workers = 2
            batch_size = 50_000
        else:
            n_workers = min(4, multiprocessing.cpu_count())
            batch_size = max(1000, N // (n_workers * 4))

    # Pre-allocate result array and prepare batch metadata
    Zs_norm = np.empty_like(Zs)
    batch_metadata = []
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_metadata.append((start_idx, end_idx))

    # Use ThreadPoolExecutor instead of multiprocessing.Pool
    # - No serialization overhead (shares memory)
    # - Lower memory footprint
    def process_single_batch(batch_info):
        start_idx, end_idx = batch_info
        batch_U = U[start_idx:end_idx]
        batch_norms = norms[start_idx:end_idx]
        result = _fib_process_batch((batch_U, batch_norms, G, r_env, m, sigma))
        return start_idx, result

    n_batches = len(batch_metadata)
    completed = 0

    # ThreadPoolExecutor with limited queue size to control memory
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_batch, bm): bm for bm in batch_metadata}

        # Collect results as they complete
        for future in as_completed(futures):
            if cancel_event is not None and cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                raise InterruptedError("Cancelled by user")

            start_idx, batch_result = future.result()
            end_idx = start_idx + batch_result.shape[0]
            Zs_norm[start_idx:end_idx] = batch_result
            completed += 1
            pct = 35 + int(55 * completed / n_batches)
            _cb(pct, "Local interpolation (parallel computation)…")

    _cb(95, "Finalizing…")
    r_global = float(np.median(r_env))

    # Cleanup to help GC
    del U, norms, labels

    return Zs_norm, r_global


# ------------ Fibonacci worker thread ------------
class FibNormWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)   # (value 0-100, message)
    finished = QtCore.pyqtSignal(object, object)  # (normalised image or None, Zs_norm or None)
    error    = QtCore.pyqtSignal(str)

    def __init__(self, img_srgb_orig, mu_lin, W_lin, Winv_lin, paint_mask=None, push_factor=1.0, parent=None):
        super().__init__(parent)
        self.img_srgb_orig = img_srgb_orig
        self.mu_lin   = mu_lin
        self.W_lin    = W_lin
        self.Winv_lin = Winv_lin
        self.paint_mask = paint_mask
        self.push_factor = push_factor  # stored for reference only; NOT applied in run() — the shader applies it in real-time via u_push
        self._cancel_event = multiprocessing.Event()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        try:
            img_lin = ColorUtils.srgb_to_linear_np(np.clip(self.img_srgb_orig, 0, 1))
            Xl = img_lin.reshape(-1, 3).astype(np.float32)
            del img_lin
            # If a ROI mask is provided, recompute ZCA on painted pixels only
            # (same logic as reprocess()) so the radial push uses local statistics.
            if self.paint_mask is not None and self.paint_mask.sum() >= 10:
                mu_loc, W_loc, Winv_loc = ColorUtils.zca_from_data(Xl[self.paint_mask])
            else:
                mu_loc  = self.mu_lin
                W_loc   = self.W_lin
                Winv_loc = self.Winv_lin
            Zs = (Xl - mu_loc) @ W_loc.T
            del Xl
            Zs_norm, r_global = radial_normalisation_fibonacci(
                Zs,
                progress_cb=lambda v, m: self.progress.emit(v, m),
                paint_mask=self.paint_mask,
                cancel_event=self._cancel_event
            )
            Xl_norm = np.clip(mu_loc + ((Zs_norm * r_global) @ Winv_loc.T), 0.0, 1.0)
            img_norm = ColorUtils.linear_to_srgb_np(Xl_norm).reshape(self.img_srgb_orig.shape)
            self.finished.emit(img_norm.astype(np.float32), Zs_norm.astype(np.float32))
        except InterruptedError:
            self.finished.emit(None, None)
        except MemoryError:
            self.error.emit(
                "Not enough memory to process this image.\n\n"
                "Suggestions:\n"
                "  • Close other applications to free RAM.\n"
                "  • Reduce the image size via Modify Image before using Radial Push."
            )
            self.finished.emit(None, None)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(None, None)

# ------------ GLSL ------------
VERT = """
#ifdef GL_ES
precision highp float;
precision highp int;
#endif
attribute vec2 a_position;
attribute vec2 a_texcoord;
uniform float u_zoom_scale;
uniform vec2 u_zoom_offset;
uniform vec2 u_aspect_correction;
varying vec2 v_texcoord;
void main() {
    vec2 pos = a_position * u_zoom_scale + u_zoom_offset;
    pos *= u_aspect_correction;
    gl_Position = vec4(pos, 0.0, 1.0);
    v_texcoord = vec2(a_texcoord.x, 1.0 - a_texcoord.y);
}
"""
FRAG = """
#ifdef GL_ES
precision highp float;
precision highp int;
#endif

uniform sampler2D u_tex_orig;
uniform sampler2D u_tex_file_orig;

uniform sampler2D u_lut_s2l;
uniform sampler2D u_lut_l2s;
uniform float u_lut_size;

uniform mat3 W_lin;
uniform mat3 Winv_lin;
uniform vec3 mu_lin;

uniform mat3 Ruser;
uniform float u_push;
uniform float u_var_restore;
uniform float u_contrast;
uniform float u_contrast_center;

uniform int u_mode;

uniform sampler2D u_brush;
uniform int u_use_brush;

uniform int u_split_active;
uniform float u_split_pos;
uniform float u_split_line_w;

varying vec2 v_texcoord;

vec3 lut_lookup_rgb(sampler2D lut, float size, vec3 x) {
    float s = 1.0 / max(size, 1.0);
    float u0 = 0.5 * s;
    float v0 = 0.5;
    float ur = mix(u0, 1.0 - u0, clamp(x.r, 0.0, 1.0));
    float ug = mix(u0, 1.0 - u0, clamp(x.g, 0.0, 1.0));
    float ub = mix(u0, 1.0 - u0, clamp(x.b, 0.0, 1.0));
    vec3 samp = texture2D(lut, vec2(ur, v0)).rgb;
    vec3 sampg = texture2D(lut, vec2(ug, v0)).rgb;
    vec3 sampb = texture2D(lut, vec2(ub, v0)).rgb;
    return vec3(samp.r, sampg.g, sampb.b);
}

void main() {
    vec3 xs_srgb_orig = texture2D(u_tex_orig, v_texcoord).rgb;
    vec3 xl0          = lut_lookup_rgb(u_lut_s2l, u_lut_size, xs_srgb_orig);

    vec3 z  = W_lin * (xl0 - mu_lin);
    vec3 z2 = Ruser * z * u_push;
    mat3 eff_Winv = mat3(1.0) + u_var_restore * (Winv_lin - mat3(1.0));
    vec3 xo = clamp(mu_lin + eff_Winv * z2, 0.0, 1.0);
    vec3 base_color = lut_lookup_rgb(u_lut_l2s, u_lut_size, xo);
    
    if (u_mode == 1) { float v = base_color.r; base_color = vec3(v); }
    else if (u_mode == 2) { float v = base_color.g; base_color = vec3(v); }
    else if (u_mode == 3) { float v = base_color.b; base_color = vec3(v); }
    else if (u_mode == 4) { base_color = texture2D(u_tex_file_orig, v_texcoord).rgb; }

    // Split view: left side shows original, right side shows transformed
    if (u_split_active == 1 && v_texcoord.x < u_split_pos) {
        base_color = texture2D(u_tex_file_orig, v_texcoord).rgb;
    }

    // Split line indicator (constant screen-pixel width)
    if (u_split_active == 1 && abs(v_texcoord.x - u_split_pos) < u_split_line_w) {
        base_color = vec3(1.0, 1.0, 1.0);
    }

    // Manual contrast adjustment (1.0 = no change), only on transformed views.
    // The left side of the split view (original) must stay unmodified.
    bool is_original = (u_mode == 4) || (u_split_active == 1 && v_texcoord.x < u_split_pos);
    if (!is_original) {
        base_color = clamp((base_color - u_contrast_center) * u_contrast + u_contrast_center, 0.0, 1.0);
    }

    if (u_use_brush == 1) {
        float a = texture2D(u_brush, v_texcoord).r;
        vec3 brushColor = vec3(1.0, 1.0, 0.0);
        base_color = mix(base_color, brushColor, clamp(a, 0.0, 0.5));
    }

    gl_FragColor = vec4(base_color, 1.0);
}
"""

# ------------ EXIF orientation utility ------------
def _apply_exif_orientation(img_array, path):
    """Rotate/flip a numpy image array according to EXIF Orientation tag.
    Returns the corrected array (or the original if no EXIF data)."""
    # RAW files: orientation already applied by rawpy/LibRaw during postprocess
    if _is_raw_file(path):
        return img_array
    try:
        from PIL import Image
        pil = Image.open(path)
        exif = pil.getexif()
        orientation = exif.get(274)  # 274 = Orientation tag
        if orientation is None or orientation == 1:
            return img_array
        if orientation == 2:
            return np.fliplr(img_array)
        elif orientation == 3:
            return np.rot90(img_array, 2)
        elif orientation == 4:
            return np.flipud(img_array)
        elif orientation == 5:
            return np.rot90(np.fliplr(img_array), 1)
        elif orientation == 6:
            return np.rot90(img_array, 3)
        elif orientation == 7:
            return np.rot90(np.flipud(img_array), 1)
        elif orientation == 8:
            return np.rot90(img_array, 1)
    except Exception:
        pass
    return img_array

def _normalize_image_array(arr):
    """Normalise a raw numpy image array to float32 [0, 1] RGB.

    Rules (applied in order, without mutating arr before dtype detection):
    - grayscale (2-D or single-channel) → replicate to 3 channels
    - keep only the first 3 channels
    - float arrays: divide by max only if max > 1 (HDR float); otherwise keep as-is
    - uint16 → divide by 65535
    - anything else (uint8, int, …) → divide by 255
    - final clip to [0, 1]
    """
    a = arr
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=2)
    elif a.shape[2] == 1:
        a = np.concatenate([a] * 3, axis=2)
    a = a[..., :3]
    if np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float32)
        m = float(np.nanmax(a))
        if m > 1.0:
            a = a / m
    elif a.dtype == np.uint16:
        mx = int(np.max(a))
        # Detect sub-16-bit data (e.g. 10-bit, 12-bit, 14-bit TIFF stored as uint16)
        if mx > 0 and mx <= 4095:
            a = a.astype(np.float32) / 4095.0    # 12-bit
        elif mx > 4095 and mx <= 16383:
            a = a.astype(np.float32) / 16383.0   # 14-bit
        else:
            a = a.astype(np.float32) / 65535.0    # true 16-bit
    else:
        a = a.astype(np.float32) / 255.0
    return np.clip(a, 0.0, 1.0)


# ------------ Colour / ZCA utilities ------------
class ColorUtils:
    @staticmethod
    def make_lut_srgb_to_linear(n=LUT_SIZE):
        x = np.linspace(0, 1, n, dtype=np.float32); a=0.055
        return np.where(x<=0.04045, x/12.92, ((x+a)/(1+a))**2.4).astype(np.float32)

    @staticmethod
    def make_lut_linear_to_srgb(n=LUT_SIZE):
        x = np.linspace(0, 1, n, dtype=np.float32); a=0.055
        return np.where(x<=0.0031308, 12.92*x, (1+a)*np.power(np.maximum(x,0), 1/2.4)-a).astype(np.float32)

    @staticmethod
    def srgb_to_linear_np(c):
        a=0.055; out=np.where(c<=0.04045, c/12.92, ((c+a)/(1+a))**2.4); return out.astype(np.float32)

    @staticmethod
    def linear_to_srgb_np(c):
        a=0.055; out=np.where(c<=0.0031308, 12.92*c, (1+a)*np.power(np.maximum(c,0), 1/2.4)-a); return out.astype(np.float32)

    @staticmethod
    def zca_from_data(X):
        mu = X.mean(axis=0).astype(np.float32)
        Xc = (X - mu).astype(np.float32)
        Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
        l, U = np.linalg.eigh(Sigma.astype(np.float64))
        l = np.maximum(l, 1e-6)
        W    = (U @ np.diag(1.0/np.sqrt(l)) @ U.T).astype(np.float32)
        Winv = (U @ np.diag(np.sqrt(l))     @ U.T).astype(np.float32)
        return mu, W, Winv

# ------------ Rotations ------------
def rot_x(a): c,s=np.cos(a),np.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]],dtype=np.float32)
def rot_y(a): c,s=np.cos(a),np.sin(a); return np.array([[c,0,s],[0,1,0],[-s,0,c]],dtype=np.float32)
def rot_z(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)

class RotationState:
    def __init__(self):
        self.ang_x = 0.0; self.ang_y = 0.0; self.ang_z = 0.0
        self.Ruser = np.eye(3, dtype=np.float32)

    def recompose(self):
        R = rot_z(self.ang_z) @ rot_y(self.ang_y) @ rot_x(self.ang_x)
        u, _, vh = np.linalg.svd(R.astype(np.float64), full_matrices=False)
        self.Ruser = (u @ vh).astype(np.float32)

    @staticmethod
    def R_to_euler_zyx(R):
        """Decomposes R = rot_z @ rot_y @ rot_x into (ang_x, ang_y, ang_z) in radians."""
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy > 1e-6:
            ax = np.arctan2( R[2, 1], R[2, 2])
            ay = np.arctan2(-R[2, 0], sy)
            az = np.arctan2( R[1, 0], R[0, 0])
        else:
            ax = np.arctan2(-R[1, 2], R[1, 1])
            ay = np.arctan2(-R[2, 0], sy)
            az = 0.0
        return float(ax), float(ay), float(az)

    @staticmethod
    def R_to_quat(R):
        t = np.trace(R)
        if t > 0:
            S = np.sqrt(t+1.0)*2; w=0.25*S
            x=(R[2,1]-R[1,2])/S; y=(R[0,2]-R[2,0])/S; z=(R[1,0]-R[0,1])/S
        elif (R[0,0]>R[1,1]) and (R[0,0]>R[2,2]):
            S=np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/S
            x=0.25*S; y=(R[0,1]+R[1,0])/S; z=(R[0,2]+R[2,0])/S
        elif R[1,1]>R[2,2]:
            S=np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/S
            x=(R[0,1]+R[1,0])/S; y=0.25*S; z=(R[1,2]+R[2,1])/S
        else:
            S=np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/S
            x=(R[0,2]+R[2,0])/S; y=(R[1,2]+R[2,1])/S; z=0.25*S
        q = np.array([w,x,y,z], dtype=np.float32); return q/np.linalg.norm(q)

    @staticmethod
    def quat_to_R(q):
        q = q/np.linalg.norm(q); w,x,y,z = q
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
            [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]
        ], dtype=np.float32)
        return R


# ------------ Simple OBJ/MTL loader + textured 3D widget ------------
class MeshData:
    def __init__(self, vertices=None, uvs=None, indices=None, texture_path=None, source_path=None):
        self.vertices = np.asarray(vertices if vertices is not None else np.zeros((0, 3), np.float32), dtype=np.float32)
        self.uvs = np.asarray(uvs if uvs is not None else np.zeros((0, 2), np.float32), dtype=np.float32)
        self.indices = np.asarray(indices if indices is not None else np.zeros((0,), np.uint32), dtype=np.uint32)
        self.texture_path = texture_path
        self.source_path = source_path

    @property
    def is_valid(self):
        return self.vertices.size > 0 and self.indices.size > 0


class ObjLoader:
    @staticmethod
    def _parse_mtl(mtl_path):
        """Parse MTL file for map_Kd texture path.
        Handles options like 'map_Kd -s 1 1 1 texture.jpg' by extracting the last token."""
        tex_path = None
        if not mtl_path or not os.path.isfile(mtl_path):
            return None
        base = os.path.dirname(mtl_path)
        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.lower().startswith('map_kd'):
                    # Handle various formats:
                    # map_Kd texture.jpg
                    # map_Kd -s 1 1 1 texture.jpg
                    # map_Kd -o 0 0 0 -s 1 1 1 texture.jpg
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    # The texture path is usually the last non-option token
                    # Options start with '-', so find the last non-dash-starting token
                    candidates = [p for p in parts[1:] if not p.startswith('-')]
                    if candidates:
                        rel = candidates[-1]
                        tex_path = os.path.normpath(os.path.join(base, rel))
                        if os.path.isfile(tex_path):
                            break
        return tex_path if tex_path and os.path.isfile(tex_path) else None

    @staticmethod
    def load_obj(path):
        positions = []
        texcoords = []
        unique = {}
        out_vertices = []
        out_uvs = []
        out_indices = []
        mtl_path = None

        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('mtllib '):
                    rel = line.split(None, 1)[1].strip()
                    mtl_path = os.path.normpath(os.path.join(os.path.dirname(path), rel))
                elif line.startswith('v '):
                    vals = line.split()[1:4]
                    if len(vals) == 3:
                        positions.append([float(v) for v in vals])
                elif line.startswith('vt '):
                    vals = line.split()[1:3]
                    if len(vals) >= 2:
                        texcoords.append([float(vals[0]), float(vals[1])])
                elif line.startswith('f '):
                    parts = line.split()[1:]
                    face_idx = []
                    for part in parts:
                        elems = part.split('/')
                        if not elems or not elems[0]:
                            continue
                        vi = int(elems[0])
                        ti = int(elems[1]) if len(elems) > 1 and elems[1] else 0
                        key = (vi, ti)
                        if key not in unique:
                            pos = positions[vi - 1] if vi > 0 else positions[len(positions) + vi]
                            if ti != 0 and texcoords:
                                uv = texcoords[ti - 1] if ti > 0 else texcoords[len(texcoords) + ti]
                            else:
                                uv = [0.0, 0.0]
                            unique[key] = len(out_vertices)
                            out_vertices.append(pos)
                            out_uvs.append(uv)
                        face_idx.append(unique[key])
                    for i in range(1, len(face_idx) - 1):
                        out_indices.extend([face_idx[0], face_idx[i], face_idx[i + 1]])

        # Try to get texture from MTL, fallback to same-name texture lookup
        texture_path = ObjLoader._parse_mtl(mtl_path)
        if texture_path is None:
            # Fallback: look for texture with same base name as OBJ (like PLY loader)
            base = os.path.splitext(path)[0]
            for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
                candidate = base + ext
                if os.path.isfile(candidate):
                    texture_path = candidate
                    break

        mesh = MeshData(
            vertices=np.asarray(out_vertices, dtype=np.float32),
            uvs=np.asarray(out_uvs, dtype=np.float32),
            indices=np.asarray(out_indices, dtype=np.uint32),
            texture_path=texture_path,
            source_path=path
        )
        if mesh.vertices.size == 0 or mesh.indices.size == 0:
            raise ValueError("OBJ file contains no textured triangles that could be loaded.")
        return mesh


class PlyLoader:
    _TYPE_MAP = {'float': ('f', 4), 'float32': ('f', 4),
                 'double': ('d', 8), 'float64': ('d', 8),
                 'uchar': ('B', 1), 'uint8': ('B', 1),
                 'char': ('b', 1), 'int8': ('b', 1),
                 'int': ('i', 4), 'int32': ('i', 4),
                 'uint': ('I', 4), 'uint32': ('I', 4),
                 'short': ('h', 2), 'int16': ('h', 2),
                 'ushort': ('H', 2), 'uint16': ('H', 2)}

    @staticmethod
    def load_ply(path):
        import struct
        TM = PlyLoader._TYPE_MAP
        raw_verts = []          # list of (x, y, z)
        raw_uvs_vert = []       # per-vertex UVs (if present)
        n_verts = 0
        n_faces = 0
        vert_props = []         # [(name, fmt_char, byte_size)]
        face_list_props = []    # list-type properties on face element
        face_scalar_props = []  # scalar properties on face element
        is_binary_le = False
        current_element = None
        has_face_texcoord = False  # per-face UV list
        # index type for vertex_indices list
        idx_count_fmt = 'B'
        idx_val_fmt = 'i'

        with open(path, 'rb') as fh:
            # ── Parse header ──
            while True:
                raw = fh.readline()
                if not raw:
                    break
                line = raw.decode('ascii', errors='ignore').strip()
                if line == 'end_header':
                    break
                if line.startswith('format'):
                    if 'binary_little_endian' in line:
                        is_binary_le = True
                    elif 'binary_big_endian' in line:
                        raise ValueError("binary_big_endian PLY format is not supported yet. Please convert to binary_little_endian or ASCII.")
                elif line.startswith('element vertex'):
                    n_verts = int(line.split()[-1])
                    current_element = 'vertex'
                elif line.startswith('element face'):
                    n_faces = int(line.split()[-1])
                    current_element = 'face'
                elif line.startswith('element'):
                    current_element = 'other'
                elif line.startswith('property') and current_element == 'vertex':
                    parts = line.split()
                    ptype = parts[1]
                    pname = parts[-1]
                    fmt, sz = TM.get(ptype, ('f', 4))
                    vert_props.append((pname, fmt, sz))
                elif line.startswith('property list') and current_element == 'face':
                    # e.g. "property list uchar int vertex_indices"
                    #   or "property list uchar float texcoord"
                    parts = line.split()
                    cnt_type = parts[2]
                    val_type = parts[3]
                    list_name = parts[4] if len(parts) > 4 else parts[3]
                    cf, _ = TM.get(cnt_type, ('B', 1))
                    vf, _ = TM.get(val_type, ('i', 4))
                    if list_name in ('vertex_indices', 'vertex_index'):
                        idx_count_fmt = cf
                        idx_val_fmt = vf
                    elif list_name in ('texcoord', 'texcoords'):
                        has_face_texcoord = True
                    face_list_props.append((list_name, cf, vf))
                elif line.startswith('property') and current_element == 'face':
                    parts = line.split()
                    ptype = parts[1]
                    fmt, sz = TM.get(ptype, ('f', 4))
                    face_scalar_props.append((parts[-1], fmt, sz))

            # ── Vertex property indices ──
            vp_names = [p[0] for p in vert_props]
            ix = vp_names.index('x') if 'x' in vp_names else 0
            iy = vp_names.index('y') if 'y' in vp_names else 1
            iz = vp_names.index('z') if 'z' in vp_names else 2
            has_vert_uv = False
            iu = iv = 0
            for u_name, v_name in [('s', 't'), ('texture_u', 'texture_v'), ('u', 'v')]:
                if u_name in vp_names and v_name in vp_names:
                    iu = vp_names.index(u_name)
                    iv = vp_names.index(v_name)
                    has_vert_uv = True
                    break

            idx_count_sz = struct.calcsize(idx_count_fmt)
            idx_val_sz = struct.calcsize(idx_val_fmt)

            # ── Read data ──
            if is_binary_le:
                vert_fmt = '<' + ''.join(p[1] for p in vert_props)
                vert_sz = struct.calcsize(vert_fmt)

                # --- Bulk-read all vertices at once ---
                vert_block = fh.read(n_verts * vert_sz)
                # Build a NumPy dtype matching the vertex layout
                _np_dtype_map = {'f': np.float32, 'd': np.float64,
                                 'B': np.uint8, 'b': np.int8,
                                 'H': np.uint16, 'h': np.int16,
                                 'I': np.uint32, 'i': np.int32}
                vert_dt = np.dtype([(p[0], _np_dtype_map.get(p[1], np.float32))
                                    for p in vert_props])
                vert_arr = np.frombuffer(vert_block, dtype=vert_dt, count=n_verts)
                raw_verts_np = np.column_stack([
                    vert_arr[vert_props[ix][0]].astype(np.float32),
                    vert_arr[vert_props[iy][0]].astype(np.float32),
                    vert_arr[vert_props[iz][0]].astype(np.float32),
                ])
                if has_vert_uv:
                    raw_uvs_np = np.column_stack([
                        vert_arr[vert_props[iu][0]].astype(np.float32),
                        vert_arr[vert_props[iv][0]].astype(np.float32),
                    ])
                else:
                    raw_uvs_np = None
                # Keep list versions for per-face-UV duplication path
                raw_verts = raw_verts_np  # numpy array (N,3)

                # --- Bulk-read the entire face block ---
                face_block = fh.read()  # rest of file
                buf = memoryview(face_block)
                off = 0
                face_scalar_total = sum(sz for _, _, sz in face_scalar_props)
                # Pre-compute struct sizes for list props
                _list_cnt_sizes = []
                _list_val_sizes = []
                _list_val_dtypes = []
                for lname, cf, vf in face_list_props:
                    _list_cnt_sizes.append(struct.calcsize(cf))
                    _list_val_sizes.append(struct.calcsize(vf))
                    _list_val_dtypes.append('<' + vf)

                out_verts = []
                out_uvs = []
                out_indices = []

                # Fast path: common case — only vertex_indices list, all triangles, no face texcoord
                only_vidx = (len(face_list_props) == 1
                             and face_list_props[0][0] in ('vertex_indices', 'vertex_index')
                             and not has_face_texcoord
                             and face_scalar_total == 0)

                if only_vidx:
                    cnt_sz = _list_cnt_sizes[0]
                    val_sz = _list_val_sizes[0]
                    cnt_np = _np_dtype_map.get(face_list_props[0][1], np.uint8)
                    val_np = _np_dtype_map.get(face_list_props[0][2], np.int32)
                    # Try all-triangle fast path: fixed stride = cnt_byte + 3 * val_byte
                    face_stride = cnt_sz + 3 * val_sz
                    expected_len = n_faces * face_stride
                    all_tris = (len(face_block) >= expected_len)
                    if all_tris:
                        # Verify all count bytes are 3
                        face_dt = np.dtype([('n', cnt_np), ('v0', val_np), ('v1', val_np), ('v2', val_np)])
                        face_arr = np.frombuffer(face_block, dtype=face_dt, count=n_faces)
                        all_tris = np.all(face_arr['n'] == 3)
                    if all_tris:
                        out_indices = np.column_stack([
                            face_arr['v0'], face_arr['v1'], face_arr['v2']
                        ]).astype(np.uint32).ravel()
                    else:
                        # Fallback: mixed polygon sizes — loop with struct.unpack_from
                        cnt_fmt = '<' + face_list_props[0][1]
                        tri_indices = []
                        off2 = 0
                        for _ in range(n_faces):
                            cnt = struct.unpack_from(cnt_fmt, buf, off2)[0]; off2 += cnt_sz
                            vidx = struct.unpack_from(f'<{cnt}{face_list_props[0][2]}', buf, off2)
                            off2 += cnt * val_sz
                            for j in range(1, cnt - 1):
                                tri_indices.append(vidx[0])
                                tri_indices.append(vidx[j])
                                tri_indices.append(vidx[j + 1])
                        out_indices = tri_indices
                else:
                    # Try vectorized fast path for triangle faces with per-face texcoords
                    _can_vectorize_faces = (
                        has_face_texcoord
                        and len(face_list_props) == 2
                        and face_scalar_total == 0
                    )
                    # Identify which list prop is vertex_indices and which is texcoord
                    _vidx_li = _tc_li = -1
                    if _can_vectorize_faces:
                        for li, (lname, cf, vf) in enumerate(face_list_props):
                            if lname in ('vertex_indices', 'vertex_index'):
                                _vidx_li = li
                            elif lname in ('texcoord', 'texcoords'):
                                _tc_li = li
                        _can_vectorize_faces = (_vidx_li >= 0 and _tc_li >= 0)

                    if _can_vectorize_faces:
                        # Compute fixed stride for a triangle face:
                        #   list0: count_byte + N * val_byte
                        #   list1: count_byte + N * val_byte
                        # For vertex_indices: count=3, for texcoord: count=6 (3 verts * 2 floats)
                        vidx_cnt_sz = _list_cnt_sizes[_vidx_li]
                        vidx_val_sz = _list_val_sizes[_vidx_li]
                        tc_cnt_sz = _list_cnt_sizes[_tc_li]
                        tc_val_sz = _list_val_sizes[_tc_li]
                        tri_stride = (vidx_cnt_sz + 3 * vidx_val_sz +
                                      tc_cnt_sz + 6 * tc_val_sz)
                        expected_face_len = n_faces * tri_stride
                        _all_tris_tc = (len(face_block) >= expected_face_len)

                        if _all_tris_tc:
                            # Build a structured dtype for one face record
                            vidx_np = _np_dtype_map.get(face_list_props[_vidx_li][2], np.int32)
                            tc_np = _np_dtype_map.get(face_list_props[_tc_li][2], np.float32)
                            cnt_np_vidx = _np_dtype_map.get(face_list_props[_vidx_li][1], np.uint8)
                            cnt_np_tc = _np_dtype_map.get(face_list_props[_tc_li][1], np.uint8)
                            # Build dtype in order of list props
                            fields = []
                            for li in range(2):
                                if li == _vidx_li:
                                    fields.append(('nv', cnt_np_vidx))
                                    fields.append(('v0', vidx_np))
                                    fields.append(('v1', vidx_np))
                                    fields.append(('v2', vidx_np))
                                else:
                                    fields.append(('nt', cnt_np_tc))
                                    fields.append(('u0', tc_np)); fields.append(('t0', tc_np))
                                    fields.append(('u1', tc_np)); fields.append(('t1', tc_np))
                                    fields.append(('u2', tc_np)); fields.append(('t2', tc_np))
                            face_dt = np.dtype(fields)
                            if face_dt.itemsize == tri_stride:
                                face_arr = np.frombuffer(face_block, dtype=face_dt, count=n_faces)
                                _all_tris_tc = (np.all(face_arr['nv'] == 3) and
                                                np.all(face_arr['nt'] == 6))
                            else:
                                _all_tris_tc = False

                        if _all_tris_tc:
                            # Vectorized: gather vertices by index, build UVs directly
                            vidx_all = np.column_stack([
                                face_arr['v0'].astype(np.int64),
                                face_arr['v1'].astype(np.int64),
                                face_arr['v2'].astype(np.int64),
                            ]).ravel()  # (n_faces*3,)
                            out_verts = raw_verts_np[vidx_all]  # (n_faces*3, 3)
                            out_uvs = np.column_stack([
                                face_arr['u0'].astype(np.float32),
                                face_arr['t0'].astype(np.float32),
                                face_arr['u1'].astype(np.float32),
                                face_arr['t1'].astype(np.float32),
                                face_arr['u2'].astype(np.float32),
                                face_arr['t2'].astype(np.float32),
                            ]).reshape(-1, 2)  # (n_faces*3, 2)
                            out_indices = np.arange(n_faces * 3, dtype=np.uint32)
                        else:
                            _can_vectorize_faces = False  # fall through to loop

                    if not _can_vectorize_faces:
                        # Fallback: generic per-face loop
                        for _ in range(n_faces):
                            face_vidx = []
                            face_tc = []
                            for li, (lname, cf, vf) in enumerate(face_list_props):
                                cnt_sz = _list_cnt_sizes[li]
                                val_sz = _list_val_sizes[li]
                                cnt = struct.unpack_from('<' + cf, buf, off)[0]; off += cnt_sz
                                vals = struct.unpack_from(f'<{cnt}{vf}', buf, off); off += cnt * val_sz
                                if lname in ('vertex_indices', 'vertex_index'):
                                    face_vidx = vals
                                elif lname in ('texcoord', 'texcoords'):
                                    face_tc = vals
                            off += face_scalar_total

                            if has_face_texcoord and face_tc and len(face_tc) == 2 * len(face_vidx):
                                base_idx = len(out_verts)
                                for k, vi in enumerate(face_vidx):
                                    out_verts.append(raw_verts_np[vi])
                                    out_uvs.append([float(face_tc[2*k]), float(face_tc[2*k+1])])
                                for j in range(1, len(face_vidx) - 1):
                                    out_indices.extend([base_idx, base_idx + j, base_idx + j + 1])
                            else:
                                for j in range(1, len(face_vidx) - 1):
                                    out_indices.extend([face_vidx[0], face_vidx[j], face_vidx[j + 1]])
            else:
                # ── ASCII mode ──
                fh.seek(0)
                in_header = True
                vert_count = 0
                out_verts = []
                out_uvs = []
                out_indices = []
                for raw_line in fh:
                    line = raw_line.decode('ascii', errors='ignore').strip()
                    if in_header:
                        if line == 'end_header':
                            in_header = False
                        continue
                    if vert_count < n_verts:
                        vals = line.split()
                        raw_verts.append([float(vals[ix]), float(vals[iy]), float(vals[iz])])
                        if has_vert_uv:
                            raw_uvs_vert.append([float(vals[iu]), float(vals[iv])])
                        vert_count += 1
                    else:
                        vals = line.split()
                        if not vals:
                            continue
                        pos = 0
                        face_vidx = []
                        face_tc = []
                        for lname, cf, vf in face_list_props:
                            cnt = int(vals[pos]); pos += 1
                            lvals = vals[pos:pos+cnt]; pos += cnt
                            if lname in ('vertex_indices', 'vertex_index'):
                                face_vidx = [int(v) for v in lvals]
                            elif lname in ('texcoord', 'texcoords'):
                                fvals = [float(v) for v in lvals]
                                face_tc = [(fvals[k], fvals[k+1]) for k in range(0, len(fvals), 2)]
                        # skip scalar face properties (already consumed by pos)

                        if has_face_texcoord and face_tc and len(face_tc) == len(face_vidx):
                            base_idx = len(out_verts)
                            for k, vi in enumerate(face_vidx):
                                out_verts.append(raw_verts[vi])
                                out_uvs.append(list(face_tc[k]))
                            for j in range(1, len(face_vidx) - 1):
                                out_indices.extend([base_idx, base_idx + j, base_idx + j + 1])
                        else:
                            for j in range(1, len(face_vidx) - 1):
                                out_indices.extend([face_vidx[0], face_vidx[j], face_vidx[j + 1]])

        # If no per-face texcoord, use shared vertices + per-vertex UVs
        if not has_face_texcoord:
            if isinstance(raw_verts, np.ndarray):
                # Binary path: raw_verts is already (N,3) float32
                out_verts = raw_verts
                out_uvs = raw_uvs_np if raw_uvs_np is not None else np.zeros((len(raw_verts), 2), dtype=np.float32)
            else:
                out_verts = raw_verts
                out_uvs = raw_uvs_vert if has_vert_uv else [[0.0, 0.0]] * len(raw_verts)

        # Look for a texture file with the same base name
        base = os.path.splitext(path)[0]
        texture_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
            candidate = base + ext
            if os.path.isfile(candidate):
                texture_path = candidate
                break

        mesh = MeshData(
            vertices=np.asarray(out_verts, dtype=np.float32),
            uvs=np.asarray(out_uvs, dtype=np.float32),
            indices=np.asarray(out_indices, dtype=np.uint32),
            texture_path=texture_path,
            source_path=path
        )
        if mesh.vertices.size == 0 or mesh.indices.size == 0:
            raise ValueError("PLY file contains no triangles that could be loaded.")
        return mesh


class _ModelOverlay(QtWidgets.QWidget):
    """Transparent overlay that sits on top of TexturedModelWidget.
    Receives all mouse/wheel events (being the topmost widget) and
    delegates them to the underlying 3D widget."""

    SENS_MOUSE = 0.5
    SENS_WHEEL = 3.0

    def __init__(self, model_widget, parent=None):
        super().__init__(parent)
        self._model = model_widget
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self._last_pos = None
        self._is_painting_3d = False
        self._last_paint_ij = None

    def _in_paint_mode(self):
        return getattr(self._model, 'state', None) is not None and self._model.state.paint_mode

    def wheelEvent(self, event):
        event.accept()
        self._model.on_wheel(event.angleDelta().y(), event.modifiers())

    def mousePressEvent(self, event):
        self.setFocus()
        self._last_pos = event.pos()
        # 3D painting: left click without Ctrl in paint mode
        if (self._in_paint_mode()
                and event.button() == QtCore.Qt.LeftButton
                and not _is_ctrl_qt(event.modifiers())):
            self._is_painting_3d = True
            self._last_paint_ij = None
            ij = self._model.paint_at_screen(event.x(), event.y(), last_ij=None)
            if ij is not None:
                self._last_paint_ij = ij

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        if self._is_painting_3d:
            self._is_painting_3d = False
            self._last_paint_ij = None

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        # 3D painting drag
        if self._is_painting_3d and (event.buttons() & QtCore.Qt.LeftButton):
            ij = self._model.paint_at_screen(event.x(), event.y(),
                                              last_ij=self._last_paint_ij)
            if ij is not None:
                self._last_paint_ij = ij
            self._last_pos = event.pos()
            return
        dx = event.x() - self._last_pos.x()
        dy = event.y() - self._last_pos.y()
        self._last_pos = event.pos()
        m = self._model
        m.on_drag(dx, dy=dy,
                  right_button=bool(event.buttons() & QtCore.Qt.RightButton),
                  ctrl=_is_ctrl_qt(event.modifiers()))


class _ModelHost(QtWidgets.QWidget):
    """Hosts the GL widget and an input overlay as siblings.
    This avoids QOpenGLWidget input quirks by ensuring the overlay is a normal QWidget
    at the top of the local stacking order.
    """

    def __init__(self, model_widget, parent=None):
        super().__init__(parent)
        self._model = model_widget
        self._overlay = _ModelOverlay(model_widget, parent=self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(model_widget.minimumSizeHint())

        model_widget.setParent(self)
        model_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        model_widget.show()
        self._overlay.show()
        self._overlay.raise_()

    def sizeHint(self):
        return self._model.sizeHint()

    def minimumSizeHint(self):
        return self._model.minimumSizeHint()

    def resizeEvent(self, event):
        r = self.rect()
        self._model.setGeometry(r)
        self._overlay.setGeometry(r)
        self._overlay.raise_()


class TexturedModelWidget(QOpenGLWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self.state = app_state
        self.setMinimumSize(240, 240)
        self._program = None
        self._vao = None
        self._vbo = None
        self._ebo = None
        self._tex      = None   # u_tex      : img_srgb_view  (float32 RGB)
        self._tex_orig = None   # u_tex_orig : img_srgb_orig  (float32 RGB)
        self._tex_lut_s2l = None
        self._tex_lut_l2s = None
        self._tex_brush = None
        self._index_count = 0
        self._yaw = 25.0
        self._pitch = -20.0
        self._roll = 0.0
        self._distance = 2.6
        self._pan = np.zeros(2, dtype=np.float32)
        self._brush_radius_px = 50
        self.control_level = 'basic'
        self._center = np.zeros(3, dtype=np.float32)
        self._scale = 1.0
        self._has_mesh = False
        self._vertices_interleaved = None
        self._indices = None
        self._tri_V0 = self._tri_edge1 = self._tri_edge2 = None
        self._tri_UV0 = self._tri_UV1 = self._tri_UV2 = None
        self._screen_cache = None

    def sizeHint(self):
        return QtCore.QSize(420, 320)

    def minimumSizeHint(self):
        return QtCore.QSize(120, 120)

    def set_mesh(self, mesh):
        self._has_mesh = mesh is not None and mesh.is_valid
        if not self._has_mesh:
            self._vertices_interleaved = None
            self._indices = None
            self._index_count = 0
            self._tri_V0 = self._tri_edge1 = self._tri_edge2 = None
            self._tri_UV0 = self._tri_UV1 = self._tri_UV2 = None
            self._screen_cache = None
            self.update()
            return
        verts = mesh.vertices.astype(np.float32)
        uvs = mesh.uvs.astype(np.float32)
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        self._center = ((mins + maxs) * 0.5).astype(np.float32)
        span = float(np.max(maxs - mins))
        self._scale = 1.0 / max(span, 1e-6)
        self._vertices_interleaved = np.hstack([verts, uvs]).astype(np.float32)
        self._indices = mesh.indices.astype(np.uint32)
        self._index_count = int(self._indices.size)
        # Pre-compute triangle arrays for fast ray picking
        self._cache_triangles()
        if self.context() is not None and self.context().isValid():
            self.makeCurrent()
            self._upload_mesh()
            self.doneCurrent()
        self.update()

    def _cache_triangles(self):
        """Pre-compute per-triangle geometry from interleaved vertex data."""
        verts = self._vertices_interleaved
        idx = self._indices.reshape(-1, 3)
        self._tri_V0 = np.ascontiguousarray(verts[idx[:, 0], :3])
        V1 = verts[idx[:, 1], :3]
        V2 = verts[idx[:, 2], :3]
        self._tri_edge1 = np.ascontiguousarray(V1 - self._tri_V0)
        self._tri_edge2 = np.ascontiguousarray(V2 - self._tri_V0)
        self._tri_UV0 = np.ascontiguousarray(verts[idx[:, 0], 3:5])
        self._tri_UV1 = np.ascontiguousarray(verts[idx[:, 1], 3:5])
        self._tri_UV2 = np.ascontiguousarray(verts[idx[:, 2], 3:5])
        self._screen_cache = None  # invalidate

    def reset_camera(self):
        self._yaw = 25.0
        self._pitch = -20.0
        self._roll = 0.0
        self._distance = 2.6
        self._pan = np.zeros(2, dtype=np.float32)
        self._screen_cache = None
        self.update()

    def set_control_level(self, level):
        self.control_level = 'expert' if level == 'expert' else 'basic'

    def on_drag(self, dx, dy=0, right_button=False, ctrl=False):
        if ctrl and not right_button:
            # Ctrl + left drag = pan (translate view)
            speed = self._distance * 0.002
            self._pan[0] -= dx * speed
            self._pan[1] += dy * speed
        elif right_button:
            # Distinct controls in 3D: right drag = pitch
            self._pitch += dx * _ModelOverlay.SENS_MOUSE
        else:
            self._yaw += dx * _ModelOverlay.SENS_MOUSE
        self._screen_cache = None  # camera changed — invalidate projection cache
        self.update()

    def set_texture_image(self):
        """Uploads img_srgb_orig and img_srgb_view from AppState to GPU — no CPU transform."""
        if self._tex is None or self._tex_orig is None:
            return
        if self.context() is None or not self.context().isValid():
            return
        self.makeCurrent()
        self._upload_float_tex(self._tex_orig, self.state.img_srgb_orig)
        self._upload_float_tex(self._tex,      self.state.img_srgb_view)
        self.doneCurrent()
        self.update()

    # kept for API compatibility — now a no-op (shader does the work)
    def update_texture_from_state(self):
        self.set_texture_image()

    def initializeGL(self):
        try:
            self._initializeGL_impl()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[RASCAL] initializeGL FAILED: {e}")

    def _initializeGL_impl(self):
        if IS_MACOS:
            # macOS default context is GL 2.1 (for vispy compat) → use GLSL 120
            _glsl_ver = "#version 120"
            _attr = "attribute"
            _vary_out = "varying"   # vertex shader output
            _vary_in  = "varying"   # fragment shader input
            _tex_fn   = "texture2D"
            _frag_out_decl = ""     # gl_FragColor is built-in
            _frag_out      = "gl_FragColor"
        elif IS_LINUX:
            _glsl_ver = "#version 150"
            _attr = "in"
            _vary_out = "out"
            _vary_in  = "in"
            _tex_fn   = "texture"
            _frag_out_decl = "out vec4 fragColor;"
            _frag_out      = "fragColor"
        else:
            _glsl_ver = "#version 150 core"
            _attr = "in"
            _vary_out = "out"
            _vary_in  = "in"
            _tex_fn   = "texture"
            _frag_out_decl = "out vec4 fragColor;"
            _frag_out      = "fragColor"

        vert = f"""
        {_glsl_ver}
        {_attr} vec3 a_position;
        {_attr} vec2 a_uv;
        {_vary_out} vec2 v_uv;
        uniform mat4 u_mvp;
        void main() {{
            v_uv = a_uv;
            gl_Position = u_mvp * vec4(a_position, 1.0);
        }}
        """
        frag = f"""
        {_glsl_ver}
        {_vary_in} vec2 v_uv;
        {_frag_out_decl}

        uniform sampler2D u_tex;
        uniform sampler2D u_tex_orig;
        uniform sampler2D u_lut_s2l;
        uniform sampler2D u_lut_l2s;
        uniform float u_lut_size;

        uniform mat3 W_lin;
        uniform mat3 Winv_lin;
        uniform vec3 mu_lin;
        uniform mat3 Ruser;
        uniform float u_push;
        uniform float u_var_restore;
        uniform float u_contrast;
        uniform float u_contrast_center;

        uniform sampler2D u_brush;
        uniform int u_use_brush;

        vec3 lut_lookup_rgb(sampler2D lut, float size, vec3 x) {{
            float s  = 1.0 / max(size, 1.0);
            float u0 = 0.5 * s;
            float ur = mix(u0, 1.0 - u0, clamp(x.r, 0.0, 1.0));
            float ug = mix(u0, 1.0 - u0, clamp(x.g, 0.0, 1.0));
            float ub = mix(u0, 1.0 - u0, clamp(x.b, 0.0, 1.0));
            vec3 sr = {_tex_fn}(lut, vec2(ur, 0.5)).rgb;
            vec3 sg = {_tex_fn}(lut, vec2(ug, 0.5)).rgb;
            vec3 sb = {_tex_fn}(lut, vec2(ub, 0.5)).rgb;
            return vec3(sr.r, sg.g, sb.b);
        }}

        void main() {{
            vec3 xs_orig = {_tex_fn}(u_tex_orig, v_uv).rgb;
            vec3 xl0     = lut_lookup_rgb(u_lut_s2l, u_lut_size, xs_orig);
            vec3 z       = W_lin * (xl0 - mu_lin);
            vec3 z2      = Ruser * z * u_push;
            mat3 eff_Winv = mat3(1.0) + u_var_restore * (Winv_lin - mat3(1.0));
            vec3 xo      = clamp(mu_lin + eff_Winv * z2, 0.0, 1.0);
            vec3 color   = lut_lookup_rgb(u_lut_l2s, u_lut_size, xo);

            // Manual contrast adjustment (1.0 = no change)
            color = clamp((color - u_contrast_center) * u_contrast + u_contrast_center, 0.0, 1.0);

            if (u_use_brush == 1) {{
                float a = {_tex_fn}(u_brush, v_uv).r;
                vec3 brushColor = vec3(1.0, 1.0, 0.0);
                color = mix(color, brushColor, clamp(a, 0.0, 0.5));
            }}

            {_frag_out}    = vec4(color, 1.0);
        }}
        """
        self._program = GL.glCreateProgram()
        vs = self._compile_shader(vert, GL.GL_VERTEX_SHADER)
        fs = self._compile_shader(frag, GL.GL_FRAGMENT_SHADER)
        GL.glAttachShader(self._program, vs)
        GL.glAttachShader(self._program, fs)
        GL.glLinkProgram(self._program)
        if GL.glGetProgramiv(self._program, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetProgramInfoLog(self._program).decode('utf-8', errors='ignore'))
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)

        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)

        # Allocate five textures (4 image/LUT + 1 brush overlay)
        texids = GL.glGenTextures(5)
        self._tex, self._tex_orig, self._tex_lut_s2l, self._tex_lut_l2s, self._tex_brush = texids

        for tid in texids:
            GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        GL.glEnable(GL.GL_DEPTH_TEST)
        self._upload_mesh()

        # Upload LUTs (1×N×3 float32)
        self._upload_lut(self._tex_lut_s2l, self.state.lut_s2l_tex)
        self._upload_lut(self._tex_lut_l2s, self.state.lut_l2s_tex)

        # Upload image textures
        self._upload_float_tex(self._tex_orig, self.state.img_srgb_orig)
        self._upload_float_tex(self._tex,      self.state.img_srgb_view)

        # Upload initial brush texture (1×1 black = no overlay)
        self._upload_brush_tex(self.state.brush_alpha)

    def _upload_brush_tex(self, brush_alpha):
        """Upload a (H,W,1) float32 brush alpha to the brush GL texture."""
        if self._tex_brush is None:
            return
        if brush_alpha is None:
            brush_alpha = np.zeros((1, 1, 1), dtype=np.float32)
        h, w = brush_alpha.shape[:2]
        data = np.ascontiguousarray(brush_alpha[::-1, :, 0], dtype=np.float32)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_brush)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F,
                        w, h, 0, GL.GL_RED, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def set_brush_texture(self, brush_alpha):
        """Public method: upload brush overlay and refresh the 3D view."""
        if self._tex_brush is None:
            return
        if self.context() is None or not self.context().isValid():
            return
        self.makeCurrent()
        self._upload_brush_tex(brush_alpha)
        self.doneCurrent()
        self.update()

    def _compile_shader(self, source, shader_type):
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            raise RuntimeError(GL.glGetShaderInfoLog(shader).decode('utf-8', errors='ignore'))
        return shader

    def _upload_mesh(self):
        if self._program is None or self._vertices_interleaved is None or self._indices is None:
            return
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._vertices_interleaved.nbytes,
                        self._vertices_interleaved, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self._indices.nbytes,
                        self._indices, GL.GL_STATIC_DRAW)
        stride = 5 * 4
        loc_pos = GL.glGetAttribLocation(self._program, 'a_position')
        loc_uv  = GL.glGetAttribLocation(self._program, 'a_uv')
        GL.glEnableVertexAttribArray(loc_pos)
        GL.glVertexAttribPointer(loc_pos, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(loc_uv)
        GL.glVertexAttribPointer(loc_uv,  2, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(12))
        GL.glBindVertexArray(0)

    def _upload_float_tex(self, tid, img_f32):
        """Upload a float32 (H,W,3) sRGB image to a GL texture."""
        h, w = img_f32.shape[:2]
        data = np.ascontiguousarray(np.clip(img_f32[::-1], 0, 1), dtype=np.float32)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F,
                        w, h, 0, GL.GL_RGB, GL.GL_FLOAT, data)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def _upload_lut(self, tid, lut):
        """Upload a (1, N, 3) float32 LUT as a 1D-style 2D texture."""
        data = np.ascontiguousarray(lut, dtype=np.float32)
        n = data.shape[1]
        GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB32F,
                        n, 1, 0, GL.GL_RGB, GL.GL_FLOAT, data.reshape(1, n, 3))
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def resizeGL(self, w, h):
        GL.glViewport(0, 0, max(1, w), max(1, h))

    def paintGL(self):
        GL.glClearColor(0.08, 0.08, 0.08, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self._has_mesh or self._program is None or self._index_count == 0:
            return
        proj  = self._perspective(45.0, self.width() / float(max(1, self.height())), 0.01, 100.0)
        pan3  = np.array([self._pan[0], self._pan[1], 0.0], dtype=np.float32)
        view  = self._look_at(self._camera_position() + pan3,
                               pan3.copy(),
                               np.array([0, 1, 0], dtype=np.float32))
        model = self._model_matrix()
        mvp   = proj @ view @ model

        st = self.state
        GL.glUseProgram(self._program)

        # MVP
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self._program, 'u_mvp'),
                               1, GL.GL_TRUE, mvp.astype(np.float32))

        # ZCA uniforms (updated every frame — pure GPU, zero CPU cost)
        def _loc(name): return GL.glGetUniformLocation(self._program, name)
        GL.glUniformMatrix3fv(_loc('W_lin'),    1, GL.GL_FALSE, st.W_lin)
        GL.glUniformMatrix3fv(_loc('Winv_lin'), 1, GL.GL_FALSE, st.Winv_lin)
        GL.glUniform3fv(_loc('mu_lin'), 1, st.mu_lin)
        GL.glUniformMatrix3fv(_loc('Ruser'),    1, GL.GL_FALSE, st.rot.Ruser)
        GL.glUniform1f(_loc('u_push'), float(st.push_factor))
        GL.glUniform1f(_loc('u_var_restore'), float(st.variance_restore))
        GL.glUniform1f(_loc('u_contrast'), float(st.contrast))
        GL.glUniform1f(_loc('u_contrast_center'), float(st.contrast_center))
        GL.glUniform1f(_loc('u_lut_size'), float(LUT_SIZE))

        # Brush overlay
        use_brush = 1 if (st.paint_mode and st.brush_alpha is not None and st.brush_alpha.max() > 0) else 0
        GL.glUniform1i(_loc('u_use_brush'), use_brush)

        # Textures
        for unit, (name, tid) in enumerate([
            ('u_tex',      self._tex),
            ('u_tex_orig', self._tex_orig),
            ('u_lut_s2l',  self._tex_lut_s2l),
            ('u_lut_l2s',  self._tex_lut_l2s),
            ('u_brush',    self._tex_brush),
        ]):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tid)
            GL.glUniform1i(_loc(name), unit)

        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        for unit in range(5):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

    # ---- 3D painting: ray casting + brush ----

    def _build_mvp(self):
        """Recomputes the current MVP matrix (same logic as paintGL)."""
        proj = self._perspective(45.0, self.width() / float(max(1, self.height())), 0.01, 100.0)
        pan3 = np.array([self._pan[0], self._pan[1], 0.0], dtype=np.float32)
        view = self._look_at(self._camera_position() + pan3,
                              pan3.copy(),
                              np.array([0, 1, 0], dtype=np.float32))
        model = self._model_matrix()
        return proj @ view @ model

    def _unproject_ray(self, screen_x, screen_y):
        """Convert a screen pixel (screen_x, screen_y) to a 3D ray (origin, direction)."""
        w, h = self.width(), self.height()
        # Normalised device coordinates
        ndc_x = (2.0 * screen_x / w) - 1.0
        ndc_y = 1.0 - (2.0 * screen_y / h)

        mvp = self._build_mvp()
        inv_mvp = np.linalg.inv(mvp)

        # Near and far points in clip space
        near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        far_clip  = np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)

        near_world = inv_mvp @ near_clip
        far_world  = inv_mvp @ far_clip
        near_world /= near_world[3]
        far_world  /= far_world[3]

        origin = near_world[:3]
        direction = far_world[:3] - origin
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction /= norm
        return origin, direction

    def _ensure_screen_cache(self):
        """Project all vertices to screen-space and cache per-triangle 2D AABBs."""
        if self._screen_cache is not None:
            return
        mvp = self._build_mvp()
        verts3 = self._vertices_interleaved[:, :3]  # (N, 3)
        N = verts3.shape[0]
        ones = np.ones((N, 1), dtype=np.float32)
        clip = (mvp @ np.hstack([verts3, ones]).T).T  # (N, 4)
        w = clip[:, 3:4]
        ndc = clip[:, :3] / np.where(np.abs(w) > 1e-12, w, 1e-12)  # (N, 3)
        W, H = float(self.width()), float(self.height())
        sx = (ndc[:, 0] * 0.5 + 0.5) * W
        sy = (0.5 - ndc[:, 1] * 0.5) * H
        # Per-triangle screen AABB
        idx = self._indices.reshape(-1, 3)
        sx0, sx1, sx2 = sx[idx[:, 0]], sx[idx[:, 1]], sx[idx[:, 2]]
        sy0, sy1, sy2 = sy[idx[:, 0]], sy[idx[:, 1]], sy[idx[:, 2]]
        self._screen_cache = (
            np.minimum(np.minimum(sx0, sx1), sx2),  # xmin
            np.maximum(np.maximum(sx0, sx1), sx2),  # xmax
            np.minimum(np.minimum(sy0, sy1), sy2),  # ymin
            np.maximum(np.maximum(sy0, sy1), sy2),  # ymax
        )

    def pick_uv(self, screen_x, screen_y):
        """Cast a ray from screen coords and return (u_tex, v_tex) or None.
        Uses cached triangle arrays + screen-space AABB pre-filter + vectorized
        Möller–Trumbore on the candidate subset."""
        if not self._has_mesh or self._tri_V0 is None:
            return None

        # Screen-space pre-filter: only test triangles near the click
        self._ensure_screen_cache()
        xmin, xmax, ymin, ymax = self._screen_cache
        MARGIN = 2.0
        candidates = ((xmin - MARGIN) <= screen_x) & (screen_x <= (xmax + MARGIN)) & \
                     ((ymin - MARGIN) <= screen_y) & (screen_y <= (ymax + MARGIN))
        cand_idx = np.flatnonzero(candidates)
        if cand_idx.size == 0:
            return None

        origin, direction = self._unproject_ray(screen_x, screen_y)

        V0    = self._tri_V0[cand_idx]
        edge1 = self._tri_edge1[cand_idx]
        edge2 = self._tri_edge2[cand_idx]

        pvec = np.cross(direction, edge2)
        det = np.einsum('ij,ij->i', edge1, pvec)

        valid = np.abs(det) > 1e-10
        inv_det = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)

        tvec = origin - V0
        u = np.einsum('ij,ij->i', tvec, pvec) * inv_det
        valid &= (u >= 0.0) & (u <= 1.0)

        qvec = np.cross(tvec, edge1)
        v = np.einsum('j,ij->i', direction, qvec) * inv_det
        valid &= (v >= 0.0) & ((u + v) <= 1.0)

        t = np.einsum('ij,ij->i', edge2, qvec) * inv_det
        valid &= (t > 1e-6)

        if not np.any(valid):
            return None

        t_masked = np.where(valid, t, np.inf)
        best_local = np.argmin(t_masked)
        best_tri = cand_idx[best_local]
        bu, bv = u[best_local], v[best_local]

        tex_uv = ((1.0 - bu - bv) * self._tri_UV0[best_tri]
                  + bu * self._tri_UV1[best_tri]
                  + bv * self._tri_UV2[best_tri])
        return tex_uv

    def paint_at_screen(self, screen_x, screen_y, last_ij=None):
        """Paint on brush_alpha at the UV found by ray casting. Returns (i,j) or None."""
        st = self.state
        if st.brush_alpha is None:
            return None
        uv = self.pick_uv(screen_x, screen_y)
        if uv is None:
            return None
        H, W = st.brush_alpha.shape[:2]
        j = int(np.clip(uv[0], 0.0, 1.0) * (W - 1))
        i = int((1.0 - np.clip(uv[1], 0.0, 1.0)) * (H - 1))
        r = int(getattr(self, '_brush_radius_px', 50))
        if last_ij is not None:
            i0, j0 = last_ij
            di, dj = i - i0, j - j0
            dist = float(np.hypot(di, dj))
            # Skip segment interpolation if we crossed a UV seam
            # (distance in texture space much larger than brush radius)
            if dist > r * 4:
                self._paint_disk(st.brush_alpha, i, j, r)
            elif dist == 0.0:
                self._paint_disk(st.brush_alpha, i, j, r)
            else:
                step_px = max(1.0, r * 0.5)
                n_steps = int(np.ceil(dist / step_px))
                for t in np.linspace(0.0, 1.0, n_steps + 1):
                    ii = int(round(i0 + t * di))
                    jj = int(round(j0 + t * dj))
                    self._paint_disk(st.brush_alpha, ii, jj, r)
        else:
            self._paint_disk(st.brush_alpha, i, j, r)
        # Sync brush to all 2D widgets
        for w in st.widgets_2d:
            w.brush_alpha = st.brush_alpha
            if hasattr(w, 'brush_tex'):
                w.brush_tex.set_data(st.brush_alpha.astype(np.float32))
            w.canvas.update()
        if st.model_tex_widget is not None:
            st.model_tex_widget.brush_alpha = st.brush_alpha
            if hasattr(st.model_tex_widget, 'brush_tex'):
                st.model_tex_widget.brush_tex.set_data(st.brush_alpha.astype(np.float32))
            st.model_tex_widget.canvas.update()
        # Update 3D brush overlay
        self.set_brush_texture(st.brush_alpha)
        if callable(st.on_paint_updated):
            st.on_paint_updated()
        return (i, j)

    @staticmethod
    def _paint_disk(brush_alpha, i, j, r):
        H, W = brush_alpha.shape[:2]
        i0 = max(0, i - r); i1 = min(H - 1, i + r)
        j0 = max(0, j - r); j1 = min(W - 1, j + r)
        if i0 > i1 or j0 > j1:
            return
        yy, xx = np.ogrid[i0:i1+1, j0:j1+1]
        mask = (yy - i)**2 + (xx - j)**2 <= r*r
        brush_alpha[i0:i1+1, j0:j1+1, 0][mask] = 1.0

    def _model_matrix(self):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = -self._center
        S = np.diag([self._scale, self._scale, self._scale, 1.0]).astype(np.float32)
        Rx = self._rotation_x(np.radians(self._pitch))
        Ry = self._rotation_y(np.radians(self._yaw))
        Rz = self._rotation_z(np.radians(self._roll))
        return Ry @ Rx @ Rz @ S @ T

    def _camera_position(self):
        return np.array([0.0, 0.0, self._distance], dtype=np.float32)

    def _rotation_x(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], dtype=np.float32)

    def _rotation_y(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

    def _rotation_z(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)

    def _perspective(self, fovy, aspect, znear, zfar):
        f = 1.0 / np.tan(np.radians(fovy) / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / max(aspect, 1e-6)
        m[1, 1] = f
        m[2, 2] = (zfar + znear) / (znear - zfar)
        m[2, 3] = (2 * zfar * znear) / (znear - zfar)
        m[3, 2] = -1.0
        return m

    def _look_at(self, eye, target, up):
        f = target - eye
        f = f / max(np.linalg.norm(f), 1e-8)
        s = np.cross(f, up)
        s = s / max(np.linalg.norm(s), 1e-8)
        u = np.cross(s, f)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        t = np.eye(4, dtype=np.float32)
        t[:3, 3] = -eye
        return m @ t

    def on_wheel(self, delta, mods):
        if delta == 0:
            return
        if _is_ctrl_qt(mods):
            factor = 0.9 if delta > 0 else 1.1
            self._distance = float(np.clip(self._distance * factor, 0.3, 20.0))
        else:
            # Wheel rotates model in 3D mode (roll); zoom is Ctrl+wheel only.
            self._roll += delta * 0.03
        self._screen_cache = None  # camera changed — invalidate projection cache
        self.update()

# ------------ Application state ------------
class AppState:
    """
    Application state managing image data and transformation parameters.

    TODO: Future refactor could replace the scattered image fields below with
    a dedicated ImageState class for clearer lifecycle management.

    Image field glossary (non-destructive documentation):
    - img_original_view : Truly immutable original view. Set once at image load
      or Modify Image, then NEVER modified by ROI, Radial Push, or ICA.
      Used for the "Original" tab and as the ultimate reference.
    - img_roi_base : Stable base image for each new ROI operation.
      Set at load / Modify Image, never overwritten by reprocess().
      This is what the user "sees" as their working image before painting.
    - img_srgb_orig : Current working image used by the shader (ZCA space).
      Changes during ROI operations and when Radial Push is toggled.
    - img_srgb_view : Display/texture image (may include view-only transforms).
      Usually mirrors img_srgb_orig but can differ during certain operations.
    - img_srgb : Alias to img_srgb_view (legacy compatibility).
    - img_file_orig : Original image exactly as loaded from file.
      Legacy field, kept for backward compatibility.
    - _img_file_initial : Initial image state (before any ROI/Modify operations).
      Legacy internal field.
    - _img_pre_roi : Image state preserved before ROI operations.
      Legacy field to allow ROI replay in older sessions.
    """
    def __init__(self):
        lut_s2l = ColorUtils.make_lut_srgb_to_linear(LUT_SIZE)
        lut_l2s = ColorUtils.make_lut_linear_to_srgb(LUT_SIZE)
        self.lut_s2l_tex = np.tile(lut_s2l[np.newaxis,:,np.newaxis], (1,1,3)).astype(np.float32)
        self.lut_l2s_tex = np.tile(lut_l2s[np.newaxis,:,np.newaxis], (1,1,3)).astype(np.float32)

        self.active_axis    = 'x'  # last manipulated axis: 'x', 'y', or 'z'

        self.rot = RotationState()
        self.current_image_path = ""  # Path of the currently loaded file
        self._source_dtype = None     # dtype of the original loaded file (e.g. uint8, uint16, float32)

        # Display angles for the left panel (when rotating in RGB mode)
        self.disp_ax = 0.0
        self.disp_ay = 0.0
        self.disp_az = 0.0

        # Fibonacci normalisation
        self.fib_norm_mode = False
        self._fib_img_cached = None       # normalised image cache (invalidated on image load)
        self._fib_Zs_norm_cached = None   # ZCA coords after Fibonacci normalisation
        self.samples_zca_fib = None       # sampled subset of _fib_Zs_norm_cached

        # Brush variables
        self.paint_mode = False
        self.brush_alpha = None
        self.last_brush_alpha = None       # saved after re-process
        self.last_img_srgb_orig = None       # image before re-process
        self.last_img_srgb_reprocessed = None  # image after re-process
        self.on_paint_updated = None  # callback fired after each brush stroke
        self.on_rotation_started = None  # callback fired at the start of a rotation drag
        self.on_ica_deactivated = None  # callback fired when manual rotation overrides ICA
        self.last_rot_quat = None  # quaternion used during the last re-process
        self.push_factor = 1.0  # norm multiplier in the transformed space
        self.variance_restore = 1.0  # 1.0 = full variance restore, 0.0 = identity (no restore)
        self.contrast = 1.0  # 1.0 = no contrast change, <1.0 flattens, >1.0 increases contrast
        self.contrast_center = 0.5  # pivot for contrast adjustment (0.5 = mid-gray)
        self.ica_active = False  # True if ICA has been applied and hasn't been overridden by manual rotation
        self.modify_params = None  # dict: {crop_rect, brightness, contrast} from last Modify Image

        # Preset checkpoints (cleared on image load / reset)
        self.presets = []  # list of dicts: {rot, push_factor, fib_norm_mode, ...}

        self.widgets_2d = []
        self.widget_3d  = None
        self.model_widget = None
        self.model_tex_widget = None
        self.mesh_data = None
        self.model_path = None

        # Initialise with a grey 4×4 placeholder image (no file loading at startup)
        self._init_placeholder()

    def _init_placeholder(self):
        """Initialises all structures with a neutral 4×4 grey image (fast startup).

        See AppState class docstring for image field glossary.
        """
        img = np.full((4, 4, 3), 0.5, dtype=np.float32)
        self.img_file_orig = img.copy()
        self._img_file_initial = img.copy()
        self._img_pre_roi = img.copy()
        self.img_original_view = img.copy()  # truly immutable — never overwritten by ROI/fib/ICA
        self.img_roi_base = img.copy()       # stable base for each new ROI (set at load / modify)
        self.img_srgb_orig = img.copy()
        self.img_srgb_view = img.copy()
        self.img_srgb      = img
        self.current_image_path = ""
        Xl = img.reshape(-1, 3)
        self.mu_lin, self.W_lin, self.Winv_lin = ColorUtils.zca_from_data(Xl)
        self._mu_lin_orig   = self.mu_lin.copy()
        self._W_lin_orig    = self.W_lin.copy()
        self._Winv_lin_orig = self.Winv_lin.copy()
        self.brush_alpha    = np.zeros((4, 4, 1), np.float32)
        self.samples_lin    = Xl.copy()
        self.disp_ax = self.disp_ay = self.disp_az = 0.0
        self._fib_img_cached = None
        self._fib_Zs_norm_cached = None
        self.samples_zca_fib = None
        self.presets = []
        self.mesh_data = None
        self.model_path = None

    def load_image_array(self, img, pseudo_path=""):
        """Initialise state from a numpy array (any dtype) instead of a file path.
        pseudo_path is stored as current_image_path (can be the original session
        image_path so it never points to a deleted temporary file)."""
        orig_dtype = img.dtype
        img = _normalize_image_array(img)
        self._source_dtype = orig_dtype   # preserve original file bit depth
        self.img_file_orig       = img.copy()
        self._img_file_initial   = img.copy()
        self._img_pre_roi        = img.copy()
        self.img_original_view   = img.copy()
        self.img_roi_base        = img.copy()
        self.img_srgb_orig       = img.copy()
        self.img_srgb_view       = img.copy()
        self.img_srgb            = self.img_srgb_orig
        self.current_image_path  = pseudo_path
        self._raw_img_for_session = None
        self.modify_params = None

        img_lin = ColorUtils.srgb_to_linear_np(np.clip(img, 0, 1))
        Xl = img_lin.reshape(-1, 3).astype(np.float32)
        self.mu_lin, self.W_lin, self.Winv_lin = ColorUtils.zca_from_data(Xl)
        self._mu_lin_orig   = self.mu_lin.copy()
        self._W_lin_orig    = self.W_lin.copy()
        self._Winv_lin_orig = self.Winv_lin.copy()
        self.brush_alpha = np.zeros(img.shape[:2] + (1,), np.float32)
        rng = np.random.default_rng(42)
        N = min(N_SAMPLES_3D, Xl.shape[0])
        idx = rng.choice(Xl.shape[0], N, replace=False)
        self.samples_lin = Xl[idx]
        self.disp_ax = self.disp_ay = self.disp_az = 0.0
        self._fib_img_cached     = None
        self._fib_Zs_norm_cached = None
        self.samples_zca_fib     = None
        self.presets             = []
        if self.mesh_data is None:
            self.model_path = None

    def load_image(self, path):
        raw = _imread_any(path)
        raw = _apply_exif_orientation(raw, path)
        orig_dtype = raw.dtype
        img = _normalize_image_array(raw)
        self.img_file_orig = img.copy()   # FILE IMAGE, never modified
        self._img_file_initial = img.copy()  # truly immutable copy of the loaded file
        self._img_pre_roi = img.copy()       # copy before any ROI reprocess (for ROI deactivation reset)
        self.img_original_view = img.copy()  # truly immutable — never overwritten by ROI/fib/ICA
        self.img_roi_base = img.copy()       # stable base for each new ROI (set at load / modify)
        self.img_srgb_orig = img.copy()   # frozen ORIGINAL
        self.img_srgb_view = img.copy()   # display copy (may change later if needed)
        self.img_srgb = self.img_srgb_orig  # alias for brush compatibility
        self.current_image_path = path    # update path of the currently loaded file
        self._source_dtype = orig_dtype   # remember original file bit depth for session save warning
        self._raw_img_for_session = None
        self.modify_params = None

        img_lin = ColorUtils.srgb_to_linear_np(np.clip(self.img_srgb_orig,0,1))
        Xl = img_lin.reshape(-1,3).astype(np.float32)

        # Frozen ZCA (never touched afterwards)
        self.mu_lin, self.W_lin, self.Winv_lin = ColorUtils.zca_from_data(Xl)
        # Save original ZCA to restore it in Fibonacci mode
        self._mu_lin_orig   = self.mu_lin.copy()
        self._W_lin_orig    = self.W_lin.copy()
        self._Winv_lin_orig = self.Winv_lin.copy()

        # Brush initialised to blank
        self.brush_alpha = np.zeros(img.shape[:2] + (1,), np.float32)

        rng = np.random.default_rng(42)
        N = min(N_SAMPLES_3D, Xl.shape[0])
        idx = rng.choice(Xl.shape[0], N, replace=False)
        self.samples_lin = Xl[idx]

        self.disp_ax = self.disp_ay = self.disp_az = 0.0
        self._fib_img_cached = None   # invalidate Fibonacci cache
        self._fib_Zs_norm_cached = None
        self.samples_zca_fib = None
        self.presets = []
        # Only clear mesh data when not loading a texture for an existing 3D model
        if self.mesh_data is None:
            self.model_path = None


    def load_model(self, path):
        """Loads an OBJ or PLY model and stores it in mesh_data / model_path."""
        if path.lower().endswith('.ply'):
            mesh = PlyLoader.load_ply(path)
        else:
            mesh = ObjLoader.load_obj(path)
        self.mesh_data = mesh
        self.model_path = path
        return mesh

    def refresh_all(self):
        for w in self.widgets_2d:
            w.canvas.update()
        if self.widget_3d is not None:
            self.widget_3d.update_markers()
            self.widget_3d.canvas.update()
        if self.model_widget is not None:
            self.model_widget.update()   # shader recalculates ZCA every frame — no CPU work
        if self.model_tex_widget is not None:
            self.model_tex_widget.canvas.update()

# ------------ 2D widget ------------
class VisImageWidget(QtWidgets.QWidget):
    """Mode: 0=RGB, 1=R greyscale, 2=G greyscale, 3=B greyscale"""
    SENS_MOUSE = 0.02
    SENS_WHEEL = 0.08
    clicked = QtCore.pyqtSignal()

    def __init__(self, app_state, mode=0, compact=False, parent=None):
        super().__init__(parent)
        self.state = app_state
        self.mode  = int(mode)
        self.compact = compact
        init_size = WIN_SIZE_2D if not compact else COMPACT_SIZE_2D

        self.canvas = app.Canvas(keys='interactive', size=init_size, show=False)
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        # Make the canvas shrinkable
        # - an Expanding SizePolicy and small minimum size help the QSplitter shrink the column
        self.canvas.native.setMinimumSize(50, 50)
        sp = self.canvas.native.sizePolicy()
        sp.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        sp.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.canvas.native.setSizePolicy(sp)
        # Apply to the Qt wrapper too to avoid implicit minima
        wsp = self.sizePolicy()
        wsp.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        wsp.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.setSizePolicy(wsp)
        lay.addWidget(self.canvas.native)

        positions = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], np.float32)
        texcoords = np.array([[0,0],[0,1],[1,0],[1,1]], np.float32)
        self.prog = gloo.Program(VERT, FRAG)
        self.prog['a_position'] = gloo.VertexBuffer(positions)
        self.prog['a_texcoord'] = gloo.VertexBuffer(texcoords)

        # Textures: working image (for ZCA) + file original (for mode 4)
        self.tex_img_orig      = gloo.Texture2D(self.state.img_srgb_orig,      interpolation='linear')
        self.tex_img_file_orig = gloo.Texture2D(self.state.img_original_view,  interpolation='linear')
        self.tex_lut_s2l   = gloo.Texture2D(self.state.lut_s2l_tex,     interpolation='linear')
        self.tex_lut_l2s   = gloo.Texture2D(self.state.lut_l2s_tex,     interpolation='linear')

        self.prog['u_tex_orig']      = self.tex_img_orig
        self.prog['u_tex_file_orig'] = self.tex_img_file_orig
        self.prog['u_lut_s2l']   = self.tex_lut_s2l
        self.prog['u_lut_l2s']   = self.tex_lut_l2s
        self.prog['u_lut_size']  = float(LUT_SIZE)

        self.prog['W_lin']     = self.state.W_lin
        self.prog['Winv_lin']  = self.state.Winv_lin
        self.prog['mu_lin']    = self.state.mu_lin
        self.prog['u_mode']    = int(self.mode)

        # Zoom parameter initialisation
        self.zoom_level  = 1.0
        self.zoom_center = np.array([0.0, 0.0], dtype=np.float32)
        self.prog['u_zoom_scale']  = 1.0
        self.prog['u_zoom_offset'] = [0.0, 0.0]
        self.prog['u_aspect_correction'] = [1.0, 1.0]
        self.prog['u_push'] = float(self.state.push_factor)
        self.prog['u_var_restore'] = float(self.state.variance_restore)
        self.prog['u_contrast'] = float(self.state.contrast)
        self.prog['u_contrast_center'] = float(self.state.contrast_center)

        # Initialise shader variables with default values
        self.prog['u_use_brush'] = 0
        self.prog['u_mode'] = int(self.mode)

        # Split view state
        self.split_active = False
        self.split_pos = 0.5  # normalised 0..1 (left=original, right=transformed)
        self._split_dragging = False
        self.prog['u_split_active'] = 0
        self.prog['u_split_pos'] = 0.5
        self.prog['u_split_line_w'] = 0.002
        
        # Initialise brush texture with a neutral value
        self.brush_tex = gloo.Texture2D(np.zeros((1,1,1), dtype=np.float32))
        self.prog['u_brush'] = self.brush_tex

        # Brush: default radius and buffer shared with state
        self.brush_radius_px = 30.0
        self.brush_alpha = self.state.brush_alpha

        # Initialise required matrices and vectors
        eye3 = np.eye(3, dtype=np.float32)

        self.prog['Ruser'] = eye3
        self.canvas.events.resize.connect(self._on_resize)
        self.canvas.events.draw.connect(self._on_draw)
        # Connect mouse events (brush handling included)
        self.canvas.events.mouse_press.connect(self._on_press)
        self.canvas.events.mouse_move.connect(self._on_move)
        self.canvas.events.mouse_release.connect(self._on_release)
        self.canvas.events.mouse_wheel.connect(self._on_wheel)

        self.last_pos = None
        self._is_painting = False
        self._last_paint_ij = None

    def update_image_texture(self):
        # Ensure data is contiguous for safe GPU upload
        img_orig = np.ascontiguousarray(self.state.img_srgb_orig, dtype=np.float32)
        img_file = np.ascontiguousarray(self.state.img_original_view, dtype=np.float32)

        # Update image textures. Recreate the texture object if the image
        # size changed to avoid glTexSubImage2D corruption that can show the
        # image as a tiled/puzzle pattern (observed after Modify Image -> Reset).
        if self.tex_img_orig.shape[:2] != img_orig.shape[:2]:
            self.tex_img_orig = gloo.Texture2D(
                img_orig, interpolation='linear', wrapping='clamp_to_edge'
            )
            self.prog['u_tex_orig'] = self.tex_img_orig
        else:
            self.tex_img_orig.set_data(img_orig)

        if self.tex_img_file_orig.shape[:2] != img_file.shape[:2]:
            self.tex_img_file_orig = gloo.Texture2D(
                img_file, interpolation='linear', wrapping='clamp_to_edge'
            )
            self.prog['u_tex_file_orig'] = self.tex_img_file_orig
        else:
            self.tex_img_file_orig.set_data(img_file)

        # Disable brush overlay
        self.prog['u_use_brush'] = 0

        # Update brush buffer if image changed
        self.brush_alpha = self.state.brush_alpha
        self.brush_tex.set_data(np.ascontiguousarray(
            self.state.brush_alpha.astype(np.float32)
        ))

        # Frozen ZCA
        self.prog['W_lin']     = self.state.W_lin
        self.prog['Winv_lin']  = self.state.Winv_lin
        self.prog['mu_lin']    = self.state.mu_lin
        self.prog['u_contrast'] = float(self.state.contrast)
        self.prog['u_contrast_center'] = float(self.state.contrast_center)

    def toggle_split(self, active=None):
        """Toggle or set the split view mode."""
        if active is None:
            self.split_active = not self.split_active
        else:
            self.split_active = bool(active)
        if self.split_active:
            # Place the split at the centre of the currently visible area
            if self.zoom_level < 1.0:
                # Zoomed in: compute the visible texture-X range from zoom params
                scale = 1.0 / self.zoom_level
                off_x = float(self.zoom_center[0] * (1.0 - scale))
                # Visible NDC range is -1..+1, map to attribute then texture U
                u_left  = ((-1.0 - off_x) / scale + 1.0) * 0.5
                u_right = (( 1.0 - off_x) / scale + 1.0) * 0.5
                self.split_pos = max(0.0, min(1.0, (u_left + u_right) * 0.5))
            else:
                self.split_pos = 0.5
        self.prog['u_split_active'] = 1 if self.split_active else 0
        self.prog['u_split_pos'] = self.split_pos
        if not self.split_active:
            self.canvas.native.unsetCursor()
        self.canvas.update()

    def _push_dynamics(self):
        st = self.state
        # Use the same rotation for both modes
        self.prog['Ruser'] = st.rot.Ruser
        self.prog['u_push'] = float(st.push_factor)
        self.prog['u_var_restore'] = float(st.variance_restore)
        self.prog['u_contrast'] = float(st.contrast)
        self.prog['u_contrast_center'] = float(st.contrast_center)

        # Split line: keep constant ~2 px width on screen regardless of zoom
        if self.split_active:
            try:
                img_w = st.img_srgb_view.shape[1]
                _, _, vp_w, _ = self._letterbox()
                if vp_w > 0 and img_w > 0:
                    pixels_per_texel = vp_w / float(img_w) * (1.0 / max(self.zoom_level, 0.01))
                    self.prog['u_split_line_w'] = max(1.5 / max(pixels_per_texel * img_w, 1.0), 0.0005)
                else:
                    self.prog['u_split_line_w'] = 0.002
            except Exception:
                self.prog['u_split_line_w'] = 0.002

        # Enable selection brush overlay (yellow) on all 2D views (colour + R/G/B)
        self.prog['u_use_brush'] = 1 if (st.paint_mode and st.brush_alpha is not None and st.brush_alpha.max() > 0) else 0

    def _on_resize(self, ev):
        gloo.set_viewport(0, 0, *self.canvas.physical_size)

    def _on_draw(self, ev):
        self._push_dynamics()
        gloo.clear((0,0,0,1))
        
        # When zoomed in, use full canvas with aspect correction;
        # otherwise use letterboxing
        if self.zoom_level < 1.0:
            # Zoom mode: use the full canvas area
            cw, ch = self.canvas.physical_size
            x, y, w, h = 0, 0, cw, ch
            
            # Compute aspect correction to preserve image ratio
            H, W = self.state.img_srgb_view.shape[:2]
            img_ratio = W / float(H)
            canvas_ratio = cw / float(ch) if ch > 0 else 1.0
            
            if canvas_ratio > img_ratio:
                # Canvas wider: shrink width
                aspect_x = img_ratio / canvas_ratio
                aspect_y = 1.0
            else:
                # Canvas taller: shrink height
                aspect_x = 1.0
                aspect_y = canvas_ratio / img_ratio
        else:
            # Normal mode: letterbox
            x, y, w, h = self._letterbox()
            aspect_x, aspect_y = 1.0, 1.0

        if self.zoom_level != 1.0:
            scale = 1.0 / self.zoom_level
            off   = self.zoom_center*(1.0 - scale)
            self.prog['u_zoom_scale']  = scale
            self.prog['u_zoom_offset'] = [float(off[0]), float(off[1])]
        else:
            self.prog['u_zoom_scale']  = 1.0
            self.prog['u_zoom_offset'] = [0.0, 0.0]
        
        self.prog['u_aspect_correction'] = [float(aspect_x), float(aspect_y)]

        if w>0 and h>0:
            gloo.set_viewport(x,y,w,h)
            self.prog.draw('triangle_strip')
        gloo.set_viewport(0,0,*self.canvas.physical_size)

    def _letterbox(self):
        H, W = self.state.img_srgb_view.shape[:2]
        cw, ch = self.canvas.physical_size
        if cw <= 0 or ch <= 0: return 0,0,0,0
        img_ratio, canvas_ratio = W/float(H), cw/float(ch)
        if canvas_ratio > img_ratio:
            h = ch; w = int(round(h*img_ratio)); x=(cw-w)//2; y=0
        else:
            w = cw; h = int(round(w/img_ratio)); x=0; y=(ch-h)//2
        return x,y,w,h

    # ---------- Brush helpers ----------
    def _mouse_to_texcoord(self, ev):
        x,y,w,h = self._letterbox()
        # ev.pos is in logical pixels; _letterbox uses physical pixels.
        # Scale by pixel_ratio so coordinates match (Retina = 2x, normal = 1x).
        px = getattr(self.canvas, 'pixel_scale', 1.0)
        mx, my = ev.pos[0] * px, ev.pos[1] * px
        if not (x <= mx <= x+w and y <= my <= y+h):
            return None
        ndc_x = ( (mx - x) / float(w) ) * 2.0 - 1.0
        ndc_y = ( 1.0 - (my - y) / float(h) ) * 2.0 - 1.0

        scale = (1.0 / self.zoom_level) if self.zoom_level != 1.0 else 1.0
        off = self.zoom_center*(1.0 - scale) if self.zoom_level != 1.0 else np.array([0.0,0.0], np.float32)
        ax = (ndc_x - off[0]) / scale
        ay = (ndc_y - off[1]) / scale

        u = (ax + 1.0) * 0.5
        v = 1.0 - ( (ay + 1.0) * 0.5 )
        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
            return None
        return (u, v)

    def _paint_disk_ij(self, i, j):
        H, W = self.brush_alpha.shape[:2]
        r = int(self.brush_radius_px)
        i0 = max(0, i - r); i1 = min(H - 1, i + r)
        j0 = max(0, j - r); j1 = min(W - 1, j + r)
        if i0 > i1 or j0 > j1:
            return
        yy, xx = np.ogrid[i0:i1+1, j0:j1+1]
        mask = (yy - i)**2 + (xx - j)**2 <= r*r
        self.brush_alpha[i0:i1+1, j0:j1+1, 0][mask] = 1.0

    def _paint_segment(self, i0j0, i1j1):
        (i0, j0) = i0j0; (i1, j1) = i1j1
        di = i1 - i0; dj = j1 - j0
        dist = float(np.hypot(di, dj))
        if dist == 0.0:
            self._paint_disk_ij(i0, j0); return
        step_px = max(1.0, self.brush_radius_px * 0.5)
        n_steps = int(np.ceil(dist / step_px))
        for t in np.linspace(0.0, 1.0, n_steps + 1):
            ii = int(round(i0 + t * di))
            jj = int(round(j0 + t * dj))
            self._paint_disk_ij(ii, jj)

    def _after_paint_updated(self):
        # Refresh the GPU brush texture on this widget
        self.brush_tex.set_data(self.brush_alpha.astype(np.float32))
        self.state.brush_alpha = self.brush_alpha
        self.canvas.update()
        # Propagate brush to all other 2D widgets (R, G, B)
        for w in (self.state.widgets_2d or []):
            if w is self:
                continue
            w.brush_alpha = self.state.brush_alpha
            w.brush_tex.set_data(self.state.brush_alpha.astype(np.float32))
            w.canvas.update()
        if callable(self.state.on_paint_updated):
            self.state.on_paint_updated()

    def _paint_at_event(self, ev):
        if not self.state.paint_mode:
            return
        uv = self._mouse_to_texcoord(ev)
        if uv is None:
            return
        u, v = uv
        H, W = self.brush_alpha.shape[:2]
        j = int(u * (W - 1))
        i = int(v * (H - 1))

        if getattr(ev, "is_dragging", False) and self._last_paint_ij is not None:
            self._paint_segment(self._last_paint_ij, (i, j))
        else:
            self._paint_disk_ij(i, j)
        self._last_paint_ij = (i, j)
        self._after_paint_updated()

    def _tex_x_from_event(self, ev):
        """Convert mouse event to texture X coordinate, zoom-aware.
        Unlike _mouse_to_texcoord, does NOT clamp to 0..1 so the split
        line can reach the visible edges even when zoomed."""
        x, y, w, h = self._letterbox()
        px = getattr(self.canvas, 'pixel_scale', 1.0)
        mx, my = ev.pos[0] * px, ev.pos[1] * px
        if w <= 0 or h <= 0:
            return None
        # Mouse → NDC
        ndc_x = ((mx - x) / float(w)) * 2.0 - 1.0
        # Invert zoom transform: NDC → attribute space
        scale = (1.0 / self.zoom_level) if self.zoom_level != 1.0 else 1.0
        off_x = float(self.zoom_center[0] * (1.0 - scale)) if self.zoom_level != 1.0 else 0.0
        ax = (ndc_x - off_x) / scale
        # Attribute space → texture U
        u = (ax + 1.0) * 0.5
        return u

    def _on_press(self, ev):
        # Single click on a compact view => swap signal (never paint on thumbnails)
        if self.compact and getattr(ev, 'button', 1) == 1:
            mods = set(ev.modifiers or [])
            if not ('Alt' in mods or _is_ctrl_vispy(mods) or 'Shift' in mods):
                self.clicked.emit()
                return
        # Split view drag: check if click is near the split line
        if not self.compact and self.split_active and ev.button == 1:
            tx = self._tex_x_from_event(ev)
            grab = 0.02 / max(self.zoom_level, 0.01)  # wider grab zone when zoomed
            if tx is not None and abs(tx - self.split_pos) < grab:
                self._split_dragging = True
                return
        # Paint mode (main view only, not compact)
        if not self.compact and self.state.paint_mode and ev.button == 1:
            self._is_painting = True
            self._last_paint_ij = None
            self._paint_at_event(ev)
            return
        self.last_pos = ev.pos

        if callable(self.state.on_rotation_started):
            self.state.on_rotation_started()

        mods = set(ev.modifiers or [])
        btn  = getattr(ev, 'button', 1)
        is_x = ('Alt' in mods) or _is_ctrl_vispy(mods)
        is_y = (btn in (2,3))
        is_z = not (is_x or is_y)

        # Compute the instantaneous axis at click time and freeze it for the entire drag
        ay = float(self.state.rot.ang_y)
        az = float(self.state.rot.ang_z)
        ex = np.array([1,0,0], np.float32)
        ey = np.array([0,1,0], np.float32)
        ez = np.array([0,0,1], np.float32)
        if is_x:   self.state.active_axis = 'x'
        elif is_y: self.state.active_axis = 'y'
        else:      self.state.active_axis = 'z'
        self.state.refresh_all()

    def _on_release(self, ev):
        if self._split_dragging:
            self._split_dragging = False
            return
        if not self.compact and ev.button == 1 and self.state.paint_mode:
            self._is_painting = False
            self._last_paint_ij = None
            return
        self.last_pos = None

    def _on_move(self, ev):
        # Split line dragging
        if self._split_dragging and getattr(ev, "is_dragging", False):
            tx = self._tex_x_from_event(ev)
            if tx is not None:
                self.split_pos = max(0.0, min(1.0, tx))
                self.prog['u_split_pos'] = self.split_pos
                self.canvas.update()
            return
        # Split cursor feedback: show ↔ when hovering near the split line
        if not self.compact and self.split_active and not getattr(ev, "is_dragging", False):
            tx = self._tex_x_from_event(ev)
            grab = 0.02 / max(self.zoom_level, 0.01)
            near = (tx is not None and abs(tx - self.split_pos) < grab)
            native = self.canvas.native
            if near:
                if native.cursor().shape() != QtCore.Qt.SplitHCursor:
                    native.setCursor(QtCore.Qt.SplitHCursor)
            else:
                if native.cursor().shape() != QtCore.Qt.ArrowCursor:
                    native.unsetCursor()
        if not self.compact and self.state.paint_mode:
            if getattr(ev, "is_dragging", False) and self._is_painting:
                self._paint_at_event(ev)
            return
        if self.last_pos is None or not ev.is_dragging:
            return

        x,y = ev.pos; lx,ly = self.last_pos; dx,dy = x-lx, y-ly; self.last_pos=(x,y)
        btns = set(int(b) for b in getattr(ev,'buttons',[]) or [])
        is_y = (2 in btns) or (3 in btns)
        is_z = not is_y

        dth = dx * self.SENS_MOUSE
        if dth == 0.0:
            return

        st = self.state

        # Do not recompute the axis during the drag: it was frozen at _on_press

        # Simple Euler-angle rotation: left drag = Z, right drag = Y
        if is_y:   st.rot.ang_y += dth; st.disp_ay += dth
        else:      st.rot.ang_z += dth; st.disp_az += dth
        st.rot.recompose()
        
        # Deactivate ICA when manual rotation is applied
        if st.ica_active:
            st.ica_active = False
            if callable(st.on_ica_deactivated):
                st.on_ica_deactivated()

        st.refresh_all()

    def _on_wheel(self, ev):
        px = getattr(self.canvas, 'pixel_scale', 1.0)
        mx, my = ev.pos[0] * px, ev.pos[1] * px
        # When zoomed in, the render covers the full canvas; use letterbox only at zoom_level == 1.0
        if self.zoom_level < 1.0:
            cw, ch = self.canvas.physical_size
            x, y, w, h = 0, 0, cw, ch
        else:
            x, y, w, h = self._letterbox()
        
        # Check if the mouse is over the image area
        if not (x <= mx <= x+w and y <= my <= y+h):
            return
        
        delta = ev.delta
        if hasattr(delta,'__iter__'):
            delta = delta[1] if len(delta)>1 else (delta[0] if len(delta)>0 else 0)
        
        mods = set(ev.modifiers or [])
        
        if _is_ctrl_vispy(mods):
            # Ctrl + wheel => zoom
            zoom_factor = 0.9 if delta>0 else 1.1
            old_zoom = self.zoom_level
            new_zoom = np.clip(old_zoom * zoom_factor, 0.2, 1.0)
            
            if new_zoom == old_zoom:
                return
            
            if new_zoom == 1.0:
                self.zoom_level = 1.0
                self.zoom_center = np.array([0.0, 0.0], dtype=np.float32)
                self.canvas.update()
                return
            
            ndc_x = ((mx - x) / float(w)) * 2.0 - 1.0
            ndc_y = (1.0 - (my - y) / float(h)) * 2.0 - 1.0
            mouse_ndc = np.array([ndc_x, ndc_y], dtype=np.float32)
            
            old_scale = 1.0 / old_zoom
            old_offset = self.zoom_center * (1.0 - old_scale)
            image_point = (mouse_ndc - old_offset) / old_scale
            new_scale = 1.0 / new_zoom
            new_offset = mouse_ndc - image_point * new_scale
            
            self.zoom_level = new_zoom
            self.zoom_center = new_offset / (1.0 - new_scale)
            self.canvas.update()
        else:
            # Wheel alone => rotate around X axis
            dth = delta * self.SENS_WHEEL
            if dth == 0.0:
                return
            if callable(self.state.on_rotation_started):
                self.state.on_rotation_started()
            st = self.state
            st.active_axis = 'x'
            st.rot.ang_x += dth; st.disp_ax += dth
            st.rot.recompose()
            
            # Deactivate ICA when manual rotation is applied
            if st.ica_active:
                st.ica_active = False
                if callable(st.on_ica_deactivated):
                    st.on_ica_deactivated()
            
            st.refresh_all()

# ------------ 3D widget ------------
class VisScatter3DWidget(QtWidgets.QWidget):
    def __init__(self, app_state, compact=True, parent=None):
        super().__init__(parent)
        self.state = app_state

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(2)
        self.canvas = scene.SceneCanvas(keys='interactive', size=(400,400), show=False)
        self.canvas.native.setMinimumSize(50, 50)
        sp3 = self.canvas.native.sizePolicy()
        sp3.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        sp3.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.canvas.native.setSizePolicy(sp3)
        vbox.addWidget(self.canvas.native, 1)

        self._cloud_mode_label = QtWidgets.QLabel("sRGB cube")
        self._cloud_mode_label.setAlignment(QtCore.Qt.AlignCenter)
        self._cloud_mode_label.setStyleSheet("color: black;")
        vbox.addWidget(self._cloud_mode_label, 0)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=45, azimuth=45, elevation=30, distance=2.2, center=(0.5,0.5,0.5)
        )

        cube_edges = np.array([
            [0,0,0],[1,0,0],[1,0,0],[1,1,0],[1,1,0],[0,1,0],[0,1,0],[0,0,0],
            [0,0,1],[1,0,1],[1,0,1],[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,0,1],
            [0,0,0],[0,0,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1],[0,1,0],[0,1,1]
        ], np.float32)
        cube_line = visuals.Line(pos=cube_edges, color=(0.6,0.6,0.6,1.0), connect='segments', width=1)
        self.view.add(cube_line)

        eps = 0.02
        lbl_kwargs = dict(color='white', parent=self.view.scene, font_size=12,
                          anchor_x='left', anchor_y='baseline')
        self._cube_labels = [
            scene.visuals.Text('0', pos=(0, -eps, -eps), **lbl_kwargs),
            scene.visuals.Text('1', pos=(1, -eps, -eps), **lbl_kwargs),
            scene.visuals.Text('0', pos=(-eps, 0, -eps), **lbl_kwargs),
            scene.visuals.Text('1', pos=(-eps, 1, -eps), **lbl_kwargs),
            scene.visuals.Text('0', pos=(-eps, -eps, 0), **lbl_kwargs),
            scene.visuals.Text('1', pos=(-eps, -eps, 1), **lbl_kwargs),
        ]

        self.markers = visuals.Markers()
        self.view.add(self.markers)

        # RGB axes for normal mode (R=X red, G=Y green, B=Z blue)
        rgb_axis_pos = np.array([
            [0,0,0],[1,0,0],  # R axis
            [0,0,0],[0,1,0],  # G axis
            [0,0,0],[0,0,1],  # B axis
        ], dtype=np.float32)
        rgb_axis_colors = np.array([
            [1.0, 0.2, 0.2, 1.0],[1.0, 0.2, 0.2, 1.0],
            [0.2, 1.0, 0.2, 1.0],[0.2, 1.0, 0.2, 1.0],
            [0.2, 0.4, 1.0, 1.0],[0.2, 0.4, 1.0, 1.0],
        ], dtype=np.float32)
        self._rgb_axes = visuals.Line(
            pos=rgb_axis_pos, color=rgb_axis_colors, connect='segments', width=3
        )
        self.view.add(self._rgb_axes)
        self._rgb_axes.visible = True

        # Axes lines + labels for decorrelated mode (hidden by default)
        self._axis_lines = visuals.Line(
            pos=np.zeros((6, 3), dtype=np.float32),
            color=(1.0, 1.0, 0.3, 0.7), connect='segments', width=1
        )
        self.view.add(self._axis_lines)
        self._axis_lines.visible = False

        lbl_zero_kwargs = dict(color=(1.0, 1.0, 0.3, 1.0), parent=self.view.scene,
                               font_size=12, anchor_x='left', anchor_y='baseline')
        self._zero_labels = [
            scene.visuals.Text('0', pos=(0, 0, 0), **lbl_zero_kwargs),
            scene.visuals.Text('0', pos=(0, 0, 0), **lbl_zero_kwargs),
            scene.visuals.Text('0', pos=(0, 0, 0), **lbl_zero_kwargs),
        ]
        for lbl in self._zero_labels:
            lbl.visible = False

        self.show_decorrelated = False

        self.update_markers()
        self.canvas.events.mouse_wheel.connect(lambda ev: None)

    def update_mode_label(self, fib_norm_mode=False):
        if fib_norm_mode:
            text = "Radial push"
        elif self.show_decorrelated:
            text = "ZCA space"
        else:
            text = "sRGB cube"
        self._cloud_mode_label.setText(text)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        lbl_h = self._cloud_mode_label.sizeHint().height()
        side = max(1, min(self.width(), self.height() - lbl_h - 4))
        self.canvas.size = (side, side)
        self.canvas.update()

    def toggle_decorrelated(self, fib_norm_mode=False):
        self.show_decorrelated = not self.show_decorrelated
        for lbl in self._cube_labels:
            lbl.visible = not self.show_decorrelated
        self._rgb_axes.visible = not self.show_decorrelated
        self._axis_lines.visible = self.show_decorrelated
        for lbl in self._zero_labels:
            lbl.visible = self.show_decorrelated
        self.update_mode_label(fib_norm_mode)
        self.update_markers()
        self.canvas.update()

    def update_markers(self):
        st = self.state; samples = st.samples_lin
        R = st.rot.Ruser
        z  = (samples - st.mu_lin) @ st.W_lin.T
        z2 = z @ R

        if self.show_decorrelated:
            if st.fib_norm_mode and hasattr(st, 'samples_zca_fib') and st.samples_zca_fib is not None:
                zf = st.samples_zca_fib @ R
                half = getattr(st, '_fib_zca_half', 1.0)
                pos = zf * float(st.push_factor) / (2.0 * half) + 0.5
            else:
                norms_z = np.linalg.norm(z, axis=1)
                half = float(np.percentile(norms_z, 95))
                if half < 1e-8:
                    half = 1.0
                coords = z2 * float(st.push_factor)
                pos = coords / (2.0 * half) + 0.5
            eff_Winv = np.eye(3, dtype=np.float32) + float(st.variance_restore) * (st.Winv_lin - np.eye(3, dtype=np.float32))
            xo_col = np.clip(st.mu_lin + (z2 @ eff_Winv.T), 0.0, 1.0)
            col = ColorUtils.linear_to_srgb_np(xo_col)
            # Origine toujours au centre du cube
            origin = np.array([0.5, 0.5, 0.5])
            eps = 0.02
            # Trois lignes d'axes traversant le centre
            self._axis_lines.set_data(pos=np.array([
                [0, 0.5, 0.5], [1, 0.5, 0.5],  # X axis
                [0.5, 0, 0.5], [0.5, 1, 0.5],  # Y axis
                [0.5, 0.5, 0], [0.5, 0.5, 1],  # Z axis
            ], dtype=np.float32))
            # Labels "0" fixes au centre sur chaque axe
            self._zero_labels[0].pos = (0.5, -eps, -eps)   # X
            self._zero_labels[1].pos = (-eps, 0.5, -eps)   # Y
            self._zero_labels[2].pos = (-eps, -eps, 0.5)   # Z
        else:
            eff_Winv = np.eye(3, dtype=np.float32) + float(st.variance_restore) * (st.Winv_lin - np.eye(3, dtype=np.float32))
            xo = np.clip(st.mu_lin + (z2 * float(st.push_factor) @ eff_Winv.T), 0.0, 1.0)
            pos = xo
            col = ColorUtils.linear_to_srgb_np(xo)

        rgba = np.c_[col, np.ones((col.shape[0],1), dtype=np.float32)]
        self.markers.set_data(pos=pos, face_color=rgba, size=2.0, edge_width=0.0)


# ------------ Info panel ------------
def _deg_norm(rad):
    d = np.degrees(rad)
    return ((d + 180.0) % 360.0) - 180.0

class InfoPanel(QtWidgets.QGroupBox):
    def __init__(self, app_state, parent=None):
        super().__init__("Rotation info", parent)
        self.state = app_state
        self.setMinimumWidth(260)
        lay = QtWidgets.QFormLayout(self); lay.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.lbl_ax = QtWidgets.QLabel("—")
        self.lbl_ay = QtWidgets.QLabel("—")
        self.lbl_az = QtWidgets.QLabel("—")
        self.lbl_qw = QtWidgets.QLabel("—")
        self.lbl_qx = QtWidgets.QLabel("—")
        self.lbl_qy = QtWidgets.QLabel("—")
        self.lbl_qz = QtWidgets.QLabel("—")

        font = self.font(); font.setFamily("Menlo" if IS_MACOS else "Monospace" if IS_LINUX else "Consolas")
        for w in (self.lbl_ax,self.lbl_ay,self.lbl_az,self.lbl_qw,self.lbl_qx,self.lbl_qy,self.lbl_qz):
            w.setFont(font)

        lay.addRow("angle X (°):", self.lbl_ax)
        lay.addRow("angle Y (°):", self.lbl_ay)
        lay.addRow("angle Z (°):", self.lbl_az)
        line = QtWidgets.QFrame(); line.setFrameShape(QtWidgets.QFrame.HLine); line.setFrameShadow(QtWidgets.QFrame.Sunken)
        lay.addRow(line)
        lay.addRow("Quaternion:", QtWidgets.QLabel(""))
        lay.addRow("  w:", self.lbl_qw)
        lay.addRow("  x:", self.lbl_qx)
        lay.addRow("  y:", self.lbl_qy)
        lay.addRow("  z:", self.lbl_qz)

    def refresh(self):
        ax=_deg_norm(self.state.rot.ang_x); ay=_deg_norm(self.state.rot.ang_y); az=_deg_norm(self.state.rot.ang_z)
        self.lbl_ax.setText(f"{ax:7.2f}"); self.lbl_ay.setText(f"{ay:7.2f}"); self.lbl_az.setText(f"{az:7.2f}")
        Rq = self.state.rot.Ruser
        q = RotationState.R_to_quat(Rq)
        self.lbl_qw.setText(f"{q[0]: 7.4f}"); self.lbl_qx.setText(f"{q[1]: 7.4f}")
        self.lbl_qy.setText(f"{q[2]: 7.4f}"); self.lbl_qz.setText(f"{q[3]: 7.4f}")

def _make_separator():
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Sunken)
    return line


# ------------ Startup controls hint dialog ------------
class ControlsHintDialog(QtWidgets.QDialog):
    """One-shot dialog that summarises mouse controls for rotation.
    Shown at startup unless the user checked 'Don't show again'."""

    SETTINGS_KEY = "show_controls_hint"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Controls")
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QtWidgets.QLabel("<b>Mouse/key controls & Rotation</b>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        hint = QtWidgets.QLabel("Try rotating the image along all three axes: X, Y and Z... and hope!")
        hint.setAlignment(QtCore.Qt.AlignCenter)
        hint.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(hint)

        layout.addWidget(_make_separator())

        # Controls table
        grid = QtWidgets.QGridLayout()
        grid.setColumnMinimumWidth(0, 160)
        grid.setColumnMinimumWidth(1, 200)
        _mod = "⌘" if IS_MACOS else "Ctrl"
        rows = [
            ("Scroll wheel or <b>Q</b> / <b>D</b>",  "Rotate around <b>X</b> axis"),
            ("Right drag or <b>Z</b> / <b>S</b>",   "Rotate around <b>Y</b> axis"),
            ("Left drag or <b>A</b> / <b>E</b>",    "Rotate around <b>Z</b> axis"),
            (f"{_mod} + Scroll wheel",               "Zoom in / out"),
            ("<b>R</b> key",              "Reset to base state (shown in toolbar)"),
            ("<b>N</b> key",              "Toggle 3D view: sRGB cube / decorrelated space"),
            (f"{_mod}+1 / 2 / 3",        "Swap colour view with R / G / B"),
            ("<b>P</b> key",              "Save a preset checkpoint (shown in toolbar)"),
            ("<b>O</b> key",              "Toggle ROI paint mode (shown in toolbar)"),
            ("<b>I</b> key",              "Toggle ICA mode (shown in toolbar)"),
            ("<b>F</b> key",              "Toggle Radial Push mode (shown in toolbar)"),
            ("<b>T</b> key",              "Toggle Split view (before / after)"),
        ]
        for row_idx, (action, effect) in enumerate(rows):
            lbl_action = QtWidgets.QLabel(action)
            lbl_effect = QtWidgets.QLabel(effect)
            lbl_effect.setTextFormat(QtCore.Qt.RichText)
            grid.addWidget(lbl_action, row_idx, 0)
            grid.addWidget(lbl_effect, row_idx, 1)
        layout.addLayout(grid)

        layout.addWidget(_make_separator())

        # Don't show again checkbox + OK button
        bottom = QtWidgets.QHBoxLayout()
        self.chk_no_show = QtWidgets.QCheckBox("Don't show again")
        bottom.addWidget(self.chk_no_show)
        bottom.addStretch()
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self._on_ok)
        bottom.addWidget(ok_btn)
        layout.addLayout(bottom)

    def _on_ok(self):
        if self.chk_no_show.isChecked():
            QtCore.QSettings("Rascal", "rascal").setValue(self.SETTINGS_KEY, False)
        else:
            QtCore.QSettings("Rascal", "rascal").setValue(self.SETTINGS_KEY, True)
        self.accept()

    @staticmethod
    def should_show():
        val = QtCore.QSettings("Rascal", "rascal").value(
            ControlsHintDialog.SETTINGS_KEY, defaultValue=True
        )
        # QSettings returns strings on some platforms
        if isinstance(val, str):
            return val.lower() != "false"
        return bool(val)

    @classmethod
    def show_if_needed(cls, parent=None):
        if cls.should_show():
            dlg = cls(parent)
            dlg.exec_()


# ------------ Image quality assessment ------------
def _sci_gaussian_blur(gray, sigma=1.0):
    h, w = gray.shape
    y = np.fft.fftfreq(h).reshape(-1, 1)
    x = np.fft.fftfreq(w).reshape(1, -1)
    kernel = np.exp(-2 * (np.pi ** 2) * sigma ** 2 * (x ** 2 + y ** 2))
    return np.real(np.fft.ifft2(np.fft.fft2(gray) * kernel))


def sci_score(path, progress_cb=None):
    def _progress(value, message):
        if progress_cb is not None:
            progress_cb(value, message)

    from PIL import Image as _Image
    _raw = _Image.open(path)
    pil_img = _raw.convert("RGB")
    img = np.array(pil_img)
    h, w, _ = img.shape
    gray = np.array(pil_img.convert("L")).astype(np.float32)

    # Resolution
    R = (h * w) / 2_000_000
    _progress(25, "Image quality: resolution (25%)")

    # Quantization (1.0 for lossless formats without JPEG quantization tables)
    try:
        q = np.mean([np.mean(v) for v in _raw.quantization.values()])
        Q = float(np.clip(1 - q / 255, 0, 1))
    except Exception:
        # PNG, TIFF, and other lossless formats have no quantization tables
        Q = 1.0
    _progress(50, "Image quality: quantization (50%)")

    # Blockiness
    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    cols = np.arange(7, w - 1, 8)
    rows = np.arange(7, h - 1, 8)
    block_h = diff_h[:, cols].mean() if len(cols) > 0 else 0
    block_v = diff_v[rows, :].mean() if len(rows) > 0 else 0
    G_block = (block_h + block_v) / 2
    G_nat = (diff_h.mean() + diff_v.mean()) / 2 + 1e-6
    B = float(2 / (G_block / G_nat + 1))
    _progress(75, "Image quality: blockiness (75%)")

    # High-frequency
    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f))
    r = int(min(h, w) * 0.05)
    HF = 1 - (mag[h//2 - r:h//2 + r, w//2 - r:w//2 + r].sum() / (mag.sum() + 1e-8))
    blur = _sci_gaussian_blur(gray, sigma=1.0)
    mag_blur = np.abs(np.fft.fftshift(np.fft.fft2(blur)))
    HF_ref = 1 - (mag_blur[h//2 - r:h//2 + r, w//2 - r:w//2 + r].sum() / (mag_blur.sum() + 1e-8))
    HF = float(np.clip(HF / (HF_ref + 1e-8), 0, 1))
    _progress(100, "Image quality: high frequency (100%)")

    # Labels
    R_label = ("sufficient resolution" if R > 0.8 else
               "borderline" if R >= 0.5 else
               "low" if R >= 0.2 else "unusable")
    Q_label = ("near lossless" if Q > 0.9 else
               "light compression" if Q >= 0.7 else
               "moderate compression" if Q >= 0.4 else "heavy compression")
    B_label = ("no block artefacts" if B > 0.9 else
               "slight" if B >= 0.7 else
               "visible" if B >= 0.5 else "strong")
    HF_label = ("very detailed" if HF > 0.9 else
                "good" if HF >= 0.7 else
                "notable loss" if HF >= 0.5 else "smoothed image")

    # SCI includes resolution R with equal weighting (25% each component)
    SCI = 100 * (0.25 * min(R, 1.0) + 0.25 * Q + 0.25 * B + 0.25 * HF)

    return dict(h=h, w=w, R=R, R_label=R_label,
                Q=Q, Q_label=Q_label,
                B=B, B_label=B_label,
                HF=HF, HF_label=HF_label,
                SCI=SCI)


class _CropLabel(QtWidgets.QLabel):
    """QLabel subclass that lets the user draw a rubber-band rectangle over the image."""
    cropChanged = QtCore.pyqtSignal()

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self._original_pixmap = pixmap
        self.setPixmap(pixmap)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self._origin = None
        self._rect = QtCore.QRect()
        self._drawing = False
        self._crop_disabled = False

    def _image_rect(self):
        pw, ph = self.pixmap().width(), self.pixmap().height()
        lw, lh = self.width(), self.height()
        x = max((lw - pw) // 2, 0)
        y = max((lh - ph) // 2, 0)
        return QtCore.QRect(x, y, pw, ph)

    def mousePressEvent(self, event):
        if self._crop_disabled:
            return
        if event.button() == QtCore.Qt.LeftButton:
            self._origin = event.pos()
            self._rect = QtCore.QRect(self._origin, QtCore.QSize())
            self._drawing = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._crop_disabled:
            return
        if self._drawing and self._origin is not None:
            self._rect = QtCore.QRect(self._origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._drawing = False
            self.cropChanged.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._rect.isNull():
            painter = QtGui.QPainter(self)
            pen = QtGui.QPen(QtGui.QColor(255, 200, 0), 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtGui.QColor(255, 255, 0, 40))
            painter.drawRect(self._rect)
            painter.end()

    def get_crop_rect_normalized(self):
        """Returns (x, y, w, h) in normalized [0,1] coordinates relative to the pixmap, or None."""
        if self._rect.isNull():
            return None
        ir = self._image_rect()
        if ir.width() <= 0 or ir.height() <= 0:
            return None
        clipped = self._rect.intersected(ir)
        if clipped.width() < 4 or clipped.height() < 4:
            return None
        x = (clipped.x() - ir.x()) / ir.width()
        y = (clipped.y() - ir.y()) / ir.height()
        w = clipped.width() / ir.width()
        h = clipped.height() / ir.height()
        return (x, y, w, h)


class ImageCropDialog(QtWidgets.QDialog):
    """Dialog that displays an image and allows the user to select a rectangular crop region.
    Includes brightness and contrast sliders with live preview."""

    def __init__(self, image_path, parent=None, disable_crop_resize=False, image_array=None, prev_params=None):
        super().__init__(parent)
        self.setWindowTitle("Modify Image — Crop")
        # Use parent's screen if available (multi-monitor support), fallback to primary
        if parent is not None:
            parent_screen = parent.screen()
            self._screen_geo = parent_screen.availableGeometry() if parent_screen else QtWidgets.QApplication.primaryScreen().availableGeometry()
        else:
            self._screen_geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.setMaximumWidth(self._screen_geo.width() - 20)
        self.cropped_array = None  # will hold the cropped numpy array on accept
        self._image_path = image_path
        self._disable_crop_resize = disable_crop_resize
        self._prev_params = prev_params or {}
        self._contrast = self._prev_params.get('contrast', 0)
        # HSL per-channel: keys are 'Master','Reds','Greens','Blues'
        self._hsl = self._prev_params.get('hsl') or {
            'Master': [0, 0, 0], 'Reds': [0, 0, 0],
            'Greens': [0, 0, 0], 'Blues': [0, 0, 0],
        }
        # Ensure all channels exist (forward compat)
        for ch in ('Master', 'Reds', 'Greens', 'Blues'):
            if ch not in self._hsl:
                self._hsl[ch] = [0, 0, 0]

        if image_array is not None:
            raw = image_array
        else:
            raw = imageio.imread(pathlib.Path(image_path))
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=2)
        raw = raw[..., :3]
        if image_array is None:
            raw = _apply_exif_orientation(raw, image_path)
        self._full_image = raw

        # Normalise to float32 [0,1] for adjustments
        if self._full_image.dtype == np.uint16:
            self._full_f32 = self._full_image.astype(np.float32) / 65535.0
        elif self._full_image.dtype in (np.float32, np.float64):
            self._full_f32 = self._full_image.astype(np.float32)
        else:
            self._full_f32 = self._full_image.astype(np.float32) / 255.0

        # Precompute a display-sized downscaled copy for fast live preview
        h_img, w_img = self._full_f32.shape[:2]
        _scr = self._screen_geo
        _CTRL_OVERHEAD_H = 520  # title bar + info label + contrast + HSL box + resize row + buttons + all margins (tuned for 1080p screens)
        _max_disp_w = _scr.width() // 2
        _max_disp_h = min(800, _scr.height() - _CTRL_OVERHEAD_H)
        _max_disp_w = max(200, _max_disp_w)
        _max_disp_h = max(200, _max_disp_h)
        self._display_max = min(_max_disp_w, _max_disp_h)  # kept for API compat
        self._scale = min(_max_disp_w / w_img, _max_disp_h / h_img, 1.0)
        self._disp_w = int(w_img * self._scale)
        self._disp_h = int(h_img * self._scale)
        # Minimum size prevents window from being resized smaller than the image preview
        _min_w = max(400, self._disp_w + 40)  # image width + margins + controls
        _min_h = max(400, self._disp_h + 300)  # image height + space for controls
        self.setMinimumSize(_min_w, _min_h)
        if self._scale < 1.0:
            from PIL import Image
            pil_img = Image.fromarray((np.clip(self._full_f32, 0, 1) * 255).astype(np.uint8))
            pil_small = pil_img.resize((self._disp_w, self._disp_h), Image.LANCZOS)
            self._preview_f32 = np.asarray(pil_small, dtype=np.float32) / 255.0
        else:
            self._preview_f32 = self._full_f32

        # Pre-compute HSL of preview for fast slider updates
        self._preview_hsl = self._rgb_to_hsl(self._preview_f32)

        # Debounce timer for slider updates (50ms)
        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(50)
        self._update_timer.timeout.connect(self._do_update_preview)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # reduce margins to save vertical space
        layout.setSpacing(4)

        if self._disable_crop_resize:
            info = QtWidgets.QLabel("Adjust contrast and HSL, then click <b>OK</b>.<br>(Crop and resize are disabled in 3D mode.)")
        else:
            info = QtWidgets.QLabel("Draw a rectangle on the image to define the crop area, then click <b>OK</b>.")
        info.setAlignment(QtCore.Qt.AlignCenter)
        info.setWordWrap(True)
        layout.addWidget(info)

        # Display image
        self._crop_label = _CropLabel(self._build_pixmap())
        self._crop_label.setFixedSize(self._disp_w + 4, self._disp_h + 4)  # minimal padding around image
        if self._disable_crop_resize:
            self._crop_label._crop_disabled = True
        crop_row = QtWidgets.QHBoxLayout()
        crop_row.addStretch()
        crop_row.addWidget(self._crop_label)
        self._crop_clear_btn = QtWidgets.QToolButton()
        self._crop_clear_btn.setText("\u2715")
        self._crop_clear_btn.setToolTip("Clear crop selection")
        self._crop_clear_btn.setFixedSize(20, 20)
        self._crop_clear_btn.clicked.connect(self._clear_crop)
        if self._disable_crop_resize:
            self._crop_clear_btn.setVisible(False)
        crop_row.addWidget(self._crop_clear_btn, alignment=QtCore.Qt.AlignTop)
        crop_row.addStretch()
        layout.addLayout(crop_row)

        # Contrast slider
        ct_layout = QtWidgets.QHBoxLayout()
        ct_layout.addWidget(QtWidgets.QLabel("Contrast:"))
        self._ct_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._ct_slider.setRange(-100, 100)
        self._ct_slider.setValue(self._contrast)
        self._ct_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._ct_slider.setTickInterval(25)
        self._ct_label = QtWidgets.QLabel(str(self._contrast))
        self._ct_label.setFixedWidth(35)
        self._ct_label.setAlignment(QtCore.Qt.AlignCenter)
        ct_layout.addWidget(self._ct_slider)
        ct_layout.addWidget(self._ct_label)
        self._ct_clear_btn = QtWidgets.QToolButton()
        self._ct_clear_btn.setText("\u2715")
        self._ct_clear_btn.setToolTip("Reset contrast to 0")
        self._ct_clear_btn.setFixedSize(20, 20)
        self._ct_clear_btn.clicked.connect(self._clear_contrast)
        ct_layout.addWidget(self._ct_clear_btn)
        layout.addLayout(ct_layout)

        self._ct_slider.valueChanged.connect(self._on_slider_changed)

        # --- HSL per-channel controls ---
        hsl_group = QtWidgets.QGroupBox("HSL Adjustments")
        hsl_vlayout = QtWidgets.QVBoxLayout(hsl_group)
        hsl_vlayout.setContentsMargins(6, 4, 6, 4)
        hsl_vlayout.setSpacing(4)

        ch_layout = QtWidgets.QHBoxLayout()
        ch_layout.addWidget(QtWidgets.QLabel("Channel:"))
        self._hsl_channel_combo = QtWidgets.QComboBox()
        self._hsl_channel_combo.addItems(['Master', 'Reds', 'Greens', 'Blues'])
        ch_layout.addWidget(self._hsl_channel_combo)
        ch_layout.addStretch()
        self._hsl_clear_btn = QtWidgets.QToolButton()
        self._hsl_clear_btn.setText("\u2715")
        self._hsl_clear_btn.setToolTip("Reset all HSL channels to 0")
        self._hsl_clear_btn.setFixedSize(20, 20)
        self._hsl_clear_btn.clicked.connect(self._clear_hsl)
        ch_layout.addWidget(self._hsl_clear_btn)
        hsl_vlayout.addLayout(ch_layout)

        # Hue slider (-180..+180)
        hue_layout = QtWidgets.QHBoxLayout()
        hue_layout.addWidget(QtWidgets.QLabel("Hue:"))
        self._hue_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._hue_slider.setRange(-180, 180)
        self._hue_slider.setValue(self._hsl['Master'][0])
        self._hue_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._hue_slider.setTickInterval(45)
        self._hue_label = QtWidgets.QLabel(str(self._hsl['Master'][0]))
        self._hue_label.setFixedWidth(35)
        self._hue_label.setAlignment(QtCore.Qt.AlignCenter)
        hue_layout.addWidget(self._hue_slider)
        hue_layout.addWidget(self._hue_label)
        hsl_vlayout.addLayout(hue_layout)

        # Saturation slider (-100..+100)
        sat_layout = QtWidgets.QHBoxLayout()
        sat_layout.addWidget(QtWidgets.QLabel("Saturation:"))
        self._sat_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._sat_slider.setRange(-100, 100)
        self._sat_slider.setValue(self._hsl['Master'][1])
        self._sat_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._sat_slider.setTickInterval(25)
        self._sat_label = QtWidgets.QLabel(str(self._hsl['Master'][1]))
        self._sat_label.setFixedWidth(35)
        self._sat_label.setAlignment(QtCore.Qt.AlignCenter)
        sat_layout.addWidget(self._sat_slider)
        sat_layout.addWidget(self._sat_label)
        hsl_vlayout.addLayout(sat_layout)

        # Lightness slider (-100..+100)
        lit_layout = QtWidgets.QHBoxLayout()
        lit_layout.addWidget(QtWidgets.QLabel("Lightness:"))
        self._lit_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._lit_slider.setRange(-100, 100)
        self._lit_slider.setValue(self._hsl['Master'][2])
        self._lit_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self._lit_slider.setTickInterval(25)
        self._lit_label = QtWidgets.QLabel(str(self._hsl['Master'][2]))
        self._lit_label.setFixedWidth(35)
        self._lit_label.setAlignment(QtCore.Qt.AlignCenter)
        lit_layout.addWidget(self._lit_slider)
        lit_layout.addWidget(self._lit_label)
        hsl_vlayout.addLayout(lit_layout)

        layout.addWidget(hsl_group)

        self._hsl_channel_combo.currentTextChanged.connect(self._on_hsl_channel_changed)
        self._hue_slider.valueChanged.connect(self._on_hsl_slider_changed)
        self._sat_slider.valueChanged.connect(self._on_hsl_slider_changed)
        self._lit_slider.valueChanged.connect(self._on_hsl_slider_changed)

        # Resize controls
        h_img, w_img = self._full_image.shape[:2]
        self._orig_w, self._orig_h = w_img, h_img
        self._aspect = w_img / h_img
        self._resize_lock = False  # prevent recursive updates

        rz_layout = QtWidgets.QHBoxLayout()
        rz_layout.addWidget(QtWidgets.QLabel("Width:"))
        self._w_spin = QtWidgets.QSpinBox()
        self._w_spin.setRange(4, w_img)
        self._w_spin.setValue(w_img)
        self._w_spin.setSuffix(" px")
        rz_layout.addWidget(self._w_spin)
        rz_layout.addWidget(QtWidgets.QLabel("Height:"))
        self._h_spin = QtWidgets.QSpinBox()
        self._h_spin.setRange(4, h_img)
        self._h_spin.setValue(h_img)
        self._h_spin.setSuffix(" px")
        rz_layout.addWidget(self._h_spin)
        self._rz_clear_btn = QtWidgets.QToolButton()
        self._rz_clear_btn.setText("\u2715")
        self._rz_clear_btn.setToolTip("Reset to original dimensions")
        self._rz_clear_btn.setFixedSize(20, 20)
        self._rz_clear_btn.clicked.connect(self._clear_resize)
        rz_layout.addWidget(self._rz_clear_btn)
        rz_layout.addStretch()
        layout.addLayout(rz_layout)

        if self._disable_crop_resize:
            self._w_spin.setEnabled(False)
            self._h_spin.setEnabled(False)
            self._rz_clear_btn.setVisible(False)

        self._w_spin.valueChanged.connect(self._on_w_changed)
        self._h_spin.valueChanged.connect(self._on_h_changed)
        self._crop_label.cropChanged.connect(self._on_crop_changed)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        # Show Reset only when the image was previously modified
        _has_modifications = bool(self._prev_params) and (
            self._prev_params.get('contrast', 0) != 0
            or self._prev_params.get('crop_rect') is not None
            or any(any(v != 0 for v in self._prev_params.get('hsl', {}).get(ch, [0, 0, 0]))
                   for ch in ('Master', 'Reds', 'Greens', 'Blues'))
        )
        self._has_modifications = _has_modifications
        if _has_modifications:
            self._btn_reset = QtWidgets.QPushButton("Reset")
            self._btn_reset.setToolTip("Reset all adjustments to the original image")
            self._btn_reset.clicked.connect(self._on_reset)
            btn_layout.addWidget(self._btn_reset)
        btn_layout.addStretch()
        self._btn_ok = QtWidgets.QPushButton("OK")
        self._btn_ok.clicked.connect(self._on_ok)
        btn_layout.addWidget(self._btn_ok)
        layout.addLayout(btn_layout)

        # Restore previous crop rectangle if available
        prev_crop = self._prev_params.get('crop_rect')
        if prev_crop is not None and not self._disable_crop_resize:
            nx, ny, nw, nh = prev_crop
            ir = self._crop_label._image_rect()
            px = int(round(nx * ir.width())) + ir.x()
            py = int(round(ny * ir.height())) + ir.y()
            pw = int(round(nw * ir.width()))
            ph = int(round(nh * ir.height()))
            self._crop_label._rect = QtCore.QRect(px, py, pw, ph)
            self._crop_label.update()
            self._on_crop_changed()

    def showEvent(self, event):
        super().showEvent(event)
        _scr = self._screen_geo
        _x = _scr.x() + (_scr.width() - self.width()) // 2
        _y = _scr.y() + max(20, int(_scr.height() * 0.05))
        self.move(_x, _y)

    def _on_hsl_channel_changed(self, channel_name):
        """When the user switches channel, update sliders to reflect stored values."""
        # Ensure the channel exists in _hsl dict
        if channel_name not in self._hsl:
            self._hsl[channel_name] = [0, 0, 0]
        vals = self._hsl[channel_name]
        self._hue_slider.blockSignals(True)
        self._sat_slider.blockSignals(True)
        self._lit_slider.blockSignals(True)
        self._hue_slider.setValue(vals[0])
        self._sat_slider.setValue(vals[1])
        self._lit_slider.setValue(vals[2])
        self._hue_label.setText(str(vals[0]))
        self._sat_label.setText(str(vals[1]))
        self._lit_label.setText(str(vals[2]))
        self._hue_slider.blockSignals(False)
        self._sat_slider.blockSignals(False)
        self._lit_slider.blockSignals(False)

    def _on_hsl_slider_changed(self):
        """Store current HSL slider values for the active channel and refresh preview."""
        ch = self._hsl_channel_combo.currentText()
        h_val = self._hue_slider.value()
        s_val = self._sat_slider.value()
        l_val = self._lit_slider.value()
        # Store as new list to avoid reference issues
        self._hsl[ch] = [int(h_val), int(s_val), int(l_val)]
        self._hue_label.setText(str(h_val))
        self._sat_label.setText(str(s_val))
        self._lit_label.setText(str(l_val))
        # Refresh preview
        self._on_slider_changed()

    @staticmethod
    def _rgb_to_hsl(rgb):
        """Convert float32 RGB [0,1] array (H,W,3) to HSL (H,W,3). H in [0,360], S,L in [0,1]."""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        # Lightness
        L = (cmax + cmin) * 0.5
        # Saturation
        S = np.where(delta == 0, 0.0,
                     delta / (1.0 - np.abs(2.0 * L - 1.0) + 1e-7))
        # Hue
        H = np.zeros_like(L)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        H[mask_r] = (60.0 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-7)) % 6.0))
        H[mask_g] = (60.0 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-7)) + 2.0))
        H[mask_b] = (60.0 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-7)) + 4.0))
        hsl = np.stack([H, S, L], axis=-1)
        return hsl

    @staticmethod
    def _hsl_to_rgb(hsl):
        """Convert HSL (H,W,3) back to float32 RGB [0,1]. H in [0,360], S,L in [0,1]."""
        H, S, L = hsl[..., 0], hsl[..., 1], hsl[..., 2]
        C = (1.0 - np.abs(2.0 * L - 1.0)) * S
        H2 = H / 60.0
        X = C * (1.0 - np.abs(H2 % 2.0 - 1.0))
        m = L - C * 0.5

        r = np.zeros_like(H)
        g = np.zeros_like(H)
        b = np.zeros_like(H)

        idx = (H2 >= 0) & (H2 < 1)
        r[idx] = C[idx]; g[idx] = X[idx]
        idx = (H2 >= 1) & (H2 < 2)
        r[idx] = X[idx]; g[idx] = C[idx]
        idx = (H2 >= 2) & (H2 < 3)
        g[idx] = C[idx]; b[idx] = X[idx]
        idx = (H2 >= 3) & (H2 < 4)
        g[idx] = X[idx]; b[idx] = C[idx]
        idx = (H2 >= 4) & (H2 < 5)
        r[idx] = X[idx]; b[idx] = C[idx]
        idx = (H2 >= 5) & (H2 < 6)
        r[idx] = C[idx]; b[idx] = X[idx]

        rgb = np.stack([r + m, g + m, b + m], axis=-1)
        return np.clip(rgb, 0.0, 1.0)

    def _apply_hsl(self, img_f32, precomputed_hsl=None):
        """Apply HSL adjustments to a float32 [0,1] RGB image.

        Master acts on all pixels. Reds/Greens/Blues act with a soft mask based
        on the actual RGB contribution of each channel, rather than with a
        narrow hue-only selector. This avoids the previous behaviour where the
        R/G/B sliders appeared inactive on desaturated or yellow-beige images.
        """
        any_nonzero = any(
            any(v != 0 for v in self._hsl.get(ch, [0, 0, 0]))
            for ch in ('Master', 'Reds', 'Greens', 'Blues')
        )
        if not any_nonzero:
            return img_f32

        if precomputed_hsl is not None:
            hsl = precomputed_hsl
        else:
            hsl = self._rgb_to_hsl(img_f32)
        H = hsl[..., 0].copy()
        S = hsl[..., 1].copy()
        L = hsl[..., 2].copy()

        # Soft RGB-channel masks. They sum approximately to 1 for each pixel.
        # A parchment/yellowish pixel still has a strong red/green contribution,
        # whereas the previous strict hue masks could be exactly zero.
        rgb = np.clip(img_f32, 0.0, 1.0).astype(np.float32)
        denom = np.sum(rgb, axis=-1) + 1e-6
        masks = {
            'Reds':   (rgb[..., 0] / denom).astype(np.float32),
            'Greens': (rgb[..., 1] / denom).astype(np.float32),
            'Blues':  (rgb[..., 2] / denom).astype(np.float32),
        }

        for ch_name in ('Master', 'Reds', 'Greens', 'Blues'):
            if ch_name not in self._hsl:
                continue
            dh, ds, dl = self._hsl[ch_name]
            if dh == 0 and ds == 0 and dl == 0:
                continue
            if ch_name == 'Master':
                w = np.float32(1.0)
            else:
                w = masks[ch_name]
            if dh != 0:
                H += np.float32(dh) * w
            if ds != 0:
                S *= (1.0 + np.float32(ds / 100.0) * w)
            if dl != 0:
                L += (np.float32(dl / 100.0) * 0.5) * w

        H %= 360.0
        np.clip(S, 0.0, 1.0, out=S)
        np.clip(L, 0.0, 1.0, out=L)
        return self._hsl_to_rgb(np.stack([H, S, L], axis=-1))

    def _apply_bc(self, img_f32):
        """Apply contrast to a float32 [0,1] image. Returns float32 [0,1]."""
        c = self._contrast / 100.0    # -1..+1
        if c == 0:
            return img_f32
        # Contrast factor: map [-1,+1] to [0.5, 2.0] via exponential
        factor = np.float32(2.0 ** c)
        out = (img_f32 - 0.5) * factor + 0.5
        return np.clip(out, 0.0, 1.0)

    def _build_pixmap(self):
        """Build the display QPixmap with current contrast/HSL (uses downscaled preview)."""
        adjusted = self._apply_bc(self._preview_f32)
        # HSL cache is only valid when contrast is 0 (contrast changes RGB values,
        # so the precomputed HSL of the original image would be invalid)
        hsl_cache = self._preview_hsl if self._contrast == 0 else None
        adjusted = self._apply_hsl(adjusted, precomputed_hsl=hsl_cache)
        disp_u8 = (adjusted * 255).astype(np.uint8)
        self._disp_u8 = np.ascontiguousarray(disp_u8)
        h_img, w_img = disp_u8.shape[:2]
        self._qimg = QtGui.QImage(self._disp_u8.data, w_img, h_img, w_img * 3, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(self._qimg)

    def _on_slider_changed(self):
        self._contrast = self._ct_slider.value()
        self._ct_label.setText(str(self._contrast))
        # Debounce: schedule preview update
        self._update_timer.start()

    def _do_update_preview(self):
        """Actually rebuild the preview pixmap (called after debounce timer)."""
        old_rect = self._crop_label._rect
        pixmap = self._build_pixmap()
        self._crop_label._original_pixmap = pixmap
        self._crop_label.setPixmap(pixmap)
        self._crop_label._rect = old_rect
        self._crop_label.update()

    def _on_crop_changed(self):
        """Update resize spinboxes to reflect the crop region dimensions."""
        rect = self._crop_label.get_crop_rect_normalized()
        if rect is not None:
            x, y, w, h = rect
            crop_w = max(4, int(round(w * self._orig_w)))
            crop_h = max(4, int(round(h * self._orig_h)))
        else:
            crop_w, crop_h = self._orig_w, self._orig_h
        self._resize_lock = True
        self._aspect = crop_w / crop_h
        self._w_spin.setMaximum(crop_w)
        self._h_spin.setMaximum(crop_h)
        self._w_spin.setValue(crop_w)
        self._h_spin.setValue(crop_h)
        self._resize_lock = False

    def _on_w_changed(self, val):
        if self._resize_lock:
            return
        self._resize_lock = True
        new_h = max(4, int(round(val / self._aspect)))
        self._h_spin.setValue(min(new_h, self._orig_h))
        self._resize_lock = False

    def _on_h_changed(self, val):
        if self._resize_lock:
            return
        self._resize_lock = True
        new_w = max(4, int(round(val * self._aspect)))
        self._w_spin.setValue(min(new_w, self._orig_w))
        self._resize_lock = False

    def _clear_crop(self):
        """Clear the crop rectangle."""
        self._crop_label._rect = QtCore.QRect()
        self._crop_label.update()
        self._on_crop_changed()
        self._do_update_preview()

    def _clear_contrast(self):
        """Reset contrast to 0."""
        self._contrast = 0
        self._ct_slider.blockSignals(True)
        self._ct_slider.setValue(0)
        self._ct_label.setText("0")
        self._ct_slider.blockSignals(False)
        self._do_update_preview()

    def _clear_hsl(self):
        """Reset all HSL channels to 0."""
        for ch in ('Master', 'Reds', 'Greens', 'Blues'):
            self._hsl[ch] = [0, 0, 0]
        self._hsl_channel_combo.setCurrentText('Master')
        self._hue_slider.blockSignals(True)
        self._sat_slider.blockSignals(True)
        self._lit_slider.blockSignals(True)
        self._hue_slider.setValue(0)
        self._sat_slider.setValue(0)
        self._lit_slider.setValue(0)
        self._hue_label.setText("0")
        self._sat_label.setText("0")
        self._lit_label.setText("0")
        self._hue_slider.blockSignals(False)
        self._sat_slider.blockSignals(False)
        self._lit_slider.blockSignals(False)
        self._do_update_preview()

    def _clear_resize(self):
        """Reset dimensions to original image size."""
        self._resize_lock = True
        self._w_spin.setValue(self._orig_w)
        self._h_spin.setValue(self._orig_h)
        self._resize_lock = False

    def _on_reset(self):
        """Reset all adjustments to zero, clear crop, restore dimensions, and apply immediately."""
        # Reset contrast
        self._contrast = 0
        self._ct_slider.blockSignals(True)
        self._ct_slider.setValue(0)
        self._ct_label.setText("0")
        self._ct_slider.blockSignals(False)
        # Reset HSL for all channels
        for ch in ('Master', 'Reds', 'Greens', 'Blues'):
            self._hsl[ch] = [0, 0, 0]
        self._hsl_channel_combo.setCurrentText('Master')
        self._hue_slider.blockSignals(True)
        self._sat_slider.blockSignals(True)
        self._lit_slider.blockSignals(True)
        self._hue_slider.setValue(0)
        self._sat_slider.setValue(0)
        self._lit_slider.setValue(0)
        self._hue_label.setText("0")
        self._sat_label.setText("0")
        self._lit_label.setText("0")
        self._hue_slider.blockSignals(False)
        self._sat_slider.blockSignals(False)
        self._lit_slider.blockSignals(False)
        # Clear crop rectangle
        self._crop_label._rect = QtCore.QRect()
        self._crop_label.update()
        self._on_crop_changed()
        # Reset resize to original dimensions
        self._resize_lock = True
        self._w_spin.setValue(self._orig_w)
        self._h_spin.setValue(self._orig_h)
        self._resize_lock = False
        # Apply immediately (Reset = restore original + close dialog)
        self._on_ok()

    @staticmethod
    def _resize_array(img_u8, new_w, new_h):
        """Resize a uint8 RGB numpy array using QImage smooth scaling."""
        h, w = img_u8.shape[:2]
        buf = np.ascontiguousarray(img_u8)
        qimg = QtGui.QImage(buf.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        scaled = qimg.scaled(new_w, new_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        scaled = scaled.convertToFormat(QtGui.QImage.Format_RGB888)
        bpl = scaled.bytesPerLine()
        ptr = scaled.bits()
        ptr.setsize(new_h * bpl)
        raw = np.frombuffer(ptr, dtype=np.uint8).reshape(new_h, bpl)
        return raw[:, :new_w * 3].reshape(new_h, new_w, 3).copy()

    def _on_ok(self):
        progress = QtWidgets.QProgressDialog("Applying modifications…", None, 0, 6, self)
        progress.setWindowTitle("Modify Image")
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(350)
        progress.setValue(0)
        QtWidgets.QApplication.processEvents()
        try:
            rect = self._crop_label.get_crop_rect_normalized()
            self.crop_rect = rect      # (x, y, w, h) normalised or None
            self.contrast   = self._contrast
            self.hsl = {k: list(v) for k, v in self._hsl.items()}

            # Step 1: Crop
            progress.setLabelText("Cropping…")
            progress.setValue(1)
            QtWidgets.QApplication.processEvents()
            if rect is not None:
                x, y, w, h = rect
                img_H, img_W = self._full_f32.shape[:2]
                x0 = int(round(x * img_W))
                y0 = int(round(y * img_H))
                x1 = int(round((x + w) * img_W))
                y1 = int(round((y + h) * img_H))
                x0, x1 = max(0, x0), min(img_W, x1)
                y0, y1 = max(0, y0), min(img_H, y1)
                if (x1 - x0) < 4 or (y1 - y0) < 4:
                    progress.close()
                    QtWidgets.QMessageBox.warning(self, "Crop", "Selected area is too small.")
                    return
                result_f32 = self._full_f32[y0:y1, x0:x1].copy()
            else:
                result_f32 = self._full_f32.copy()

            # Step 2: Contrast
            progress.setLabelText("Applying contrast…")
            progress.setValue(2)
            QtWidgets.QApplication.processEvents()
            result_f32 = self._apply_bc(result_f32)

            # Step 3: HSL
            progress.setLabelText("Applying HSL adjustments…")
            progress.setValue(3)
            QtWidgets.QApplication.processEvents()
            result_f32 = self._apply_hsl(result_f32)

            # Step 4: Resize
            rh, rw = result_f32.shape[:2]
            target_w = self._w_spin.value()
            target_h = self._h_spin.value()
            if target_w != rw or target_h != rh:
                progress.setLabelText("Resizing…")
                progress.setValue(4)
                QtWidgets.QApplication.processEvents()
                result_u8 = np.clip(result_f32 * 255, 0, 255).astype(np.uint8)
                result_u8 = self._resize_array(result_u8, target_w, target_h)
                result_f32 = result_u8.astype(np.float32) / 255.0

            # Step 5: Convert dtype
            progress.setLabelText("Finalising…")
            progress.setValue(5)
            QtWidgets.QApplication.processEvents()
            if self._full_image.dtype == np.uint16:
                self.cropped_array = (result_f32 * 65535).astype(np.uint16)
            elif self._full_image.dtype in (np.float32, np.float64):
                self.cropped_array = result_f32
            else:
                self.cropped_array = (result_f32 * 255).astype(np.uint8)

            progress.setValue(6)
            progress.close()
            self.accept()
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(self, "Modify Image Error", f"Error applying modifications:\n{e}")
            import traceback; traceback.print_exc()


class ImageQualityDialog(QtWidgets.QDialog):
    def __init__(self, metrics, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Quality")
        self.setMinimumWidth(360)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QtWidgets.QLabel("Image Quality Report")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)

        # SCI score bar
        sci_val = metrics['SCI']
        sci_color = ("#4caf50" if sci_val >= 70 else
                     "#ff9800" if sci_val >= 45 else "#f44336")
        sci_label = QtWidgets.QLabel(f"SCI Score: <b>{sci_val:.1f} / 100</b>")
        sci_label.setAlignment(QtCore.Qt.AlignCenter)
        sci_label.setStyleSheet(f"font-size: 18px; color: {sci_color};")
        layout.addWidget(sci_label)

        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(sci_val))
        bar.setTextVisible(False)
        bar.setFixedHeight(12)
        bar.setStyleSheet(f"""
            QProgressBar {{ border: 1px solid #555; border-radius: 6px; background: #333; }}
            QProgressBar::chunk {{ background: {sci_color}; border-radius: 6px; }}
        """)
        layout.addWidget(bar)

        layout.addWidget(_make_separator())

        # Metrics grid
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        def add_row(row, name, value, label, color):
            grid.addWidget(QtWidgets.QLabel(f"<b>{name}</b>"), row, 0)
            val_lbl = QtWidgets.QLabel(f"{value:.3f}")
            val_lbl.setAlignment(QtCore.Qt.AlignCenter)
            grid.addWidget(val_lbl, row, 1)
            tag = QtWidgets.QLabel(label)
            tag.setAlignment(QtCore.Qt.AlignCenter)
            tag.setStyleSheet(f"color: {color}; font-style: italic;")
            grid.addWidget(tag, row, 2)

        def metric_color(val):
            return "#4caf50" if val >= 0.7 else "#ff9800" if val >= 0.4 else "#f44336"

        add_row(0, "Resolution (R)", metrics['R'], metrics['R_label'],
                metric_color(min(metrics['R'], 1.0)))
        add_row(1, "Quantization (Q)", metrics['Q'], metrics['Q_label'],
                metric_color(metrics['Q']))
        add_row(2, "Blockiness (B)", metrics['B'], metrics['B_label'],
                metric_color(metrics['B']))
        add_row(3, "High-freq. (HF)", metrics['HF'], metrics['HF_label'],
                metric_color(metrics['HF']))

        layout.addLayout(grid)
        layout.addWidget(_make_separator())

        # Dimensions
        dim_lbl = QtWidgets.QLabel(
            f"Dimensions: {metrics['w']} × {metrics['h']} px  "
            f"({metrics['w'] * metrics['h'] / 1e6:.2f} Mpx)"
        )
        dim_lbl.setAlignment(QtCore.Qt.AlignCenter)
        dim_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(dim_lbl)

        # Buttons: OK
        btn_layout = QtWidgets.QHBoxLayout()
        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)


# ------------ Session browser dialog with thumbnails ------------
class _SessionBrowserDialog(QtWidgets.QDialog):
    """Dialog that displays .rasc session files as a thumbnail grid."""

    def __init__(self, initial_dir="", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load session")
        _geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
        _small = _geo.width() < 1400 or _geo.height() < 800
        self.THUMB_SIZE = 128 if _small else 192
        self.PREVIEW_SIZE = 256 if _small else 384
        self.setMinimumSize(
            min(900, int(_geo.width() * 0.85)),
            min(560, int(_geo.height() * 0.80)),
        )
        self.selected_path = None
        self._current_dir = initial_dir or os.path.expanduser("~")
        self._thumb_cache = {}  # path -> QPixmap (full-size decoded)

        layout = QtWidgets.QVBoxLayout(self)

        # Directory bar
        dir_row = QtWidgets.QHBoxLayout()
        self._dir_label = QtWidgets.QLabel()
        self._dir_label.setStyleSheet("font-weight: bold;")
        dir_row.addWidget(self._dir_label, 1)
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_directory)
        dir_row.addWidget(btn_browse)
        btn_up = QtWidgets.QPushButton("↑")
        btn_up.setMaximumWidth(30)
        btn_up.clicked.connect(self._go_up)
        dir_row.addWidget(btn_up)
        layout.addLayout(dir_row)

        # Main content: grid on left, preview on right
        content_row = QtWidgets.QHBoxLayout()

        # Thumbnail grid (QListWidget in icon mode)
        self._list = QtWidgets.QListWidget()
        self._list.setViewMode(QtWidgets.QListView.IconMode)
        self._list.setIconSize(QtCore.QSize(self.THUMB_SIZE, self.THUMB_SIZE))
        self._list.setResizeMode(QtWidgets.QListView.Adjust)
        self._list.setSpacing(12)
        self._list.setWordWrap(True)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        content_row.addWidget(self._list, 1)

        # Preview panel
        preview_panel = QtWidgets.QWidget()
        preview_panel.setMinimumWidth(self.PREVIEW_SIZE + 20)
        preview_panel.setMaximumWidth(self.PREVIEW_SIZE + 40)
        preview_layout = QtWidgets.QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(8, 0, 0, 0)

        self._preview_label = QtWidgets.QLabel()
        self._preview_label.setFixedSize(self.PREVIEW_SIZE, self.PREVIEW_SIZE)
        self._preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self._preview_label.setStyleSheet(
            "background: #1e1e1e; border: 1px solid #555;"
        )
        preview_layout.addWidget(self._preview_label)

        self._preview_name = QtWidgets.QLabel()
        self._preview_name.setWordWrap(True)
        self._preview_name.setAlignment(QtCore.Qt.AlignCenter)
        self._preview_name.setStyleSheet("font-weight: bold; margin-top: 4px;")
        preview_layout.addWidget(self._preview_name)

        self._preview_path = QtWidgets.QLabel()
        self._preview_path.setWordWrap(True)
        self._preview_path.setAlignment(QtCore.Qt.AlignCenter)
        self._preview_path.setStyleSheet("color: #aaa; font-size: 10px;")
        preview_layout.addWidget(self._preview_path)

        preview_layout.addStretch()
        content_row.addWidget(preview_panel, 0)
        layout.addLayout(content_row)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_file = QtWidgets.QPushButton("Open file…")
        btn_file.setToolTip("Use standard file dialog")
        btn_file.clicked.connect(self._open_file_dialog)
        btn_row.addWidget(btn_file)
        btn_row.addStretch()
        self._btn_open = QtWidgets.QPushButton("Open")
        self._btn_open.setEnabled(False)
        self._btn_open.clicked.connect(self._accept_selection)
        btn_row.addWidget(self._btn_open)
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        self._list.currentItemChanged.connect(self._on_selection_changed)

        self._populate(self._current_dir)

    def _populate(self, directory):
        """Scan directory for .rasc files and populate the grid."""
        self._current_dir = directory
        self._dir_label.setText(directory)
        self._list.clear()

        if not os.path.isdir(directory):
            return

        rasc_files = sorted(
            [f for f in os.listdir(directory) if f.lower().endswith('.rasc')],
            key=str.lower
        )

        placeholder_pixmap = QtGui.QPixmap(self.THUMB_SIZE, self.THUMB_SIZE)
        placeholder_pixmap.fill(QtGui.QColor(60, 60, 60))

        self._thumb_cache.clear()
        for fname in rasc_files:
            fpath = os.path.join(directory, fname)
            full_pixmap = self._load_thumbnail_full(fpath)
            if full_pixmap is not None:
                self._thumb_cache[fpath] = full_pixmap
                icon_pixmap = full_pixmap.scaled(
                    self.THUMB_SIZE, self.THUMB_SIZE,
                    QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
            else:
                icon_pixmap = placeholder_pixmap

            item = QtWidgets.QListWidgetItem()
            item.setIcon(QtGui.QIcon(icon_pixmap))
            display_name = os.path.splitext(fname)[0]
            item.setText(display_name)
            item.setData(QtCore.Qt.UserRole, fpath)
            item.setToolTip(fname)
            self._list.addItem(item)

    def _load_thumbnail_full(self, rasc_path):
        """Read the thumbnail field from a .rasc JSON file without parsing the full JSON.
        Scans line by line for the "thumbnail" key and decodes only that value."""
        try:
            thumb_b64 = None
            with open(rasc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped = line.lstrip()
                    if not stripped.startswith('"thumbnail"'):
                        continue
                    # Line looks like:  "thumbnail": "...base64...",
                    colon = stripped.index(':')
                    value_part = stripped[colon + 1:].strip().rstrip(',').strip()
                    if value_part in ('null', '""', ''):
                        return None
                    # Strip surrounding quotes
                    if value_part.startswith('"') and value_part.endswith('"'):
                        thumb_b64 = value_part[1:-1]
                    break
            if not thumb_b64:
                return None
            thumb_bytes = base64.b64decode(thumb_b64)
            pixmap = QtGui.QPixmap()
            pixmap.loadFromData(thumb_bytes)
            if pixmap.isNull():
                return None
            return pixmap
        except Exception as e:
            print(f"[RASCAL] thumbnail loading failed: {e}")
            return None

    def _on_selection_changed(self, current, previous):
        self._btn_open.setEnabled(current is not None)
        if current is None:
            self._preview_label.clear()
            self._preview_name.setText("")
            self._preview_path.setText("")
            return
        fpath = current.data(QtCore.Qt.UserRole)
        # Show enlarged preview
        pixmap = self._thumb_cache.get(fpath)
        if pixmap is not None:
            scaled = pixmap.scaled(
                self.PREVIEW_SIZE, self.PREVIEW_SIZE,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self._preview_label.setPixmap(scaled)
        else:
            self._preview_label.clear()
        name = os.path.splitext(os.path.basename(fpath))[0]
        self._preview_name.setText(name)
        self._preview_path.setText(os.path.dirname(fpath))

    def _on_double_click(self, item):
        self.selected_path = item.data(QtCore.Qt.UserRole)
        self.accept()

    def _accept_selection(self):
        item = self._list.currentItem()
        if item:
            self.selected_path = item.data(QtCore.Qt.UserRole)
            self.accept()

    def _pick_directory(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose directory", self._current_dir)
        if d:
            self._populate(d)

    def _go_up(self):
        parent = os.path.dirname(self._current_dir)
        if parent and parent != self._current_dir:
            self._populate(parent)

    def _open_file_dialog(self):
        """Fallback to standard file dialog."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load session", self._current_dir,
            "Rascal sessions (*.rasc);;All files (*)"
        )
        if path:
            self.selected_path = path
            self.accept()


# ------------ Main window ------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rascal")
        self.setAcceptDrops(True)
        _here = os.path.dirname(os.path.abspath(__file__))
        _icon_path = os.path.join(_here, "rascal_clean.png" if (IS_MACOS or IS_LINUX) else "rascal_clean.ico")
        if not os.path.isfile(_icon_path):
            _icon_path = os.path.join(_here, "rascal_clean.ico")
        if os.path.isfile(_icon_path):
            self.setWindowIcon(QtGui.QIcon(_icon_path))

        # Status bar: left label (directory/file) + right label (size/weight)
        self.status_label_left = QtWidgets.QLabel()
        self.status_label_left.setStyleSheet("QLabel { padding: 2px; }")
        self.statusBar().addWidget(self.status_label_left)
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("QLabel { padding: 2px; }")
        self.statusBar().addPermanentWidget(self.status_label)

        # Memory indicator (permanent, rightmost)
        self._mem_label = QtWidgets.QLabel()
        self._mem_label.setStyleSheet(
            "QLabel { padding: 2px 6px; font-size: 11px; color: #888; }"
        )
        self.statusBar().addPermanentWidget(self._mem_label)
        self._mem_timer = QtCore.QTimer(self)
        self._mem_timer.setInterval(5000)  # refresh every 5 seconds
        self._mem_timer.timeout.connect(self._refresh_memory_label)
        self._mem_timer.start()
        self._refresh_memory_label()

        self.state = AppState()
        self._update_status_bar()
        self._session_dirty = False   # True = unsaved changes since last save
        self._last_session_path = None  # path of last successful save
        self._precision_warning_shown = False  # shown at most once per loaded image

        self.export_flags = {'original': False, 'color': True, 'R': False, 'G': False, 'B': False, '3d': False}

        self.info_panel = InfoPanel(self.state)
        self.model_info_panel = InfoPanel(self.state)
        self.w_color = VisImageWidget(self.state, mode=0, compact=False)
        self.w_R  = VisImageWidget(self.state, mode=1, compact=True)
        self.w_G  = VisImageWidget(self.state, mode=2, compact=True)
        self.w_B  = VisImageWidget(self.state, mode=3, compact=True)
        self.w_3d = VisScatter3DWidget(self.state, compact=True)
        self.w_model = TexturedModelWidget(self.state)
        self.w_model_tex = VisImageWidget(self.state, mode=0, compact=False)
        self.model_ui_level = 'basic'
        self._model_expert_push_init = False

        self.state.widgets_2d = [self.w_color, self.w_R, self.w_G, self.w_B]
        self.state.widget_3d  = self.w_3d
        self.state.model_widget = self.w_model
        self.state.model_tex_widget = self.w_model_tex
        QtWidgets.QApplication.instance().installEventFilter(self)

        # Map mode -> title
        self.mode_titles = {
            0: "Colour view (shader)",
            1: "R channel (grey)",
            2: "G channel (grey)",
            3: "B channel (grey)",
            4: "Original (file)"
        }
        self._orig_swap_prev_mode = None
        self._swapped_channel = None

        # UI widgets initialized later — set to None for clean hasattr replacement
        self.ica_btn = None
        self.pol_btn = None
        self.fib_norm_btn = None
        self.paint_mode_btn = None
        self.model_level_combo = None
        self._stack = None
        self._model_host = None
        self._model_left_box = None
        self._preset_action = None
        self._preset_save_action = None
        self._level_row_widget = None
        self.auto_rotate_action = None
        self.auto_rotate_btn = None
        self._rand_rot_action = None
        self._rand_rot_timer = None
        self.push_slider = None
        self.push_val_label = None
        self.var_restore_slider = None
        self.var_restore_val_label = None
        self.contrast_label = None
        self.contrast_slider = None
        self.contrast_val_label = None
        self.auto_contrast_btn = None
        self.reprocess_btn = None
        self.model_reprocess_btn = None
        self.clear_brush_btn = None
        self.model_clear_brush_btn = None
        self.brush_panel = None  # already set above but ensure consistency
        self.model_brush_panel = None
        self._save_3d_action = None
        self._load_session_top_act = None
        self._model_content_splitter = None
        self._preset_buttons = []
        self._fib_worker = None
        self._fib_progress_dlg = None
        self._key_rot_timer = None
        self._pressed_rot_keys = set()

        split_main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split_main.setChildrenCollapsible(True)
        split_center_right = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        split_center_right.setChildrenCollapsible(True)
        split_right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split_right.setChildrenCollapsible(True)
        split_right.setMinimumHeight(0)

        left_box = QtWidgets.QWidget()
        self._left_box = left_box
        left_box.setMinimumWidth(270)
        vb_left = QtWidgets.QVBoxLayout(left_box); vb_left.setContentsMargins(6,6,6,6)
        vb_left.addWidget(self.info_panel)

        # Brush panel (visible only in paint mode)
        self.brush_panel = QtWidgets.QGroupBox("Brush")
        brush_lay = QtWidgets.QFormLayout(self.brush_panel)
        self.brush_size_spin = QtWidgets.QSpinBox()
        self.brush_size_spin.setRange(1, 100)
        self.brush_size_spin.setValue(30)
        self.brush_size_spin.setSuffix(" px")
        self.brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        brush_lay.addRow("Size:", self.brush_size_spin)
        self.clear_brush_btn = QtWidgets.QPushButton("Clear")
        self.clear_brush_btn.clicked.connect(self.clear_current_brush)
        self.clear_brush_btn.setEnabled(False)
        brush_lay.addRow(self.clear_brush_btn)
        self.reprocess_btn = QtWidgets.QPushButton("Re-process")
        self.reprocess_btn.clicked.connect(self.reprocess)
        self.reprocess_btn.setEnabled(False)
        brush_lay.addRow(self.reprocess_btn)
        self.brush_panel.setVisible(False)
        self.model_brush_panel = None
        vb_left.addWidget(self.brush_panel)

        vb_left.addStretch()

        # UI level combo — built here, placed in the main page below
        self.ui_level = 'basic'
        level_row = QtWidgets.QHBoxLayout()
        level_row.addSpacing(20)
        self.level_label = QtWidgets.QLabel("Level:")
        level_row.addWidget(self.level_label)
        self.ui_level_combo = QtWidgets.QComboBox()
        self.ui_level_combo.addItems(["Basic mode", "Expert mode"])
        self.ui_level_combo.setFocusPolicy(QtCore.Qt.NoFocus)
        self.ui_level_combo.currentIndexChanged.connect(self._on_ui_level_changed)
        level_row.addWidget(self.ui_level_combo)
        self.model_level_combo = QtWidgets.QComboBox()
        self.model_level_combo.addItems(["Basic mode", "Expert mode"])
        self.model_level_combo.setFocusPolicy(QtCore.Qt.NoFocus)
        self.model_level_combo.currentIndexChanged.connect(self._on_model_ui_level_changed)
        self.model_level_combo.setVisible(False)
        level_row.addWidget(self.model_level_combo)
        level_row.addSpacing(16)
        self.push_label = QtWidgets.QLabel("Push:")
        level_row.addWidget(self.push_label)
        self.push_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.push_slider.setRange(10, 500)  # displayed as 0.10 .. 5.00
        self.push_slider.setValue(100)
        self.push_slider.setFixedWidth(100)
        self.push_slider.setToolTip("Push factor: 1.00 = normal, 0.10 = minimum, 5.00 = maximum")
        self.push_slider.valueChanged.connect(lambda v: self._set_push(v / 100.0))
        level_row.addWidget(self.push_slider)
        self.push_val_label = QtWidgets.QLabel("1.00")
        self.push_val_label.setFixedWidth(40)
        level_row.addWidget(self.push_val_label)
        level_row.addSpacing(12)
        self.var_restore_label = QtWidgets.QLabel("Var:")
        level_row.addWidget(self.var_restore_label)
        self.var_restore_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.var_restore_slider.setRange(0, 100)
        self.var_restore_slider.setValue(100)
        self.var_restore_slider.setFixedWidth(100)
        self.var_restore_slider.setToolTip("Variance restore: 1 = full original variance, 0 = no restore (identity)")
        self.var_restore_slider.valueChanged.connect(lambda v: self._set_var_restore(v / 100.0))
        level_row.addWidget(self.var_restore_slider)
        self.var_restore_val_label = QtWidgets.QLabel("1.00")
        self.var_restore_val_label.setFixedWidth(40)
        level_row.addWidget(self.var_restore_val_label)
        level_row.addSpacing(12)
        self.contrast_label = QtWidgets.QLabel("Contrast:")
        level_row.addWidget(self.contrast_label)
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(50, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedWidth(100)
        self.contrast_slider.setToolTip("Contrast: 1.0 = no change, 0.5 = flatter, 3.0 = more contrast")
        self.contrast_slider.valueChanged.connect(lambda v: self._set_contrast(v / 100.0))
        level_row.addWidget(self.contrast_slider)
        self.contrast_val_label = QtWidgets.QLabel("1.00")
        self.contrast_val_label.setFixedWidth(40)
        level_row.addWidget(self.contrast_val_label)
        level_row.addSpacing(6)
        self.auto_contrast_btn = QtWidgets.QPushButton("Auto")
        self.auto_contrast_btn.setCheckable(True)
        self.auto_contrast_btn.setToolTip("Auto contrast: adjust contrast automatically for the current view. Turns off when rotation or parameters change.")
        self.auto_contrast_btn.clicked.connect(self._on_auto_contrast_clicked)
        level_row.addWidget(self.auto_contrast_btn)
        level_row.addStretch()

        split_main.addWidget(left_box); split_main.addWidget(split_center_right)

        # Store GroupBox references to allow title updates
        self.color_box = self._wrap_with_export_checkbox(
            self.w_color, self.mode_titles[0], key='color',
            checked=self.export_flags.get('color', True)
        )

        split_center_right.addWidget(self.color_box)

        split_center_right.addWidget(split_right)
        # Original thumbnail (top of right column, never modified)
        class _ThumbLabel(QtWidgets.QLabel):
            def sizeHint(self): return QtCore.QSize(420, 260)
            def minimumSizeHint(self): return QtCore.QSize(50, 60)
            def resizeEvent(self_, e):
                QtWidgets.QLabel.resizeEvent(self_, e)
                self._update_original_thumbnail()
            def mousePressEvent(self_, e):
                if e.button() == QtCore.Qt.LeftButton:
                    self._toggle_original_swap()
        self.w_original_thumb = _ThumbLabel()
        self.w_original_thumb.setAlignment(QtCore.Qt.AlignCenter)
        self.w_original_thumb.setCursor(QtCore.Qt.PointingHandCursor)
        self.w_original_thumb.setStyleSheet("background-color: black;")
        sp_thumb = self.w_original_thumb.sizePolicy()
        sp_thumb.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        sp_thumb.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        self.w_original_thumb.setSizePolicy(sp_thumb)
        self.orig_group = self._wrap_with_export_checkbox(
            self.w_original_thumb, "Original", key='original',
            checked=self.export_flags.get('original', False)
        )
        split_right.addWidget(self.orig_group)
        self._update_original_thumbnail()

        self.R_box = self._wrap_with_export_checkbox(self.w_R, self.mode_titles[1], key='R', checked=self.export_flags.get('R', False))
        self.G_box = self._wrap_with_export_checkbox(self.w_G, self.mode_titles[2], key='G', checked=self.export_flags.get('G', False))
        self.B_box = self._wrap_with_export_checkbox(self.w_B, self.mode_titles[3], key='B', checked=self.export_flags.get('B', False))
        split_right.addWidget(self.R_box)
        split_right.addWidget(self.G_box)
        split_right.addWidget(self.B_box)
        self.box_3d = self._wrap_with_export_checkbox(self.w_3d, "3D cloud", key='3d', checked=self.export_flags.get('3d', False))
        split_right.addWidget(self.box_3d)
        self.model_box = self._wrap(self.w_model, "3D model")
        split_right.addWidget(self.model_box)

        # Click on a right-panel view => swap with the main view
        self.w_R.clicked.connect(lambda: self.swap_views(self.w_color, self.w_R, self.color_box, self.R_box))
        self.w_G.clicked.connect(lambda: self.swap_views(self.w_color, self.w_G, self.color_box, self.G_box))
        self.w_B.clicked.connect(lambda: self.swap_views(self.w_color, self.w_B, self.color_box, self.B_box))

        split_main.setStretchFactor(0,0); split_main.setStretchFactor(1,1)
        split_center_right.setStretchFactor(0,4); split_center_right.setStretchFactor(1,1)
        for i in range(6): split_right.setStretchFactor(i, 1)

        # Pack the main interface into a stack page (without level row)
        self._main_page = QtWidgets.QWidget()
        main_lay = QtWidgets.QVBoxLayout(self._main_page)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)
        main_lay.addWidget(split_main, 1)

        # Welcome page
        self._welcome_page = self._build_welcome_page()

        # 3D page: texture (left) + 3D model (right), no R/G/B/Original panels
        self._3d_page = self._build_3d_page()

        # QStackedWidget: page 0 = welcome, page 1 = image, page 2 = 3D
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._welcome_page)   # index 0
        self._stack.addWidget(self._main_page)       # index 1
        self._stack.addWidget(self._3d_page)         # index 2
        self._stack.setCurrentIndex(0)

        # Global central layout: shared level row + stacked pages (welcome/2D/3D)
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._level_row_widget = QtWidgets.QWidget()
        self._level_row_widget.setLayout(level_row)
        lay.addWidget(self._level_row_widget)
        lay.addWidget(self._stack, 1)
        self.setCentralWidget(central)

        self._build_menus(); self._build_toolbar()
        # In welcome mode, hide non-essential menus/buttons
        self._set_welcome_mode(True)
        self._apply_ui_level('basic')
        
        # Keyboard shortcuts (ApplicationShortcut context to work everywhere)
        shortcut_r = QtWidgets.QShortcut(QtGui.QKeySequence("R"), self)
        shortcut_r.setContext(QtCore.Qt.ApplicationShortcut)
        shortcut_r.activated.connect(self.reset_rotation)

        shortcut_p = QtWidgets.QShortcut(QtGui.QKeySequence("P"), self)
        shortcut_p.setContext(QtCore.Qt.ApplicationShortcut)
        shortcut_p.activated.connect(
            lambda: self._save_preset()
            if (self.ui_level == 'expert') or
               (self._stack.currentIndex() == 2 and self.model_ui_level == 'expert')
            else None
        )

        shortcut_n = QtWidgets.QShortcut(QtGui.QKeySequence("N"), self)
        shortcut_n.setContext(QtCore.Qt.ApplicationShortcut)
        shortcut_n.activated.connect(self._toggle_3d_decorrelated)

        shortcut_i = QtWidgets.QShortcut(QtGui.QKeySequence("I"), self)
        shortcut_i.setContext(QtCore.Qt.ApplicationShortcut)
        shortcut_i.activated.connect(
            lambda: self._run_ica()
            if (
                (self._stack.currentIndex() == 1 and self.ui_level == 'expert') or
                (self._stack.currentIndex() == 2 and self.model_ui_level == 'expert')
            )
            else None
        )

        # Shortcuts to swap views (Ctrl+1/2/3 to swap with R/G/B)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+1"), self, activated=lambda: self.swap_views(self.w_color, self.w_R, self.color_box, self.R_box))
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+2"), self, activated=lambda: self.swap_views(self.w_color, self.w_G, self.color_box, self.G_box))
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+3"), self, activated=lambda: self.swap_views(self.w_color, self.w_B, self.color_box, self.B_box))

        self.auto=False
        self.timer = app.Timer(interval=1/60, connect=self._tick, start=False)

        self.ui_timer = QtCore.QTimer(self); self.ui_timer.setInterval(50)
        self.ui_timer.timeout.connect(self._refresh_info_panels); self.ui_timer.start()
        self._refresh_info_panels()

        self.state.on_paint_updated = self._update_reprocess_btn
        self.state.on_rotation_started = self._on_rotation_started
        self.state.on_ica_deactivated = self._deactivate_ica_btn

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_welcome_mode', False):
            self._load_welcome_image()

    def _toggle_3d_decorrelated(self):
        if self._stack is not None and self._stack.currentIndex() == 0:
            return
        if self.w_3d is not None:
            entering_decorrelated = not self.w_3d.show_decorrelated
            self.w_3d.toggle_decorrelated(fib_norm_mode=self.state.fib_norm_mode and entering_decorrelated)

    def _deactivate_ica_btn(self):
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(False)
            self.ica_btn.blockSignals(False)

    def _on_rotation_started(self):
        self._stop_autorotate()
        self._invalidate_auto_contrast()
        self._session_dirty = True

    def _stop_autorotate(self):
        self._stop_random_rotation()
        if self.auto:
            self.auto = False
            if self.auto_rotate_action is not None:
                self.auto_rotate_action.setChecked(False)
            self.timer.stop()

    def _wrap(self, widget, title):
        """Wraps a widget in a titled group box."""
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(widget)
        layout.setContentsMargins(2, 15, 2, 2)
        group.setLayout(layout)
        # Allow shrinking
        group.setMinimumWidth(50)
        group.setMinimumHeight(0)
        gsp = group.sizePolicy()
        gsp.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        gsp.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        group.setSizePolicy(gsp)
        return group

    def _update_original_thumbnail(self):
        """Refreshes the Original thumbnail centered on a black background."""
        img = self.state.img_original_view
        h, w = img.shape[:2]
        rgb8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        src = QtGui.QImage(rgb8.tobytes(), w, h, w * 3, QtGui.QImage.Format_RGB888)

        thumb = self.w_original_thumb
        tw = thumb.width() or 200
        th = thumb.height() or 150

        # Scale preserving aspect ratio
        scaled = src.scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        # Compose onto black canvas same size as the container
        canvas = QtGui.QPixmap(tw, th)
        canvas.fill(QtGui.QColor(0, 0, 0))
        painter = QtGui.QPainter(canvas)
        x_off = (tw - scaled.width()) // 2
        y_off = (th - scaled.height()) // 2
        painter.drawImage(x_off, y_off, scaled)
        painter.end()

        thumb.setPixmap(canvas)

    def _wrap_with_export_checkbox(self, widget, title, key, checked=True):
        """Wraps a widget with an export checkbox."""
        group = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout()

        # Add export checkbox
        cb = QtWidgets.QCheckBox("Export")
        cb.setChecked(checked)
        cb.stateChanged.connect(lambda state, k=key: self._on_export_checkbox_changed(k, state))

        layout.addWidget(cb)
        layout.addWidget(widget)
        layout.setContentsMargins(2, 15, 2, 2)
        group.setLayout(layout)
        # Allow shrinking
        group.setMinimumWidth(50)
        group.setMinimumHeight(0)
        gsp2 = group.sizePolicy()
        gsp2.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        gsp2.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        group.setSizePolicy(gsp2)
        return group

    def _on_export_checkbox_changed(self, key, state):
        """Handles export checkbox state change."""
        self.export_flags[key] = state == QtCore.Qt.Checked

    def _build_3d_page(self):
        """Builds the 3D page: texture view (left) + manipulable 3D model (right)."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        page_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        page_split.setChildrenCollapsible(True)

        # Left panel (same place as 2D): rotation info/quaternion in expert mode
        self._model_left_box = QtWidgets.QWidget()
        self._model_left_box.setMinimumWidth(270)
        vb_left = QtWidgets.QVBoxLayout(self._model_left_box)
        vb_left.setContentsMargins(6, 6, 6, 6)
        vb_left.addWidget(self.model_info_panel)

        # Brush panel in 3D mode (same controls as 2D)
        self.model_brush_panel = QtWidgets.QGroupBox("Brush")
        model_brush_lay = QtWidgets.QFormLayout(self.model_brush_panel)
        self.model_brush_size_spin = QtWidgets.QSpinBox()
        self.model_brush_size_spin.setRange(1, 100)
        self.model_brush_size_spin.setValue(50)
        self.model_brush_size_spin.setSuffix(" px")
        self.model_brush_size_spin.valueChanged.connect(self._on_brush_size_changed)
        model_brush_lay.addRow("Size:", self.model_brush_size_spin)
        self.model_clear_brush_btn = QtWidgets.QPushButton("Clear")
        self.model_clear_brush_btn.clicked.connect(self.clear_current_brush)
        self.model_clear_brush_btn.setEnabled(False)
        model_brush_lay.addRow(self.model_clear_brush_btn)
        self.model_reprocess_btn = QtWidgets.QPushButton("Re-process")
        self.model_reprocess_btn.clicked.connect(self.reprocess)
        self.model_reprocess_btn.setEnabled(False)
        model_brush_lay.addRow(self.model_reprocess_btn)
        self.model_brush_panel.setVisible(False)
        vb_left.addWidget(self.model_brush_panel)

        if hasattr(self.w_model_tex, 'brush_radius_px'):
            self.w_model_tex.brush_radius_px = 50.0

        vb_left.addStretch()
        page_split.addWidget(self._model_left_box)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        self._model_content_splitter = splitter

        # Left: texture view (VisImageWidget mode=0, full size)
        tex_group = QtWidgets.QGroupBox("Texture")
        tex_layout = QtWidgets.QVBoxLayout()
        tex_layout.setContentsMargins(2, 15, 2, 2)
        tex_layout.addWidget(self.w_model_tex)
        tex_group.setLayout(tex_layout)
        gsp_l = tex_group.sizePolicy()
        gsp_l.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        gsp_l.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        tex_group.setSizePolicy(gsp_l)
        splitter.addWidget(tex_group)

        # Right: 3D model widget
        model_group = QtWidgets.QGroupBox("3D model")
        model_layout = QtWidgets.QVBoxLayout()
        model_layout.setContentsMargins(2, 15, 2, 2)

        self._model_host = _ModelHost(self.w_model)
        model_layout.addWidget(self._model_host)
        model_group.setLayout(model_layout)
        gsp_r = model_group.sizePolicy()
        gsp_r.setHorizontalPolicy(QtWidgets.QSizePolicy.Expanding)
        gsp_r.setVerticalPolicy(QtWidgets.QSizePolicy.Expanding)
        model_group.setSizePolicy(gsp_r)
        splitter.addWidget(model_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        page_split.addWidget(splitter)
        page_split.setStretchFactor(0, 0)
        page_split.setStretchFactor(1, 1)

        lay.addWidget(page_split, 1)
        return page

    def _equalize_3d_views(self):
        if self._model_content_splitter is None:
            return
        sp = self._model_content_splitter
        if sp.count() < 2:
            return
        total = max(2, sp.width())
        half = max(1, total // 2)
        sp.setSizes([half, total - half])

    def _refresh_info_panels(self):
        self.info_panel.refresh()
        if self.model_info_panel is not None:
            self.model_info_panel.refresh()

    def _build_welcome_page(self):
        """Builds the welcome page with rascal_splash.png centred."""
        page = QtWidgets.QWidget()
        page.setStyleSheet("background-color: #1a1a1a;")
        layout = QtWidgets.QVBoxLayout(page)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setSpacing(32)

        # Central image
        self._welcome_img_label = QtWidgets.QLabel()
        self._welcome_img_label.setAlignment(QtCore.Qt.AlignCenter)
        self._welcome_img_label.setStyleSheet("background: transparent; border: none;")
        self._load_welcome_image()
        layout.addWidget(self._welcome_img_label)

        # Message
        msg = QtWidgets.QLabel("Open or drag & drop an image, a 3D model (.obj / .ply) or a session (.rasc) to get started")
        msg.setAlignment(QtCore.Qt.AlignCenter)
        msg.setStyleSheet("color: #aaa; font-size: 15px; background: transparent;")
        layout.addWidget(msg)

        return page

    def _load_welcome_image(self):
        """Loads and displays rascal_splash.png at 60% of the smallest window dimension."""
        _here = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(_here, "rascal_splash.png")
        try:
            img = imageio.imread(img_path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=2)
            img = img[..., :3]
            h, w = img.shape[:2]
            target = int(min(self.width(), self.height()) * 0.35)
            if target < 100:
                target = 300
            img8 = np.clip(img, 0, 255).astype(np.uint8)
            qimg = QtGui.QImage(img8.data, w, h, img8.strides[0], QtGui.QImage.Format_RGB888)
            px = QtGui.QPixmap.fromImage(qimg.copy())
            if h >= w:
                px = px.scaledToHeight(target, QtCore.Qt.SmoothTransformation)
            else:
                px = px.scaledToWidth(target, QtCore.Qt.SmoothTransformation)
            self._welcome_img_label.setPixmap(px)
        except Exception:
            self._welcome_img_label.setText("rascal_splash.png not found")
            self._welcome_img_label.setStyleSheet("color:#888; font-size:14px;")

    def _set_welcome_mode(self, welcome: bool):
        """Enables/disables welcome-page mode: reduced menus and toolbar."""
        self._welcome_mode = welcome
        # Hide non-File menus
        for action in self.menuBar().actions():
            if action.text() not in ('File', '?'):
                action.setVisible(not welcome)
        # Load session top-level: only visible on welcome screen
        if self._load_session_top_act is not None:
            self._load_session_top_act.setVisible(welcome)
        # Hide advanced File actions (Session, Export) in welcome mode
        for action in self._advanced_file_actions:
            action.setVisible(not welcome)
        # Save 3D model is only visible when on the 3D page
        if not welcome and self._save_3d_action is not None:
            is_3d_page = self._stack is not None and self._stack.currentIndex() == 2
            self._save_3d_action.setVisible(is_3d_page)
        # Toolbar: keep only Open and Close
        for act in self.toolbar.actions():
            if act not in (self._tb_open_act, self._tb_close_act):
                act.setVisible(not welcome)
        # _preset_action must be hidden explicitly in welcome mode
        if self._preset_action is not None:
            self._preset_action.setVisible(False)
        if self._preset_save_action is not None:
            self._preset_save_action.setVisible(False)
        if self._level_row_widget is not None:
            self._level_row_widget.setVisible(not welcome)
        # Re-enforce ui_level visibility when leaving welcome mode
        if not welcome:
            if self._stack is not None and self._stack.currentIndex() == 2:
                self._apply_model_ui_level(self.model_ui_level)
            else:
                self._apply_ui_level(self.ui_level)

    def _build_menus(self):
        """Builds the application menus."""
        menubar = self.menuBar()
        self._advanced_file_actions = []

        # File menu
        file_menu = menubar.addMenu('File')

        open_action = QtWidgets.QAction('Open file...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.action_open)
        file_menu.addAction(open_action)

        # Load session: visible on welcome screen only, hidden once an image is loaded
        self._load_session_top_act = QtWidgets.QAction('Load session...', self)
        self._load_session_top_act.setShortcut('Ctrl+Shift+L')
        self._load_session_top_act.triggered.connect(self.load_quaternion)
        file_menu.addAction(self._load_session_top_act)

        file_menu.addSeparator()

        # Session sub-menu (hidden on welcome screen)
        quat_menu = file_menu.addMenu('Session')
        act_save_q = QtWidgets.QAction('Save session...', self)
        act_save_q.setShortcut('Ctrl+Shift+S')
        act_save_q.triggered.connect(self.save_quaternion)
        quat_menu.addAction(act_save_q)
        act_load_q = QtWidgets.QAction('Load session...', self)
        act_load_q.setShortcut('Ctrl+Shift+L')
        act_load_q.triggered.connect(self.load_quaternion)
        quat_menu.addAction(act_load_q)
        self._advanced_file_actions.append(quat_menu.menuAction())

        export_action = QtWidgets.QAction('Export views...', self)
        export_action.triggered.connect(self.export_selected_views_to_jpg)
        file_menu.addAction(export_action)
        self._advanced_file_actions.append(export_action)

        self._save_3d_action = QtWidgets.QAction('Save 3D model...', self)
        self._save_3d_action.triggered.connect(self.save_3d_model)
        self._save_3d_action.setVisible(False)
        file_menu.addAction(self._save_3d_action)
        self._advanced_file_actions.append(self._save_3d_action)

        file_menu.addSeparator()

        quality_action = QtWidgets.QAction('Image Quality...', self)
        quality_action.triggered.connect(self.action_image_quality)
        file_menu.addAction(quality_action)
        self._advanced_file_actions.append(quality_action)

        modify_action = QtWidgets.QAction('Modify Image...', self)
        modify_action.triggered.connect(self.action_modify_image)
        file_menu.addAction(modify_action)
        self._advanced_file_actions.append(modify_action)

        file_menu.addSeparator()
        quit_action = QtWidgets.QAction('Quit', self)
        quit_action.setShortcut('Ctrl+Q')
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Rotation menu
        rot_menu = menubar.addMenu('Rotation')

        reset_action = QtWidgets.QAction('Reset to base state\tR', self)
        # Shortcut displayed as hint only — actual binding handled by QShortcut to avoid conflict
        reset_action.triggered.connect(self.reset_rotation)
        rot_menu.addAction(reset_action)

        self.auto_rotate_action = QtWidgets.QAction('Auto-rotate', self, checkable=True)
        self.auto_rotate_action.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space))
        self.auto_rotate_action.triggered.connect(self.toggle_autorotate)
        rot_menu.addAction(self.auto_rotate_action)

        self._rand_rot_action = QtWidgets.QAction('Random Rotation Explorer', self, checkable=True)
        self._rand_rot_action.setShortcut(QtGui.QKeySequence('Ctrl+Shift+R'))
        self._rand_rot_action.triggered.connect(self._toggle_random_rotation)
        self._rand_rot_action.setVisible(False)
        rot_menu.addAction(self._rand_rot_action)

        # Help menu
        help_menu = menubar.addMenu('?')

        controls_action = QtWidgets.QAction('Controls...', self)
        controls_action.triggered.connect(lambda: ControlsHintDialog(self).exec_())
        help_menu.addAction(controls_action)

        help_menu.addSeparator()

        about_action = QtWidgets.QAction('About...', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def _build_toolbar(self):
        """Builds the toolbar."""
        self.toolbar = self.addToolBar('Tools')
        _dpr = QtWidgets.QApplication.primaryScreen().devicePixelRatio()
        _icon_px = 20 if _dpr > 1.25 else 32
        self.toolbar.setIconSize(QtCore.QSize(_icon_px, _icon_px))
        self.toolbar.setStyleSheet("""
            QToolBar { spacing: 4px; }
            QToolButton {
                font-size: 12px;
                font-weight: bold;
                padding: 4px 8px;
                border: 1px solid #999;
                border-radius: 4px;
                background: #e8e8e8;
                color: #222;
                min-width: 28px;
            }
            QToolButton:hover {
                background: #d0d8f0;
                border-color: #5580cc;
            }
            QToolButton:checked {
                background: #4a7acc;
                color: #ffffff;
                border-color: #2a5aaa;
            }
            QToolButton:pressed {
                background: #3a6ab8;
                color: #ffffff;
            }
        """)

        _here = os.path.dirname(os.path.abspath(__file__))

        def _svg_icon(name):
            path = os.path.join(_here, name)
            px = QtGui.QPixmap(path)
            if px.isNull():
                return QtGui.QIcon()
            return QtGui.QIcon(px)

        # Open file button
        open_act = QtWidgets.QAction(_svg_icon('file-upload.svg'), 'Open', self)
        open_act.setToolTip('Open an image file')
        open_act.triggered.connect(self.action_open)
        self.toolbar.addAction(open_act)
        self._tb_open_act = open_act

        # Reset to base state button
        reset_act = QtWidgets.QAction(_svg_icon('reset.svg'), 'Reset', self)
        reset_act.setToolTip('Reset to base state')
        reset_act.triggered.connect(self.reset_rotation)
        self.toolbar.addAction(reset_act)

        # Export button
        export_act = QtWidgets.QAction(_svg_icon('camera.svg'), 'Export', self)
        export_act.setToolTip('Export views as JPG')
        export_act.triggered.connect(self.export_selected_views_to_jpg)
        self.toolbar.addAction(export_act)

        # Close action (no toolbar icon, kept for welcome-mode logic)
        close_act = QtWidgets.QAction('Close', self)
        close_act.triggered.connect(self.close)
        self._tb_close_act = close_act

        self.toolbar.addSeparator()

        # ROI (paint mode) button
        self.paint_mode_btn = QtWidgets.QAction('ROI', self)
        self.paint_mode_btn.setCheckable(True)
        self.paint_mode_btn.setShortcut('O')
        self.paint_mode_btn.triggered.connect(self._on_roi_btn_clicked)
        self.toolbar.addAction(self.paint_mode_btn)

        # ICA button (expert mode only)
        self.ica_btn = QtWidgets.QAction('ICA', self)
        self.ica_btn.setCheckable(True)
        self.ica_btn.triggered.connect(self._run_ica)
        self.toolbar.addAction(self.ica_btn)

        # Polarity inversion button (expert mode only)
        self.pol_btn = QtWidgets.QAction('Pol. +/-', self)
        self.pol_btn.setToolTip('Invert rotation polarity: negate X, Y, Z angles')
        self.pol_btn.triggered.connect(self._on_pol_btn_clicked)
        self.toolbar.addAction(self.pol_btn)

        # Radial Push (Fibonacci normalisation) button
        self.fib_norm_btn = QtWidgets.QAction('Radial Push', self)
        self.fib_norm_btn.setCheckable(True)
        self.fib_norm_btn.setShortcut('F')
        self.fib_norm_btn.triggered.connect(self.toggle_fib_norm)
        self.toolbar.addAction(self.fib_norm_btn)

        self.toolbar.addSeparator()

        # Split view button (expert mode only)
        self._split_btn = QtWidgets.QAction('Split', self)
        self._split_btn.setToolTip("Toggle before/after split view (T)")
        self._split_btn.setCheckable(True)
        self._split_btn.triggered.connect(self._on_split_btn_toggled)
        self._split_btn.setVisible(False)
        self.toolbar.addAction(self._split_btn)

        # Presets button (expert mode only) — same style as ICA / ROI
        self._preset_save_action = QtWidgets.QAction('Presets', self)
        self._preset_save_action.setToolTip("Save current state as a preset checkpoint (P)")
        self._preset_save_action.triggered.connect(self._save_preset)
        self._preset_save_action.setVisible(False)
        self.toolbar.addAction(self._preset_save_action)

        # Numbered preset buttons container (populated dynamically, expert mode only)
        self._preset_container = QtWidgets.QWidget()
        self._preset_layout = QtWidgets.QHBoxLayout(self._preset_container)
        self._preset_layout.setContentsMargins(2, 0, 2, 0)
        self._preset_layout.setSpacing(3)
        # Use QWidgetAction so visibility is controlled via action (like other toolbar items)
        self._preset_action = QtWidgets.QWidgetAction(self)
        self._preset_action.setDefaultWidget(self._preset_container)
        self._preset_action.setVisible(False)  # hidden until expert mode is activated
        self.toolbar.addAction(self._preset_action)
        self._preset_buttons = []  # list of QPushButton

        self.statusBar()

        settings = QtCore.QSettings("Rascal", "rascal")
        geometry = settings.value("main_geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self._resize_to_screen()

    def _save_preset(self):
        """Captures current rotation + push + full image state as a numbered checkpoint (expert mode only)."""
        if self.state.paint_mode:
            self.statusBar().showMessage("Preset not available in ROI mode", 2000)
            return
        in_expert = (self.ui_level == 'expert') or (self._stack.currentIndex() == 2 and self.model_ui_level == 'expert')
        if not in_expert:
            return
        st = self.state
        if len(st.presets) >= 15:
            QtWidgets.QMessageBox.information(self, "Presets", "Maximum 15 presets reached.")
            return
        if not self._check_memory_before_heavy_op("Save Preset"):
            return
        preset = {
            'Ruser': st.rot.Ruser.copy(),
            'ang_x': st.rot.ang_x,
            'ang_y': st.rot.ang_y,
            'ang_z': st.rot.ang_z,
            'push_factor': st.push_factor,
            'variance_restore': st.variance_restore,
            'contrast': st.contrast,
            'fib_norm_mode': st.fib_norm_mode,
            'fib_img_cached': st._fib_img_cached.copy() if st._fib_img_cached is not None else None,
            'fib_Zs_norm_cached': st._fib_Zs_norm_cached.copy() if st._fib_Zs_norm_cached is not None else None,
            'fib_img_backup': getattr(st, '_fib_img_backup', None).copy() if getattr(st, '_fib_img_backup', None) is not None else None,
            'fib_mu_lin_orig': getattr(st, '_mu_lin_orig', None).copy() if getattr(st, '_mu_lin_orig', None) is not None else None,
            'fib_W_lin_orig': getattr(st, '_W_lin_orig', None).copy() if getattr(st, '_W_lin_orig', None) is not None else None,
            'fib_Winv_lin_orig': getattr(st, '_Winv_lin_orig', None).copy() if getattr(st, '_Winv_lin_orig', None) is not None else None,
            'fib_samples_zca': getattr(st, 'samples_zca_fib', None).copy() if getattr(st, 'samples_zca_fib', None) is not None else None,
            'fib_zca_half': getattr(st, '_fib_zca_half', None),
            # Full image + ZCA snapshot so ROI applied after this point can be undone
            'img_srgb_orig': st.img_srgb_orig.copy(),
            'mu_lin': st.mu_lin.copy(),
            'W_lin': st.W_lin.copy(),
            'Winv_lin': st.Winv_lin.copy(),
            'samples_lin': st.samples_lin.copy(),
            # ICA state
            'ica_active': st.ica_active,
            # ROI / brush state
            'paint_mode': st.paint_mode,
            'brush_alpha': st.brush_alpha.copy() if st.brush_alpha is not None else None,
            'last_brush_alpha': st.last_brush_alpha.copy() if st.last_brush_alpha is not None else None,
            'last_rot_quat': list(st.last_rot_quat) if st.last_rot_quat is not None else None,
            'last_img_srgb_orig': st.last_img_srgb_orig.copy() if st.last_img_srgb_orig is not None else None,
            'last_img_srgb_reprocessed': st.last_img_srgb_reprocessed.copy() if st.last_img_srgb_reprocessed is not None else None,
            # File-level image references (needed for correct save/reset/export after restore)
            'img_file_orig': st.img_file_orig.copy(),
            '_img_file_initial': st._img_file_initial.copy(),
            '_img_pre_roi': st._img_pre_roi.copy() if hasattr(st, '_img_pre_roi') else None,
            'img_roi_base': st.img_roi_base.copy() if hasattr(st, 'img_roi_base') else None,
            'img_original_view': st.img_original_view.copy() if hasattr(st, 'img_original_view') and st.img_original_view is not None else None,
            '_raw_img_for_session': getattr(st, '_raw_img_for_session', None).copy()
                                    if getattr(st, '_raw_img_for_session', None) is not None else None,
            'modify_params': dict(st.modify_params) if st.modify_params is not None else None,
        }
        st.presets.append(preset)
        idx = len(st.presets)  # 1-based number
        btn = QtWidgets.QPushButton(str(idx))
        btn.setFixedSize(24, 24)
        btn.setToolTip(f"Restore preset {idx} (push={preset['push_factor']:.2f}, var={preset.get('variance_restore', 1.0):.2f})")
        btn.setStyleSheet(
            "QPushButton { font-size: 11px; font-weight: bold; border: 1px solid #888; "
            "border-radius: 3px; background: #dde; } "
            "QPushButton:hover { background: #aac; }"
        )
        btn.clicked.connect(lambda checked, i=idx-1: self._restore_preset(i))
        self._preset_layout.addWidget(btn)
        self._preset_buttons.append(btn)
        self.statusBar().showMessage(f"Preset {idx} saved.", 2000)

    def _restore_preset(self, idx):
        """Restores full image state + rotation + push from checkpoint at index idx (0-based)."""
        st = self.state
        if idx < 0 or idx >= len(st.presets):
            return
        preset = st.presets[idx]

        # Restore rotation
        st.rot.Ruser = preset['Ruser'].copy()
        st.rot.ang_x = preset['ang_x']
        st.rot.ang_y = preset['ang_y']
        st.rot.ang_z = preset['ang_z']
        self._set_push(preset['push_factor'], mark_dirty=False)
        self._set_var_restore(preset.get('variance_restore', 1.0), mark_dirty=False)
        self._set_contrast(preset.get('contrast', 1.0), mark_dirty=False)

        # Restore full image + ZCA (cancels any ROI applied after the preset was saved)
        st.img_srgb_orig = preset['img_srgb_orig'].copy()
        st.img_srgb_view = preset['img_srgb_orig'].copy()
        st.img_srgb = st.img_srgb_orig
        st.mu_lin    = preset['mu_lin'].copy()
        st.W_lin     = preset['W_lin'].copy()
        st.Winv_lin  = preset['Winv_lin'].copy()
        st.samples_lin = preset['samples_lin'].copy()

        # Restore file-level image references
        if 'img_file_orig' in preset:
            st.img_file_orig = preset['img_file_orig'].copy()
        if '_img_file_initial' in preset:
            st._img_file_initial = preset['_img_file_initial'].copy()
        _pri = preset.get('_img_pre_roi')
        if _pri is not None:
            st._img_pre_roi = _pri.copy()
        # Restore stable ROI base (use img_roi_base if saved, fall back to img_srgb_orig)
        _rb = preset.get('img_roi_base')
        st.img_roi_base = _rb.copy() if _rb is not None else st.img_srgb_orig.copy()
        # Restore immutable original view
        iov = preset.get('img_original_view')
        if iov is not None:
            st.img_original_view = iov.copy()
        _rfs = preset.get('_raw_img_for_session')
        st._raw_img_for_session = _rfs.copy() if _rfs is not None else None
        st.modify_params = dict(preset['modify_params']) if preset.get('modify_params') is not None else None

        # Restore ROI / brush state
        st.paint_mode = preset.get('paint_mode', False)
        brush = preset.get('brush_alpha')
        st.brush_alpha = brush.copy() if brush is not None else (np.zeros_like(st.brush_alpha) if st.brush_alpha is not None else None)
        lba = preset.get('last_brush_alpha')
        st.last_brush_alpha = lba.copy() if lba is not None else None
        st.last_rot_quat = list(preset['last_rot_quat']) if preset.get('last_rot_quat') is not None else None
        lio = preset.get('last_img_srgb_orig')
        st.last_img_srgb_orig = lio.copy() if lio is not None else None
        lir = preset.get('last_img_srgb_reprocessed')
        st.last_img_srgb_reprocessed = lir.copy() if lir is not None else None
        st.ica_active = preset.get('ica_active', False)
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.setChecked(st.paint_mode)
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(st.ica_active)
            self.ica_btn.blockSignals(False)
        self._update_brush_panels_visibility()
        QtCore.QTimer.singleShot(0, self._equalize_3d_views)

        # Update all paintable textures (2D + model texture)
        for w in self._paint_widgets():
            w.brush_alpha = st.brush_alpha
            if hasattr(w, 'brush_tex') and st.brush_alpha is not None:
                w.brush_tex.set_data(st.brush_alpha.astype(np.float32))
            w.update_image_texture()
        if st.model_widget is not None:
            st.model_widget.update_texture_from_state()

        # Restore Fibonacci state directly — never via toggle_fib_norm()
        self._restore_fib_state_from_preset(preset)

        if st.widget_3d:
            st.widget_3d.update_markers()
            st.widget_3d.canvas.update()
        st.refresh_all()
        self.info_panel.refresh()
        self._update_reprocess_btn()
        self._update_original_thumbnail()
        self.statusBar().showMessage(f"Preset {idx+1} restored.", 2000)

    def _restore_fib_state_from_preset(self, preset):
        """Directly restores all Fibonacci/Radial-Push state from a preset dict.
        Never calls toggle_fib_norm() — avoids contamination from live _fib_img_backup
        and _mu_lin_orig that may belong to a different state."""
        st = self.state
        target_fib = preset.get('fib_norm_mode', False)

        # 1. Always restore the cached computation artefacts unconditionally
        fic = preset.get('fib_img_cached')
        st._fib_img_cached = fic.copy() if fic is not None else None

        fzc = preset.get('fib_Zs_norm_cached')
        st._fib_Zs_norm_cached = fzc.copy() if fzc is not None else None

        fib = preset.get('fib_img_backup')
        st._fib_img_backup = fib.copy() if fib is not None else None

        fmo = preset.get('fib_mu_lin_orig')
        if fmo is not None:
            st._mu_lin_orig = fmo.copy()
        fwo = preset.get('fib_W_lin_orig')
        if fwo is not None:
            st._W_lin_orig = fwo.copy()
        fwio = preset.get('fib_Winv_lin_orig')
        if fwio is not None:
            st._Winv_lin_orig = fwio.copy()

        fsz = preset.get('fib_samples_zca')
        st.samples_zca_fib = fsz.copy() if fsz is not None else None
        fzh = preset.get('fib_zca_half')
        if fzh is not None:
            st._fib_zca_half = float(fzh)

        # 2. Restore fib_norm_mode flag and sync the button
        st.fib_norm_mode = target_fib
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.blockSignals(True)
            self.fib_norm_btn.setChecked(target_fib)
            self.fib_norm_btn.blockSignals(False)

        # 3. If fib was active at snapshot time, img_srgb_orig is already the fib image
        #    (it was saved that way in _save_preset via 'img_srgb_orig').
        #    ZCA matrices and samples_lin were also saved as the fib-space values.
        #    Nothing more to swap — the caller already restored those fields.

        # 4. Update 3D mode label
        if st.widget_3d is not None:
            st.widget_3d.update_mode_label(fib_norm_mode=target_fib)

    def _restore_fib_state_from_session(self, fib_img, fib_mode):
        """Directly restores Radial Push state when loading a session.
        Never calls toggle_fib_norm() — avoids creating a spurious _fib_state_backup
        from whatever happens to be in memory at load time.

        At the point this is called, img_srgb_orig / mu_lin / W_lin / Winv_lin /
        samples_lin already hold the pre-fib working state (loaded from state_img /
        roi_base).  We capture those as the backup, then swap img_srgb_orig to fib_img.
        """
        st = self.state
        st._fib_img_cached = fib_img.copy()

        if fib_mode:
            # Build backup from the current pre-fib state
            st._fib_state_backup = {
                'img_srgb_orig': st.img_srgb_orig.copy(),
                'img_srgb_view': st.img_srgb_view.copy(),
                'mu_lin':        st.mu_lin.copy(),
                'W_lin':         st.W_lin.copy(),
                'Winv_lin':      st.Winv_lin.copy(),
                'samples_lin':   st.samples_lin.copy(),
            }
            st._fib_img_backup = st._fib_state_backup['img_srgb_orig']  # compat alias

            # Swap working image to the fib image
            st.img_srgb_orig = fib_img.copy()
            st.img_srgb_view = st.img_srgb_orig.copy()
            st.img_srgb      = st.img_srgb_orig

            # Recompute samples_lin from fib image (ZCA matrices stay as pre-fib backup)
            img_lin = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
            Xl = img_lin.reshape(-1, 3).astype(np.float32)
            rng = np.random.default_rng(42)
            N = min(N_SAMPLES_3D, Xl.shape[0])
            idx = rng.choice(Xl.shape[0], N, replace=False)
            st.samples_lin = Xl[idx]

            # Deactivate ICA — incompatible with Radial Push
            st.ica_active = False
            if self.ica_btn is not None:
                self.ica_btn.blockSignals(True)
                self.ica_btn.setChecked(False)
                self.ica_btn.blockSignals(False)

        st.fib_norm_mode = fib_mode
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.blockSignals(True)
            self.fib_norm_btn.setChecked(fib_mode)
            self.fib_norm_btn.blockSignals(False)
        if st.widget_3d is not None:
            st.widget_3d.update_mode_label(fib_norm_mode=fib_mode)
        for w in st.widgets_2d:
            w.update_image_texture()
        if st.widget_3d:
            st.widget_3d.update_markers()
            st.widget_3d.canvas.update()

    def _clear_presets(self):
        """Removes all preset buttons and data."""
        if not self._preset_buttons:
            return
        for btn in self._preset_buttons:
            self._preset_layout.removeWidget(btn)
            btn.deleteLater()
        self._preset_buttons.clear()
        self.state.presets.clear()

    _SUPPORTED_IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp') + tuple(_RAW_EXTENSIONS)
    _SUPPORTED_MODEL_EXT = ('.obj', '.ply')
    _SUPPORTED_SESSION_EXT = ('.rasc',)
    _SUPPORTED_EXT = _SUPPORTED_IMAGE_EXT + _SUPPORTED_MODEL_EXT + _SUPPORTED_SESSION_EXT

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(self._SUPPORTED_EXT):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = url.toLocalFile()
            if path.lower().endswith(self._SUPPORTED_SESSION_EXT):
                if not self._prompt_save_if_dirty():
                    return
                self._load_session_from_path(path)
                return
            elif path.lower().endswith(self._SUPPORTED_MODEL_EXT):
                if not self._prompt_save_if_dirty():
                    return
                self.action_open_model(path)
                return
            elif path.lower().endswith(self._SUPPORTED_IMAGE_EXT):
                if not self._prompt_save_if_dirty():
                    return
                self._open_image(path)

    def _has_save_context(self):
        """Returns True if there is enough context to propose saving a session."""
        return bool(
            getattr(self.state, 'current_image_path', None) or
            getattr(self.state, 'model_path', None) or
            getattr(self, '_last_session_path', None)
        )

    def _prompt_save_if_dirty(self):
        """If unsaved changes exist, ask Save/Discard/Cancel. Returns True to proceed, False to abort."""
        if not (self._session_dirty and self._has_save_context()):
            return True
        reply = QtWidgets.QMessageBox.question(
            self, "Save session?",
            "You have unsaved changes.\nSave session before continuing?",
            QtWidgets.QMessageBox.Save |
            QtWidgets.QMessageBox.Discard |
            QtWidgets.QMessageBox.Cancel
        )
        if reply == QtWidgets.QMessageBox.Cancel:
            return False
        if reply == QtWidgets.QMessageBox.Save:
            if self._last_session_path:
                default = self._last_session_path
            elif self.state.current_image_path:
                default = os.path.splitext(self.state.current_image_path)[0] + '.rasc'
            else:
                default = 'untitled.rasc'
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save session", default,
                "Rascal sessions (*.rasc);;All files (*)"
            )
            if not path:
                return False
            try:
                self._save_session_to_path(path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Save failed: {e}")
                return False
        return True

    def _resize_to_screen(self):
        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.availableGeometry()
        w = min(1920, int(geo.width() * 0.92))
        h = min(1080, int(geo.height() * 0.88))
        w = max(800, w)
        h = max(600, h)
        self.resize(w, h)
        self.move(
            geo.x() + (geo.width() - w) // 2,
            geo.y() + (geo.height() - h) // 2,
        )

    def closeEvent(self, event):
        if not self._prompt_save_if_dirty():
            event.ignore()
            return
        settings = QtCore.QSettings("Rascal", "rascal")
        settings.setValue("main_geometry", self.saveGeometry())
        event.accept()

    def action_open(self):
        if not self._prompt_save_if_dirty():
            return
        _raw_glob = ' '.join('*' + e for e in sorted(_RAW_EXTENSIONS))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", "",
            f"All supported (*.png *.jpg *.jpeg *.tif *.tiff *.bmp {_raw_glob} *.obj *.ply);;"
            f"Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp {_raw_glob});;"
            "3D model (*.obj *.ply)"
        )
        if not path:
            return
        if path.lower().endswith(('.obj', '.ply')):
            self.action_open_model(path)
        else:
            self._open_image(path)

    def _open_image(self, path):
        self._stop_autorotate()
        if self.w_color.split_active:
            self.w_color.toggle_split(False)
            if self._split_btn is not None:
                self._split_btn.setChecked(False)

        prog = QtWidgets.QProgressDialog(
            f"Reading {os.path.basename(path)}…", "", 0, 4, self)
        prog.setWindowTitle("Opening image")
        prog.setCancelButton(None)
        prog.setMinimumWidth(400)
        prog.setMinimumDuration(200)
        prog.setWindowModality(QtCore.Qt.ApplicationModal)
        prog.setValue(0)
        prog.show()
        QtWidgets.QApplication.processEvents()

        self._clear_presets()
        self._precision_warning_shown = False

        # Step 1 — read file + apply EXIF orientation
        raw = _imread_any(path)
        raw = _apply_exif_orientation(raw, path)
        orig_dtype = raw.dtype

        prog.setLabelText(f"Decoding {os.path.basename(path)}…")
        prog.setValue(1); QtWidgets.QApplication.processEvents()

        # Step 2 — normalise to float32 [0,1] and set image state
        img = _normalize_image_array(raw)
        del raw

        # Memory warning for very large images
        h, w = img.shape[:2]
        mpix = (h * w) / 1_000_000
        if mpix > _MPIX_WARN_THRESHOLD:
            est_mb = _estimate_image_memory_mb(h, w)
            total_ram = _get_total_ram_mb()
            ram_info = f"  (system RAM: {_format_memory_mb(total_ram)})" if total_ram else ""
            reply = QtWidgets.QMessageBox.warning(
                self, "Large image",
                f"This image is {w}×{h} ({mpix:.0f} Mpx).\n"
                f"Estimated memory usage: {_format_memory_mb(est_mb)}{ram_info}.\n\n"
                f"Each preset will add ~{_format_memory_mb(_estimate_image_memory_mb(h, w, 1) - est_mb)}.\n\n"
                "Continue loading?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            if reply == QtWidgets.QMessageBox.No:
                prog.close()
                return

        self.state._source_dtype        = orig_dtype
        self.state.img_file_orig        = img.copy()
        self.state._img_file_initial    = img.copy()
        self.state._img_pre_roi         = img.copy()
        self.state.img_original_view    = img.copy()
        self.state.img_roi_base         = img.copy()
        self.state.img_srgb_orig        = img.copy()
        self.state.img_srgb_view        = img.copy()
        self.state.img_srgb             = self.state.img_srgb_orig
        self.state.current_image_path   = path
        self.state._raw_img_for_session = None
        self.state.modify_params        = None
        self.state.brush_alpha          = np.zeros(img.shape[:2] + (1,), np.float32)
        self.state.presets              = []
        if self.state.mesh_data is None:
            self.state.model_path = None
        self.state._fib_img_cached      = None
        self.state._fib_Zs_norm_cached  = None
        self.state.samples_zca_fib      = None

        prog.setLabelText("Computing colour model (ZCA)…")
        prog.setValue(2); QtWidgets.QApplication.processEvents()

        # Step 3 — ZCA
        img_lin = ColorUtils.srgb_to_linear_np(np.clip(img, 0, 1))
        Xl = img_lin.reshape(-1, 3).astype(np.float32)
        self.state.mu_lin, self.state.W_lin, self.state.Winv_lin = \
            ColorUtils.zca_from_data(Xl)
        self.state._mu_lin_orig   = self.state.mu_lin.copy()
        self.state._W_lin_orig    = self.state.W_lin.copy()
        self.state._Winv_lin_orig = self.state.Winv_lin.copy()
        rng = np.random.default_rng(42)
        N = min(N_SAMPLES_3D, Xl.shape[0])
        self.state.samples_lin    = Xl[rng.choice(Xl.shape[0], N, replace=False)]
        self.state.disp_ax = self.state.disp_ay = self.state.disp_az = 0.0
        del Xl

        prog.setLabelText("Rendering…")
        prog.setValue(3); QtWidgets.QApplication.processEvents()
        self._set_push(1.0, mark_dirty=False)
        self._set_var_restore(1.0, mark_dirty=False)
        self._set_contrast(1.0, mark_dirty=False)
        self.reset_rotation()
        if self.state.widget_3d:
            self.state.widget_3d.update_markers(); self.state.widget_3d.canvas.update()
        self.info_panel.refresh()
        self._update_original_thumbnail()

        # Switch to image page
        self._stack.setCurrentIndex(1)
        self._save_3d_action.setVisible(False)
        self._apply_ui_level(self.ui_level)
        self._update_status_bar()
        if getattr(self, '_welcome_mode', False):
            self._set_welcome_mode(False)
        self._session_dirty = False

        # Flush all pending paint events so textures are uploaded and the
        # image becomes visible before we close the progress dialog.
        QtWidgets.QApplication.processEvents()
        prog.setValue(4); prog.close()

    def action_image_quality(self):
        """Launches the Image Quality assessment dialog for the current image."""
        st = self.state
        path = st.current_image_path
        if not path or not os.path.isfile(path):
            if getattr(st, 'img_file_orig', None) is not None:
                QtWidgets.QMessageBox.warning(
                    self, "Image Quality",
                    "The image is loaded in memory, but this operation requires "
                    "the original image file on disk."
                )
            else:
                QtWidgets.QMessageBox.warning(self, "Image Quality", "No image loaded.")
            return

        progress = QtWidgets.QProgressDialog("Image quality: starting...", "", 0, 100, self)
        progress.setWindowTitle("Image Quality")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setMinimumWidth(450)
        progress.setValue(0)
        progress.setWindowModality(QtCore.Qt.ApplicationModal)
        progress.show()
        QtWidgets.QApplication.processEvents()

        try:
            def _on_quality_progress(value, message):
                progress.setLabelText(message)
                progress.setValue(int(value))
                QtWidgets.QApplication.processEvents()

            metrics = sci_score(path, progress_cb=_on_quality_progress)
            progress.close()
            dlg = ImageQualityDialog(metrics, parent=self)
            dlg.exec_()
        except Exception as e:
            progress.close()
            QtWidgets.QMessageBox.critical(
                self,
                "Image Quality",
                f"Image quality computation failed:\n{e}"
            )

    def action_modify_image(self):
        """Launches the Modify Image (crop/contrast/HSL) dialog.
        Always operates on the original raw image so that adjustments are
        cumulative from the true source, not from an already-modified image."""
        st = self.state
        path = st.current_image_path
        _has_file = path and os.path.isfile(path)
        _has_memory = getattr(st, 'img_file_orig', None) is not None
        if not _has_file and not _has_memory:
            QtWidgets.QMessageBox.warning(self, "Modify Image", "No image loaded.")
            return

        is_3d = self._stack is not None and self._stack.currentIndex() == 2
        _prev = st.modify_params

        # Always feed the ORIGINAL raw image into the dialog so that
        # resetting sliders to 0 truly restores the original.
        _raw_src = getattr(st, '_raw_img_for_session', None)
        if _raw_src is not None:
            # Re-modify: use the saved pre-modify original
            crop_dlg = ImageCropDialog(
                path or "", parent=self,
                disable_crop_resize=is_3d,
                image_array=_raw_src,
                prev_params=_prev,
            )
        elif _has_file:
            crop_dlg = ImageCropDialog(path, parent=self, disable_crop_resize=is_3d, prev_params=_prev)
        else:
            # Disk file absent (portable session) — use the in-memory base image
            crop_dlg = ImageCropDialog(
                path or "", parent=self,
                disable_crop_resize=is_3d,
                image_array=(st._img_file_initial
                             if getattr(st, '_img_file_initial', None) is not None
                             else st.img_file_orig),
                prev_params=_prev,
            )
        if crop_dlg.exec_() == QtWidgets.QDialog.Accepted and crop_dlg.cropped_array is not None:
            new_params = {
                'crop_rect':  getattr(crop_dlg, 'crop_rect', None),
                'contrast':   getattr(crop_dlg, 'contrast', 0),
                'hsl':        getattr(crop_dlg, 'hsl', None),
            }
            # Detect if this is a full reset (all params at default)
            _is_reset_params = (
                new_params.get('crop_rect') is None
                and new_params.get('contrast', 0) == 0
                and not any(any(v != 0 for v in (new_params.get('hsl') or {}).get(ch, [0,0,0]))
                           for ch in ('Master', 'Reds', 'Greens', 'Blues'))
            )
            cropped = crop_dlg.cropped_array
            original_path = path
            # Save the original raw image only on the FIRST modify (preserve across re-modifies)
            _saved_raw = getattr(st, '_raw_img_for_session', None)
            if _saved_raw is None:
                _saved_raw = self.state._img_file_initial.copy()
            _is_reset = _is_reset_params
            # ---- Save exploration state BEFORE load_image_array ----
            # (load_image_array resets brush_alpha, disp_ax/ay/az, fib caches, presets)
            _prev_ang_x = st.rot.ang_x
            _prev_ang_y = st.rot.ang_y
            _prev_ang_z = st.rot.ang_z
            _prev_Ruser = st.rot.Ruser.copy()
            _prev_push  = st.push_factor
            _prev_var   = st.variance_restore
            _prev_fib   = st.fib_norm_mode
            _prev_brush  = st.brush_alpha.copy() if st.brush_alpha is not None else None
            _prev_paint  = st.paint_mode
            _prev_last_ba = st.last_brush_alpha
            _prev_last_orig = st.last_img_srgb_orig
            _prev_last_repr = st.last_img_srgb_reprocessed
            _prev_presets = list(st.presets)  # shallow copy of preset list
            _prev_source_dtype = st._source_dtype  # preserve original file dtype

            # Progress bar covering the heavy post-dialog work
            _prog = QtWidgets.QProgressDialog("Recomputing colour model…", None, 0, 4, self)
            _prog.setWindowTitle("Modify Image")
            _prog.setWindowModality(QtCore.Qt.ApplicationModal)
            _prog.setMinimumDuration(0)
            _prog.setMinimumWidth(350)
            _prog.setValue(0)
            _prog.show()
            QtWidgets.QApplication.processEvents()

            # Reload cropped image directly from array — ZCA is recomputed on cropped pixels
            # This preserves full bit depth (no 8-bit PNG degradation)
            # NOTE: load_image_array resets modify_params, _raw_img_for_session,
            #       brush_alpha, presets, fib caches, disp_ax/ay/az.
            self.state.load_image_array(cropped, pseudo_path=original_path)
            self.state._source_dtype = _prev_source_dtype  # restore original dtype to avoid false warning
            if _is_reset:
                # Full reset: clean state completely
                self.state.modify_params = None
                self.state._raw_img_for_session = None
            else:
                self.state.modify_params = new_params
                self.state._raw_img_for_session = _saved_raw

            # ---- Lightweight post-modify restore ----
            _prog.setLabelText("Restoring state…")
            _prog.setValue(1); QtWidgets.QApplication.processEvents()

            # Restore presets
            st.presets = _prev_presets

            # Restore rotation
            st.rot.ang_x = _prev_ang_x
            st.rot.ang_y = _prev_ang_y
            st.rot.ang_z = _prev_ang_z
            st.rot.Ruser = _prev_Ruser
            st.rot.recompose()
            st.disp_ax = _prev_ang_x
            st.disp_ay = _prev_ang_y
            st.disp_az = _prev_ang_z
            self._set_push(_prev_push, mark_dirty=False)
            self._set_var_restore(_prev_var, mark_dirty=False)

            # Restore brush/ROI if dimensions match (no crop/resize)
            new_h, new_w = st.img_srgb_orig.shape[:2]
            if (_prev_brush is not None
                    and _prev_brush.shape[0] == new_h
                    and _prev_brush.shape[1] == new_w):
                st.brush_alpha = _prev_brush
                st.paint_mode = _prev_paint
                st.last_brush_alpha = _prev_last_ba
                st.last_img_srgb_orig = _prev_last_orig
                st.last_img_srgb_reprocessed = _prev_last_repr
            else:
                # Dimensions changed (crop/resize) — brush no longer valid
                st.brush_alpha = np.zeros((new_h, new_w, 1), np.float32)
                st.paint_mode = False
                st.last_brush_alpha = None
                st.last_img_srgb_orig = None
                st.last_img_srgb_reprocessed = None
            if self.paint_mode_btn is not None:
                self.paint_mode_btn.setChecked(st.paint_mode)
            self._update_brush_panels_visibility()
            self._update_reprocess_btn()

            # Invalidate Fibonacci cache, re-trigger if was active
            st._fib_img_cached = None
            st._fib_Zs_norm_cached = None
            st.samples_zca_fib = None
            st.fib_norm_mode = False
            if self.fib_norm_btn is not None:
                self.fib_norm_btn.blockSignals(True)
                self.fib_norm_btn.setChecked(False)
                self.fib_norm_btn.blockSignals(False)
            if st.widget_3d is not None:
                st.widget_3d.update_mode_label(fib_norm_mode=False)

            # Reset zoom on 2D widgets
            for w in st.widgets_2d:
                if hasattr(w, "zoom_level"): w.zoom_level = 1.0
                if hasattr(w, "zoom_center"): w.zoom_center = np.array([0.0, 0.0], dtype=np.float32)

            # Update all view textures
            _prog.setLabelText("Rendering…")
            _prog.setValue(2); QtWidgets.QApplication.processEvents()
            for w in self._paint_widgets():
                w.update_image_texture()
                if st.brush_alpha is not None and hasattr(w, 'brush_tex'):
                    w.brush_alpha = st.brush_alpha
                    w.brush_tex.set_data(st.brush_alpha.astype(np.float32))
            if st.model_widget is not None:
                st.model_widget.update_texture_from_state()
            st.refresh_all()

            # Channel swap reset
            self._orig_swap_prev_mode = None
            self._swapped_channel = None
            self.w_color.mode = 0
            self.w_R.mode = 1
            self.w_G.mode = 2
            self.w_B.mode = 3
            self.w_color.prog['u_mode'] = 0
            self.w_R.prog['u_mode'] = 1
            self.w_G.prog['u_mode'] = 2
            self.w_B.prog['u_mode'] = 3

            if self.state.widget_3d:
                self.state.widget_3d.update_markers()
                self.state.widget_3d.canvas.update()
            self.info_panel.refresh()
            self._update_original_thumbnail()
            self._update_status_bar()
            self._session_dirty = True

            # Flush pending paint events so the image is visible
            _prog.setLabelText("Finalising…")
            _prog.setValue(3); QtWidgets.QApplication.processEvents()
            QtWidgets.QApplication.processEvents()
            _prog.setValue(4); _prog.close()

            # Re-trigger Fibonacci if it was active (after all state is set)
            if _prev_fib:
                self.toggle_fib_norm()
            if _is_reset:
                QtWidgets.QMessageBox.information(
                    self, "Modify Image",
                    "Image restored to original."
                )
            else:
                self.statusBar().showMessage(
                    f"Image modified ({cropped.shape[1]}×{cropped.shape[0]} px) — recomputed", 4000
                )

    def action_open_model(self, path=None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open 3D model", "", "3D models (*.obj *.ply);;Wavefront OBJ (*.obj);;Stanford PLY (*.ply)"
            )
        if not path:
            return

        prog = QtWidgets.QProgressDialog("Loading 3D model…", None, 0, 6, self)
        prog.setWindowTitle("Opening model")
        prog.setWindowModality(QtCore.Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setMinimumWidth(400)
        prog.setValue(0)
        QtWidgets.QApplication.processEvents()

        try:
            # 1. Load mesh locally FIRST without modifying global state
            prog.setLabelText("Parsing model file… This may take a while for large files.")
            prog.setValue(1); QtWidgets.QApplication.processEvents()
            if path.lower().endswith('.ply'):
                mesh = PlyLoader.load_ply(path)
            else:
                mesh = ObjLoader.load_obj(path)

            # 2. Verify texture BEFORE any state reset
            if not mesh.texture_path or not os.path.isfile(mesh.texture_path):
                prog.close()
                QtWidgets.QMessageBox.warning(
                    self, "3D model",
                    f"No texture found for this model.\n\n"
                    f"A texture image (.png / .jpg) must be present alongside the model file.\n"
                    f"The model was not loaded."
                )
                return

            # 3. NOW safe to reset state and assign model globally
            self._stop_autorotate()
            self._clear_presets()
            self._model_expert_push_init = False
            self.reset_rotation()

            self.state.mesh_data = mesh
            self.state.model_path = path

            prog.setLabelText("Building mesh…")
            prog.setValue(2); QtWidgets.QApplication.processEvents()
            self.w_model.set_mesh(mesh)
            self.w_model.reset_camera()
            # Opening a model always starts in 3D basic mode
            self._apply_model_ui_level('basic')
            if self.model_level_combo is not None:
                self.model_level_combo.blockSignals(True)
                self.model_level_combo.setCurrentIndex(0)
                self.model_level_combo.blockSignals(False)

            # Load the texture image as the working image
            prog.setLabelText("Loading texture image…")
            prog.setValue(3); QtWidgets.QApplication.processEvents()
            self._precision_warning_shown = False
            self.state.load_image(mesh.texture_path)

            prog.setLabelText("Uploading textures to GPU…")
            prog.setValue(4); QtWidgets.QApplication.processEvents()
            self._set_push(1.0, mark_dirty=False)
            self._set_var_restore(1.0, mark_dirty=False)
            for w in self.state.widgets_2d:
                w.update_image_texture()
            self.w_model_tex.update_image_texture()
            self.w_model.update_texture_from_state()

            prog.setLabelText("Finalizing…")
            prog.setValue(5); QtWidgets.QApplication.processEvents()
            # Switch to 3D page
            self._stack.setCurrentIndex(2)
            self._save_3d_action.setVisible(True)
            self._apply_model_ui_level(self.model_ui_level)
            self._update_status_bar()
            QtCore.QTimer.singleShot(0, self._equalize_3d_views)
            if getattr(self, '_welcome_mode', False):
                self._set_welcome_mode(False)

            prog.setValue(6)
            self.statusBar().showMessage(
                f"3D model loaded: {os.path.basename(path)} | texture: {os.path.basename(mesh.texture_path)}", 5000
            )
        except Exception as e:
            prog.close()
            QtWidgets.QMessageBox.critical(self, "3D model", f"Could not load model: {e}")

    def save_3d_model(self):
        """Export the current 3D model to OBJ or PLY format with the modified texture."""
        mesh = self.state.mesh_data
        if mesh is None or not mesh.is_valid:
            QtWidgets.QMessageBox.warning(self, "Save 3D model", "No 3D model loaded.")
            return

        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 3D model", "",
            "Wavefront OBJ (*.obj);;Stanford PLY (*.ply)"
        )
        if not path:
            return

        # Determine format from extension or filter
        ext = os.path.splitext(path)[1].lower()
        if ext not in ('.obj', '.ply'):
            if 'PLY' in selected_filter:
                path += '.ply'
                ext = '.ply'
            else:
                path += '.obj'
                ext = '.obj'

        try:
            # Render the modified texture at full resolution (with ZCA/rotation/push applied)
            tex_image = self._render_fullres_image(0)
            if ext == '.ply':
                self._write_ply(path, mesh, tex_image)
            else:
                self._write_obj(path, mesh, tex_image)
            self.statusBar().showMessage(f"3D model saved: {os.path.basename(path)}", 5000)
            QtWidgets.QMessageBox.information(self, "Save 3D model", f"Model saved successfully:\n{os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Save 3D model", f"Error saving model:\n{e}")

    def _write_obj(self, path, mesh, tex_image=None):
        """Write mesh data to Wavefront OBJ format (with MTL and modified texture)."""
        base = os.path.splitext(path)[0]
        mtl_name = os.path.basename(base) + '.mtl'
        has_texture = tex_image is not None

        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"# Exported by RASCAL\n")
            if has_texture:
                f.write(f"mtllib {mtl_name}\n")
            f.write(f"usemtl material0\n\n")

            # Vertices
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Texture coordinates
            has_uvs = mesh.uvs is not None and len(mesh.uvs) == len(mesh.vertices)
            if has_uvs:
                for uv in mesh.uvs:
                    f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Faces (1-indexed)
            indices = mesh.indices
            for i in range(0, len(indices), 3):
                i0, i1, i2 = int(indices[i]) + 1, int(indices[i+1]) + 1, int(indices[i+2]) + 1
                if has_uvs:
                    f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")
                else:
                    f.write(f"f {i0} {i1} {i2}\n")

        # Save the modified texture and write MTL file
        if has_texture:
            tex_filename = os.path.basename(base) + '_texture.png'
            out_dir = os.path.dirname(path)
            tex_out_path = os.path.join(out_dir, tex_filename)
            imageio.imwrite(pathlib.Path(tex_out_path), tex_image)
            mtl_path = base + '.mtl'
            with open(mtl_path, 'w', encoding='utf-8') as f:
                f.write(f"# Exported by RASCAL\n")
                f.write(f"newmtl material0\n")
                f.write(f"Ka 1.0 1.0 1.0\n")
                f.write(f"Kd 1.0 1.0 1.0\n")
                f.write(f"map_Kd {tex_filename}\n")

    def _write_ply(self, path, mesh, tex_image=None):
        """Write mesh data to binary little-endian PLY format."""
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.indices) // 3
        has_uvs = mesh.uvs is not None and len(mesh.uvs) == n_verts

        # Save modified texture
        tex_filename = None
        if tex_image is not None:
            tex_filename = os.path.splitext(os.path.basename(path))[0] + '_texture.png'
            out_dir = os.path.dirname(path)
            tex_out_path = os.path.join(out_dir, tex_filename)
            imageio.imwrite(pathlib.Path(tex_out_path), tex_image)

        header = "ply\n"
        header += "format binary_little_endian 1.0\n"
        if tex_filename:
            header += f"comment TextureFile {tex_filename}\n"
        header += f"element vertex {n_verts}\n"
        header += "property float x\n"
        header += "property float y\n"
        header += "property float z\n"
        if has_uvs:
            header += "property float s\n"
            header += "property float t\n"
        header += f"element face {n_faces}\n"
        header += "property list uchar int vertex_indices\n"
        header += "end_header\n"

        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))

            # Vertices - vectorized for performance
            if has_uvs:
                # Interleave xyz + uv: [x0,y0,z0,u0,v0, x1,y1,z1,u1,v1, ...]
                verts_uv = np.empty((n_verts, 5), dtype=np.float32)
                verts_uv[:, :3] = mesh.vertices
                verts_uv[:, 3:5] = mesh.uvs
                f.write(verts_uv.astype(np.float32).tobytes())
            else:
                f.write(mesh.vertices.astype(np.float32).tobytes())

            # Faces - vectorized: interleave [3, i0, i1, i2] for each triangle
            indices = mesh.indices.astype(np.int32)
            # Create structured array: uint8 count + 3 int32 indices per face
            face_dtype = np.dtype([('count', np.uint8), ('i0', np.int32), ('i1', np.int32), ('i2', np.int32)])
            face_data = np.empty(n_faces, dtype=face_dtype)
            face_data['count'] = 3
            tri_indices = indices.reshape(n_faces, 3)
            face_data['i0'] = tri_indices[:, 0]
            face_data['i1'] = tri_indices[:, 1]
            face_data['i2'] = tri_indices[:, 2]
            f.write(face_data.tobytes())

    _KEY_ROT_STEP = 0.035  # ~2° per key press (single tap)
    _KEY_ROT_HELD = 0.018   # ~1° per frame at 60 fps when held
    _ROT_KEYS = {
        QtCore.Qt.Key_Q, QtCore.Qt.Key_D,
        QtCore.Qt.Key_Z, QtCore.Qt.Key_S,
        QtCore.Qt.Key_A, QtCore.Qt.Key_E,
    }

    def _ensure_key_timer(self):
        if self._key_rot_timer is None:
            self._pressed_rot_keys = set()
            self._key_rot_timer = QtCore.QTimer(self)
            self._key_rot_timer.setInterval(16)  # ~60 fps
            self._key_rot_timer.timeout.connect(self._on_key_rot_tick)

    def _on_key_rot_tick(self):
        keys = self._pressed_rot_keys
        if not keys:
            self._key_rot_timer.stop()
            return
        st = self.state
        dth = self._KEY_ROT_HELD
        changed = False
        if QtCore.Qt.Key_Q in keys:
            st.rot.ang_x -= dth; st.disp_ax -= dth; st.active_axis = 'x'; changed = True
        if QtCore.Qt.Key_D in keys:
            st.rot.ang_x += dth; st.disp_ax += dth; st.active_axis = 'x'; changed = True
        if QtCore.Qt.Key_Z in keys:
            st.rot.ang_y -= dth; st.disp_ay -= dth; st.active_axis = 'y'; changed = True
        if QtCore.Qt.Key_S in keys:
            st.rot.ang_y += dth; st.disp_ay += dth; st.active_axis = 'y'; changed = True
        if QtCore.Qt.Key_A in keys:
            st.rot.ang_z -= dth; st.disp_az -= dth; st.active_axis = 'z'; changed = True
        if QtCore.Qt.Key_E in keys:
            st.rot.ang_z += dth; st.disp_az += dth; st.active_axis = 'z'; changed = True
        if changed:
            st.rot.recompose()
            if st.ica_active:
                st.ica_active = False
                if callable(st.on_ica_deactivated):
                    st.on_ica_deactivated()
            self._invalidate_auto_contrast()
            st.refresh_all()
            self._session_dirty = True

    def keyPressEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        in_2d_expert = (self._stack.currentIndex() == 1 and self.ui_level == 'expert')
        in_3d_expert = (self._stack.currentIndex() == 2 and self.model_ui_level == 'expert')
        can_adjust_push = in_2d_expert or in_3d_expert
        if can_adjust_push and key == QtCore.Qt.Key_Up:
            self._set_push(self.state.push_factor + 0.05)
        elif can_adjust_push and key == QtCore.Qt.Key_Down:
            self._set_push(self.state.push_factor - 0.05)
        # Keyboard rotation: q/d = X, z/s = Y, a/e = Z — continuous while held
        elif key in self._ROT_KEYS:
            self._stop_autorotate()
            self._ensure_key_timer()
            self._pressed_rot_keys.add(key)
            # Apply one step immediately so the first frame feels responsive
            st = self.state
            dth = self._KEY_ROT_STEP
            if key == QtCore.Qt.Key_Q:
                st.rot.ang_x -= dth; st.disp_ax -= dth; st.active_axis = 'x'
            elif key == QtCore.Qt.Key_D:
                st.rot.ang_x += dth; st.disp_ax += dth; st.active_axis = 'x'
            elif key == QtCore.Qt.Key_Z:
                st.rot.ang_y -= dth; st.disp_ay -= dth; st.active_axis = 'y'
            elif key == QtCore.Qt.Key_S:
                st.rot.ang_y += dth; st.disp_ay += dth; st.active_axis = 'y'
            elif key == QtCore.Qt.Key_A:
                st.rot.ang_z -= dth; st.disp_az -= dth; st.active_axis = 'z'
            elif key == QtCore.Qt.Key_E:
                st.rot.ang_z += dth; st.disp_az += dth; st.active_axis = 'z'
            st.rot.recompose()
            if st.ica_active:
                st.ica_active = False
                if callable(st.on_ica_deactivated):
                    st.on_ica_deactivated()
            st.refresh_all()
            if not self._key_rot_timer.isActive():
                self._key_rot_timer.start()
        elif key == QtCore.Qt.Key_T and in_2d_expert:
            self._toggle_split_view()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.isAutoRepeat():
            return
        key = event.key()
        if self._pressed_rot_keys:
            self._pressed_rot_keys.discard(key)
            if not self._pressed_rot_keys and self._key_rot_timer is not None:
                self._key_rot_timer.stop()
        super().keyReleaseEvent(event)

    def eventFilter(self, obj, event):
        # Intercept Escape globally to exit ROI mode without propagating to vispy/Qt
        if event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
            if self.state.paint_mode:
                self.exit_paint_mode_only()
            return True
        # In 3D page, route wheel to model controls ONLY when cursor is over model area.
        if (
            event.type() == QtCore.QEvent.Wheel
            and self._stack is not None
            and self._stack.currentIndex() == 2
            and self._model_host is not None
        ):
            try:
                top_left = self._model_host.mapToGlobal(QtCore.QPoint(0, 0))
                rect = QtCore.QRect(top_left, self._model_host.size())
                if rect.contains(QtGui.QCursor.pos()):
                    self.w_model.on_wheel(event.angleDelta().y(), event.modifiers())
                    return True
            except Exception:
                pass
        return super().eventFilter(obj, event)

    def _set_push(self, value, mark_dirty=True):
        value = round(max(0.1, min(5.0, value)), 2)
        self.state.push_factor = value
        if self.push_slider is not None:
            self.push_slider.blockSignals(True)
            self.push_slider.setValue(int(value * 100))
            self.push_slider.blockSignals(False)
        if self.push_val_label is not None:
            self.push_val_label.setText(f"{value:.2f}")
        self._invalidate_auto_contrast()
        self.state.refresh_all()
        if mark_dirty:
            self._session_dirty = True

    def _set_var_restore(self, value, mark_dirty=True):
        value = round(max(0.0, min(1.0, value)), 2)
        self.state.variance_restore = value
        if self.var_restore_slider is not None:
            self.var_restore_slider.blockSignals(True)
            self.var_restore_slider.setValue(int(value * 100))
            self.var_restore_slider.blockSignals(False)
        if self.var_restore_val_label is not None:
            self.var_restore_val_label.setText(f"{value:.2f}")
        self._invalidate_auto_contrast()
        self.state.refresh_all()
        if mark_dirty:
            self._session_dirty = True

    def _set_contrast(self, value, mark_dirty=True, invalidate_auto=True):
        value = round(max(0.5, min(3.0, value)), 2)
        self.state.contrast = value
        # When returning to manual contrast, reset the pivot to mid-gray
        if self.auto_contrast_btn is not None and not self.auto_contrast_btn.isChecked():
            self.state.contrast_center = 0.5
        if self.contrast_slider is not None:
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(int(value * 100))
            self.contrast_slider.blockSignals(False)
        if self.contrast_val_label is not None:
            self.contrast_val_label.setText(f"{value:.2f}")
        if invalidate_auto:
            self._invalidate_auto_contrast()
        self.state.refresh_all()
        if mark_dirty:
            self._session_dirty = True

    def _invalidate_auto_contrast(self):
        if self.auto_contrast_btn is not None and self.auto_contrast_btn.isChecked():
            self.auto_contrast_btn.setChecked(False)

    def _compute_auto_contrast(self):
        """Compute an automatic contrast factor and pivot for the current transformed view.
        Returns a value in [0.5, 3.0] and sets state.contrast_center to the median luminance."""
        try:
            rgb_u8 = self._render_fullres_image(0)
            H, W = rgb_u8.shape[:2]
            # Downsample large images for speed
            if max(H, W) > 1024:
                from PIL import Image
                scale = 1024 / max(H, W)
                new_w = int(W * scale)
                new_h = int(H * scale)
                pil = Image.fromarray(rgb_u8)
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                rgb_u8 = np.array(pil)
            # Rec. 601 luminance
            rgb = rgb_u8.astype(np.float32) / 255.0
            lum = (
                0.299 * rgb[..., 0] +
                0.587 * rgb[..., 1] +
                0.114 * rgb[..., 2]
            )
            p2, p98 = np.percentile(lum, [2.0, 98.0])
            dynamic = p98 - p2

            if dynamic < 0.03:
                self.state.contrast_center = float(np.median(lum))
                return 1.0

            factor = 1.0 / dynamic
            factor = float(np.clip(factor, 0.8, 3.0))

            self.state.contrast_center = float(np.median(lum))
            return round(factor, 2)
        except Exception as e:
            print(f"[RASCAL] Auto contrast computation failed: {e}")
            self.state.contrast_center = 0.5
            return 1.0

    def _on_auto_contrast_clicked(self, checked):
        if checked:
            factor = self._compute_auto_contrast()
            self._set_contrast(factor, mark_dirty=True, invalidate_auto=False)
        else:
            self.state.contrast_center = 0.5
            self._set_contrast(1.0, mark_dirty=True, invalidate_auto=False)

    def _on_pol_btn_clicked(self):
        st = self.state
        st.rot.ang_x = -st.rot.ang_x
        st.rot.ang_y = -st.rot.ang_y
        st.rot.ang_z = -st.rot.ang_z
        st.rot.recompose()
        if st.widget_3d is not None:
            st.widget_3d.update_markers()
            st.widget_3d.canvas.update()
        self._invalidate_auto_contrast()
        st.refresh_all()
        self.info_panel.refresh()
        self._session_dirty = True

    def toggle_fib_norm(self):
        if self.state.paint_mode:
            self.statusBar().showMessage("Radial Push not available in ROI mode", 2000)
            if self.fib_norm_btn is not None:
                self.fib_norm_btn.blockSignals(True)
                self.fib_norm_btn.setChecked(False)
                self.fib_norm_btn.blockSignals(False)
            return
        
        # Cancel any running worker before toggling state
        if self._fib_worker is not None and self._fib_worker.isRunning():
            self._fib_cancel()
            self.state.fib_norm_mode = False
            if self.fib_norm_btn is not None:
                self.fib_norm_btn.blockSignals(True)
                self.fib_norm_btn.setChecked(False)
                self.fib_norm_btn.blockSignals(False)
            self.statusBar().showMessage("Radial Push calculation cancelled.", 2000)
            return
        
        st = self.state
        if st.fib_norm_mode:
            # Currently active → deactivate
            self.disable_fib_norm()
            return
        # Currently inactive → activate
        st.fib_norm_mode = True
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.setChecked(True)
        if st.fib_norm_mode:
            if st._fib_img_cached is not None:
                # Reuse cache — no recalculation needed
                st._fib_state_backup = {
                    'img_srgb_orig': st.img_srgb_orig.copy(),
                    'img_srgb_view': st.img_srgb_view.copy(),
                    'mu_lin':        st.mu_lin.copy(),
                    'W_lin':         st.W_lin.copy(),
                    'Winv_lin':      st.Winv_lin.copy(),
                    'samples_lin':   st.samples_lin.copy(),
                }
                st._fib_img_backup = st._fib_state_backup['img_srgb_orig']  # kept for compat
                st.img_srgb_orig = st._fib_img_cached.copy()
                st.img_srgb_view = st.img_srgb_orig.copy()
                st.img_srgb = st.img_srgb_orig
                img_lin2 = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
                Xl2 = img_lin2.reshape(-1, 3).astype(np.float32)
                # Restore pre-Radial Push ZCA from backup (not from _orig which may be post-ROI stale)
                _bak = st._fib_state_backup
                st.mu_lin   = _bak['mu_lin'].copy()
                st.W_lin    = _bak['W_lin'].copy()
                st.Winv_lin = _bak['Winv_lin'].copy()
                rng = np.random.default_rng(42)
                N = min(N_SAMPLES_3D, Xl2.shape[0])
                idx = rng.choice(Xl2.shape[0], N, replace=False)
                st.samples_lin = Xl2[idx]
                if hasattr(st, '_fib_Zs_norm_cached') and st._fib_Zs_norm_cached is not None:
                    st.samples_zca_fib = st._fib_Zs_norm_cached[idx]
                    st._fib_zca_half = float(np.linalg.norm(st._fib_Zs_norm_cached, axis=1).max())
                    if st._fib_zca_half < 1e-8:
                        st._fib_zca_half = 1.0
                for w in st.widgets_2d:
                    w.update_image_texture()
                if st.widget_3d:
                    w3 = st.widget_3d
                    w3.update_markers()
                    w3.update_mode_label(fib_norm_mode=True)
                    w3.canvas.update()
                # Deactivate ICA — incompatible with Radial Push
                st.ica_active = False
                if self.ica_btn is not None:
                    self.ica_btn.blockSignals(True)
                    self.ica_btn.setChecked(False)
                    self.ica_btn.blockSignals(False)
                self._invalidate_auto_contrast()
                st.refresh_all()
                self._session_dirty = True
                self.statusBar().showMessage("Radial Push applied (cached).", 4000)
                return
            self._fib_start_calc()
        else:
            self.disable_fib_norm()

    def disable_fib_norm(self):
        """Deactivates Radial Push and fully restores image + ZCA from _fib_state_backup.
        Safe to call from any context (Basic mode switch, reset, etc.)."""
        st = self.state
        if not st.fib_norm_mode:
            return
        st.fib_norm_mode = False
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.blockSignals(True)
            self.fib_norm_btn.setChecked(False)
            self.fib_norm_btn.blockSignals(False)
        if st.widget_3d:
            w3 = st.widget_3d
            w3.update_mode_label(fib_norm_mode=False)
            if w3.show_decorrelated:
                w3.show_decorrelated = False
                for lbl in w3._cube_labels:
                    lbl.visible = True
                w3._rgb_axes.visible = True
                w3._axis_lines.visible = False
                for lbl in w3._zero_labels:
                    lbl.visible = False
        _bak = getattr(st, '_fib_state_backup', None)
        if _bak is not None:
            st.img_srgb_orig = _bak['img_srgb_orig'].copy()
            st.img_srgb_view = _bak['img_srgb_view'].copy()
            st.img_srgb      = st.img_srgb_orig
            st.mu_lin        = _bak['mu_lin'].copy()
            st.W_lin         = _bak['W_lin'].copy()
            st.Winv_lin      = _bak['Winv_lin'].copy()
            st.samples_lin   = _bak['samples_lin'].copy()
        elif hasattr(st, '_fib_img_backup'):
            # Legacy fallback: no backup dict, recompute from saved image
            st.img_srgb_orig = st._fib_img_backup.copy()
            st.img_srgb_view = st.img_srgb_orig.copy()
            st.img_srgb = st.img_srgb_orig
            img_lin = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
            Xl = img_lin.reshape(-1, 3).astype(np.float32)
            st.mu_lin, st.W_lin, st.Winv_lin = ColorUtils.zca_from_data(Xl)
            rng = np.random.default_rng(42)
            N = min(N_SAMPLES_3D, Xl.shape[0])
            idx = rng.choice(Xl.shape[0], N, replace=False)
            st.samples_lin = Xl[idx]
        for w in st.widgets_2d:
            w.update_image_texture()
        if st.widget_3d:
            st.widget_3d.update_markers()
            st.widget_3d.canvas.update()
        self._invalidate_auto_contrast()
        st.refresh_all()
        self._session_dirty = True
        self.statusBar().showMessage("Radial Push disabled.", 3000)

    def _fib_start_calc(self):
        if not self._check_memory_before_heavy_op("Radial Push"):
            self.state.fib_norm_mode = False
            if self.fib_norm_btn is not None:
                self.fib_norm_btn.blockSignals(True)
                self.fib_norm_btn.setChecked(False)
                self.fib_norm_btn.blockSignals(False)
            return
        st = self.state
        self._fib_progress_dlg = QtWidgets.QProgressDialog(
            "Radial Push in progress…", "Cancel", 0, 100, self
        )
        self._fib_progress_dlg.setWindowTitle("Radial Push")
        self._fib_progress_dlg.setMinimumWidth(400)
        self._fib_progress_dlg.setWindowModality(QtCore.Qt.NonModal)
        self._fib_progress_dlg.setMinimumDuration(0)
        self._fib_progress_dlg.setAutoClose(False)
        self._fib_progress_dlg.setValue(0)
        self._fib_progress_dlg.canceled.connect(self._fib_cancel)
        # Mask of painted pixels (last_brush_alpha saved after reprocess)
        self._fib_progress_dlg.show()
        QtWidgets.QApplication.processEvents()

        paint_mask = None
        brush = st.last_brush_alpha if st.last_brush_alpha is not None else st.brush_alpha
        if brush is not None and brush.max() > 0.5:
            paint_mask = brush[..., 0].ravel() > 0.5

        self._fib_err_msg = None
        self._fib_cancelled = False
        self._fib_worker = FibNormWorker(
            st.img_srgb_orig.copy(), st.mu_lin.copy(),
            st.W_lin.copy(), st.Winv_lin.copy(),
            paint_mask=paint_mask, push_factor=st.push_factor, parent=self
        )
        self._fib_worker.progress.connect(self._fib_on_progress)
        self._fib_worker.finished.connect(self._fib_on_finished)
        self._fib_worker.error.connect(self._fib_on_error)
        self._fib_worker.start()

    def _fib_on_progress(self, value, message):
        if self._fib_progress_dlg is not None:
            self._fib_progress_dlg.setValue(value)
            self._fib_progress_dlg.setLabelText(message)

    def _fib_cancel(self):
        if self._fib_worker is not None:
            self._fib_worker.cancel()
        self._fib_cancelled = True

    def _fib_on_error(self, msg):
        self._fib_err_msg = msg

    def _fib_on_finished(self, img_norm, Zs_norm):
        if self._fib_progress_dlg is not None:
            self._fib_progress_dlg.close()
            self._fib_progress_dlg = None
        st = self.state
        if img_norm is None:
            st.fib_norm_mode = False
            if self.fib_norm_btn is not None:
                self.fib_norm_btn.setChecked(False)
            if getattr(self, '_fib_cancelled', False):
                self._fib_cancelled = False
                self.statusBar().showMessage("Radial Push cancelled.", 3000)
            else:
                err = getattr(self, '_fib_err_msg', 'Unknown error')
                QtWidgets.QMessageBox.critical(self, "Radial Push error", err)
                self.statusBar().showMessage("Radial Push failed.", 4000)
            return
        
        # Validate that fib mode is still expected to be active
        # (user may have toggled off during calculation)
        if not st.fib_norm_mode:
            self.statusBar().showMessage("Radial Push result discarded (mode was disabled).", 2000)
            return
        
        st._fib_img_cached = img_norm
        st._fib_state_backup = {
            'img_srgb_orig': st.img_srgb_orig.copy(),
            'img_srgb_view': st.img_srgb_view.copy(),
            'mu_lin':        st.mu_lin.copy(),
            'W_lin':         st.W_lin.copy(),
            'Winv_lin':      st.Winv_lin.copy(),
            'samples_lin':   st.samples_lin.copy(),
        }
        st._fib_img_backup = st._fib_state_backup['img_srgb_orig']  # kept for compat
        st.img_srgb_orig = st._fib_img_cached.copy()
        st.img_srgb_view = st.img_srgb_orig.copy()
        st.img_srgb = st.img_srgb_orig
        img_lin2 = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
        Xl2 = img_lin2.reshape(-1, 3).astype(np.float32)
        # Restore pre-Radial Push ZCA from backup (not from _orig which may be post-ROI stale)
        _bak = st._fib_state_backup
        st.mu_lin   = _bak['mu_lin'].copy()
        st.W_lin    = _bak['W_lin'].copy()
        st.Winv_lin = _bak['Winv_lin'].copy()
        rng = np.random.default_rng(42)
        N = min(N_SAMPLES_3D, Xl2.shape[0])
        idx = rng.choice(Xl2.shape[0], N, replace=False)
        st.samples_lin = Xl2[idx]
        # Store sampled Zs_norm for direct 3D visualisation
        st._fib_Zs_norm_cached = Zs_norm
        st.samples_zca_fib = Zs_norm[idx]
        st._fib_zca_half = float(np.linalg.norm(Zs_norm, axis=1).max())
        if st._fib_zca_half < 1e-8:
            st._fib_zca_half = 1.0
        for w in st.widgets_2d:
            w.update_image_texture()
        if st.widget_3d:
            w3 = st.widget_3d
            w3.update_markers()
            w3.update_mode_label(fib_norm_mode=True)
            w3.canvas.update()
        # Deactivate ICA — incompatible with Radial Push
        st.ica_active = False
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(False)
            self.ica_btn.blockSignals(False)
        self._invalidate_auto_contrast()
        st.refresh_all()
        self._session_dirty = True
        self.statusBar().showMessage("Radial Push applied.", 4000)

    def _on_brush_size_changed(self, value):
        sender = self.sender()
        if sender is getattr(self, 'model_brush_size_spin', None):
            targets = [self.state.model_tex_widget] if self.state.model_tex_widget is not None else []
            # Also update 3D model widget brush radius
            if self.state.model_widget is not None:
                self.state.model_widget._brush_radius_px = int(value)
        else:
            targets = list(self.state.widgets_2d)
        for w in targets:
            if hasattr(w, 'brush_radius_px'):
                w.brush_radius_px = float(value)

    def _paint_widgets(self):
        widgets = list(self.state.widgets_2d)
        if self.state.model_tex_widget is not None and self.state.model_tex_widget not in widgets:
            widgets.append(self.state.model_tex_widget)
        return widgets

    def _update_brush_panels_visibility(self):
        in_3d_page = self._stack is not None and self._stack.currentIndex() == 2
        show_2d = (self.ui_level == 'expert') and self.state.paint_mode and not in_3d_page
        show_3d = (self.model_ui_level == 'expert') and self.state.paint_mode and in_3d_page
        if self.brush_panel is not None:
            self.brush_panel.setVisible(show_2d)
        if self.model_brush_panel is not None:
            self.model_brush_panel.setVisible(show_3d)

    def _on_roi_btn_clicked(self, checked=None):
        """Called when the ROI button or shortcut O is triggered.
        Toggles paint mode based on current state.paint_mode (ignores checked arg)."""
        if self.state.paint_mode:
            # Was ON → turn OFF: full deactivation
            self._deactivate_roi()
        else:
            # Was OFF → turn ON: activate paint mode
            self.state.paint_mode = True
            if self.paint_mode_btn is not None:
                self.paint_mode_btn.blockSignals(True)
                self.paint_mode_btn.setChecked(True)
                self.paint_mode_btn.blockSignals(False)
            self._update_brush_panels_visibility()
            if self.state.last_brush_alpha is not None:
                self.state.brush_alpha = self.state.last_brush_alpha.copy()
            for w in self._paint_widgets():
                w.brush_alpha = self.state.brush_alpha
                if hasattr(w, 'brush_tex'):
                    w.brush_tex.set_data(self.state.brush_alpha.astype(np.float32))
                w.update_image_texture()
            if self.state.model_widget is not None:
                self.state.model_widget.update_texture_from_state()
            self._update_reprocess_btn()
            self.state.refresh_all()
            self.info_panel.refresh()

    def exit_paint_mode_only(self):
        """Exits paint mode and clears the current brush stroke, without resetting the image,
        rotation, ICA, or Radial Push. Use this when the user simply wants to stop painting
        (e.g. Escape key) without discarding the current working state."""
        st = self.state
        blank = np.zeros_like(st.brush_alpha) if st.brush_alpha is not None else None
        if blank is not None:
            st.brush_alpha = blank
        st.paint_mode = False
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.blockSignals(True)
            self.paint_mode_btn.setChecked(False)
            self.paint_mode_btn.blockSignals(False)
        for w in self._paint_widgets():
            if blank is not None:
                w.brush_alpha = blank
                if hasattr(w, 'brush_tex'):
                    w.brush_tex.set_data(blank.astype(np.float32))
            w.canvas.update()
        self._update_brush_panels_visibility()
        self._update_reprocess_btn()
        self.state.refresh_all()
        self.info_panel.refresh()
        self.statusBar().showMessage("Paint mode disabled", 2000)

    def reset_to_roi_base(self):
        """Full ROI deactivation: resets image to img_roi_base, clears rotation, ICA,
        Radial Push, and brush state. Presets are preserved."""
        st = self.state

        # 1. Reset rotation to zero
        st.rot.ang_x = st.rot.ang_y = st.rot.ang_z = 0.0
        st.rot.recompose()
        st.disp_ax = st.disp_ay = st.disp_az = 0.0
        self._set_push(1.0, mark_dirty=False)
        self._set_var_restore(1.0, mark_dirty=False)
        for w in st.widgets_2d:
            if hasattr(w, "zoom_level"): w.zoom_level = 1.0
            if hasattr(w, "zoom_center"): w.zoom_center = np.array([0.0, 0.0], dtype=np.float32)

        # 2. ICA off
        st.ica_active = False
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(False)
            self.ica_btn.blockSignals(False)

        # 3. Radial Push off
        st._fib_img_cached = None
        st._fib_Zs_norm_cached = None
        st.samples_zca_fib = None
        st.fib_norm_mode = False
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.blockSignals(True)
            self.fib_norm_btn.setChecked(False)
            self.fib_norm_btn.blockSignals(False)
        if st.widget_3d is not None:
            st.widget_3d.update_mode_label(fib_norm_mode=False)

        # 4. Original image (before any ROI) — img_roi_base is stable after load/modify
        _rb = getattr(st, 'img_roi_base', None)
        _src = _rb if _rb is not None else st.img_file_orig
        st._img_file_initial = _src.copy()
        st.img_file_orig = _src.copy()
        st._img_pre_roi = _src.copy()
        st.img_srgb_orig = _src.copy()
        st.img_srgb_view = _src.copy()
        st.img_srgb = st.img_srgb_orig

        # 5. Recompute ZCA on original image
        img_lin = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
        Xl = img_lin.reshape(-1, 3).astype(np.float32)
        st.mu_lin, st.W_lin, st.Winv_lin = ColorUtils.zca_from_data(Xl)
        rng = np.random.default_rng(42)
        idx = rng.choice(Xl.shape[0], min(N_SAMPLES_3D, Xl.shape[0]), replace=False)
        st.samples_lin = Xl[idx]

        # 6. Clear brush
        blank = np.zeros_like(st.brush_alpha) if st.brush_alpha is not None else None
        if blank is not None:
            st.brush_alpha = blank
        st.last_brush_alpha = None
        st.last_img_srgb_orig = None
        st.last_img_srgb_reprocessed = None
        st.paint_mode = False
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.blockSignals(True)
            self.paint_mode_btn.setChecked(False)
            self.paint_mode_btn.blockSignals(False)

        # 7. Update all GPU textures
        for w in self._paint_widgets():
            w.update_image_texture()
            if blank is not None:
                w.brush_alpha = blank
                if hasattr(w, 'brush_tex'):
                    w.brush_tex.set_data(blank.astype(np.float32))
            w.canvas.update()

        if st.model_widget is not None:
            st.model_widget.update_texture_from_state()
        if st.widget_3d:
            st.widget_3d.update_markers()
            st.widget_3d.canvas.update()

        # 8. UI
        self._update_brush_panels_visibility()
        self._update_reprocess_btn()
        self.state.refresh_all()
        self.info_panel.refresh()
        self.statusBar().showMessage("ROI disabled — image and parameters reset", 2000)

    def _deactivate_roi(self):
        """Full ROI deactivation (kept for backward compatibility). Calls reset_to_roi_base()."""
        self.reset_to_roi_base()

    def toggle_state(self, name):
        setattr(self.state, name, not getattr(self.state, name))
        if name=="paint_mode":
            if self.paint_mode_btn is not None:
                self.paint_mode_btn.setChecked(self.state.paint_mode)
            self._update_brush_panels_visibility()
            if self.state.paint_mode:
                # Activation: restore the stroke mask so the user can repaint.
                # Do NOT restore img_srgb_orig from last_img_srgb_orig: the shader
                # reads u_tex_orig and applies the current ZCA to it; swapping the
                # pixel data without updating ZCA causes a contrast jump.
                # reprocess() already reads last_img_srgb_orig directly when needed.
                if self.state.last_brush_alpha is not None:
                    self.state.brush_alpha = self.state.last_brush_alpha.copy()
                for w in self._paint_widgets():
                    w.brush_alpha = self.state.brush_alpha
                    if hasattr(w, 'brush_tex'):
                        w.brush_tex.set_data(self.state.brush_alpha.astype(np.float32))
                    w.update_image_texture()
            else:
                # Deactivation: full reset preserving preset checkpoints
                self._reset_core(clear_presets=False)
                return
            if self.state.model_widget is not None:
                self.state.model_widget.update_texture_from_state()
            self._update_reprocess_btn()
        self.state.refresh_all()
        self.info_panel.refresh()

    # ---------- UI level ----------
    def _on_ui_level_changed(self, index):
        levels = ['basic', 'expert']
        self._apply_ui_level(levels[index])

    def _on_model_ui_level_changed(self, index):
        levels = ['basic', 'expert']
        self._apply_model_ui_level(levels[index])

    def _apply_model_ui_level(self, level):
        self.model_ui_level = 'expert' if level == 'expert' else 'basic'
        self.w_model.set_control_level(self.model_ui_level)
        is_expert = self.model_ui_level == 'expert'

        # Level row: 3D selector in the same location as 2D selector
        self.ui_level_combo.setVisible(False)
        self.model_level_combo.setVisible(True)
        idx = 1 if is_expert else 0
        self.model_level_combo.blockSignals(True)
        self.model_level_combo.setCurrentIndex(idx)
        self.model_level_combo.blockSignals(False)

        # 3D model page controls: expert => ICA + ROI + presets; no radial push
        self.ica_btn.setVisible(is_expert)
        self.pol_btn.setVisible(is_expert)
        self.paint_mode_btn.setVisible(is_expert)
        if self._preset_action is not None:
            self._preset_action.setVisible(is_expert)
        if self._preset_save_action is not None:
            self._preset_save_action.setVisible(is_expert)
        self.fib_norm_btn.setVisible(False)
        # Split not available in 3D mode
        if self._split_btn is not None:
            self._split_btn.setVisible(False)
            if self.w_color.split_active:
                self.w_color.toggle_split(False)
                self._split_btn.setChecked(False)
        self.push_label.setVisible(is_expert)
        self.push_slider.setVisible(is_expert)
        self.push_val_label.setVisible(is_expert)
        self.var_restore_label.setVisible(is_expert)
        self.var_restore_slider.setVisible(is_expert)
        self.var_restore_val_label.setVisible(is_expert)
        self.contrast_label.setVisible(is_expert)
        self.contrast_slider.setVisible(is_expert)
        self.contrast_val_label.setVisible(is_expert)
        if self.auto_contrast_btn is not None:
            self.auto_contrast_btn.setVisible(is_expert)

        # 3D expert left panel: rotation/quaternion (same place as 2D)
        if self._model_left_box is not None:
            self._model_left_box.setVisible(is_expert)

        # Initial push in 3D expert mode
        if is_expert and not self._model_expert_push_init:
            self._set_push(1.0, mark_dirty=False)
            self._model_expert_push_init = True

        # Keep only model view focused in this page
        self.box_3d.setVisible(False)
        self.model_box.setVisible(True)

        if not is_expert:
            if self.state.paint_mode:
                self.state.paint_mode = False
                if self.paint_mode_btn is not None:
                    self.paint_mode_btn.setChecked(False)
            if self._model_left_box is not None:
                self._model_left_box.setVisible(False)

        self._update_brush_panels_visibility()
        QtCore.QTimer.singleShot(0, self._equalize_3d_views)

    def _apply_ui_level(self, level):
        self.ui_level = level
        is_expert = level == 'expert'

        # Level row: 2D selector visible, 3D selector hidden
        self.ui_level_combo.setVisible(True)
        self.model_level_combo.setVisible(False)
        idx = 1 if is_expert else 0
        self.ui_level_combo.blockSignals(True)
        self.ui_level_combo.setCurrentIndex(idx)
        self.ui_level_combo.blockSignals(False)

        # Rotation info panel (left)
        self.info_panel.setVisible(is_expert)
        self._left_box.setVisible(is_expert)

        # Toolbar buttons
        self.ica_btn.setVisible(is_expert)
        self.pol_btn.setVisible(is_expert)
        self.paint_mode_btn.setVisible(is_expert)
        self.fib_norm_btn.setVisible(is_expert)
        if self._rand_rot_action is not None:
            self._rand_rot_action.setVisible(is_expert)
        if not is_expert:
            self._stop_random_rotation()

        # Split view: visible only in expert mode
        if self._split_btn is not None:
            self._split_btn.setVisible(is_expert)
            if not is_expert and self.w_color.split_active:
                self.w_color.toggle_split(False)
                self._split_btn.setChecked(False)

        # Preset checkpoints: visible only in expert mode
        if self._preset_action is not None:
            self._preset_action.setVisible(is_expert)
        if self._preset_save_action is not None:
            self._preset_save_action.setVisible(is_expert)

        # Push: visible only in expert mode
        self.push_label.setVisible(is_expert)
        self.push_slider.setVisible(is_expert)
        self.push_val_label.setVisible(is_expert)
        self.var_restore_label.setVisible(is_expert)
        self.var_restore_slider.setVisible(is_expert)
        self.var_restore_val_label.setVisible(is_expert)
        self.contrast_label.setVisible(is_expert)
        self.contrast_slider.setVisible(is_expert)
        self.contrast_val_label.setVisible(is_expert)
        if self.auto_contrast_btn is not None:
            self.auto_contrast_btn.setVisible(is_expert)

        # 3D cube: visible only in expert mode
        self.box_3d.setVisible(is_expert)
        # 3D model: never visible on the 2D image page
        self.model_box.setVisible(False)

        # If downgrading level, disable now-hidden active modes
        if not is_expert and self.state.paint_mode:
            self.state.paint_mode = False
            if self.paint_mode_btn is not None:
                self.paint_mode_btn.setChecked(False)
        if not is_expert and self.state.fib_norm_mode:
            self.disable_fib_norm()

        # Left panel tied to paint mode
        self._update_brush_panels_visibility()

    def _update_reprocess_btn(self):
        """Enables the Clear and Re-process buttons if brush_alpha contains painted pixels.
        Also keeps ROI button checked when a ROI is active."""
        has_paint = bool(self.state.brush_alpha is not None and
                         self.state.brush_alpha.max() > 0.5)
        has_roi = bool(self.state.last_brush_alpha is not None and
                      self.state.last_brush_alpha.max() > 0.5)
        
        if self.reprocess_btn is not None:
            self.reprocess_btn.setEnabled(has_paint)
        if self.model_reprocess_btn is not None:
            self.model_reprocess_btn.setEnabled(has_paint)
        if self.clear_brush_btn is not None:
            self.clear_brush_btn.setEnabled(has_paint)
        if self.model_clear_brush_btn is not None:
            self.model_clear_brush_btn.setEnabled(has_paint)
        
        # Keep ROI button checked if there's an active ROI (last_brush_alpha)
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.setChecked(has_roi or self.state.paint_mode)
        
        # Update brush overlay on 3D model widget
        if self.state.model_widget is not None:
            self.state.model_widget.set_brush_texture(self.state.brush_alpha)

    def clear_current_brush(self):
        """Clears only the current painted strokes (brush_alpha).
        Does NOT touch last_brush_alpha, last_img_srgb_orig, last_img_srgb_reprocessed
        or last_rot_quat — the memory of the last applied ROI is preserved."""
        st = self.state
        st.brush_alpha = np.zeros_like(st.brush_alpha)
        for w in self._paint_widgets():
            w.brush_alpha = st.brush_alpha
            if hasattr(w, 'brush_tex'):
                w.brush_tex.set_data(st.brush_alpha.astype(np.float32))
            w.canvas.update()
        if st.model_widget is not None:
            st.model_widget.set_brush_texture(st.brush_alpha)
        self._update_reprocess_btn()

    def clear_applied_roi_memory(self):
        """Clears the memory of the last applied ROI (last_brush_alpha,
        last_img_srgb_orig, last_img_srgb_reprocessed, last_rot_quat).
        Not wired to any button — call explicitly when a full ROI reset is needed."""
        st = self.state
        st.last_brush_alpha = None
        st.last_img_srgb_orig = None
        st.last_img_srgb_reprocessed = None
        st.last_rot_quat = None

    def reprocess(self):
        """Applies ZCA+rotation to painted pixels, saves the mask, clears strokes."""
        st = self.state
        if st.brush_alpha is None or st.brush_alpha.max() <= 0.5:
            return
        if not self._check_memory_before_heavy_op("ROI Reprocess"):
            return

        source_img = st.img_roi_base if hasattr(st, 'img_roi_base') else st.img_file_orig

        mask = st.brush_alpha[..., 0] > 0.5  # (H, W) bool
        yy, xx = np.nonzero(mask)
        if len(yy) == 0:
            return
        MIN_ROI_PIXELS = 100
        if len(yy) < MIN_ROI_PIXELS:
            QtWidgets.QMessageBox.warning(
                self, "ROI too small",
                f"The ROI contains only {len(yy)} painted pixel(s).\n"
                f"Minimum required: {MIN_ROI_PIXELS} pixels for a stable ZCA."
            )
            return

        # 1. Save mask, rotation and image before transformation
        st.last_brush_alpha = st.brush_alpha.copy()
        st.last_rot_quat = RotationState.R_to_quat(st.rot.Ruser).tolist()
        st.last_img_srgb_orig = source_img.copy()
        st.last_img_srgb_reprocessed = None  # filled after transformation

        # 2. Retrieve original FILE sRGB pixels under the mask
        pixels_srgb = source_img[yy, xx, :]  # (N, 3) float32 [0,1]

        # 3. Convert to linear
        pixels_lin = ColorUtils.srgb_to_linear_np(pixels_srgb.astype(np.float32))

        # 4. Compute ZCA strictly on painted pixels
        mu_b, W_b, Winv_b = ColorUtils.zca_from_data(pixels_lin)

        # 5. Apply transform to ALL image pixels
        #    - whitening: local ZCA of painted pixels (mu_b, W_b)
        #    - back to RGB: inverse ZCA from the ROI-local model (mu_b, Winv_b)
        R = st.rot.Ruser
        all_lin = ColorUtils.srgb_to_linear_np(
            np.clip(source_img, 0, 1).reshape(-1, 3).astype(np.float32)
        )

        z_all  = (all_lin - mu_b) @ W_b.T
        z2_all = z_all @ R
        xo_all = np.clip(mu_b + (z2_all @ Winv_b.T), 0.0, 1.0)

        all_out = ColorUtils.linear_to_srgb_np(xo_all).reshape(source_img.shape)

        # 6. Write result into img_srgb_view (full transformed image)
        st.img_srgb_view = all_out.astype(np.float32)

        # 7. Promote img_srgb_view as new reference
        st.last_img_srgb_reprocessed = st.img_srgb_view.copy()
        st.img_srgb_orig = st.img_srgb_view.copy()
        st.img_srgb = st.img_srgb_orig

        # 9. Keep ROI-local ZCA as the new global ZCA (no global recomputation)
        st.mu_lin = mu_b
        st.W_lin = W_b
        st.Winv_lin = Winv_b
        img_lin_new = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
        Xl = img_lin_new.reshape(-1, 3).astype(np.float32)
        rng = np.random.default_rng(42)
        N = min(N_SAMPLES_3D, Xl.shape[0])
        idx = rng.choice(Xl.shape[0], N, replace=False)
        st.samples_lin = Xl[idx]

        # 10. Reset rotation
        st.rot.ang_x = st.rot.ang_y = st.rot.ang_z = 0.0
        st.rot.recompose()
        st.disp_ax = st.disp_ay = st.disp_az = 0.0

        # 11. Clear brush
        st.brush_alpha = np.zeros_like(st.brush_alpha)

        # 12. Disable paint mode
        st.paint_mode = False
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.setChecked(False)
        self._update_brush_panels_visibility()
        self._update_reprocess_btn()

        self._session_dirty = True

        # 13. Update all views
        for w in self._paint_widgets():
            w.brush_alpha = st.brush_alpha
            if hasattr(w, 'brush_tex'):
                w.brush_tex.set_data(st.brush_alpha.astype(np.float32))
            w.update_image_texture()
            w.canvas.update()
        if st.model_widget is not None:
            st.model_widget.update_texture_from_state()
        if st.widget_3d:
            st.widget_3d.update_markers()
            st.widget_3d.canvas.update()
        self.info_panel.refresh()

        # 14. Invalidate Fibonacci cache and promote new image as the file reference
        st._fib_img_cached = None
        st._fib_Zs_norm_cached = None
        st.samples_zca_fib = None
        st.fib_norm_mode = False
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.setChecked(False)
        if st.widget_3d is not None:
            st.widget_3d.update_mode_label(fib_norm_mode=False)
        # The ROI result becomes the current working/display image,
        # but must NOT replace img_roi_base or any original image reference.
        # Future ROI operations must still start from img_roi_base.
        self._update_original_thumbnail()

        self.statusBar().showMessage("ROI applied — image recomputed", 4000)

    def _reset_core(self, clear_presets=True):
        """Full reset of image, rotation, ZCA, Fibonacci and brush state.
        If clear_presets is False, preset checkpoints are preserved."""
        if clear_presets:
            self._clear_presets()
        # Reset rotation
        self.state.rot.ang_x = self.state.rot.ang_y = self.state.rot.ang_z = 0.0
        self.state.rot.recompose()
        self.state.disp_ax = self.state.disp_ay = self.state.disp_az = 0.0
        self._set_push(1.0, mark_dirty=False)
        self._set_var_restore(1.0, mark_dirty=False)
        self._set_contrast(1.0, mark_dirty=False)
        for w in self.state.widgets_2d:
            if hasattr(w, "zoom_level"): w.zoom_level = 1.0
            if hasattr(w, "zoom_center"): w.zoom_center = np.array([0.0, 0.0], dtype=np.float32)

        # Restore original file image + recompute ZCA
        st = self.state
        st._fib_img_cached = None   # invalidate Fibonacci cache
        st._fib_Zs_norm_cached = None
        st.samples_zca_fib = None
        st.fib_norm_mode = False
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.setChecked(False)
        if st.widget_3d is not None:
            st.widget_3d.update_mode_label(fib_norm_mode=False)
        # Restore stable base image — img_roi_base is set at load/modify and never overwritten by reprocess
        _rb = getattr(st, 'img_roi_base', None)
        _src = _rb if _rb is not None else st.img_file_orig
        st.img_file_orig = _src.copy()
        st._img_pre_roi = _src.copy()
        st.img_srgb_orig = _src.copy()
        st.img_srgb_view = _src.copy()
        st.img_srgb = st.img_srgb_orig
        img_lin = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
        Xl = img_lin.reshape(-1, 3).astype(np.float32)
        st.mu_lin, st.W_lin, st.Winv_lin = ColorUtils.zca_from_data(Xl)
        rng = np.random.default_rng(42)
        N = min(N_SAMPLES_3D, Xl.shape[0])
        idx = rng.choice(Xl.shape[0], N, replace=False)
        st.samples_lin = Xl[idx]

        # Clear all brush memory
        if st.brush_alpha is not None:
            st.brush_alpha = np.zeros_like(st.brush_alpha)
        st.last_brush_alpha = None
        st.last_img_srgb_orig = None
        st.last_img_srgb_reprocessed = None
        st.paint_mode = False
        st.ica_active = False  # Deactivate ICA on reset
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.blockSignals(True)
            self.paint_mode_btn.setChecked(False)
            self.paint_mode_btn.blockSignals(False)
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(False)
            self.ica_btn.blockSignals(False)
        self._update_brush_panels_visibility()
        self._update_reprocess_btn()
        blank = np.zeros_like(st.brush_alpha) if st.brush_alpha is not None else None
        for w in self._paint_widgets():
            w.update_image_texture()
            if blank is not None:
                w.brush_alpha = blank
                if hasattr(w, 'brush_tex'):
                    w.brush_tex.set_data(blank.astype(np.float32))

        if st.model_widget is not None:
            st.model_widget.update_texture_from_state()
        
        # Reset views to initial state (RGB centre, R/G/B right)
        self._orig_swap_prev_mode = None
        self._swapped_channel = None
        self.w_color.mode = 0
        self.w_R.mode = 1
        self.w_G.mode = 2
        self.w_B.mode = 3

        # Update shaders
        self.w_color.prog['u_mode'] = 0
        self.w_R.prog['u_mode'] = 1
        self.w_G.prog['u_mode'] = 2
        self.w_B.prog['u_mode'] = 3

        # Update GroupBox titles
        self.color_box.setTitle(self.mode_titles[0])
        self.R_box.setTitle(self.mode_titles[1])
        self.G_box.setTitle(self.mode_titles[2])
        self.B_box.setTitle(self.mode_titles[3])
        
        self.state.refresh_all()
        self.statusBar().showMessage("Full reset (rotation + views)", 2000)

    def reset_rotation(self):
        self._reset_core(clear_presets=False)

    def _reset_state_keep_image(self):
        """Resets rotation, push, brush and UI state without touching the current image.
        Used after action_modify_image where load_image already set the correct image."""
        self._clear_presets()
        st = self.state
        st.rot.ang_x = st.rot.ang_y = st.rot.ang_z = 0.0
        st.rot.recompose()
        st.disp_ax = st.disp_ay = st.disp_az = 0.0
        self._set_push(1.0, mark_dirty=False)
        self._set_var_restore(1.0, mark_dirty=False)
        self._set_contrast(1.0, mark_dirty=False)
        for w in st.widgets_2d:
            if hasattr(w, "zoom_level"): w.zoom_level = 1.0
            if hasattr(w, "zoom_center"): w.zoom_center = np.array([0.0, 0.0], dtype=np.float32)
        st._fib_img_cached = None
        st._fib_Zs_norm_cached = None
        st.samples_zca_fib = None
        st.fib_norm_mode = False
        if self.fib_norm_btn is not None:
            self.fib_norm_btn.setChecked(False)
        if st.widget_3d is not None:
            st.widget_3d.update_mode_label(fib_norm_mode=False)
        # ZCA already computed by load_image — no image reset needed
        if st.brush_alpha is not None:
            st.brush_alpha = np.zeros_like(st.brush_alpha)
        st.last_brush_alpha = None
        st.last_img_srgb_orig = None
        st.last_img_srgb_reprocessed = None
        st.paint_mode = False
        st.ica_active = False
        if self.paint_mode_btn is not None:
            self.paint_mode_btn.setChecked(False)
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(False)
            self.ica_btn.blockSignals(False)
        self._update_brush_panels_visibility()
        self._update_reprocess_btn()
        blank = np.zeros_like(st.brush_alpha) if st.brush_alpha is not None else None
        for w in self._paint_widgets():
            w.update_image_texture()
            if blank is not None:
                w.brush_alpha = blank
                if hasattr(w, 'brush_tex'):
                    w.brush_tex.set_data(blank.astype(np.float32))
        if st.model_widget is not None:
            st.model_widget.update_texture_from_state()
        self._orig_swap_prev_mode = None
        self._swapped_channel = None
        self.w_color.mode = 0
        self.w_R.mode = 1
        self.w_G.mode = 2
        self.w_B.mode = 3
        self.w_color.prog['u_mode'] = 0
        self.w_R.prog['u_mode'] = 1
        self.w_G.prog['u_mode'] = 2
        self.w_B.prog['u_mode'] = 3
        self.color_box.setTitle(self.mode_titles[0])
        self.R_box.setTitle(self.mode_titles[1])
        self.G_box.setTitle(self.mode_titles[2])
        self.B_box.setTitle(self.mode_titles[3])
        st.refresh_all()

    # ---------- ICA ----------
    def _run_ica(self):
        """Compute ICA rotation on pixels using the existing ZCA.
        If ROI is active, uses only ROI pixels (up to 100 000).
        Otherwise uses 100 000 random pixels from the full image.
        The resulting rotation matrix is decomposed into Euler angles (x, y, z)
        and applied as the current Ruser."""
        if self.state.paint_mode:
            self.statusBar().showMessage("ICA not available in ROI mode", 2000)
            if self.ica_btn is not None:
                self.ica_btn.blockSignals(True)
                self.ica_btn.setChecked(False)
                self.ica_btn.blockSignals(False)
            return
        if self.state.fib_norm_mode:
            if self.ica_btn is not None:
                self.ica_btn.blockSignals(True)
                self.ica_btn.setChecked(False)
                self.ica_btn.blockSignals(False)
            QtWidgets.QMessageBox.warning(
                self, "ICA unavailable",
                "ICA cannot be used while <b>Radial Push</b> is active.<br><br>"
                "Deactivate Radial Push first, then retry ICA."
            )
            return
        st = self.state
        img_lin = ColorUtils.srgb_to_linear_np(np.clip(st.img_srgb_orig, 0, 1))
        Xl = img_lin.reshape(-1, 3).astype(np.float64)

        # Check if ROI is active (last_brush_alpha contains the ROI mask)
        if st.last_brush_alpha is not None and st.last_brush_alpha.max() > 0.5:
            # ROI mode: use only painted pixels
            mask = st.last_brush_alpha[..., 0] > 0.5  # (H, W) bool
            yy, xx = np.nonzero(mask)
            n_roi_pixels = len(yy)
            
            if n_roi_pixels == 0:
                self.statusBar().showMessage("No ROI pixels found for ICA.", 2000)
                return
            
            # Get ROI pixels and limit to 100 000 if necessary
            roi_indices = yy * img_lin.shape[1] + xx  # Convert 2D to 1D indices
            if n_roi_pixels <= 100_000:
                # Use all ROI pixels
                X_sub = Xl[roi_indices]
                message = f"ICA computed on all {n_roi_pixels} ROI pixels."
            else:
                # Sample 100 000 random pixels from ROI
                rng = np.random.default_rng(0)
                idx = rng.choice(n_roi_pixels, 100_000, replace=False)
                X_sub = Xl[roi_indices[idx]]
                message = f"ICA computed on 100 000 sampled ROI pixels (out of {n_roi_pixels})."
        else:
            # Full image mode: sample up to 100 000 random pixels
            n_samples = min(100_000, Xl.shape[0])
            rng = np.random.default_rng(0)
            idx = rng.choice(Xl.shape[0], n_samples, replace=False)
            X_sub = Xl[idx]
            message = f"ICA computed on {n_samples} random pixels from full image."

        # Whiten using existing ZCA parameters
        mu = st.mu_lin.astype(np.float64)
        W = st.W_lin.astype(np.float64)
        Z = (X_sub - mu) @ W.T  # (n_samples, 3) whitened data

        # FastICA (fixed-point, kurtosis/logcosh) on whitened data to find rotation
        R = self._fastica_rotation(Z)

        # Ensure R is a proper rotation (det=+1)
        if np.linalg.det(R) < 0:
            R[-1, :] *= -1
    
        # Decompose into Euler angles and apply
        R32 = R.astype(np.float32)
        ax, ay, az = RotationState.R_to_euler_zyx(R32)
        st.rot.ang_x = ax; st.rot.ang_y = ay; st.rot.ang_z = az
        st.rot.Ruser = R32
        st.disp_ax = ax; st.disp_ay = ay; st.disp_az = az
        st.ica_active = True  # Mark ICA as active
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(True)
            self.ica_btn.blockSignals(False)
        self._invalidate_auto_contrast()
        st.refresh_all()
        self._session_dirty = True
        self.statusBar().showMessage(f"ICA rotation applied. {message}", 3000)

    @staticmethod
    def _fastica_rotation(Z, max_iter=200, tol=1e-6):
        """FastICA deflation on pre-whitened 3D data. Returns a 3×3 rotation matrix.
        Returns identity if the data is degenerate (homogeneous, dark, or near-zero variance)."""
        n, p = Z.shape  # p == 3

        # Early exit: if the data has no usable variance, ICA is undefined
        if n < p or np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            return np.eye(p, dtype=np.float64)

        W_ica = np.zeros((p, p), dtype=np.float64)

        for i in range(p):
            w = np.random.default_rng(i).standard_normal(p)
            norm_w = np.linalg.norm(w)
            if norm_w < 1e-12:
                w = np.eye(p)[i]
            else:
                w /= norm_w

            for _ in range(max_iter):
                # g(u) = tanh(u),  g'(u) = 1 - tanh²(u)
                proj = Z @ w                          # (n,)
                g = np.tanh(proj)                     # (n,)
                gp = 1.0 - g ** 2                     # (n,)
                w_new = (Z.T @ g) / n - gp.mean() * w

                # Deflation: orthogonalise against previous components
                for j in range(i):
                    w_new -= np.dot(w_new, W_ica[j]) * W_ica[j]

                # Guard: zero norm or NaN means degenerate update — keep current w
                norm_new = np.linalg.norm(w_new)
                if norm_new < 1e-12 or np.any(np.isnan(w_new)):
                    break
                w_new /= norm_new

                # Convergence check
                if abs(abs(np.dot(w_new, w)) - 1.0) < tol:
                    w = w_new
                    break
                w = w_new

            W_ica[i] = w

        # Safety net: if W_ica is invalid fall back to identity
        if np.any(np.isnan(W_ica)) or np.any(np.isinf(W_ica)):
            return np.eye(p, dtype=np.float64)

        # W_ica rows are the unmixing directions; form rotation via SVD to ensure orthogonality
        U, _, Vt = np.linalg.svd(W_ica)
        R = (U @ Vt)  # closest orthogonal matrix
        return R

    def _toggle_random_rotation(self):
        if self._rand_rot_timer is not None and self._rand_rot_timer.isActive():
            self._stop_random_rotation()
            return
        self._stop_autorotate()
        if self._rand_rot_timer is None:
            self._rand_rot_timer = QtCore.QTimer(self)
            self._rand_rot_timer.setInterval(3000)
            self._rand_rot_timer.timeout.connect(self._apply_random_rotation)
        if self._rand_rot_action is not None:
            self._rand_rot_action.setChecked(True)
        self._apply_random_rotation()
        self._rand_rot_timer.start()

    def _stop_random_rotation(self):
        if self._rand_rot_timer is not None:
            self._rand_rot_timer.stop()
        if self._rand_rot_action is not None:
            self._rand_rot_action.setChecked(False)

    def _apply_random_rotation(self):
        st = self.state
        M = np.random.randn(3, 3)
        Q, R_qr = np.linalg.qr(M)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        st.rot.Ruser = Q.astype(np.float32)
        ax, ay, az = st.rot.R_to_euler_zyx(Q)
        st.rot.ang_x = ax; st.rot.ang_y = ay; st.rot.ang_z = az
        st.disp_ax = ax; st.disp_ay = ay; st.disp_az = az
        st.ica_active = False
        if self.ica_btn is not None:
            self.ica_btn.blockSignals(True)
            self.ica_btn.setChecked(False)
            self.ica_btn.blockSignals(False)
        self._invalidate_auto_contrast()
        st.refresh_all()
        self._session_dirty = True

    def toggle_autorotate(self):
        self.auto = not self.auto
        if self.auto_rotate_btn is not None:
            self.auto_rotate_btn.setChecked(self.auto)
        if self.auto_rotate_action is not None:
            self.auto_rotate_action.setChecked(self.auto)
        if self.auto:
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self, _):
        if self.auto:
            st = self.state
            dth = 0.02
            if st.active_axis == 'x':
                st.rot.ang_x += dth
            elif st.active_axis == 'y':
                st.rot.ang_y += dth
            else:
                st.rot.ang_z += dth
            st.rot.recompose()
            
            # Deactivate ICA when auto-rotation is applied
            if st.ica_active:
                st.ica_active = False
                if callable(st.on_ica_deactivated):
                    st.on_ica_deactivated()
            self._invalidate_auto_contrast()
            st.refresh_all()
            self._session_dirty = True

    def _encode_img_png_b64(self, img, label="state_img"):
        """Encode a float [0,1] RGB image as base64 PNG. Returns None on failure."""
        if img is None or img.size == 0:
            return None
        tmp = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            imageio.imwrite(tmp.name, img_u8)
            with open(tmp.name, 'rb') as f:
                data_bytes = f.read()
            return base64.b64encode(data_bytes).decode('ascii')
        except Exception as e:
            print(f"[RASCAL] Failed to encode {label}: {e}")
            return None
        finally:
            if tmp is not None and os.path.exists(tmp.name):
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

    def _encode_numpy_b64(self, arr, label="array"):
        """Encode a numpy array as base64 npy. Returns None on failure."""
        if arr is None:
            return None
        try:
            buf = io.BytesIO()
            np.save(buf, arr)
            return base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as e:
            print(f"[RASCAL] Failed to encode {label}: {e}")
            return None

    def _decode_numpy_b64(self, b64_str, label="array"):
        """Decode a base64 npy string to numpy array. Returns None on failure."""
        if b64_str is None:
            return None
        try:
            buf = io.BytesIO(base64.b64decode(b64_str))
            return np.load(buf)
        except Exception as e:
            print(f"[RASCAL] Failed to decode {label}: {e}")
            return None

    def _save_session_to_path(self, path):
        """Saves the current session to the given .rasc path (no dialog)."""
        st = self.state

        # Warn once if the source image had >8-bit or float precision,
        # since all images embedded in .rasc are stored as 8-bit PNG.
        _src_dtype = getattr(st, '_source_dtype', None)
        if _src_dtype is not None:
            _is_high_depth = (np.issubdtype(_src_dtype, np.floating) or
                              _src_dtype == np.uint16 or
                              getattr(_src_dtype, 'itemsize', 1) > 1)
            if _is_high_depth and not self._precision_warning_shown:
                QtWidgets.QMessageBox.information(
                    self, "Session save — precision notice",
                    f"The source image has {_src_dtype} precision.\n\n"
                    "Images embedded in the .rasc session file are stored as 8-bit PNG.\n"
                    "The session will restore visually, but precision beyond 8-bit is lost in embedded session images."
                )
                self._precision_warning_shown = True

        Rq = st.rot.Ruser
        q = RotationState.R_to_quat(Rq)

        def _mask_to_b64(alpha_2d):
            mask_u8 = (np.clip(alpha_2d, 0, 1) * 255).astype(np.uint8)
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.close()
            imageio.imwrite(tmp.name, mask_u8)
            with open(tmp.name, 'rb') as f:
                data_bytes = f.read()
            os.unlink(tmp.name)
            return base64.b64encode(data_bytes).decode('ascii')

        # Encode brush mask as base64 PNG
        brush_b64 = None
        if st.brush_alpha is not None and st.brush_alpha.max() > 0.0:
            brush_b64 = _mask_to_b64(st.brush_alpha[..., 0])

        # Encode last_brush_alpha mask if available (before re-process)
        last_brush_b64 = None
        if st.last_brush_alpha is not None and st.last_brush_alpha.max() > 0.0:
            last_brush_b64 = _mask_to_b64(st.last_brush_alpha[..., 0])

        # Re-process quaternion (used to replay the ROI transform on load)
        rot_q = st.last_rot_quat if st.last_rot_quat is not None else \
                [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        # Current quaternion (the rotation the user is at right now — saved separately)
        current_q = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

        # Encode the exact current working image state unconditionally.
        # This covers ROI, Modify Image, and Radial Push — any state that diverges
        # from the raw file on reload. On load, this is used directly, skipping all replay.
        # When fib is active, use the full pre-fib backup (image + matrices) so that
        # state_img and state_{mu,W,Winv}_lin are always consistent with each other.
        state_img_b64 = None
        _fib_state_bak = getattr(st, '_fib_state_backup', None)
        if st.fib_norm_mode and _fib_state_bak is not None:
            _state_src   = _fib_state_bak['img_srgb_orig']
            state_mu_lin   = _fib_state_bak['mu_lin'].tolist()
            state_W_lin    = _fib_state_bak['W_lin'].tolist()
            state_Winv_lin = _fib_state_bak['Winv_lin'].tolist()
        else:
            _state_src     = st.img_srgb_orig
            state_mu_lin   = st.mu_lin.tolist()   if st.mu_lin   is not None else None
            state_W_lin    = st.W_lin.tolist()    if st.W_lin    is not None else None
            state_Winv_lin = st.Winv_lin.tolist() if st.Winv_lin is not None else None
        if _state_src is not None and _state_src.size > 0:
            state_img_b64 = self._encode_img_png_b64(_state_src, "state_img")

        # Encode the truly immutable original view image — only if it differs from state_img
        _ov_src = getattr(st, 'img_original_view', None)
        if _ov_src is not None and _state_src is not None and np.array_equal(_ov_src, _state_src):
            original_view_img_b64 = None  # identical to state_img, skip
        else:
            original_view_img_b64 = self._encode_img_png_b64(_ov_src, "original_view_img")

        # Encode the stable ROI base image — only if it differs from state_img
        _rb_src = getattr(st, 'img_roi_base', None)
        if _rb_src is not None and _state_src is not None and np.array_equal(_rb_src, _state_src):
            roi_base_img_b64 = None  # identical to state_img, skip
        else:
            roi_base_img_b64 = self._encode_img_png_b64(_rb_src, "roi_base_img")

        # Encode Fibonacci image if active
        fib_img_b64 = None
        if st.fib_norm_mode and st._fib_img_cached is not None:
            fib_img_b64 = self._encode_img_png_b64(st._fib_img_cached, "fib_img")

        # Encode the raw original image (before any crop/BC) for session replay.
        # Prefer _img_pre_roi (set at load_image, never overwritten by reprocess)
        # over _img_file_initial (which reprocess() overwrites with the result).
        raw_img_b64 = None
        raw_src = getattr(st, '_raw_img_for_session', None)
        if raw_src is None and st.modify_params is None:
            _pre = getattr(st, '_img_pre_roi', None)
            raw_src = _pre if _pre is not None else st._img_file_initial
        if raw_src is not None:
            if _state_src is not None and np.array_equal(raw_src, _state_src):
                raw_img_b64 = None  # identical to state_img, skip
            else:
                raw_img_b64 = self._encode_img_png_b64(raw_src, "raw_img")

        # 3D model path (absolute) if in 3D mode
        is_3d_mode = self._stack is not None and self._stack.currentIndex() == 2
        model_path_abs = None
        if is_3d_mode and st.model_path:
            model_path_abs = os.path.abspath(st.model_path)

        # Generate thumbnail from the active view (3D model widget or 2D colour canvas).
        thumbnail_b64 = None
        try:
            if is_3d_mode and self.w_model is not None:
                frame_widget = self.w_model
                frame_widget.update()
                QtWidgets.QApplication.processEvents()
                frame = frame_widget.grabFramebuffer() if hasattr(frame_widget, 'grabFramebuffer') else None
            else:
                canvas_widget = self.w_color.canvas.native
                self.w_color.canvas.update()
                QtWidgets.QApplication.processEvents()
                frame = canvas_widget.grabFramebuffer() if hasattr(canvas_widget, 'grabFramebuffer') else None
            if frame is not None and not frame.isNull():
                # Scale to 384px on the longest side, keeping aspect ratio
                thumb = frame.scaled(384, 384,
                                     QtCore.Qt.KeepAspectRatio,
                                     QtCore.Qt.SmoothTransformation)
                buf = QtCore.QBuffer()
                buf.open(QtCore.QIODevice.WriteOnly)
                thumb.save(buf, 'JPEG', 95)
                thumbnail_b64 = base64.b64encode(
                    buf.data().data()).decode('ascii')
        except Exception as e:
            print(f"[RASCAL] Thumbnail generation failed: {e}")
            thumbnail_b64 = None

        image_path_saved = os.path.abspath(st.current_image_path) if st.current_image_path else ""

        data = {
            'version': 6,
            'thumbnail': thumbnail_b64,
            'image_path': image_path_saved,
            'quaternion': rot_q,
            'current_quaternion': current_q,
            'brush_alpha': brush_b64,
            'last_brush_alpha': last_brush_b64,
            'paint_mode': st.paint_mode,
            'push_factor': float(st.push_factor),
            'variance_restore': float(st.variance_restore),
            'contrast': float(st.contrast),
            'ui_level': self.ui_level,
            'fib_norm_mode': bool(st.fib_norm_mode),
            'fib_img': fib_img_b64,
            'raw_img': raw_img_b64,
            'modify_params': st.modify_params,
            'ica_active': bool(st.ica_active),
            'model_path': model_path_abs,
            'model_ui_level': self.model_ui_level if is_3d_mode else None,
            'state_img': state_img_b64,
            'state_mu_lin': state_mu_lin,
            'state_W_lin': state_W_lin,
            'state_Winv_lin': state_Winv_lin,
            'original_view_img': original_view_img_b64,
            'roi_base_img': roi_base_img_b64,
            'fib_samples_zca': self._encode_numpy_b64(getattr(st, 'samples_zca_fib', None), "fib_samples_zca"),
            'fib_zca_half': getattr(st, '_fib_zca_half', None),
        }

        if not path.endswith('.rasc'):
            path += '.rasc'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        self._session_dirty = False
        self._last_session_path = path
        QtCore.QSettings("Rascal", "rascal").setValue("last_session_dir", os.path.dirname(os.path.abspath(path)))

    def save_quaternion(self):
        """Saves the session: quaternion + brush mask + image path to a .rasc (JSON) file."""
        if self.state.paint_mode:
            self.statusBar().showMessage("Save not available in ROI mode — apply (Reprocess) or cancel first", 3000)
            return
        st = self.state
        if st.current_image_path:
            base_path, _ = os.path.splitext(st.current_image_path)
            default_path = base_path + ".rasc"
        elif getattr(self, '_last_session_path', None):
            default_path = self._last_session_path
        else:
            default_path = "untitled.rasc"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save session", default_path,
            "Rascal sessions (*.rasc);;All files (*)"
        )
        if not path:
            return
        try:
            self._save_session_to_path(path)
            self.statusBar().showMessage(f"Session saved: {os.path.basename(path)}", 4000)
            QtWidgets.QMessageBox.information(
                self, "Session Saved",
                f"Session saved successfully:\n{path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Save failed: {e}")

    def _browse_sessions(self):
        """Opens a custom dialog showing .rasc files with thumbnails.
        Returns the selected file path or None if cancelled."""
        # Determine initial directory: last session path > persisted QSettings dir
        init_dir = ""
        if self._last_session_path:
            init_dir = os.path.dirname(self._last_session_path)
        if not init_dir:
            init_dir = QtCore.QSettings("Rascal", "rascal").value("last_session_dir", "")

        dlg = _SessionBrowserDialog(init_dir, self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            return dlg.selected_path
        return None

    def _load_session_from_path(self, path):
        """Loads a .rasc session from the given path (no dialog)."""
        self._stop_autorotate()
        # Clear stale preset buttons from a previous image/session so the
        # numbered toolbar buttons stay in sync with state.presets (which is
        # reset by load_image / load_image_array below).
        self._clear_presets()
        if self.w_color.split_active:
            self.w_color.toggle_split(False)
            if self._split_btn is not None:
                self._split_btn.setChecked(False)
        try:
            _prog = QtWidgets.QProgressDialog(
                f"Reading {os.path.basename(path)}…", "", 0, 10, self)
            _prog.setWindowTitle("Loading session")
            _prog.setCancelButton(None)
            _prog.setMinimumDuration(0)
            _prog.setWindowModality(QtCore.Qt.ApplicationModal)
            _prog.setValue(0)
            _prog.show()
            QtWidgets.QApplication.processEvents()

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 1. Resolve image source: disk file > embedded raw_img > embedded work_img (legacy v3)
            img_path = data.get('image_path', '')
            if not os.path.isfile(img_path):
                img_path = os.path.join(os.path.dirname(path), os.path.basename(img_path))
            raw_img_b64  = data.get('raw_img')
            work_img_b64 = data.get('work_img')  # legacy v3 fallback
            loaded_from_embedded_raw = False
            if not os.path.isfile(img_path):
                fallback_b64 = raw_img_b64 or work_img_b64 or data.get('state_img')
                if not fallback_b64:
                    QtWidgets.QMessageBox.warning(self, "Session",
                        f"Image not found:\n{data.get('image_path', '')}\n"
                        "Please open the image manually.")
                    return
                # Decode embedded image as the base source.
                # raw_img_b64 is intentionally kept so the legacy replay path
                # (modify_params) can still be applied when state_img is absent.
                tmp_w = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tmp_w.write(base64.b64decode(fallback_b64))
                tmp_w.close()
                try:
                    _emb_raw = imageio.imread(tmp_w.name)
                finally:
                    try: os.unlink(tmp_w.name)
                    except OSError: pass
                _emb_f32 = _normalize_image_array(_emb_raw)
                _prog.setLabelText("Decoding embedded image…")
                _prog.setValue(1); QtWidgets.QApplication.processEvents()
                self.state.load_image_array(_emb_f32,
                                            pseudo_path=data.get('image_path', ''))
                loaded_from_embedded_raw = True
            else:
                _prog.setLabelText("Loading image from disk…")
                _prog.setValue(1); QtWidgets.QApplication.processEvents()
                self.state.load_image(img_path)
            self.reset_rotation()  # cleanly resets everything
            if getattr(self, '_welcome_mode', False):
                self._stack.setCurrentIndex(1)
                self._set_welcome_mode(False)

            # 1b. v6+ fast path: state_img is the exact saved working image — use it
            # directly, skipping all raw_img / modify_params / ROI replay.
            _state_img_b64_early = data.get('state_img')
            _state_img_applied = False
            if _state_img_b64_early and data.get('version', 0) >= 6:
                try:
                    _prog.setLabelText("Restoring working image…")
                    _prog.setValue(3); QtWidgets.QApplication.processEvents()
                    tmp_si = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    tmp_si.write(base64.b64decode(_state_img_b64_early))
                    tmp_si.close()
                    _si = _normalize_image_array(imageio.imread(tmp_si.name))
                    os.unlink(tmp_si.name)
                    s_mu   = data.get('state_mu_lin')
                    s_W    = data.get('state_W_lin')
                    s_Winv = data.get('state_Winv_lin')
                    if s_mu is not None and s_W is not None and s_Winv is not None:
                        self.state.mu_lin   = np.array(s_mu,   dtype=np.float32)
                        self.state.W_lin    = np.array(s_W,    dtype=np.float32)
                        self.state.Winv_lin = np.array(s_Winv, dtype=np.float32)
                    else:
                        Xl = ColorUtils.srgb_to_linear_np(
                            np.clip(_si, 0, 1)).reshape(-1, 3).astype(np.float32)
                        self.state.mu_lin, self.state.W_lin, self.state.Winv_lin = \
                            ColorUtils.zca_from_data(Xl)
                        del Xl
                    # _si is the working image (may be ROI-processed) — only assign to display vars
                    self.state.img_srgb_orig     = _si.copy()
                    self.state.img_srgb_view     = _si.copy()
                    self.state.img_srgb          = _si
                    self.state.modify_params     = data.get('modify_params')
                    # Restore truly immutable original view if saved explicitly
                    _ov_b64 = data.get('original_view_img')
                    if _ov_b64:
                        _prog.setLabelText("Restoring original view…")
                        _prog.setValue(5); QtWidgets.QApplication.processEvents()
                        try:
                            tmp_ov2 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            tmp_ov2.write(base64.b64decode(_ov_b64))
                            tmp_ov2.close()
                            _ov = _normalize_image_array(imageio.imread(tmp_ov2.name))
                            os.unlink(tmp_ov2.name)
                            self.state.img_original_view = _ov
                        except Exception as e:
                            print(f"[RASCAL] Failed to restore original_view_img: {e}")
                            self.state.img_original_view = _si.copy()
                    else:
                        # original_view was identical to state_img at save time
                        self.state.img_original_view = _si.copy()
                    # Restore stable ROI base from roi_base_img (set at load/modify, never by reprocess)
                    _rb_b64 = data.get('roi_base_img')
                    if _rb_b64:
                        _prog.setLabelText("Restoring ROI base…")
                        _prog.setValue(6); QtWidgets.QApplication.processEvents()
                        try:
                            tmp_rb2 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            tmp_rb2.write(base64.b64decode(_rb_b64))
                            tmp_rb2.close()
                            _rb = _normalize_image_array(imageio.imread(tmp_rb2.name))
                            os.unlink(tmp_rb2.name)
                            self.state.img_roi_base      = _rb.copy()
                            self.state._img_file_initial = _rb.copy()
                            self.state.img_file_orig     = _rb.copy()
                        except Exception as e:
                            print(f"[RASCAL] Failed to restore roi_base_img: {e}")
                            # Fall back: use the working image as base (safe default)
                            self.state.img_roi_base      = _si.copy()
                            self.state._img_file_initial = _si.copy()
                            self.state.img_file_orig     = _si.copy()
                    else:
                        # No roi_base_img saved (older session): use working image as base
                        self.state.img_roi_base      = _si.copy()
                        self.state._img_file_initial = _si.copy()
                        self.state.img_file_orig     = _si.copy()
                    # Restore _raw_img_for_session and _img_pre_roi from raw_img
                    # so that Modify Image can access the original pre-modify image.
                    if raw_img_b64 and data.get('modify_params'):
                        try:
                            tmp_raw = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            tmp_raw.write(base64.b64decode(raw_img_b64))
                            tmp_raw.close()
                            _raw = _normalize_image_array(imageio.imread(tmp_raw.name))
                            os.unlink(tmp_raw.name)
                            self.state._raw_img_for_session = _raw.copy()
                            self.state._img_pre_roi = _raw.copy()
                        except Exception as e:
                            print(f"[RASCAL] Failed to restore _raw_img_for_session: {e}")
                            self.state._img_pre_roi = self.state.img_roi_base.copy()
                    elif raw_img_b64 and not data.get('modify_params'):
                        # No modify but raw_img saved (unmodified session):
                        # decode raw_img as _img_pre_roi for ROI deactivation reset
                        try:
                            tmp_raw = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            tmp_raw.write(base64.b64decode(raw_img_b64))
                            tmp_raw.close()
                            _raw = _normalize_image_array(imageio.imread(tmp_raw.name))
                            os.unlink(tmp_raw.name)
                            self.state._img_pre_roi = _raw.copy()
                        except Exception as e:
                            print(f"[RASCAL] Failed to restore _img_pre_roi: {e}")
                            self.state._img_pre_roi = self.state.img_roi_base.copy()
                    else:
                        # No raw_img available: use roi_base as fallback
                        self.state._img_pre_roi = self.state.img_roi_base.copy()
                    img_lin = ColorUtils.srgb_to_linear_np(np.clip(_si, 0, 1))
                    Xl2 = img_lin.reshape(-1, 3).astype(np.float32)
                    rng = np.random.default_rng(42)
                    N = min(N_SAMPLES_3D, Xl2.shape[0])
                    self.state.samples_lin = Xl2[rng.choice(Xl2.shape[0], N, replace=False)]
                    for w in self.state.widgets_2d:
                        w.update_image_texture()
                    _state_img_applied = True
                except Exception as e:
                    print(f"[RASCAL] Failed to restore state_img, falling back to legacy replay: {e}")
                    _state_img_applied = False

            # 1b-legacy. No state_img (v5 and earlier): replay raw_img + modify_params.
            if not _state_img_applied and raw_img_b64:
                _prog.setLabelText("Replaying image modifications…")
                _prog.setValue(4); QtWidgets.QApplication.processEvents()
                tmp_r = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tmp_r.write(base64.b64decode(raw_img_b64))
                tmp_r.close()
                raw_img = _normalize_image_array(imageio.imread(tmp_r.name))
                os.unlink(tmp_r.name)
                # Replay modify_params (crop + brightness/contrast)
                mp = data.get('modify_params')
                if mp:
                    b = mp.get('brightness', 0) / 100.0
                    c = mp.get('contrast', 0) / 100.0
                    factor = np.float32(2.0 ** c)
                    work_img = np.clip((raw_img - 0.5) * factor + 0.5 + b, 0.0, 1.0)
                    rect = mp.get('crop_rect')
                    if rect is not None:
                        x, y, w_r, h_r = rect
                        H, W = work_img.shape[:2]
                        x0 = max(0, int(round(x * W)))
                        y0 = max(0, int(round(y * H)))
                        x1 = min(W, int(round((x + w_r) * W)))
                        y1 = min(H, int(round((y + h_r) * H)))
                        if (x1 - x0) >= 4 and (y1 - y0) >= 4:
                            work_img = work_img[y0:y1, x0:x1].copy()
                else:
                    work_img = raw_img
                # Apply as working image (keeps _img_file_initial = raw)
                self.state._img_file_initial = raw_img.copy()
                self.state._img_pre_roi = raw_img.copy()
                self.state.img_original_view = raw_img.copy()
                self.state.img_file_orig = work_img.copy()
                self.state.img_roi_base = work_img.copy()
                self.state.img_srgb_orig = work_img.copy()
                self.state.img_srgb_view = work_img.copy()
                self.state.img_srgb = work_img
                self.state.modify_params = mp
                if mp:
                    self.state._raw_img_for_session = raw_img.copy()
                Xl = ColorUtils.srgb_to_linear_np(np.clip(work_img, 0, 1)).reshape(-1, 3)
                self.state.mu_lin, self.state.W_lin, self.state.Winv_lin = ColorUtils.zca_from_data(Xl)
                del Xl; gc.collect()
                for w in self.state.widgets_2d:
                    w.update_image_texture()

            self._update_original_thumbnail()

            _prog.setLabelText("Restoring scene…")
            _prog.setValue(7); QtWidgets.QApplication.processEvents()

            # 2a. Restore 3D model if session was saved in 3D mode
            _model_path = data.get('model_path')
            if _model_path and os.path.isfile(_model_path):
                mesh = self.state.load_model(_model_path)
                self.w_model.set_mesh(mesh)
                self.w_model.reset_camera()
                for w in self.state.widgets_2d:
                    w.update_image_texture()
                self.w_model_tex.update_image_texture()
                self.w_model.update_texture_from_state()
                self._stack.setCurrentIndex(2)
                self._save_3d_action.setVisible(True)
                # Restore model UI level
                m_lvl = data.get('model_ui_level', 'basic')
                self._apply_model_ui_level(m_lvl)
                if self.model_level_combo is not None:
                    idx_m = ['basic', 'expert'].index(m_lvl) if m_lvl in ['basic', 'expert'] else 0
                    self.model_level_combo.blockSignals(True)
                    self.model_level_combo.setCurrentIndex(idx_m)
                    self.model_level_combo.blockSignals(False)
                QtCore.QTimer.singleShot(0, self._equalize_3d_views)
            else:
                # 2D session: ensure we are on the 2D page and clear any leftover 3D model
                self.state.mesh_data = None
                self.state.model_path = None
                self._stack.setCurrentIndex(1)
                if self._save_3d_action is not None:
                    self._save_3d_action.setVisible(False)

            # 2b. Restore push_factor, variance_restore, contrast and ui_level
            push = data.get('push_factor', 1.0)
            self._set_push(push, mark_dirty=False)
            self._set_var_restore(data.get('variance_restore', 1.0), mark_dirty=False)
            self._set_contrast(data.get('contrast', 1.0), mark_dirty=False)
            ui_lvl = data.get('ui_level', 'basic')
            self._apply_ui_level(ui_lvl)
            idx_lvl = ['basic', 'expert'].index(ui_lvl) if ui_lvl in ['basic', 'expert'] else 0
            self.ui_level_combo.blockSignals(True)
            self.ui_level_combo.setCurrentIndex(idx_lvl)
            self.ui_level_combo.blockSignals(False)
            # Re-apply model UI level if in 3D mode (ui_level may have overridden it)
            if _model_path and os.path.isfile(_model_path):
                m_lvl = data.get('model_ui_level', 'basic')
                self._apply_model_ui_level(m_lvl)

            # 2c. Decode Fibonacci image if available (activation deferred after reprocess)
            fib_img_b64 = data.get('fib_img')
            _restore_fib = False
            if fib_img_b64 and data.get('fib_norm_mode', False):
                _prog.setLabelText("Restoring Fibonacci state…")
                _prog.setValue(8); QtWidgets.QApplication.processEvents()
                tmp3 = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tmp3.write(base64.b64decode(fib_img_b64))
                tmp3.close()
                _fib_img_decoded = _normalize_image_array(imageio.imread(tmp3.name))
                os.unlink(tmp3.name)
                _restore_fib = True
                # Restore lightweight 3D decorrelated coordinates
                fsz = self._decode_numpy_b64(data.get('fib_samples_zca'), "fib_samples_zca")
                if fsz is not None:
                    self.state.samples_zca_fib = fsz
                fzh = data.get('fib_zca_half')
                if fzh is not None:
                    self.state._fib_zca_half = float(fzh)

            # 2. Restore brush mask (last_brush_alpha = mask before re-process)
            _prog.setLabelText("Restoring brush mask…")
            _prog.setValue(9); QtWidgets.QApplication.processEvents()
            last_brush_b64 = data.get('last_brush_alpha')
            brush = None
            if last_brush_b64:
                tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                tmp.write(base64.b64decode(last_brush_b64))
                tmp.close()
                mask_u8 = imageio.imread(tmp.name)
                os.unlink(tmp.name)
                if mask_u8.ndim == 3:
                    mask_u8 = mask_u8[..., 0]
                mask_f = mask_u8.astype(np.float32) / 255.0
                H, W = self.state.img_file_orig.shape[:2]
                if mask_f.shape[:2] == (H, W):
                    brush = mask_f.reshape(H, W, 1)
                    self.state.brush_alpha = brush.copy()
                    self.state.last_brush_alpha = brush.copy()
                    for w in self.state.widgets_2d:
                        w.brush_alpha = self.state.brush_alpha
                        if hasattr(w, 'brush_tex'):
                            w.brush_tex.set_data(self.state.brush_alpha.astype(np.float32))
                    n_painted = int((brush[..., 0] > 0.5).sum())
                    self.statusBar().showMessage(
                        f"Mask restored: {n_painted} painted pixels — re-processing...", 3000
                    )
                    QtWidgets.QApplication.processEvents()
                # else: mask size mismatch — silently ignored, session loads without ROI

            if brush is not None:
                # v6+: state_img was already applied above (or ROI image already in place).
                # For ROI sessions, also mark last_img_srgb_reprocessed for consistency.
                _used_state_img = _state_img_applied

                if not _state_img_applied:
                    # --- Legacy path (v5 and earlier): replay the ROI transform ---
                    q = np.array(data['quaternion'], dtype=np.float32)
                    if q.size != 4:
                        raise ValueError("Invalid quaternion")
                    self.state.rot.Ruser = RotationState.quat_to_R(q)
                    self.state.rot.ang_x, self.state.rot.ang_y, self.state.rot.ang_z = \
                        RotationState.R_to_euler_zyx(self.state.rot.Ruser)

                    self.state.ica_active = False
                    if self.ica_btn is not None:
                        self.ica_btn.blockSignals(True)
                        self.ica_btn.setChecked(False)
                        self.ica_btn.blockSignals(False)

                    self.reprocess()
                else:
                    # state_img already applied — mark reprocessed reference
                    self.state.last_img_srgb_reprocessed = self.state.img_srgb_orig.copy()
                    self._update_original_thumbnail()

                # 5. Restore the CURRENT rotation
                cur_q_data = data.get('current_quaternion', data['quaternion'])
                cur_q = np.array(cur_q_data, dtype=np.float32)
                self.state.rot.Ruser = RotationState.quat_to_R(cur_q)
                self.state.rot.ang_x, self.state.rot.ang_y, self.state.rot.ang_z = \
                    RotationState.R_to_euler_zyx(self.state.rot.Ruser)

                # 5b. Restore ICA state
                ica_was_active = data.get('ica_active', False)
                self.state.ica_active = ica_was_active
                if self.ica_btn is not None:
                    self.ica_btn.blockSignals(True)
                    self.ica_btn.setChecked(ica_was_active)
                    self.ica_btn.blockSignals(False)

                # 6. Save stroke so paint mode can be restored later
                self.state.last_brush_alpha = brush.copy()
                self.state.last_img_srgb_orig = self.state.img_file_orig.copy()

                # 7. Restore Fibonacci / Radial Push after reprocess
                if _restore_fib:
                    self._restore_fib_state_from_session(_fib_img_decoded, True)

                # 8. Refresh views with restored rotation
                for w in self.state.widgets_2d:
                    w.update_image_texture(); w.canvas.update()
                if self.state.widget_3d:
                    self.state.widget_3d.update_markers(); self.state.widget_3d.canvas.update()
                self.info_panel.refresh()

                self._session_dirty = False
                self._update_status_bar()
                _prog.setValue(10); _prog.close()
                _load_msg = "Session loaded" if _used_state_img else "Session loaded and re-processed"
                self.statusBar().showMessage(
                    f"{_load_msg}: {os.path.basename(path)}", 5000
                )
            else:
                # No mask (or incompatible mask): restore the current rotation
                cur_q_data = data.get('current_quaternion', data.get('quaternion'))
                q = np.array(cur_q_data, dtype=np.float32)
                self.state.rot.Ruser = RotationState.quat_to_R(q)
                self.state.rot.ang_x, self.state.rot.ang_y, self.state.rot.ang_z = \
                    RotationState.R_to_euler_zyx(self.state.rot.Ruser)

                # Restore ICA state
                ica_was_active = data.get('ica_active', False)
                self.state.ica_active = ica_was_active
                if self.ica_btn is not None:
                    self.ica_btn.blockSignals(True)
                    self.ica_btn.setChecked(ica_was_active)
                    self.ica_btn.blockSignals(False)

                # Restore Fibonacci / Radial Push (no brush case)
                if _restore_fib:
                    self._restore_fib_state_from_session(_fib_img_decoded, True)

                # Restore active brush (painted but not yet reprocessed)
                active_brush_b64 = data.get('brush_alpha')
                if active_brush_b64:
                    try:
                        tmp_ab = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        tmp_ab.write(base64.b64decode(active_brush_b64))
                        tmp_ab.close()
                        ab_u8 = imageio.imread(tmp_ab.name)
                        os.unlink(tmp_ab.name)
                        if ab_u8.ndim == 3:
                            ab_u8 = ab_u8[..., 0]
                        ab_f = ab_u8.astype(np.float32) / 255.0
                        H_ab, W_ab = self.state.img_file_orig.shape[:2]
                        if ab_f.shape[:2] == (H_ab, W_ab):
                            active_brush = ab_f.reshape(H_ab, W_ab, 1)
                            self.state.brush_alpha = active_brush.copy()
                            for w in self.state.widgets_2d:
                                w.brush_alpha = self.state.brush_alpha
                                if hasattr(w, 'brush_tex'):
                                    w.brush_tex.set_data(self.state.brush_alpha.astype(np.float32))
                    except Exception as e:
                        print(f"[RASCAL] Failed to restore active brush_alpha: {e}")

                # Restore paint_mode state and update UI button
                self.state.paint_mode = data.get('paint_mode', False)
                if self.paint_mode_btn is not None:
                    self.paint_mode_btn.setChecked(self.state.paint_mode)
                self._update_brush_panels_visibility()

                self.state.refresh_all()
                self.info_panel.refresh()
                self._session_dirty = False
                self._update_status_bar()
                _prog.setValue(10); _prog.close()
                self.statusBar().showMessage(f"Session loaded: {os.path.basename(path)}", 5000)

        except Exception as e:
            try: _prog.close()
            except Exception: pass
            QtWidgets.QMessageBox.critical(self, "Error", f"Load failed: {e}")
            self.info_panel.refresh()

    def load_quaternion(self):
        """Loads a .rasc session via the thumbnail browser dialog."""
        if not self._prompt_save_if_dirty():
            return
        path = self._browse_sessions()
        if not path:
            return
        self._load_session_from_path(path)
        QtCore.QSettings("Rascal", "rascal").setValue("last_session_dir", os.path.dirname(os.path.abspath(path)))

    # ---------- JPG export ----------
    @staticmethod
    def _lut_apply(lut_1d, img_f32):
        """Apply a 1-D LUT (length N, float32) to an (H,W,3) float32 image.
        Replicates the shader's lut_lookup_rgb: linear interpolation, clamped to [0,1].
        This matches the GPU shader behaviour exactly (same 256-entry LUT, same interpolation).
        """
        n = len(lut_1d)
        x = np.clip(img_f32, 0.0, 1.0) * (n - 1)
        lo = np.floor(x).astype(np.int32)
        hi = np.minimum(lo + 1, n - 1)
        frac = (x - lo).astype(np.float32)
        return lut_1d[lo] * (1.0 - frac) + lut_1d[hi] * frac

    def _render_fullres_image(self, mode):
        """Replicate the 2D shader on the CPU at full image resolution.

        mode 0: colour (RGB after ZCA transform)
        mode 1: R channel (greyscale)
        mode 2: G channel (greyscale)
        mode 3: B channel (greyscale)
        mode 4: original file image (no transform)

        Returns an (H, W, 3) uint8 array.
        Uses the same LUT-based gamma and float32 precision as the GPU shader
        so the exported image is pixel-accurate to what is displayed on screen.
        """
        st = self.state
        if mode == 4:
            return np.clip(st.img_original_view * 255.0, 0, 255).astype(np.uint8)

        # Build the same 1-D LUTs used by the shader (shape: N float32)
        lut_s2l = ColorUtils.make_lut_srgb_to_linear(LUT_SIZE)  # sRGB → linear
        lut_l2s = ColorUtils.make_lut_linear_to_srgb(LUT_SIZE)  # linear → sRGB

        # 1. sRGB → linear  (via LUT, identical to shader's lut_lookup_rgb)
        xs = np.clip(st.img_srgb_orig, 0.0, 1.0).astype(np.float32)
        H, W, _ = xs.shape
        xl0 = self._lut_apply(lut_s2l, xs)

        # 2. ZCA forward: z = W * (xl0 - mu)  — float32, matching GPU mediump precision
        flat = xl0.reshape(-1, 3)
        mu     = st.mu_lin.astype(np.float32)
        W_lin  = st.W_lin.astype(np.float32)
        Winv   = st.Winv_lin.astype(np.float32)
        Ruser  = st.rot.Ruser.astype(np.float32)
        push   = np.float32(st.push_factor)
        var_r  = np.float32(st.variance_restore)

        z = (flat - mu) @ W_lin.T

        # 3. Rotation + push
        # vispy uploads Ruser as column-major → GLSL sees Ruser_numpy.T.
        # GLSL: z2 = Ruser_glsl * z = Ruser_numpy.T @ z_i
        # CPU row-major equivalent: z @ Ruser  (not z @ Ruser.T)
        z2 = z @ Ruser * push

        # 4. Effective inverse: eff_Winv = I + var_restore * (Winv - I)
        I3 = np.eye(3, dtype=np.float32)
        eff_Winv = I3 + var_r * (Winv - I3)

        # 5. Reconstruct linear
        xo = np.clip(mu + z2 @ eff_Winv.T, 0.0, 1.0).astype(np.float32).reshape(H, W, 3)

        # 6. linear → sRGB  (via LUT, identical to shader's lut_lookup_rgb)
        color = self._lut_apply(lut_l2s, xo)

        # 7. Manual contrast adjustment (must match GPU shader:
        #    base_color = clamp((base_color - u_contrast_center) * u_contrast + u_contrast_center, 0.0, 1.0))
        color = np.clip((color - np.float32(st.contrast_center)) * np.float32(st.contrast) + np.float32(st.contrast_center), 0.0, 1.0)

        # 8. Channel extraction for R/G/B modes
        if mode == 1:
            grey = color[..., 0]
            color = np.stack([grey, grey, grey], axis=-1)
        elif mode == 2:
            grey = color[..., 1]
            color = np.stack([grey, grey, grey], axis=-1)
        elif mode == 3:
            grey = color[..., 2]
            color = np.stack([grey, grey, grey], axis=-1)

        return np.clip(color * 255.0, 0, 255).astype(np.uint8)

    def export_selected_views_to_jpg(self):
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose output folder", ""
        )
        if not out_dir:
            return

        base, ok = QtWidgets.QInputDialog.getText(
            self, "File prefix", "Prefix (leave blank for 'export'):",
            text="export"
        )
        if not ok:
            return
        base = base.strip() or "export"

        orig_name = os.path.splitext(os.path.basename(self.state.current_image_path))[0]
        ts = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")
        stem = f"{base}_{ts}_{orig_name}"

        # 3D page: export only one RGB output (texture view)
        if self._stack.currentIndex() == 2:
            try:
                rgb = self._render_fullres_image(0)
                fname = f"{stem}_rgb.jpg"
                fpath = os.path.join(out_dir, fname)
                imageio.imwrite(pathlib.Path(fpath), rgb, quality=95)
                # Auto-save session
                try:
                    self._save_session_to_path(os.path.join(out_dir, f"{stem}.rasc"))
                except Exception as e:
                    print(f"[RASCAL] Auto-save session failed: {e}")
                    QtWidgets.QMessageBox.warning(
                        self, "Session not saved",
                        f"Image exported to:\n{fpath}\n\n"
                        f"But the session could not be saved:\n{e}"
                    )
                self.statusBar().showMessage("JPG export done (1 file).", 4000)
                QtWidgets.QMessageBox.information(self, "Export", "Export complete: 1 file written.")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export", f"Export failed for RGB: {e}")
            return

        # Canonical export definitions — always use fixed mode numbers regardless of swap_views() state
        canonical_exports = [
            ("color", 0),
            ("R",     1),
            ("G",     2),
            ("B",     3),
        ]
        export_3d = self.export_flags.get('3d', False)
        export_original = self.export_flags.get('original', False)

        has_channel = any(self.export_flags.get(s, False) for s, _ in canonical_exports)
        if not has_channel and not export_3d and not export_original:
            QtWidgets.QMessageBox.information(self, "Export", "No view selected for export.")
            return

        count = 0

        # Export original image at full resolution
        if export_original:
            try:
                orig = self.state.img_original_view
                orig_u8 = np.clip(orig * 255, 0, 255).astype(np.uint8)
                fname = f"{stem}_original.jpg"
                fpath = os.path.join(out_dir, fname)
                # Try to preserve the ICC profile from the source file so that
                # colour-managed viewers show the same colours as the source.
                _icc = None
                _src = getattr(self.state, 'current_image_path', None)
                if _src and os.path.isfile(_src):
                    try:
                        from PIL import Image as _PILImage
                        with _PILImage.open(_src) as _pil:
                            _icc = _pil.info.get('icc_profile')
                    except Exception:
                        pass
                if _icc:
                    from PIL import Image as _PILImage
                    _pil_out = _PILImage.fromarray(orig_u8)
                    _pil_out.save(fpath, format='JPEG', quality=95, icc_profile=_icc)
                else:
                    imageio.imwrite(pathlib.Path(fpath), orig_u8, quality=95)
                count += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export", f"Export failed for Original: {e}")

        for suffix, mode in canonical_exports:
            if not self.export_flags.get(suffix, False):
                continue
            try:
                rgb = self._render_fullres_image(mode)
                fname = f"{stem}_{suffix}.jpg"
                fpath = os.path.join(out_dir, fname)
                imageio.imwrite(pathlib.Path(fpath), rgb, quality=95)
                count += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export",
                    f"Export failed for {suffix}: {e}")

        if export_3d:
            try:
                self.w_3d.canvas.update()
                QtWidgets.QApplication.processEvents()
                frame = self.w_3d.canvas.render(alpha=False)
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[..., :3]
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                fname = f"{stem}_3d.jpg"
                fpath = os.path.join(out_dir, fname)
                imageio.imwrite(pathlib.Path(fpath), frame, quality=95)
                count += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export", f"Export failed for 3D: {e}")

        # Auto-save session alongside the exported images
        try:
            self._save_session_to_path(os.path.join(out_dir, f"{stem}.rasc"))
        except Exception as e:
            print(f"[RASCAL] Auto-save session failed: {e}")
            QtWidgets.QMessageBox.warning(
                self, "Session not saved",
                f"Image(s) exported to:\n{out_dir}\n\n"
                f"But the session could not be saved:\n{e}"
            )

        self.statusBar().showMessage(f"JPG export done ({count} file(s)).", 4000)
        QtWidgets.QMessageBox.information(self, "Export", f"Export complete: {count} file(s) written.")

    def _refresh_memory_label(self):
        """Update the permanent memory indicator in the status bar."""
        proc_mb = _get_process_memory_mb()
        if proc_mb is None:
            self._mem_label.setText("RAM: N/A")
            self._mem_label.setToolTip("Install psutil for memory monitoring")
            return
        pct = _get_memory_percent()
        total_mb = _get_total_ram_mb()
        text = f"RAM: {_format_memory_mb(proc_mb)}"
        if pct is not None:
            text += f" ({pct:.0f}%)"
        self._mem_label.setText(text)
        tooltip = f"Process: {_format_memory_mb(proc_mb)}"
        if total_mb:
            tooltip += f" / System: {_format_memory_mb(total_mb)}"
        self._mem_label.setToolTip(tooltip)
        # Colour-code based on usage
        if pct is not None and pct > _MEM_HIGH_PERCENT:
            self._mem_label.setStyleSheet(
                "QLabel { padding: 2px 6px; font-size: 11px; color: #ff4444; font-weight: bold; }"
            )
        elif pct is not None and pct > 60:
            self._mem_label.setStyleSheet(
                "QLabel { padding: 2px 6px; font-size: 11px; color: #ddaa00; }"
            )
        else:
            self._mem_label.setStyleSheet(
                "QLabel { padding: 2px 6px; font-size: 11px; color: #888; }"
            )

    def _check_memory_before_heavy_op(self, op_name="Operation"):
        """Check memory usage before a heavy operation. Returns True if OK to proceed."""
        pct = _get_memory_percent()
        if pct is not None and pct > _MEM_HIGH_PERCENT:
            proc_mb = _get_process_memory_mb()
            total_mb = _get_total_ram_mb()
            reply = QtWidgets.QMessageBox.warning(
                self, f"High memory usage",
                f"Rascal is using {_format_memory_mb(proc_mb)} "
                f"({pct:.0f}% of {_format_memory_mb(total_mb)} RAM).\n\n"
                f"Proceeding with '{op_name}' may cause slowdowns or crashes.\n\n"
                "Continue anyway?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            return reply == QtWidgets.QMessageBox.Yes
        return True

    def _update_status_bar(self):
        path = self.state.current_image_path
        if not path:
            self.status_label_left.setText("")
            self.status_label.setText("")
            return
        is_3d = self._stack is not None and self._stack.currentIndex() == 2
        mode_tag = "3D" if is_3d else "2D"
        if is_3d and self.state.model_path:
            display_path = self.state.model_path
        else:
            display_path = path
        filename = os.path.basename(display_path)
        if os.path.isfile(display_path):
            directory = os.path.dirname(os.path.abspath(display_path))
            self.status_label_left.setText(f"[{mode_tag}]  Directory: {directory}  |  File: {filename}")
        else:
            self.status_label_left.setText(f"[{mode_tag}]  File: {filename}  (embedded)")
        # Image dimensions and file size
        H, W = self.state.img_file_orig.shape[:2]
        try:
            size_bytes = os.path.getsize(os.path.abspath(path))
            if size_bytes >= 1_048_576:
                size_str = f"{size_bytes / 1_048_576:.1f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.0f} KB"
            else:
                size_str = f"{size_bytes} B"
        except OSError:
            size_str = "?"
        # In 3D mode, prepend mesh vertex/face counts
        mesh_info = ""
        if self._stack is not None and self._stack.currentIndex() == 2:
            mesh = self.state.mesh_data
            if mesh is not None and mesh.is_valid:
                n_verts = len(mesh.vertices)
                n_faces = len(mesh.indices) // 3
                mesh_info = f"{n_verts:,} vertices  |  {n_faces:,} faces  |  "
        self.status_label.setText(f"{mesh_info}{W} × {H} px  |  {size_str}")
    
    def _on_split_btn_toggled(self, checked):
        """Handle the Split toolbar button click."""
        self._toggle_split_view()

    def _toggle_split_view(self):
        """Toggle the before/after split view on the main colour widget."""
        if self._stack is not None and self._stack.currentIndex() == 0:
            return  # no image loaded (welcome page)
        if self._stack is not None and self._stack.currentIndex() == 2:
            return  # not available in 3D mode
        self.w_color.toggle_split()
        active = self.w_color.split_active
        # Keep button in sync
        if self._split_btn is not None:
            self._split_btn.blockSignals(True)
            self._split_btn.setChecked(active)
            self._split_btn.blockSignals(False)
        if active:
            self.statusBar().showMessage("Split view ON – drag the line to compare (T to toggle)", 4000)
        else:
            self.statusBar().showMessage("Split view OFF", 2000)

    def _toggle_original_swap(self):
        """Toggle main view between current mode and Original (mode 4)."""
        if self._stack.currentIndex() != 1:
            return
        w = self.w_color
        if self._orig_swap_prev_mode is None:
            # Switch to Original
            self._orig_swap_prev_mode = w.mode
            w.mode = 4
            w.prog['u_mode'] = 4
            self.color_box.setTitle(self.mode_titles[4])
        else:
            # Restore previous mode
            w.mode = self._orig_swap_prev_mode
            w.prog['u_mode'] = int(w.mode)
            self.color_box.setTitle(self.mode_titles[w.mode])
            self._orig_swap_prev_mode = None
        w.canvas.update()

    def swap_views(self, widget1, widget2, box1, box2):
        """Swaps the colour view with a single R/G/B channel.

        Only colour (mode 0) ↔ channel swaps are allowed.
        If colour is already swapped with a different channel, that swap is
        undone first.  Clicking the same channel again restores the original
        layout (toggle behaviour).
        """
        # Cancel any active original-image swap first
        if self._orig_swap_prev_mode is not None:
            self.w_color.mode = self._orig_swap_prev_mode
            self.w_color.prog['u_mode'] = int(self.w_color.mode)
            self._orig_swap_prev_mode = None

        # _swapped_channel tracks which (widget, box) is currently swapped
        # with the colour view.  None means no swap is active.
        prev = self._swapped_channel

        if prev is not None:
            # A swap is active → always restore the original layout first
            prev_w, prev_box = prev
            m_main, m_prev = self.w_color.mode, prev_w.mode
            self.w_color.mode = m_prev
            prev_w.mode = m_main
            self.w_color.prog['u_mode'] = int(self.w_color.mode)
            prev_w.prog['u_mode'] = int(prev_w.mode)
            self.color_box.setTitle(self.mode_titles[self.w_color.mode])
            prev_box.setTitle(self.mode_titles[prev_w.mode])
            prev_w.canvas.update()
            self.w_color.canvas.update()
            self._swapped_channel = None
            self.statusBar().showMessage("Views restored", 2000)
            return

        # No swap active → perform colour ↔ requested channel
        mode1, mode2 = self.w_color.mode, widget2.mode
        self.w_color.mode = mode2
        widget2.mode = mode1
        self.w_color.prog['u_mode'] = int(self.w_color.mode)
        widget2.prog['u_mode'] = int(widget2.mode)
        self.color_box.setTitle(self.mode_titles[self.w_color.mode])
        box2.setTitle(self.mode_titles[widget2.mode])
        self.w_color.canvas.update()
        widget2.canvas.update()
        self._swapped_channel = (widget2, box2)

        self.statusBar().showMessage(f"Views swapped: {self.mode_titles[mode1]} ↔ {self.mode_titles[mode2]}", 2000)
    
    def show_about(self):
        """Shows the About dialog."""
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("About Rascal")
        dlg.setTextFormat(QtCore.Qt.RichText)
        dlg.setText(
            "Rascal — Colour visualisation tool<br><br>"
            "Version 1.8.8<br>"
            "Contact: Fabrice.Monna@ube.fr<br><br>"
            "© 2026 - Fabrice Monna - All rights reserved"
        )
        dlg.exec_()

# ------------ Main ------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # On Linux, default multiprocessing start method is "fork" which can
    # deadlock when combined with Qt / OpenGL.  Force "spawn" everywhere
    # for consistency (Windows and macOS already default to "spawn").
    if IS_LINUX:
        multiprocessing.set_start_method("spawn", force=True)
    # On Linux under Wayland, Qt + OpenGL can misbehave; force the XCB
    # (X11) platform plugin unless the user explicitly set something else.
    if IS_LINUX and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    if IS_MACOS:
        # Layer-backed views are required for correct GL compositing on macOS;
        # without this, the framebuffer may show mirrored/garbage content.
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    # Allow GL contexts to share textures across vispy canvases and QOpenGLWidget.
    # On macOS this can fail with NSOpenGL shared contexts, so keep contexts unshared.
    if not IS_MACOS:
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app.use_app('pyqt5')
    fmt = QtGui.QSurfaceFormat()
    if IS_MACOS:
        fmt.setVersion(2, 1)
    else:
        fmt.setVersion(3, 2)
    if IS_LINUX:
        # Mesa / Intel / AMD drivers often report Core Profile as
        # unsupported while Compatibility Profile works fine at 3.2+.
        fmt.setProfile(QtGui.QSurfaceFormat.CompatibilityProfile)
    elif IS_MACOS:
        fmt.setProfile(QtGui.QSurfaceFormat.NoProfile)
    else:
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    qapp = QtWidgets.QApplication(sys.argv)
    _dpr = qapp.primaryScreen().devicePixelRatio()
    if _dpr > 1.25:
        _f = qapp.font()
        _f.setPointSizeF(min(_f.pointSizeF(), 8.0))
        qapp.setFont(_f)
    w = MainWindow(); w.show()
    ControlsHintDialog.show_if_needed(w)
    sys.exit(qapp.exec_())
