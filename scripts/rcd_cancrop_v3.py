import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None


# ===================== color table / config =====================

@dataclass
class BandColor:
    name: str
    digit: Optional[int] = None
    multiplier: Optional[float] = None
    tolerance: Optional[str] = None
    debug_bgr: Tuple[int, int, int] = (255, 255, 255)


COLOR_DB: Dict[str, BandColor] = {
    "BLACK": BandColor("BLACK", 0, 1, None, (0, 0, 0)),
    "BROWN": BandColor("BROWN", 1, 10, "±1%", (19, 69, 139)),
    "RED": BandColor("RED", 2, 100, "±2%", (0, 0, 255)),
    "ORANGE": BandColor("ORANGE", 3, 1_000, None, (0, 140, 255)),
    "YELLOW": BandColor("YELLOW", 4, 10_000, None, (0, 255, 255)),
    "GREEN": BandColor("GREEN", 5, 100_000, "±0.5%", (0, 255, 0)),
    "BLUE": BandColor("BLUE", 6, 1_000_000, "±0.25%", (255, 0, 0)),
    "VIOLET": BandColor("VIOLET", 7, 10_000_000, "±0.1%", (255, 0, 180)),
    "GRAY": BandColor("GRAY", 8, 100_000_000, "±0.05%", (128, 128, 128)),
    "WHITE": BandColor("WHITE", 9, 1_000_000_000, None, (255, 255, 255)),
    "GOLD": BandColor("GOLD", None, 0.1, "±5%", (0, 215, 255)),
    "SILVER": BandColor("SILVER", None, 0.01, "±10%", (192, 192, 192)),
}

TOLERANCE_COLORS = {"GOLD", "SILVER"}
CONFIRMED_D_COLORS = {"RED", "ORANGE", "YELLOW", "VIOLET", "BLUE", "GREEN"}
FONT = cv.FONT_HERSHEY_SIMPLEX

FIXED_CROP: Optional[Tuple[int, int, int, int]] = None
USE_FIXED_CROP = False
#要不要背景差分
USE_BACKGROUND_SUBTRACTION = False
BACKGROUND_IMAGE_PATH = "bg.jpg"
BACKGROUND_IMAGE_ALIASES = ["bg.jpg"]

TARGET_BODY_WIDTH = 700
BODY_EDGE_TRIM_RATIO = 0.018
MIN_BODY_WIDTH_RATIO = 0.24


# ===================== basic utilities =====================

def fixed_crop_image(img: np.ndarray, crop: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if crop is None:
        return img
    x1, y1, x2, y2 = crop
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    return img[y1:y2, x1:x2].copy()


def contiguous_segments(indices: np.ndarray) -> List[Tuple[int, int]]:
    if indices is None or len(indices) == 0:
        return []
    segments = []
    start = prev = int(indices[0])
    for val in indices[1:]:
        val = int(val)
        if val == prev + 1:
            prev = val
        else:
            segments.append((start, prev))
            start = prev = val
    segments.append((start, prev))
    return segments


def merge_close_segments(segments: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged = [[segments[0][0], segments[0][1]]]
    for a, b in segments[1:]:
        if a - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def circular_hue_median(hues: np.ndarray) -> float:
    if len(hues) == 0:
        return 0.0
    hues = hues.astype(np.float32)
    if np.mean((hues <= 8) | (hues >= 172)) > 0.25:
        adjusted = hues.copy()
        adjusted[adjusted >= 172] -= 180
        med = float(np.median(adjusted))
        return med + 180 if med < 0 else med
    return float(np.median(hues))


def local_find_peaks(signal: np.ndarray, prominence: float, distance: int):
    if find_peaks is not None:
        return find_peaks(signal, prominence=prominence, distance=distance)
    peaks = []
    for i in range(1, len(signal) - 1):
        if signal[i] >= signal[i - 1] and signal[i] > signal[i + 1] and signal[i] >= prominence:
            peaks.append(i)
    selected = []
    prominences = []
    for idx in sorted(peaks, key=lambda p: signal[p], reverse=True):
        if all(abs(idx - p) >= distance for p in selected):
            selected.append(idx)
            prominences.append(signal[idx])
    order = np.argsort(selected)
    return np.array(selected)[order], {"prominences": np.array(prominences)[order]}


def format_resistance(value_ohm: float) -> str:
    if value_ohm >= 1_000_000:
        return f"{value_ohm / 1_000_000:.3g} MΩ"
    if value_ohm >= 1_000:
        return f"{value_ohm / 1_000:.3g} kΩ"
    return f"{value_ohm:.3g} Ω"


def format_resistance_ascii(value_ohm: float) -> str:
    if value_ohm >= 1_000_000:
        return f"{value_ohm / 1_000_000:.3g} MOhm"
    if value_ohm >= 1_000:
        return f"{value_ohm / 1_000:.3g} kOhm"
    return f"{value_ohm:.3g} Ohm"


# ===================== body crop =====================

def build_body_likelihood_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV).astype(np.float32)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)

    border_size = max(3, int(0.05 * min(h, w)))
    border = np.zeros((h, w), np.uint8)
    border[:border_size, :] = 1
    border[-border_size:, :] = 1
    border[:, :border_size] = 1
    border[:, -border_size:] = 1

    bg_lab = np.median(lab[border > 0], axis=0)
    bg_gray = float(np.median(gray[border > 0]))

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    dist = np.linalg.norm(lab - bg_lab, axis=2)

    color_or_dark = (
        ((dist > 12) & (V < 252)) |
        ((S > 20) & (V < 252)) |
        (gray < bg_gray - 18)
    )
    warm_body = (
        (B > bg_lab[2] + 3.0) &
        (S > 6) &
        (V > 65) &
        (V < 252)
    )
    soft_shadow_body = (
        (gray < bg_gray - 9) &
        (B > bg_lab[2] + 1.2) &
        (V > 58) &
        (V < 248)
    )

    mask = (color_or_dark | warm_body | soft_shadow_body).astype(np.uint8) * 255
    mask[: int(0.04 * h), :] = 0
    mask[int(0.96 * h):, :] = 0
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    close_kernel = np.ones((max(3, h // 32), max(15, w // 18)), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=1)
    return mask


def column_thickness_profile(mask: np.ndarray, y1: int, y2: int) -> np.ndarray:
    h, w = mask.shape[:2]
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    mid = mask[y1:y2, :]
    profile = np.zeros(w, dtype=np.float32)
    for x in range(w):
        ys = np.where(mid[:, x] > 0)[0]
        if len(ys) >= 2:
            profile[x] = ys[-1] - ys[0] + 1
    return cv.GaussianBlur(profile.reshape(1, -1), (0, 0), sigmaX=max(2, w / 120)).reshape(-1)


def column_coverage_profile(mask: np.ndarray, y1: int, y2: int) -> np.ndarray:
    h, w = mask.shape[:2]
    y1 = max(0, min(h - 1, y1))
    y2 = max(y1 + 1, min(h, y2))
    col = np.mean(mask[y1:y2, :] > 0, axis=0).astype(np.float32)
    return cv.GaussianBlur(col.reshape(1, -1), (0, 0), sigmaX=max(2, w / 120)).reshape(-1)


def body_color_column_score(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    y1, y2 = int(0.30 * h), int(0.70 * h)
    strip = img[y1:y2, :]
    hsv = cv.cvtColor(strip, cv.COLOR_BGR2HSV).astype(np.float32)
    lab = cv.cvtColor(strip, cv.COLOR_BGR2LAB).astype(np.float32)
    gray = cv.cvtColor(strip, cv.COLOR_BGR2GRAY).astype(np.float32)

    S = np.median(hsv[:, :, 1], axis=0)
    V = np.median(hsv[:, :, 2], axis=0)
    B = np.median(lab[:, :, 2], axis=0)
    G = np.median(gray, axis=0)

    edge_w = max(2, int(0.06 * w))
    edge_idx = np.r_[0:edge_w, w - edge_w:w]
    bg_s = float(np.median(S[edge_idx]))
    bg_b = float(np.median(B[edge_idx]))
    bg_g = float(np.median(G[edge_idx]))

    score = (
        1.00 * np.maximum(S - max(18.0, bg_s + 4.0), 0) +
        3.00 * np.maximum(B - bg_b - 2.0, 0) +
        0.55 * np.maximum(bg_g - G - 8.0, 0)
    )
    score[V > 252] *= 0.4
    return cv.GaussianBlur(score.reshape(1, -1), (0, 0), sigmaX=max(2, w / 80)).reshape(-1)


def select_body_x_range_from_color_score(img: np.ndarray) -> Optional[Tuple[int, int]]:
    h, w = img.shape[:2]
    hsv = cv.cvtColor(img[int(0.30 * h):int(0.70 * h), :], cv.COLOR_BGR2HSV).astype(np.float32)
    S = np.median(hsv[:, :, 1], axis=0)
    edge_w = max(2, int(0.08 * w))
    left_edge_s = float(np.median(S[:edge_w]))
    right_edge_s = float(np.median(S[w - edge_w:]))

    if left_edge_s >= 34 and right_edge_s >= 34:
        return None

    score = body_color_column_score(img)
    if score.max() <= 3:
        return None

    best = None
    thresholds = [0.30, 0.24, 0.20, 0.16, 0.12, 0.09, 0.06]
    min_ratio = max(0.18, MIN_BODY_WIDTH_RATIO * 0.75)

    for frac in thresholds:
        thr = max(3.0, float(frac * score.max()))
        xs = np.where(score >= thr)[0]
        segments = merge_close_segments(contiguous_segments(xs), max_gap=max(8, int(0.14 * w)))
        for a, b in segments:
            seg_w = b - a + 1
            if seg_w < min_ratio * w or seg_w > 0.985 * w:
                continue
            center_penalty = abs(((a + b) / 2) - w / 2) / w
            mean_score = float(np.mean(score[a:b + 1]) / (score.max() + 1e-6))
            rank = 0.90 * (seg_w / w) + 0.55 * mean_score - 0.28 * center_penalty
            if best is None or rank > best[0]:
                best = (rank, int(a), int(b))

    if best is None:
        return None

    _, x1, x2 = best
    if x2 - x1 + 1 < 0.22 * w:
        return None

    pad = max(4, int(0.040 * w), int(0.080 * (x2 - x1 + 1)))
    x1 = max(0, x1 - pad)
    x2 = min(w - 1, x2 + pad)

    crop_w = x2 - x1 + 1
    if crop_w > 120:
        trim = max(2, int(BODY_EDGE_TRIM_RATIO * crop_w))
        x1 = min(x2 - 1, x1 + trim)
        x2 = max(x1 + 1, x2 - trim)

    if x2 - x1 + 1 < 0.26 * w:
        return None
    return x1, x2


def select_body_x_range_from_mask(mask: np.ndarray, *, small_body_mode: bool) -> Optional[Tuple[int, int]]:
    h, w = mask.shape[:2]
    profile = column_thickness_profile(mask, int(0.12 * h), int(0.88 * h))
    if profile.max() <= 2:
        profile = column_coverage_profile(mask, int(0.18 * h), int(0.82 * h)) * h
    if profile.max() <= 2:
        return None

    min_ratio = MIN_BODY_WIDTH_RATIO if small_body_mode else max(MIN_BODY_WIDTH_RATIO, 0.34)
    thresholds = [0.66, 0.58, 0.50, 0.42, 0.34, 0.28, 0.22] if small_body_mode else [0.58, 0.52, 0.46, 0.40, 0.34, 0.28]
    best = None

    for frac in thresholds:
        xs = np.where(profile >= frac * profile.max())[0]
        segments = merge_close_segments(contiguous_segments(xs), max_gap=max(8, int(0.08 * w)))
        for a, b in segments:
            seg_w = b - a + 1
            if seg_w < min_ratio * w or seg_w > 0.96 * w:
                continue
            center_penalty = abs(((a + b) / 2) - w / 2) / w
            mean_thick = float(np.mean(profile[a:b + 1]) / (profile.max() + 1e-6))
            score = 0.70 * (seg_w / w) + 0.55 * mean_thick - 0.35 * center_penalty
            if best is None or score > best[0]:
                best = (score, int(a), int(b))
        if best is not None:
            break

    if best is None:
        return None

    _, x1, x2 = best
    pad = max(3, int(0.010 * w), int(0.018 * (x2 - x1 + 1)))
    x1 = max(0, x1 - pad)
    x2 = min(w - 1, x2 + pad)

    crop_w = x2 - x1 + 1
    if crop_w > 120:
        trim = max(2, int(BODY_EDGE_TRIM_RATIO * crop_w))
        x1 = min(x2 - 1, x1 + trim)
        x2 = max(x1 + 1, x2 - trim)

    if x2 - x1 + 1 < 0.18 * w:
        return None
    return x1, x2


def select_body_y_range_from_mask(mask: np.ndarray, x1: int, x2: int) -> Optional[Tuple[int, int]]:
    h, _ = mask.shape[:2]
    sub = mask[:, x1:x2 + 1]
    row = np.mean(sub > 0, axis=1).astype(np.float32)
    row = cv.GaussianBlur(row.reshape(-1, 1), (0, 0), sigmaY=max(1.5, h / 90), sigmaX=0).reshape(-1)

    if row.max() <= 0.02:
        return None

    ys = np.where(row >= max(0.06, 0.13 * row.max()))[0]
    segments = contiguous_segments(ys)
    if not segments:
        return None

    y1, y2 = max(segments, key=lambda p: p[1] - p[0])
    pad_y = max(3, int(0.10 * (y2 - y1 + 1)))
    y1 = max(0, y1 - pad_y)
    y2 = min(h - 1, y2 + pad_y)

    if y2 - y1 + 1 < 0.45 * h:
        cy = (y1 + y2) // 2
        half = int(0.26 * h)
        y1 = max(0, cy - half)
        y2 = min(h - 1, cy + half)
    return y1, y2


def crop_to_body_from_mask(img: np.ndarray, mask: np.ndarray, *, small_body_mode: bool) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    h, w = img.shape[:2]
    x_range = select_body_x_range_from_mask(mask, small_body_mode=small_body_mode)
    if x_range is None:
        x1, x2 = int(0.02 * w), int(0.98 * w) - 1
    else:
        x1, x2 = x_range

    y_range = select_body_y_range_from_mask(mask, x1, x2)
    if y_range is None:
        y1, y2 = int(0.05 * h), int(0.95 * h) - 1
    else:
        y1, y2 = y_range

    body = img[y1:y2 + 1, x1:x2 + 1].copy()
    body_mask = mask[y1:y2 + 1, x1:x2 + 1].copy()

    if body.shape[1] < TARGET_BODY_WIDTH:
        scale = TARGET_BODY_WIDTH / body.shape[1]
        body = cv.resize(body, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        body_mask = cv.resize(body_mask, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

    return body, (body_mask > 0).astype(np.uint8) * 255, (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def crop_to_body_without_background(slot_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    img = slot_img.copy()
    if img.shape[0] > img.shape[1] * 1.15:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    if w < 80 or h < 30:
        return img, np.ones((h, w), np.uint8) * 255, (0, 0, w, h)

    mask = build_body_likelihood_mask(img)
    x_range = select_body_x_range_from_color_score(img)
    if x_range is None:
        x_range = select_body_x_range_from_mask(mask, small_body_mode=True)

    if x_range is None:
        x1, x2 = int(0.02 * w), int(0.98 * w) - 1
    else:
        x1, x2 = x_range

    y_range = select_body_y_range_from_mask(mask, x1, x2)
    if y_range is None:
        y1, y2 = int(0.05 * h), int(0.95 * h) - 1
    else:
        y1, y2 = y_range

    body = img[y1:y2 + 1, x1:x2 + 1].copy()
    body_mask = mask[y1:y2 + 1, x1:x2 + 1].copy()

    if body.shape[1] < TARGET_BODY_WIDTH:
        scale = TARGET_BODY_WIDTH / body.shape[1]
        body = cv.resize(body, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        body_mask = cv.resize(body_mask, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

    return body, (body_mask > 0).astype(np.uint8) * 255, (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def resize_background_to_match(bg: np.ndarray, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if bg.shape[:2] != (h, w):
        bg = cv.resize(bg, (w, h), interpolation=cv.INTER_AREA)
    return bg


def build_foreground_mask_by_background(img: np.ndarray, bg: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    bg = resize_background_to_match(bg, img)

    img_blur = cv.GaussianBlur(img, (3, 3), 0)
    bg_blur = cv.GaussianBlur(bg, (3, 3), 0)

    lab_img = cv.cvtColor(img_blur, cv.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv.cvtColor(bg_blur, cv.COLOR_BGR2LAB).astype(np.float32)
    hsv_img = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv_bg = cv.cvtColor(bg_blur, cv.COLOR_BGR2HSV).astype(np.float32)
    gray_img = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY).astype(np.float32)
    gray_bg = cv.cvtColor(bg_blur, cv.COLOR_BGR2GRAY).astype(np.float32)

    border_size = max(3, int(0.05 * min(h, w)))
    border = np.zeros((h, w), np.uint8)
    border[:border_size, :] = 1
    border[-border_size:, :] = 1
    border[:, :border_size] = 1
    border[:, -border_size:] = 1

    lab_offset = np.median(lab_img[border > 0], axis=0) - np.median(lab_bg[border > 0], axis=0)
    gray_offset = float(np.median(gray_img[border > 0]) - np.median(gray_bg[border > 0]))
    lab_bg_aligned = lab_bg + lab_offset
    gray_bg_aligned = gray_bg + gray_offset

    d_l = np.abs(lab_img[:, :, 0] - lab_bg_aligned[:, :, 0])
    d_a = np.abs(lab_img[:, :, 1] - lab_bg_aligned[:, :, 1])
    d_b = np.abs(lab_img[:, :, 2] - lab_bg_aligned[:, :, 2])
    lab_score = 0.45 * d_l + 1.20 * d_a + 1.20 * d_b

    s_img = hsv_img[:, :, 1]
    s_bg = hsv_bg[:, :, 1]
    v_img = hsv_img[:, :, 2]
    gray_diff = np.abs(gray_img - gray_bg_aligned)

    bg_noise = lab_score[border > 0]
    score_thr = max(10.0, float(np.percentile(bg_noise, 95)) + 5.0)

    mask = (
        (lab_score > score_thr) |
        ((s_img > s_bg + 14) & (s_img > 30) & (v_img < 252)) |
        ((gray_diff > 26) & (v_img < 250))
    ).astype(np.uint8) * 255

    mask[: int(0.04 * h), :] = 0
    mask[int(0.96 * h):, :] = 0
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    close_kernel = np.ones((max(3, h // 45), max(7, w // 45)), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, close_kernel, iterations=1)

    fg_ratio = float(np.mean(mask > 0))
    if fg_ratio < 0.015 or fg_ratio > 0.70:
        return build_body_likelihood_mask(img)
    return mask


def crop_to_body_with_background(slot_img: np.ndarray, background_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    img = slot_img.copy()
    bg = background_img.copy()

    if img.shape[0] > img.shape[1] * 1.15:
        img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        bg = cv.rotate(bg, cv.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    if w < 80 or h < 30:
        return img, np.ones((h, w), np.uint8) * 255, (0, 0, w, h)

    mask = build_foreground_mask_by_background(img, bg)
    return crop_to_body_from_mask(img, mask, small_body_mode=False)


def crop_to_body(slot_img: np.ndarray, background_img: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    if background_img is None:
        return crop_to_body_without_background(slot_img)
    return crop_to_body_with_background(slot_img, background_img)


def load_background_image(image_path: str) -> Optional[np.ndarray]:
    if not USE_BACKGROUND_SUBTRACTION:
        return None

    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    names = []
    if BACKGROUND_IMAGE_PATH:
        names.append(BACKGROUND_IMAGE_PATH)
    for alias in BACKGROUND_IMAGE_ALIASES:
        if alias not in names:
            names.append(alias)

    candidates = []
    for name in names:
        candidates.extend([
            name,
            os.path.join(os.path.dirname(image_path), name),
            os.path.join(os.getcwd(), name),
            os.path.join(script_dir, name),
        ])

    for path in candidates:
        if path and os.path.exists(path):
            bg = cv.imread(path)
            if bg is not None:
                if USE_FIXED_CROP:
                    bg = fixed_crop_image(bg, FIXED_CROP)
                return bg
    return None


# ===================== signal and candidate detection =====================

def column_profiles(body: np.ndarray):
    h, w = body.shape[:2]
    bands = [(0.24, 0.42), (0.38, 0.58), (0.54, 0.74)]
    signals, hs, ss, vs = [], [], [], []

    for a, b in bands:
        y1, y2 = int(a * h), max(int(a * h) + 1, int(b * h))
        strip = body[y1:y2, :]
        hsv = cv.cvtColor(strip, cv.COLOR_BGR2HSV).astype(np.float32)
        lab = cv.cvtColor(strip, cv.COLOR_BGR2LAB).astype(np.float32)

        med_h = np.zeros(w, np.float32)
        med_s = np.zeros(w, np.float32)
        med_v = np.zeros(w, np.float32)
        med_a = np.zeros(w, np.float32)
        med_b = np.zeros(w, np.float32)

        for x in range(w):
            pix_hsv = hsv[:, x, :].reshape(-1, 3)
            pix_lab = lab[:, x, :].reshape(-1, 3)
            v = pix_hsv[:, 2]
            lo, hi = np.percentile(v, 8), np.percentile(v, 92)
            keep = (v >= lo) & (v <= hi)
            if np.sum(keep) >= 3:
                pix_hsv = pix_hsv[keep]
                pix_lab = pix_lab[keep]
            med_h[x] = circular_hue_median(pix_hsv[:, 0])
            med_s[x] = np.median(pix_hsv[:, 1])
            med_v[x] = np.median(pix_hsv[:, 2])
            med_a[x] = np.median(pix_lab[:, 1])
            med_b[x] = np.median(pix_lab[:, 2])

        sigma = max(14, w / 9)
        base_s = cv.GaussianBlur(med_s.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        base_v = cv.GaussianBlur(med_v.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        base_a = cv.GaussianBlur(med_a.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        base_b = cv.GaussianBlur(med_b.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)

        color_delta = np.sqrt((med_a - base_a) ** 2 + (med_b - base_b) ** 2)
        sig = np.maximum(med_s - base_s, 0) + 0.35 * np.maximum(base_v - med_v, 0) + 0.42 * color_delta
        sig = cv.GaussianBlur(sig.reshape(1, -1), (0, 0), sigmaX=max(1.5, w / 220)).reshape(-1)
        signals.append(sig)
        hs.append(med_h)
        ss.append(med_s)
        vs.append(med_v)

    signal = 0.70 * np.mean(np.vstack(signals), axis=0) + 0.30 * np.min(np.vstack(signals), axis=0)
    signal = cv.GaussianBlur(signal.reshape(1, -1), (0, 0), sigmaX=max(1.5, w / 220)).reshape(-1)
    return signal, np.mean(np.vstack(hs), axis=0), np.mean(np.vstack(ss), axis=0), np.mean(np.vstack(vs), axis=0)


def patch_stats(body: np.ndarray, peak_x: int, half: Optional[int] = None) -> Dict[str, float]:
    h, w = body.shape[:2]
    if half is None:
        half = max(5, int(0.014 * w))
    x1, x2 = max(0, peak_x - half), min(w, peak_x + half + 1)
    y1, y2 = int(0.18 * h), int(0.82 * h)
    patch = body[y1:y2, x1:x2]
    if patch.size == 0:
        return {"valid": 0.0}

    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV).astype(np.float32)
    bgr = patch.astype(np.float32)
    h_vals = hsv[:, :, 0].reshape(-1)
    s_vals = hsv[:, :, 1].reshape(-1)
    v_vals = hsv[:, :, 2].reshape(-1)

    salience = 1.20 * s_vals + 0.25 * np.abs(v_vals - np.median(v_vals))
    threshold = np.percentile(salience, 60)
    salient = salience >= threshold
    if np.sum(salient) < max(5, 0.10 * len(h_vals)):
        salient = np.ones_like(h_vals, dtype=bool)

    b = bgr[:, :, 0].reshape(-1)[salient]
    g = bgr[:, :, 1].reshape(-1)[salient]
    r = bgr[:, :, 2].reshape(-1)[salient]

    return {
        "valid": 1.0,
        "h50": circular_hue_median(h_vals),
        "s50": float(np.median(s_vals)),
        "v50": float(np.median(v_vals)),
        "v10": float(np.percentile(v_vals, 10)),
        "v20": float(np.percentile(v_vals, 20)),
        "h_band": circular_hue_median(h_vals[salient]),
        "s_band": float(np.median(s_vals[salient])),
        "v_band": float(np.median(v_vals[salient])),
        "r_med": float(np.median(r)),
        "g_med": float(np.median(g)),
        "b_med": float(np.median(b)),
    }


def build_white_signal(med_s: np.ndarray, med_v: np.ndarray) -> np.ndarray:
    w = len(med_s)
    base_v = cv.GaussianBlur(med_v.reshape(1, -1), (0, 0), sigmaX=max(14, w / 9)).reshape(-1)
    bright = np.maximum(med_v - base_v, 0)
    low_s = np.clip((92.0 - med_s) / 36.0, 0.0, 1.0)
    white = bright * low_s
    white[(med_v < 185) | (med_s > 92)] = 0
    return cv.GaussianBlur(white.reshape(1, -1), (0, 0), sigmaX=max(1.5, w / 260)).reshape(-1)


def band_box(peak_x: int, signal: np.ndarray, body: np.ndarray, is_tolerance: bool = False) -> Tuple[int, int, int, int]:
    h, w = body.shape[:2]
    peak = float(signal[peak_x])
    low = max(float(np.percentile(signal, 45)), peak * (0.45 if is_tolerance else 0.34))
    left = peak_x
    while left > 0 and signal[left - 1] >= low:
        left -= 1
    right = peak_x
    while right < w - 1 and signal[right + 1] >= low:
        right += 1

    min_w = max(8, int(0.014 * w))
    max_w = max(min_w + 2, int((0.060 if is_tolerance else 0.078) * w))
    if right - left + 1 < min_w:
        half = min_w // 2
        left, right = max(0, peak_x - half), min(w - 1, peak_x + half)
    if right - left + 1 > max_w:
        half = max_w // 2
        left, right = max(0, peak_x - half), min(w - 1, peak_x + half)
    y1, y2 = int(0.18 * h), int(0.82 * h)
    return (left, y1, right - left + 1, y2 - y1 + 1)


def detect_candidates(body: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
    h, w = body.shape[:2]
    signal, med_h, med_s, med_v = column_profiles(body)

    prom = max(2.5, float(np.std(signal) * 0.20))
    peaks, props = local_find_peaks(signal, prominence=prom, distance=max(8, w // 22))

    reps = []
    for p, pr in zip(peaks, props.get("prominences", np.ones_like(peaks))):
        p, pr = int(p), float(pr)
        if not reps or p - reps[-1][-1][0] > max(12, w // 18):
            reps.append([(p, pr)])
        else:
            reps[-1].append((p, pr))
    selected = [max(group, key=lambda item: item[1]) for group in reps]

    candidates = []
    for p, pr in selected:
        stats = patch_stats(body, p)
        digit = classify_digit(stats)
        mult = classify_multiplier(stats)
        tol = classify_tolerance(stats)
        candidates.append({"peak": p, "prom": pr, "source": "color", **stats, "digit_color": digit, "mult_color": mult, "tol_color": tol})

    white_signal = build_white_signal(med_s, med_v)
    w_peaks, w_props = local_find_peaks(white_signal, prominence=max(3.0, float(np.std(white_signal) * 0.35)), distance=max(10, w // 18))
    for p, pr in zip(w_peaks, w_props.get("prominences", np.ones_like(w_peaks))):
        p = int(p)
        if p < 0.07 * w or p > 0.93 * w:
            continue
        if any(abs(p - c["peak"]) < max(10, w // 35) for c in candidates):
            continue
        stats = patch_stats(body, p)
        if classify_white(stats) is False:
            continue
        candidates.append({"peak": p, "prom": float(max(pr, white_signal[p])), "source": "white", **stats, "digit_color": "WHITE", "mult_color": "WHITE", "tol_color": "UNKNOWN"})

    candidates = sorted(candidates, key=lambda c: c["peak"])
    for c in candidates:
        c["box"] = band_box(c["peak"], signal, body, c["tol_color"] in TOLERANCE_COLORS)
    return candidates, signal


# ===================== color classification =====================

def classify_white(st: Dict[str, float]) -> bool:
    s = st.get("s50", 255)
    v = st.get("v50", 0)
    v20 = st.get("v20", 0)
    sb = st.get("s_band", s)
    vb = st.get("v_band", v)
    return bool(v >= 180 and v20 >= 145 and s <= 55 and sb <= 65 and vb >= 155)


def classify_black(st: Dict[str, float]) -> bool:
    if classify_white(st):
        return False
    h = st.get("h50", 0)
    s = st.get("s50", 255)
    v = st.get("v50", 255)
    v20 = st.get("v20", 255)
    sb = st.get("s_band", s)
    vb = st.get("v_band", v)
    r_med = st.get("r_med", 0)
    g_med = st.get("g_med", 0)
    b_med = st.get("b_med", 0)

    # Reject black classification for dark bands with brown/orange hue
    if 8 <= h <= 26 and s >= 60 and sb >= 70 and 80 <= v <= 110 and r_med > g_med > b_med:
        return False

    return bool(
        (s <= 72 and (v < 72 or v20 < 58 or vb < 65))
        or (s <= 85 and sb <= 85 and v <= 95 and v20 <= 105)
        or (s <= 55 and sb <= 78 and v <= 104 and v20 <= 95)
        or (s <= 35 and sb <= 45 and v <= 112 and v20 <= 82)
        or (s <= 38 and sb <= 58 and v <= 155 and v20 <= 95 and vb <= 125)
    )


def classify_gray(st: Dict[str, float]) -> bool:
    if classify_white(st):
        return False
    s = st.get("s50", 255)
    v = st.get("v50", 255)
    v20 = st.get("v20", 255)
    sb = st.get("s_band", s)
    vb = st.get("v_band", v)
    r_med = st.get("r_med", 0)
    g_med = st.get("g_med", 0)
    b_med = st.get("b_med", 0)
    gray_rgb = max(r_med, g_med, b_med) - min(r_med, g_med, b_med)

    return bool(
        (s <= 64 and sb <= 92 and 82 <= v <= 132 and 62 <= v20 <= 122 and vb <= 140)
        or (s <= 24 and sb <= 50 and 82 <= v <= 140 and 55 <= v20 <= 135 and vb <= 155)
        or (s <= 40 and sb <= 65 and 82 <= v <= 140 and 50 <= v20 <= 130 and vb <= 150 and gray_rgb <= 32)
    )


def classify_silver(st: Dict[str, float]) -> bool:
    if classify_white(st) or classify_gray(st) or classify_black(st):
        return False
    s = st.get("s50", 255)
    v = st.get("v50", 0)
    v20 = st.get("v20", 0)
    sb = st.get("s_band", s)
    return bool(s < 48 and sb < 75 and 145 <= v <= 230 and v20 >= 118)


def classify_gold(st: Dict[str, float], tolerance: bool = False) -> bool:
    if classify_white(st) or classify_gray(st) or classify_black(st):
        return False
    h = st.get("h50", 0)
    s = st.get("s50", 0)
    v = st.get("v50", 0)
    hb = st.get("h_band", h)
    sb = st.get("s_band", s)
    vb = st.get("v_band", v)
    hue_ok = (14 <= h <= 34) or (14 <= hb <= 34)
    sat_ok = 25 <= s <= (210 if tolerance else 140) or 40 <= sb <= (220 if tolerance else 145)
    bright_ok = v >= 75 and vb >= 65
    orange_like = (h <= 13 or hb <= 13) and (s >= 135 or sb >= 165)
    return bool(hue_ok and sat_ok and bright_ok and not orange_like)


def yellow_as_digit_or_multiplier(st: Dict[str, float]) -> bool:
    h = st.get("h50", 0)
    s = st.get("s50", 0)
    v = st.get("v50", 0)
    hb = st.get("h_band", h)
    sb = st.get("s_band", s)
    vb = st.get("v_band", v)

    direct_yellow = (
        22 <= h <= 42 and
        s >= 120 and
        v >= 155
    )

    band_yellow = (
        22 <= hb <= 42 and
        sb >= 120 and
        vb >= 150
    )

    return bool(direct_yellow or band_yellow)


def classify_violet(st: Dict[str, float]) -> bool:
    h = st.get("h50", 0)
    hb = st.get("h_band", h)
    s = st.get("s50", 0)
    sb = st.get("s_band", s)

    r = st.get("r_med", 0)
    g = st.get("g_med", 0)
    b = st.get("b_med", 0)

    hue_near_violet = (112 <= h <= 155) or (112 <= hb <= 155)
    enough_color = (s >= 28 or sb >= 35)

    purple_rgb = (
        b >= g + 8 and
        r >= g + 5 and
        b >= 0.82 * r
    )

    strong_violet_hue = (128 <= h <= 170) or (128 <= hb <= 170)

    return bool(enough_color and (strong_violet_hue or (hue_near_violet and purple_rgb)))


def classify_digit(st: Dict[str, float]) -> str:
    if st.get("valid", 0) < 1:
        return "UNKNOWN"
    h = st.get("h50", 0)
    s = st.get("s50", 0)
    v = st.get("v50", 0)
    v20 = st.get("v20", 0)
    hb = st.get("h_band", h)
    sb = st.get("s_band", s)
    vb = st.get("v_band", v)

    if v < 55 or (v20 < 38 and s < 185):
        return "BLACK"
    if classify_violet(st):
        return "VIOLET"
    if 88 < h <= 128 and s > 38:
        return "BLUE"
    if 128 < h <= 170 and s > 35:
        return "VIOLET"
    if 35 < h <= 88 and s > 38:
        return "GREEN"
    if classify_black(st):
        return "BLACK"
    if classify_gray(st):
        return "GRAY"
    if classify_white(st):
        return "WHITE"
    if 22 <= h <= 42 and s >= 120 and v >= 155:
        return "YELLOW"

    if h <= 7 or h >= 173:
        if h <= 4 or h >= 176:
            return "RED" if (v >= 82 and s >= 82 and sb >= 90) else "BROWN"
        return "RED" if (v >= 168 and s >= 92 and sb >= 105 and st.get("r_med", 0) / (st.get("g_med", 1) + 1) >= 1.50) else "BROWN"
    if 7 < h <= 12:
        if h >= 8 and s >= 130 and v >= 165 and sb >= 135:
            return "ORANGE"
        return "BROWN"
    if 12 < h <= 22:
        if (hb <= 6 or hb >= 173) and sb >= 100 and vb >= 100:
            return "RED"
        if v >= 165 and s >= 120 and sb >= 130:
            return "ORANGE"
        return "BROWN"
    if 22 < h <= 34:
        if s >= 125 and v >= 150:
            return "ORANGE" if h < 27 else "YELLOW"
        return "BROWN"
    return "UNKNOWN"


def classify_multiplier(st: Dict[str, float]) -> str:
    if classify_silver(st):
        return "SILVER"
    h = st.get("h50", 0)
    hb = st.get("h_band", h)
    v = st.get("v50", 0)
    vb = st.get("v_band", v)
    if classify_gold(st, tolerance=False) and h >= 16 and hb >= 15 and v >= 145 and vb >= 135:
        return "GOLD"
    if classify_black(st) or classify_gray(st):
        return "BLACK"
    if classify_white(st):
        return "WHITE"
    return classify_digit(st)


def classify_tolerance(st: Dict[str, float]) -> str:
    if classify_silver(st):
        return "SILVER"

    if yellow_as_digit_or_multiplier(st):
        return "UNKNOWN"

    if classify_gold(st, tolerance=True):
        return "GOLD"

    return "UNKNOWN"


# ===================== band selection / decoding =====================

def role_color(c: Dict, role: str) -> str:
    if role in {"digit1", "digit2"}:
        return c["digit_color"]
    if role == "mult":
        if is_confirmed_digit_candidate(c):
            return c["digit_color"]
        return c["mult_color"]
    return c["tol_color"]


def role_score(c: Dict, role: str, w: int) -> float:
    score = float(c.get("prom", 0))
    edge = min(c["peak"], w - 1 - c["peak"])
    color = role_color(c, role)

    if role in {"digit1", "digit2"}:
        if is_confirmed_gold_candidate(c):
            score -= 170
        if color in COLOR_DB and COLOR_DB[color].digit is not None:
            score += 24
        else:
            score -= 120
        if role == "digit1" and color == "WHITE":
            score -= 85
        elif color == "WHITE":
            score -= 16
        if c["tol_color"] in TOLERANCE_COLORS:
            score -= 65
        if edge < 0.025 * w:
            score -= 25
    elif role == "mult":
        if color in COLOR_DB and COLOR_DB[color].multiplier is not None:
            score += 18
        else:
            score -= 100
        confirmed_gold = is_confirmed_gold_candidate(c) and color == "GOLD"
        if confirmed_gold:
            score += 70
        if color in TOLERANCE_COLORS and not confirmed_gold:
            score -= 70
        if c["tol_color"] in TOLERANCE_COLORS and color in TOLERANCE_COLORS and not confirmed_gold:
            score -= 45
    else:
        if color in TOLERANCE_COLORS:
            score += 55
        else:
            score -= 150
        if is_confirmed_gold_candidate(c):
            score += 55
        if edge < 0.025 * w:
            score -= 35
        score += 0.10 * max(0, 0.25 * w - edge)
        looks_like_gold_tolerance = c.get("mult_color") == "GOLD" and c.get("tol_color") == "GOLD"
        if not looks_like_gold_tolerance and c["digit_color"] in {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "VIOLET", "BLACK", "GRAY", "WHITE"}:
            score -= 35
    return score


def is_strong_digit_candidate(c: Dict) -> bool:
    return c.get("digit_color") in {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "VIOLET", "BLACK", "GRAY"}


def is_confirmed_digit_candidate(c: Dict) -> bool:
    looks_like_gold_tolerance = c.get("mult_color") == "GOLD" and c.get("tol_color") == "GOLD"
    return c.get("digit_color") in CONFIRMED_D_COLORS and not looks_like_gold_tolerance


def is_confirmed_gold_candidate(c: Dict) -> bool:
    return c.get("digit_color") == "ORANGE" and c.get("mult_color") == "GOLD" and c.get("tol_color") == "GOLD"


def is_weak_noise_candidate(c: Dict) -> bool:
    return c.get("digit_color") == "WHITE" or c.get("source") == "white"


def skipped_candidate_penalty(candidates: List[Dict], selected: List[Dict], ordered: List[Dict]) -> float:
    selected_ids = {id(c) for c in selected}
    penalty = 0.0

    for a, b in [(ordered[0], ordered[1]), (ordered[1], ordered[2])]:
        lo, hi = sorted((a["peak"], b["peak"]))
        for c in candidates:
            if id(c) in selected_ids:
                continue
            if lo < c["peak"] < hi:
                if is_strong_digit_candidate(c):
                    penalty += 85.0
                elif is_weak_noise_candidate(c):
                    penalty += 8.0
                elif c.get("digit_color") == "BROWN" and c.get("tol_color") not in TOLERANCE_COLORS:
                    penalty += 18.0

    return penalty


def ordered_gap_penalty(ordered: List[Dict], w: int) -> float:
    peaks = [c["peak"] for c in ordered]
    gaps = [abs(peaks[i] - peaks[i + 1]) for i in range(3)]
    if min(gaps) <= 0:
        return 200.0

    penalty = 0.0
    digit_gap_ref = max(1.0, min(gaps[0], gaps[1]))
    if gaps[1] > 1.85 * digit_gap_ref:
        penalty += 0.55 * (gaps[1] - 1.85 * digit_gap_ref)
    if gaps[0] > 2.30 * max(1.0, gaps[1]):
        penalty += 0.28 * (gaps[0] - 2.30 * max(1.0, gaps[1]))
    if gaps[2] < 0.35 * max(gaps[0], gaps[1], 1):
        penalty += 18.0
    if gaps[0] < max(7, w // 55) or gaps[1] < max(7, w // 55):
        penalty += 45.0
    return penalty


def digit_gap_regularity_score(ordered: List[Dict], w: int) -> float:
    g1 = abs(ordered[0]["peak"] - ordered[1]["peak"])
    g2 = abs(ordered[1]["peak"] - ordered[2]["peak"])
    normal_gap = max(1.0, 0.5 * (g1 + g2))
    diff_ratio = abs(g1 - g2) / normal_gap

    score = 170.0 * float(np.exp(-((diff_ratio / 0.18) ** 2)))
    if diff_ratio > 0.28:
        score -= min(95.0, 230.0 * (diff_ratio - 0.28))
    if min(g1, g2) < max(10, w // 50):
        score -= 120.0

    return score


def tolerance_gap_score(ordered: List[Dict], w: int) -> float:
    peaks = [c["peak"] for c in ordered]
    gaps = [abs(peaks[i] - peaks[i + 1]) for i in range(3)]
    if min(gaps) <= 0:
        return -240.0

    digit_gaps = gaps[:2]
    tol_gap = float(gaps[2])
    normal_gap = max(1.0, float(np.median(digit_gaps)))
    min_digit_gap = max(1.0, float(min(digit_gaps)))

    score = 0.0
    if tol_gap < 0.75 * min_digit_gap:
        score -= 190.0 + 1.4 * (0.75 * min_digit_gap - tol_gap)
    elif tol_gap < 1.00 * normal_gap:
        score -= 75.0 + 0.65 * (1.00 * normal_gap - tol_gap)
    elif tol_gap < 1.25 * normal_gap:
        score += 24.0 * ((tol_gap - normal_gap) / max(1.0, 0.25 * normal_gap))
    elif tol_gap <= 2.15 * normal_gap:
        center = 1.65 * normal_gap
        score += 68.0 - 18.0 * (abs(tol_gap - center) / max(1.0, 0.50 * normal_gap))
    else:
        score += 36.0
        score -= min(95.0, 0.55 * (tol_gap - 2.15 * normal_gap))

    if tol_gap < max(10, w // 45):
        score -= 110.0

    return score


def selected_box_overlap_penalty(ordered: List[Dict], signal: np.ndarray, body: np.ndarray, colors: List[str]) -> float:
    roles = ["digit1", "digit2", "mult", "tol"]
    boxes = [
        band_box(c["peak"], signal, body, role == "tol" and color in TOLERANCE_COLORS)
        for c, role, color in zip(ordered, roles, colors)
    ]
    penalty = 0.0

    for i in range(len(boxes)):
        x1, _, w1, _ = boxes[i]
        r1 = x1 + w1
        for j in range(i + 1, len(boxes)):
            x2, _, w2, _ = boxes[j]
            r2 = x2 + w2
            overlap = min(r1, r2) - max(x1, x2)
            if overlap <= 0:
                continue
            ratio = overlap / max(1.0, float(min(w1, w2)))
            adjacent = j == i + 1
            penalty += (160.0 if adjacent else 80.0) + (280.0 if adjacent else 150.0) * ratio

    return penalty


def selected_color_bonus(colors: List[str]) -> float:
    bonus = 0.0
    vivid = {"RED", "ORANGE", "YELLOW", "GREEN", "BLUE", "VIOLET", "BLACK", "GRAY"}
    for c in colors[:3]:
        if c in vivid:
            bonus += 12.0
    if "VIOLET" in colors[:3]:
        bonus += 18.0
    if "GREEN" in colors[:3]:
        bonus += 10.0
    if colors[0] == "WHITE":
        bonus -= 70.0
    if colors[1] == "WHITE":
        bonus -= 8.0
    if colors[2] in TOLERANCE_COLORS:
        bonus -= 85.0
    return bonus


def confirmed_digit_combo_score(candidates: List[Dict], selected: List[Dict], ordered: List[Dict], colors: List[str]) -> float:
    confirmed = [c for c in candidates if is_confirmed_digit_candidate(c) or is_confirmed_gold_candidate(c)]
    if not confirmed:
        return 0.0

    selected_ids = {id(c) for c in selected}
    score = 0.0

    for c in confirmed:
        if id(c) in selected_ids:
            score += 180.0
        else:
            score -= 10000.0 if len(confirmed) <= 4 else 430.0

    confirmed_digit_role_indexes = [i for i, c in enumerate(ordered) if is_confirmed_digit_candidate(c)]
    confirmed_gold_role_indexes = [i for i, c in enumerate(ordered) if is_confirmed_gold_candidate(c)]
    confirmed_first_three = [i for i in confirmed_digit_role_indexes if i < 3]

    for i in confirmed_digit_role_indexes:
        if i == 3:
            score -= 260.0
        else:
            score += 45.0

    for i in confirmed_gold_role_indexes:
        if i in {2, 3}:
            score += 95.0
        else:
            score -= 240.0

    count = len(confirmed_first_three)
    if count >= 2:
        distinct_colors = len({colors[i] for i in confirmed_first_three})
        score += 55.0 * (count - 1)
        score += 16.0 * distinct_colors
    if count == 3:
        score += 85.0

    return score


def pick_four_bands(body: np.ndarray):
    candidates, signal = detect_candidates(body)
    if len(candidates) < 4:
        return candidates, [], [], [], signal

    w = body.shape[1]
    confirmed = [c for c in candidates if is_confirmed_digit_candidate(c) or is_confirmed_gold_candidate(c)]
    ranked = sorted(candidates, key=lambda c: c["prom"] + (12 if c["tol_color"] in TOLERANCE_COLORS else 0), reverse=True)[:16]
    ranked_ids = {id(c) for c in ranked}
    ranked.extend(c for c in confirmed if id(c) not in ranked_ids)
    ranked = sorted(ranked, key=lambda c: c["peak"])

    import itertools
    best = None
    for subset in itertools.combinations(ranked, 4):
        spatial = sorted(subset, key=lambda c: c["peak"])
        gaps = np.diff([c["peak"] for c in spatial]).astype(np.float32)
        if np.min(gaps) < max(7, w // 48):
            continue
        span = spatial[-1]["peak"] - spatial[0]["peak"]
        if span < 0.28 * w:
            continue

        geometry_penalty = 0.07 * float(np.sum(np.abs(gaps - np.median(gaps))))
        if np.max(gaps) / max(np.min(gaps), 1) > 4.6:
            geometry_penalty += 30

        for direction in ["LEFT_TO_RIGHT", "RIGHT_TO_LEFT"]:
            ordered = spatial if direction == "LEFT_TO_RIGHT" else list(reversed(spatial))
            roles = ["digit1", "digit2", "mult", "tol"]
            colors = [role_color(c, r) for c, r in zip(ordered, roles)]
            score = sum(role_score(c, r, w) for c, r in zip(ordered, roles))
            score -= geometry_penalty
            score -= ordered_gap_penalty(ordered, w)
            score -= skipped_candidate_penalty(candidates, list(subset), ordered)
            score += selected_color_bonus(colors)
            score += confirmed_digit_combo_score(candidates, list(subset), ordered, colors)
            score += digit_gap_regularity_score(ordered, w)
            score += tolerance_gap_score(ordered, w)
            score -= selected_box_overlap_penalty(ordered, signal, body, colors)

            if colors[0] not in COLOR_DB or COLOR_DB[colors[0]].digit is None:
                score -= 200
            if colors[1] not in COLOR_DB or COLOR_DB[colors[1]].digit is None:
                score -= 200
            if colors[2] not in COLOR_DB or COLOR_DB[colors[2]].multiplier is None:
                score -= 160
            if colors[3] not in TOLERANCE_COLORS:
                score -= 260

            if colors[3] in TOLERANCE_COLORS:
                tol_edge = min(ordered[3]["peak"], w - 1 - ordered[3]["peak"])
                digit1_edge = min(ordered[0]["peak"], w - 1 - ordered[0]["peak"])
                if tol_edge > digit1_edge + 0.12 * w:
                    score -= 45

            if best is None or score > best["score"]:
                best = {"score": score, "ordered": ordered, "colors": colors, "direction": direction}

    if best is None:
        return candidates, [], [], [], signal

    selected = best["ordered"]
    boxes = [band_box(c["peak"], signal, body, c["tol_color"] in TOLERANCE_COLORS) for c in selected]
    peaks = [c["peak"] for c in selected]
    return candidates, boxes, best["colors"], peaks, signal


def decode_four_band(colors: List[str]) -> Tuple[float, str]:
    if len(colors) != 4:
        raise ValueError("need exactly 4 bands")
    c1, c2, c3, c4 = colors
    if c4 not in TOLERANCE_COLORS:
        raise ValueError(f"tolerance band must be GOLD or SILVER, got {c4}")
    b1, b2, b3, b4 = COLOR_DB[c1], COLOR_DB[c2], COLOR_DB[c3], COLOR_DB[c4]
    if b1.digit is None or b2.digit is None:
        raise ValueError(f"first two bands must be digit colors, got {c1}, {c2}")
    if b3.multiplier is None:
        raise ValueError(f"third band must be multiplier color, got {c3}")
    if b4.tolerance is None:
        raise ValueError(f"invalid tolerance color: {c4}")
    return (10 * b1.digit + b2.digit) * b3.multiplier, b4.tolerance


# ===================== debug output =====================

def draw_candidate_image(body: np.ndarray, candidates: List[Dict]) -> np.ndarray:
    img = body.copy()
    for c in candidates:
        x, y, w, h = c["box"]
        name = c["digit_color"] if c["digit_color"] != "UNKNOWN" else c["mult_color"]
        color = COLOR_DB.get(name, BandColor("UNKNOWN", debug_bgr=(255, 255, 255))).debug_bgr
        cv.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv.putText(img, name, (x, max(16, y - 4)), FONT, 0.42, color, 1, cv.LINE_AA)
    return img


def draw_final_image(body: np.ndarray, boxes: List[Tuple[int, int, int, int]], colors: List[str], text: str, direction: str, ok: bool) -> np.ndarray:
    img = body.copy()
    for i, (box, color_name) in enumerate(zip(boxes, colors), start=1):
        x, y, w, h = box
        color = COLOR_DB.get(color_name, BandColor("UNKNOWN", debug_bgr=(255, 255, 255))).debug_bgr
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv.putText(img, f"{i}:{color_name}", (x, max(18, y - 5)), FONT, 0.55, color, 2, cv.LINE_AA)
    cv.putText(img, text, (10, 28), FONT, 0.75, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, f"Order: {direction}", (10, 56), FONT, 0.60, (0, 255, 255), 2, cv.LINE_AA)
    cv.putText(img, f"Status: {'OK' if ok else 'FAIL'}", (10, 82), FONT, 0.60, (0, 255, 0) if ok else (0, 0, 255), 2, cv.LINE_AA)
    return img




def summarize_candidate_tolerance(candidates: List[Dict]) -> Tuple[List[Dict], str]:
    summary = []
    lines = []
    for i, c in enumerate(candidates, start=1):
        item = {
            "index": i,
            "peak": int(c.get("peak", -1)),
            "digit_color": c.get("digit_color", "UNKNOWN"),
            "mult_color": c.get("mult_color", "UNKNOWN"),
            "tol_color": c.get("tol_color", "UNKNOWN"),
            "source": c.get("source", "UNKNOWN"),
        }
        summary.append(item)
        lines.append(
            f"candidate {i:02d}: peak={item['peak']}, "
            f"D={item['digit_color']}, M={item['mult_color']}, "
            f"T={item['tol_color']}, source={item['source']}"
        )
    return summary, "\n".join(lines)

# ===================== main API =====================

def analyze_resistor_image(image_path: str, save_path: str = "annotated_result.jpg", strict: bool = False) -> Dict:
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if USE_FIXED_CROP:
        image = fixed_crop_image(image, FIXED_CROP)

    background = load_background_image(image_path)
    body, body_mask, crop_box = crop_to_body(image, background)
    candidates, band_boxes, band_colors, peaks, signal = pick_four_bands(body)

    ok = False
    value_ohm = None
    tolerance = None
    message = "failed"
    direction = "UNKNOWN"
    result_text = "Detected bands: not 4"

    if len(band_boxes) == 4 and len(band_colors) == 4 and len(peaks) == 4:
        direction = "RIGHT_TO_LEFT" if peaks[0] > peaks[-1] else "LEFT_TO_RIGHT"
        try:
            value_ohm, tolerance = decode_four_band(band_colors)
            ok = True
            message = "success"
            result_text = f"Resistance: {format_resistance_ascii(value_ohm)} {tolerance.replace('±', '+/-')}"
        except Exception as exc:
            message = str(exc)
            result_text = "Decode failed"

    save_dir = os.path.dirname(save_path) or "."
    base = os.path.splitext(os.path.basename(save_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    candidate_path = os.path.join(save_dir, f"{base}_candidates.jpg")
    candidate_tol_summary, candidate_tol_text = summarize_candidate_tolerance(candidates)
    candidate_tol_path = os.path.join(save_dir, f"{base}_candidate_tol.txt")

    cv.imwrite(candidate_path, draw_candidate_image(body, candidates))
    cv.imwrite(save_path, draw_final_image(body, band_boxes, band_colors, result_text, direction, ok))
    with open(candidate_tol_path, "w", encoding="utf-8") as f:
        f.write(candidate_tol_text + ("\n" if candidate_tol_text else ""))

    return {
        "ok": ok,
        "message": message,
        "body_crop_box": crop_box,
        "band_boxes": band_boxes,
        "peaks": peaks,
        "band_colors": band_colors,
        "resistance_ohm": value_ohm,
        "resistance_text": format_resistance(value_ohm) if value_ohm is not None else None,
        "tolerance": tolerance,
        "direction": direction,
        "candidate_tol_colors": candidate_tol_summary,
        "candidate_tol_text": candidate_tol_text,
        "candidate_tol_text_file": candidate_tol_path,
        "candidate_debug_image": candidate_path,
        "result_debug_image": save_path,
    }


if __name__ == "__main__":
    test_paths = [
        "50.jpg","8.2.jpg",
        
    ]
    for path in test_paths:
        if not os.path.exists(path):
            continue
        out = f"annotated_{os.path.splitext(os.path.basename(path))[0]}.jpg"
        result = analyze_resistor_image(path, out)
        print("\n==============================")
        print(f"Image: {path}")
        for key, value in result.items():
            print(f"{key}: {value}")
