import os
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

@dataclass
class BandColor:
    name: str
    digit: Optional[int] = None
    multiplier: Optional[float] = None
    tolerance: Optional[str] = None
    debug_bgr: Tuple[int, int, int] = (255, 255, 255)

COLOR_DB: Dict[str, BandColor] = {
    "BLACK":  BandColor("BLACK",  0, 1,             None,     (0, 0, 0)),
    "BROWN":  BandColor("BROWN",  1, 10,            "±1%",    (19, 69, 139)),
    "RED":    BandColor("RED",    2, 100,           "±2%",    (0, 0, 255)),
    "ORANGE": BandColor("ORANGE", 3, 1_000,         None,     (0, 140, 255)),
    "YELLOW": BandColor("YELLOW", 4, 10_000,        None,     (0, 255, 255)),
    "GREEN":  BandColor("GREEN",  5, 100_000,       "±0.5%",  (0, 255, 0)),
    "BLUE":   BandColor("BLUE",   6, 1_000_000,     "±0.25%", (255, 0, 0)),
    "VIOLET": BandColor("VIOLET", 7, 10_000_000,    "±0.1%",  (255, 0, 180)),
    "GRAY":   BandColor("GRAY",   8, 100_000_000,   "±0.05%", (128, 128, 128)),
    "WHITE":  BandColor("WHITE",  9, 1_000_000_000, None,     (255, 255, 255)),
    "GOLD":   BandColor("GOLD",   None, 0.1,        "±5%",    (0, 215, 255)),
    "SILVER": BandColor("SILVER", None, 0.01,       "±10%",   (192, 192, 192)),
}
FOUR_BAND_TOLERANCE_COLORS = {"GOLD", "SILVER"}
FONT = cv.FONT_HERSHEY_SIMPLEX

# ===================== config =====================
ASSUME_CROPPED_INPUT = True
TARGET_ROI_WIDTH = 700
# If you later want fixed crop, set (x1, y1, x2, y2). Keep None for now.
FIXED_CROP: Optional[Tuple[int, int, int, int]] = None

# ===================== utility =====================
def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    best_idx = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == best_idx] = 255
    return out

def fixed_crop_image(img: np.ndarray, crop: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    if crop is None:
        return img
    x1, y1, x2, y2 = crop
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(x1 + 1, min(w, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(y1 + 1, min(h, y2))
    return img[y1:y2, x1:x2].copy()

def rotate_image_keep_bounds(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv.warpAffine(image, M, (new_w, new_h), flags=cv.INTER_LINEAR,
                         borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

def normalize_angle_to_horizontal(rect) -> float:
    (_, _), (rw, rh), angle = rect
    # minAreaRect angle has awkward convention. This maps long-axis angle to roughly [-45, 45].
    if rw < rh:
        angle = angle + 90
    while angle < -45:
        angle += 90
    while angle > 45:
        angle -= 90
    return angle

def robust_foreground_mask(img: np.ndarray) -> np.ndarray:
    """
    對白色 3D 列印紋理板比較穩的前景 mask。
    不是只用 gray < 245，而是用邊框估背景顏色，再抓和背景 Lab 距離大的區域。
    """
    h, w = img.shape[:2]
    bd = max(3, int(0.05 * min(h, w)))
    border = np.zeros((h, w), np.uint8)
    border[:bd, :] = 1; border[-bd:, :] = 1; border[:, :bd] = 1; border[:, -bd:] = 1

    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    bg_lab = np.median(lab[border > 0], axis=0)
    bg_gray = float(np.median(gray[border > 0]))
    dist = np.linalg.norm(lab - bg_lab, axis=2)

    # 對紋理背景：背景本身也有 dist，所以門檻要依 border 紋理自適應。
    border_dist = dist[border > 0]
    dist_thr = max(18.0, float(np.percentile(border_dist, 90)) + 12.0)

    fg = (
        (dist > dist_thr) |
        ((hsv[:, :, 1] > 25) & (gray < 245)) |
        (gray < bg_gray - 35)
    ).astype(np.uint8) * 255

    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    # 用稍大的 close 讓導線和本體連起來，對實拍小圖比較重要。
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    # 如果還是整張太多，代表背景被吃掉，改用更保守的 mask。
    if np.mean(fg > 0) > 0.65:
        fg = (((dist > max(35.0, dist_thr + 12.0)) | ((hsv[:, :, 1] > 45) & (gray < 235)) | (gray < bg_gray - 50))).astype(np.uint8) * 255
        fg = cv.morphologyEx(fg, cv.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    return largest_component(fg)

def get_thickness_profile(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    thickness = np.zeros(w, dtype=np.float32)
    for x in range(w):
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) > 0:
            thickness[x] = ys[-1] - ys[0] + 1
    thickness = cv.GaussianBlur(thickness.reshape(1, -1), (0, 0), sigmaX=max(3, w / 80)).reshape(-1)
    return thickness

def find_body_x_range_from_thickness(thickness: np.ndarray) -> Tuple[int, int]:
    if thickness.max() <= 0:
        return 0, len(thickness) - 1
    # 實拍小圖可能本體和導線厚度差沒網路圖大，0.45 比 0.55 寬容。
    thr = 0.45 * thickness.max()
    valid = np.where(thickness >= thr)[0]
    if len(valid) == 0:
        return 0, len(thickness) - 1
    peak = int(np.argmax(thickness))
    left = peak
    while left > 0 and thickness[left] >= thr:
        left -= 1
    right = peak
    while right < len(thickness) - 1 and thickness[right] >= thr:
        right += 1
    pad = max(2, int(0.025 * len(thickness)))
    return max(0, left - pad), min(len(thickness) - 1, right + pad)

def format_resistance(value_ohm: float) -> str:
    if value_ohm >= 1_000_000:
        return f"{value_ohm / 1_000_000:.3g} MΩ"
    if value_ohm >= 1_000:
        return f"{value_ohm / 1_000:.3g} kΩ"
    return f"{value_ohm:.3g} Ω"

def format_resistance_ascii(value_ohm: float) -> str:
    # OpenCV putText 不支援 Ω / ± 這類字元，debug 圖上用 ASCII 避免變成 ???。
    if value_ohm >= 1_000_000:
        return f"{value_ohm / 1_000_000:.3g} MOhm"
    if value_ohm >= 1_000:
        return f"{value_ohm / 1_000:.3g} kOhm"
    return f"{value_ohm:.3g} Ohm"

# ===================== ROI / body =====================
def detect_resistor_roi(image: np.ndarray):
    """
    給目前手動裁切圖使用：
    1. 可先固定裁切（目前預設不使用）
    2. 用 robust mask 估計電阻方向並旋平
    3. 用厚度 profile 把導線裁掉，留下本體附近
    4. resize 到固定寬度，避免實拍圖色環像素太少
    """
    image = fixed_crop_image(image, FIXED_CROP)

    fg = robust_foreground_mask(image)
    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("找不到前景輪廓")

    cnt = max(contours, key=cv.contourArea)
    rect = cv.minAreaRect(cnt)
    rotate_angle = normalize_angle_to_horizontal(rect)

    # 實拍手動裁切圖常常只歪一點點；小角度旋轉會引入大量白邊，反而讓背景 mask 變差。
    # 所以角度很小時不旋轉，直接用原圖與原 mask。
    if abs(rotate_angle) < 10:
        rotated = image.copy()
        fg_rot = fg.copy()
    else:
        rotated = rotate_image_keep_bounds(image, rotate_angle)
        fg_rot = robust_foreground_mask(rotated)

    contours2, _ = cv.findContours(fg_rot, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours2:
        raise RuntimeError("旋正後找不到前景輪廓")

    cnt2 = max(contours2, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt2)
    rough = rotated[y:y+h, x:x+w]
    rough_mask = fg_rot[y:y+h, x:x+w]

    thickness = get_thickness_profile(rough_mask)
    bx1, bx2 = find_body_x_range_from_thickness(thickness)

    submask = rough_mask[:, bx1:bx2+1]
    ys, _ = np.where(submask > 0)
    if len(ys) == 0:
        by1, by2 = 0, h - 1
    else:
        by1 = max(0, int(np.percentile(ys, 1)))
        by2 = min(h - 1, int(np.percentile(ys, 99)))

    pad_x = max(3, int(0.04 * (bx2 - bx1 + 1)))
    pad_y = max(3, int(0.12 * (by2 - by1 + 1)))
    bx1 = max(0, bx1 - pad_x); bx2 = min(w - 1, bx2 + pad_x)
    by1 = max(0, by1 - pad_y); by2 = min(h - 1, by2 + pad_y)

    roi = rough[by1:by2+1, bx1:bx2+1].copy()
    roi_mask = rough_mask[by1:by2+1, bx1:bx2+1].copy()

    # 如果旋轉/裁切後仍是直向，保險轉成水平。
    if roi.shape[0] > roi.shape[1]:
        roi = cv.rotate(roi, cv.ROTATE_90_CLOCKWISE)
        roi_mask = cv.rotate(roi_mask, cv.ROTATE_90_CLOCKWISE)

    # 放大：實拍圖通常電阻太小，先標準化寬度。
    if roi.shape[1] < TARGET_ROI_WIDTH:
        scale = TARGET_ROI_WIDTH / roi.shape[1]
        roi = cv.resize(roi, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        roi_mask = cv.resize(roi_mask, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

    # 實拍紋理背景下，重新用 ROI 邊框估背景可能會把本體吃掉；
    # 因此保留前面由整張裁切圖得到的 mask，只做 resize。
    roi_mask = (roi_mask > 0).astype(np.uint8) * 255

    info = {
        "rotated_image": rotated,
        "rotation_angle": rotate_angle,
        "rough_box_in_rotated": (x, y, w, h),
        "body_crop_in_rough": (bx1, by1, bx2 - bx1 + 1, by2 - by1 + 1),
        "target_width": TARGET_ROI_WIDTH,
    }
    return roi, roi_mask, info

def detect_body_box(roi: np.ndarray, roi_mask: np.ndarray):
    # ROI 已經大致是 body crop，這裡只做保守內縮，不再讓整張背景進 body。
    mask = largest_component(roi_mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = roi.shape[:2]
        return (0, 0, w, h), np.ones((h, w), dtype=np.uint8) * 255

    cnt = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(cnt)

    # 用厚度再修一次 x 範圍，避免導線殘留太多。
    th = get_thickness_profile(mask)
    bx1, bx2 = find_body_x_range_from_thickness(th)
    x1 = max(0, min(x + max(1, int(0.01*w)), bx1))
    x2 = min(roi.shape[1], max(x + w - max(1, int(0.01*w)), bx2))

    y1 = max(0, y + max(1, int(0.06 * h)))
    y2 = min(roi.shape[0], y + h - max(1, int(0.06 * h)))

    if x2 <= x1 or y2 <= y1:
        return (0, 0, roi.shape[1], roi.shape[0]), mask
    return (x1, y1, x2 - x1, y2 - y1), mask

# ===================== signal / color =====================
def build_signal_and_profiles(roi: np.ndarray, body_box, body_mask: np.ndarray):
    x, y, w, h = body_box
    body = roi[y:y+h, x:x+w]
    mask = body_mask[y:y+h, x:x+w]

    # 三條水平帶，加強穩定性：真正色環在多條水平帶都會有 peak。
    bands = [(0.30, 0.46), (0.42, 0.58), (0.54, 0.70)]
    signals = []
    med_h_all = []
    med_s_all = []
    med_v_all = []
    core_mask_acc = np.zeros((max(1, int(0.70*h)-int(0.30*h)), w), dtype=np.uint8)

    for a, b in bands:
        cy1 = int(a * h); cy2 = max(cy1 + 1, int(b * h))
        core = body[cy1:cy2, :]
        core_mask = mask[cy1:cy2, :]
        hsv = cv.cvtColor(core, cv.COLOR_BGR2HSV).astype(np.float32)
        lab = cv.cvtColor(core, cv.COLOR_BGR2LAB).astype(np.float32)

        med_h = np.zeros(w, dtype=np.float32)
        med_s = np.zeros(w, dtype=np.float32)
        med_v = np.zeros(w, dtype=np.float32)
        med_a = np.zeros(w, dtype=np.float32)
        med_b = np.zeros(w, dtype=np.float32)
        valid = np.zeros(w, dtype=np.uint8)
        for i in range(w):
            m = core_mask[:, i] > 0
            if np.sum(m) < 2:
                continue
            pix_hsv = hsv[:, i, :][m]
            pix_lab = lab[:, i, :][m]
            V = pix_hsv[:, 2]
            lo, hi = np.percentile(V, 8), np.percentile(V, 92)
            keep = (V >= lo) & (V <= hi)
            if np.sum(keep) >= 2:
                pix_hsv = pix_hsv[keep]; pix_lab = pix_lab[keep]
            med_h[i] = np.median(pix_hsv[:, 0])
            med_s[i] = np.median(pix_hsv[:, 1])
            med_v[i] = np.median(pix_hsv[:, 2])
            med_a[i] = np.median(pix_lab[:, 1])
            med_b[i] = np.median(pix_lab[:, 2])
            valid[i] = 1
        if not np.any(valid):
            continue
        # fill missing
        for i in range(w):
            if not valid[i] and i > 0:
                med_h[i] = med_h[i-1]; med_s[i] = med_s[i-1]; med_v[i] = med_v[i-1]; med_a[i] = med_a[i-1]; med_b[i] = med_b[i-1]
        for i in range(w-2, -1, -1):
            if not valid[i]:
                med_h[i] = med_h[i+1]; med_s[i] = med_s[i+1]; med_v[i] = med_v[i+1]; med_a[i] = med_a[i+1]; med_b[i] = med_b[i+1]

        sigma = max(15, w / 9)
        base_s = cv.GaussianBlur(med_s.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        base_v = cv.GaussianBlur(med_v.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        base_a = cv.GaussianBlur(med_a.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
        base_b = cv.GaussianBlur(med_b.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)

        sig = (np.maximum(med_s - base_s, 0)
               + 0.28 * np.maximum(base_v - med_v, 0)
               + 0.35 * np.sqrt((med_a - base_a)**2 + (med_b - base_b)**2))
        sig = cv.GaussianBlur(sig.reshape(1, -1), (0, 0), sigmaX=max(2, w / 180)).reshape(-1)
        signals.append(sig)
        med_h_all.append(med_h); med_s_all.append(med_s); med_v_all.append(med_v)

    if not signals:
        return None

    # 多條帶取平均，並用最小值加權，抑制只在一條帶出現的陰影/噪音。
    stack = np.vstack(signals)
    signal = 0.70 * np.mean(stack, axis=0) + 0.30 * np.min(stack, axis=0)
    signal = cv.GaussianBlur(signal.reshape(1, -1), (0, 0), sigmaX=max(2, w / 200)).reshape(-1)

    return {
        "body": body,
        "core_mask": mask[int(0.30*h):max(int(0.30*h)+1, int(0.70*h)), :],
        "signal": signal,
        "med_h": np.mean(np.vstack(med_h_all), axis=0),
        "med_s": np.mean(np.vstack(med_s_all), axis=0),
        "med_v": np.mean(np.vstack(med_v_all), axis=0),
    }

def get_patch_stats(body: np.ndarray, body_h: int, peak_x: int, half: int = 6):
    cx1 = max(0, peak_x - half); cx2 = min(body.shape[1], peak_x + half + 1)
    cy1 = int(0.18 * body_h); cy2 = int(0.82 * body_h)
    patch = body[cy1:cy2, cx1:cx2]
    if patch.size == 0:
        return {"h50": 0.0, "s50": 0.0, "v50": 0.0, "v10": 0.0, "v20": 0.0, "vstd": 0.0, "sstd": 0.0,
                "h_band": 0.0, "s_band": 0.0, "v_band": 0.0, "b_med": 0.0, "g_med": 0.0, "r_med": 0.0, "is_valid": False}
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV).astype(np.float32)
    H = hsv[:, :, 0].reshape(-1); S = hsv[:, :, 1].reshape(-1); V = hsv[:, :, 2].reshape(-1)
    B = patch[:, :, 0].astype(np.float32).reshape(-1); G = patch[:, :, 1].astype(np.float32).reshape(-1); R = patch[:, :, 2].astype(np.float32).reshape(-1)

    h50 = float(np.median(H)); s50 = float(np.median(S)); v50 = float(np.median(V))
    v_med = np.median(V)
    # 實拍偏暗，讓高飽和像素權重更高，避免被灰白本體稀釋。
    salience = 1.25 * S + 0.35 * np.maximum(v_med - V, 0) + 0.15 * np.abs(V - v_med)
    thr = np.percentile(salience, 65)
    m = salience >= thr
    if np.sum(m) < max(5, 0.10 * len(H)):
        m = salience >= np.percentile(salience, 55)
    if np.sum(m) == 0:
        m = np.ones_like(H, dtype=bool)

    h_vals = H[m]
    if np.mean((h_vals <= 8) | (h_vals >= 172)) > 0.30:
        h_adj = h_vals.copy(); h_adj[h_adj >= 172] -= 180
        hb = float(np.median(h_adj));
        if hb < 0: hb += 180
    else:
        hb = float(np.median(h_vals))

    return {
        "h50": h50, "s50": s50, "v50": v50,
        "v10": float(np.percentile(V, 10)), "v20": float(np.percentile(V, 20)),
        "vstd": float(np.std(V)), "sstd": float(np.std(S)),
        "h_band": hb, "s_band": float(np.median(S[m])), "v_band": float(np.median(V[m])),
        "b_med": float(np.median(B[m])), "g_med": float(np.median(G[m])), "r_med": float(np.median(R[m])),
        "is_valid": True,
    }

def white_like(stats: dict) -> bool:
    """
    白色色環在實拍裡常不是純白，而是「低飽和 + 高亮度 + 窄亮帶」。
    這裡刻意要求 v20 也要夠高，避免把普通高光/米色本體誤當白環。
    """
    s = stats["s50"]; v = stats["v50"]; v20 = stats["v20"]
    sb = stats.get("s_band", s); vb = stats.get("v_band", v)
    return (v >= 205 and v20 >= 155 and s <= 78 and sb <= 98 and vb >= 165)

def gray_like(stats: dict) -> bool:
    """
    灰色色環在這組實拍圖會接近「低飽和、偏暗的窄帶」。
    注意它和銀色都低飽和，但銀色通常更亮、更像金屬反光；灰色則是穩定偏暗。
    """
    if white_like(stats):
        return False
    s = stats["s50"]; v = stats["v50"]; v20 = stats["v20"]
    sb = stats.get("s_band", s); vb = stats.get("v_band", v)

    low_sat = (s <= 58 and sb <= 82)
    mid_dark = (65 <= v <= 175 and 45 <= v20 <= 165 and vb <= 170)
    return low_sat and mid_dark

def silver_like(stats: dict) -> bool:
    # 很亮的低飽和區在這組照片多半是 WHITE，不應該先被 SILVER 吃掉。
    # 很暗的低飽和窄帶則更像 GRAY 色碼，不是銀色誤差環。
    if white_like(stats) or gray_like(stats):
        return False
    s = stats["s50"]; v = stats["v50"]; v20 = stats["v20"]
    sb = stats.get("s_band", s)
    return s < 48 and sb < 72 and 125 <= v <= 230 and v20 >= 95

def gold_like(stats: dict) -> bool:
    h = stats["h50"]; s = stats["s50"]; v = stats["v50"]
    hb = stats.get("h_band", h); sb = stats.get("s_band", s)
    # 高飽和的橘/棕色環常落在 h=10~16，不要把它誤當金色。
    if sb > 135 or s > 135:
        return False
    if h > 16 and (hb <= 6 or hb >= 173) and sb >= 145:
        return False
    return (13 <= h <= 32 and 28 <= s <= 125 and v >= 75)

def gold_tolerance_like(stats: dict) -> bool:
    """
    容差金環在實拍會因為陰影變得很髒，局部飽和度可能比一般 gold_like 高。
    但它的 hue 仍偏黃褐，不會像橘環那樣落在偏紅橘的 8~13 附近。
    """
    if white_like(stats) or gray_like(stats):
        return False
    h = stats["h50"]; s = stats["s50"]; v = stats["v50"]; v20 = stats["v20"]
    hb = stats.get("h_band", h); sb = stats.get("s_band", s); vb = stats.get("v_band", v)
    hue_ok = (15 <= h <= 33) or (15 <= hb <= 33)
    sat_ok = (30 <= s <= 195) or (45 <= sb <= 215)
    bright_ok = v >= 75 and v20 >= 55 and vb >= 65
    # 真正橘環通常 h/hb 偏 8~13 且飽和度很高，這種不要當金色。
    orange_red = (h <= 13 or hb <= 13) and (s >= 140 or sb >= 170)
    return hue_ok and sat_ok and bright_ok and not orange_red

def classify_digit_from_stats(stats: dict) -> str:
    h = stats["h50"]; s = stats["s50"]; v = stats["v50"]; v20 = stats["v20"]
    hb = stats.get("h_band", h); sb = stats.get("s_band", s); vb = stats.get("v_band", v)

    if v < 58 or (v20 < 38 and s < 185):
        return "BLACK"
    if white_like(stats):
        return "WHITE"
    if gray_like(stats):
        return "GRAY"
    if 88 < h <= 132 and s > 45:
        return "BLUE"
    if 132 < h <= 170 and s > 35:
        return "VIOLET"
    if 35 < h <= 88 and s > 45:
        return "GREEN"

    # yellow / orange / brown 這幾個在手機自動白平衡下很容易互相飄。
    # 這裡把 YELLOW 門檻拉高，避免金色/米色高光被當成黃色有效數字。
    if 20 <= h <= 42 and s >= 115 and v >= 145:
        return "YELLOW"

    # red / brown / orange
    if h <= 7 or h >= 173:
        return "RED" if (v >= 85 and s >= 50) else "BROWN"
    if 7 < h <= 12:
        # 這組實拍橘環因白平衡/反光會掉到 h≈10~12；用亮度與飽和度和棕色分開。
        if s >= 165 and v >= 140 and sb >= 175:
            return "ORANGE"
        if (hb <= 6 or hb >= 174) and sb >= 90 and vb >= 85:
            return "RED"
        return "BROWN"
    if 12 < h <= 22:
        if (hb <= 6 or hb >= 173) and sb >= 90:
            return "RED"
        # 棕環在實拍中也可能有 h=13~15、s 高，但通常亮度低；
        # 只有又亮又飽和才當 ORANGE。
        if v >= 150 and s >= 145 and sb >= 155:
            return "ORANGE"
        return "BROWN"
    if 22 < h <= 34:
        if s >= 120 and v >= 145:
            return "ORANGE" if h < 26 else "YELLOW"
        return "BROWN"
    return "UNKNOWN"

def classify_multiplier_from_stats(stats: dict) -> str:
    if silver_like(stats):
        return "SILVER"
    if gold_like(stats):
        return "GOLD"
    if white_like(stats):
        return "WHITE"
    if gray_like(stats):
        return "GRAY"

    h = stats["h50"]; s = stats["s50"]; v = stats["v50"]
    hb = stats.get("h_band", h); sb = stats.get("s_band", s); vb = stats.get("v_band", v)

    # 倍率第三環可能是橘色，但不能只因 h=13~15 就判 ORANGE；
    # 棕環常因光線被拍得偏橘，差異主要在亮度與飽和度。
    orange_strong = (
        (8 <= h <= 13 and s >= 160 and v >= 135 and sb >= 170) or
        (11 <= h <= 23 and s >= 155 and v >= 145 and sb >= 165)
    )
    if orange_strong:
        return "ORANGE"

    return classify_digit_from_stats(stats)

def classify_tolerance_from_stats(stats: dict) -> str:
    if silver_like(stats): return "SILVER"
    if gold_tolerance_like(stats): return "GOLD"
    return "UNKNOWN"

# ===================== candidates =====================
def local_find_peaks(signal: np.ndarray, prominence: float, distance: int):
    if find_peaks is not None:
        peaks, props = find_peaks(signal, prominence=prominence, distance=distance)
        return peaks, props
    # simple fallback if scipy is unavailable
    candidates = []
    for i in range(1, len(signal)-1):
        if signal[i] >= signal[i-1] and signal[i] > signal[i+1] and signal[i] >= prominence:
            candidates.append((signal[i], i))
    candidates.sort(reverse=True)
    selected = []
    prominences = []
    for val, idx in candidates:
        if all(abs(idx - j) >= distance for j in selected):
            selected.append(idx); prominences.append(val)
    order = np.argsort(selected)
    return np.array(selected)[order], {"prominences": np.array(prominences)[order]}

def build_white_signal(med_s: np.ndarray, med_v: np.ndarray, w: int) -> np.ndarray:
    """
    專門補白色色環：白環的飽和度低、亮度高，而且是局部窄亮帶。
    不能只看 V，否則本體反光會一堆假峰；所以同時壓制高飽和與過寬高光。
    """
    sigma = max(15, w / 9)
    base_v = cv.GaussianBlur(med_v.reshape(1, -1), (0, 0), sigmaX=sigma).reshape(-1)
    bright = np.maximum(med_v - base_v, 0)

    # s <= 60 幾乎全保留；s 到 95 逐漸歸零。
    low_s_weight = np.clip((95.0 - med_s) / 35.0, 0.0, 1.0)
    white_signal = bright * low_s_weight

    # 白色環通常真的很亮；過暗的米色本體不要納入。
    white_signal[(med_v < 218) | (med_s > 92)] = 0
    white_signal = cv.GaussianBlur(white_signal.reshape(1, -1), (0, 0), sigmaX=max(1.5, w / 260)).reshape(-1)
    return white_signal

def band_box_from_peak(peak_x: int, signal: np.ndarray, core_mask: np.ndarray, body_box, is_tolerance: bool = False):
    x, y, w, h = body_box
    global_low = max(float(np.percentile(signal, 50)), 0.8)
    peak_value = signal[peak_x]
    low_ratio = 0.45 if is_tolerance else 0.33
    low = max(global_low, peak_value * low_ratio)
    left = peak_x
    while left > 0 and signal[left - 1] >= low:
        left -= 1
    right = peak_x
    while right < len(signal) - 1 and signal[right + 1] >= low:
        right += 1
    min_w = max(8, int(0.014 * w))
    max_w = max(min_w + 2, int((0.055 if is_tolerance else 0.075) * w))
    if right - left + 1 < min_w:
        half = min_w // 2; left = max(0, peak_x - half); right = min(len(signal)-1, peak_x + half)
    if right - left + 1 > max_w:
        half = max_w // 2; left = max(0, peak_x - half); right = min(len(signal)-1, peak_x + half)
    # vertical box: use conservative central region, not full mask, to avoid board/slot edges.
    yy1 = int(0.18 * h); yy2 = int(0.82 * h)
    return (x + left, y + yy1, right - left + 1, yy2 - yy1 + 1)

def edge_tolerance_hint(roi: np.ndarray, body_box):
    x, y, w, h = body_box
    body = roi[y:y+h, x:x+w]
    def score_at(i: int):
        stats = get_patch_stats(body, h, i, half=max(4, int(0.012*w)))
        h50, s50, v50 = stats["h50"], stats["s50"], stats["v50"]
        gold_score = (2.2*np.exp(-((h50-18)**2)/(2*7*7)) +
                      1.1*np.exp(-((s50-105)**2)/(2*70*70)) +
                      0.5*np.exp(-((v50-145)**2)/(2*85*85)))
        silver_score = 1.4 if silver_like(stats) else 0.0
        return max(gold_score, silver_score), stats
    inner = max(8, int(0.04*w)); outer_l = int(0.34*w); outer_r = int(0.66*w)
    lb = (-1, None, None)
    for i in range(inner, max(inner+1, outer_l)):
        sc, st = score_at(i)
        if sc > lb[0]: lb = (sc, i, st)
    rb = (-1, None, None)
    for i in range(max(inner, outer_r), w-inner):
        sc, st = score_at(i)
        if sc > rb[0]: rb = (sc, i, st)
    return {"left": {"score": lb[0], "peak": lb[1], "stats": lb[2]},
            "right": {"score": rb[0], "peak": rb[1], "stats": rb[2]}}

def detect_candidates(roi: np.ndarray, body_box, body_mask: np.ndarray):
    built = build_signal_and_profiles(roi, body_box, body_mask)
    if built is None:
        return [], np.zeros(body_box[2], dtype=np.float32), np.zeros((body_box[3], body_box[2]), dtype=np.uint8)
    body = built["body"]; core_mask = built["core_mask"]; signal = built["signal"]
    x, y, w, h = body_box

    prom = max(2.5, float(np.std(signal) * 0.18))
    dist = max(8, w // 22)
    peaks, props = local_find_peaks(signal, prominence=prom, distance=dist)

    # keep more candidates, merge only very close duplicates
    merge_gap = max(10, w // 22)
    groups = []
    for p, pr in zip(peaks, props.get("prominences", np.ones_like(peaks))):
        if not groups or int(p) - groups[-1][-1][0] > merge_gap:
            groups.append([(int(p), float(pr))])
        else:
            groups[-1].append((int(p), float(pr)))
    reps = [max(g, key=lambda t: t[1]) for g in groups]

    candidates = []
    for p, pr in reps:
        stats = get_patch_stats(body, h, p, half=max(5, int(0.015*w)))
        candidates.append({
            "peak": int(p), "prom": float(pr), **stats,
            "digit_color": classify_digit_from_stats(stats),
            "mult_color": classify_multiplier_from_stats(stats),
            "tol_color": classify_tolerance_from_stats(stats),
            "source": "peak",
        })

    # 白色環補點：白環的原本 signal 可能很低，所以另外掃描低飽和窄亮帶。
    white_signal = build_white_signal(built["med_s"], built["med_v"], w)
    white_prom = max(4.0, float(np.std(white_signal) * 0.35))
    white_peaks, white_props = local_find_peaks(white_signal, prominence=white_prom, distance=max(10, w // 18))
    for p, pr in zip(white_peaks, white_props.get("prominences", np.ones_like(white_peaks))):
        p = int(p)
        # 太靠邊常是本體端點或背景反光，不當白環。
        if p < 0.08*w or p > 0.92*w:
            continue
        if any(abs(p - c["peak"]) < max(10, w//35) for c in candidates):
            continue
        stats = get_patch_stats(body, h, p, half=max(5, int(0.014*w)))
        if not white_like(stats):
            continue
        candidates.append({
            "peak": p, "prom": float(max(pr, white_signal[p]) + 18.0), **stats,
            "digit_color": "WHITE",
            "mult_color": "WHITE",
            "tol_color": "UNKNOWN",
            "source": "white_scan",
        })

    # Edge gold/silver補點，實拍容差環不一定有強 peak。
    hints = edge_tolerance_hint(roi, body_box)
    for side in ["left", "right"]:
        hint = hints[side]
        if hint["peak"] is None or hint["score"] < 2.45:
            continue
        p = int(hint["peak"])
        if any(abs(p - c["peak"]) < max(8, w//32) for c in candidates):
            continue
        stats = get_patch_stats(body, h, p, half=max(4, int(0.012*w)))
        tol = classify_tolerance_from_stats(stats)
        if tol not in FOUR_BAND_TOLERANCE_COLORS:
            continue
        candidates.append({
            "peak": p, "prom": float(hint["score"] * 10), **stats,
            "digit_color": classify_digit_from_stats(stats),
            "mult_color": classify_multiplier_from_stats(stats),
            "tol_color": tol,
            "source": "edge_scan",
        })

    candidates = sorted(candidates, key=lambda c: c["peak"])
    ps = [c["peak"] for c in candidates]
    for c in candidates:
        others = [abs(c["peak"] - p) for p in ps if p != c["peak"]]
        c["gap"] = min(others) if others else w

    band_mask = np.zeros((h, w), dtype=np.uint8)
    for c in candidates:
        bx, by, bw, bh = band_box_from_peak(c["peak"], signal, core_mask, body_box, is_tolerance=(c["tol_color"] in FOUR_BAND_TOLERANCE_COLORS))
        band_mask[by-y:by-y+bh, bx-x:bx-x+bw] = 255
    return candidates, signal, band_mask

def color_role_score(c: dict, role: str, w: int) -> float:
    edge = min(c["peak"], w - 1 - c["peak"])
    score = float(c["prom"])
    if role in {"digit1", "digit2"}:
        if c["digit_color"] in {"UNKNOWN", "GOLD", "SILVER"}:
            score -= 80
        else:
            score += 25

        # 金/銀容差候選很常是端帽或反光；除非它是白環補點，不應該拿來當有效數字。
        if c["tol_color"] in FOUR_BAND_TOLERANCE_COLORS and c.get("source") != "white_scan":
            score -= 75

        if c["digit_color"] == "WHITE":
            score += 42
            if c.get("source") == "white_scan":
                score += 18
        if c["digit_color"] == "GRAY":
            score += 52
        if c["digit_color"] == "BROWN":
            score += 10

        if edge < 0.035*w:
            score -= 25

    elif role == "mult":
        if c["mult_color"] == "UNKNOWN":
            score -= 60
        else:
            score += 18
        if c["mult_color"] in {"GOLD", "SILVER"}:
            score += 8
        if c["mult_color"] == "BROWN":
            score += 14
        if c["mult_color"] == "GRAY":
            score += 20
        # 容差金色候選若被拿來當倍率，通常是高光或端帽，稍微扣分。
        if c["tol_color"] in FOUR_BAND_TOLERANCE_COLORS and c["mult_color"] not in {"GOLD", "SILVER"}:
            score -= 20

    elif role == "tol":
        if c["tol_color"] in FOUR_BAND_TOLERANCE_COLORS:
            score += 58
        else:
            score -= 95
        if edge < max(5, 0.025*w):
            score -= 45
        # 太靠本體端點且 prom 很弱的「金色」常是端帽/反光，不是真正色環。
        if edge < 0.09*w and c.get("prom", 0) < 20:
            score -= 35
        score += 0.20 * max(0, 0.28*w - edge)
        score += 0.30 * c.get("gap", 0)
        if c["source"] == "edge_scan":
            score += 8
    return score

def pick_four_bands(roi: np.ndarray, body_box, body_mask: np.ndarray):
    x, y, w, h = body_box
    built = build_signal_and_profiles(roi, body_box, body_mask)
    if built is None:
        return [], [], [], np.zeros(w, dtype=np.float32), np.zeros((h, w), dtype=np.uint8)
    core_mask = built["core_mask"]
    candidates, signal, band_mask = detect_candidates(roi, body_box, body_mask)
    if not candidates:
        return [], [], [], signal, band_mask

    def q(c):
        val = c["prom"]
        if c["digit_color"] != "UNKNOWN": val += 12
        if c["digit_color"] == "WHITE": val += 30
        if c["digit_color"] == "GRAY": val += 35
        if c.get("source") == "white_scan": val += 18
        if c["mult_color"] in {"GOLD", "SILVER"}: val += 8
        if c["tol_color"] in FOUR_BAND_TOLERANCE_COLORS: val += 18
        if c["peak"] < 0.02*w or c["peak"] > 0.98*w: val -= 20
        return val
    candidates = sorted(candidates, key=q, reverse=True)[:12]
    candidates = sorted(candidates, key=lambda c: c["peak"])

    import itertools
    best = None
    for subset in itertools.combinations(candidates, 4):
        subset = sorted(subset, key=lambda c: c["peak"])
        gaps = [subset[i+1]["peak"] - subset[i]["peak"] for i in range(3)]
        if min(gaps) < max(6, w//45):
            continue
        for direction in ["LEFT_TO_RIGHT", "RIGHT_TO_LEFT"]:
            ordered = list(subset) if direction == "LEFT_TO_RIGHT" else list(reversed(subset))
            roles = ["digit1", "digit2", "mult", "tol"]
            score = 0.0; colors = []
            for c, role in zip(ordered, roles):
                score += color_role_score(c, role, w)
                colors.append(c["digit_color"] if role in {"digit1", "digit2"} else c["mult_color"] if role == "mult" else c["tol_color"])
            if colors[0] in {"UNKNOWN", "GOLD", "SILVER"}: score -= 200
            if colors[1] in {"UNKNOWN", "GOLD", "SILVER"}: score -= 200
            if colors[2] == "UNKNOWN": score -= 150
            if colors[3] not in FOUR_BAND_TOLERANCE_COLORS: score -= 250
            tol_edge = min(ordered[-1]["peak"], w - 1 - ordered[-1]["peak"])
            score -= 0.08 * tol_edge
            logical_peaks = [c["peak"] for c in ordered]
            gap_23 = abs(logical_peaks[3] - logical_peaks[2]); gap_01 = abs(logical_peaks[1] - logical_peaks[0])
            if gap_23 > gap_01: score += min(12, 0.035*(gap_23-gap_01))
            if best is None or score > best["score"]:
                best = {"score": score, "ordered": ordered, "colors": colors, "direction": direction}
    if best is None:
        return [], [], [], signal, band_mask
    boxes = []; peaks = []
    for c, role in zip(best["ordered"], ["digit1", "digit2", "mult", "tol"]):
        boxes.append(band_box_from_peak(c["peak"], signal, core_mask, body_box, is_tolerance=(role == "tol")))
        peaks.append(c["peak"])
    return boxes, best["colors"], peaks, signal, band_mask

# ===================== decode / debug =====================
def decode_four_band(color_names: List[str]):
    if len(color_names) != 4:
        raise ValueError(f"四環解碼需要 4 個色環，目前偵測到 {len(color_names)} 個")
    c1, c2, c3, c4 = color_names
    if c1 in {"GOLD", "SILVER"} or c2 in {"GOLD", "SILVER"}:
        raise ValueError(f"前兩個有效數字不能是金或銀，目前得到：{c1}, {c2}")
    if c4 not in FOUR_BAND_TOLERANCE_COLORS:
        raise ValueError(f"四環容差環目前限定為金或銀，目前得到：{c4}")
    b1, b2, b3, b4 = COLOR_DB[c1], COLOR_DB[c2], COLOR_DB[c3], COLOR_DB[c4]
    if b1.digit is None or b2.digit is None: raise ValueError(f"前兩環必須是有效數字色碼，目前得到：{c1}, {c2}")
    if b3.multiplier is None: raise ValueError(f"第三環必須是倍率色碼，目前得到：{c3}")
    if b4.tolerance is None: raise ValueError(f"第四環必須是誤差色碼，目前得到：{c4}")
    return (10 * b1.digit + b2.digit) * b3.multiplier, b4.tolerance

def draw_debug_image(roi, body_box, band_boxes, band_colors, resistance_text, direction_text):
    dbg = roi.copy()
    x, y, w, h = body_box
    cv.rectangle(dbg, (x, y), (x+w, y+h), (255, 255, 0), 2)
    cv.putText(dbg, "BODY", (x, max(18, y-8)), FONT, 0.6, (255,255,0), 2, cv.LINE_AA)
    for i, (box, cname) in enumerate(zip(band_boxes, band_colors), start=1):
        bx, by, bw, bh = box
        color = COLOR_DB[cname].debug_bgr if cname in COLOR_DB else (255,255,255)
        cv.rectangle(dbg, (bx, by), (bx+bw, by+bh), color, 2)
        cv.putText(dbg, f"{i}:{cname}", (bx, max(18, by-6)), FONT, 0.55, color, 2, cv.LINE_AA)
    cv.putText(dbg, resistance_text, (10, 28), FONT, 0.75, (0,255,255), 2, cv.LINE_AA)
    cv.putText(dbg, direction_text, (10, 56), FONT, 0.65, (0,255,255), 2, cv.LINE_AA)
    return dbg

def analyze_resistor_image(image_path: str, save_path: str = "annotated_result.jpg", strict: bool = False):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"讀不到圖片：{image_path}")
    roi, roi_mask, roi_info = detect_resistor_roi(image)
    body_box, body_mask = detect_body_box(roi, roi_mask)
    band_boxes, band_colors, peaks, signal, band_mask = pick_four_bands(roi, body_box, body_mask)

    ok = True; message = "成功"; value_ohm = None; tolerance = None
    direction = "RIGHT_TO_LEFT" if (len(peaks) == 4 and peaks[0] > peaks[-1]) else "LEFT_TO_RIGHT" if len(peaks)==4 else "UNKNOWN"
    resistance_text = "Resistance: decode failed"
    if len(band_boxes) == 4 and len(band_colors) == 4 and len(peaks) == 4:
        try:
            value_ohm, tolerance = decode_four_band(band_colors)
            tolerance_ascii = tolerance.replace("±", "+/-") if tolerance else ""
            resistance_text = f"Resistance: {format_resistance_ascii(value_ohm)} {tolerance_ascii}"
        except Exception as e:
            ok = False; message = f"找到 4 個色環，但解碼失敗：{e}"; resistance_text = f"Decode failed: {e}"
    else:
        ok = False; message = f"偵測到 {len(band_boxes)} 個候選色環，不是 4 個"; resistance_text = f"Detected {len(band_boxes)} band(s), not 4"

    debug_img = draw_debug_image(roi, body_box, band_boxes, band_colors, resistance_text, f"Order: {direction}")
    status_color = (0,255,0) if ok else (0,0,255)
    cv.putText(debug_img, f"Status: {'OK' if ok else 'FAIL'}", (10,84), FONT, 0.65, status_color, 2, cv.LINE_AA)
    img_message = "OK" if ok else "Detect/decode failed"
    cv.putText(debug_img, img_message, (10,112), FONT, 0.5, status_color, 1, cv.LINE_AA)
    cv.imwrite(save_path, debug_img)

    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    base = os.path.splitext(os.path.basename(save_path))[0]
    cv.imwrite(os.path.join(save_dir, f"{base}_roi.jpg"), roi)
    cv.imwrite(os.path.join(save_dir, f"{base}_body_mask.jpg"), body_mask)
    cv.imwrite(os.path.join(save_dir, f"{base}_band_mask.jpg"), band_mask)
    return {
        "ok": ok, "message": message, "body_box": body_box,
        "band_boxes": band_boxes, "peaks": peaks, "band_colors": band_colors,
        "resistance_ohm": value_ohm, "resistance_text": format_resistance(value_ohm) if value_ohm is not None else None,
        "tolerance": tolerance, "direction": direction, "saved_image": save_path, "roi_info": roi_info,
    }

if __name__ == "__main__":
    test_paths = ["test56.jpg", "test56_2.jpg", "test390.jpg", "testresistor.jpg"]
    for image_path in test_paths:
        if not os.path.exists(image_path):
            continue
        out_name = f"annotated_{os.path.splitext(os.path.basename(image_path))[0]}.jpg"
        result = analyze_resistor_image(image_path, out_name, strict=False)
        print("\n==============================")
        print(f"Image: {image_path}")
        for k, v in result.items():
            if k == "roi_info":
                print(f"{k}: (略，內含 rotated_image)")
            else:
                print(f"{k}: {v}")
