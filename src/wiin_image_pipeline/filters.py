"""Cheap CPU rules that run before any vision model.

The point is to never spend a vision call on something a sub-second rule can
reject: watermarks, retail-box shots, collages, heavily cropped products, and
obviously broken sources. Each check returns a reason string or None.
"""
from __future__ import annotations
import numpy as np
import cv2

MIN_DIMENSION = 900
BLUR_THRESHOLD = 80.0          # Laplacian variance below this is mushy
DARK_RATIO_MAX = 0.55
UNIFORM_RATIO_MAX = 0.85
EDGE_DENSITY_MAX = 0.25


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def validate_raw(bgr: np.ndarray) -> str | None:
    """Fast sanity checks on the raw candidate."""
    h, w = bgr.shape[:2]
    if min(h, w) < MIN_DIMENSION:
        return "TOO_SMALL"
    g = _gray(bgr)
    if float(np.mean(g < 60)) > DARK_RATIO_MAX:
        return "TOO_DARK"
    # dominant-color share via coarse histogram
    hist = cv2.calcHist([g], [0], None, [16], [0, 256]).flatten()
    if hist.max() / hist.sum() > UNIFORM_RATIO_MAX:
        return "UNIFORM_COLOR"
    edges = cv2.Canny(g, 100, 200)
    if float(np.mean(edges > 0)) > EDGE_DENSITY_MAX:
        return "HIGH_EDGE_DENSITY"      # spec sheets, collages
    if cv2.Laplacian(g, cv2.CV_64F).var() < BLUR_THRESHOLD:
        return "BLURRY"
    return None


def is_cropped_product(bgr: np.ndarray, white_thresh: int = 230) -> str | None:
    """Product touching 3+ edges is cut off."""
    g = _gray(bgr)
    b = 6
    sides = [g[:b], g[-b:], g[:, :b], g[:, -b:]]
    touching = sum(1 for s in sides if float(np.mean(s < white_thresh)) > 0.5)
    return "CROPPED_PRODUCT" if touching >= 3 else None


def has_text_overlay(bgr: np.ndarray, expected_words: set[str] | None = None,
                     reader=None) -> str | None:
    """Watermark / banner detection.

    Watermarks are often light-on-silver, so boost contrast (CLAHE) before OCR.
    Pass an easyocr.Reader as `reader` to enable; without it this is a no-op so
    the dependency stays optional.
    """
    if reader is None:
        return None
    expected = expected_words or set()
    g = _gray(bgr)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(g)
    for item in reader.readtext(enhanced, detail=0):
        word = str(item).strip().lower()
        if len(word) > 5 and word not in expected and word not in _GENERIC_WHITELIST:
            return f"TEXT_OVERLAY:{word}"
    return None


_GENERIC_WHITELIST = {
    "settings", "search", "today", "battery", "signal", "messages",
    "photos", "camera", "music", "safari", "phone", "mail", "calendar",
}


def screen(bgr: np.ndarray, expected_words: set[str] | None = None,
           ocr_reader=None) -> str | None:
    """Run all cheap rules, return the first rejection reason or None."""
    for check in (
        validate_raw(bgr),
        is_cropped_product(bgr),
        has_text_overlay(bgr, expected_words, ocr_reader),
    ):
        if check:
            return check
    return None
