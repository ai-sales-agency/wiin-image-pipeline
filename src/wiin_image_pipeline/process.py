"""Local processing stages: crop, background, upscale, standardize, normalize.

Heavy deps (ultralytics, rembg, realesrgan) are imported lazily so the rest of
the package works without a GPU. The background step follows the recipe from
recipe.py instead of always running a neural net.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image

from .recipe import plan, retention_ok

TARGET = 2048
UPSCALE_BELOW = 1800


def yolo_crop(bgr: np.ndarray, conf: float = 0.25, pad: float = 0.10):
    """Crop the main object. Lazy YOLOv8n. Skip if the largest box is tiny (logo)."""
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    res = model.predict(bgr, conf=conf, verbose=False, device=0)
    boxes = res[0].boxes
    if boxes is None or len(boxes) == 0:
        return bgr
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    i = int(areas.argmax())
    x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
    h, w = bgr.shape[:2]
    if (x2 - x1) * (y2 - y1) < 0.15 * h * w:    # logo, not the product
        return bgr
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]


def remove_background(bgr: np.ndarray, mode: str):
    """mode from recipe: skip | threshold | neural."""
    if mode == "skip":
        return bgr
    if mode == "threshold":
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(g, 230, 255, cv2.THRESH_BINARY_INV)
        out = bgr.copy()
        out[mask == 0] = (255, 255, 255)
        return out
    # neural (BiRefNet via rembg) with a retention guard
    from rembg import new_session, remove
    try:
        session = new_session("birefnet-general")
    except Exception:
        session = new_session("u2net")
    before = bgr.shape[0] * bgr.shape[1]
    rgba = remove(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), session=session)
    rgba = np.asarray(rgba)
    kept = int((rgba[:, :, 3] > 10).sum()) if rgba.shape[-1] == 4 else before
    if not retention_ok(before, kept):           # overcut -> fall back to threshold
        return remove_background(bgr, "threshold")
    white = np.full((rgba.shape[0], rgba.shape[1], 3), 255, np.uint8)
    a = (rgba[:, :, 3:4] / 255.0)
    comp = (rgba[:, :, :3] * a + white * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)


def upscale_4x(bgr: np.ndarray, model_path: str):
    """Real-ESRGAN x4, tiled to avoid OOM on 12 GB."""
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    up = RealESRGANer(scale=4, model_path=model_path, model=net, half=True, tile=400)
    out, _ = up.enhance(bgr, outscale=4)
    return out


def standardize(bgr: np.ndarray, size: int = TARGET, margin: float = 0.05) -> np.ndarray:
    """Center the product on a white square canvas with a small margin."""
    h, w = bgr.shape[:2]
    inner = int(size * (1 - 2 * margin))
    scale = min(inner / w, inner / h)
    rw, rh = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - rh) // 2, (size - rw) // 2
    canvas[y0:y0 + rh, x0:x0 + rw] = resized
    return canvas


def normalize_white(bgr: np.ndarray) -> np.ndarray:
    """Per-channel white balance using the 95th percentile as the white point."""
    out = bgr.astype(np.float32)
    for c in range(3):
        wp = np.percentile(out[:, :, c], 95)
        if wp > 0:
            out[:, :, c] = np.clip(out[:, :, c] * (255.0 / wp), 0, 255)
    return out.astype(np.uint8)


def process_one(image_path: str, esrgan_model_path: str | None = None,
                trusted_source: bool = False) -> np.ndarray:
    """End-to-end for one image, following the recipe."""
    pil = Image.open(image_path).convert("RGB")
    recipe = plan(pil, trusted_source=trusted_source)
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    bgr = yolo_crop(bgr)
    bgr = remove_background(bgr, recipe["rembg_mode"])
    if recipe["upscale"] and esrgan_model_path:
        bgr = upscale_4x(bgr, esrgan_model_path)
    bgr = standardize(bgr)
    bgr = normalize_white(bgr)
    return bgr
