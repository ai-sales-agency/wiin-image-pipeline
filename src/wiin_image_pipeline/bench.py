"""Per-stage benchmark harness: cold load + warm average.

Usage:
    python -m wiin_image_pipeline.bench path/to/image.jpg [path/to/RealESRGAN_x4plus.pth]

Measures each stage independently so you can see where time goes on your GPU.
On a single RTX 4070 Super we got roughly: YOLO ~14 ms, rembg ~1.2-2.4 s,
Real-ESRGAN 4x ~9.5 s, pHash ~3 ms.
"""
from __future__ import annotations
import sys
import time
import json
import statistics as st
from PIL import Image


def _timeit(fn, warm: int = 5) -> dict:
    t0 = time.perf_counter(); fn(); cold = time.perf_counter() - t0
    ts = []
    for _ in range(warm):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return {"cold_s": round(cold, 3), "warm_avg_s": round(st.mean(ts), 3),
            "warm_min_s": round(min(ts), 3), "runs": warm}


def run(image_path: str, esrgan_model_path: str | None = None) -> dict:
    import numpy as np
    import cv2
    out: dict = {"image": image_path, "stages": {}}
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        out["gpu"] = None

    img = Image.open(image_path).convert("RGB")
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    out["image_size"] = img.size

    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        out["stages"]["yolo_crop"] = _timeit(
            lambda: yolo.predict(bgr, device=0, verbose=False, conf=0.25))
    except Exception as e:
        out["stages"]["yolo_crop"] = {"error": str(e)[:160]}

    try:
        from rembg import new_session, remove
        try:
            sess = new_session("birefnet-general"); used = "birefnet-general"
        except Exception:
            sess = new_session("u2net"); used = "u2net"
        raw = open(image_path, "rb").read()
        r = _timeit(lambda: remove(raw, session=sess), warm=3); r["model"] = used
        out["stages"]["rembg"] = r
    except Exception as e:
        out["stages"]["rembg"] = {"error": str(e)[:160]}

    if esrgan_model_path:
        try:
            from .process import upscale_4x
            small = cv2.resize(bgr, (512, 512))
            out["stages"]["realesrgan_4x"] = _timeit(
                lambda: upscale_4x(small, esrgan_model_path), warm=3)
        except Exception as e:
            out["stages"]["realesrgan_4x"] = {"error": str(e)[:160]}

    try:
        import imagehash
        out["stages"]["phash"] = _timeit(lambda: imagehash.phash(img), warm=20)
    except Exception as e:
        out["stages"]["phash"] = {"error": str(e)[:160]}

    try:
        import torch
        if torch.cuda.is_available():
            out["vram_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m wiin_image_pipeline.bench IMAGE [ESRGAN_MODEL.pth]")
        raise SystemExit(1)
    model = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(run(sys.argv[1], model), ensure_ascii=False, indent=2))
