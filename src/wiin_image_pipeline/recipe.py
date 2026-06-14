"""Recipe planner: decide *when not to touch the background*.

There is no off-the-shelf background-removal model that wins on every catalog
shot. The biggest practical gain came not from a better model but from skipping
removal when the background is already clean, because any neural net only makes
a clean shot worse, and the heavy steps (removal, upscale) are what cost time.

We measure a couple of cheap properties of the source and pick a mode.
"""
from __future__ import annotations
import numpy as np
from PIL import Image

# thresholds tuned on catalog product shots
EDGE_WHITE_SKIP = 0.75      # >75% white in the border -> background already clean
CORNER_STD_SKIP = 20.0      # low corner variance -> uniform background
EDGE_WHITE_THRESHOLD = 0.40 # mostly-white -> safe OpenCV threshold is enough
RETENTION_MIN = 0.25        # if neural removal keeps <25% pixels, it overcut -> fall back
UPSCALE_BELOW = 1800        # upscale only if the short side is smaller than this


def source_props(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    b = 10
    edges = np.concatenate([
        arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
        arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3),
    ])
    edge_whiteness = float(np.mean(np.all(edges > 235, axis=1)))
    cs = 50
    corners = np.concatenate([
        arr[:cs, :cs].reshape(-1, 3), arr[:cs, -cs:].reshape(-1, 3),
        arr[-cs:, :cs].reshape(-1, 3), arr[-cs:, -cs:].reshape(-1, 3),
    ])
    corner_std = float(np.mean(np.std(corners, axis=0)))
    return {"edge_whiteness": round(edge_whiteness, 3),
            "corner_std": round(corner_std, 2),
            "short_side": min(h, w)}


def plan(image: Image.Image, trusted_source: bool = False) -> dict:
    """Return the processing recipe for one source image."""
    p = source_props(image)
    if p["edge_whiteness"] > EDGE_WHITE_SKIP and p["corner_std"] < CORNER_STD_SKIP:
        rembg_mode = "skip"          # clean already, do not touch
    elif p["edge_whiteness"] > EDGE_WHITE_THRESHOLD or trusted_source:
        rembg_mode = "threshold"     # cheap OpenCV threshold, CPU
    else:
        rembg_mode = "neural"        # BiRefNet on GPU, the expensive path
    return {
        "props": p,
        "rembg_mode": rembg_mode,
        "upscale": p["short_side"] < UPSCALE_BELOW,
        "retention_min": RETENTION_MIN,
    }


def retention_ok(pixels_before: int, pixels_after: int,
                 retention_min: float = RETENTION_MIN) -> bool:
    """After neural removal, if too few pixels remain it overcut the product."""
    if pixels_before <= 0:
        return False
    return (pixels_after / pixels_before) >= retention_min
