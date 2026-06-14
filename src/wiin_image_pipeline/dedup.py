"""Perceptual-hash dedup and reference matching.

The same product photo arrives from several sources in different sizes and
re-compressions, so a byte hash is useless. We use pHash + Hamming distance.

Empirical thresholds (catalog product shots):
  <= 4   same image (minor crop/resize)   -> drop as duplicate
  <= 8   very close to a reference         -> ranking bonus
  > 32   different product or color        -> ranking penalty
"""
from __future__ import annotations
from PIL import Image
import imagehash

DUP_MAX = 4
NEAR_REF = 8
FAR_REF = 32


def phash(path_or_image) -> imagehash.ImageHash:
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    return imagehash.phash(img.convert("RGB"))


def hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    return a - b


def dedup(paths: list[str], dup_max: int = DUP_MAX) -> list[str]:
    """Keep the first of each near-identical group."""
    kept: list[tuple[str, imagehash.ImageHash]] = []
    for p in paths:
        try:
            h = phash(p)
        except Exception:
            continue
        if all(hamming(h, kh) > dup_max for _, kh in kept):
            kept.append((p, h))
    return [p for p, _ in kept]


def reference_bonus(candidate_hash, reference_hash) -> float:
    """Score adjustment when a GTIN reference image is available."""
    d = hamming(candidate_hash, reference_hash)
    if d <= NEAR_REF:
        return 0.5
    if d > FAR_REF:
        return -0.3
    return 0.0
