"""Cheap CPU rules keep junk out of the expensive vision step.

Pins the rejection thresholds so they stay documented.
"""
import numpy as np
from wiin_image_pipeline.filters import validate_raw, is_cropped_product, screen


def _solid(h, w, color):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = color
    return img


def test_too_small_is_rejected():
    assert validate_raw(_solid(200, 200, (200, 200, 200))) == "TOO_SMALL"


def test_uniform_placeholder_is_rejected():
    # a flat gray card is a placeholder, not a product
    assert validate_raw(_solid(1000, 1000, (200, 200, 200))) in {"UNIFORM_COLOR", "BLURRY"}


def test_too_dark_is_rejected():
    assert validate_raw(_solid(1000, 1000, (10, 10, 10))) == "TOO_DARK"


def test_cropped_product_detected():
    # product fills the frame and touches all edges -> cut off
    img = _solid(1000, 1000, (40, 40, 40))
    assert is_cropped_product(img) == "CROPPED_PRODUCT"


def test_clean_white_with_object_passes_crop_check():
    img = _solid(1000, 1000, (255, 255, 255))
    img[300:700, 300:700] = (50, 50, 55)                  # centered object, edges white
    assert is_cropped_product(img) is None


def test_screen_returns_first_reason_or_none():
    assert screen(_solid(200, 200, (128, 128, 128))) == "TOO_SMALL"
