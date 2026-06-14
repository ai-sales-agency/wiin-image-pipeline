"""The recipe planner is the core idea: decide when NOT to touch the background.

These tests pin the behavior so the gating thresholds stay documented and stable.
"""
from PIL import Image, ImageDraw
from wiin_image_pipeline.recipe import plan, retention_ok, source_props


def _clean_white():
    im = Image.new("RGB", (1000, 1000), (255, 255, 255))
    ImageDraw.Draw(im).rounded_rectangle((300, 200, 700, 800), radius=40, fill=(60, 60, 65))
    return im


def _dark_textured():
    im = Image.new("RGB", (1000, 1000), (38, 40, 44))
    ImageDraw.Draw(im).rounded_rectangle((300, 200, 700, 800), radius=40, fill=(18, 18, 22))
    return im


def test_clean_white_is_skipped():
    # value: a clean shot must never be sent through a neural net (it only degrades it)
    r = plan(_clean_white())
    assert r["rembg_mode"] == "skip"
    assert r["props"]["edge_whiteness"] > 0.75


def test_dark_background_goes_neural():
    # value: only the genuinely hard backgrounds pay the GPU cost
    r = plan(_dark_textured())
    assert r["rembg_mode"] == "neural"


def test_source_props_shape():
    p = source_props(_clean_white())
    assert set(p) == {"edge_whiteness", "corner_std", "short_side"}
    assert p["short_side"] == 1000


def test_retention_guard():
    # value: if neural removal keeps too few pixels it overcut the product -> reject
    assert retention_ok(1000, 500) is True
    assert retention_ok(1000, 100) is False
    assert retention_ok(0, 0) is False
