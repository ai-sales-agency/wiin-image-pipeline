"""Dedup groups the same photo (which arrives from several sources) and keeps
distinct products. Deterministic tests for the pHash + Hamming logic and the
reference-bonus thresholds.

Note: pHash is robust to re-compression and resize on real, detailed photos.
Here we use exact copies and clearly distinct products so the test is stable
regardless of how a particular hash implementation rounds flat synthetic art.
"""
from PIL import Image, ImageDraw
from wiin_image_pipeline.dedup import phash, hamming, dedup, reference_bonus, NEAR_REF, FAR_REF


def _device(body, screen=(20, 22, 28)):
    im = Image.new("RGB", (800, 800), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((250, 150, 550, 650), radius=30, fill=body)
    d.rounded_rectangle((280, 190, 520, 610), radius=18, fill=screen)
    return im


def _save(im, path):
    im.save(path)
    return str(path)


def test_identical_copy_is_duplicate(tmp_path):
    base = _device((60, 60, 65))
    a = _save(base, tmp_path / "a.png")
    b = _save(base.copy(), tmp_path / "b.png")            # same photo, second source
    assert hamming(phash(a), phash(b)) == 0


def test_dedup_drops_duplicates_keeps_distinct(tmp_path):
    a = _save(_device((60, 60, 65)), tmp_path / "a.png")
    b = _save(_device((60, 60, 65)), tmp_path / "b.png")  # duplicate of a
    c = _save(_device((210, 40, 40), screen=(250, 250, 250)), tmp_path / "c.png")  # different
    kept = dedup([a, b, c])
    assert a in kept and c in kept and b not in kept


def test_reference_bonus_signs(tmp_path):
    h = phash(_save(_device((60, 60, 65)), tmp_path / "x.png"))
    assert reference_bonus(h, h) == 0.5                   # identical -> bonus
    far = phash(_save(_device((210, 40, 40), screen=(250, 250, 250)), tmp_path / "y.png"))
    assert reference_bonus(h, far) <= 0.0                 # very different -> penalty/none
    assert NEAR_REF < FAR_REF
