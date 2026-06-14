# wiin-image-pipeline

Build clean catalog product shots automatically: find candidate images in public
sources, throw away junk with cheap rules, score the rest with a vision model,
and process the winners locally on a single consumer GPU.

Extracted and anonymized from a real project that collected product photos for a
large device catalog. The modules are meant to be dropped into your own pipeline.
Tested on CPU in CI, the GPU stages verified on an RTX 4070 Super. Built by
[wiin.agency](https://wiin.agency).

## What you actually get

Concrete, hard-won pieces you can reuse today:

- **A planner that decides when *not* to remove the background** (`recipe.py`).
  This is the headline win. On our hardware the upscale step alone is ~9.5 s per
  image, so skipping background removal and upscale on already-clean shots saves
  multiples of GPU time, not percentages. The decision is two cheap measurements
  (edge whiteness, corner variance) plus a retention guard.
- **A scoring scheme that kills the "crane-as-tablet" class of bug** (`score.py`).
  A vision model with no reference scores only background cleanliness, not
  identity, and will accept a photo of a bird as a tablet with a 10/10. The fix
  is grounding: pass a reference image and force the prompt to verify category,
  brand and model. The prompt is in the repo.
- **Calibrated thresholds for the boring-but-critical parts**: pHash dedup
  distances (`dedup.py`), CPU pre-filters that keep junk out of the expensive
  vision call (`filters.py`), candidate search with an early stop to save API
  credits (`collect.py`).
- **A benchmark harness** (`bench.py`) so you can measure your own GPU.

Everything except neural background removal is solved and tested. Background
removal on hard shots is an open problem, and instead of hiding that we show
exactly where it breaks in the [failure gallery](docs/examples/). That honesty is
the point: you will know which shots are safe to automate and which are not.

## Measured benchmarks (single NVIDIA GeForce RTX 4070 Super, 12 GB)

Image 1200×1200, warm run, average:

| Step | Cold load | Per image (warm) |
|---|---|---|
| YOLOv8n crop | 0.76 s | ~14 ms |
| rembg (BiRefNet-general) | 0.82 s | ~1.2 to 2.4 s |
| Real-ESRGAN 4x (512→2048, 400px tiles) | — | ~9.5 s |
| pHash | — | ~3 ms |

Peak VRAM on a single step was ~1.5 GB of 12. The bottleneck is time on the heavy
steps, not memory, which is exactly why the planner pays off.

## Layout

```
src/wiin_image_pipeline/
  collect.py   Serper image search: query cascade (GTIN → MPN → name), junk + URL dedup
  filters.py   cheap CPU rules before any AI (watermark/OCR, retail box, collage, cropped)
  dedup.py     perceptual hash (pHash) + Hamming distance, calibrated thresholds
  recipe.py    "when not to touch the background": edge-whiteness / corner-std gating
  process.py   YOLO crop → rembg (per recipe) → Real-ESRGAN → standardize → normalize
  score.py     vision scoring with an explicit identity-verification prompt
  bench.py     per-stage benchmark harness (cold load + warm average)
examples/
  make_samples.py  generate synthetic, license-clean test images
  run_one.py       end-to-end on one image
docs/examples/     before/after gallery, including a real failure case
tests/             CPU-only tests for the gating, dedup and filter thresholds
```

## Install and run

```bash
pip install -e .            # core (CPU). Add the GPU stages with: pip install -e ".[gpu]"
cp .env.example .env        # your own SERPER_API_KEY / vision key
python examples/make_samples.py docs/examples/regenerated
python examples/run_one.py docs/examples/regenerated/textured_bg.png
pytest -q                   # CPU tests
```

The local processing path (recipe → threshold background → standardize →
normalize) runs without a GPU. The neural background and 4x upscale stages need
the optional `[gpu]` extras.

## Reusing a single module

You do not need the whole thing. The two most reused pieces:

```python
from wiin_image_pipeline.recipe import plan
plan(pil_image)        # -> {"rembg_mode": "skip|threshold|neural", "upscale": bool, ...}

from wiin_image_pipeline.score import score
score("candidate.jpg", "Apple iPad Air", reference_path="reference.jpg")
```

## Contributing

The interesting open question is background removal on hard shots (dark devices,
reflective screens, clutter). If you have solved it, or have shots that break the
gallery cases, open an issue or PR. Real failing examples are especially welcome.

## License

MIT. See [LICENSE](LICENSE).
