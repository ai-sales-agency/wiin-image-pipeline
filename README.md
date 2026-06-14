# wiin-image-pipeline

Reference pipeline for building clean catalog product shots automatically: find candidate images in public sources, throw away junk with cheap rules, score the rest with a vision model, and process the winners locally on a single consumer GPU.

This is an extracted, anonymized reference from a real project (collecting product photos for a large device catalog). It is meant to be read and reused, not as a turnkey product. Built and maintained by [wiin.agency](https://wiin.agency).

## Honest status

Two parts work well and save a lot of manual labor:

- **Collection + scoring.** Multi-source search (Serper Google Images first), then a vision model that verifies *what* is on the image, not just whether the background is clean.
- **Cheap pre-filters.** Rule-based checks (watermark/OCR, retail box, collage, cropped) run on CPU in under a second and keep junk out of the expensive vision step.

One part is still an open problem:

- **Background removal.** Off-the-shelf models did not solve it on hard shots (dark devices on dark backgrounds, reflective screens). What helped was not picking yet another model, it was deciding *when not to touch the background at all* (see `recipe.py`). A follow-up will cover the post-processing once it is solved.

## Why it is useful

If you have a similar task, you get ready-to-use thresholds, a scoring scheme with grounding, and a planner that skips heavy GPU steps when they are not needed. The planner matters: on our hardware the upscale step dominates total time, so skipping it on already-good images saves multiples, not percentages.

## Measured benchmarks (single NVIDIA GeForce RTX 4070 Super, 12 GB)

Image 1200×1200, warm run, average:

| Step | Cold load | Per image (warm) |
|---|---|---|
| YOLOv8n crop | 0.76 s | ~14 ms |
| rembg (BiRefNet-general) | 0.82 s | ~1.2 to 2.4 s |
| Real-ESRGAN 4x (512→2048, 400px tiles) | — | ~9.5 s |
| pHash | — | ~3 ms |

Peak VRAM on a single step was about 1.5 GB out of 12. The bottleneck is time on the heavy steps (upscale, background removal), not memory.

## Layout

```
src/wiin_image_pipeline/
  collect.py   Serper image search: query cascade (GTIN → MPN → name), junk + URL dedup
  filters.py   cheap CPU rules before any AI (watermark/OCR, retail box, collage, cropped)
  dedup.py     perceptual hash (pHash) + Hamming distance
  recipe.py    "when not to touch the background": edge-whiteness / corner-std gating
  process.py   YOLO crop → rembg (per recipe) → Real-ESRGAN → standardize → normalize
  score.py     vision scoring with an explicit identity-verification prompt
  bench.py     per-stage benchmark harness (cold load + warm average)
examples/
  run_one.py   end-to-end on one image
```

## Quick start

```bash
pip install -r requirements.txt           # core; GPU stages are optional, see requirements.txt
cp .env.example .env                       # put your own SERPER_API_KEY / vision key
python examples/run_one.py path/to/image.jpg
```

## A note on the vision step

The single most expensive mistake we made: a vision model with no reference photo scores only background cleanliness, not identity. It once accepted a photo of a crowned crane as a tablet with a 10/10. The fix is grounding: feed it a reference image by GTIN and force the prompt to verify category, brand and model, not "does this look clean". See `score.py`.

## License

MIT. See [LICENSE](LICENSE).
