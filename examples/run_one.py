"""End-to-end on a single image: plan a recipe, process, save the result.

    python examples/run_one.py path/to/source.jpg [path/to/RealESRGAN_x4plus.pth]

Collection (collect.py) and scoring (score.py) need API keys from .env; this
example only runs the local image processing so it works offline on a CPU box
(the neural background and upscale stages need their optional GPU deps).
"""
import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wiin_image_pipeline.recipe import plan          # noqa: E402
from wiin_image_pipeline.process import process_one  # noqa: E402
from PIL import Image                                  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("usage: python examples/run_one.py SOURCE.jpg [ESRGAN_MODEL.pth]")
        raise SystemExit(1)
    src = sys.argv[1]
    esrgan = sys.argv[2] if len(sys.argv) > 2 else None

    recipe = plan(Image.open(src).convert("RGB"))
    print("recipe:", recipe)

    result = process_one(src, esrgan_model_path=esrgan)
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "result.jpg")
    cv2.imwrite(out_path, result)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
