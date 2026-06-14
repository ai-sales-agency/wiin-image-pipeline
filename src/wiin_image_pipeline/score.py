"""Vision scoring with grounding.

The expensive lesson: a vision model with no reference photo scores only
background cleanliness, not identity, and will confidently accept a crane bird
as a tablet. The fix is grounding. Force the prompt to verify category, brand
and model, and pass a reference image when you have one (e.g. by GTIN).

Provider-agnostic: any OpenAI-compatible chat endpoint with vision works
(set VISION_API_BASE / VISION_API_KEY / VISION_MODEL).
"""
from __future__ import annotations
import os
import json
import base64
import requests

ACCEPT_THRESHOLD = 7  # score >= 7 -> accept

SYSTEM = (
    "You verify product catalog photos. Judge IDENTITY first, not background. "
    "If the image shows a different object category (animal, sculpture, website "
    "screenshot, logo, promo banner) or a similar product in a different color or "
    "generation, REJECT it regardless of how clean the background is."
)

PROMPT = (
    "Product expected: {name}.\n"
    "Return strict JSON: {{\"is_this_product\": bool, \"angle\": "
    "\"front|back|side|3-quarter|top|detail\", \"score\": 0-10, \"reason\": str}}.\n"
    "score 10 = clean white background, centered, sharp, correct product. "
    "score 0 = not this product at all."
)


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _image_part(path: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{_b64(path)}"}}


def score(image_path: str, product_name: str, reference_path: str | None = None) -> dict:
    base = os.environ.get("VISION_API_BASE", "https://api.mistral.ai/v1")
    key = os.environ.get("VISION_API_KEY")
    model = os.environ.get("VISION_MODEL", "pixtral-12b-latest")
    if not key:
        raise RuntimeError("VISION_API_KEY is not set (see .env.example)")

    content = [{"type": "text", "text": PROMPT.format(name=product_name)}]
    if reference_path:
        content.append({"type": "text", "text": "Reference photo of the expected product:"})
        content.append(_image_part(reference_path))
    content.append({"type": "text", "text": "Candidate to judge:"})
    content.append(_image_part(image_path))

    r = requests.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": content}]},
        timeout=60,
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1]) if start >= 0 else {"score": 0, "reason": text[:200]}
    data["accepted"] = bool(data.get("is_this_product")) and int(data.get("score", 0)) >= ACCEPT_THRESHOLD
    return data
