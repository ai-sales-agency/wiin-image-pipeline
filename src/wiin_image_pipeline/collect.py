"""Candidate collection via Serper Google Images.

Query cascade goes from exact to broad. Serper has a dedicated images endpoint.
Pricing (2026): from $1 / 1000 credits on the starter pack, cheaper at volume,
2500 free credits to evaluate, credits valid 6 months.

The real-world catch we hit: a multi-source cascade degrades over time as
scrapers go stale, so keep an honest fallback and stop early once you have
enough trusted candidates instead of burning credits.
"""
from __future__ import annotations
import os
import time
from urllib.parse import urlparse
import requests

SERPER_IMAGES_URL = "https://google.serper.dev/images"
JUNK_HINTS = ("logo", "favicon", "icon", "sprite", "avatar", "placeholder")
MIN_SIDE = 900  # hard reject smaller than this on the short side


def build_queries(name: str, gtin: str | None = None, mpn: str | None = None) -> list[str]:
    """From exact (barcode) to broad (name)."""
    q: list[str] = []
    if gtin:
        q.append(gtin)
    if mpn:
        q.append(f"{mpn} product photo")
    q.append(f"{name} product photo white background")
    return q


def _norm_url(url: str) -> str:
    p = urlparse(url)
    return f"{p.netloc.lower()}{p.path}"


def search_images(query: str, num: int = 10, gl: str = "us", hl: str = "en") -> list[dict]:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise RuntimeError("SERPER_API_KEY is not set (see .env.example)")
    r = requests.post(
        SERPER_IMAGES_URL,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "gl": gl, "hl": hl, "num": num},
        timeout=30,
    )
    r.raise_for_status()
    return r.json().get("images", [])


def collect(name: str, gtin: str | None = None, mpn: str | None = None,
            target: int = 10, rate_limit_s: float = 0.3) -> list[dict]:
    """Run the query cascade, drop junk, dedup by normalized URL, stop early."""
    seen: set[str] = set()
    out: list[dict] = []
    for query in build_queries(name, gtin, mpn):
        for img in search_images(query):
            url = img.get("imageUrl") or img.get("link") or ""
            if not url:
                continue
            low = url.lower()
            if any(h in low for h in JUNK_HINTS):
                continue
            if min(int(img.get("imageWidth", 0)), int(img.get("imageHeight", 0))) and \
               min(int(img.get("imageWidth", 0)), int(img.get("imageHeight", 0))) < MIN_SIDE:
                continue
            nu = _norm_url(url)
            if nu in seen:
                continue
            seen.add(nu)
            out.append({"url": url, "query": query,
                        "w": img.get("imageWidth"), "h": img.get("imageHeight")})
        if len(out) >= target:           # early stop, save credits
            break
        time.sleep(rate_limit_s)
    return out[:target]
