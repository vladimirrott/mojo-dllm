#!/usr/bin/env python3
"""Build the wordmark and social preview SVGs from assets/logo/mojo-dllm.svg.

The logo is inlined rather than referenced, because GitHub (and most image
viewers) will not load an <image> inside an SVG served through <img>.
Rasterize afterwards with a headless browser (see assets/README.md).
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGO = (ROOT / "assets/logo/mojo-dllm.svg").read_text()

_m = re.search(r"<defs>(.*?)</defs>", LOGO, re.S)
if _m is None:
    raise SystemExit("assets/logo/mojo-dllm.svg has no <defs> block")
defs = _m.group(1)
body = LOGO.split("</defs>", 1)[1].rsplit("</svg>", 1)[0]


def logo_group(x: float, y: float, size: float, prefix: str) -> tuple[str, str]:
    d = defs.replace('id="', f'id="{prefix}').replace("url(#", f"url(#{prefix}")
    b = body.replace("url(#", f"url(#{prefix}")
    s = size / 256.0
    return d, f'<g transform="translate({x} {y}) scale({s})">{b}</g>'


def wordmark() -> str:
    d, g = logo_group(10, 10, 180, "l")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 200" width="720" height="200" role="img" aria-labelledby="t">
  <title id="t">mojo-dllm</title>
  <defs>{d}
    <linearGradient id="word" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFB45A"/><stop offset="1" stop-color="#FF6B1A"/></linearGradient>
  </defs>
  <rect width="720" height="200" rx="32" fill="#0B0F13"/>
  {g}
  <text x="210" y="112" font-family="JetBrains Mono, Fira Code, Menlo, Consolas, monospace" font-size="62" font-weight="700" fill="#F0EFE9">mojo<tspan fill="url(#word)">-dllm</tspan></text>
  <text x="212" y="152" font-family="Inter, Helvetica, Arial, sans-serif" font-size="22" fill="#5EEAD4">diffusion language models, in Mojo</text>
</svg>
"""


def social() -> str:
    # GitHub's repo card template keeps everything 80 px from each edge, so
    # all content sits in x 80..1200, y 80..560 (checked by check_social.py).
    d, g = logo_group(120, 180, 280, "s")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 640" width="1280" height="640" role="img" aria-labelledby="t">
  <title id="t">mojo-dllm: diffusion language models, in Mojo</title>
  <defs>{d}
    <linearGradient id="word" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFB45A"/><stop offset="1" stop-color="#FF6B1A"/></linearGradient>
  </defs>
  <rect width="1280" height="640" fill="#0B0F13"/>
  {g}
  <text x="450" y="292" font-family="JetBrains Mono, Fira Code, Menlo, Consolas, monospace" font-size="84" font-weight="700" fill="#F0EFE9">mojo<tspan fill="url(#word)">-dllm</tspan></text>
  <text x="454" y="346" font-family="Inter, Helvetica, Arial, sans-serif" font-size="30" fill="#5EEAD4">Local diffusion-LLM inference, written in Mojo</text>
  <text x="454" y="400" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" fill="#9AA4AE">LLaDA-8B and Dream-7B from GGUF files</text>
  <text x="454" y="434" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" fill="#9AA4AE">No Python at runtime · CPU first</text>
</svg>
"""


def main() -> None:
    (ROOT / "assets/logo/wordmark.svg").write_text(wordmark())
    (ROOT / "assets/social-preview.svg").write_text(social())
    print("wrote assets/logo/wordmark.svg and assets/social-preview.svg")


if __name__ == "__main__":
    main()
