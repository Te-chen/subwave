"""Video probing and subtitle layout geometry.

No aspect-ratio assumption: every video is measured independently.

Key idea -- most short-form and many long-form videos are letterboxed, leaving a
band of pure black above or below the artwork. That band is where a translated
caption belongs: it does not cover the picture and does not fight a burned-in
original caption.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Geometry:
    width: int
    height: int
    content_top: int
    content_bottom: int

    @property
    def band_top(self) -> int:
        return self.content_bottom

    @property
    def band_height(self) -> int:
        return max(0, self.height - self.content_bottom)

    @property
    def has_bottom_band(self) -> bool:
        return self.band_height >= self.height * 0.06


def probe_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def detect_content_box(path: Path) -> tuple[int, int]:
    """Detect the non-black artwork box via cropdetect. Returns (top, bottom)."""
    width, height = probe_size(path)
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-ss", "2", "-t", "3", "-i", str(path),
         "-vf", "cropdetect=limit=24:round=2:reset=0", "-f", "null", "-"],
        capture_output=True,
    )
    log = result.stderr.decode("utf-8", "replace")
    hits = [ln for ln in log.split("\n") if "crop=" in ln]
    if not hits:
        return 0, int(height * 0.78)

    match = re.search(r"crop=(\d+):(\d+):(\d+):(\d+)", hits[-1])
    if not match:
        return 0, int(height * 0.78)
    y, h = int(match.group(4)), int(match.group(2))
    # guard against a degenerate detection
    if h < height * 0.25:
        return 0, int(height * 0.78)
    return max(0, y), min(height, y + h)


def analyze(path: Path) -> Geometry:
    width, height = probe_size(path)
    top, bottom = detect_content_box(path)
    return Geometry(width, height, top, bottom)


def font_size_for(width: int) -> int:
    """Scale type with canvas width so lines hold a constant character count."""
    return max(16, int(round(width * 0.0426)))


def margins_for(geom: Geometry) -> tuple[int, int, int]:
    """Return (margin_lr, margin_v, max_px) for the caption block."""
    margin_lr = int(geom.width * 0.055)

    if geom.has_bottom_band:
        # centre-ish inside the black band, biased slightly above its middle
        margin_v = max(int(geom.band_height * 0.40) + 30, int(geom.height * 0.03))
    else:
        # no band: sit above the bottom edge, clear of platform UI overlays
        margin_v = max(int(geom.height * 0.10), 40)

    max_px = geom.width - margin_lr * 2
    return margin_lr, margin_v, max_px


def describe(geom: Geometry) -> dict:
    return {
        "size": f"{geom.width}x{geom.height}",
        "content_bottom": geom.content_bottom,
        "band_height": geom.band_height,
        "uses_band": geom.has_bottom_band,
        "font_size": font_size_for(geom.width),
    }


def ffprobe_json(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,width,height", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}
