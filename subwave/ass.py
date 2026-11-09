"""ASS subtitle generation."""
from __future__ import annotations

from pathlib import Path

from .geometry import Geometry, font_size_for, margins_for
from .layout import Cue


def _ts(ms: int) -> str:
    ms = max(0, int(ms))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{ms // 10:02d}"


def build(geom: Geometry, cues: list[Cue], font: str = "SimHei") -> str:
    size = font_size_for(geom.width)
    margin_lr, margin_v, _ = margins_for(geom)
    outline = max(1.4, round(size * 0.05, 1))
    shadow = max(0.8, round(size * 0.03, 1))

    header = f"""[Script Info]
Title: translated subtitle
ScriptType: v4.00+
PlayResX: {geom.width}
PlayResY: {geom.height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: SUB,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H78000000,0,0,0,0,100,100,0.5,0,1,{outline},{shadow},2,{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = [
        f"Dialogue: 0,{_ts(c.start_ms)},{_ts(c.end_ms)},SUB,,0,0,0,,"
        f"{{\\fn{font}\\fs{size}}}{c.text}"
        for c in cues
    ]
    return header + "\n".join(events) + "\n"


def write(geom: Geometry, cues: list[Cue], path: Path, font: str = "SimHei") -> Path:
    path.write_text(build(geom, cues, font), encoding="utf-8")
    return path
