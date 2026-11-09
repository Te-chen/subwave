"""ffmpeg rendering and font discovery."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def pick_font(candidates: list[str] | None = None) -> str:
    """Return a font family name that ffmpeg/libass can resolve."""
    if candidates is None:
        candidates = [
            ("SimHei", r"C:\Windows\Fonts\simhei.ttf"),
            ("Microsoft YaHei", r"C:\Windows\Fonts\msyh.ttc"),
            ("Noto Sans CJK SC", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            ("PingFang SC", "/System/Library/Fonts/PingFang.ttc"),
        ]
    for name, path in candidates:
        if Path(path).exists():
            return name
    on_windows = sys.platform.startswith("win")
    return "SimHei" if on_windows else "Noto Sans CJK SC"


def render(source: Path, ass_path: Path, target: Path, *,
           crf: int = 20, preset: str = "medium",
           audio_bitrate: str = "192k", encoder: str = "libx264") -> None:
    """Burn the ASS file into the video. Runs ffmpeg from the ASS directory so the
    filter argument stays a bare filename (a Windows drive colon breaks parsing)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # absolute paths: ffmpeg runs from the ASS directory, so relative source
    # and target paths would resolve against the wrong base
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(Path(source).resolve()),
        "-vf", f"ass='{ass_path.name}'",
        "-c:v", encoder, "-crf", str(crf), "-preset", preset,
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(Path(target).resolve()),
    ]
    result = subprocess.run(cmd, cwd=str(ass_path.parent), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n" + result.stderr.decode("utf-8", "replace")[-1500:])
