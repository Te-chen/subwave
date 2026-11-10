"""Quality gate. Checks things that make an output unusable, not style."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import geometry, pipeline


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


# Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
_DIALOGUE = re.compile(r"^Dialogue:\s*-?\d+,([\d:.]+),([\d:.]+),(.*)$")
_ASS_TIME = re.compile(r"^(\d+):(\d{1,2}):(\d{1,2})\.(\d{1,2})$")


def _parse_ts(value: str) -> int | None:
    match = _ASS_TIME.match(value.strip())
    if not match:
        return None
    h, m, s, cs = match.groups()
    # ASS centiseconds are two digits; be tolerant of one
    centi = int(cs.ljust(2, "0")[:2])
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + centi * 10


def parse_ass(path: Path) -> list[tuple[int, int, str]]:
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = _DIALOGUE.match(line.strip())
        if not match:
            continue
        start = _parse_ts(match.group(1))
        end = _parse_ts(match.group(2))
        if start is None or end is None:
            continue
        text = re.sub(r"\{[^}]*\}", "", match.group(3)).strip()
        out.append((start, end, text))
    return out


def _bright_pixels(path: Path, height: int, width: int, at_s: int) -> int:
    import subprocess
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(at_s), "-i", str(path),
         "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True).stdout
    if not raw:
        return 0
    top = int(height * 0.78)
    hits = 0
    for y in range(top, height, 2):
        row = y * width
        for x in range(0, width, 2):
            if row + x < len(raw) and raw[row + x] > 200:
                hits += 1
    return hits


def _caption_is_visible(path: Path, height: int, width: int, duration: float) -> tuple[bool, int]:
    """Sample several points across the video and look for caption pixels.

    A single sample is unreliable: a gap between cues can land exactly on the
    chosen timestamp and produce a false negative. Captions are dense in these
    videos, so if any of the samples finds them, the burn worked.
    """
    if duration <= 0:
        return False, 0
    points = [duration * r for r in (0.08, 0.25, 0.45, 0.65, 0.85)]
    best = 0
    for point in points:
        best = max(best, _bright_pixels(path, height, width, int(point)))
        if best > 30:
            return True, best
    return best > 30, best


def verify(src: Path, *, lang: str, out_dir: Path | None = None,
           names_file: Path | None = None,
           verbose: bool = True) -> list[Check]:
    src = Path(src)
    # outputs live at <base>/<creator>/<lang>; verify accepts either the base
    lang_dir = pipeline.output_root(src, out_dir, lang)
    if not lang_dir.is_dir():
        raise SystemExit(f"output directory not found: {lang_dir}")

    checks: list[Check] = []

    names = pipeline.load_names(names_file) if names_file else None
    sources = pipeline.discover(src, lang, names=names)

    # with an explicit name map, outputs are matched by that map; otherwise by
    # the "<lang> - <source name>" convention
    if names:
        expected = {i.out_name: i.path for i in sources if i.video_id in names}
        outputs = sorted(p for p in lang_dir.glob("*.mp4") if p.name in expected)
        stray = sorted(p.name for p in lang_dir.glob("*.mp4") if p.name not in expected)
        checks.append(Check("no unexpected outputs", not stray,
                            f"{len(stray)} not in name map: {stray[:3]}"))
    else:
        expected = {}
        outputs = sorted(p for p in lang_dir.glob("*.mp4") if p.name.startswith(f"{lang} - "))
        missing_source = [o.name for o in outputs
                          if not (src / o.name[len(lang) + 3:]).exists()]
        checks.append(Check("every output has a source", not missing_source,
                            f"{len(missing_source)} unmatched"))

    checks.append(Check("outputs were produced", len(outputs) > 0,
                        f"{len(outputs)} of {len(sources)} source video(s)"))

    for out in outputs:
        info = geometry.ffprobe_json(out)
        streams = info.get("streams") or []
        if not streams:
            # truncated or missing moov atom -- a partial write, not a style issue
            checks.append(Check(f"readable {out.name[:32]}", False,
                                f"ffprobe found no streams ({out.stat().st_size} bytes)"))
            continue

        kinds = [s.get("codec_type") for s in streams]
        checks.append(Check(f"streams {out.name[:32]}",
                            "video" in kinds and "audio" in kinds, str(kinds)))

        # duration must match the source video it came from
        if names:
            source_path = expected.get(out.name)
        else:
            source_path = src / out.name[len(lang) + 3:]
        if source_path and Path(source_path).exists():
            src_info = geometry.ffprobe_json(source_path)
            d_out = float(info.get("format", {}).get("duration", 0) or 0)
            d_src = float(src_info.get("format", {}).get("duration", 0) or 0)
            checks.append(Check(f"duration {out.name[:32]}", abs(d_out - d_src) < 1.5,
                                f"{d_out:.1f} vs {d_src:.1f}"))

            w = int(streams[0]["width"])
            h = int(streams[0]["height"])
            visible, hits = _caption_is_visible(out, h, w, d_out)
            checks.append(Check(f"caption visible {out.name[:32]}", visible, f"{hits}px"))

    # subtitle files live in the work area: empty text, overlaps, runaway durations
    ass_dir = lang_dir.parent / "_work" / "ass"
    for ass_path in sorted(ass_dir.glob("*.ass") if ass_dir.is_dir() else []):
        cues = parse_ass(ass_path)
        if not cues:
            checks.append(Check(f"ass readable {ass_path.name[:32]}", False, "no events"))
            continue
        empty = sum(1 for _, _, t in cues if not t)
        overlaps = sum(1 for a, b in zip(cues, cues[1:]) if b[0] < a[1] - 30)
        runaway = sum(1 for s, e, _ in cues if e - s > 9000)
        checks.append(Check(f"ass content {ass_path.name[:32]}", empty == 0, f"{empty} empty"))
        checks.append(Check(f"ass overlap {ass_path.name[:32]}", overlaps == 0, f"{overlaps}"))
        checks.append(Check(f"ass duration {ass_path.name[:32]}", runaway == 0, f"{runaway} >9s"))

    if verbose:
        print("=== checks ===")
        for check in checks:
            flag = "PASS" if check.ok else "FAIL"
            detail = f"  [{check.detail}]" if check.detail and not check.ok else ""
            print(f"  {flag}  {check.name}{detail}")
        passed = sum(1 for c in checks if c.ok)
        print(f"\n{passed}/{len(checks)} passed")

    return checks
