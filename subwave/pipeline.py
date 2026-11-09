"""The pipeline: discover videos, fetch captions, translate, render.

Design notes that matter for correctness:

* Sources are treated as read-only. Output goes to a sibling tree keyed by
  creator then language, so two channels with similar titles never mix.
* Output filenames keep the original name verbatim with a ``<lang> - `` prefix.
  The ``[VIDEOID]`` inside the original name is the caption lookup key, which
  means the pipeline does not need any separate manifest to stay consistent.
* Every stage is idempotent. Existing valid outputs are skipped, so a run that
  died halfway resumes without redoing (or re-billing) finished work.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import ImageFont

from . import ass as ass_mod
from . import captions, geometry, layout, render
from .translate import Translator

VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}
ID_IN_NAME = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


@dataclass
class Item:
    path: Path
    video_id: str | None
    out_name: str


@dataclass
class Result:
    item: Item
    ok: bool
    cues: int = 0
    sentences: int = 0
    size: str = ""
    reason: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def video_id_from_name(name: str) -> str | None:
    match = ID_IN_NAME.search(name)
    return match.group(1) if match else None


def discover(src: Path, lang: str, recursive: bool = False) -> list[Item]:
    src = Path(src)
    pattern = "**/*" if recursive else "*"
    items = []
    for path in sorted(src.glob(pattern)):
        if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
            continue
        items.append(Item(path, video_id_from_name(path.name), f"{lang} - {path.name}"))
    return items


def output_root(src: Path, out_dir: Path | None, lang: str) -> Path:
    """<out>/<creator>/<lang> -- creator is the source directory name."""
    base = Path(out_dir) if out_dir else Path("out")
    return base / Path(src).resolve().name / lang


# --------------------------------------------------------------------------
# caption acquisition
# --------------------------------------------------------------------------

CAPTION_LANGS = ("en-orig", "en", "en-US", "en-GB")


def fetch_captions(video_id: str, work: Path, langs=CAPTION_LANGS) -> Path | None:
    """Download platform captions as json3. Returns the file path or None.

    Looks in the work directory first, so a caption fetched once (or supplied
    from an earlier batch) is never re-requested -- important because the
    platform rate-limits caption endpoints aggressively.
    """
    work.mkdir(parents=True, exist_ok=True)
    existing = sorted(work.glob(f"{video_id}.*.json3"))
    if existing:
        return existing[0]

    cmd = [
        "yt-dlp", "--no-warnings", "--skip-download",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", ",".join(langs), "--sub-format", "json3",
        "--sleep-requests", "0.3",
        "-o", str(work / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True)
    got = sorted(work.glob(f"{video_id}.*.json3"))
    if got:
        return got[0]
    if result.returncode != 0:
        tail = result.stderr.decode("utf-8", "replace").strip().split("\n")[-1]
        if "not a bot" in tail or "Sign in" in tail:
            raise RuntimeError(
                "platform requires authentication (bot check). "
                "Export cookies with `--cookies-from-browser` or provide captions "
                "in the source directory.")
        raise RuntimeError(f"caption fetch failed: {tail[:200]}")
    return None


def local_caption_for(video: Path) -> Path | None:
    """Sidecar subtitles next to the video (works for non-YouTube sources too)."""
    for suffix in (".json3", ".vtt", ".srt"):
        candidate = video.with_suffix(suffix)
        if candidate.exists():
            return candidate
        for found in video.parent.glob(f"{video.stem}.*{suffix}"):
            return found
    return None


def seed_captions(video_ids: list[str], source: Path, work: Path) -> int:
    """Copy already-downloaded captions into the work directory.

    Lets a batch reuse captions gathered earlier instead of re-hitting an
    endpoint that rate-limits or demands authentication.
    """
    if not source or not Path(source).is_dir():
        return 0
    work.mkdir(parents=True, exist_ok=True)
    copied = 0
    for video_id in video_ids:
        if any(work.glob(f"{video_id}.*.json3")):
            continue
        for found in Path(source).glob(f"{video_id}.*.json3"):
            shutil.copy2(found, work / found.name)
            copied += 1
            break
    return copied


# --------------------------------------------------------------------------
# measurement helper (PIL, real glyph widths)
# --------------------------------------------------------------------------

def make_measure(size: int, font_file: Path | None):
    if font_file and Path(font_file).exists():
        handle = ImageFont.truetype(str(font_file), size)
    else:
        handle = ImageFont.load_default()

    def measure(text: str) -> int:
        box = handle.getbbox(text)
        return box[2] - box[0]

    return measure


def font_file_for(font_name: str) -> Path | None:
    candidates = {
        "SimHei": [r"C:\Windows\Fonts\simhei.ttf"],
        "Microsoft YaHei": [r"C:\Windows\Fonts\msyh.ttc"],
        "Noto Sans CJK SC": [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        ],
        "PingFang SC": ["/System/Library/Fonts/PingFang.ttc"],
    }
    for path in candidates.get(font_name, []):
        if Path(path).exists():
            return Path(path)
    fallback = Path(r"C:\Windows\Fonts\simhei.ttf")
    return fallback if fallback.exists() else None


# --------------------------------------------------------------------------
# one video
# --------------------------------------------------------------------------

def process_video(item: Item, *, out_root: Path, work_dir: Path,
                  translator: Translator, lang: str, font: str,
                  crf: int, skip_existing: bool = True,
                  caption_lang_filter: tuple[str, ...] = CAPTION_LANGS) -> Result:
    target = out_root / item.out_name

    if skip_existing and target.exists() and target.stat().st_size > 10_000:
        return Result(item, True, reason="exists", size=_size_of(target))

    # 1. captions -----------------------------------------------------------
    work_dir.mkdir(parents=True, exist_ok=True)
    cap_dir = work_dir / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    cap_path = local_caption_for(item.path)
    if cap_path is None:
        if not item.video_id:
            return Result(item, False, reason="no caption file and no video id")
        try:
            cap_path = fetch_captions(item.video_id, cap_dir, caption_lang_filter)
        except RuntimeError as exc:
            return Result(item, False, reason=str(exc))
    if cap_path is None:
        return Result(item, False, reason="no captions available")

    words = captions.parse_any(cap_path)
    if not words:
        return Result(item, False, reason="caption file had no words")

    sentences = captions.build_sentences(words)

    # 2. translation --------------------------------------------------------
    translations = translator.translate([s.text for s in sentences])
    missing = sum(1 for t in translations if not t)

    # 3. geometry + layout --------------------------------------------------
    geom = geometry.analyze(item.path)
    size_px = geometry.font_size_for(geom.width)
    measure = make_measure(size_px, font_file_for(font))
    _, _, max_px = geometry.margins_for(geom)
    max_chars = max(8, int(max_px / size_px * 1.02))

    cues = []
    for sentence, translated in zip(sentences, translations):
        if translated:
            cues.extend(layout.layout(sentence, translated, measure, max_px, max_chars))
    cues = layout.dedup(cues)
    if not cues:
        return Result(item, False, reason="no cues produced")

    # 4. render -------------------------------------------------------------
    stage = work_dir / "ass"
    stage.mkdir(parents=True, exist_ok=True)
    # fixed safe name: titles contain quotes/colons/emoji that break the ffmpeg
    # filter argument, and CSS-style escaping is not portable across shells
    safe_stem = item.video_id or f"video{abs(hash(item.path.name)) % 10**8}"
    ass_path = ass_mod.write(geom, cues, stage / f"{safe_stem}.ass", font).resolve()

    try:
        render.render(item.path, ass_path, target, crf=crf)
    except RuntimeError as exc:
        return Result(item, False, reason=str(exc)[:400])

    notes = []
    if missing:
        notes.append(f"{missing} untranslated sentence(s)")
    if not geom.has_bottom_band:
        notes.append("no letterbox band: caption sits above bottom edge")

    return Result(item, True, cues=len(cues), sentences=len(sentences),
                  size=_size_of(target), notes=notes)


def _size_of(path: Path) -> str:
    mb = path.stat().st_size / (1024 * 1024)
    return f"{mb:.1f} MB"


# --------------------------------------------------------------------------
# run whole directory
# --------------------------------------------------------------------------

def run(src: Path, *, lang: str, out_dir: Path | None = None,
        limit: int | None = None, force: bool = False, dry_run: bool = False,
        api_key: str | None = None, model: str | None = None,
        batch_size: int = 20, crf: int = 20, font: str | None = None,
        caption_langs: tuple[str, ...] = CAPTION_LANGS,
        captions_dir: Path | None = None,
        verbose: bool = True) -> list[Result]:
    src = Path(src)
    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")

    out_root = output_root(src, out_dir, lang)
    work_dir = out_root.parent / "_work"
    items = discover(src, lang)
    if limit:
        items = items[:limit]

    if captions_dir:
        copied = seed_captions([i.video_id for i in items if i.video_id],
                               Path(captions_dir), work_dir / "captions")
        if verbose and copied:
            print(f"seeded {copied} caption file(s) from {captions_dir}")

    font = font or render.pick_font()
    translator = Translator(api_key=api_key, model=model,
                            cache_dir=work_dir / "cache", dst_lang=lang)

    if verbose:
        print(f"source   : {src}")
        print(f"output   : {out_root}")
        print(f"videos   : {len(items)}")
        print(f"font     : {font}")
        if dry_run:
            print("mode     : dry run\n")

    results: list[Result] = []
    for index, item in enumerate(items, 1):
        if verbose:
            print(f"[{index}/{len(items)}] {item.path.name}", flush=True)
        if dry_run:
            cap = local_caption_for(item.path)
            has = "sidecar" if cap else ("remote" if item.video_id else "none")
            print(f"    captions: {has} | out: {item.out_name}", flush=True)
            continue

        result = process_video(item, out_root=out_root, work_dir=work_dir,
                               translator=translator, lang=lang, font=font,
                               crf=crf, skip_existing=not force,
                               caption_lang_filter=caption_langs)
        results.append(result)
        if verbose:
            if result.ok:
                extra = f" ({'; '.join(result.notes)})" if result.notes else ""
                print(f"    ok  cues={result.cues} {result.size}{extra}", flush=True)
            else:
                print(f"    FAILED  {result.reason}", flush=True)

    if verbose and results:
        ok = sum(1 for r in results if r.ok)
        print(f"\n{ok}/{len(results)} produced")
        print(f"translation: {translator.stats['requests']} request(s), "
              f"{translator.stats['cached']} cache hit(s)")

    report = out_root / "report.json"
    if not dry_run and results:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(
            [{"video": r.item.path.name, "out": r.item.out_name, "ok": r.ok,
              "cues": r.cues, "sentences": r.sentences, "size": r.size,
              "reason": r.reason, "notes": r.notes} for r in results],
            ensure_ascii=False, indent=2), encoding="utf-8")

    return results


def prepare(src: Path, *, verbose: bool = True,
            langs: tuple[str, ...] = CAPTION_LANGS) -> dict:
    """Fetch captions up front and build an inventory. No rendering."""
    src = Path(src)
    items = discover(src, lang="xx")
    work_dir = output_root(src, None, "xx").parent / "_work"
    inventory = []
    for index, item in enumerate(items, 1):
        cap = local_caption_for(item.path)
        status = "sidecar" if cap else "missing"
        if cap is None and item.video_id:
            try:
                cap = fetch_captions(item.video_id, work_dir, langs)
                status = "fetched" if cap else "unavailable"
            except RuntimeError as exc:
                status = f"error: {exc}"[:80]
        inventory.append({
            "video": item.path.name, "video_id": item.video_id,
            "captions": str(cap) if cap else None, "status": status,
        })
        if verbose:
            print(f"[{index}/{len(items)}] {status:12} {item.path.name[:60]}", flush=True)

    out = work_dir / "inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"count": len(inventory), "inventory": str(out)}
