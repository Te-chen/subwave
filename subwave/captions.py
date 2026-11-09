"""Platform caption parsing: json3 / vtt / srt -> word timeline -> sentences.

The messy part of this project lives here. Captions from video platforms are not
clean sentence lists; they are progressively revealed fragments.

YouTube json3 specifics this module handles:
  * ``aAppend`` events carry only a newline -- they mark a line reveal, not text.
  * Content events hold per-word segments; ``tOffsetMs`` is relative to the
    event start. Some segments omit it and must be interpolated.
  * A sentence is routinely split across events, so joining must re-insert
    spaces that the event split removed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NOISE = re.compile(r"\[(?:music|applause|laughter|sound|inaudible|silence)[^\]]*\]", re.I)
SENT_END = re.compile(r"[.?!。？！]$")
# do not treat common abbreviations as sentence ends
ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|vs|etc|No|Jr|Sr|Prof|Inc|Ltd|Co)\.$", re.I)


@dataclass
class Word:
    t: int
    w: str


@dataclass
class Sentence:
    start_ms: int
    end_ms: int
    text: str
    words: list[Word] = field(default_factory=list)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _timeline_for_event(event: dict) -> list[int]:
    """Absolute ms for each segment, interpolating the ones without tOffsetMs."""
    segs = event.get("segs") or []
    base = int(event.get("tStartMs") or 0)
    dur = int(event.get("dDurationMs") or 0)

    times: list[int | None] = [None] * len(segs)
    for i, seg in enumerate(segs):
        if "tOffsetMs" in seg:
            times[i] = base + int(seg["tOffsetMs"])

    if times and times[0] is None:
        times[0] = base
    if times and times[-1] is None:
        times[-1] = base + dur if dur else times[0]

    i = 0
    while i < len(times):
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < len(times) and times[j] is None:
            j += 1
        lo = times[i - 1] if i > 0 else base
        hi = times[j] if j < len(times) else lo
        span = j - i + 1
        for k in range(i, j):
            times[k] = lo + int((hi - lo) * (k - i + 1) / span)
        i = j
    return [int(t) for t in times]


def parse_json3(path: Path) -> list[Word]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    words: list[Word] = []
    for event in data.get("events", []):
        if event.get("aAppend"):
            continue
        segs = event.get("segs") or []
        if not segs:
            continue
        for seg, t in zip(segs, _timeline_for_event(event)):
            raw = str(seg.get("utf8", "")).replace("\n", " ").replace(">>", " ")
            raw = NOISE.sub(" ", raw)
            for piece in raw.split():
                words.append(Word(t, piece))
    return words


_VTT_TIME = re.compile(
    r"(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})[.,](\d{1,3})")


def _to_ms(h: str | None, m: str, s: str, frac: str) -> int:
    return (int(h or 0) * 3600 + int(m) * 60 + int(s)) * 1000 + int(frac.ljust(3, "0"))


def parse_vtt(path: Path) -> list[Word]:
    """VTT/SRT have cue-level timing only; split each cue into timed words."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    words: list[Word] = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue
        m = _VTT_TIME.search(lines[idx])
        if not m:
            continue
        start = _to_ms(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _to_ms(m.group(5), m.group(6), m.group(7), m.group(8))
        body = " ".join(lines[idx + 1:])
        body = re.sub(r"<[^>]+>", "", body)
        body = NOISE.sub(" ", body)
        toks = body.split()
        if not toks:
            continue
        step = max(1, (end - start) // len(toks))
        for k, tok in enumerate(toks):
            words.append(Word(start + k * step, tok))
    return words


def parse_any(path: Path) -> list[Word]:
    suffix = Path(path).suffix.lower()
    if suffix == ".json3":
        return parse_json3(path)
    if suffix in (".vtt", ".srt"):
        return parse_vtt(path)
    raise ValueError(f"unsupported caption format: {path}")


# --------------------------------------------------------------------------
# sentence assembly
# --------------------------------------------------------------------------

def join_words(words: list[Word]) -> str:
    text = " ".join(w.w for w in words)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return re.sub(r"\s+", " ", text).strip()


def build_sentences(words: list[Word], max_dur_ms: int = 15000) -> list[Sentence]:
    """Aggregate words into sentence units that carry their real time span."""
    out: list[Sentence] = []
    cur: list[Word] = []

    def flush() -> None:
        nonlocal cur
        if cur:
            text = join_words(cur)
            if text:
                out.append(Sentence(cur[0].t, cur[-1].t + 420, text, list(cur)))
        cur = []

    for word in words:
        cur.append(word)
        text = join_words(cur)
        if SENT_END.search(text) and not ABBREV.search(text):
            flush()
        elif cur[-1].t - cur[0].t >= max_dur_ms and len(cur) > 4:
            flush()
    flush()

    for a, b in zip(out, out[1:]):
        if a.end_ms > b.start_ms - 30:
            a.end_ms = max(a.start_ms + 400, b.start_ms - 30)
    for s in out:
        if s.end_ms <= s.start_ms:
            s.end_ms = s.start_ms + 800
    return out
