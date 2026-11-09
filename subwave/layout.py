"""Turn a translated sentence into timed display cues.

Two problems this solves:

1. Timing. Chinese and English clause counts rarely match by luck -- but they
   often do match, because punctuation survives translation. When they match,
   clause i inherits English clause i's real time span, keeping the Chinese in
   step with what the original speaker is saying. Otherwise fall back to
   character-proportional distribution across the sentence.

2. Wrapping. Width is measured in pixels with the actual font, not estimated
   from character count. CJK and Latin glyphs differ hugely in width, so a
   character budget alone produces overflowing lines on some canvases and
   half-empty lines on others.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .captions import Sentence

EN_CLAUSE = re.compile(r"(?<=[,;:])\s+")
ZH_CLAUSE = re.compile(r"(?<=[，；：、])")
HAS_TEXT = re.compile(r"[\w\u3400-\u4dbf\u4e00-\u9fff]")
BREAK_CHARS = "，；：、,;:"


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    text: str


Measure = Callable[[str], int]


def en_clauses(sentence: Sentence) -> list[tuple[int, int]]:
    """Time spans of the English clauses inside one sentence."""
    words = sentence.words
    if not words:
        return []
    spans: list[tuple[int, int]] = []
    start = words[0].t
    for i, word in enumerate(words):
        if word.w.endswith((",", ";", ":")) and i < len(words) - 1:
            spans.append((start, words[i + 1].t))
            start = words[i + 1].t
    spans.append((start, sentence.end_ms))
    return spans


def zh_clauses(text: str) -> list[str]:
    parts = [p.strip() for p in ZH_CLAUSE.split(text) if p.strip()]
    # drop fragments that carry no actual text (a bare "。" is not a cue)
    parts = [p for p in parts if HAS_TEXT.search(p)]
    return parts or ([text.strip()] if text.strip() else [])


def split_to_width(text: str, measure: Measure, max_px: int, max_chars: int) -> list[str]:
    """Break text into pieces that fit both the pixel and character budgets."""
    text = text.strip()
    if not text:
        return []
    if measure(text) <= max_px and len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    rest = text
    while rest:
        if measure(rest) <= max_px and len(rest) <= max_chars:
            pieces.append(rest)
            break

        cut = min(len(rest) - 1, max_chars)
        while cut > 1 and measure(rest[:cut]) > max_px:
            cut -= 1

        # prefer breaking at punctuation, but not so early the line looks empty
        floor = max(1, int(cut * 0.55))
        for idx in range(cut - 1, floor - 1, -1):
            if rest[idx] in BREAK_CHARS:
                cut = idx + 1
                break

        head, tail = rest[:cut].strip(), rest[cut:].strip()
        if tail and not HAS_TEXT.search(tail) and head:
            pieces.append(head + tail)   # never emit a punctuation-only cue
            break
        pieces.append(head)
        rest = tail

    return [p for p in pieces if p]


def layout(sentence: Sentence, translated: str, measure: Measure,
           max_px: int, max_chars: int, min_cue_ms: int = 320) -> list[Cue]:
    """Build timed cues for one sentence."""
    zc = zh_clauses(translated)
    if not zc:
        return []

    groups = [split_to_width(c, measure, max_px, max_chars) for c in zc]
    ec = en_clauses(sentence)

    cues: list[Cue] = []
    if len(zc) == len(ec) and len(zc) > 1:
        for pieces, (span_start, span_end) in zip(groups, ec):
            total = sum(len(p) for p in pieces) or 1
            span = max(span_end - span_start, min_cue_ms * len(pieces))
            cursor = span_start
            for i, piece in enumerate(pieces):
                if i == len(pieces) - 1:
                    end = span_end
                else:
                    end = cursor + int(span * len(piece) / total)
                cues.append(Cue(cursor, max(end, cursor + min_cue_ms), piece))
                cursor = cues[-1].end_ms
    else:
        flat = [p for group in groups for p in group]
        total = sum(len(p) for p in flat) or 1
        span = max(sentence.end_ms - sentence.start_ms, min_cue_ms * len(flat))
        cursor = sentence.start_ms
        for i, piece in enumerate(flat):
            if i == len(flat) - 1:
                end = sentence.end_ms
            else:
                end = cursor + int(span * len(piece) / total)
            cues.append(Cue(cursor, max(end, cursor + min_cue_ms), piece))
            cursor = cues[-1].end_ms

    return cues


def dedup(cues: list[Cue], min_cue_ms: int = 320, gap_ms: int = 20) -> list[Cue]:
    """Enforce a single global timeline.

    Shrink the earlier cue when it has room; otherwise push the later one
    forward. Never drop a cue -- a dropped cue is a missing subtitle.
    """
    cues = sorted(cues, key=lambda c: (c.start_ms, c.end_ms))
    out: list[Cue] = []
    for cue in cues:
        if out:
            prev = out[-1]
            if cue.start_ms < prev.end_ms + gap_ms:
                target = cue.start_ms - gap_ms
                if target - prev.start_ms >= min_cue_ms:
                    prev.end_ms = target
                else:
                    prev.end_ms = prev.start_ms + min_cue_ms
                    cue.start_ms = prev.end_ms + gap_ms
        out.append(cue)

    for cue in out:
        if cue.end_ms <= cue.start_ms:
            cue.end_ms = cue.start_ms + min_cue_ms
    return out
