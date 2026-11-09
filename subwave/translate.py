"""Translation via an OpenAI-compatible chat endpoint (DeepSeek by default).

Token economy matters here: these are short narrative clips, so we translate
sentence by sentence in small batches and cache aggressively. A batch whose line
count comes back wrong is retried once as a whole, then degraded to single lines
-- the model merging two short lines is common and must not silently desync the
subtitle timeline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "You are a professional subtitle translator working from {src} into {dst}. "
    "Translate each numbered line into natural spoken {dst}, keeping the speaker's "
    "tone and rhythm. Keep proper nouns, brands and numbers intact. "
    "Output exactly one line per input line, same order, same count. "
    "Never merge or split lines. Output translations only."
)

_NUMBER_PREFIX = re.compile(r"^\d+[\.\)、]\s*")


class Translator:
    def __init__(self, api_key: str | None = None, endpoint: str | None = None,
                 model: str | None = None, cache_dir: Path | None = None,
                 src_lang: str = "English", dst_lang: str = "Simplified Chinese",
                 timeout: int = 180):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.endpoint = endpoint or os.environ.get("SUBWAVE_ENDPOINT", DEFAULT_ENDPOINT)
        self.model = model or os.environ.get("SUBWAVE_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.stats = {"requests": 0, "lines": 0, "cached": 0}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache -------------------------------------------------------------

    def _cache_path(self, text: str) -> Path | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(
            f"{self.model}|{self.src_lang}|{self.dst_lang}|{text}".encode()
        ).hexdigest()[:32]
        return self.cache_dir / f"{key}.txt"

    def _cache_get(self, text: str) -> str | None:
        path = self._cache_path(text)
        if path and path.exists():
            self.stats["cached"] += 1
            return path.read_text(encoding="utf-8")
        return None

    def _cache_put(self, text: str, value: str) -> None:
        path = self._cache_path(text)
        if path:
            path.write_text(value, encoding="utf-8")

    # -- request -----------------------------------------------------------

    def _request(self, lines: list[str]) -> list[str]:
        prompt = "\n".join(f"{i}. {t}" for i, t in enumerate(lines, 1))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(
                    src=self.src_lang, dst=self.dst_lang)},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        self.stats["requests"] += 1
        content = body["choices"][0]["message"]["content"]
        return [ln.strip() for ln in content.split("\n") if ln.strip()]

    def _translate_batch(self, lines: list[str]) -> list[str]:
        for attempt in range(4):
            try:
                raw = self._request(lines)
                # strip list numbering whichever layer produced it
                got = [_NUMBER_PREFIX.sub("", ln).strip() for ln in raw]
                got = [g for g in got if g]
                if len(got) == len(lines):
                    return got
                if len(lines) > 1:
                    # model merged or split lines: isolate each one
                    return [self._translate_batch([ln])[0] for ln in lines]
                # single line but garbled output -- retry
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError,
                    OSError) as exc:
                if attempt == 3:
                    raise RuntimeError(f"translation failed: {exc}") from exc
                time.sleep(1.5 ** attempt)
        return ["" for _ in lines]

    def translate(self, lines: list[str], batch_size: int = 20) -> list[str]:
        """Translate lines, honouring the cache and preserving input order.

        Batch size trades prompt overhead against the risk of a batch coming
        back misaligned: each request re-sends the system prompt, so very small
        batches waste most of their tokens on instructions. 20 keeps the prompt
        share near 10% while staying well inside reliable output lengths.
        """
        results: list[str | None] = [None] * len(lines)

        pending: list[tuple[int, str]] = []
        for i, text in enumerate(lines):
            cached = self._cache_get(text)
            if cached is not None:
                results[i] = cached
            else:
                pending.append((i, text))

        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            translated = self._translate_batch([t for _, t in chunk])
            for (i, src), dst in zip(chunk, translated):
                results[i] = dst
                self._cache_put(src, dst)

        self.stats["lines"] += len(lines)
        return [r or "" for r in results]


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK ~1 token/char, Latin ~1 token/4 chars."""
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return cjk + max(1, other // 4)
