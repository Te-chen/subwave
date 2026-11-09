"""Tests for the translator's batching, caching and count-mismatch recovery.

No network: the request method is stubbed so these run offline.
"""
import json

from subwave import translate


class StubTranslator(translate.Translator):
    """Returns predictable output; optionally misbehaves on purpose."""

    def __init__(self, *args, merge_lines=False, fail_times=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.merge_lines = merge_lines
        self.fail_times = fail_times
        self.seen_batches = []

    def _request(self, lines):
        self.stats["requests"] += 1
        self.seen_batches.append(list(lines))
        if self.fail_times > 0:
            self.fail_times -= 1
            raise OSError("temporary network failure")
        if self.merge_lines and len(lines) > 1:
            # simulate the model merging two short lines into one
            return ["合并后的中文"] * (len(lines) - 1)
        return [f"译:{line}" for line in lines]


def test_translates_in_order():
    t = StubTranslator(api_key="x")
    out = t.translate(["one", "two", "three"], batch_size=2)
    assert out == ["译:one", "译:two", "译:three"]


def test_cache_avoids_second_request(tmp_path):
    t = StubTranslator(api_key="x", cache_dir=tmp_path / "cache")
    first = t.translate(["hello"], batch_size=4)
    requests_after_first = t.stats["requests"]
    second = t.translate(["hello"], batch_size=4)
    assert first == second
    assert t.stats["requests"] == requests_after_first
    assert t.stats["cached"] >= 1


def test_merged_output_falls_back_to_single_lines():
    t = StubTranslator(api_key="x", merge_lines=True)
    out = t.translate(["a", "b", "c"], batch_size=4)
    # each line must still get its own translation, in order
    assert len(out) == 3
    assert all(out)


def test_retries_on_transient_failure():
    t = StubTranslator(api_key="x", fail_times=2)
    out = t.translate(["hello"], batch_size=1)
    assert out == ["译:hello"]


def test_batch_size_is_respected():
    t = StubTranslator(api_key="x")
    t.translate([f"line{i}" for i in range(10)], batch_size=4)
    sizes = [len(b) for b in t.seen_batches]
    assert max(sizes) <= 4


def test_empty_input():
    t = StubTranslator(api_key="x")
    assert t.translate([], batch_size=4) == []


def test_estimate_tokens_is_sane():
    assert translate.estimate_tokens("") <= 2
    latin = translate.estimate_tokens("hello world this is a test")
    assert 3 <= latin <= 20
    cjk = translate.estimate_tokens("这是一句中文")
    assert cjk >= 6


def test_numbering_prefix_is_stripped():
    t = StubTranslator(api_key="x")
    original = t._request
    t._request = lambda lines: ["1. 第一句", "2) 第二句"]  # noqa: E731
    out = t._translate_batch(["a", "b"])
    assert out == ["第一句", "第二句"]
    t._request = original
