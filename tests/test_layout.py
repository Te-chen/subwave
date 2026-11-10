"""Tests for cue layout, wrapping and timeline de-overlap.

The wrapping tests use a fixed fake measure function so they assert behaviour,
not font metrics from whichever machine runs them.
"""
import pytest

from subwave import captions, layout


def fixed_measure(per_char: int = 46):
    return lambda text: len(text) * per_char


def sentence(text: str, start: int, end: int, clause_len: int = 3) -> captions.Sentence:
    words = []
    parts = text.split()
    step = max(1, (end - start) // max(1, len(parts)))
    for i, word in enumerate(parts):
        words.append(captions.Word(start + i * step, word))
    return captions.Sentence(start, end, text, words)


class TestZhClauses:
    def test_splits_on_chinese_punctuation(self):
        assert layout.zh_clauses("他来了，然后走了。") == ["他来了，", "然后走了。"]

    def test_drops_punctuation_only_fragments(self):
        # a bare "。" must not become its own cue
        chunks = layout.zh_clauses("结果必须经批准后才能公布。")
        assert all(chunk.strip("。，；：、 ") for chunk in chunks)

    def test_keeps_single_clause_intact(self):
        assert layout.zh_clauses("没有标点的一句话") == ["没有标点的一句话"]

    def test_empty_input(self):
        assert layout.zh_clauses("") == []


class TestWrap:
    def test_short_text_untouched(self):
        pieces = layout.split_to_width("短句", fixed_measure(), max_px=1000, max_chars=20)
        assert pieces == ["短句"]

    def test_long_text_is_split_within_budget(self):
        text = "这是一段相当长的中文字幕需要被切分成多条才能显示完整"
        pieces = layout.split_to_width(text, fixed_measure(), max_px=460, max_chars=10)
        assert len(pieces) > 1
        assert all(len(p) <= 10 for p in pieces)
        assert "".join(pieces) == text

    def test_prefers_punctuation_break(self):
        text = "第一句话在这里结束，第二句话从这里开始"
        pieces = layout.split_to_width(text, fixed_measure(), max_px=460, max_chars=12)
        assert pieces[0].endswith("，")

    def test_never_emits_punctuation_only_piece(self):
        text = "铅是脑毒，人人皆知已有两千年。"
        pieces = layout.split_to_width(text, fixed_measure(), max_px=510, max_chars=11)
        for piece in pieces:
            assert any(c.isalnum() or "\u4e00" <= c <= "\u9fff" for c in piece)

    def test_hard_cut_does_not_orphan_a_word_fragment(self):
        # regression: greedy fill cut "美国农业部门" into "美国农业部" + "门的主管"
        text = "那三位教授中的一位成为了美国农业部营养部门的主管。"
        pieces = layout.split_to_width(text, fixed_measure(), max_px=962, max_chars=21)
        assert "".join(pieces) == text
        assert not any(p.startswith("门") for p in pieces)
        # halves should be roughly even rather than 21 chars then 5
        assert max(len(p) for p in pieces) - min(len(p) for p in pieces) <= 8

    def test_latin_acronym_is_never_split(self):
        text = "他成了 USDA 的主管并推动了整件事"
        pieces = layout.split_to_width(text, fixed_measure(), max_px=380, max_chars=12)
        joined = "".join(pieces)
        assert joined == text
        # the token must survive intact in one piece, not be cut in half
        assert any("USDA" in p for p in pieces)


class TestLayout:
    def test_matching_clause_counts_use_english_timing(self):
        sent = sentence("They went home, and then they slept.", 0, 4000)
        cues = layout.layout(sent, "他们回家了，然后就睡了。",
                             fixed_measure(), max_px=1000, max_chars=30)
        assert len(cues) == 2
        # first cue should start at the sentence start
        assert cues[0].start_ms == 0
        # second cue starts at the English clause boundary, not a guess
        assert cues[1].start_ms >= cues[0].end_ms

    def test_mismatched_clause_counts_fall_back_to_proportional(self):
        sent = sentence("One two three four five six.", 0, 3000)
        cues = layout.layout(sent, "一二三四五六七八九十一二三四五六",
                             fixed_measure(), max_px=200, max_chars=8)
        assert len(cues) >= 2
        assert cues[0].start_ms == 0
        assert cues[-1].end_ms == 3000

    def test_cues_cover_sentence_span(self):
        sent = sentence("A b c d e f g h.", 1000, 5000)
        cues = layout.layout(sent, "甲乙丙丁戊己庚辛",
                             fixed_measure(), max_px=140, max_chars=4)
        assert cues[0].start_ms == 1000
        assert cues[-1].end_ms == 5000

    def test_empty_translation_yields_no_cues(self):
        sent = sentence("Ignored.", 0, 1000)
        assert layout.layout(sent, "", fixed_measure(), 1000, 20) == []


class TestDedup:
    def test_overlap_is_removed(self):
        cues = [layout.Cue(0, 2000, "a"), layout.Cue(1500, 3000, "b")]
        out = layout.dedup(cues)
        assert out[0].end_ms <= out[1].start_ms

    def test_nothing_is_dropped(self):
        cues = [layout.Cue(0, 3000, "a"), layout.Cue(100, 4000, "b"),
                layout.Cue(200, 5000, "c")]
        out = layout.dedup(cues)
        assert len(out) == 3
        assert [c.text for c in out] == ["a", "b", "c"]

    def test_short_previous_cue_pushes_next_forward(self):
        # previous cue is already at the minimum length, so the later one moves
        cues = [layout.Cue(0, 320, "a"), layout.Cue(330, 2000, "b")]
        out = layout.dedup(cues, min_cue_ms=320, gap_ms=20)
        assert out[1].start_ms >= out[0].end_ms + 20

    def test_already_clean_timeline_is_unchanged(self):
        cues = [layout.Cue(0, 1000, "a"), layout.Cue(1200, 2000, "b")]
        out = layout.dedup(cues)
        assert [(c.start_ms, c.end_ms) for c in out] == [(0, 1000), (1200, 2000)]
