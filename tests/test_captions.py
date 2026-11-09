"""Tests for caption parsing and sentence assembly.

These target the failure modes that actually occurred while building the tool:
lost spaces at event boundaries, aAppend events leaking as content, and
abbreviation false-positives splitting sentences.
"""
import json
from pathlib import Path

from subwave import captions


def _write_json3(tmp_path: Path, events: list[dict]) -> Path:
    path = tmp_path / "test.en-orig.json3"
    path.write_text(json.dumps({"events": events}), encoding="utf-8")
    return path


def test_aappend_events_are_not_content(tmp_path):
    path = _write_json3(tmp_path, [
        {"tStartMs": 0, "dDurationMs": 3000, "wWinId": 1,
         "segs": [{"utf8": "hello"}, {"utf8": " world", "tOffsetMs": 300}]},
        {"tStartMs": 1200, "dDurationMs": 1800, "aAppend": 1,
         "segs": [{"utf8": "\n"}]},
    ])
    words = captions.parse_json3(path)
    assert [w.w for w in words] == ["hello", "world"]


def test_words_across_events_keep_their_separating_space(tmp_path):
    path = _write_json3(tmp_path, [
        {"tStartMs": 0, "dDurationMs": 2000, "wWinId": 1,
         "segs": [{"utf8": "still"}, {"utf8": " not", "tOffsetMs": 500}]},
        {"tStartMs": 2100, "dDurationMs": 2000, "wWinId": 1,
         "segs": [{"utf8": "invented"}, {"utf8": " it", "tOffsetMs": 400}]},
    ])
    text = captions.join_words(captions.parse_json3(path))
    assert "not invented" in text
    assert "notinvented" not in text


def test_missing_t_offsets_are_interpolated(tmp_path):
    path = _write_json3(tmp_path, [
        {"tStartMs": 1000, "dDurationMs": 1000,
         "segs": [{"utf8": "a"}, {"utf8": " b"}, {"utf8": " c"}]},
    ])
    words = captions.parse_json3(path)
    times = [w.t for w in words]
    assert times == sorted(times)
    assert times[0] == 1000
    assert times[-1] == 2000


def test_sentences_break_on_terminal_punctuation():
    words = [
        captions.Word(0, "Hello"), captions.Word(100, "there."),
        captions.Word(200, "How"), captions.Word(300, "are"), captions.Word(400, "you?"),
    ]
    sents = captions.build_sentences(words)
    assert len(sents) == 2
    assert sents[0].text == "Hello there."
    assert sents[1].text == "How are you?"


def test_abbreviations_do_not_end_a_sentence():
    words = [
        captions.Word(0, "Mr."), captions.Word(100, "Smith"),
        captions.Word(200, "arrived."),
    ]
    sents = captions.build_sentences(words)
    assert len(sents) == 1
    assert "Mr. Smith arrived." in sents[0].text


def test_music_markers_are_stripped(tmp_path):
    path = _write_json3(tmp_path, [
        {"tStartMs": 0, "dDurationMs": 1000,
         "segs": [{"utf8": "[Music]"}, {"utf8": " real", "tOffsetMs": 300},
                  {"utf8": " words.", "tOffsetMs": 600}]},
    ])
    text = captions.join_words(captions.parse_json3(path))
    assert "Music" not in text
    assert "real words." in text


def test_vtt_parsing(tmp_path):
    path = tmp_path / "a.vtt"
    path.write_text(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello there friend\n",
        encoding="utf-8")
    words = captions.parse_vtt(path)
    assert [w.w for w in words] == ["Hello", "there", "friend"]
    assert words[0].t == 1000
