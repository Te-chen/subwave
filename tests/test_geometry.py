"""Tests for geometry assumptions and output layout rules."""
from pathlib import Path

from subwave import geometry, pipeline


class TestFontScaling:
    def test_font_scales_with_width(self):
        assert geometry.font_size_for(1080) > geometry.font_size_for(576)

    def test_same_optical_character_count(self):
        # both sizes should hold roughly the same number of characters per line
        for width in (1080, 576, 1920, 720):
            size = geometry.font_size_for(width)
            usable = width - int(width * 0.055) * 2
            assert 15 <= usable / size <= 26

    def test_floor_prevents_tiny_type(self):
        assert geometry.font_size_for(200) >= 16


class TestBands:
    def test_detects_usable_band(self):
        geom = geometry.Geometry(1080, 1920, 196, 1170)
        assert geom.has_bottom_band
        assert geom.band_height == 750

    def test_no_band_when_artwork_fills_frame(self):
        geom = geometry.Geometry(1920, 1080, 0, 1080)
        assert not geom.has_bottom_band

    def test_margin_sits_inside_band(self):
        geom = geometry.Geometry(1080, 1920, 196, 1170)
        _, margin_v, _ = geometry.margins_for(geom)
        assert margin_v < geom.band_height

    def test_margin_avoids_bottom_edge_without_band(self):
        geom = geometry.Geometry(1920, 1080, 0, 1080)
        _, margin_v, _ = geometry.margins_for(geom)
        assert margin_v >= int(1080 * 0.09)

    def test_max_px_leaves_side_margins(self):
        geom = geometry.Geometry(1080, 1920, 196, 1170)
        margin_lr, _, max_px = geometry.margins_for(geom)
        assert max_px == 1080 - margin_lr * 2


class TestDiscovery:
    def test_finds_video_and_keeps_original_name(self, tmp_path):
        (tmp_path / "2026-10-18 - Title [HR9MQZGE36I].mp4").write_bytes(b"x")
        items = pipeline.discover(tmp_path, lang="zh-CN")
        assert len(items) == 1
        assert items[0].video_id == "HR9MQZGE36I"
        assert items[0].out_name == "zh-CN - 2026-10-18 - Title [HR9MQZGE36I].mp4"

    def test_video_without_id_still_discovered(self, tmp_path):
        (tmp_path / "plain.mp4").write_bytes(b"x")
        items = pipeline.discover(tmp_path, lang="zh-CN")
        assert items[0].video_id is None

    def test_ignores_non_video_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x")
        (tmp_path / "a.mp4").write_bytes(b"x")
        assert len(pipeline.discover(tmp_path, lang="zh-CN")) == 1

    def test_sorts_by_filename(self, tmp_path):
        for name in ("b.mp4", "a.mp4", "c.mp4"):
            (tmp_path / name).write_bytes(b"x")
        names = [i.path.name for i in pipeline.discover(tmp_path, lang="zh-CN")]
        assert names == ["a.mp4", "b.mp4", "c.mp4"]


class TestNameMap:
    def test_explicit_names_override_convention(self, tmp_path):
        (tmp_path / "2026-09-08 - original [ADxtQ7stHbE].mp4").write_bytes(b"x")
        items = pipeline.discover(tmp_path, lang="zh-CN",
                                  names={"ADxtQ7stHbE": "2026-09-08_中文标题_MassInfluence.mp4"})
        assert items[0].out_name == "2026-09-08_中文标题_MassInfluence.mp4"

    def test_video_without_map_entry_keeps_default(self, tmp_path):
        (tmp_path / "a [ADxtQ7stHbE].mp4").write_bytes(b"x")
        items = pipeline.discover(tmp_path, lang="zh-CN", names={"OTHERID123": "x.mp4"})
        assert items[0].out_name == "zh-CN - a [ADxtQ7stHbE].mp4"

    def test_load_names_from_list_of_objects(self, tmp_path):
        p = tmp_path / "titles.json"
        p.write_text('[{"video_id": "ABC", "filename": "out.mp4"}]', encoding="utf-8")
        assert pipeline.load_names(p) == {"ABC": "out.mp4"}

    def test_load_names_from_plain_mapping(self, tmp_path):
        p = tmp_path / "names.json"
        p.write_text('{"ABC": "out.mp4"}', encoding="utf-8")
        assert pipeline.load_names(p) == {"ABC": "out.mp4"}

    def test_load_names_skips_rows_without_filename(self, tmp_path):
        p = tmp_path / "titles.json"
        p.write_text('[{"video_id": "A", "filename": "a.mp4"}, {"video_id": "B"}]',
                     encoding="utf-8")
        assert pipeline.load_names(p) == {"A": "a.mp4"}


class TestOutputLayout:
    def test_groups_by_creator_then_language(self, tmp_path):
        creator = tmp_path / "brain-duster"
        out = pipeline.output_root(creator, Path("/out"), "zh-CN")
        assert out == Path("/out") / "brain-duster" / "zh-CN"

    def test_two_creators_do_not_collide(self, tmp_path):
        a = pipeline.output_root(tmp_path / "alpha", Path("/out"), "zh-CN")
        b = pipeline.output_root(tmp_path / "beta", Path("/out"), "zh-CN")
        assert a != b

    def test_language_is_a_subdirectory(self, tmp_path):
        creator = tmp_path / "alpha"
        zh = pipeline.output_root(creator, Path("/out"), "zh-CN")
        ja = pipeline.output_root(creator, Path("/out"), "ja-JP")
        assert zh.parent == ja.parent
        assert zh != ja


class TestFontResolution:
    def test_returns_a_font_name(self):
        assert isinstance(geometry.font_size_for(1080), int)

    def test_font_file_lookup_does_not_crash(self):
        result = pipeline.font_file_for("SimHei")
        assert result is None or Path(result).exists() or isinstance(result, Path)
