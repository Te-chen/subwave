"""Command line entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline, verify


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subwave",
        description="Add translated hard subtitles to videos using their own captions.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="fetch captions and build an inventory")
    p_prep.add_argument("src", type=Path)

    p_make = sub.add_parser("make", help="translate and render subtitles")
    p_make.add_argument("src", type=Path)
    p_make.add_argument("--lang", default="zh-CN", help="target language tag (default zh-CN)")
    p_make.add_argument("--out", type=Path, default=None,
                        help="output base directory (default ./out)")
    p_make.add_argument("--limit", type=int, default=None, help="only first N videos")
    p_make.add_argument("--force", action="store_true", help="redo existing outputs")
    p_make.add_argument("--dry-run", action="store_true", help="show the plan only")
    p_make.add_argument("--model", default=None, help="translation model")
    p_make.add_argument("--batch-size", type=int, default=20,
                        help="sentences per translation request (default 8)")
    p_make.add_argument("--crf", type=int, default=20, help="x264 quality (default 20)")
    p_make.add_argument("--font", default=None, help="caption font family")
    p_make.add_argument("--caption-langs", default="en-orig,en",
                        help="caption languages to try, comma separated")
    p_make.add_argument("--captions-dir", type=Path, default=None,
                        help="reuse caption files already downloaded to this directory")

    p_verify = sub.add_parser("verify", help="run the quality gate")
    p_verify.add_argument("src", type=Path)
    p_verify.add_argument("--lang", default="zh-CN")
    p_verify.add_argument("--out", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "prepare":
        pipeline.prepare(args.src)
        return 0

    if args.command == "make":
        langs = tuple(l.strip() for l in args.caption_langs.split(",") if l.strip())
        results = pipeline.run(
            args.src, lang=args.lang, out_dir=args.out, limit=args.limit,
            force=args.force, dry_run=args.dry_run, model=args.model,
            batch_size=args.batch_size, crf=args.crf, font=args.font,
            caption_langs=langs, captions_dir=args.captions_dir,
        )
        if args.dry_run:
            return 0
        return 0 if all(r.ok for r in results) else 1

    if args.command == "verify":
        checks = verify.verify(args.src, lang=args.lang, out_dir=args.out)
        return 0 if all(c.ok for c in checks) else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
