"""Lightweight data preparation and transparent Swift SFT entry point."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence

from audio_swift import __version__
from audio_swift.backends.swift.engine import build_sft_command, run_sft
from audio_swift.data import add_prepare_parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="audio-swift", allow_abbrev=False)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_prepare_parser(commands)
    commands.add_parser("sft", help="Forward native arguments to the installed Swift engine.")
    try:
        if arguments[:1] == ["sft"]:
            native = arguments[1:]
            dry_run = native[:1] == ["--dry-run"]
            if dry_run:
                native = native[1:]
            if native[:1] == ["--"]:
                native = native[1:]
            if dry_run:
                print(shlex.join(build_sft_command(native)))
                return 0
            return run_sft(native)
        args = parser.parse_args(arguments)
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"audio-swift: error: {exc}\n")
