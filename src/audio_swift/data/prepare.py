"""Convert local, single-utterance speech records to Swift's dataset format."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

AUDIO_KEYS = ("audios", "audio", "wav", "wav_path", "path")
TEXT_KEYS = ("text", "txt", "response", "target", "transcript", "transcription")


def _records(input_path: Path) -> Iterator[tuple[int, Any]]:
    with input_path.open(encoding="utf-8") as source:
        if input_path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Line {line_number}: invalid JSON: {exc.msg}") from exc
        else:
            try:
                rows = json.load(source)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {exc.lineno}: {exc.msg}") from exc
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list):
                raise ValueError("JSON input must be a record or a list of records")
            yield from enumerate(rows, 1)


def _first_field(record: dict[str, Any], keys: tuple[str, ...], kind: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    raise ValueError(f"missing {kind}; expected one of {', '.join(keys)}")


def _audio_path(record: dict[str, Any], input_dir: Path) -> Path:
    audio = _first_field(record, AUDIO_KEYS, "audio")
    if isinstance(audio, list):
        if len(audio) != 1:
            raise ValueError("exactly one audio file is required per record")
        audio = audio[0]
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError("audio must be a nonempty local file path")
    if "://" in audio:
        raise ValueError("audio must be a local file path, not a URL")
    path = Path(audio).expanduser()
    if not path.is_absolute():
        path = input_dir / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"audio file does not exist: {path}")
    return path


def _messages(
    record: dict[str, Any],
    prompt: str,
    response_prefix: str,
    response_prefix_pattern: re.Pattern[str] | None,
) -> list[dict[str, Any]]:
    if "messages" in record:
        original = record["messages"]
        if not isinstance(original, list) or not original:
            raise ValueError("messages must be a nonempty list")
        messages = []
        for message in original:
            if not isinstance(message, dict):
                raise ValueError("each message must be an object")
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("message roles must be system, user, or assistant")
            if not isinstance(message.get("content"), str):
                raise ValueError("message content must be text")
            messages.append(dict(message))
    else:
        text = _first_field(record, TEXT_KEYS, "transcript")
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]

    roles = [message["role"] for message in messages]
    if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
        raise ValueError(
            "single-turn messages require an optional initial system, "
            "exactly one user, and exactly one assistant transcript last"
        )
    user, assistant = messages[-2:]
    transcript = assistant["content"]
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("transcript must be nonempty text")

    if any("<audio>" in message["content"] for message in messages if message["role"] != "user"):
        raise ValueError("<audio> markers are allowed only in the user message")
    audio_markers = user["content"].count("<audio>")
    if audio_markers > 1:
        raise ValueError("exactly one <audio> marker is allowed per record")
    if not audio_markers:
        user_prompt = user["content"]
        user["content"] = "<audio>" + (f"\n{user_prompt}" if user_prompt else "")

    prefix_match = response_prefix_pattern.match(transcript) if response_prefix_pattern else None
    if prefix_match is not None:
        prefix_end = prefix_match.end()
        if prefix_end == 0:
            raise ValueError("response_prefix_pattern must match a nonempty prefix")
    elif response_prefix and transcript.startswith(response_prefix):
        prefix_end = len(response_prefix)
    else:
        prefix_end = 0
        assistant["content"] = response_prefix + transcript
    if not transcript[prefix_end:].strip():
        raise ValueError("transcript after response prefix must be nonempty")
    return messages


def _same_file(first: Path, second: Path) -> bool:
    return first == second or (second.exists() and first.samefile(second))


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    *,
    prompt: str = "<audio>",
    response_prefix: str = "",
    response_prefix_pattern: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write validated JSONL and return its record count.

    Relative audio paths are resolved against the input file's directory. Only
    the first ``limit`` records are selected when a limit is supplied. Invalid
    selected records raise an error without committing a partial output file.
    ``prompt`` supplies the user message for raw records; existing messages are
    retained. A response already matching the supplied pattern or literal
    prefix is preserved; otherwise ``response_prefix`` is prepended.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be text")
    if not isinstance(response_prefix, str):
        raise ValueError("response_prefix must be text")
    if "<audio>" in response_prefix:
        raise ValueError("response_prefix must not contain an <audio> marker")
    prefix_pattern = None
    if response_prefix_pattern is not None:
        if not isinstance(response_prefix_pattern, str) or not response_prefix_pattern:
            raise ValueError("response_prefix_pattern must be nonempty text")
        try:
            prefix_pattern = re.compile(response_prefix_pattern)
        except re.error as exc:
            raise ValueError(f"invalid response_prefix_pattern: {exc}") from exc
        if prefix_pattern.match("") is not None:
            raise ValueError("response_prefix_pattern must match a nonempty prefix")
    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")
    if _same_file(input_path, output_path):
        raise ValueError("output must not overwrite the input file")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; use --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as target:
            temporary_path = Path(target.name)
            records = iter(_records(input_path))
            while limit is None or count < limit:
                try:
                    index, record = next(records)
                except StopIteration:
                    break
                try:
                    if not isinstance(record, dict):
                        raise ValueError("record must be an object")
                    audio_path = _audio_path(record, input_path.parent)
                    if _same_file(audio_path, output_path):
                        raise ValueError("output must not overwrite an audio file")
                    messages = _messages(record, prompt, response_prefix, prefix_pattern)
                except ValueError as exc:
                    raise ValueError(f"Record {index}: {exc}") from exc
                row = {"messages": messages, "audios": [str(audio_path)]}
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            if count == 0:
                raise ValueError("input dataset contains no records")
        if overwrite:
            os.replace(temporary_path, output_path)
        else:
            # Linking commits atomically while refusing a destination created
            # after the initial existence check.
            os.link(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return count


def add_prepare_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register the ``prepare-data`` subcommand on the project CLI."""
    parser = subparsers.add_parser("prepare-data", help="prepare local speech JSON or JSONL data")
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        default="<audio>",
        help="user prompt for raw audio/transcript records (default: <audio>)",
    )
    parser.add_argument("--response-prefix", default="", help="prepend this text to unprefixed responses")
    parser.add_argument("--response-prefix-pattern", help="regex identifying an existing response prefix")
    parser.add_argument("--limit", type=int, help="select the first N records")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.set_defaults(func=run_prepare)
    return parser


def run_prepare(args: argparse.Namespace) -> int:
    count = prepare_dataset(
        args.input_path,
        args.output_path,
        prompt=args.prompt,
        response_prefix=args.response_prefix,
        response_prefix_pattern=args.response_prefix_pattern,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"Prepared {count} records: {Path(args.output_path).expanduser().resolve()}")
    return 0
