import argparse
import json

import pytest

from audio_swift.data import add_prepare_parser, prepare_dataset, run_prepare


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "audio" / "utterance.wav"
    path.parent.mkdir()
    path.write_bytes(b"test audio")
    return path


def write_input(path, records):
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def read_output(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("audio_key", ["audio", "audios", "wav", "wav_path", "path"])
@pytest.mark.parametrize("text_key", ["text", "txt", "response", "target", "transcript", "transcription"])
def test_aliases_and_relative_audio_paths(tmp_path, audio, audio_key, text_key):
    audio_value = "audio/utterance.wav"
    if audio_key == "audios":
        audio_value = [audio_value]
    source = write_input(tmp_path / "source.json", [{audio_key: audio_value, text_key: "你好世界"}])
    output = tmp_path / "prepared.jsonl"

    assert prepare_dataset(source, output) == 1
    assert read_output(output) == [
        {
            "messages": [
                {"role": "user", "content": "<audio>"},
                {"role": "assistant", "content": "你好世界"},
            ],
            "audios": [str(audio)],
        }
    ]


def test_standard_messages_and_pattern_matched_prefix_are_preserved(tmp_path, audio):
    messages = [
        {"role": "system", "content": "You transcribe audio."},
        {"role": "user", "content": "<audio>Transcribe this."},
        {"role": "assistant", "content": "[caption:en] Hello."},
    ]
    source = write_input(tmp_path / "source.json", {"audios": [str(audio)], "messages": messages})
    output = tmp_path / "prepared.jsonl"

    prepare_dataset(source, output, response_prefix="[caption:zh] ", response_prefix_pattern=r"^\[caption:[a-z]+\] ")

    assert read_output(output)[0]["messages"] == messages
    second_output = tmp_path / "prepared_twice.jsonl"
    prepare_dataset(
        output, second_output, response_prefix="[caption:ja] ", response_prefix_pattern=r"^\[caption:[a-z]+\] "
    )
    assert read_output(second_output) == read_output(output)


def test_explicit_prefix_and_prompt_audio_marker(tmp_path, audio):
    source = write_input(
        tmp_path / "source.json",
        {
            "audio": str(audio),
            "messages": [
                {"role": "user", "content": "Please transcribe."},
                {"role": "assistant", "content": "Hello."},
            ],
        },
    )
    output = tmp_path / "prepared.jsonl"

    prepare_dataset(source, output, prompt="ignored", response_prefix="[caption:en] ")

    assert read_output(output)[0]["messages"] == [
        {"role": "user", "content": "<audio>\nPlease transcribe."},
        {"role": "assistant", "content": "[caption:en] Hello."},
    ]


def test_plain_transcript_and_jsonl_limit(tmp_path, audio):
    source = tmp_path / "source.jsonl"
    row = {"audio": str(audio), "text": "Hello, 世界!"}
    source.write_text(json.dumps(row) + "\n" + "invalid unselected row\n", encoding="utf-8")
    output = tmp_path / "prepared.jsonl"

    assert prepare_dataset(source, output, limit=1) == 1
    assert read_output(output)[0]["messages"][-1]["content"] == "Hello, 世界!"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("<audio>Write the transcript.", "<audio>Write the transcript."),
        ("Write the transcript.", "<audio>\nWrite the transcript."),
        ("", "<audio>"),
    ],
)
def test_explicit_prompt_for_raw_records(tmp_path, audio, prompt, expected):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": "hello"})
    output = tmp_path / "prepared.jsonl"

    prepare_dataset(source, output, prompt=prompt)

    assert read_output(output)[0]["messages"][0]["content"] == expected


@pytest.mark.parametrize(
    ("transcript", "options", "expected"),
    [
        ("hello", {"response_prefix": "[caption] "}, "[caption] hello"),
        ("[caption] hello", {"response_prefix": "[caption] "}, "[caption] hello"),
        (
            "[caption] hello",
            {"response_prefix": "[caption] ", "response_prefix_pattern": r"^\[other\] "},
            "[caption] hello",
        ),
        (
            "[caption:en] hello",
            {"response_prefix_pattern": r"^\[caption:[a-z]+\] "},
            "[caption:en] hello",
        ),
    ],
)
def test_response_prefix_is_generic_and_idempotent(tmp_path, audio, transcript, options, expected):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": transcript})
    output = tmp_path / "prepared.jsonl"

    prepare_dataset(source, output, **options)

    assert read_output(output)[0]["messages"][-1]["content"] == expected
    second_output = tmp_path / "prepared_twice.jsonl"
    prepare_dataset(output, second_output, **options)
    assert read_output(second_output) == read_output(output)


@pytest.mark.parametrize(
    ("transcript", "options"),
    [
        ("[caption] ", {"response_prefix": "[caption] "}),
        ("[caption]   ", {"response_prefix": "[caption] "}),
        (
            "[caption:en] ",
            {"response_prefix": "[caption:zh] ", "response_prefix_pattern": r"^\[caption:[a-z]+\] "},
        ),
        ("[caption:en]   ", {"response_prefix_pattern": r"^\[caption:[a-z]+\] "}),
    ],
)
def test_prefix_without_transcript_is_rejected(tmp_path, audio, transcript, options):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": transcript})
    output = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match="transcript after response prefix must be nonempty"):
        prepare_dataset(source, output, **options)

    assert not output.exists()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"prompt": None}, "prompt must be text"),
        ({"response_prefix": None}, "response_prefix must be text"),
        ({"response_prefix": "<audio>"}, "must not contain an <audio> marker"),
        ({"response_prefix_pattern": 42}, "must be nonempty text"),
        ({"response_prefix_pattern": ""}, "must be nonempty text"),
        ({"response_prefix_pattern": "["}, "invalid response_prefix_pattern"),
        ({"response_prefix_pattern": "^"}, "must match a nonempty prefix"),
        ({"response_prefix_pattern": r"\b"}, "must match a nonempty prefix"),
        ({"prompt": "<audio><audio>"}, "exactly one <audio> marker"),
    ],
)
def test_invalid_prefix_or_prompt_configuration(tmp_path, audio, options, message):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": "hello"})
    output = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match=message):
        prepare_dataset(source, output, **options)

    assert not output.exists()


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"text": "hello"}, "missing audio"),
        ({"audio": "audio/utterance.wav"}, "missing transcript"),
        ({"audio": "audio/utterance.wav", "text": " "}, "nonempty text"),
        ({"audio": "audio/utterance.wav", "text": None}, "nonempty text"),
        ({"audio": "absent.wav", "text": "hello"}, "audio file does not exist"),
        ({"audio": [], "text": "hello"}, "exactly one audio"),
        ({"audios": ["a.wav", "b.wav"], "text": "hello"}, "exactly one audio"),
        ({"audio": "https://example.com/a.wav", "text": "hello"}, "local file path"),
        ({"audio": "audio/utterance.wav", "messages": []}, "nonempty list"),
        (
            {
                "audio": "audio/utterance.wav",
                "messages": [
                    {"role": "user", "content": "<audio>"},
                ],
            },
            "exactly one assistant",
        ),
        (
            {
                "audio": "audio/utterance.wav",
                "messages": [
                    {"role": "user", "content": "<audio><audio>"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
            "one <audio> marker",
        ),
        (["not an object"], "record must be an object"),
    ],
)
def test_invalid_records_do_not_create_partial_output(tmp_path, audio, record, message):
    source = write_input(tmp_path / "source.json", [{"audio": str(audio), "text": "valid"}, record])
    output = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match=message):
        prepare_dataset(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.jsonl.*"))


@pytest.mark.parametrize("prefix_options", [{}, {"response_prefix": "[caption] "}], ids=["plain", "prefix"])
@pytest.mark.parametrize(
    "roles",
    [
        ["assistant", "user"],
        ["user", "assistant", "user"],
        ["user", "user", "assistant"],
        ["system", "system", "user", "assistant"],
        ["user", "system", "assistant"],
        ["system", "assistant", "user"],
        ["user", "assistant", "assistant"],
    ],
)
def test_messages_must_be_single_turn_in_order(tmp_path, audio, prefix_options, roles):
    messages = [{"role": role, "content": "<audio>" if role == "user" else "text"} for role in roles]
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "messages": messages})
    output = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match="single-turn messages"):
        prepare_dataset(source, output, **prefix_options)

    assert not output.exists()


@pytest.mark.parametrize("prefix_options", [{}, {"response_prefix": "[caption] "}], ids=["plain", "prefix"])
@pytest.mark.parametrize("marker_role", ["system", "assistant"])
@pytest.mark.parametrize("user_marker", [False, True])
def test_audio_marker_is_only_allowed_in_user(tmp_path, audio, prefix_options, marker_role, user_marker):
    messages = [
        {"role": "system", "content": "Transcribe."},
        {"role": "user", "content": "<audio>" if user_marker else "Transcribe."},
        {"role": "assistant", "content": "hello"},
    ]
    next(message for message in messages if message["role"] == marker_role)["content"] += "<audio>"
    source = write_input(tmp_path / "source.json", {"audios": [str(audio)], "messages": messages})
    output = tmp_path / "prepared.jsonl"

    with pytest.raises(ValueError, match="only in the user message"):
        prepare_dataset(source, output, **prefix_options)

    assert not output.exists()


def test_overwrite_requires_explicit_opt_in(tmp_path, audio):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": "hello"})
    output = tmp_path / "prepared.jsonl"
    output.write_text("existing data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_dataset(source, output)
    assert output.read_text(encoding="utf-8") == "existing data"

    assert prepare_dataset(source, output, overwrite=True) == 1
    assert read_output(output)[0]["messages"][-1]["content"] == "hello"


def test_failed_overwrite_preserves_existing_output(tmp_path, audio):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": ""})
    output = tmp_path / "prepared.jsonl"
    output.write_text("existing data", encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty text"):
        prepare_dataset(source, output, overwrite=True)

    assert output.read_text(encoding="utf-8") == "existing data"


def test_source_and_audio_cannot_be_overwritten(tmp_path, audio):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": "hello"})
    original = source.read_bytes()

    with pytest.raises(ValueError, match="input file"):
        prepare_dataset(source, source, overwrite=True)
    assert source.read_bytes() == original
    hard_link = tmp_path / "source_hard_link.json"
    hard_link.hardlink_to(source)
    with pytest.raises(ValueError, match="input file"):
        prepare_dataset(source, hard_link, overwrite=True)
    with pytest.raises(ValueError, match="audio file"):
        prepare_dataset(source, audio, overwrite=True)
    assert audio.read_bytes() == b"test audio"


@pytest.mark.parametrize("limit", [0, -1, True])
def test_limit_must_be_positive(tmp_path, audio, limit):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": "hello"})

    with pytest.raises(ValueError, match="positive integer"):
        prepare_dataset(source, tmp_path / "output.jsonl", limit=limit)


@pytest.mark.parametrize("contents", ["", "[]", "42", '{"audio":'])
def test_invalid_or_empty_input(tmp_path, contents):
    source = tmp_path / "source.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        prepare_dataset(source, tmp_path / "output.jsonl")


def test_invalid_jsonl_reports_line_and_preserves_output(tmp_path, audio):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"audio": str(audio), "text": "hello"}) + "\n{bad}\n",
        encoding="utf-8",
    )
    output = tmp_path / "output.jsonl"

    with pytest.raises(ValueError, match="Line 2"):
        prepare_dataset(source, output)
    assert not output.exists()


def test_prepare_data_cli_arguments(tmp_path, audio, capsys):
    source = write_input(tmp_path / "source.json", {"audio": str(audio), "text": "hello"})
    output = tmp_path / "prepared.jsonl"
    parser = argparse.ArgumentParser()
    add_prepare_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(
        [
            "prepare-data",
            "--input",
            str(source),
            "--output",
            str(output),
            "--prompt",
            "<audio>Transcribe.",
            "--response-prefix",
            "[caption:en] ",
            "--response-prefix-pattern",
            r"^\[caption:[a-z]+\] ",
            "--limit",
            "1",
        ]
    )

    assert args.func is run_prepare
    assert args.func(args) == 0
    assert "Prepared 1 records" in capsys.readouterr().out
    assert read_output(output)[0]["messages"][0]["content"] == "<audio>Transcribe."
    assert read_output(output)[0]["messages"][-1]["content"] == "[caption:en] hello"
