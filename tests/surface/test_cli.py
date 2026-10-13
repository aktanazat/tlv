"""The command line reads the same commands and flags as the TypeScript tool: each command line
becomes one operation call, and each mistake exits with status 2 and says what is wrong."""

import shlex

import pytest

from tlv import cli


@pytest.fixture(autouse=True)
def _no_index_env(monkeypatch):
    monkeypatch.delenv("TLV_INDEX", raising=False)


@pytest.mark.parametrize(
    ("line", "name", "kwargs"),
    [
        ('add ~/v.mp4 --title "Game 7" --no-wait', "add", {"source": "~/v.mp4", "title": "Game 7", "wait": False}),
        ("add https://youtu.be/x", "add", {"source": "https://youtu.be/x", "wait": True}),
        ("list", "list", {}),
        ("status v1", "status", {"video_id": "v1"}),
        (
            "index v1 --stage speech --stage spans --force",
            "index",
            {"video_id": "v1", "stages": ["speech", "spans"], "force": True},
        ),
        ("index v1", "index", {"video_id": "v1", "force": False}),
        (
            'search lebron scores --then celebrates --then "runs back" --modality visual --modality transcription '
            "--max-gap 8 --limit 5 --video a --video b",
            "search",
            {
                "query": "lebron scores",
                "then": ["celebrates", "runs back"],
                "modalities": ["visual", "transcription"],
                "max_gap": 8.0,
                "limit": 5,
                "video_ids": ["a", "b"],
            },
        ),
        (
            'ask v1 who wins? --from 1:23.4 --to 00:01:30 --schema \'{"type": "object"}\'',
            "ask",
            {"video_id": "v1", "question": "who wins?", "start": 83.4, "end": 90.0, "json_schema": {"type": "object"}},
        ),
        (
            "find dunk --then flex --max-candidates 4 --pad 2.5 --min-confidence 0.7 --no-cut --video v1",
            "find",
            {
                "query": "dunk",
                "then": ["flex"],
                "max_candidates": 4,
                "pad": 2.5,
                "min_confidence": 0.7,
                "cut": False,
                "video_ids": ["v1"],
            },
        ),
        ("find dunk", "find", {"query": "dunk", "cut": True}),
        (
            "cut v1 --from 4:15 --to 4:22 --frame sheet",
            "cut",
            {"video_id": "v1", "start": 255.0, "end": 262.0, "frame": "sheet"},
        ),
        (
            "--from=10 frames v1 --count 4 --to 12.5",
            "frames",
            {"video_id": "v1", "start": 10.0, "end": 12.5, "count": 4},
        ),
        ("timeline v1 --from 4:15 --to 4:22", "timeline", {"video_id": "v1", "start": 255.0, "end": 262.0}),
        (
            "entity-add LeBron James --video v1 --at 83.4 --at 7:00 --image a.jpg",
            "entity-add",
            {"name": "LeBron James", "images": ["a.jpg"], "video_id": "v1", "times": [83.4, 420.0]},
        ),
        ("entity-add Coco --image https://x/c.png", "entity-add", {"name": "Coco", "images": ["https://x/c.png"]}),
        ("entities", "entities", {}),
        ("entity-name ent-3 LeBron James", "entity-name", {"target": "ent-3", "name": "LeBron James"}),
        ("entity-same ent-3 app-9", "entity-same", {"a": "ent-3", "b": "app-9"}),
        ("entity-different ent-3 ent-4", "entity-different", {"a": "ent-3", "b": "ent-4"}),
        ("person @LeBron --video v1 --strip", "person", {"who": "@LeBron", "video_ids": ["v1"], "strip": True}),
        (
            "jockey who scored most? --session 0123456789ab --video v1",
            "jockey",
            {"question": "who scored most?", "session_id": "0123456789ab", "video_ids": ["v1"]},
        ),
        ("search dunk --clock Q4", "search", {"query": "dunk", "clock": "Q4"}),
        ('find dunk --clock "last 2 min"', "find", {"query": "dunk", "cut": True, "clock": "last 2 min"}),
        ("more now only fourth quarter", "more", {"instruction": "now only fourth quarter"}),
        ("at v1 4th 1:12 --cut", "at", {"video_id": "v1", "clock": "4th 1:12", "cut": True}),
        ('at v1 "Q4 2:00-1:00"', "at", {"video_id": "v1", "clock": "Q4 2:00-1:00", "cut": False}),
    ],
)
def test_a_command_line_becomes_one_operation_call(line, name, kwargs):
    call = cli.parse(shlex.split(line))
    assert (call.name, call.kwargs) == (name, {"index": "tlv", **kwargs})


@pytest.mark.parametrize(
    ("line", "env", "index", "as_json"),
    [
        ("list", None, "tlv", False),
        ("list", "games", "games", False),
        ("list --index practice --json", "games", "practice", True),
    ],
)
def test_index_and_json_apply_to_every_command(monkeypatch, line, env, index, as_json):
    if env:
        monkeypatch.setenv("TLV_INDEX", env)
    call = cli.parse(shlex.split(line))
    assert (call.kwargs["index"], call.json) == (index, as_json)


@pytest.mark.parametrize(
    ("line", "message", "hint"),
    [
        ("cut v1 --from 4:15 --to x", '--to: "x" is not a time', "usage: tlv cut"),
        ("cut v1 --to 5", "--from is required", "usage: tlv cut"),
        ("cut v1 --from 1 --to 2 --frame all", "--frame must be one of mid, sheet, none", "usage: tlv cut"),
        ("cut v1 --from 1 --to 2 --pad 3", "cut does not take --pad", "usage: tlv cut"),
        ("timeline v1 --from 1", "--to is required", "usage: tlv timeline"),
        ("search x --limit 2.5", '--limit needs a whole number, got "2.5"', "usage: tlv search"),
        ("search x --modality smell", "--modality must be one of visual, audio, transcription", "usage: tlv search"),
        ("find x --pad abc", '--pad needs a number, got "abc"', "usage: tlv find"),
        ("find x --min-confidence nan", '--min-confidence needs a number, got "nan"', "usage: tlv find"),
        ("ask v1 q --schema [1]", "--schema must be a JSON object", "usage: tlv ask"),
        ("ask v1 q --schema {", "--schema must be inline JSON", "usage: tlv ask"),
        ("ask v1", "expected a question", "usage: tlv ask"),
        ("entity-add x --video v1", "--video and --at go together", "usage: tlv entity-add"),
        ("entity-add x --video a --video b --at 1", "entity-add takes one --video", "usage: tlv entity-add"),
        ("status", "expected a video_id", "usage: tlv status"),
        ("status a b", "expected a video_id", "usage: tlv status"),
        ("list extra", 'unexpected argument "extra"', "usage: tlv list"),
        ("entity-same a", "expected two entity or appearance ids", "usage: tlv entity-same"),
        ("person --strip", "expected a person: @Name, a name or an entity id", "usage: tlv person"),
        ("more", 'expected a follow-up, like "now only the 4th quarter"', "usage: tlv more"),
        ("at", "expected a video_id", "usage: tlv at"),
        ("at v1 --cut", 'expected a game clock, like "4th 1:12"', "usage: tlv at"),
        ("find x --cut", "find does not take --cut", "usage: tlv find"),
        ("mcp --json", "mcp takes no arguments", "usage: tlv mcp"),
        ("frobnicate", 'unknown command "frobnicate"', "Run tlv --help for usage."),
        ("list --bogus", "unrecognized arguments: --bogus", "Run tlv --help for usage."),
        ("search x --lim 3", "unrecognized arguments: --lim", "Run tlv --help for usage."),
    ],
)
def test_mistakes_exit_2_and_say_what_is_wrong(capsys, line, message, hint):
    assert cli.main(shlex.split(line)) == 2
    err = capsys.readouterr().err
    assert f"tlv: {message}" in err
    assert err.rstrip().splitlines()[-1].startswith(hint)


@pytest.mark.parametrize(
    ("line", "code", "stdout_starts", "stderr_starts"),
    [
        ("--help", 0, "tlv: find, verify and cut", ""),
        ("help", 0, "tlv: find, verify and cut", ""),
        ("", 2, "", "tlv: find, verify and cut"),
        ("cut -h", 0, "usage: tlv cut <video_id> --from T --to T", ""),
        ("mcp --help", 0, "usage: tlv mcp", ""),
    ],
)
def test_help_goes_to_stdout_when_asked_and_to_stderr_otherwise(capsys, line, code, stdout_starts, stderr_starts):
    assert cli.main(shlex.split(line)) == code
    out, err = capsys.readouterr()
    assert (out[: len(stdout_starts)], err[: len(stderr_starts)]) == (stdout_starts, stderr_starts)


def test_help_lists_every_command():
    commands = cli.HELP.split("commands:")[1].split("options for every command:")[0]
    listed = {line.split()[0] for line in commands.splitlines() if line[2:3].isalpha()}
    assert listed == {*cli.COMMANDS, "mcp"}
