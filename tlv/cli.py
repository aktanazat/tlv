"""tlv: local video understanding for terminal agents. Each command runs one core operation;
`tlv mcp` serves the same operations as MCP tools over stdio.

The command line follows the TypeScript tool it replaces: every option parses anywhere on the
line, and a command given an option it does not take is a usage error (exit 2)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NoReturn

from tlv import core
from tlv.timecode import parse_time

HELP = """\
tlv: find, verify and cut moments in videos, on this Mac

usage: tlv <command> [arguments] [options]

commands:
  add <source>                 add a local file or a URL (YouTube, direct .mp4) and index it
      --title T                title (default: the video's own title or file name)
      --no-wait                return once the video is added; it indexes in the background
  list                         videos in the index
  status <video_id>            indexing stages and row counts of one video
  index <video_id>             run the indexing stages that are missing or out of date
      --stage S                only this stage (repeatable): shots, frames, speech, vision,
                               sounds, spans, entities
      --force                  redo the stages even when they are current
  search <query>               fast search: likely moments, not verified
      --then STEP              next step of an ordered action (repeatable)
      --modality M             visual, audio or transcription (repeatable; default all)
      --max-gap S              most seconds between steps (default 12)
      --limit N                most results (default 10)
      --video ID               search only this video (repeatable)
  ask <video_id> <question>    ask the vision-language model about a video
      --from T --to T          only this window
      --schema JSON            JSON Schema the answer must follow
  find <query>                 verified clips: search, then look at every candidate's frames
      --then STEP              next step (repeatable)
      --max-candidates N       candidates to check (default 8)
      --pad S                  seconds added around each candidate (default 3)
      --min-confidence C       0 to 1 (default 0.6)
      --no-cut                 do not cut the confirmed clips
      --video ID               search only this video (repeatable)
  cut <video_id> --from T --to T [--frame mid|sheet|none]
                               cut a clip from the local copy
  frames <video_id> --from T --to T [--count N]
                               contact sheet of a range
  timeline <video_id> --from T --to T
                               speech, on-screen text, sounds, captions and people in a range
  entity-add <name> [--image PATH|URL]... [--video ID --at T...]
                               teach a person for @Name in queries
  entities                     people found in the videos, named or not
  entity-name <id> <name>      name the person of an entity or appearance id
  entity-same <a> <b>          two entity or appearance ids are the same person
  entity-different <a> <b>     two entity or appearance ids are different people
  jockey <question> [--session ID] [--video ID]...
                               answer a question about the whole collection, citing clips
  mcp                          serve these commands as MCP tools over stdio

options for every command:
  --index NAME                 index (default: $TLV_INDEX, else tlv), stored in $TLV_HOME/NAME.db
  --json                       print the structured result as JSON
  -h, --help                   help

Times are seconds (83.4) or timecodes (1:23.4, 00:01:23.4).
Everything runs on this Mac; data lives in $TLV_HOME (default ~/.local/share/tlv).
"""

TEXT_OPTIONS = ("index", "title", "max-gap", "limit", "from", "to", "schema", "max-candidates", "pad")
TEXT_OPTIONS += ("min-confidence", "frame", "count", "session")
LIST_OPTIONS = ("then", "modality", "video", "image", "at", "stage")
FLAG_OPTIONS = ("json", "no-wait", "no-cut", "force")
COMMON_OPTIONS = ("index", "json", "help")


class UsageError(Exception):
    """A mistake in the command line: exits with status 2 and a pointer to the usage."""

    def __init__(self, message: str, hint: str = "Run tlv --help for usage.") -> None:
        super().__init__(message)
        self.hint = hint


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="tlv", add_help=False, allow_abbrev=False)
    parser.add_argument("-h", "--help", action="store_true", default=None)
    for name in TEXT_OPTIONS:
        parser.add_argument(f"--{name}")
    for name in LIST_OPTIONS:
        parser.add_argument(f"--{name}", action="append")
    for name in FLAG_OPTIONS:
        parser.add_argument(f"--{name}", action="store_true", default=None)
    parser.add_argument("positionals", nargs="*")
    return parser


Values = dict[str, Any]


def _one(args: list[str], what: str) -> str:
    if len(args) != 1:
        raise UsageError(f"expected {what}")
    return args[0]


def _words(args: list[str], what: str) -> str:
    text = " ".join(args).strip()
    if not text:
        raise UsageError(f"expected {what}")
    return text


def _none(args: list[str]) -> None:
    if args:
        raise UsageError(f'unexpected argument "{args[0]}"')


def _number(value: str | None, flag: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        number = float("nan")
    if number != number or number in (float("inf"), float("-inf")):
        raise UsageError(f'--{flag} needs a number, got "{value}"')
    return number


def _whole(value: str | None, flag: str) -> int | None:
    number = _number(value, flag)
    if number is not None and not number.is_integer():
        raise UsageError(f'--{flag} needs a whole number, got "{value}"')
    return None if number is None else int(number)


def _seconds(value: str | None, flag: str) -> float | None:
    if value is None:
        return None
    try:
        return parse_time(value)
    except ValueError as error:
        raise UsageError(f"--{flag}: {error}") from None


def _given[T](value: T | None, flag: str) -> T:
    if value is None:
        raise UsageError(f"--{flag} is required")
    return value


def _one_of(value: str, allowed: tuple[str, ...], flag: str) -> str:
    if value not in allowed:
        raise UsageError(f'--{flag} must be one of {", ".join(allowed)}, got "{value}"')
    return value


def _present(**kwargs: Any) -> dict[str, Any]:
    """The keyword arguments that were given; the operation's defaults cover the rest."""
    return {key: value for key, value in kwargs.items() if value is not None}


def _add(args: list[str], v: Values) -> dict[str, Any]:
    return _present(source=_one(args, "a file path or URL"), title=v.get("title"), wait=not v.get("no-wait"))


def _video_id(args: list[str], _v: Values) -> dict[str, Any]:
    return {"video_id": _one(args, "a video_id")}


def _list(args: list[str], _v: Values) -> dict[str, Any]:
    _none(args)
    return {}


def _index(args: list[str], v: Values) -> dict[str, Any]:
    """Stage names are checked by the pipeline, which owns them."""
    return _present(video_id=_one(args, "a video_id"), stages=v.get("stage"), force=bool(v.get("force")))


def _search(args: list[str], v: Values) -> dict[str, Any]:
    return _present(
        query=_words(args, "a query"),
        then=v.get("then"),
        modalities=[_one_of(m, core.MODALITIES, "modality") for m in v["modality"]] if "modality" in v else None,
        max_gap=_number(v.get("max-gap"), "max-gap"),
        limit=_whole(v.get("limit"), "limit"),
        video_ids=v.get("video"),
    )


def _ask(args: list[str], v: Values) -> dict[str, Any]:
    if not args:
        raise UsageError("expected a video_id")
    schema = None
    if "schema" in v:
        try:
            schema = json.loads(v["schema"])
        except json.JSONDecodeError:
            raise UsageError("--schema must be inline JSON") from None
        if not isinstance(schema, dict):
            raise UsageError("--schema must be a JSON object")
    return _present(
        video_id=args[0],
        question=_words(args[1:], "a question"),
        start=_seconds(v.get("from"), "from"),
        end=_seconds(v.get("to"), "to"),
        json_schema=schema,
    )


def _find(args: list[str], v: Values) -> dict[str, Any]:
    return _present(
        query=_words(args, "a query"),
        then=v.get("then"),
        max_candidates=_whole(v.get("max-candidates"), "max-candidates"),
        pad=_number(v.get("pad"), "pad"),
        min_confidence=_number(v.get("min-confidence"), "min-confidence"),
        cut=not v.get("no-cut"),
        video_ids=v.get("video"),
    )


def _range(args: list[str], v: Values) -> dict[str, Any]:
    return {
        "video_id": _one(args, "a video_id"),
        "start": _given(_seconds(v.get("from"), "from"), "from"),
        "end": _given(_seconds(v.get("to"), "to"), "to"),
    }


def _cut(args: list[str], v: Values) -> dict[str, Any]:
    frame = v.get("frame")
    return _present(**_range(args, v), frame=None if frame is None else _one_of(frame, core.FRAMES, "frame"))


def _frames(args: list[str], v: Values) -> dict[str, Any]:
    return _present(**_range(args, v), count=_whole(v.get("count"), "count"))


def _entity_add(args: list[str], v: Values) -> dict[str, Any]:
    videos = v.get("video", [])
    if len(videos) > 1:
        raise UsageError("entity-add takes one --video")
    times = [_seconds(t, "at") for t in v.get("at", ())]
    if bool(videos) != bool(times):
        raise UsageError("--video and --at go together")
    return _present(
        name=_words(args, "a name"), images=v.get("image"), video_id=videos[0] if videos else None, times=times or None
    )


def _entity_name(args: list[str], _v: Values) -> dict[str, Any]:
    if not args:
        raise UsageError("expected an entity or appearance id")
    return {"target": args[0], "name": _words(args[1:], "a name")}


def _pair(args: list[str], _v: Values) -> dict[str, Any]:
    if len(args) != 2:
        raise UsageError("expected two entity or appearance ids")
    return {"a": args[0], "b": args[1]}


def _jockey(args: list[str], v: Values) -> dict[str, Any]:
    return _present(question=_words(args, "a question"), session_id=v.get("session"), video_ids=v.get("video"))


@dataclass(frozen=True)
class Command:
    usage: str
    options: tuple[str, ...]
    """Options this command takes besides --index, --json and --help."""
    parse: Callable[[list[str], Values], dict[str, Any]]
    run: Callable[..., dict[str, Any]]
    render: Callable[[dict[str, Any]], str]
    reports_progress: bool = False


COMMANDS: dict[str, Command] = {
    "add": Command(
        "tlv add <source> [--title T] [--no-wait]",
        ("title", "no-wait"),
        _add,
        core.video_add,
        core.add_text,
        reports_progress=True,
    ),
    "list": Command("tlv list", (), _list, core.video_list, core.list_text),
    "status": Command("tlv status <video_id>", (), _video_id, core.video_status, core.status_text),
    "index": Command(
        "tlv index <video_id> [--stage S]... [--force]",
        ("stage", "force"),
        _index,
        core.video_index,
        core.status_text,
        reports_progress=True,
    ),
    "search": Command(
        "tlv search <query> [--then STEP]... [--modality M]... [--max-gap S] [--limit N] [--video ID]...",
        ("then", "modality", "max-gap", "limit", "video"),
        _search,
        core.video_search,
        core.search_text,
    ),
    "ask": Command(
        "tlv ask <video_id> <question> [--from T] [--to T] [--schema JSON]",
        ("from", "to", "schema"),
        _ask,
        core.video_ask,
        core.ask_text,
    ),
    "find": Command(
        "tlv find <query> [--then STEP]... [--max-candidates N] [--pad S] [--min-confidence C] [--no-cut]"
        " [--video ID]...",
        ("then", "max-candidates", "pad", "min-confidence", "no-cut", "video"),
        _find,
        core.video_find,
        core.find_text,
        reports_progress=True,
    ),
    "cut": Command(
        "tlv cut <video_id> --from T --to T [--frame mid|sheet|none]",
        ("from", "to", "frame"),
        _cut,
        core.video_cut,
        core.cut_text,
    ),
    "frames": Command(
        "tlv frames <video_id> --from T --to T [--count N]",
        ("from", "to", "count"),
        _frames,
        core.video_frames,
        core.frames_text,
    ),
    "timeline": Command(
        "tlv timeline <video_id> --from T --to T",
        ("from", "to"),
        _range,
        core.video_timeline,
        core.timeline_text,
    ),
    "entity-add": Command(
        "tlv entity-add <name> [--image PATH|URL]... [--video ID --at T...]",
        ("image", "video", "at"),
        _entity_add,
        core.entity_add,
        core.entity_add_text,
    ),
    "entities": Command("tlv entities", (), _list, core.video_entities, core.entities_text),
    "entity-name": Command("tlv entity-name <id> <name>", (), _entity_name, core.entity_name, core.corrected_text),
    "entity-same": Command("tlv entity-same <a> <b>", (), _pair, core.entity_same, core.corrected_text),
    "entity-different": Command("tlv entity-different <a> <b>", (), _pair, core.entity_different, core.corrected_text),
    "jockey": Command(
        "tlv jockey <question> [--session ID] [--video ID]...",
        ("session", "video"),
        _jockey,
        core.video_jockey,
        core.jockey_text,
        reports_progress=True,
    ),
}


@dataclass(frozen=True)
class Call:
    """A parsed command line: the operation, its keyword arguments, and how to print the result."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    json: bool = False


@dataclass(frozen=True)
class Help:
    text: str
    asked: bool


def parse(argv: list[str]) -> Call | Help:
    """Reads a command line into the call it asks for. Raises UsageError for a mistake."""
    values = {key.replace("_", "-"): value for key, value in vars(_parser().parse_intermixed_args(argv)).items()}
    positionals = values.pop("positionals")
    values = {key: value for key, value in values.items() if value is not None}
    name, args = (positionals[0], positionals[1:]) if positionals else (None, [])
    if name is None or name == "help":
        return Help(HELP, asked=bool(values.get("help")) or name == "help")
    if name == "mcp":
        if values.get("help"):
            return Help("usage: tlv mcp (serves the tlv tools over stdio)\n", asked=True)
        if args or values:
            raise UsageError("mcp takes no arguments", "usage: tlv mcp")
        return Call("mcp")
    command = COMMANDS.get(name)
    if command is None:
        raise UsageError(f'unknown command "{name}"')
    if values.get("help"):
        return Help(f"usage: {command.usage}\n", asked=True)
    hint = f"usage: {command.usage}"
    unexpected = [key for key in values if key not in COMMON_OPTIONS and key not in command.options]
    if unexpected:
        raise UsageError(f"{name} does not take {', '.join(f'--{key}' for key in unexpected)}", hint)
    try:
        kwargs = command.parse(args, values)
    except UsageError as error:
        raise UsageError(str(error), hint) from None
    index = values.get("index") or core.default_index()
    return Call(name, {"index": index, **kwargs}, json=bool(values.get("json")))


def _progress(message: str) -> None:
    print(f"tlv: {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    try:
        call = parse(sys.argv[1:] if argv is None else argv)
    except UsageError as error:
        print(f"tlv: {error}\n{error.hint}", file=sys.stderr)
        return 2
    if isinstance(call, Help):
        print(call.text, end="", file=sys.stdout if call.asked else sys.stderr)
        return 0 if call.asked else 2
    if call.name == "mcp":
        from tlv import mcp_server

        mcp_server.serve()
        return 0
    command = COMMANDS[call.name]
    kwargs = {**call.kwargs, "progress": _progress} if command.reports_progress else call.kwargs
    try:
        result = command.run(**kwargs)
    except Exception as error:  # the command line's boundary: any failure becomes a one-line message
        print(f"tlv: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False) if call.json else command.render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
