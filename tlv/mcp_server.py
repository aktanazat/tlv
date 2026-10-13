"""The MCP server over stdio: one tool per tlv operation, with the tool names and input fields of
the Twelve Labs client it replaces, plus the entity corrections, the timeline and a person's
timeline. Each tool returns its result as structured content plus the short text rendering;
video_cut, video_frames and video_person add the frame, contact sheet or filmstrip as an image.
Long tools send progress notifications when the client passes a progress token. Operations run in
worker threads, so a long one never blocks the others."""

from __future__ import annotations

import base64
import functools
import importlib.metadata
import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import anyio.from_thread
import anyio.lowlevel
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ImageContent, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from tlv import core

Index = Annotated[str | None, Field(min_length=1, description='Index name (default: $TLV_INDEX, else "tlv")')]
VideoId = Annotated[str, Field(min_length=1, description="video_id from video_list, video_search or video_find")]
Seconds = Annotated[float, Field(ge=0, description="Seconds")]
Query = Annotated[str, Field(min_length=1, description='What happens, e.g. "@LeBron_James steals the ball"')]
Then = Annotated[
    list[Annotated[str, Field(min_length=1)]] | None,
    Field(description='Ordered follow-up steps that happen after query, e.g. ["he dunks", "he celebrates"]'),
]
VideoIds = Annotated[list[Annotated[str, Field(min_length=1)]] | None, Field(description="Only these videos")]
EntityRef = Annotated[str, Field(min_length=1, description="entity_id from video_entities, or an appearance id")]

READ = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


class FromVideo(BaseModel):
    video_id: VideoId
    times: Annotated[list[Annotated[float, Field(ge=0)]], Field(min_length=1)] = Field(
        description="Seconds where the person is clearly visible"
    )


def _reply(result: dict[str, Any], text: str, image: str | None = None) -> CallToolResult:
    content: list[TextContent | ImageContent] = [TextContent(type="text", text=text)]
    if image:
        data = base64.b64encode(Path(image).read_bytes()).decode()
        content.append(ImageContent(type="image", data=data, mime_type="image/jpeg"))
    return CallToolResult(content=content, structured_content=result)


async def _run(op: Callable[..., dict[str, Any]], ctx: Context | None = None, **kwargs: Any) -> dict[str, Any]:
    """Runs a core operation in a worker thread. With `ctx`, its progress messages become MCP
    progress notifications, from whichever thread the operation reports on. A failure reaches the
    client as a tool error carrying its message, as on the command line."""
    if ctx is not None:
        token = anyio.lowlevel.current_token()
        steps = itertools.count(1)

        def progress(message: str) -> None:
            anyio.from_thread.run(ctx.report_progress, next(steps), None, message, token=token)

        kwargs["progress"] = progress
    try:
        return await anyio.to_thread.run_sync(functools.partial(op, **kwargs))
    except Exception as error:
        raise ToolError(str(error)) from error


def server(default_index: str) -> MCPServer:
    """The tlv tools; a call without `index` uses `default_index`."""
    mcp = MCPServer(name="tlv", version=importlib.metadata.version("tlv"))

    def pick(index: str | None) -> str:
        return index or default_index

    @mcp.tool(
        name="video_add",
        title="Add a video",
        description="Add a video to the index: a local file, a YouTube (or other yt-dlp) URL, or a direct video link. "
        "Downloads it, then indexes it on this Mac: camera cuts, frames, speech, on-screen text, people, sounds and "
        "captions. Safe to repeat: the same content returns the existing video. Waits for indexing (minutes) unless "
        "wait is false, when indexing goes on in the background; video_status shows the stages. Returns video_id, "
        "title, duration, status and the local path.",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        ),
    )
    async def video_add(
        ctx: Context,
        source: Annotated[str, Field(min_length=1, description="Local file path or URL")],
        title: Annotated[str | None, Field(description="Defaults to the video's own title or file name")] = None,
        wait: Annotated[bool, Field(description="Index before returning (default true)")] = True,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(core.video_add, ctx, index=pick(index), source=source, title=title, wait=wait)
        return _reply(result, core.add_text(result))

    @mcp.tool(
        name="video_list",
        title="List videos",
        description="List the videos in the index: id, title, length, indexing status, and whether the local copy "
        "exists.",
        annotations=READ,
    )
    async def video_list(index: Index = None) -> CallToolResult:
        result = await _run(core.video_list, index=pick(index))
        return _reply(result, core.list_text(result))

    @mcp.tool(
        name="video_status",
        title="Video status",
        description="Indexing status of one video: each stage (done, stale or pending) with its seconds, the row "
        "counts it produced, and the local file.",
        annotations=READ,
    )
    async def video_status(video_id: VideoId, index: Index = None) -> CallToolResult:
        result = await _run(core.video_status, index=pick(index), video_id=video_id)
        return _reply(result, core.status_text(result))

    @mcp.tool(
        name="video_index",
        title="Index a video",
        description="Run the indexing stages of a video that are missing or out of date (all, or only the named "
        "stages; force redoes them). Returns the video's status afterwards.",
        annotations=WRITE,
    )
    async def video_index(
        ctx: Context,
        video_id: VideoId,
        stages: Annotated[
            list[Literal["shots", "frames", "speech", "vision", "sounds", "spans", "entities"]] | None,
            Field(min_length=1, description="Only these stages"),
        ] = None,
        force: Annotated[bool, Field(description="Redo the stages even when they are current")] = False,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(core.video_index, ctx, index=pick(index), video_id=video_id, stages=stages, force=force)
        return _reply(result, core.status_text(result))

    @mcp.tool(
        name="video_search",
        title="Search videos",
        description="Fast search over the indexed videos: captions, speech, on-screen text and frames. Returns likely "
        "moments with timecodes, not verified ones. For an ordered action, pass the first step as query and the next "
        "steps as then. @Name means a person from video_entities. Use video_find when the clip must be right.",
        annotations=READ,
    )
    async def video_search(
        query: Query,
        then: Then = None,
        modalities: Annotated[
            list[Literal["visual", "audio", "transcription"]] | None,
            Field(min_length=1, description="What to match against (default all)"),
        ] = None,
        max_gap: Annotated[
            float, Field(ge=0, description="Most seconds between one step's end and the next step's start")
        ] = 12,
        limit: Annotated[int, Field(ge=1, le=50, description="Most results (default 10)")] = 10,
        video_ids: VideoIds = None,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(
            core.video_search,
            index=pick(index),
            query=query,
            then=then or [],
            modalities=modalities,
            max_gap=max_gap,
            limit=limit,
            video_ids=video_ids,
        )
        return _reply(result, core.search_text(result))

    @mcp.tool(
        name="video_ask",
        title="Ask about a video",
        description="Ask the vision-language model a question about one video, or about start..end of it, from its "
        "frames and transcript. Returns text, or JSON matching json_schema when given.",
        annotations=READ,
    )
    async def video_ask(
        video_id: VideoId,
        question: Annotated[str, Field(min_length=1)],
        start: Annotated[float | None, Field(ge=0, description="Window start in seconds")] = None,
        end: Annotated[float | None, Field(ge=0, description="Window end in seconds")] = None,
        json_schema: Annotated[dict[str, Any] | None, Field(description="JSON Schema the answer must follow")] = None,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(
            core.video_ask,
            index=pick(index),
            video_id=video_id,
            question=question,
            start=start,
            end=end,
            json_schema=json_schema,
        )
        return _reply(result, core.ask_text(result))

    @mcp.tool(
        name="video_find",
        title="Find verified clips",
        description="Verified clip finder: searches, then looks at the frames of every candidate with their true "
        "times and confirms the claim, the person and the step order. Replays of the same moment are merged into one "
        "clip. Slower than video_search. Returns confirmed clips with evidence (frames, transcript, on-screen text) "
        "and cut files, plus rejected candidates with reasons.",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False),
    )
    async def video_find(
        ctx: Context,
        query: Query,
        then: Then = None,
        max_candidates: Annotated[int, Field(ge=1, le=50, description="Candidates to check (default 8)")] = 8,
        pad: Annotated[float, Field(ge=0, description="Seconds added around each candidate (default 3)")] = 3,
        min_confidence: Annotated[
            float, Field(ge=0, le=1, description="Lowest confidence to confirm (default 0.6)")
        ] = 0.6,
        cut: Annotated[bool, Field(description="Cut each confirmed clip with a contact sheet (default true)")] = True,
        video_ids: VideoIds = None,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(
            core.video_find,
            ctx,
            index=pick(index),
            query=query,
            then=then or [],
            max_candidates=max_candidates,
            pad=pad,
            min_confidence=min_confidence,
            cut=cut,
            video_ids=video_ids,
        )
        return _reply(result, core.find_text(result))

    @mcp.tool(
        name="video_cut",
        title="Cut a clip",
        description="Cut start..end of a video into an mp4 from its local copy, with its middle frame or a contact "
        "sheet returned as an image.",
        annotations=WRITE,
    )
    async def video_cut(
        video_id: VideoId,
        start: Seconds,
        end: Seconds,
        frame: Annotated[
            Literal["mid", "sheet", "none"], Field(description="Middle frame (default), 6-frame sheet, or none")
        ] = "mid",
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(core.video_cut, index=pick(index), video_id=video_id, start=start, end=end, frame=frame)
        return _reply(result, core.cut_text(result), result.get("frame_path"))

    @mcp.tool(
        name="video_frames",
        title="Look at a range",
        description="Contact sheet of evenly spaced frames from start..end of a video, returned as an image, without "
        "cutting a clip.",
        annotations=READ,
    )
    async def video_frames(
        video_id: VideoId,
        start: Seconds,
        end: Seconds,
        count: Annotated[int, Field(ge=1, le=12, description="Frames (default 6)")] = core.SHEET_FRAMES,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(core.video_frames, index=pick(index), video_id=video_id, start=start, end=end, count=count)
        return _reply(result, core.frames_text(result), result["sheet_path"])

    @mcp.tool(
        name="video_timeline",
        title="Evidence in a range",
        description="Everything indexed about start..end of a video, each with its times: speech word by word, "
        "on-screen text (scores, names), sounds, the captions of the spans it touches (marked when a replay), and "
        "who is visible. Use it to check a claim.",
        annotations=READ,
    )
    async def video_timeline(video_id: VideoId, start: Seconds, end: Seconds, index: Index = None) -> CallToolResult:
        result = await _run(core.video_timeline, index=pick(index), video_id=video_id, start=start, end=end)
        return _reply(result, core.timeline_text(result))

    @mcp.tool(
        name="video_entity_add",
        title="Teach a person",
        description="Create or extend a person (entity) from reference images (paths or URLs) or from frames of an "
        "indexed video at given times. Afterwards @Name in video_search and video_find queries means this person.",
        annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True),
    )
    async def video_entity_add(
        name: Annotated[str, Field(min_length=1, description='Name as written after @, e.g. "LeBron James"')],
        images: Annotated[
            list[Annotated[str, Field(min_length=1)]] | None,
            Field(description="Image paths or URLs showing the person clearly"),
        ] = None,
        from_: Annotated[FromVideo | None, Field(alias="from")] = None,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(
            core.entity_add,
            index=pick(index),
            name=name,
            images=images or [],
            video_id=from_.video_id if from_ else None,
            times=from_.times if from_ else [],
        )
        return _reply(result, core.entity_add_text(result))

    @mcp.tool(
        name="video_entities",
        title="List people",
        description="List the people (entities) found in the index's videos, named or not: entity_id, name, how "
        "often and in how many videos they appear, and a face crop. @Name in queries refers to a named one.",
        annotations=READ,
    )
    async def video_entities(index: Index = None) -> CallToolResult:
        result = await _run(core.video_entities, index=pick(index))
        return _reply(result, core.entities_text(result))

    @mcp.tool(
        name="video_entity_name",
        title="Name a person",
        description="Name the person an entity or appearance id belongs to. The name is kept across re-clustering. "
        "Returns the person.",
        annotations=WRITE,
    )
    async def video_entity_name(
        target: EntityRef,
        name: Annotated[str, Field(min_length=1, description='e.g. "LeBron James"')],
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(core.entity_name, index=pick(index), target=target, name=name)
        return _reply(result, core.corrected_text(result))

    @mcp.tool(
        name="video_entity_same",
        title="Same person",
        description="Record that two entity or appearance ids are the same person, and merge them. Kept across "
        "re-clustering. Returns the merged person.",
        annotations=WRITE,
    )
    async def video_entity_same(a: EntityRef, b: EntityRef, index: Index = None) -> CallToolResult:
        result = await _run(core.entity_same, index=pick(index), a=a, b=b)
        return _reply(result, core.corrected_text(result))

    @mcp.tool(
        name="video_entity_different",
        title="Different people",
        description="Record that two entity or appearance ids are different people, and split them. Kept across "
        "re-clustering. Returns both people.",
        annotations=WRITE,
    )
    async def video_entity_different(a: EntityRef, b: EntityRef, index: Index = None) -> CallToolResult:
        result = await _run(core.entity_different, index=pick(index), a=a, b=b)
        return _reply(result, core.corrected_text(result))

    @mcp.tool(
        name="video_person",
        title="A person's timeline",
        description="One person across every video, from the index alone (no reference photos, no model): each "
        "stretch they are on screen with its clearest crop, seconds on screen per video, where their name is said "
        "or shown on screen, and the moments video_find verified about them. strip adds a filmstrip still of one "
        "crop per stretch, returned as an image.",
        annotations=READ,
    )
    async def video_person(
        who: Annotated[
            str, Field(min_length=1, description='"@LeBron_James", "LeBron" or an entity_id from video_entities')
        ],
        strip: Annotated[bool, Field(description="Also return a filmstrip still, one crop per stretch")] = False,
        video_ids: VideoIds = None,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(core.person, index=pick(index), who=who, strip=strip, video_ids=video_ids)
        return _reply(result, core.person_text(result), result.get("strip_path"))

    @mcp.tool(
        name="video_jockey",
        title="Ask about the collection",
        description="Answer a question about the whole collection: plans, searches, checks clips and answers in prose "
        "with citations to videos and times. Slower; use it for corpus-level questions, video_find for verified "
        "clips. Pass session_id from an earlier answer to follow up.",
        annotations=READ,
    )
    async def video_jockey(
        ctx: Context,
        question: Annotated[str, Field(min_length=1)],
        session_id: Annotated[
            str | None, Field(min_length=1, description="session_id of an earlier answer, to follow up")
        ] = None,
        video_ids: VideoIds = None,
        index: Index = None,
    ) -> CallToolResult:
        result = await _run(
            core.video_jockey, ctx, index=pick(index), question=question, session_id=session_id, video_ids=video_ids
        )
        return _reply(result, core.jockey_text(result))

    return mcp


def serve() -> None:
    """Serves the tools over stdio until the client disconnects."""
    server(core.default_index()).run("stdio")
