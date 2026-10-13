"""`tlv mcp` serves the TypeScript tool's MCP tools plus the new ones over stdio; results carry
structured content, text, and the image for frames; failures and progress reach the client."""

import base64
import os
import sys
from pathlib import Path

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from tlv import core, mcp_server

TS_TOOLS = {
    "video_add",
    "video_list",
    "video_status",
    "video_search",
    "video_ask",
    "video_find",
    "video_cut",
    "video_frames",
    "video_entity_add",
    "video_entities",
    "video_jockey",
}
NEW_TOOLS = {
    "video_index",
    "video_timeline",
    "video_entity_name",
    "video_entity_same",
    "video_entity_different",
    "video_person",
}
NEW_TOOLS |= {"video_at", "video_more"}
NEW_TOOLS |= {"video_reel"}


def test_stdio_server_lists_the_tools_and_answers_video_list(video, home):
    tlv = Path(sys.executable).parent / "tlv"
    params = StdioServerParameters(command=str(tlv), args=["mcp"], env={**os.environ, "TLV_HOME": str(home)})

    async def session():
        async with Client(params) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            return tools, await client.call_tool("video_list", {})

    tools, listed = anyio.run(session)
    assert tools == TS_TOOLS | NEW_TOOLS
    assert [v["video_id"] for v in listed.structured_content["videos"]] == [video]
    assert listed.content[0].text.startswith(f"{video}  ready  00:00:10.0")


def call(name, arguments, **options):
    async def session():
        async with Client(mcp_server.server("tlv")) as client:
            return await client.call_tool(name, arguments, **options)

    return anyio.run(session)


def test_frames_return_the_contact_sheet_as_an_image(video):
    result = call("video_frames", {"video_id": video, "start": 2, "end": 6, "count": 4})
    text, image = result.content
    sheet = Path(result.structured_content["sheet_path"])
    assert text.text.splitlines()[1] == f"sheet: {sheet}"
    assert (image.mime_type, base64.b64decode(image.data)) == ("image/jpeg", sheet.read_bytes())


def test_a_failure_reaches_the_client_with_its_reason(video):
    result = call("video_cut", {"video_id": video, "start": 12, "end": 15})
    assert result.is_error
    assert "Empty range 00:00:12.0 -> 00:00:15.0" in result.content[0].text


def test_progress_messages_become_progress_notifications(monkeypatch, home):
    def add(index, source, title=None, wait=True, progress=None):
        progress("downloading")
        progress("indexing: speech")
        return {"video_id": "v", "title": "T", "duration": 5.0, "status": "ready", "local_path": source}

    monkeypatch.setattr(core, "video_add", add)
    monkeypatch.setattr(core, "add_text", lambda result: result["status"])
    seen = []

    async def record(progress, total, message):
        seen.append((progress, message))

    call("video_add", {"source": "/x.mp4"}, progress_callback=record)
    assert seen == [(1, "downloading"), (2, "indexing: speech")]
