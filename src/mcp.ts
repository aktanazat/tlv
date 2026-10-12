// The MCP server over stdio: one tool per core operation. Each tool returns its result as
// structured content plus a short text rendering; video_cut and video_frames also return
// the frame image so the agent can see it.

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type { RequestHandlerExtra } from "@modelcontextprotocol/sdk/shared/protocol.js";
import type { CallToolResult, ServerNotification, ServerRequest } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import packageJson from "../package.json" with { type: "json" };
import * as core from "./core.ts";

type Extra = RequestHandlerExtra<ServerRequest, ServerNotification>;

/** Sends progress notifications when the client asked for them with a progress token. */
function contextFor(index: string | undefined, extra: Extra): core.Context {
  const token = extra._meta?.progressToken;
  if (token === undefined) return { index };
  let step = 0;
  return {
    index,
    progress: (message) => {
      step += 1;
      extra
        .sendNotification({ method: "notifications/progress", params: { progressToken: token, progress: step, message } })
        .catch((error: unknown) => console.error(`tlv: progress notification failed: ${error}`));
    },
  };
}

async function reply(result: Record<string, unknown>, text: string, imagePath?: string): Promise<CallToolResult> {
  const content: CallToolResult["content"] = [{ type: "text", text }];
  if (imagePath) {
    const data = Buffer.from(await Bun.file(imagePath).arrayBuffer()).toString("base64");
    content.push({ type: "image", data, mimeType: "image/jpeg" });
  }
  return { content, structuredContent: result };
}

const index = z.string().min(1).optional().describe('Index name (default: $TLV_INDEX, else "tlv")');
const videoId = z.string().min(1).describe("video_id from video_list, video_search or video_find");
const seconds = z.number().nonnegative();
const query = z.string().min(1).describe('What happens, e.g. "@LeBron_James steals the ball"');
const then = z
  .array(z.string().min(1))
  .optional()
  .describe('Ordered follow-up steps that happen after query, e.g. ["he dunks", "he celebrates"]');

export async function serveMcp(): Promise<void> {
  const server = new McpServer({ name: "tlv", version: packageJson.version });

  server.registerTool(
    "video_add",
    {
      title: "Add a video",
      description:
        "Add a video to the index: a local file, a YouTube (or other yt-dlp) URL, or a direct video link. " +
        "Downloads, uploads, indexes, and adds it to the Jockey store. Safe to repeat: the same source returns " +
        "the existing video and restores a missing local copy. Waits for indexing (minutes) unless wait is false.",
      inputSchema: {
        source: z.string().min(1).describe("Local file path or URL"),
        title: z.string().optional().describe("Defaults to the video's own title or file name"),
        wait: z.boolean().optional().describe("Wait until indexing finishes (default true)"),
        index,
      },
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoAdd(input, contextFor(name, extra));
      return reply(result, core.addText(result));
    },
  );

  server.registerTool(
    "video_list",
    {
      title: "List videos",
      description: "List the videos in the index: id, title, length, indexing status, and whether the local copy exists.",
      inputSchema: { index },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ index: name }, extra) => {
      const result = await core.videoList(contextFor(name, extra));
      return reply(result, core.listText(result));
    },
  );

  server.registerTool(
    "video_status",
    {
      title: "Video status",
      description: "Indexing status of one video (for adds with wait=false), its Jockey store status, and its local file.",
      inputSchema: { video_id: videoId, index },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoStatus(input, contextFor(name, extra));
      return reply(result, core.statusText(result));
    },
  );

  server.registerTool(
    "video_search",
    {
      title: "Search videos",
      description:
        "Fast semantic search (Marengo) over the indexed videos. Returns likely moments with timecodes, not verified ones. " +
        "For an ordered action, pass the first step as query and the next steps as then. @Name means a person added " +
        "with video_entity_add. Use video_find when the clip must be right.",
      inputSchema: {
        query,
        then,
        modalities: z
          .array(z.enum(["visual", "audio", "transcription"]))
          .min(1)
          .optional()
          .describe("What to match against (default visual and audio)"),
        max_gap: seconds.optional().describe("Most seconds between one step's end and the next step's start (default 12)"),
        limit: z.number().int().min(1).max(50).optional().describe("Most results (default 10)"),
        video_ids: z.array(z.string().min(1)).optional().describe("Search only these videos"),
        index,
      },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoSearch(input, contextFor(name, extra));
      return reply(result, core.searchText(result));
    },
  );

  server.registerTool(
    "video_ask",
    {
      title: "Ask about a video",
      description:
        "Ask Pegasus a question about one video, or about start..end of it. Returns text, or JSON matching json_schema when given.",
      inputSchema: {
        video_id: videoId,
        question: z.string().min(1),
        start: seconds.optional().describe("Window start in seconds"),
        end: seconds.optional().describe("Window end in seconds"),
        json_schema: z.record(z.string(), z.unknown()).optional().describe("JSON Schema the answer must follow"),
        index,
      },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoAsk(input, contextFor(name, extra));
      return reply(result, core.askText(result));
    },
  );

  server.registerTool(
    "video_find",
    {
      title: "Find verified clips",
      description:
        "Verified clip finder: searches, then has Pegasus watch every candidate and confirm the claim, the person " +
        "and the step order. Slower than video_search and spends Pegasus minutes. Returns confirmed clips with " +
        "evidence and cut files, plus rejected candidates with reasons.",
      inputSchema: {
        query,
        then,
        max_candidates: z.number().int().min(1).max(50).optional().describe("Candidates to check (default 8)"),
        pad: seconds.optional().describe("Seconds added around each candidate before checking (default 3)"),
        min_confidence: z.number().min(0).max(1).optional().describe("Lowest confidence to confirm (default 0.6)"),
        cut: z.boolean().optional().describe("Cut each confirmed clip and its middle frame (default true)"),
        index,
      },
      annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoFind(input, contextFor(name, extra));
      return reply(result, core.findText(result));
    },
  );

  server.registerTool(
    "video_cut",
    {
      title: "Cut a clip",
      description:
        "Cut start..end of a video into an mp4 from its local copy, with its middle frame or a contact sheet returned as an image.",
      inputSchema: {
        video_id: videoId,
        start: seconds.describe("Seconds"),
        end: seconds.describe("Seconds"),
        frame: z.enum(["mid", "sheet", "none"]).optional().describe("Middle frame (default), 6-frame sheet, or none"),
        index,
      },
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoCut(input, contextFor(name, extra));
      return reply(result, core.cutText(result), result.frame_path);
    },
  );

  server.registerTool(
    "video_frames",
    {
      title: "Look at a range",
      description: "Contact sheet of evenly spaced frames from start..end of a video, returned as an image, without cutting a clip.",
      inputSchema: {
        video_id: videoId,
        start: seconds.describe("Seconds"),
        end: seconds.describe("Seconds"),
        count: z.number().int().min(1).max(12).optional().describe("Frames (default 6)"),
        index,
      },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoFrames(input, contextFor(name, extra));
      return reply(result, core.framesText(result), result.sheet_path);
    },
  );

  server.registerTool(
    "video_entity_add",
    {
      title: "Teach a person",
      description:
        "Create or extend a person (entity) from reference images (paths or URLs) or from frames of an indexed video " +
        "at given times. Afterwards @Name in video_search and video_find queries means this person.",
      inputSchema: {
        name: z.string().min(1).describe('Name as written after @, e.g. "LeBron James"'),
        images: z.array(z.string().min(1)).optional().describe("Image paths or URLs showing the person clearly"),
        from: z
          .object({ video_id: videoId, times: z.array(seconds).min(1).describe("Seconds where the person is clearly visible") })
          .optional(),
        index,
      },
      annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoEntityAdd(input, contextFor(name, extra));
      return reply(result, core.entityAddText(result));
    },
  );

  server.registerTool(
    "video_entities",
    {
      title: "List people",
      description: "List the people (entities) that @Name in queries can refer to. They are shared by all indexes.",
      inputSchema: {},
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async () => {
      const result = await core.videoEntities();
      return reply(result, core.entitiesText(result));
    },
  );

  server.registerTool(
    "video_jockey",
    {
      title: "Ask Jockey",
      description:
        "Ask Jockey, TwelveLabs' research-preview agent, a question about the whole collection. Slower; answers in " +
        "prose with citations to videos and times. Use it for corpus-level questions; video_find is the verified " +
        "clip finder. Pass session_id to follow up.",
      inputSchema: {
        question: z.string().min(1),
        session_id: z.string().min(1).optional().describe("session_id of an earlier answer, to follow up"),
        video_ids: z.array(z.string().min(1)).optional().describe("Answer from these videos only"),
        index,
      },
      annotations: { readOnlyHint: true, openWorldHint: true },
    },
    async ({ index: name, ...input }, extra) => {
      const result = await core.videoJockey(input, contextFor(name, extra));
      return reply(result, core.jockeyText(result));
    },
  );

  await server.connect(new StdioServerTransport());
}
