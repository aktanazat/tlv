#!/usr/bin/env bun
// tlv: TwelveLabs video search for terminal agents. Each command runs one core operation;
// `tlv mcp` serves the same operations as MCP tools over stdio.

import { parseArgs } from "node:util";
import { z } from "zod";
import * as core from "./core.ts";
import { serveMcp } from "./mcp.ts";
import { parseTime } from "./search.ts";

const HELP = `tlv: search, verify and cut moments in videos with TwelveLabs

usage: tlv <command> [arguments] [options]

commands:
  add <source>                 add a local file or a URL (YouTube, direct .mp4) to the index
      --title T                title (default: the video's own title or file name)
      --no-wait                return once indexing has started
  list                         videos in the index
  status <video_id>            indexing and Jockey store status of one video
  search <query>               fast semantic search: likely moments, not verified
      --then STEP              next step of an ordered action (repeatable)
      --modality M             visual, audio or transcription (repeatable; default visual, audio)
      --max-gap S              most seconds between steps (default 12)
      --limit N                most results (default 10)
      --video ID               search only this video (repeatable)
  ask <video_id> <question>    ask Pegasus about a video
      --from T --to T          only this window
      --schema JSON            JSON Schema the answer must follow
  find <query>                 verified clips: search, then Pegasus checks each candidate
      --then STEP              next step (repeatable)
      --max-candidates N       candidates to check (default 8)
      --pad S                  seconds added around each candidate (default 3)
      --min-confidence C       0 to 1 (default 0.6)
      --no-cut                 do not cut the confirmed clips
  cut <video_id> --from T --to T [--frame mid|sheet|none]
                               cut a clip from the local copy
  frames <video_id> --from T --to T [--count N]
                               contact sheet of a range
  entity-add <name> [--image PATH|URL]... [--video ID --at T...]
                               teach a person for @Name in queries
  entities                     people that @Name can refer to
  jockey <question> [--session ID] [--video ID]...
                               ask Jockey (research preview) about the whole collection
  mcp                          serve these commands as MCP tools over stdio

options for every command:
  --index NAME                 index (default: $TLV_INDEX, else tlv)
  --json                       print the structured result as JSON
  -h, --help                   help

Times are seconds (83.4) or timecodes (1:23.4, 00:01:23.4).
The API key comes from TWELVELABS_API_KEY: mem-secret run TWELVELABS_API_KEY -- tlv <command>
`;

const OPTIONS = {
  index: { type: "string" },
  json: { type: "boolean" },
  help: { type: "boolean", short: "h" },
  title: { type: "string" },
  "no-wait": { type: "boolean" },
  then: { type: "string", multiple: true },
  modality: { type: "string", multiple: true },
  "max-gap": { type: "string" },
  limit: { type: "string" },
  video: { type: "string", multiple: true },
  from: { type: "string" },
  to: { type: "string" },
  schema: { type: "string" },
  "max-candidates": { type: "string" },
  pad: { type: "string" },
  "min-confidence": { type: "string" },
  "no-cut": { type: "boolean" },
  frame: { type: "string" },
  count: { type: "string" },
  image: { type: "string", multiple: true },
  at: { type: "string", multiple: true },
  session: { type: "string" },
} as const;
const COMMON_OPTIONS = ["index", "json", "help"];
const MODALITIES = ["visual", "audio", "transcription"] as const;
const FRAMES = ["mid", "sheet", "none"] as const;

/** The parsed command-line options; keys follow OPTIONS. */
interface Values {
  index?: string;
  json?: boolean;
  help?: boolean;
  title?: string;
  "no-wait"?: boolean;
  then?: string[];
  modality?: string[];
  "max-gap"?: string;
  limit?: string;
  video?: string[];
  from?: string;
  to?: string;
  schema?: string;
  "max-candidates"?: string;
  pad?: string;
  "min-confidence"?: string;
  "no-cut"?: boolean;
  frame?: string;
  count?: string;
  image?: string[];
  at?: string[];
  session?: string;
}

/** A mistake in the command line: exits with status 2 and the command's usage. */
class UsageError extends Error {}

interface Command {
  usage: string;
  /** Options this command takes besides --index, --json and --help. */
  options: string[];
  run(args: string[], values: Values, ctx: core.Context): Promise<{ result: unknown; text: string }>;
}

function one(args: string[], what: string): string {
  const [value, ...extra] = args;
  if (value === undefined || extra.length > 0) throw new UsageError(`expected ${what}`);
  return value;
}

function words(args: string[], what: string): string {
  const text = args.join(" ").trim();
  if (!text) throw new UsageError(`expected ${what}`);
  return text;
}

function none(args: string[]): void {
  if (args.length > 0) throw new UsageError(`unexpected argument "${args[0]}"`);
}

function number(value: string | undefined, flag: string): number | undefined {
  if (value === undefined) return undefined;
  const parsed = Number(value);
  if (value.trim() === "" || !Number.isFinite(parsed)) throw new UsageError(`--${flag} needs a number, got "${value}"`);
  return parsed;
}

function seconds(value: string | undefined, flag: string): number | undefined {
  if (value === undefined) return undefined;
  try {
    return parseTime(value);
  } catch (error) {
    throw new UsageError(`--${flag}: ${error instanceof Error ? error.message : String(error)}`);
  }
}

function given<T>(value: T | undefined, flag: string): T {
  if (value === undefined) throw new UsageError(`--${flag} is required`);
  return value;
}

function oneOf<T extends string>(value: string, allowed: readonly T[], flag: string): T {
  const match = allowed.find((option) => option === value);
  if (!match) throw new UsageError(`--${flag} must be one of ${allowed.join(", ")}, got "${value}"`);
  return match;
}

const COMMANDS: Record<string, Command> = {
  add: {
    usage: "tlv add <source> [--title T] [--no-wait]",
    options: ["title", "no-wait"],
    async run(args, v, ctx) {
      const result = await core.videoAdd({ source: one(args, "a file path or URL"), title: v.title, wait: !v["no-wait"] }, ctx);
      return { result, text: core.addText(result) };
    },
  },
  list: {
    usage: "tlv list",
    options: [],
    async run(args, _v, ctx) {
      none(args);
      const result = await core.videoList(ctx);
      return { result, text: core.listText(result) };
    },
  },
  status: {
    usage: "tlv status <video_id>",
    options: [],
    async run(args, _v, ctx) {
      const result = await core.videoStatus({ video_id: one(args, "a video_id") }, ctx);
      return { result, text: core.statusText(result) };
    },
  },
  search: {
    usage: "tlv search <query> [--then STEP]... [--modality M]... [--max-gap S] [--limit N] [--video ID]...",
    options: ["then", "modality", "max-gap", "limit", "video"],
    async run(args, v, ctx) {
      const input = {
        query: words(args, "a query"),
        then: v.then,
        modalities: v.modality?.map((m) => oneOf(m, MODALITIES, "modality")),
        max_gap: number(v["max-gap"], "max-gap"),
        limit: number(v.limit, "limit"),
        video_ids: v.video,
      };
      const result = await core.videoSearch(input, ctx);
      return { result, text: core.searchText(result) };
    },
  },
  ask: {
    usage: "tlv ask <video_id> <question> [--from T] [--to T] [--schema JSON]",
    options: ["from", "to", "schema"],
    async run([videoId, ...question], v, ctx) {
      if (!videoId) throw new UsageError("expected a video_id");
      let schema: Record<string, unknown> | undefined;
      if (v.schema !== undefined) {
        let json: unknown;
        try {
          json = JSON.parse(v.schema);
        } catch {
          throw new UsageError("--schema must be inline JSON");
        }
        const object = z.record(z.string(), z.unknown()).safeParse(json);
        if (!object.success) throw new UsageError("--schema must be a JSON object");
        schema = object.data;
      }
      const input = {
        video_id: videoId,
        question: words(question, "a question"),
        start: seconds(v.from, "from"),
        end: seconds(v.to, "to"),
        json_schema: schema,
      };
      const result = await core.videoAsk(input, ctx);
      return { result, text: core.askText(result) };
    },
  },
  find: {
    usage: "tlv find <query> [--then STEP]... [--max-candidates N] [--pad S] [--min-confidence C] [--no-cut]",
    options: ["then", "max-candidates", "pad", "min-confidence", "no-cut"],
    async run(args, v, ctx) {
      const input = {
        query: words(args, "a query"),
        then: v.then,
        max_candidates: number(v["max-candidates"], "max-candidates"),
        pad: number(v.pad, "pad"),
        min_confidence: number(v["min-confidence"], "min-confidence"),
        cut: !v["no-cut"],
      };
      const result = await core.videoFind(input, ctx);
      return { result, text: core.findText(result) };
    },
  },
  cut: {
    usage: "tlv cut <video_id> --from T --to T [--frame mid|sheet|none]",
    options: ["from", "to", "frame"],
    async run(args, v, ctx) {
      const input = {
        video_id: one(args, "a video_id"),
        start: given(seconds(v.from, "from"), "from"),
        end: given(seconds(v.to, "to"), "to"),
        frame: v.frame === undefined ? undefined : oneOf(v.frame, FRAMES, "frame"),
      };
      const result = await core.videoCut(input, ctx);
      return { result, text: core.cutText(result) };
    },
  },
  frames: {
    usage: "tlv frames <video_id> --from T --to T [--count N]",
    options: ["from", "to", "count"],
    async run(args, v, ctx) {
      const input = {
        video_id: one(args, "a video_id"),
        start: given(seconds(v.from, "from"), "from"),
        end: given(seconds(v.to, "to"), "to"),
        count: number(v.count, "count"),
      };
      const result = await core.videoFrames(input, ctx);
      return { result, text: core.framesText(result) };
    },
  },
  "entity-add": {
    usage: "tlv entity-add <name> [--image PATH|URL]... [--video ID --at T...]",
    options: ["image", "video", "at"],
    async run(args, v, ctx) {
      const [videoId, ...otherVideos] = v.video ?? [];
      if (otherVideos.length > 0) throw new UsageError("entity-add takes one --video");
      const times = (v.at ?? []).map((t) => given(seconds(t, "at"), "at"));
      if ((videoId === undefined) !== (times.length === 0)) throw new UsageError("--video and --at go together");
      const input = {
        name: words(args, "a name"),
        images: v.image,
        from: videoId === undefined ? undefined : { video_id: videoId, times },
      };
      const result = await core.videoEntityAdd(input, ctx);
      return { result, text: core.entityAddText(result) };
    },
  },
  entities: {
    usage: "tlv entities",
    options: [],
    async run(args) {
      none(args);
      const result = await core.videoEntities();
      return { result, text: core.entitiesText(result) };
    },
  },
  jockey: {
    usage: "tlv jockey <question> [--session ID] [--video ID]...",
    options: ["session", "video"],
    async run(args, v, ctx) {
      const input = { question: words(args, "a question"), session_id: v.session, video_ids: v.video };
      const result = await core.videoJockey(input, ctx);
      return { result, text: core.jockeyText(result) };
    },
  },
};

async function main(argv: string[]): Promise<number> {
  let values: Values;
  let positionals: string[];
  try {
    ({ values, positionals } = parseArgs({ args: argv, options: OPTIONS, allowPositionals: true }));
  } catch (error) {
    console.error(`tlv: ${error instanceof Error ? error.message : String(error)}\nRun tlv --help for usage.`);
    return 2;
  }
  const [name, ...args] = positionals;
  if (name === undefined || name === "help") {
    const asked = Boolean(values.help) || name === "help";
    (asked ? process.stdout : process.stderr).write(HELP);
    return asked ? 0 : 2;
  }
  if (name === "mcp") {
    if (values.help) {
      console.log("usage: tlv mcp (serves the tlv tools over stdio)");
      return 0;
    }
    if (args.length > 0 || Object.keys(values).length > 0) {
      console.error("tlv: mcp takes no arguments\nusage: tlv mcp");
      return 2;
    }
    await serveMcp();
    return 0;
  }
  const command = COMMANDS[name];
  if (!command) {
    console.error(`tlv: unknown command "${name}"\nRun tlv --help for usage.`);
    return 2;
  }
  if (values.help) {
    console.log(`usage: ${command.usage}`);
    return 0;
  }
  try {
    const unexpected = Object.keys(values).filter((key) => !COMMON_OPTIONS.includes(key) && !command.options.includes(key));
    if (unexpected.length > 0) throw new UsageError(`${name} does not take ${unexpected.map((key) => `--${key}`).join(", ")}`);
    const ctx: core.Context = { index: values.index, progress: (message) => console.error(`tlv: ${message}`) };
    const { result, text } = await command.run(args, values, ctx);
    console.log(values.json ? JSON.stringify(result, null, 2) : text);
    return 0;
  } catch (error) {
    if (error instanceof UsageError) {
      console.error(`tlv: ${error.message}\nusage: ${command.usage}`);
      return 2;
    }
    console.error(`tlv: ${error instanceof Error ? error.message : String(error)}`);
    return 1;
  }
}

process.exitCode = await main(process.argv.slice(2));
