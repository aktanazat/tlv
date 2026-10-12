// The tlv operations. The CLI and the MCP server are thin layers over this module: each
// operation returns a compact JSON result, and its *Text function renders that result as
// a few short lines.

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { basename, extname, join, resolve } from "node:path";
import { z } from "zod";
import * as api from "./api.ts";
import { contactSheet, cut, dataDir, download, probe, slug, still } from "./media.ts";
import {
  type Candidate,
  type Range,
  formatRange,
  formatTimecode,
  inspectionWindow,
  judge,
  observationId,
  rankClips,
  rankSteps,
  resolveMentions,
} from "./search.ts";

export type Modality = "visual" | "audio" | "transcription";
export type Frame = "mid" | "sheet" | "none";

/** Receives short progress messages from long operations. */
export type Progress = (message: string) => void;

export interface Context {
  /** Index name; defaults to $TLV_INDEX, then "tlv". */
  index?: string;
  progress?: Progress;
}

const ENTITY_COLLECTION = "tlv-people";
const MIN_VIDEO_SECONDS = 4;
const ASK_MIN_WINDOW_SECONDS = 4;
const FIND_MIN_WINDOW_SECONDS = 6;
const FIND_CONCURRENCY = 4;
const LOOKUP_CONCURRENCY = 8;
const SHEET_FRAMES = 6;
const STILL_WIDTH = 960;
const PROGRESS_EVERY_MS = 30_000;

const indexName = (ctx: Context) => ctx.index || process.env.TLV_INDEX || "tlv";
const round = (seconds: number) => Math.round(seconds * 100) / 100;
const isUrl = (text: string) => /^https?:\/\//i.test(text);

function metaText(meta: Record<string, unknown> | null | undefined, key: string): string {
  const value = meta?.[key];
  return typeof value === "string" ? value : "";
}

/** Maps with at most `limit` calls in flight, keeping input order. */
async function mapLimit<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  const queue = [...items.entries()];
  const worker = async () => {
    for (let entry = queue.shift(); entry; entry = queue.shift()) {
      const [i, item] = entry;
      results[i] = await fn(item);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return results;
}

/**
 * Polls `read` until `settled` holds or `limitMs` passes, and returns the last value.
 * Reports progress when the description changes, and every 30 s otherwise.
 */
async function pollUntil<T>(
  read: () => Promise<T>,
  settled: (value: T) => boolean,
  options: { everyMs: number; limitMs: number; describe: (value: T) => string; progress?: Progress },
): Promise<T> {
  const startedAt = Date.now();
  let lastMessage = "";
  let lastReportAt = 0;
  for (;;) {
    const value = await read();
    const elapsed = Date.now() - startedAt;
    if (settled(value) || elapsed >= options.limitMs) return value;
    const message = options.describe(value);
    if (message !== lastMessage || Date.now() - lastReportAt >= PROGRESS_EVERY_MS) {
      options.progress?.(`${message} (${Math.round(elapsed / 1000)} s)`);
      lastMessage = message;
      lastReportAt = Date.now();
    }
    await Bun.sleep(options.everyMs);
  }
}

async function requireIndex(name: string): Promise<string> {
  const id = await api.findIndex(name);
  if (!id) throw new Error(`Index "${name}" does not exist yet. Add a video first (video_add, or tlv add <source>).`);
  return id;
}

/** A video of the index, with the metadata tlv stores on it. */
interface Video {
  video_id: string;
  asset_id: string;
  title: string;
  source: string;
  local_path: string;
  duration: number;
  status: string;
}

function toVideo(asset: api.IndexedAsset): Video {
  return {
    video_id: asset._id,
    asset_id: asset.asset_id,
    title: metaText(asset.user_metadata, "title") || asset.system_metadata?.filename || asset._id,
    source: metaText(asset.user_metadata, "source"),
    local_path: metaText(asset.user_metadata, "local_path"),
    duration: asset.system_metadata?.duration ?? 0,
    status: asset.status,
  };
}

/**
 * Retrieves videos by id. An id is the indexed asset id that search results and
 * video_list carry; the asset id of the video is accepted too.
 */
async function lookupVideos(indexId: string, ids: string[]): Promise<Map<string, Video>> {
  const listed = await api.listIndexedAssets(indexId);
  const found = new Map<string, Video>();
  await mapLimit([...new Set(ids)], LOOKUP_CONCURRENCY, async (videoId) => {
    const match = listed.find((asset) => asset._id === videoId || asset.asset_id === videoId);
    if (match) found.set(videoId, toVideo(await api.getIndexedAsset(indexId, match._id)));
  });
  return found;
}

async function getVideo(indexId: string, videoId: string): Promise<Video> {
  const video = (await lookupVideos(indexId, [videoId])).get(videoId);
  if (!video) throw new Error(`No video "${videoId}" in this index. video_list shows the ids.`);
  return video;
}

function requireLocalFile(video: Video): void {
  if (video.local_path && existsSync(video.local_path)) return;
  const where = video.local_path ? `(${video.local_path}) is missing` : "was never stored";
  const fix = isUrl(video.source)
    ? `Re-add the source to download it again: video_add source="${video.source}" (tlv add "${video.source}").`
    : `Put the file back at ${video.source || video.local_path}, then re-add it with video_add.`;
  throw new Error(`The local file of "${video.title}" ${where}. ${fix}`);
}

/** Clamps a requested range to the video and checks it is not empty. */
function clampRange(range: Range, video: Video): Range {
  const start = Math.max(0, range.start);
  const end = video.duration > 0 ? Math.min(range.end, video.duration) : range.end;
  if (!(end > start)) {
    throw new Error(`Empty range ${formatRange(range.start, range.end)} for "${video.title}" (${formatTimecode(video.duration)} long)`);
  }
  return { start, end };
}

/** File name stem for media cut from a range of a video. */
function mediaStem(video: Video, range: Range): string {
  return `${slug(video.title)}-${video.video_id.slice(-6)}_${range.start.toFixed(1)}-${range.end.toFixed(1)}`;
}

type CutFiles = {
  clip_path: string;
  frame_path?: string;
  frame_times?: number[];
};

async function cutRange(video: Video, range: Range, frame: Frame): Promise<CutFiles> {
  requireLocalFile(video);
  const stem = mediaStem(video, range);
  const clipPath = join(await dataDir("clips"), `${stem}.mp4`);
  await cut(video.local_path, range, clipPath);
  if (frame === "none") return { clip_path: clipPath };
  const frames = await dataDir("frames");
  if (frame === "sheet") {
    const sheetPath = join(frames, `${stem}_sheet.jpg`);
    const times = await contactSheet(video.local_path, range, SHEET_FRAMES, sheetPath);
    return { clip_path: clipPath, frame_path: sheetPath, frame_times: times.map(round) };
  }
  const middle = (range.start + range.end) / 2;
  const stillPath = join(frames, `${stem}_mid.jpg`);
  await still(video.local_path, middle, stillPath, STILL_WIDTH);
  return { clip_path: clipPath, frame_path: stillPath, frame_times: [round(middle)] };
}

async function waitAsset(assetId: string, ctx: Context): Promise<void> {
  const asset = await pollUntil(
    () => api.getAsset(assetId),
    (a) => a.status === "ready" || a.status === "failed",
    { everyMs: 2000, limitMs: 30 * 60_000, describe: (a) => `asset ${a.status}`, progress: ctx.progress },
  );
  if (asset.status === "failed") {
    throw new Error(`TwelveLabs could not process asset ${assetId}: ${JSON.stringify(asset.error ?? "no detail")}`);
  }
  if (asset.status !== "ready") throw new Error(`Asset ${assetId} is still ${asset.status} after 30 minutes`);
}

/** Waits up to two hours for indexing; returns the last status. A failure throws. */
async function waitIndexed(indexId: string, videoId: string, ctx: Context): Promise<string> {
  const asset = await pollUntil(
    () => api.getIndexedAsset(indexId, videoId),
    (a) => a.status === "ready" || a.status === "failed",
    { everyMs: 5000, limitMs: 2 * 3600_000, describe: (a) => `indexing: ${a.status}`, progress: ctx.progress },
  );
  if (asset.status === "failed") throw new Error(`Indexing failed for video ${videoId}. Run video_add again to retry.`);
  return asset.status;
}

// video_add

/** Where the video stands in the Jockey knowledge store that mirrors the index. */
export type StoreState = { item_id: string; status: string } | { error: string };

export type AddResult = {
  video_id: string;
  asset_id: string;
  title: string;
  duration: number;
  status: string;
  local_path: string;
  already_indexed: boolean;
  store: StoreState;
};

/** The store item of an asset, preferring one that has not failed. */
function storeItemFor(items: api.StoreItem[], assetId: string): api.StoreItem | undefined {
  const matches = items.filter((item) => item.asset_id === assetId);
  return matches.find((item) => item.status !== "failed") ?? matches[0];
}

/**
 * Adds the video to the knowledge store named after the index, unless a working item for
 * its asset is there already. The index is the primary record, so a store failure is
 * reported in the result; running video_add again retries it.
 */
async function mirrorToStore(storeName: string, video: Video, checkExisting: boolean): Promise<StoreState> {
  try {
    const storeId = (await api.findKnowledgeStore(storeName)) ?? (await api.createKnowledgeStore(storeName));
    const existing = checkExisting ? storeItemFor(await api.listStoreItems(storeId), video.asset_id) : undefined;
    if (existing && existing.status !== "failed") return { item_id: existing._id, status: existing.status ?? "unknown" };
    const metadata = { title: video.title, source: video.source, video_id: video.video_id };
    const item = await api.createStoreItem(storeId, video.asset_id, metadata);
    return { item_id: item._id, status: item.status ?? "queued" };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

/** A local path (absolute, with ~ expanded) or an http(s) URL. */
function normalizeSource(source: string): string {
  const trimmed = source.trim();
  if (isUrl(trimmed)) return trimmed;
  const path = resolve(trimmed.replace(/^~(?=$|\/)/, homedir()));
  if (!existsSync(path)) throw new Error(`"${source}" is neither an existing file nor an http(s) URL`);
  return path;
}

/** Every video of the index with its metadata: one list call, then one retrieval per video. */
async function allVideos(indexId: string): Promise<Video[]> {
  const listed = await api.listIndexedAssets(indexId);
  return mapLimit(listed, LOOKUP_CONCURRENCY, async (asset) => toVideo(await api.getIndexedAsset(indexId, asset._id)));
}

export async function videoAdd(input: { source: string; title?: string; wait?: boolean }, ctx: Context = {}): Promise<AddResult> {
  const name = indexName(ctx);
  const source = normalizeSource(input.source);
  const wait = input.wait ?? true;
  const indexId = (await api.findIndex(name)) ?? (await api.createIndex(name));

  // The index is the registry: a video that has not failed with the same source is this one.
  const existing = (await allVideos(indexId)).find((video) => video.status !== "failed" && video.source === source);
  if (existing) {
    ctx.progress?.(`already indexed as ${existing.video_id}`);
    let localPath = existing.local_path;
    if (isUrl(source) && !(localPath && existsSync(localPath))) {
      ctx.progress?.(`local copy missing; downloading ${source}`);
      localPath = (await download(source)).path;
      await api.updateIndexedAssetMetadata(indexId, existing.video_id, { local_path: localPath });
    }
    const store = await mirrorToStore(name, existing, true);
    const status = wait && existing.status !== "ready" ? await waitIndexed(indexId, existing.video_id, ctx) : existing.status;
    const { video_id, asset_id, title, duration } = existing;
    return { video_id, asset_id, title, duration, status, local_path: localPath, already_indexed: true, store };
  }

  if (isUrl(source)) ctx.progress?.(`downloading ${source}`);
  const local = isUrl(source) ? await download(source) : { path: source, title: basename(source, extname(source)) };
  const { duration } = await probe(local.path);
  if (duration < MIN_VIDEO_SECONDS) {
    throw new Error(`${local.path} is ${duration.toFixed(1)} s long; TwelveLabs needs videos of at least ${MIN_VIDEO_SECONDS} s`);
  }
  const title = input.title?.trim() || local.title;
  ctx.progress?.(`uploading ${basename(local.path)}`);
  const assetId = await api.uploadFile(local.path, "video", (done, total) => ctx.progress?.(`uploaded part ${done} of ${total}`));
  await waitAsset(assetId, ctx);
  ctx.progress?.("uploaded; indexing");
  const videoId = await api.indexAsset(indexId, assetId, { title, source, local_path: local.path });
  const video: Video = { video_id: videoId, asset_id: assetId, title, source, local_path: local.path, duration, status: "" };
  const store = await mirrorToStore(name, video, false);
  const status = wait ? await waitIndexed(indexId, videoId, ctx) : (await api.getIndexedAsset(indexId, videoId)).status;
  return { video_id: videoId, asset_id: assetId, title, duration, status, local_path: local.path, already_indexed: false, store };
}

export function addText(r: AddResult): string {
  const store = "error" in r.store ? `not added (${r.store.error})` : `${r.store.status} (${r.store.item_id})`;
  return [
    `${r.already_indexed ? "already indexed" : "added"}: "${r.title}" (${formatTimecode(r.duration)}) video_id=${r.video_id} status=${r.status}`,
    `local: ${r.local_path}`,
    `jockey store: ${store}`,
  ].join("\n");
}

// video_list and video_status

export type ListResult = {
  index: string;
  videos: { video_id: string; title: string; duration: number; status: string; local_path: string; local_exists: boolean }[];
};

export async function videoList(ctx: Context = {}): Promise<ListResult> {
  const name = indexName(ctx);
  const indexId = await api.findIndex(name);
  if (!indexId) return { index: name, videos: [] };
  const videos = await allVideos(indexId);
  return {
    index: name,
    videos: videos.map((v) => ({
      video_id: v.video_id,
      title: v.title,
      duration: round(v.duration),
      status: v.status,
      local_path: v.local_path,
      local_exists: v.local_path !== "" && existsSync(v.local_path),
    })),
  };
}

export function listText(r: ListResult): string {
  if (r.videos.length === 0) return `no videos in index "${r.index}"`;
  return r.videos
    .map((v) => `${v.video_id}  ${v.status}  ${formatTimecode(v.duration)}  "${v.title}"${v.local_exists ? "" : "  (local file missing)"}`)
    .join("\n");
}

export type StatusResult = {
  video_id: string;
  asset_id: string;
  title: string;
  status: string;
  duration: number;
  local_path: string;
  local_exists: boolean;
  /** The video's item in the Jockey knowledge store, or null when it is not there. */
  store: { item_id: string; status: string } | null;
};

export async function videoStatus(input: { video_id: string }, ctx: Context = {}): Promise<StatusResult> {
  const name = indexName(ctx);
  const video = await getVideo(await requireIndex(name), input.video_id);
  const storeId = await api.findKnowledgeStore(name);
  const item = storeId ? storeItemFor(await api.listStoreItems(storeId), video.asset_id) : undefined;
  return {
    video_id: video.video_id,
    asset_id: video.asset_id,
    title: video.title,
    status: video.status,
    duration: round(video.duration),
    local_path: video.local_path,
    local_exists: video.local_path !== "" && existsSync(video.local_path),
    store: item ? { item_id: item._id, status: item.status ?? "unknown" } : null,
  };
}

export function statusText(r: StatusResult): string {
  return [
    `"${r.title}" video_id=${r.video_id} index: ${r.status} (${formatTimecode(r.duration)})`,
    `jockey store: ${r.store ? `${r.store.status} (${r.store.item_id})` : "not added"}`,
    `local: ${r.local_path || "none"}${r.local_exists ? "" : " (missing)"}`,
  ].join("\n");
}

// video_search

interface StepTexts {
  query: string;
  then: string[];
}

/**
 * Resolves @Name mentions. `search` carries <@entity_id> for Marengo; `plain` carries
 * the names for Pegasus prompts.
 */
async function resolveEntities(texts: StepTexts): Promise<{ search: StepTexts; plain: StepTexts }> {
  if (![texts.query, ...texts.then].some((t) => t.includes("@"))) return { search: texts, plain: texts };
  const collectionId = await api.findEntityCollection(ENTITY_COLLECTION);
  const entities = collectionId ? (await api.listEntities(collectionId)).map((e) => ({ id: e._id, name: e.name })) : [];
  const query = resolveMentions(texts.query, entities);
  const then = texts.then.map((t) => resolveMentions(t, entities));
  const unknown = [...new Set([query, ...then].flatMap((m) => m.unknown))];
  if (unknown.length > 0) {
    const known = entities.length
      ? `Known: ${entities.map((e) => `@${e.name}`).join(", ")}`
      : "No entities yet; add one with video_entity_add.";
    throw new Error(`Unknown entity ${unknown.join(", ")}. ${known}`);
  }
  return {
    search: { query: query.search, then: then.map((m) => m.search) },
    plain: { query: query.plain, then: then.map((m) => m.plain) },
  };
}

interface SearchOptions {
  modalities?: Modality[];
  max_gap?: number;
  video_ids?: string[];
}

async function searchCandidates(
  indexId: string,
  texts: StepTexts,
  options: SearchOptions,
  limit: number,
): Promise<Candidate<api.SearchClip>[]> {
  const params = { indexId, modalities: options.modalities ?? ["visual", "audio"], videoIds: options.video_ids };
  if (texts.then.length === 0) return rankClips(await api.search({ ...params, query: texts.query }), limit);
  const steps = [texts.query, ...texts.then];
  const queries = [steps.join(", then "), ...steps];
  const [sentence, ...perStep] = await Promise.all(queries.map((query) => api.search({ ...params, query })));
  return rankSteps(sentence ?? [], perStep, options.max_gap ?? 12, limit);
}

export type SearchHit = {
  video_id: string;
  title: string;
  start: number;
  end: number;
  timecode: string;
  /** clip: a single-query hit; sentence: a hit of the whole multi-step sentence; chain: one clip per step. */
  kind: "clip" | "sentence" | "chain";
  /** Lower is better: the search rank, or the sum of step ranks for a chain. */
  score: number;
  steps?: { step: string; start: number; end: number; timecode: string; rank: number }[];
  thumbnail_url?: string;
};

export type SearchResult = { query: string; then: string[]; results: SearchHit[] };

export async function videoSearch(
  input: { query: string; then?: string[]; limit?: number } & SearchOptions,
  ctx: Context = {},
): Promise<SearchResult> {
  const indexId = await requireIndex(indexName(ctx));
  const typed = { query: input.query, then: input.then ?? [] };
  const { search } = await resolveEntities(typed);
  const candidates = await searchCandidates(indexId, search, input, input.limit ?? 10);
  const labels = [typed.query, ...typed.then];
  const results = candidates.map((c): SearchHit => {
    const first = c.clips[0];
    return {
      video_id: c.video_id,
      title: metaText(first?.user_metadata, "title"),
      start: round(c.start),
      end: round(c.end),
      timecode: formatRange(c.start, c.end),
      kind: c.kind,
      score: c.score,
      ...(c.kind === "chain" && {
        steps: c.clips.map((clip, i) => ({
          step: labels[i] ?? "",
          start: round(clip.start),
          end: round(clip.end),
          timecode: formatRange(clip.start, clip.end),
          rank: clip.rank,
        })),
      }),
      ...(first?.thumbnail_url && { thumbnail_url: first.thumbnail_url }),
    };
  });
  return { query: typed.query, then: typed.then, results };
}

export function searchText(r: SearchResult): string {
  const asked = [r.query, ...r.then].join(", then ");
  if (r.results.length === 0) return `no results for "${asked}"`;
  const lines = [`${r.results.length} results for "${asked}":`];
  r.results.forEach((hit, i) => {
    lines.push(`${i + 1}. ${hit.timecode}  "${hit.title || hit.video_id}"  video_id=${hit.video_id}  ${hit.kind} score ${hit.score}`);
    for (const step of hit.steps ?? []) lines.push(`   ${step.timecode}  ${step.step} (rank ${step.rank})`);
  });
  return lines.join("\n");
}

// video_ask

export type AskResult = {
  video_id: string;
  title: string;
  start?: number;
  end?: number;
  timecode?: string;
  /** Pegasus's answer, when no json_schema was given. */
  answer?: string;
  /** The parsed answer, when a json_schema was given. */
  json?: unknown;
  /** True when Pegasus stopped at its token limit. */
  truncated: boolean;
};

export async function videoAsk(
  input: { video_id: string; question: string; start?: number; end?: number; json_schema?: Record<string, unknown> },
  ctx: Context = {},
): Promise<AskResult> {
  const video = await getVideo(await requireIndex(indexName(ctx)), input.video_id);
  let window: Range | undefined;
  if (input.start !== undefined || input.end !== undefined) {
    const range = clampRange({ start: input.start ?? 0, end: input.end ?? video.duration }, video);
    window = inspectionWindow(range, 0, video.duration, ASK_MIN_WINDOW_SECONDS);
  }
  ctx.progress?.("asking Pegasus");
  const analysis = await api.analyze({
    assetId: video.asset_id,
    prompt: input.question,
    start: window?.start,
    end: window?.end,
    jsonSchema: input.json_schema,
  });
  const truncated = analysis.finish_reason === "length";
  const base = {
    video_id: video.video_id,
    title: video.title,
    ...(window && { start: round(window.start), end: round(window.end), timecode: formatRange(window.start, window.end) }),
    truncated,
  };
  if (!input.json_schema) return { ...base, answer: analysis.data };
  try {
    return { ...base, json: JSON.parse(analysis.data) };
  } catch {
    const cutOff = truncated ? " (cut off at the token limit)" : "";
    throw new Error(`Pegasus returned invalid JSON${cutOff}: ${analysis.data.slice(0, 300)}`);
  }
}

export function askText(r: AskResult): string {
  const where = r.timecode ? `${r.timecode} of "${r.title}"` : `"${r.title}"`;
  const body = r.json !== undefined ? JSON.stringify(r.json, null, 2) : (r.answer ?? "");
  return `${where}${r.truncated ? " (answer cut off at the token limit)" : ""}:\n${body}`;
}

// video_find

/** The JSON Pegasus fills in for each candidate. The zod schema below parses the same shape. */
const VERDICT_JSON_SCHEMA = {
  type: "object",
  properties: {
    match: { type: "boolean" },
    confidence: { type: "number" },
    who: { type: "string" },
    evidence: { type: "string" },
    steps: {
      type: "array",
      items: {
        type: "object",
        properties: {
          step: { type: "string" },
          seen: { type: "boolean" },
          start: { type: "number" },
          end: { type: "number" },
        },
        required: ["step", "seen", "start", "end"],
      },
    },
  },
  required: ["match", "confidence", "who", "evidence", "steps"],
};

const verdictSchema = z.object({
  match: z.boolean(),
  confidence: z.number(),
  who: z.string(),
  evidence: z.string(),
  steps: z.array(z.object({ step: z.string(), seen: z.boolean(), start: z.number(), end: z.number() })),
});

function verdictPrompt(claim: string, steps: string[], seconds: number): string {
  const length = seconds.toFixed(1);
  return [
    `Check whether a claim happens in this video segment. The segment is ${length} seconds long.`,
    `Claim: ${claim}`,
    "Steps, in order:",
    ...steps.map((step, i) => `${i + 1}. ${step}`),
    "",
    "Judge only from what is visible or audible in this segment.",
    "match: true only if the whole claim happens here, with every step in this order. If the person, the action or the order is uncertain, answer false.",
    "confidence: from 0 to 1, how sure you are that the claim happens here.",
    "who: who does it, as identified on screen or by the commentary (name, jersey number, team); empty if unclear.",
    "evidence: what you see or hear that decides it: jersey, number, score bug, on-screen text, commentary.",
    `steps: one entry per step above, in the same order, with the step text, seen (true only if it clearly happens), and its start and end in seconds from the start of this segment (0 to ${length}).`,
  ].join("\n");
}

export type Observation = {
  id: string;
  video_id: string;
  title: string;
  source: string;
  local_path: string;
  start: number;
  end: number;
  timecode: string;
  claim: string;
  who: string;
  evidence: string;
  confidence: number;
  steps: { step: string; start: number; end: number; timecode: string }[];
  clip_path?: string;
  frame_path?: string;
  cut_error?: string;
};

export type Rejection = { video_id: string; start: number; end: number; reason: string };

export type FindResult = {
  claim: string;
  confirmed: Observation[];
  rejected: Rejection[];
  counts: { candidates: number; confirmed: number; rejected: number };
};

async function inspect(
  candidate: Candidate<api.SearchClip>,
  video: Video | undefined,
  claim: string,
  steps: string[],
  pad: number,
  minConfidence: number,
): Promise<{ observation: Observation } | { rejection: Rejection }> {
  const reject = (reason: string) => ({
    rejection: { video_id: candidate.video_id, start: round(candidate.start), end: round(candidate.end), reason },
  });
  if (!video) return reject("video not found in the index");
  const window = inspectionWindow(candidate, pad, video.duration, FIND_MIN_WINDOW_SECONDS);
  let analysis: api.Analysis;
  try {
    analysis = await api.analyze({
      assetId: video.asset_id,
      prompt: verdictPrompt(claim, steps, window.end - window.start),
      start: window.start,
      end: window.end,
      jsonSchema: VERDICT_JSON_SCHEMA,
    });
  } catch (error) {
    return reject(`check failed: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (analysis.finish_reason === "length") return reject("verdict cut off at the token limit");
  let verdict: z.output<typeof verdictSchema>;
  try {
    verdict = verdictSchema.parse(JSON.parse(analysis.data));
  } catch {
    return reject("unreadable verdict");
  }
  const judgement = judge(verdict, window, candidate, steps.length, minConfidence);
  if (!judgement.confirmed) return reject(judgement.reason);
  return {
    observation: {
      id: observationId(video.video_id, judgement.start, judgement.end),
      video_id: video.video_id,
      title: video.title,
      source: video.source,
      local_path: video.local_path,
      start: round(judgement.start),
      end: round(judgement.end),
      timecode: formatRange(judgement.start, judgement.end),
      claim,
      who: verdict.who,
      evidence: verdict.evidence,
      confidence: verdict.confidence,
      steps: judgement.steps.map((s) => ({ step: s.step, start: round(s.start), end: round(s.end), timecode: formatRange(s.start, s.end) })),
    },
  };
}

export async function videoFind(
  input: { query: string; then?: string[]; max_candidates?: number; pad?: number; min_confidence?: number; cut?: boolean },
  ctx: Context = {},
): Promise<FindResult> {
  const indexId = await requireIndex(indexName(ctx));
  const { search, plain } = await resolveEntities({ query: input.query, then: input.then ?? [] });
  const candidates = await searchCandidates(indexId, search, {}, input.max_candidates ?? 8);
  const steps = [plain.query, ...plain.then];
  const claim = steps.join(", then ");
  const videos = await lookupVideos(indexId, candidates.map((c) => c.video_id));
  ctx.progress?.(`checking ${candidates.length} candidates with Pegasus`);
  let checked = 0;
  const outcomes = await mapLimit(candidates, FIND_CONCURRENCY, async (candidate) => {
    const pad = input.pad ?? 3;
    const outcome = await inspect(candidate, videos.get(candidate.video_id), claim, steps, pad, input.min_confidence ?? 0.6);
    checked += 1;
    ctx.progress?.(`checked ${checked} of ${candidates.length}`);
    return outcome;
  });
  const confirmed = outcomes.flatMap((o) => ("observation" in o ? [o.observation] : []));
  const rejected = outcomes.flatMap((o) => ("rejection" in o ? [o.rejection] : []));
  if (input.cut ?? true) {
    for (const observation of confirmed) {
      const video = videos.get(observation.video_id);
      if (!video) continue;
      ctx.progress?.(`cutting ${observation.timecode} of "${observation.title}"`);
      try {
        const files = await cutRange(video, observation, "mid");
        observation.clip_path = files.clip_path;
        if (files.frame_path) observation.frame_path = files.frame_path;
      } catch (error) {
        observation.cut_error = error instanceof Error ? error.message : String(error);
      }
    }
  }
  return {
    claim,
    confirmed,
    rejected,
    counts: { candidates: candidates.length, confirmed: confirmed.length, rejected: rejected.length },
  };
}

export function findText(r: FindResult): string {
  const lines = [`confirmed ${r.counts.confirmed} of ${r.counts.candidates} candidates for "${r.claim}"`];
  for (const o of r.confirmed) {
    lines.push(`${o.id}  ${o.timecode}  "${o.title}"  video_id=${o.video_id}  confidence ${o.confidence}`);
    if (o.who) lines.push(`   who: ${o.who}`);
    lines.push(`   evidence: ${o.evidence}`);
    for (const s of o.steps) lines.push(`   ${s.timecode}  ${s.step}`);
    if (o.clip_path) lines.push(`   clip: ${o.clip_path}`);
    if (o.frame_path) lines.push(`   frame: ${o.frame_path}`);
    if (o.cut_error) lines.push(`   not cut: ${o.cut_error}`);
  }
  for (const x of r.rejected) lines.push(`rejected ${formatRange(x.start, x.end)} video_id=${x.video_id}: ${x.reason}`);
  return lines.join("\n");
}

// video_cut and video_frames

export type CutResult = { video_id: string; title: string; start: number; end: number; timecode: string } & CutFiles;

export async function videoCut(
  input: { video_id: string; start: number; end: number; frame?: Frame },
  ctx: Context = {},
): Promise<CutResult> {
  const video = await getVideo(await requireIndex(indexName(ctx)), input.video_id);
  const range = clampRange(input, video);
  const files = await cutRange(video, range, input.frame ?? "mid");
  return {
    video_id: video.video_id,
    title: video.title,
    start: round(range.start),
    end: round(range.end),
    timecode: formatRange(range.start, range.end),
    ...files,
  };
}

export function cutText(r: CutResult): string {
  const lines = [`${r.timecode} of "${r.title}"`, `clip: ${r.clip_path}`];
  if (r.frame_path) lines.push(`frame: ${r.frame_path}`);
  return lines.join("\n");
}

export type FramesResult = {
  video_id: string;
  title: string;
  start: number;
  end: number;
  timecode: string;
  sheet_path: string;
  /** Time of each tile, left to right, top to bottom. */
  times: number[];
};

export async function videoFrames(
  input: { video_id: string; start: number; end: number; count?: number },
  ctx: Context = {},
): Promise<FramesResult> {
  const count = input.count ?? SHEET_FRAMES;
  if (!Number.isInteger(count) || count < 1 || count > 12) throw new Error(`count must be a whole number from 1 to 12, got ${count}`);
  const video = await getVideo(await requireIndex(indexName(ctx)), input.video_id);
  const range = clampRange(input, video);
  requireLocalFile(video);
  const sheetPath = join(await dataDir("frames"), `${mediaStem(video, range)}_sheet${count}.jpg`);
  const times = await contactSheet(video.local_path, range, count, sheetPath);
  return {
    video_id: video.video_id,
    title: video.title,
    start: round(range.start),
    end: round(range.end),
    timecode: formatRange(range.start, range.end),
    sheet_path: sheetPath,
    times: times.map(round),
  };
}

export function framesText(r: FramesResult): string {
  return [
    `${r.timecode} of "${r.title}"`,
    `sheet: ${r.sheet_path}`,
    `tiles (left to right, top to bottom): ${r.times.map(formatTimecode).join(", ")}`,
  ].join("\n");
}

// Entities

async function entityCollection(): Promise<string> {
  const found = await api.findEntityCollection(ENTITY_COLLECTION);
  if (found) return found;
  try {
    return await api.createEntityCollection(ENTITY_COLLECTION);
  } catch (error) {
    if (!(error instanceof api.ApiError)) throw error;
    const existing = await api.entityCollectionNames();
    throw new Error(
      `Could not create the entity collection "${ENTITY_COLLECTION}": ${error.message}. ` +
        `Existing collections: ${existing.join(", ") || "none"}. ` +
        "The free plan allows one collection, and the sample collection counts: delete it in the TwelveLabs playground, then retry.",
    );
  }
}

export type EntityAddResult = { entity_id: string; name: string; status: string; added_images: number; extended: boolean };

export async function videoEntityAdd(
  input: { name: string; images?: string[]; from?: { video_id: string; times: number[] } },
  ctx: Context = {},
): Promise<EntityAddResult> {
  const name = input.name.trim().replace(/^@/, "");
  if (!name) throw new Error("An entity needs a name");
  const images = input.images ?? [];
  if (images.length === 0 && !input.from?.times.length) {
    throw new Error("Give reference images, or a video_id and times to grab frames from");
  }
  const urls = images.filter(isUrl);
  const files = images.filter((image) => !isUrl(image)).map((image) => resolve(image.replace(/^~(?=$|\/)/, homedir())));
  const missing = files.filter((file) => !existsSync(file));
  if (missing.length > 0) throw new Error(`Image not found: ${missing.join(", ")}`);
  if (input.from?.times.length) {
    const video = await getVideo(await requireIndex(indexName(ctx)), input.from.video_id);
    requireLocalFile(video);
    const dir = await dataDir("frames");
    for (const t of input.from.times) {
      if (t < 0 || (video.duration > 0 && t > video.duration)) {
        throw new Error(`${formatTimecode(t)} is outside "${video.title}" (${formatTimecode(video.duration)} long)`);
      }
      const framePath = join(dir, `${slug(video.title)}-${video.video_id.slice(-6)}_${t.toFixed(1)}_ref.jpg`);
      await still(video.local_path, t, framePath);
      files.push(framePath);
    }
  }
  ctx.progress?.(`uploading ${files.length + urls.length} reference images`);
  const assetIds = [
    ...(await mapLimit(files, 4, (file) => api.uploadFile(file, "image"))),
    ...(await mapLimit(urls, 4, (url) => api.uploadUrl(url))),
  ];
  await mapLimit(assetIds, 4, (assetId) => waitAsset(assetId, ctx));
  const collectionId = await entityCollection();
  const existing = (await api.listEntities(collectionId)).find((e) => e.name.toLowerCase() === name.toLowerCase());
  const entity = existing
    ? await api.addEntityAssets(collectionId, existing._id, assetIds)
    : await api.createEntity(collectionId, name, assetIds);
  return { entity_id: entity._id, name: entity.name, status: entity.status ?? "processing", added_images: assetIds.length, extended: Boolean(existing) };
}

export function entityAddText(r: EntityAddResult): string {
  return `${r.extended ? "extended" : "created"} @${r.name} (${r.entity_id}) with ${r.added_images} images, status ${r.status}`;
}

export type EntitiesResult = {
  collection: string;
  entities: { entity_id: string; name: string; status: string; images: number }[];
};

export async function videoEntities(): Promise<EntitiesResult> {
  const collectionId = await api.findEntityCollection(ENTITY_COLLECTION);
  if (!collectionId) return { collection: ENTITY_COLLECTION, entities: [] };
  const entities = await api.listEntities(collectionId);
  return {
    collection: ENTITY_COLLECTION,
    entities: entities.map((e) => ({ entity_id: e._id, name: e.name, status: e.status ?? "unknown", images: e.asset_ids?.length ?? 0 })),
  };
}

export function entitiesText(r: EntitiesResult): string {
  if (r.entities.length === 0) return `no entities in "${r.collection}"`;
  return r.entities.map((e) => `@${e.name}  ${e.entity_id}  ${e.status}  ${e.images} images`).join("\n");
}

// video_jockey

export type Citation = {
  /** The cited span of the answer, usually a marker such as "[1]". */
  marker: string;
  type: string;
  video_id?: string;
  item_id?: string;
  title?: string;
  start?: number;
  end?: number;
  timecode?: string;
  url?: string;
};

export type JockeyResult = {
  text: string;
  session_id: string;
  status: string;
  incomplete_reason?: string;
  citations: Citation[];
};

export async function videoJockey(
  input: { question: string; session_id?: string; video_ids?: string[] },
  ctx: Context = {},
): Promise<JockeyResult> {
  const name = indexName(ctx);
  const storeId = await api.findKnowledgeStore(name);
  if (!storeId) {
    throw new Error(`No Jockey knowledge store "${name}" yet. video_add mirrors each video into it; add a video first.`);
  }
  const items = await api.listStoreItems(storeId);
  let content = input.question;
  let itemIds: string[] | undefined;
  if (input.video_ids?.length) {
    const listed = await api.listIndexedAssets(await requireIndex(name));
    itemIds = input.video_ids.map((videoId) => {
      const asset = listed.find((a) => a._id === videoId || a.asset_id === videoId);
      const item = asset && storeItemFor(items, asset.asset_id);
      if (!item || item.status === "failed") {
        throw new Error(`Video ${videoId} is not in the Jockey store. Run video_add with its source to add it.`);
      }
      return item._id;
    });
    content += `\n\nAnswer from these videos only: ${itemIds.map((_, i) => `{{sel:${i}}}`).join(", ")}`;
  }
  ctx.progress?.("asking Jockey");
  const response = await api.createResponse({ storeId, content, sessionId: input.session_id, itemIds });
  if (response.status === "failed") throw new Error("Jockey could not answer. Try again, or narrow the question.");
  const itemsById = new Map(items.map((item) => [item._id, item]));
  // A message without a phase is the answer; other phases are intermediate output.
  const parts = (response.output ?? [])
    .filter((o) => o.type === "message" && (!o.phase || o.phase === "final_answer"))
    .flatMap((o) => o.content ?? [])
    .filter((part) => part.type === "output_text");
  const citations = parts.flatMap((part) => {
    // Annotation offsets count code points, not UTF-16 units.
    const codePoints = Array.from(part.text ?? "");
    return (part.annotations ?? []).map((a): Citation => {
      const item = a.item_id ? itemsById.get(a.item_id) : undefined;
      const videoId = metaText(item?.item_metadata, "video_id");
      const title = a.title || metaText(item?.item_metadata, "title");
      const hasRange = typeof a.start_sec === "number" && typeof a.end_sec === "number";
      return {
        marker: codePoints.slice(a.start_index, a.end_index + 1).join(""),
        type: a.type,
        ...(videoId && { video_id: videoId }),
        ...(a.item_id && { item_id: a.item_id }),
        ...(title && { title }),
        ...(hasRange && {
          start: round(a.start_sec ?? 0),
          end: round(a.end_sec ?? 0),
          timecode: formatRange(a.start_sec ?? 0, a.end_sec ?? 0),
        }),
        ...(a.url && { url: a.url }),
      };
    });
  });
  return {
    text: parts.map((part) => part.text ?? "").join("\n\n"),
    session_id: response.session_id ?? input.session_id ?? "",
    status: response.status ?? "completed",
    ...(response.incomplete_details && { incomplete_reason: response.incomplete_details.reason }),
    citations,
  };
}

export function jockeyText(r: JockeyResult): string {
  const lines = [r.text];
  if (r.incomplete_reason) lines.push(`(answer incomplete: ${r.incomplete_reason})`);
  for (const c of r.citations) {
    const where = c.url ?? [c.timecode, c.title && `"${c.title}"`, c.video_id && `video_id=${c.video_id}`].filter(Boolean).join("  ");
    lines.push(`${c.marker} ${where || c.type}`);
  }
  lines.push(`session: ${r.session_id}`);
  return lines.join("\n");
}
