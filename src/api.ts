// TwelveLabs REST client: the API key, the error type, 429 handling and the endpoints
// tlv uses. Every response is parsed with a schema so a changed API shape fails loudly.

import { basename } from "node:path";
import { z } from "zod";

const BASE_URL = "https://api.twelvelabs.io/v1.3";
const KEY_ENV = "TWELVELABS_API_KEY";
const RATE_LIMIT_RETRIES = 3;
const RATE_LIMIT_MAX_WAIT_MS = 60_000;
/** Wait used when a 429 carries neither Retry-After nor a reset header. */
const RATE_LIMIT_FALLBACK_WAIT_MS = 5_000;
/** Local files up to this size go up in one request; larger ones use multipart upload. */
const DIRECT_UPLOAD_MAX_BYTES = 200 * 1000 * 1000;
/** Pegasus and Jockey can think for minutes without sending a byte. */
const SLOW_CALL_IDLE_TIMEOUT_MS = 30 * 60_000;

/** An error response from the API, with its status, code and message. */
export class ApiError extends Error {
  override name = "ApiError";
  constructor(
    readonly status: number,
    readonly code: string,
    readonly endpoint: string,
    detail: string,
  ) {
    super(`TwelveLabs ${endpoint} failed (${status} ${code}): ${detail}`);
  }
}

function apiKey(): string {
  const key = process.env[KEY_ENV];
  if (!key) {
    throw new Error(`${KEY_ENV} is not set. Run tlv through the vault: mem-secret run ${KEY_ENV} -- tlv <command>`);
  }
  return key;
}

/** Milliseconds to wait before retrying a 429, from Retry-After or the reset headers. */
function rateLimitWait(headers: Headers): number {
  const retryAfter = Number(headers.get("retry-after"));
  let wait = retryAfter > 0 ? retryAfter * 1000 : RATE_LIMIT_FALLBACK_WAIT_MS;
  if (!(retryAfter > 0)) {
    // X-Ratelimit-<Label>-Reset holds the unix time a window resets. Wait for the
    // windows that are used up, or for all of them when none says so.
    const resets = new Map<string, number>();
    const exhausted = new Set<string>();
    for (const [name, value] of headers) {
      const match = /^x-ratelimit-(?:(.+)-)?(reset|remaining)$/i.exec(name);
      if (!match) continue;
      const label = match[1]?.toLowerCase() ?? "";
      if (match[2]?.toLowerCase() === "reset") resets.set(label, Number(value));
      else if (Number(value) === 0) exhausted.add(label);
    }
    const resetTimes = [...resets]
      .filter(([label, at]) => Number.isFinite(at) && (exhausted.size === 0 || exhausted.has(label)))
      .map(([, at]) => at);
    if (resetTimes.length > 0) wait = Math.max(...resetTimes) * 1000 - Date.now();
  }
  return Math.min(RATE_LIMIT_MAX_WAIT_MS, Math.max(1000, wait));
}

type Query = Record<string, string | number | undefined>;

interface RequestInit {
  query?: Query;
  json?: unknown;
  form?: FormData;
  idleTimeoutMs?: number;
}

async function request<S extends z.ZodType>(
  schema: S,
  method: "GET" | "POST" | "PATCH",
  path: string,
  init: RequestInit = {},
): Promise<z.output<S>> {
  const url = new URL(BASE_URL + path);
  for (const [key, value] of Object.entries(init.query ?? {})) {
    if (value !== undefined) url.searchParams.set(key, String(value));
  }
  const headers: Record<string, string> = { "x-api-key": apiKey() };
  if (init.json !== undefined) headers["content-type"] = "application/json";
  const body = init.json !== undefined ? JSON.stringify(init.json) : init.form;
  const endpoint = `${method} ${path}`;
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, { method, headers, body, timeout: init.idleTimeoutMs ?? true });
    if (response.status === 429 && attempt < RATE_LIMIT_RETRIES) {
      await Bun.sleep(rateLimitWait(response.headers));
      continue;
    }
    const text = await response.text();
    if (!response.ok) {
      let code = "http_error";
      let message = text.trim().slice(0, 500) || response.statusText;
      try {
        const parsed = errorBody.parse(JSON.parse(text));
        code = parsed.code ?? code;
        message = parsed.message ?? message;
      } catch {
        // Not a JSON error body (a 429 can be empty): keep the raw text.
      }
      throw new ApiError(response.status, code, endpoint, message);
    }
    const parsed = schema.safeParse(text ? JSON.parse(text) : {});
    if (!parsed.success) {
      throw new Error(`TwelveLabs ${endpoint} returned an unexpected shape:\n${z.prettifyError(parsed.error)}`);
    }
    return parsed.data;
  }
}

const errorBody = z.object({ code: z.string().optional(), message: z.string().optional() });
const created = z.object({ _id: z.string() });
const metadata = z.record(z.string(), z.unknown());

/** Reads every page of a list endpoint, 50 items at a time. */
async function listAll<S extends z.ZodType>(item: S, path: string, query: Query = {}): Promise<z.output<S>[]> {
  const pageSchema = z.object({
    data: z.array(item).nullish(),
    page_info: z.object({ total_page: z.number().optional() }).nullish(),
  });
  const items: z.output<S>[] = [];
  for (let page = 1; ; page++) {
    const result = await request(pageSchema, "GET", path, { query: { ...query, page, page_limit: 50 } });
    const data = result.data ?? [];
    items.push(...data);
    if (data.length === 0 || page >= (result.page_info?.total_page ?? page)) return items;
  }
}

const id = (value: string) => encodeURIComponent(value);

// Indexes

const indexSchema = z.object({ _id: z.string(), index_name: z.string() });

export async function findIndex(name: string): Promise<string | undefined> {
  const indexes = await listAll(indexSchema, "/indexes", { index_name: name });
  return indexes.find((index) => index.index_name === name)?._id;
}

export async function createIndex(name: string): Promise<string> {
  const body = {
    index_name: name,
    models: [{ model_name: "marengo3.0", model_options: ["visual", "audio"] }],
    addons: ["thumbnail"],
  };
  return (await request(created, "POST", "/indexes", { json: body }))._id;
}

// Assets

const assetSchema = z.object({ _id: z.string(), status: z.string(), error: z.unknown().optional() });
export type Asset = z.output<typeof assetSchema>;

export async function getAsset(assetId: string): Promise<Asset> {
  return request(assetSchema, "GET", `/assets/${id(assetId)}`);
}

/** Uploads an image the API fetches from a public URL. Returns the asset id. */
export async function uploadUrl(url: string): Promise<string> {
  const form = new FormData();
  form.append("method", "url");
  form.append("url", url);
  return (await request(created, "POST", "/assets", { form }))._id;
}

/**
 * Uploads a local file: one request up to 200 MB, multipart upload above that.
 * Returns the asset id; the asset may still be processing.
 */
export async function uploadFile(
  path: string,
  kind: "video" | "image",
  onChunk?: (done: number, total: number) => void,
): Promise<string> {
  const file = Bun.file(path);
  if (file.size > DIRECT_UPLOAD_MAX_BYTES) return uploadMultipart(path, kind, onChunk);
  const form = new FormData();
  form.append("method", "direct");
  form.append("file", file, basename(path));
  form.append("filename", basename(path));
  return (await request(created, "POST", "/assets", { form }))._id;
}

const uploadUrls = z.array(z.object({ chunk_index: z.number(), url: z.string() }));
const multipartSession = z.object({
  upload_id: z.string(),
  asset_id: z.string(),
  upload_urls: uploadUrls,
  upload_headers: z.record(z.string(), z.string()).nullish(),
  chunk_size: z.number(),
  total_chunks: z.number(),
});
const multipartStatus = z.object({ status: z.string() });

async function uploadMultipart(
  path: string,
  kind: "video" | "image",
  onChunk?: (done: number, total: number) => void,
): Promise<string> {
  const file = Bun.file(path);
  const session = await request(multipartSession, "POST", "/assets/multipart-uploads", {
    json: { filename: basename(path), type: kind, total_size: file.size },
  });
  const base = `/assets/multipart-uploads/${id(session.upload_id)}`;
  const urls = new Map(session.upload_urls.map((u) => [u.chunk_index, u.url]));
  const headers = { "content-type": "application/octet-stream", ...session.upload_headers };
  const total = session.total_chunks;
  let pending: { chunk_index: number; proof: string; proof_type: "etag"; chunk_size: number }[] = [];
  for (let index = 1; index <= total; index++) {
    if (!urls.has(index)) {
      const more = await request(z.object({ upload_urls: uploadUrls }), "POST", `${base}/presigned-urls`, {
        json: { start: index, count: Math.min(50, total - index + 1) },
      });
      for (const u of more.upload_urls) urls.set(u.chunk_index, u.url);
    }
    const url = urls.get(index);
    if (!url) throw new Error(`TwelveLabs gave no upload URL for chunk ${index} of ${total}`);
    const offset = (index - 1) * session.chunk_size;
    const chunk = file.slice(offset, Math.min(file.size, offset + session.chunk_size));
    const response = await fetch(url, { method: "PUT", headers, body: chunk });
    if (!response.ok) {
      throw new Error(`Uploading chunk ${index} of ${total} failed: HTTP ${response.status} ${(await response.text()).slice(0, 300)}`);
    }
    const etag = response.headers.get("etag")?.replaceAll('"', "");
    if (!etag) throw new Error(`Uploading chunk ${index} of ${total} returned no ETag`);
    pending.push({ chunk_index: index, proof: etag, proof_type: "etag", chunk_size: chunk.size });
    if (pending.length === 10 || index === total) {
      await request(z.object({}), "POST", base, { json: { completed_chunks: pending } });
      pending = [];
    }
    onChunk?.(index, total);
  }
  for (;;) {
    const { status } = await request(multipartStatus, "GET", base);
    if (status === "completed") return session.asset_id;
    if (status === "failed" || status === "expired") throw new Error(`Multipart upload of ${basename(path)} ${status}`);
    await Bun.sleep(2000);
  }
}

// Indexed assets (the videos of an index)

const indexedAssetSchema = z.object({
  _id: z.string(),
  asset_id: z.string(),
  status: z.string(),
  system_metadata: z.object({ duration: z.number().nullish(), filename: z.string().nullish() }).nullish(),
  user_metadata: metadata.nullish(),
});
export type IndexedAsset = z.output<typeof indexedAssetSchema>;

export async function indexAsset(indexId: string, assetId: string, userMetadata: Record<string, string>): Promise<string> {
  const path = `/indexes/${id(indexId)}/indexed-assets`;
  return (await request(created, "POST", path, { json: { asset_id: assetId, user_metadata: userMetadata } }))._id;
}

export async function getIndexedAsset(indexId: string, videoId: string): Promise<IndexedAsset> {
  return request(indexedAssetSchema, "GET", `/indexes/${id(indexId)}/indexed-assets/${id(videoId)}`);
}

/** Lists the videos of an index. The list carries no user_metadata; retrieve each video for that. */
export async function listIndexedAssets(indexId: string): Promise<IndexedAsset[]> {
  return listAll(indexedAssetSchema, `/indexes/${id(indexId)}/indexed-assets`);
}

export async function updateIndexedAssetMetadata(indexId: string, videoId: string, userMetadata: Record<string, string>): Promise<void> {
  const path = `/indexes/${id(indexId)}/indexed-assets/${id(videoId)}`;
  await request(z.unknown(), "PATCH", path, { json: { user_metadata: userMetadata } });
}

// Search (Marengo)

const searchClipSchema = z.object({
  video_id: z.string(),
  start: z.number(),
  end: z.number(),
  rank: z.number(),
  thumbnail_url: z.string().nullish(),
  user_metadata: metadata.nullish(),
});
export type SearchClip = z.output<typeof searchClipSchema>;

export async function search(params: {
  indexId: string;
  query: string;
  modalities: string[];
  videoIds?: string[];
}): Promise<SearchClip[]> {
  const form = new FormData();
  form.append("index_id", params.indexId);
  form.append("query_text", params.query);
  for (const modality of params.modalities) form.append("search_options", modality);
  form.append("operator", "or");
  form.append("group_by", "clip");
  form.append("page_limit", "50");
  form.append("include_user_metadata", "true");
  if (params.videoIds?.length) form.append("filter", JSON.stringify({ id: params.videoIds }));
  return (await request(z.object({ data: z.array(searchClipSchema).nullish() }), "POST", "/search", { form })).data ?? [];
}

// Analyze (Pegasus)

const analysisSchema = z.object({ data: z.string(), finish_reason: z.string().nullish() });
export type Analysis = z.output<typeof analysisSchema>;

/** Asks Pegasus about an asset, or about the window [start, end] of it when given. */
export async function analyze(params: {
  assetId: string;
  prompt: string;
  start?: number;
  end?: number;
  jsonSchema?: Record<string, unknown>;
}): Promise<Analysis> {
  const body = {
    model_name: "pegasus1.6",
    video: { type: "asset_id", asset_id: params.assetId },
    prompt: params.prompt,
    temperature: 0.2,
    stream: false,
    start_time: params.start,
    end_time: params.end,
    response_format: params.jsonSchema ? { type: "json_schema", json_schema: params.jsonSchema } : undefined,
  };
  return request(analysisSchema, "POST", "/analyze", { json: body, idleTimeoutMs: SLOW_CALL_IDLE_TIMEOUT_MS });
}

// Entities

const namedSchema = z.object({ _id: z.string(), name: z.string() });
const entitySchema = z.object({
  _id: z.string(),
  name: z.string(),
  status: z.string().nullish(),
  asset_ids: z.array(z.string()).nullish(),
});
export type Entity = z.output<typeof entitySchema>;

export async function findEntityCollection(name: string): Promise<string | undefined> {
  const collections = await listAll(namedSchema, "/entity-collections", { name });
  return collections.find((c) => c.name === name)?._id;
}

/** Names of all entity collections, for errors when creating one fails. */
export async function entityCollectionNames(): Promise<string[]> {
  return (await listAll(namedSchema, "/entity-collections")).map((c) => c.name);
}

export async function createEntityCollection(name: string): Promise<string> {
  return (await request(created, "POST", "/entity-collections", { json: { name } }))._id;
}

export async function listEntities(collectionId: string): Promise<Entity[]> {
  return listAll(entitySchema, `/entity-collections/${id(collectionId)}/entities`);
}

export async function createEntity(collectionId: string, name: string, assetIds: string[]): Promise<Entity> {
  const path = `/entity-collections/${id(collectionId)}/entities`;
  return request(entitySchema, "POST", path, { json: { name, asset_ids: assetIds } });
}

export async function addEntityAssets(collectionId: string, entityId: string, assetIds: string[]): Promise<Entity> {
  const path = `/entity-collections/${id(collectionId)}/entities/${id(entityId)}/assets`;
  return request(entitySchema, "POST", path, { json: { asset_ids: assetIds } });
}

// Knowledge stores and Jockey

const storeItemSchema = z.object({
  _id: z.string(),
  asset_id: z.string().nullish(),
  status: z.string().nullish(),
  item_metadata: metadata.nullish(),
});
export type StoreItem = z.output<typeof storeItemSchema>;

export async function findKnowledgeStore(name: string): Promise<string | undefined> {
  return (await listAll(namedSchema, "/knowledge-stores")).find((store) => store.name === name)?._id;
}

export async function createKnowledgeStore(name: string): Promise<string> {
  return (await request(created, "POST", "/knowledge-stores", { json: { name } }))._id;
}

export async function listStoreItems(storeId: string): Promise<StoreItem[]> {
  return listAll(storeItemSchema, `/knowledge-stores/${id(storeId)}/items`);
}

export async function createStoreItem(storeId: string, assetId: string, itemMetadata: Record<string, string>): Promise<StoreItem> {
  const path = `/knowledge-stores/${id(storeId)}/items`;
  return request(storeItemSchema, "POST", path, {
    json: { asset_id: assetId, asset_type: "video", item_metadata: itemMetadata },
  });
}

const annotationSchema = z.object({
  type: z.string(),
  start_index: z.number(),
  end_index: z.number(),
  item_id: z.string().nullish(),
  collection_id: z.string().nullish(),
  start_sec: z.number().nullish(),
  end_sec: z.number().nullish(),
  title: z.string().nullish(),
  url: z.string().nullish(),
});
const jockeyResponseSchema = z.object({
  status: z.string().nullish(),
  session_id: z.string().nullish(),
  incomplete_details: z.object({ reason: z.string() }).nullish(),
  output: z
    .array(
      z.object({
        type: z.string(),
        phase: z.string().nullish(),
        content: z
          .array(z.object({ type: z.string(), text: z.string().nullish(), annotations: z.array(annotationSchema).nullish() }))
          .nullish(),
      }),
    )
    .nullish(),
});
export type JockeyResponse = z.output<typeof jockeyResponseSchema>;

/** Asks Jockey a question over a knowledge store. Items narrow the scope to those items. */
export async function createResponse(params: {
  storeId: string;
  content: string;
  sessionId?: string;
  itemIds?: string[];
}): Promise<JockeyResponse> {
  const body = {
    knowledge_store_id: params.storeId,
    input: [{ type: "message", role: "user", content: params.content }],
    stream: false,
    session_id: params.sessionId,
    selections: params.itemIds?.map((itemId) => ({ kind: "item", id: itemId })),
  };
  return request(jockeyResponseSchema, "POST", "/responses", { json: body, idleTimeoutMs: SLOW_CALL_IDLE_TIMEOUT_MS });
}
