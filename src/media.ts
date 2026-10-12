// Local media work: the data directory, yt-dlp and direct downloads, ffprobe, and ffmpeg
// cuts, stills and contact sheets. Files are written under a ".partial" name and renamed
// into place, so a failed run never leaves a broken file under the final name.

import { createHash } from "node:crypto";
import { mkdir, rename, rm, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, extname, join } from "node:path";
import { z } from "zod";
import type { Range } from "./search.ts";

const YT_DLP_FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b";
/** URL paths that name a media file are fetched as-is; anything else goes through yt-dlp. */
const DIRECT_MEDIA_PATH = /\.(mp4|m4v|mov|webm|mkv|avi|mpe?g|3gp)$/i;
/** Contact sheets stay within this width so they can go back to an agent as one image. */
const SHEET_MAX_WIDTH = 1280;
/** A lock held this long was left by a crashed run and is taken over. */
const LOCK_STALE_MS = 2 * 60_000;

/** A subdirectory of $TLV_HOME (default ~/.local/share/tlv), created on first use. */
export async function dataDir(sub: "media" | "clips" | "frames" | "locks"): Promise<string> {
  const dir = join(process.env.TLV_HOME || join(homedir(), ".local", "share", "tlv"), sub);
  await mkdir(dir, { recursive: true });
  return dir;
}

/**
 * Runs `fn` holding the lock directory $TLV_HOME/locks/<name>. mkdir is atomic, so tlv
 * processes started together (agents adding videos in parallel) take turns creating a
 * named index or store; the API alone let two parallel creates make two indexes "tlv".
 */
export async function withLock<T>(name: string, fn: () => Promise<T>): Promise<T> {
  const lock = join(await dataDir("locks"), slug(name));
  for (;;) {
    try {
      await mkdir(lock);
      break;
    } catch (error) {
      if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error;
      const heldMs = await stat(lock).then((s) => Date.now() - s.mtimeMs, () => 0);
      if (heldMs > LOCK_STALE_MS) await rm(lock, { recursive: true, force: true });
      else await Bun.sleep(200);
    }
  }
  try {
    return await fn();
  } finally {
    await rm(lock, { recursive: true, force: true });
  }
}

/** Lowercase ASCII words joined by "-", at most 48 characters. */
export function slug(text: string): string {
  const words = text
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .slice(0, 48)
    .replace(/^-+|-+$/g, "");
  return words || "video";
}

async function run(command: string[]): Promise<string> {
  const child = Bun.spawn(command, { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, code] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (code !== 0) {
    const detail = stderr.trim().split("\n").slice(-6).join("\n");
    throw new Error(`${command[0]} exited with ${code}: ${detail}`);
  }
  return stdout;
}

/** Runs `write` on a ".partial" sibling of `output`, then renames it into place. */
async function writeAtomically(output: string, write: (partial: string) => Promise<unknown>): Promise<void> {
  const partial = output.replace(/(\.\w+)$/, ".partial$1");
  try {
    await write(partial);
    await rename(partial, output);
  } catch (error) {
    await rm(partial, { force: true });
    throw error;
  }
}

async function ffmpeg(args: string[], output: string): Promise<void> {
  const command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-abort_on", "empty_output", "-y", ...args];
  await writeAtomically(output, (partial) => run([...command, partial]));
}

export interface Download {
  path: string;
  title: string;
}

/** Downloads a URL into media/: a direct media file as-is, anything else with yt-dlp. */
export async function download(url: string): Promise<Download> {
  const dir = await dataDir("media");
  const { pathname } = new URL(url);
  if (DIRECT_MEDIA_PATH.test(pathname)) {
    const name = basename(decodeURIComponent(pathname));
    const ext = extname(name);
    const title = name.slice(0, -ext.length);
    const hash = createHash("sha1").update(url).digest("hex").slice(0, 8);
    const path = join(dir, `${slug(title)}-${hash}${ext.toLowerCase()}`);
    if (!(await Bun.file(path).exists())) {
      await writeAtomically(path, async (partial) => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Downloading ${url} failed: HTTP ${response.status}`);
        await Bun.write(partial, response);
      });
    }
    return { path, title };
  }
  const stdout = await run([
    "yt-dlp",
    "--no-playlist",
    "--no-progress",
    "-f",
    YT_DLP_FORMAT,
    "--merge-output-format",
    "mp4",
    "-o",
    join(dir, "%(extractor_key)s-%(id)s.%(ext)s"),
    "--print",
    "after_move:%(.{title,filepath})j",
    url,
  ]);
  const line = stdout.split("\n").findLast((printed) => printed.startsWith("{"));
  const info = z.object({ title: z.string(), filepath: z.string() }).safeParse(line ? JSON.parse(line) : null);
  if (!info.success) throw new Error(`yt-dlp printed no downloaded file for ${url}: ${stdout.trim().slice(0, 300)}`);
  return { path: info.data.filepath, title: info.data.title };
}

export interface Probe {
  duration: number;
  width: number;
  height: number;
}

/** Duration and frame size of a video file. */
export async function probe(path: string): Promise<Probe> {
  const stdout = await run([
    "ffprobe",
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-show_entries",
    "stream=width,height:format=duration",
    "-of",
    "json",
    path,
  ]);
  const info = z
    .object({
      streams: z.array(z.object({ width: z.number(), height: z.number() })),
      format: z.object({ duration: z.coerce.number() }),
    })
    .parse(JSON.parse(stdout));
  const [stream] = info.streams;
  if (!stream) throw new Error(`${path} has no video stream`);
  return { duration: info.format.duration, width: stream.width, height: stream.height };
}

/** Re-encodes [start, end] of a video into an mp4 that starts playing before it fully loads. */
export async function cut(input: string, range: Range, output: string): Promise<void> {
  await ffmpeg(
    [
      "-ss", range.start.toFixed(3), "-i", input, "-t", (range.end - range.start).toFixed(3),
      "-map", "0:v:0", "-map", "0:a:0?",
      "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
      "-c:a", "aac", "-movflags", "+faststart",
    ],
    output,
  );
}

/** Saves the frame at `t` seconds as a JPEG, scaled to `width` pixels when given. */
export async function still(input: string, t: number, output: string, width?: number): Promise<void> {
  const scale = width ? ["-vf", `scale=${width}:-2`] : [];
  await ffmpeg(["-ss", t.toFixed(3), "-i", input, "-frames:v", "1", ...scale, "-q:v", "2", "-update", "1"], output);
}

/**
 * Tiles `count` evenly spaced frames of [start, end] into one JPEG at most 1280 px wide,
 * left to right, top to bottom. Returns the time of each tile.
 */
export async function contactSheet(input: string, range: Range, count: number, output: string): Promise<number[]> {
  const { width, height } = await probe(input);
  const step = (range.end - range.start) / count;
  const times = Array.from({ length: count }, (_, i) => range.start + (i + 0.5) * step);
  const columns = count <= 3 ? count : count === 4 ? 2 : count <= 9 ? 3 : 4;
  const tileWidth = Math.floor(SHEET_MAX_WIDTH / columns / 2) * 2;
  const tileHeight = Math.max(2, Math.round((tileWidth * height) / width / 2) * 2);
  const tiles = times.map((_, i) => `[${i}:v]scale=${tileWidth}:${tileHeight},setsar=1[t${i}]`);
  const layout = times.map((_, i) => `${(i % columns) * tileWidth}_${Math.floor(i / columns) * tileHeight}`).join("|");
  const graph =
    count === 1
      ? `[0:v]scale=${tileWidth}:${tileHeight},setsar=1[out]`
      : `${tiles.join(";")};${times.map((_, i) => `[t${i}]`).join("")}xstack=inputs=${count}:layout=${layout}:fill=black[out]`;
  await ffmpeg(
    [
      ...times.flatMap((t) => ["-ss", t.toFixed(3), "-i", input]),
      "-filter_complex", graph, "-map", "[out]", "-frames:v", "1", "-q:v", "3", "-update", "1",
    ],
    output,
  );
  return times;
}
