// Pure logic shared by search and find: timecodes, multi-step chaining, candidate
// merging, inspection-window math, verdict checks and @Name entity mentions.
// Nothing here touches the network or the disk.

import { createHash } from "node:crypto";

export interface Range {
  start: number;
  end: number;
}

/** One clip returned by the search API for one query. Rank 1 is the best match. */
export interface Clip extends Range {
  video_id: string;
  rank: number;
}

/**
 * A search result before formatting.
 * - clip: a hit of a single-query search
 * - sentence: a hit of the full sentence in a multi-step search
 * - chain: one clip per step, linked in order inside one video
 */
export interface Candidate<C extends Clip> extends Range {
  video_id: string;
  kind: "clip" | "sentence" | "chain";
  /** Lower is better: the API rank for a clip, the sum of step ranks for a chain. */
  score: number;
  /** The clips behind the candidate: one per step for a chain, else the single hit. */
  clips: C[];
}

/** Seconds as HH:MM:SS.s, rounded to the nearest tenth. */
export function formatTimecode(seconds: number): string {
  const tenths = Math.round(Math.max(0, seconds) * 10);
  const hours = Math.floor(tenths / 36000);
  const minutes = Math.floor((tenths % 36000) / 600);
  const secondTenths = tenths % 600;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(hours)}:${pad(minutes)}:${pad(Math.floor(secondTenths / 10))}.${secondTenths % 10}`;
}

/** A range as "00:01:23.4 -> 00:01:31.0". */
export function formatRange(start: number, end: number): string {
  return `${formatTimecode(start)} -> ${formatTimecode(end)}`;
}

/** Parses "83.4", "1:23.4" or "00:01:23.4" into seconds. */
export function parseTime(text: string): number {
  const match = /^(?:(?:(\d+):)?(\d+):)?(\d+(?:\.\d+)?)$/.exec(text.trim());
  if (!match) throw new Error(`"${text}" is not a time: use seconds (83.4) or a timecode (1:23.4, 00:01:23.4)`);
  const [, hours, minutes, seconds] = match;
  const secs = Number(seconds);
  if (minutes !== undefined && (secs >= 60 || Number(minutes) >= 60 && hours !== undefined)) {
    throw new Error(`"${text}" is not a valid timecode`);
  }
  return Number(hours ?? 0) * 3600 + Number(minutes ?? 0) * 60 + secs;
}

/** Overlap divided by union of two ranges. */
export function overlapRatio(a: Range, b: Range): number {
  const union = Math.max(a.end, b.end) - Math.min(a.start, b.start);
  if (union <= 0) return 1;
  return Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start)) / union;
}

/**
 * Keeps the better of any two candidates in the same video whose ranges overlap by at
 * least half their union, then returns the best `limit` by score.
 */
export function mergeCandidates<C extends Clip>(candidates: Candidate<C>[], limit: number): Candidate<C>[] {
  const ordered = [...candidates].sort((a, b) => a.score - b.score || a.start - b.start);
  const kept: Candidate<C>[] = [];
  for (const candidate of ordered) {
    if (kept.length === limit) break;
    const duplicate = kept.some(
      (other) => other.video_id === candidate.video_id && overlapRatio(other, candidate) >= 0.5,
    );
    if (!duplicate) kept.push(candidate);
  }
  return kept;
}

/** Ranks the clips of a single-query search. */
export function rankClips<C extends Clip>(clips: C[], limit: number): Candidate<C>[] {
  return mergeCandidates(
    clips.map((clip) => ({ video_id: clip.video_id, start: clip.start, end: clip.end, kind: "clip", score: clip.rank, clips: [clip] })),
    limit,
  );
}

interface Link<C extends Clip> {
  clip: C;
  cost: number;
  previous?: Link<C>;
}

/**
 * Links one clip per step into chains inside the same video. A step's clip must start
 * at or after the previous step's clip start and no later than the previous end plus
 * `maxGap`. For every clip of the last step, keeps the chain with the lowest rank sum.
 */
export function chainSteps<C extends Clip>(steps: C[][], maxGap: number): Candidate<C>[] {
  const [first, ...rest] = steps;
  if (!first) return [];
  let frontier: Link<C>[] = first.map((clip) => ({ clip, cost: clip.rank }));
  for (const stepClips of rest) {
    const next: Link<C>[] = [];
    for (const clip of stepClips) {
      let best: Link<C> | undefined;
      for (const link of frontier) {
        const prev = link.clip;
        if (prev.video_id !== clip.video_id) continue;
        if (clip.start < prev.start || clip.start > prev.end + maxGap) continue;
        if (!best || link.cost < best.cost) best = link;
      }
      if (best) next.push({ clip, cost: best.cost + clip.rank, previous: best });
    }
    frontier = next;
  }
  return frontier.map((last) => {
    const clips: C[] = [];
    for (let link: Link<C> | undefined = last; link; link = link.previous) clips.unshift(link.clip);
    return {
      video_id: last.clip.video_id,
      start: Math.min(...clips.map((c) => c.start)),
      end: Math.max(...clips.map((c) => c.end)),
      kind: "chain",
      score: last.cost,
      clips,
    };
  });
}

/**
 * Ranks a multi-step search: chains of the per-step hits plus the hits of the full
 * sentence. A sentence hit scores its rank once per step, so it competes with a chain
 * whose steps all got that rank.
 */
export function rankSteps<C extends Clip>(sentence: C[], steps: C[][], maxGap: number, limit: number): Candidate<C>[] {
  const sentenceHits: Candidate<C>[] = sentence.map((clip) => ({
    video_id: clip.video_id,
    start: clip.start,
    end: clip.end,
    kind: "sentence",
    score: clip.rank * steps.length,
    clips: [clip],
  }));
  return mergeCandidates([...chainSteps(steps, maxGap), ...sentenceHits], limit);
}

/**
 * The window to inspect around a candidate: padded, clamped to the video, and widened
 * around its center to at least `minLength` seconds (or the whole video if shorter).
 */
export function inspectionWindow(range: Range, pad: number, duration: number, minLength: number): Range {
  let start = Math.max(0, range.start - pad);
  let end = Math.min(duration, range.end + pad);
  if (duration <= minLength) return { start: 0, end: duration };
  if (end - start < minLength) {
    const center = (start + end) / 2;
    start = center - minLength / 2;
    end = center + minLength / 2;
    if (start < 0) {
      end -= start;
      start = 0;
    }
    if (end > duration) {
      start -= end - duration;
      end = duration;
    }
  }
  return { start, end };
}

/** Converts a time Pegasus reports inside an analyzed window to video time. */
export function windowToAbsolute(windowStart: number, t: number): number {
  return windowStart + t;
}

/** What Pegasus answers for one candidate window. Times are seconds from the window start. */
export interface Verdict {
  match: boolean;
  confidence: number;
  who: string;
  evidence: string;
  steps: { step: string; seen: boolean; start: number; end: number }[];
}

export interface CheckedStep extends Range {
  step: string;
}

export type Judgement =
  | { confirmed: true; start: number; end: number; steps: CheckedStep[] }
  | { confirmed: false; reason: string };

/**
 * Confirms a candidate only when Pegasus matched the whole claim with enough
 * confidence and saw every step. The refined range spans the reported steps when they
 * all fall inside the window in order; otherwise the candidate range stands.
 */
export function judge(verdict: Verdict, window: Range, candidate: Range, stepCount: number, minConfidence: number): Judgement {
  if (!verdict.match) return { confirmed: false, reason: "no match" };
  if (!(verdict.confidence >= 0 && verdict.confidence <= 1)) {
    return { confirmed: false, reason: `confidence ${verdict.confidence} outside 0..1` };
  }
  if (verdict.confidence < minConfidence) {
    return { confirmed: false, reason: `confidence ${verdict.confidence} below ${minConfidence}` };
  }
  if (verdict.steps.length !== stepCount) {
    return { confirmed: false, reason: `expected ${stepCount} steps, got ${verdict.steps.length}` };
  }
  const unseen = verdict.steps.findIndex((s) => !s.seen);
  if (unseen >= 0) return { confirmed: false, reason: `step ${unseen + 1} not seen` };

  const absolute = verdict.steps.map((s) => ({
    step: s.step,
    start: windowToAbsolute(window.start, s.start),
    end: windowToAbsolute(window.start, s.end),
  }));
  const epsilon = 1e-6;
  const inside = absolute.every(
    (s) => s.start >= window.start - epsilon && s.end <= window.end + epsilon && s.end >= s.start,
  );
  const start = Math.min(...absolute.map((s) => s.start));
  const end = Math.max(...absolute.map((s) => s.end));
  const refined = inside && end > start ? { start, end } : { start: candidate.start, end: candidate.end };
  return {
    confirmed: true,
    ...refined,
    steps: absolute.map((s) => ({
      step: s.step,
      start: Math.min(window.end, Math.max(window.start, s.start)),
      end: Math.min(window.end, Math.max(window.start, s.end)),
    })),
  };
}

/** A stable id for a confirmed observation: "obs_" plus a short hash. */
export function observationId(videoId: string, start: number, end: number): string {
  const digest = createHash("sha1").update(`${videoId}:${start.toFixed(1)}:${end.toFixed(1)}`).digest("hex");
  return `obs_${digest.slice(0, 10)}`;
}

export interface Entity {
  id: string;
  name: string;
}

export interface Mentions {
  /** The text with each @Name replaced by <@entity_id>, for Marengo search. */
  search: string;
  /** The text with each @Name replaced by the entity's name, for Pegasus prompts. */
  plain: string;
  /** Mentions that match no entity, such as "@Kobe". */
  unknown: string[];
}

/**
 * Resolves @Name mentions against known entities, case-insensitively, with " ", "_" and
 * "-" interchangeable. The longest matching name wins. An "@" inside a word (an email
 * address) or right after "<" (an already-resolved <@id>) is left alone.
 */
export function resolveMentions(text: string, entities: Entity[]): Mentions {
  const patterns = [...entities]
    .sort((a, b) => b.name.length - a.name.length)
    .map((entity) => {
      const name = entity.name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/[ _-]/g, "[ _-]");
      return { entity, pattern: new RegExp(`^${name}(?![\\p{L}\\p{N}_])`, "iu") };
    });
  let search = "";
  let plain = "";
  const unknown: string[] = [];
  let i = 0;
  for (let at = text.indexOf("@"); at >= 0; at = text.indexOf("@", i)) {
    const before = text.slice(i, at);
    search += before;
    plain += before;
    i = at + 1;
    const previous = text.charAt(at - 1);
    if (at > 0 && (/[\p{L}\p{N}_]/u.test(previous) || previous === "<")) {
      search += "@";
      plain += "@";
      continue;
    }
    const rest = text.slice(i);
    const hit = patterns
      .map(({ entity, pattern }) => ({ entity, length: pattern.exec(rest)?.[0].length ?? 0 }))
      .find((candidate) => candidate.length > 0);
    if (hit) {
      search += `<@${hit.entity.id}>`;
      plain += hit.entity.name;
      i += hit.length;
      continue;
    }
    const token = /^[^\s,.;:!?)"']*/u.exec(rest)?.[0] ?? "";
    if (token) unknown.push(`@${token}`);
    search += `@${token}`;
    plain += `@${token}`;
    i += token.length;
  }
  return { search: search + text.slice(i), plain: plain + text.slice(i), unknown };
}
