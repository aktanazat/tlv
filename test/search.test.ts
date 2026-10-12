import { describe, expect, test } from "bun:test";
import {
  type Candidate,
  type Clip,
  type Verdict,
  chainSteps,
  distinctMoments,
  formatRange,
  formatTimecode,
  inspectionWindow,
  judge,
  mergeCandidates,
  observationId,
  overlapRatio,
  parseTime,
  rankClips,
  rankSteps,
  resolveMentions,
  windowToAbsolute,
} from "../src/search.ts";

const clip = (video_id: string, start: number, end: number, rank: number): Clip => ({ video_id, start, end, rank });

const hit = (video_id: string, start: number, end: number, score: number): Candidate<Clip> => ({
  video_id,
  start,
  end,
  kind: "clip",
  score,
  clips: [clip(video_id, start, end, score)],
});

/** [video_id, start, end, score] of each candidate, in order. */
const spans = (candidates: Candidate<Clip>[]) => candidates.map((c) => [c.video_id, c.start, c.end, c.score]);

describe("timecodes", () => {
  test.each([
    ["zero", 0, "00:00:00.0"],
    ["seconds with tenths", 83.44, "00:01:23.4"],
    ["rounding that carries into the minute", 59.96, "00:01:00.0"],
    ["rounding that carries into the hour", 3599.97, "01:00:00.0"],
    ["hours, minutes and seconds", 7322.5, "02:02:02.5"],
    ["negative time as zero", -2, "00:00:00.0"],
  ])("formatTimecode renders %s as HH:MM:SS.s", (_, seconds, expected) => {
    expect(formatTimecode(seconds)).toBe(expected);
  });

  test("formatRange joins two timecodes with an arrow", () => {
    expect(formatRange(83.4, 91)).toBe("00:01:23.4 -> 00:01:31.0");
  });

  test.each([
    ["83.4", 83.4],
    ["1:23.4", 83.4],
    ["00:01:23.4", 83.4],
    ["2:00:00", 7200],
    ["75:00", 4500],
    [" 5 ", 5],
  ])("parseTime reads %p as %p seconds", (text, expected) => {
    expect(parseTime(text)).toBeCloseTo(expected, 9);
  });

  test.each(["", "abc", "-5", "1:60", "1:75:00", "1:2:3:4", "1.5:00"])("parseTime rejects %p", (text) => {
    expect(() => parseTime(text)).toThrow();
  });
});

describe("candidate merging", () => {
  test.each([
    ["identical ranges", { start: 0, end: 10 }, { start: 0, end: 10 }, 1],
    ["disjoint ranges", { start: 0, end: 10 }, { start: 20, end: 30 }, 0],
    ["half-shifted ranges", { start: 0, end: 10 }, { start: 5, end: 15 }, 1 / 3],
    ["a range inside another", { start: 0, end: 10 }, { start: 2, end: 4 }, 0.2],
    ["two equal instants", { start: 5, end: 5 }, { start: 5, end: 5 }, 1],
  ])("overlapRatio of %s is overlap over union", (_, a, b, expected) => {
    expect(overlapRatio(a, b)).toBeCloseTo(expected, 9);
  });

  test.each([
    [
      "drops the worse of two same-video candidates overlapping by half or more",
      [hit("v1", 0, 10, 2), hit("v1", 0, 5, 1)],
      [["v1", 0, 5, 1]],
    ],
    [
      "keeps same-video candidates overlapping by less than half",
      [hit("v1", 0, 10, 2), hit("v1", 0, 4.9, 1)],
      [
        ["v1", 0, 4.9, 1],
        ["v1", 0, 10, 2],
      ],
    ],
    [
      "keeps overlapping candidates from different videos",
      [hit("v1", 0, 10, 1), hit("v2", 0, 10, 2)],
      [
        ["v1", 0, 10, 1],
        ["v2", 0, 10, 2],
      ],
    ],
    [
      "orders by score, then by earlier start",
      [hit("v1", 50, 60, 2), hit("v2", 30, 40, 1), hit("v3", 10, 20, 2)],
      [
        ["v2", 30, 40, 1],
        ["v3", 10, 20, 2],
        ["v1", 50, 60, 2],
      ],
    ],
  ])("mergeCandidates %s", (_, candidates, expected) => {
    expect(spans(mergeCandidates(candidates, 10))).toEqual(expected);
  });

  test("mergeCandidates returns at most limit candidates, the best ones", () => {
    const candidates = [hit("v1", 0, 1, 3), hit("v2", 0, 1, 1), hit("v3", 0, 1, 2)];
    expect(spans(mergeCandidates(candidates, 2))).toEqual([
      ["v2", 0, 1, 1],
      ["v3", 0, 1, 2],
    ]);
  });

  test("rankClips scores each clip by its search rank and merges duplicates", () => {
    const clips = [clip("v1", 10, 20, 2), clip("v1", 11, 20, 1), clip("v2", 0, 5, 3)];
    expect(spans(rankClips(clips, 10))).toEqual([
      ["v1", 11, 20, 1],
      ["v2", 0, 5, 3],
    ]);
  });
});

describe("multi-step chaining", () => {
  test.each([
    [
      "links a next step that starts within max_gap of the previous end",
      [[clip("v1", 10, 15, 1)], [clip("v1", 18, 22, 2)]],
      [["v1", 10, 22, 3]],
    ],
    [
      "links a next step that starts exactly max_gap after the previous end",
      [[clip("v1", 10, 15, 1)], [clip("v1", 20, 22, 1)]],
      [["v1", 10, 22, 2]],
    ],
    [
      "does not link a next step that starts later than max_gap after the previous end",
      [[clip("v1", 10, 15, 1)], [clip("v1", 20.1, 22, 1)]],
      [],
    ],
    [
      "does not link a next step that starts before the previous step",
      [[clip("v1", 10, 15, 1)], [clip("v1", 9, 12, 1)]],
      [],
    ],
    ["does not link steps across videos", [[clip("v1", 10, 15, 1)], [clip("v2", 16, 18, 1)]], []],
    [
      "keeps the lowest rank sum for each last-step clip",
      [[clip("v1", 10, 15, 3), clip("v1", 11, 15, 1)], [clip("v1", 16, 18, 1)]],
      [["v1", 11, 18, 2]],
    ],
    [
      "spans from the earliest start to the latest end of its steps",
      [[clip("v1", 10, 30, 1)], [clip("v1", 12, 14, 1)]],
      [["v1", 10, 30, 2]],
    ],
    [
      "chains three steps in order",
      [[clip("v1", 0, 2, 1)], [clip("v1", 3, 4, 2)], [clip("v1", 6, 9, 3)]],
      [["v1", 0, 9, 6]],
    ],
  ])("chainSteps %s", (_, steps, expected) => {
    expect(spans(chainSteps(steps, 5))).toEqual(expected);
  });

  test("chainSteps keeps one clip per step, in step order", () => {
    const first = clip("v1", 10, 15, 1);
    const second = clip("v1", 18, 22, 2);
    const [chain] = chainSteps([[first], [second]], 5);
    expect(chain?.kind).toBe("chain");
    expect(chain?.clips).toEqual([first, second]);
  });

  test("rankSteps scores a sentence hit as its rank once per step", () => {
    const steps = [[clip("v1", 10, 15, 1)], [clip("v1", 16, 20, 1)], [clip("v1", 21, 25, 1)]];
    const ranked = rankSteps([clip("v2", 0, 5, 2)], steps, 5, 10);
    expect(ranked.map((c) => [c.kind, c.video_id, c.score])).toEqual([
      ["chain", "v1", 3],
      ["sentence", "v2", 6],
    ]);
  });

  test("rankSteps merges a sentence hit into a better chain over the same moment", () => {
    const steps = [[clip("v1", 10, 15, 1)], [clip("v1", 16, 20, 1)]];
    const ranked = rankSteps([clip("v1", 10, 20, 4)], steps, 5, 10);
    expect(ranked.map((c) => [c.kind, c.start, c.end, c.score])).toEqual([["chain", 10, 20, 2]]);
  });
});

describe("inspection windows", () => {
  test.each([
    ["pads both sides", { start: 10, end: 20 }, 3, 100, { start: 7, end: 23 }],
    ["clamps padding at the video start", { start: 1, end: 5 }, 3, 100, { start: 0, end: 8 }],
    ["clamps padding at the video end", { start: 95, end: 99 }, 3, 100, { start: 92, end: 100 }],
    ["widens a short range around its center", { start: 50, end: 51 }, 0, 100, { start: 47.5, end: 53.5 }],
    ["shifts a widened window that would start before 0", { start: 0, end: 1 }, 0, 100, { start: 0, end: 6 }],
    ["shifts a widened window that would end after the video", { start: 99, end: 100 }, 0, 100, { start: 94, end: 100 }],
    ["covers a video shorter than the minimum", { start: 1, end: 2 }, 0, 5, { start: 0, end: 5 }],
    ["covers a video exactly the minimum long", { start: 1, end: 2 }, 0, 6, { start: 0, end: 6 }],
  ])("inspectionWindow %s", (_, range, pad, duration, expected) => {
    expect(inspectionWindow(range, pad, duration, 6)).toEqual(expected);
  });

  test("windowToAbsolute adds the window start to a time inside the window", () => {
    expect(windowToAbsolute(120, 12)).toBe(132);
  });
});

describe("verdicts", () => {
  const window = { start: 100, end: 110 };
  const candidate = { start: 102, end: 107 };
  const verdict = (overrides: Partial<Verdict> = {}): Verdict => ({
    match: true,
    confidence: 0.9,
    who: "LeBron James",
    evidence: "jersey 23",
    steps: [
      { step: "steals", seen: true, start: 1, end: 3 },
      { step: "dunks", seen: true, start: 5, end: 8 },
    ],
    ...overrides,
  });

  test.each([
    ["a verdict without a match", verdict({ match: false }), "no match"],
    ["a confidence above 1", verdict({ confidence: 1.5 }), "confidence 1.5 outside 0..1"],
    ["a confidence that is not a number", verdict({ confidence: Number.NaN }), "confidence NaN outside 0..1"],
    ["a confidence below the minimum", verdict({ confidence: 0.5 }), "confidence 0.5 below 0.6"],
    ["a different number of steps", verdict({ steps: [{ step: "steals", seen: true, start: 1, end: 3 }] }), "expected 2 steps, got 1"],
    [
      "an unseen step",
      verdict({ steps: [{ step: "steals", seen: true, start: 1, end: 3 }, { step: "dunks", seen: false, start: 5, end: 8 }] }),
      "step 2 not seen",
    ],
    [
      "a step that starts before the step before it",
      verdict({ steps: [{ step: "steals", seen: true, start: 5, end: 8 }, { step: "dunks", seen: true, start: 1, end: 3 }] }),
      "step 2 starts before step 1",
    ],
  ])("judge rejects %s", (_, given, reason) => {
    expect(judge(given, window, candidate, 2, 0.6)).toEqual({ confirmed: false, reason });
  });

  test("judge confirms a confidence equal to the minimum", () => {
    expect(judge(verdict({ confidence: 0.6 }), window, candidate, 2, 0.6).confirmed).toBe(true);
  });

  test("judge confirms steps that start at the same time", () => {
    const steps = [
      { step: "scores", seen: true, start: 2, end: 4 },
      { step: "celebrates", seen: true, start: 2, end: 6 },
    ];
    expect(judge(verdict({ steps }), window, candidate, 2, 0.6)).toMatchObject({ confirmed: true, start: 102, end: 106 });
  });

  test("judge refines the range to the steps, in video time, when they fit the window", () => {
    expect(judge(verdict(), window, candidate, 2, 0.6)).toEqual({
      confirmed: true,
      start: 101,
      end: 108,
      steps: [
        { step: "steals", start: 101, end: 103 },
        { step: "dunks", start: 105, end: 108 },
      ],
    });
  });

  test.each([
    ["a step ends after the window", [{ step: "a", seen: true, start: 1, end: 12 }]],
    ["a step starts before the window", [{ step: "a", seen: true, start: -1, end: 3 }]],
    [
      "a step ends before it starts",
      [
        { step: "a", seen: true, start: 1, end: 3 },
        { step: "b", seen: true, start: 6, end: 5 },
      ],
    ],
    ["the steps span no time", [{ step: "a", seen: true, start: 4, end: 4 }]],
  ])("judge keeps the candidate range when %s", (_, steps) => {
    const judgement = judge(verdict({ steps }), window, candidate, steps.length, 0.6);
    expect(judgement).toMatchObject({ confirmed: true, start: 102, end: 107 });
  });

  test("judge clamps reported steps into the window", () => {
    const steps = [{ step: "a", seen: true, start: -1, end: 12 }];
    expect(judge(verdict({ steps }), window, candidate, 1, 0.6)).toMatchObject({
      steps: [{ step: "a", start: 100, end: 110 }],
    });
  });
});

describe("distinct moments", () => {
  const moment = (id: string, video_id: string, start: number, end: number, confidence: number) => ({ id, video_id, start, end, confidence });

  test.each([
    [
      "drops the less confident of two moments in one video overlapping by half or more",
      [moment("a", "v1", 0, 10, 0.9), moment("b", "v1", 2, 12, 0.95)],
      ["b"],
      [["a", "b"]],
    ],
    [
      "drops a repeat whose times moved two seconds, under half their union",
      [moment("a", "v1", 255.5, 261, 0.95), moment("b", "v1", 257.6, 263.6, 0.95)],
      ["a"],
      [["b", "a"]],
    ],
    [
      "keeps the earlier of two equally confident repeats",
      [moment("late", "v1", 1, 11, 0.9), moment("early", "v1", 0, 10, 0.9)],
      ["early"],
      [["late", "early"]],
    ],
    [
      "keeps moments overlapping by less than half",
      [moment("a", "v1", 0, 10, 0.9), moment("b", "v1", 6, 16, 0.8)],
      ["a", "b"],
      [],
    ],
    [
      "keeps the same range in different videos",
      [moment("a", "v1", 0, 10, 0.9), moment("b", "v2", 0, 10, 0.8)],
      ["a", "b"],
      [],
    ],
  ])("distinctMoments %s", (_, moments, kept, repeats) => {
    const result = distinctMoments(moments);
    expect(result.kept.map((m) => m.id)).toEqual(kept);
    expect(result.repeats.map((r) => [r.moment.id, r.of.id])).toEqual(repeats);
  });
});

describe("observation ids", () => {
  test("observationId is obs_ and ten hex digits", () => {
    expect(observationId("v1", 83.4, 91)).toMatch(/^obs_[0-9a-f]{10}$/);
  });

  test("observationId is equal for times equal to the tenth of a second", () => {
    expect(observationId("v1", 83.41, 91.04)).toBe(observationId("v1", 83.4, 91));
  });

  test.each([
    ["video", observationId("v2", 83.4, 91)],
    ["start", observationId("v1", 83.5, 91)],
    ["end", observationId("v1", 83.4, 91.1)],
  ])("observationId changes with the %s", (_, other) => {
    expect(other).not.toBe(observationId("v1", 83.4, 91));
  });
});

describe("entity mentions", () => {
  const entities = [
    { id: "e_lebron", name: "LeBron" },
    { id: "e_lebron_james", name: "LeBron James" },
    { id: "e_aj", name: "A.J. Green" },
  ];

  test.each([
    ["a name written with underscores", "@LeBron_James dunks", "<@e_lebron_james> dunks", "LeBron James dunks", []],
    ["a name in another case, with a hyphen", "@lebron-james dunks", "<@e_lebron_james> dunks", "LeBron James dunks", []],
    ["the longest matching name", "@LeBron James scores", "<@e_lebron_james> scores", "LeBron James scores", []],
    ["a shorter name when the longer one does not match", "@LeBron scores", "<@e_lebron> scores", "LeBron scores", []],
    ["a name followed by punctuation", "@LeBron_James, then", "<@e_lebron_james>, then", "LeBron James, then", []],
    ["a name with regex characters", "@a.j._green catches", "<@e_aj> catches", "A.J. Green catches", []],
    ["no name that only matches as a pattern", "@AxJx Green catches", "@AxJx Green catches", "@AxJx Green catches", ["@AxJx"]],
    ["no name continued by more letters", "@LeBronJames dunks", "@LeBronJames dunks", "@LeBronJames dunks", ["@LeBronJames"]],
    ["an unknown name", "@Kobe passes", "@Kobe passes", "@Kobe passes", ["@Kobe"]],
    ["no email address", "mail me@x.com", "mail me@x.com", "mail me@x.com", []],
    ["no resolved mention", "<@e_lebron> dunks", "<@e_lebron> dunks", "<@e_lebron> dunks", []],
  ])("resolveMentions resolves %s", (_, text, search, plain, unknown) => {
    expect(resolveMentions(text, entities)).toEqual({ search, plain, unknown });
  });
});
