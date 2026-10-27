# tlv: local video understanding

tlv finds moments in videos and checks them against the frames. Its models run on this Mac.
If `TYPESAFE_API_KEY` is present, a confirmed play's claim and nearby transcript lines go to
TypeSafe for an additional credit check; video and frames stay here. Without that key, searching
stays local. It replaces a Twelve Labs client with the same commands and MCP tools, and aims to
address cases the earlier client got wrong: replays counted as new plays, times that drift by
seconds, and confident claims nobody checked.

## Shape

```
video ─▶ index ──▶ store (SQLite + frames) ◀── entities
                      ▲
query ─▶ plan ─▶ retrieve ─▶ connect ─▶ inspect ─▶ answer (clips with sources)
```

- **Index** (`tlv/index/`): camera cuts, frames every 0.5 s with image embeddings, speech with
  word times, on-screen text, faces and bodies, audio events, then spans of a few seconds with
  a structured caption from the vision-language model, which sees up to four of the span's
  frames; a frame repeating the picture and on-screen text shown before it is left out, so a
  still slide is shown once. Each caption becomes an observation.
  Speech is transcribed in pieces of one to two minutes of audio, one model job each, so a
  search waits behind one piece, never a whole video. Two pieces share 6 s around a seam placed
  in the quietest half second (a pause when there is one) and are joined at a word both heard
  at the same time, so no word is lost, doubled or cut at a seam. Times stay in the video's time.
- **Entities** (`tlv/entities.py`): faces and bodies grouped into one person across videos.
  Embed, retrieve a candidate pool, verify gray cases together with the vision-language model,
  cluster conservatively, then name from clues (spoken names, on-screen names, jersey numbers).
  A missed merge is preferred to a false merge. Corrections ("that's Coco") are stored against
  appearances and survive re-clustering.
- **Agent** (`tlv/agent/`): plans a query into ordered steps, retrieves candidates from every
  channel, connects steps in time order, inspects each candidate by looking at frames with
  their true timestamps, merges replays of the same play into one result, and answers only
  with confirmed clips, each with its video, time range and evidence.

## Rules every module follows

- **Store**: `tlv/store.py` owns the schema. Each table names the module that writes it. Times
  are seconds from the start of the video. Embeddings are float16 blobs (`store.to_blob`),
  normalized to unit length. `store.transaction` inside another transaction is a savepoint of
  it, so a writer's own transaction can run as one step of a larger one.
- **Models**: every model runs through `tlv.guard.ModelProcess`, which hands the work to the
  model host (`tlv/host.py`): one background process per user, shared by every `$TLV_HOME`,
  that runs one job at a time. A job someone is waiting on (a search, find, ask or follow-up)
  runs before any waiting index job; `tlv.index.pipeline.index` runs inside `guard.background()`,
  which marks every job it sends, stage threads included. After someone's job, index jobs also
  wait for that caller's next job, until it closes or 2 s pass, so a search's load and two
  embeddings run back to back and it waits at most for the one index job already running.
  Among jobs of one kind, a waiting job on a loaded model goes first (passing over another job
  at most three times). The host keeps
  models loaded between runs and lets them go when idle, over its held-memory cap or when the
  Mac is short of memory. Each model is held to its own memory cap (6 GB default).
  A model worker is a module under `tlv/models/` that calls `tlv.models.protocol.serve(load)`.
  Never import MLX, torch or onnxruntime in the parent process. Load a model once per batch of
  work, not once per item.
- **Media**: `tlv/media.py` owns ffmpeg and downloads.
- **Data**: `$TLV_HOME` (default `~/.local/share/tlv`).

## Contracts between modules

Signatures below are the interface; each owner may add private helpers freely. `db` is a
`sqlite3.Connection` from `store.connect()`; `progress` is `Callable[[str], None] | None`.

`tlv/index/pipeline.py`
- `add(source, title=None, wait=True, progress=None) -> AddResult` downloads or copies, probes,
  inserts the video, and indexes it when `wait`. Adding a file already present (same content)
  returns the existing video with `already_indexed`.
- `index(video_id, only=None, force=False, progress=None)` runs the stages, skipping any whose
  recorded version is current, each as soon as the stages it reads are done: shots, speech,
  vision and sounds start together, frames follows shots, spans waits for all five, and entities
  for spans. Vision reads the frame grid from the duration, not from frames. The entities stage
  calls `tlv.entities.index_video`.
- `status(video_id) -> dict` with each stage's version and seconds, and row counts.
- `remove(video_id, progress=None) -> RemoveResult` deletes the video's rows in one transaction
  (the tables naming a video cascade from `videos`), inside `tlv.entities.leaving`, then its
  files under `$TLV_HOME`: the media copy unless an index lists it for another video, and its
  frames, crops and work folders unless another index holds the same video.

`tlv/embed.py` (each call loads its model once under the guard; batch your inputs)
- `images(paths) -> np.ndarray` unit image vectors (SigLIP 2), shape (n, d).
- `image_queries(texts) -> np.ndarray` SigLIP 2 text vectors in the same space as `images`.
- `sentences(texts, query=False) -> np.ndarray` unit text vectors for captions, speech and
  on-screen text; `query=True` applies the model's query instruction.
- `queries(texts) -> (np.ndarray, np.ndarray)` `image_queries(texts)` and `sentences(texts, query=True)`
  from one child holding both models, so a search starts one model process, not two.

`tlv/index/vision.py`
- `read_text(paths) -> list[list[TextLine]]` Apple Vision text recognition on image files.
- `detect(paths) -> list[list[Detection]]` faces (with landmarks) and bodies on image files.

`tlv/entities.py`
- `index_video(db, video_id, progress)`: appearances for the video, then `recluster`.
- `recluster(db, progress)`: clusters the whole collection, honoring every correction.
- `leaving(db, video_id)`: a context manager for removing a video. Corrections given through its
  appearances move to the same person's best remaining appearance (or go, when no one is left),
  the caller deletes the video inside it, and everyone left is regrouped without the model.
- `entities(db) -> list[dict]`: entity_id, name, appearance and video counts, crop path.
- `resolve(db, text) -> list[str]`: entity ids a name or description in a query refers to
  ("LeBron James", "lebron", "@Coco", "number 23"), empty when unsure.
- `present(db, video_id, start, end) -> list[dict]`: who is visible in a range, with times.
- `sight_only(db, entity_ids) -> dict[str, str]`: the named people among them whose name no clue
  gives where the index sees them (the owner named them), with their names. No channel of words
  finds them, so retrieval puts first the spans showing them, and the check tells the model which
  of its frames show them and where; the model alone judges what happens.
- `sightings(db, video_id, entity_id, start, end) -> list[tuple]`: each appearance of a person in
  a range: when it is seen, face or body, and its box.
- `name(db, target, name)`, `same(db, a, b)`, `different(db, a, b)`: corrections; `target`,
  `a`, `b` are entity or appearance ids.
- `add_reference(db, name, images=(), frames=()) -> dict`: teach a person from image files or
  (video_id, t) frames, for `video_entity_add`.

`tlv/entities/timeline.py` (reads the index only; no model runs)
- `timeline(db, who, video_ids=()) -> Timeline`: one person across the videos, for `tlv person`
  and `video_person`: each stretch on screen with its clearest crop and the seconds seen, where
  the name is said or shown by the rule clue naming uses, and the verified observations tied to
  or naming the person, merged where they overlap. Clusters that share a name join; a query
  naming two different people raises.
- `presences(sightings, gap=2.0) -> list[Presence]`: sightings at most `gap` seconds apart, in
  one video, as one stretch.
- `strip(timeline, out) -> Path`: a filmstrip still, one tile per stretch, a band per video.

`tlv/agent/` (results are plain dicts shaped like those of the Twelve Labs client it replaced,
so the CLI and MCP output stays familiar)
- `search(query, then=(), limit=10, video_ids=None, max_gap=12) -> SearchResult`
- `find(query, then=(), max_candidates=8, pad=3, min_confidence=0.6, cut=True, video_ids=None,
  progress=None) -> FindResult`. Pieces of one moment found by neighbouring looks, at most 2 s
  apart, come back as one clip; for a question about what was said, so do mentions in one
  stretch of talk (no pause over 2.5 s, each within 30 s of the last). Each confirmed
  observation adds `replays`: other angles of the same moment,
  `[{video_id, start, end, timecode, reason}]`, merged into it rather than listed as separate
  clips.
- `ask(video_id, question, start=None, end=None, json_schema=None) -> AskResult`
- `answer(question, video_ids=None, progress=None) -> JockeyResult`: a collection-wide
  question answered by planning, calling the same tools, and citing clips.

`tlv/agent/clock.py` (reads the on-screen text indexing stored; no model runs)
- `Window.parse(text) -> Window`: a window of game time ("4th", "Q4 2:00-0:00", "2nd half", "OT",
  "last 2 min"); `str(window)` prints it back in words `parse` reads.
- `readings(db, video_id) -> list[Reading]`: the scoreboard in each frame showing one period and
  one time; a time under 5 s beside another time is the shot clock and is left out.
- `showings(readings, window) -> list[Showing]`: one per run of frames inside the window, the
  live play and each replay that keeps the scoreboard on screen, for `tlv at` and `video_at`.
- `coverage(db, videos, window) -> Coverage`: each video's stretches inside the window, the
  `within` ranges `find --clock` and `search --clock` pass on, and the videos never read.

`tlv/agent/followup.py`
- `parse(instruction, steps, names, titles) -> Followup | None`: rules read a follow-up to the
  last find into a `Narrow` (game clock window, videos, no replays, dropped clip ids) and a
  `Swap` (a person to find instead); None leaves it to `with_model(vlm, ...)`, whose answer
  counts only where each part names something that exists. It holds a window, a video and a
  swap, never replays: the model says to leave those out for nearly any change.
- `narrowed(result, narrow, ranges, replayed) -> dict`: the find result keeping only the
  confirmed clips inside the narrow; `swapped(steps, swap) -> list[str]` for the rerun.
- The last find of each index is kept in `$TLV_HOME/work/sessions/find-<index>.json` with what
  was asked, so `tlv more` and `video_more` follow up on it and each follow-up on the one before.

`tlv/reel.py` (reads the index and a find result; no model runs)
- `beats(song, bpm=None) -> Beats`: the song's tempo, from the lag at which its onset strength
  repeats, and its first beat; a song without a steady beat raises, so the caller passes `bpm`.
- `plan(moments, videos, grid, order) -> list[Cut]`: each clip a second either side of its
  moment, its edges moved onto camera cuts within 0.5 s, lasting whole beats, in found,
  game-clock or video order. A moment's game clock is the scoreboard the index read nearest its
  middle in the frames its clip shows.
- `make(db, claim, confirmed, rows, song, out, order, bpm, mute_video, progress) -> dict`:
  renders each clip in its own ffmpeg process on the hardware encoder, three at a time and to an
  exact frame count, joins them without re-encoding, lays the song from its first beat over the
  game's sound, and writes `.sources.md` and `.sources.json` beside the mp4.

`tlv/cli.py`, `tlv/mcp_server.py`: the commands and MCP tools of the Twelve Labs client, same
names and flags, plus entity corrections.
