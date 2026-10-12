# tlv: local video understanding

tlv finds moments in videos and proves them. It runs on this Mac with open models; nothing
leaves the machine and nothing costs money. It replaces a Twelve Labs client with the same
commands and MCP tools, and it is built to beat it on the cases Twelve Labs gets wrong:
replays counted as new plays, times that drift by seconds, and confident claims nobody checked.

## Shape

```
video ─▶ index ──▶ store (SQLite + frames) ◀── entities
                      ▲
query ─▶ plan ─▶ retrieve ─▶ connect ─▶ inspect ─▶ answer (clips with sources)
```

- **Index** (`tlv/index/`): camera cuts, frames every 0.5 s with image embeddings, speech with
  word times, on-screen text, faces and bodies, audio events, then spans of a few seconds with
  a structured caption from the vision-language model. Each caption becomes an observation.
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
  normalized to unit length.
- **Models**: every model runs through `tlv.guard.ModelProcess`, one at a time machine-wide,
  under a memory cap (5 GB default). A model worker is a module under `tlv/models/` that calls
  `tlv.models.protocol.serve(load)`. Never import MLX, torch or onnxruntime in the parent
  process. Load a model once per batch of work, not once per item.
- **Media**: `tlv/media.py` owns ffmpeg and downloads.
- **Data**: `$TLV_HOME` (default `~/.local/share/tlv`).

## Contracts between modules

Signatures below are the interface; each owner may add private helpers freely. `db` is a
`sqlite3.Connection` from `store.connect()`; `progress` is `Callable[[str], None] | None`.

`tlv/index/pipeline.py`
- `add(source, title=None, wait=True, progress=None) -> AddResult` downloads or copies, probes,
  inserts the video, and indexes it when `wait`. Adding a file already present (same content)
  returns the existing video with `already_indexed`.
- `index(video_id, only=None, force=False, progress=None)` runs the stages in order, skipping
  any whose recorded version is current: shots, frames, speech, vision, sounds, spans,
  entities. The entities stage calls `tlv.entities.index_video`.
- `status(video_id) -> dict` with each stage's version and seconds, and row counts.

`tlv/embed.py` (each call loads its model once under the guard; batch your inputs)
- `images(paths) -> np.ndarray` unit image vectors (SigLIP 2), shape (n, d).
- `image_queries(texts) -> np.ndarray` SigLIP 2 text vectors in the same space as `images`.
- `sentences(texts, query=False) -> np.ndarray` unit text vectors for captions, speech and
  on-screen text; `query=True` applies the model's query instruction.

`tlv/index/vision.py`
- `read_text(paths) -> list[list[TextLine]]` Apple Vision text recognition on image files.
- `detect(paths) -> list[list[Detection]]` faces (with landmarks) and bodies on image files.

`tlv/entities.py`
- `index_video(db, video_id, progress)`: appearances for the video, then `recluster`.
- `recluster(db, progress)`: clusters the whole collection, honoring every correction.
- `entities(db) -> list[dict]`: entity_id, name, appearance and video counts, crop path.
- `resolve(db, text) -> list[str]`: entity ids a name or description in a query refers to
  ("LeBron James", "lebron", "@Coco", "number 23"), empty when unsure.
- `present(db, video_id, start, end) -> list[dict]`: who is visible in a range, with times.
- `name(db, target, name)`, `same(db, a, b)`, `different(db, a, b)`: corrections; `target`,
  `a`, `b` are entity or appearance ids.
- `add_reference(db, name, images=(), frames=()) -> dict`: teach a person from image files or
  (video_id, t) frames, for `video_entity_add`.

`tlv/agent/` (results are plain dicts shaped like the TypeScript `core.ts` results, so the
CLI and MCP output stays familiar)
- `search(query, then=(), limit=10, video_ids=None, max_gap=12) -> SearchResult`
- `find(query, then=(), max_candidates=8, pad=3, min_confidence=0.6, cut=True, video_ids=None,
  progress=None) -> FindResult`. Each confirmed observation adds `replays`: other angles of
  the same moment, `[{video_id, start, end, timecode, reason}]`, merged into it rather than
  listed as separate clips.
- `ask(video_id, question, start=None, end=None, json_schema=None) -> AskResult`
- `answer(question, video_ids=None, progress=None) -> JockeyResult`: a collection-wide
  question answered by planning, calling the same tools, and citing clips.

`tlv/cli.py`, `tlv/mcp_server.py`: the commands and MCP tools of the TypeScript tool, same
names and flags, plus entity corrections.
