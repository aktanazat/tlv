# tlv

Local video understanding for terminal agents. `tlv` indexes videos on this Mac with open
models, finds moments in them, checks each moment by looking at its frames, and cuts clips.
Nothing leaves the machine and nothing costs money. Every answer points back to a video and an
exact time range, with the frames, transcript and on-screen text behind it.

The same operations run as CLI commands and as MCP tools (`tlv mcp`). The commands and tool
names are those of the earlier Twelve Labs client, so agents that used it need no changes.

- `search` is fast: likely moments from captions, speech, on-screen text and frames. Nothing is
  checked.
- `find` is the verified clip finder: it searches, then looks at the frames of every candidate
  with their true times and confirms the claim, the person and the order of the steps. Replays of
  the same play come back as one clip with its other angles listed, not as separate results.
- `timeline` shows the evidence for a range: speech word by word, on-screen text, sounds,
  captions and who is visible.
- `person` follows one person across every video with no reference photo: each stretch they are
  on screen, their seconds on screen, where their name is said or shown, the moments `find`
  verified about them, and a filmstrip still of their clearest crops.
- `reel` cuts the clips of a find into one video on the beat of a song: each clip starts on a
  camera cut, lasts whole beats and opens with a caption saying what happens, who and when in the
  game, and a sources file beside it names each clip's video and time range.
- `jockey` answers a question about the whole collection in prose, citing clips.

How it works is in [DESIGN.md](DESIGN.md).

## Setup

Needs Python 3.12 with [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on `PATH`
(`brew install ffmpeg`). Apple Silicon: the models run on MLX.

```sh
uv sync
uv run tlv --help
```

Model weights download from Hugging Face on first use into `~/.cache/huggingface`. Models run one
at a time, each in a child process held to a memory cap, so indexing waits its turn on a busy Mac.

A background model host, one per user, keeps the models loaded between searches, so only the
first search after a pause pays to load them. It lets a model go after 10 minutes unused
(`TLV_MODEL_IDLE_S`) and holds at most 8 GB (`TLV_HELD_MEMORY_GB`). When
`~/memory/bin/host-headroom` shows the Mac short of memory, with 8 GB or more of swap in use
(`TLV_HOLD_MAX_SWAP_GB`) or less than 25% of memory free (`TLV_HOLD_MIN_FREE_PCT`), it lets go
of every model not in use, so searches load one model at a time and keep nothing between them.
After tlv's code changes, the next run loads a fresh copy of the model rather than reuse one
running the old code. The host exits once it holds nothing, and logs what it loads and lets go to
`~/.local/share/tlv/run/models.log`, whatever `$TLV_HOME` says.

Files live under `$TLV_HOME` (default `~/.local/share/tlv`): `tlv.db` is the index, `media/`
holds downloads, `clips/` cut clips, `frames/` stills and contact sheets. `--index NAME` (or
`TLV_INDEX`) selects another index, stored as `$TLV_HOME/NAME.db`.

## MCP

`tlv mcp` serves the tools over stdio. An MCP client entry:

```json
{
  "command": "/path/to/twelvelabs-tool/.venv/bin/tlv",
  "args": ["mcp"],
  "env": { "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" }
}
```

`PATH` must reach `ffmpeg` and `ffprobe`; MCP clients often start servers with a bare `PATH`.

Tools: `video_add`, `video_list`, `video_status`, `video_index`, `video_search`, `video_ask`,
`video_find`, `video_cut`, `video_frames`, `video_timeline`, `video_entity_add`, `video_entities`,
`video_entity_name`, `video_entity_same`, `video_entity_different`, `video_person`, `video_jockey`.
Results come back as structured content plus a short text summary; `video_cut`, `video_frames` and
`video_person` also return the frame, contact sheet or filmstrip as an image. Long calls send
progress notifications when the client passes a progress token.

`video_at` finds a game time on a video's scoreboard and returns its first still as an image;
`video_more` follows up on the last `video_find`, and `video_search` and `video_find` take a `clock`
window.

`video_reel` cuts the confirmed clips of a find, new or saved, into a reel on the beat of a song
and writes its sources beside it.

## Examples

```sh
# add a YouTube video (downloads at most 720p, then indexes it here; minutes per video)
tlv add "https://www.youtube.com/watch?v=VIDEO"
tlv add ~/Movies/game.mp4 --title "Lakers at Celtics" --no-wait   # returns now, indexes in the background
tlv index VIDEO_ID                 # finish or redo indexing; one stage with --stage S [--force]
tlv list                           # video ids name the content (v_ and 10 hex digits)
tlv status VIDEO_ID

# likely moments; an ordered action takes --then steps
tlv search "steals the ball" --then "dunks" --then "celebrates" --max-gap 8

# verified clips, replays merged
tlv find "lebron james scores" --then "celebrates"

# people: found automatically; name them, fix mistakes, then refer to them by name in queries
tlv entities
tlv entity-name ENTITY_ID "LeBron James"
tlv entity-same ENTITY_A ENTITY_B
tlv entity-different ENTITY_A ENTITY_B
tlv entity-add "LeBron James" --video VIDEO_ID --at 83.4 --at 412
tlv find "LeBron James steals the ball" --then "he dunks"

# one person across every video: on screen, named, verified; --strip adds a filmstrip still
tlv person @LeBron --strip

# a game time on the scoreboard: the live play and each replay, a still of each (--cut adds clips)
tlv at VIDEO_ID "4th 1:12"
tlv find "lebron james scores" --clock 4th          # check only where the scoreboard shows the 4th
tlv search "three pointer" --clock "last 2 min"

# follow up on the last find: a narrowing takes milliseconds and no model
tlv more "now only fourth quarter"
tlv more "without the replays"
tlv more "same but steph"                           # finds again with Stephen Curry instead

# a reel of a find's clips on the beat of a song, in game-clock order, with its sources beside it
tlv reel "stephen curry makes a three pointer" --music ~/Music/beat.m4a --order game
tlv reel curry-three --from eval/results/local-fast-2026-10-12.json --music ~/Music/beat.m4a --out curry.mp4

# look at a range, check it, cut it
tlv frames VIDEO_ID --from 4:15 --to 4:22
tlv timeline VIDEO_ID --from 4:15 --to 4:22
tlv cut VIDEO_ID --from 4:15 --to 4:22 --frame sheet

# ask about one video, or about all of them
tlv ask VIDEO_ID "Who wins the tip-off?" --from 0 --to 30
tlv jockey "Which plays end in a celebration?"
tlv jockey "Show me the last one again" --session SESSION_ID
```

`--json` prints the structured result instead of the summary. Times are seconds (`83.4`) or
timecodes (`1:23.4`, `00:01:23.4`). Progress goes to stderr.

## Notes

- `add` of a file already present (same content) returns the existing video.
- `add` takes any video ffmpeg reads. One without a sound track, such as a screen recording,
  indexes with no speech and no sounds; one in a codec Apple's decoders refuse (VP9, AV1) is read
  from an H.264 copy made once under `$TLV_HOME/work/`.
- `--modality` (visual, audio, transcription) limits which channels `search` matches against.
- A `jockey` session is kept on disk under `$TLV_HOME/work/sessions/`; passing its `session_id`
  answers the follow-up with the earlier questions and answers in view.
- People belong to an index. Corrections are stored against the appearances they name, so they
  survive re-clustering when new videos are added.
- `person` reads only what indexing and `find` stored, so it runs no model and answers in
  milliseconds. Clusters that carry the same name count as one person; its filmstrip is saved
  under `$TLV_HOME/frames/`.
- `find` prefers a missed clip over a wrong one: a candidate is confirmed only when the frames
  show the claim with enough confidence and every step in order.
- `--clock` and `at` read the scoreboard from the on-screen text indexing stored, so they run no
  model. `--clock` keeps `search` and `find` to the stretches where the scoreboard shows that game
  time, before anything is ranked or checked; a video whose scoreboard was never read is left out
  and named, and a window no video shows is an error.
- `find` keeps its result as the index's last under `$TLV_HOME/work/sessions/`. `more` narrows it
  by game clock, video, replays or clip id without a model or a new search, or finds again with
  another person in place of the one searched; each follow-up builds on the one before. The rules
  read the usual phrasings and the model reads other ways to name a person, a game time or a video.
- `reel --from` cuts a saved find (`find --json`, or an eval results file and a run name) and runs
  no model. `--order game` sorts the clips by the scoreboard each one shows, read from the
  on-screen text indexing stored; clips that show none go last. The song's tempo and first beat
  are found from the song itself; pass `--bpm` for a song without a steady beat. A song shorter
  than the reel loops on whole bars. Each clip renders on the hardware video encoder, three at a
  time, so a reel of eight clips takes seconds and leaves the CPU and GPU to searches. Without
  `--music` the clips keep their own sound; `--mute-video` plays the song alone. The mp4 and its
  `.sources.md` and `.sources.json` go to `--out`, or to `$TLV_HOME/clips/`.

## How it compares

Eleven labeled queries over three NBA highlight videos, 24 events in all (`eval/`), run through
`find --max-candidates 12` here (`eval/results/local-fast-2026-10-12.json`) and through the
Twelve Labs client it replaced:

| | Twelve Labs | tlv |
|---|---|---|
| events found | 15 of 24 | 18 of 24 |
| clips returned | 52 | 31 |
| wrong clips | 25 | 9 |
| extra clips of an event already found | 5 | 0 |
| precision (ambiguous clips left out) | 0.33 | 0.67 |
| seconds per query | 42 | 44 |
| cost | paid API | none |

tlv ran on an M4 Pro with 24 GB that other work kept busy. To run the eval again and score both:

```sh
uv run python eval/run.py
uv run python eval/score.py eval/results/local-<date>.json eval/results/twelvelabs-2026-10-12.json
```

## Development

```sh
uv run --group dev pytest -q
uv run --group dev ruff check . && uv run --group dev ruff format --check .
```

Tests that load a model are marked `slow` and skipped by default; `pytest -m slow` runs them.
