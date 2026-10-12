# tlv

Video search for terminal agents, on TwelveLabs. `tlv` adds videos to a TwelveLabs index,
finds moments in them, checks those moments with Pegasus, and cuts clips from the local
copies. The same operations run as CLI commands and as MCP tools (`tlv mcp`).

- `search` is fast and cheap: Marengo returns likely moments with timecodes. Nothing is checked.
- `find` is the verified clip finder: it searches, then Pegasus watches every candidate
  window and confirms the claim, the person and the order of the steps. It is slower and
  spends Pegasus minutes (each check bills the window length, at least 6 s).
- `jockey` asks Jockey, TwelveLabs' research-preview agent, a question about the whole
  collection. It answers in prose with citations; it does not cut clips.

## Setup

Needs [Bun](https://bun.sh), `ffmpeg`/`ffprobe` and `yt-dlp` on `PATH`
(`brew install ffmpeg yt-dlp`).

```sh
bun install
bun link            # puts `tlv` on PATH (~/.bun/bin)
```

The API key comes from `TWELVELABS_API_KEY`. Keep it in the vault and run through it:

```sh
mem-secret run TWELVELABS_API_KEY -- tlv list
```

Files live under `$TLV_HOME` (default `~/.local/share/tlv`): `media/` holds downloads,
`clips/` cut clips, `frames/` stills and contact sheets. The index is `tlv` unless
`TLV_INDEX` or `--index` names another; it is created on the first `add`.

## MCP

`tlv mcp` serves the tools over stdio. An MCP client entry:

```json
{
  "command": "mem-secret",
  "args": ["run", "TWELVELABS_API_KEY", "--", "/Users/you/.bun/bin/bun", "/path/to/twelvelabs-tool/src/cli.ts", "mcp"],
  "env": { "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" }
}
```

`PATH` must reach `ffmpeg`, `ffprobe` and `yt-dlp`; MCP clients often start servers with a
bare `PATH`.

Tools: `video_add`, `video_list`, `video_status`, `video_search`, `video_ask`, `video_find`,
`video_cut`, `video_frames`, `video_entity_add`, `video_entities`, `video_jockey`. Results come
back as structured content plus a short text summary; `video_cut` and `video_frames` also return
the frame or contact sheet as an image. Long calls send progress notifications when the client
passes a progress token.

## Examples

```sh
# add a YouTube video (downloads at most 720p, uploads, indexes, waits)
tlv add "https://www.youtube.com/watch?v=VIDEO"
tlv add ~/Movies/game.mp4 --title "Lakers at Celtics" --no-wait
tlv list
tlv status 6710c5b2e1

# likely moments; an ordered action takes --then steps
tlv search "steals the ball" --then "dunks" --then "celebrates" --max-gap 8

# teach a person, then refer to them with @Name (underscores or hyphens for spaces)
tlv entity-add "LeBron James" --video 6710c5b2e1 --at 83.4 --at 412
tlv find "@LeBron_James steals the ball" --then "he dunks"

# look at a range, cut a clip
tlv frames 6710c5b2e1 --from 1:20 --to 1:35
tlv cut 6710c5b2e1 --from 1:23.4 --to 1:31 --frame sheet

# ask Pegasus about one video, or Jockey about all of them
tlv ask 6710c5b2e1 "Who wins the tip-off?" --from 0 --to 30
tlv jockey "Which games end on a buzzer beater?"
tlv jockey "Show me the last one again" --session SESSION_ID
```

`--json` prints the structured result instead of the summary. Times are seconds (`83.4`) or
timecodes (`1:23.4`, `00:01:23.4`). Progress goes to stderr.

## Notes

- The TwelveLabs index is the video registry: `title`, `source` and `local_path` live in each
  video's metadata. Adding the same source again returns the existing video, and restores the
  local copy if it was deleted.
- `add` also puts each video in a Jockey knowledge store named after the index. A failure there
  is reported in the result and retried by the next `add` of that source.
- People live in the entity collection `tlv-people`, shared by all indexes. On the free plan only
  one entity collection fits, and the sample collection counts.
- `find` prefers a missed clip over a wrong one: a candidate is confirmed only when Pegasus
  reports a match with enough confidence and saw every step.

## Development

```sh
bun test
bunx tsc --noEmit
```
