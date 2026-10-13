"""One person across the collection, read from what indexing and find already stored: each stretch
they are on screen in each video with its clearest crop, the seconds that adds up to, where their
name is said or shown on screen, and the moments a find verified about them. No model runs and no
reference photo is needed: the person is a cluster the entities stage made. Each table is read
once, SQLite keeping only the rows that could name the person, so a timeline takes milliseconds
however long the videos are."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tlv import entities, media
from tlv.entities import clues as rules
from tlv.entities.cluster import FACE_QUALITY, HALF_STEP

# Sightings with at most this long unseen between them are one stretch on screen: a cut away and
# back, or a moment hidden behind another player.
PRESENCE_GAP_S = 2.0
# Text is read once a second; frames showing the name this close are one showing, so a missed
# read does not split it.
SHOWN_GAP_S = 2.0
# "LeBron's" names LeBron; the word match would read it as another word.
POSSESSIVE = re.compile(r"'s?(?![a-z0-9])", re.IGNORECASE)
# Rows with characters outside printable ASCII can name the person once accents fold ("Dončić"),
# which SQLite's LIKE cannot see, so they always reach the word match.
NON_ASCII = "*[^ -~]*"
# The filmstrip: each stretch's crop fitted into a tile, its time and video below, as many tiles to
# a row as fit media.SHEET_MAX_WIDTH.
TILE_W, TILE_H, LABEL_H = 120, 150, 30
GAP, MARGIN, TITLE_H, BAND_H = 4, 8, 30, 24
BACKGROUND, TILE, INK, MUTED = (17, 17, 17), (34, 34, 34), (235, 235, 235), (150, 150, 150)


@dataclass(frozen=True)
class Sighting:
    """One appearance as a timeline uses it: when the person is visible, padded by half a sampling
    step within the video as entities.present pads it, and how well its crop shows them."""

    video_id: str
    start: float
    end: float
    best_t: float
    crop_path: str
    sharpness: float  # a sharp face's capture quality; 0 for a body or a blurry face
    size: float  # the crop's share of the frame height


@dataclass(frozen=True)
class Presence:
    """One stretch on screen in one video."""

    video_id: str
    start: float
    end: float
    seen: float  # seconds visible inside the stretch, its unseen moments left out
    best_t: float  # when the crop that shows the person best was taken
    crop_path: str


@dataclass(frozen=True)
class Mention:
    """The person's name said (a word of the transcript) or shown (frames whose text has it)."""

    video_id: str
    start: float
    end: float
    kind: Literal["speech", "screen"]
    text: str


@dataclass(frozen=True)
class Moment:
    """Verified observations about the person that overlap in one video, as one moment: repeated
    finds verify the same play many times."""

    video_id: str
    start: float
    end: float
    claims: tuple[str, ...]
    obs_ids: tuple[str, ...]


@dataclass(frozen=True)
class Video:
    video_id: str
    title: str
    duration: float


@dataclass(frozen=True)
class Timeline:
    entity_ids: tuple[str, ...]  # biggest first
    name: str | None
    videos: tuple[Video, ...]  # the videos looked at, in added order
    presences: tuple[Presence, ...]  # in video then time order, as are the lists below
    mentions: tuple[Mention, ...]
    verified: tuple[Moment, ...]
    seconds: dict[str, float]  # seconds on screen per video that shows the person


def timeline(db: sqlite3.Connection, who: str, video_ids: Sequence[str] = ()) -> Timeline:
    """The timeline of the person `who` names ("@LeBron", "LeBron James", an entity id, "#23")
    in every video, or only in `video_ids`."""
    entity_ids, names = person(db, who)
    videos = _videos(db, video_ids)
    order = {video.video_id: i for i, video in enumerate(videos)}
    found = presences(sorted(_sightings(db, entity_ids, videos), key=lambda s: (order[s.video_id], s.start)))
    seconds: dict[str, float] = defaultdict(float)
    for stretch in found:
        seconds[stretch.video_id] += stretch.seen
    known = [row[0] for row in db.execute("SELECT DISTINCT name FROM entities WHERE name IS NOT NULL")]
    within = (f" AND video_id IN ({_marks(video_ids)})", list(video_ids)) if video_ids else ("", [])
    mentions = _said(db, names, known, within) + _shown(db, names, known, within) if names else []
    return Timeline(
        tuple(entity_ids),
        max(names, key=lambda n: len(rules.words(n)), default=None),
        tuple(videos),
        tuple(found),
        tuple(sorted(mentions, key=lambda m: (order[m.video_id], m.start, m.kind))),
        tuple(sorted(_verified(db, entity_ids, names, known, within), key=lambda m: (order[m.video_id], m.start))),
        dict(seconds),
    )


def person(db: sqlite3.Connection, who: str) -> tuple[list[str], list[str]]:
    """The entity ids of the person `who` names, biggest first, and the names they carry. Clusters
    that share a name are one person split in parts, so their timelines join; two different names
    are a question to ask back."""
    found = entities.resolve(db, who)
    if not found:
        known = sorted({row[0] for row in db.execute("SELECT name FROM entities WHERE name IS NOT NULL")})
        named = f"Named people: {', '.join(known)}." if known else "Nobody is named yet."
        raise LookupError(f'No person "{who}" in this index. {named} tlv entities lists everyone, named or not.')
    names = list(dict.fromkeys(row[0] for row in db.execute(_select_names(found), found) if row[0]))
    if not all(rules.same_person(a, b) for a in names for b in names):
        raise ValueError(f'"{who}" could be {" or ".join(names)}: give one full name or an entity id')
    return found, names


def presences(sightings: Iterable[Sighting], gap: float = PRESENCE_GAP_S) -> list[Presence]:
    """Sightings, in video then start order, as stretches on screen: a sighting that starts at most
    `gap` seconds after its stretch so far ends, in the same video, extends it. Each stretch keeps
    the crop that shows the person best: the sharpest face, else the biggest crop."""
    stretches = []
    for run in _runs(sightings, lambda s: (s.video_id, s.start, s.end), gap):
        seen, start, end = 0.0, run[0].start, run[0].end
        for s in run[1:]:
            if s.start > end:
                seen, start = seen + end - start, s.start
            end = max(end, s.end)
        best = max(run, key=lambda s: (s.sharpness, s.size))
        stretches.append(Presence(run[0].video_id, run[0].start, end, seen + end - start, best.best_t, best.crop_path))
    return stretches


def strip(found: Timeline, out: Path) -> Path:
    """Writes a filmstrip still of `found` to `out`: under a title line, one band per video in added
    order, each a header and a tile per stretch on screen, showing the stretch's crop and labelled
    with its time and video. The font Pillow bundles draws only ASCII, so accents fold and emoji
    drop."""
    small, large = ImageFont.load_default(size=12), ImageFont.load_default(size=18)
    bands = [(v, [p for p in found.presences if p.video_id == v.video_id]) for v in found.videos]
    bands = [(video, tiles) for video, tiles in bands if tiles]
    per_row = (media.SHEET_MAX_WIDTH - 2 * MARGIN + GAP) // (TILE_W + GAP)
    total, count = sum(found.seconds.values()), len(bands)
    plural = "" if count == 1 else "s"
    title = f"{_ascii(found.name or found.entity_ids[0])}: {_clock(total)} on screen in {count} video{plural}"
    headers = [f"{v.video_id}   {_clock(found.seconds[v.video_id])} on screen   {_ascii(v.title)}" for v, _ in bands]
    tiles_w = min(per_row, max(len(tiles) for _, tiles in bands)) * (TILE_W + GAP) - GAP
    texts_w = max(large.getlength(title), *(small.getlength(header) for header in headers))
    width = min(media.SHEET_MAX_WIDTH, 2 * MARGIN + max(tiles_w, math.ceil(texts_w)))
    rows = [-(-len(tiles) // per_row) for _, tiles in bands]
    height = 2 * MARGIN + TITLE_H + sum(BAND_H + n * (TILE_H + LABEL_H + GAP) for n in rows)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN, MARGIN), _fitted(title, large, width - 2 * MARGIN), font=large, fill=INK)
    y = MARGIN + TITLE_H
    for (video, tiles), header, n in zip(bands, headers, rows, strict=True):
        draw.text((MARGIN, y + 6), _fitted(header, small, width - 2 * MARGIN), font=small, fill=MUTED)
        y += BAND_H
        for i, stretch in enumerate(tiles):
            row, column = divmod(i, per_row)
            x0, y0 = MARGIN + column * (TILE_W + GAP), y + row * (TILE_H + LABEL_H + GAP)
            draw.rectangle((x0, y0, x0 + TILE_W - 1, y0 + TILE_H - 1), fill=TILE)
            with Image.open(stretch.crop_path) as crop:
                fitted = ImageOps.contain(crop.convert("RGB"), (TILE_W, TILE_H))
            image.paste(fitted, (x0 + (TILE_W - fitted.width) // 2, y0 + (TILE_H - fitted.height) // 2))
            draw.text((x0 + 2, y0 + TILE_H + 2), _when(stretch.start, stretch.end), font=small, fill=INK)
            draw.text((x0 + 2, y0 + TILE_H + 16), video.video_id, font=small, fill=MUTED)
        y += n * (TILE_H + LABEL_H + GAP)
    media.write_atomically(out, lambda partial: image.save(partial, "JPEG", quality=90))
    return out


def _videos(db: sqlite3.Connection, video_ids: Sequence[str]) -> list[Video]:
    rows = db.execute("SELECT video_id, title, duration FROM videos ORDER BY added_at, video_id").fetchall()
    videos = [Video(row["video_id"], row["title"], row["duration"]) for row in rows]
    known = {video.video_id for video in videos}
    missing = [video_id for video_id in video_ids if video_id not in known]
    if missing:
        raise LookupError(f'No video "{missing[0]}" in this index. video_list (tlv list) shows the ids.')
    return [video for video in videos if not video_ids or video.video_id in video_ids]


def _sightings(db: sqlite3.Connection, entity_ids: Sequence[str], videos: Sequence[Video]) -> list[Sighting]:
    """The person's appearances in `videos`; reference images, which have no video, drop out."""
    duration = {video.video_id: video.duration for video in videos}
    rows = db.execute(
        "SELECT a.video_id, a.kind, a.start, a.end, a.best_t, a.crop_path, a.quality, json_extract(a.box, '$[3]')"
        f" AS height FROM entity_members m JOIN appearances a USING (appearance_id) WHERE m.entity_id IN"
        f" ({_marks(entity_ids)})",
        list(entity_ids),
    )
    return [
        Sighting(
            row["video_id"],
            max(0.0, row["start"] - HALF_STEP),
            min(duration[row["video_id"]], row["end"] + HALF_STEP),
            row["best_t"],
            row["crop_path"],
            row["quality"] if row["kind"] == "face" and row["quality"] >= FACE_QUALITY else 0.0,
            row["height"],
        )
        for row in rows
        if row["video_id"] in duration
    ]


def _said(db: sqlite3.Connection, names: list[str], known: list[str], within: tuple[str, list[str]]) -> list[Mention]:
    """Each word of the transcripts that names the person, by the rule clue naming uses: the full
    name, or one word of it that no other known name has."""
    test, args = _naming("text", names)
    mentions = []
    for row in db.execute(f"SELECT video_id, text, words FROM speech WHERE {test}{within[0]}", args + within[1]):
        words = json.loads(row["words"])
        timed = [(POSSESSIVE.sub("", word["w"]), float(i)) for i, word in enumerate(words)]
        for spelled, i in rules.said(timed, known):
            if _is(spelled, names):
                word = words[int(i)]
                mentions.append(Mention(row["video_id"], word["s"], word["e"], "speech", row["text"].strip()))
    return mentions


def _shown(db: sqlite3.Connection, names: list[str], known: list[str], within: tuple[str, list[str]]) -> list[Mention]:
    """Each run of frames whose on-screen text names the person. A frame's lines holding a word of a
    name read as one text, so a name split over two lines ("LEBRON", "JAMES") still reads whole."""
    test, args = _naming("text", names)
    parts = {word for name in names for word in rules.words(name)}
    frames: dict[tuple[str, float], list[str]] = defaultdict(list)
    sql = f"SELECT video_id, t, text FROM ocr WHERE {test}{within[0]} ORDER BY video_id, t, y, x"
    for row in db.execute(sql, args + within[1]):
        if parts.intersection(rules.words(POSSESSIVE.sub("", row["text"]))):
            frames[row["video_id"], row["t"]].append(row["text"])
    showing = [
        (video_id, t, " | ".join(lines))
        for (video_id, t), lines in frames.items()
        if _names(" | ".join(lines), names, known)
    ]
    return [
        Mention(run[0][0], run[0][1], run[-1][1], "screen", Counter(text for *_, text in run).most_common(1)[0][0])
        for run in _runs(showing, lambda f: (f[0], f[1], f[1]), SHOWN_GAP_S)
    ]


def _verified(
    db: sqlite3.Connection, entity_ids: list[str], names: list[str], known: list[str], within: tuple[str, list[str]]
) -> list[Moment]:
    """The verified observations about the person, merged where they overlap: those a find tied to
    one of the person's clusters, and those whose claim or "who" names them."""
    who = "json_extract(evidence, '$.who')"
    tests, args = [f"entity_id IN ({_marks(entity_ids)})"], list(entity_ids)
    for column in ("text", who) if names else ():
        test, column_args = _naming(column, names)
        tests.append(test)
        args += column_args
    rows = db.execute(
        f"SELECT obs_id, video_id, start, end, entity_id, text, {who} AS who FROM observations"
        f" WHERE kind = 'verified'{within[0]} AND ({' OR '.join(tests)}) ORDER BY video_id, start, obs_id",
        within[1] + args,
    )
    about = [
        row
        for row in rows
        if row["entity_id"] in entity_ids or _names(row["text"], names, known) or _names(row["who"] or "", names, known)
    ]
    return [
        Moment(
            run[0]["video_id"],
            run[0]["start"],
            max(row["end"] for row in run),
            tuple(dict.fromkeys(row["text"] for row in run)),
            tuple(row["obs_id"] for row in run),
        )
        for run in _runs(about, lambda r: (r["video_id"], r["start"], r["end"]), 0.0)
    ]


def _runs[T](items: Iterable[T], span: Callable[[T], tuple[str, float, float]], gap: float) -> list[list[T]]:
    """`items`, in video then start order, grouped into runs: an item that starts at most `gap`
    seconds after its run so far ends, in the same video, joins it."""
    runs: list[list[T]] = []
    video, end = "", 0.0
    for item in items:
        here, start, stop = span(item)
        if runs and here == video and start - end <= gap:
            runs[-1].append(item)
            end = max(end, stop)
        else:
            runs.append([item])
            video, end = here, stop
    return runs


def _names(text: str, names: Sequence[str], known: Sequence[str]) -> bool:
    """Whether `text` names the person, by the transcript rule."""
    return any(_is(spelled, names) for spelled in rules.mentions(POSSESSIVE.sub("", text), known))


def _is(spelled: str, names: Sequence[str]) -> bool:
    """Whether a name found in text is one of the person's names, spelled alike."""
    return any(rules.words(spelled) == rules.words(name) for name in names)


def _naming(column: str, names: Sequence[str]) -> tuple[str, list[str]]:
    """An SQL test keeping the rows whose `column` could name the person: it holds a word of a name
    (LIKE ignores ASCII case), or characters only the word match can fold."""
    words = sorted({word for name in names for word in rules.words(name)})
    test = " OR ".join([f"{column} LIKE ?"] * len(words) + [f"{column} GLOB ?"])
    return f"({test})", [f"%{word}%" for word in words] + [NON_ASCII]


def _select_names(entity_ids: Sequence[str]) -> str:
    return f"SELECT name FROM entities WHERE entity_id IN ({_marks(entity_ids)}) ORDER BY entity_id"


def _marks(values: Sequence[str]) -> str:
    return ", ".join("?" * len(values))


def _ascii(text: str) -> str:
    """`text` in the characters the bundled font has: accents folded ("Dončić" -> "Doncic"), the
    rest (emoji) dropped."""
    return " ".join(unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().split())


def _fitted(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: float) -> str:
    """`text`, cut short with "..." to fit `width` pixels."""
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(f"{text}...") > width:
        text = text[:-1]
    return f"{text.rstrip()}..."


def _clock(seconds: float) -> str:
    """m:ss, or h:mm:ss from an hour: short enough for a tile."""
    minutes, s = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{s:02d}" if hours else f"{minutes}:{s:02d}"


def _when(start: float, end: float) -> str:
    return _clock(start) if _clock(start) == _clock(end) else f"{_clock(start)}-{_clock(end)}"
