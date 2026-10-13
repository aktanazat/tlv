"""Transcription in pieces: a long sound track goes to the model in jobs of at most two minutes,
so a search waits for one piece, and the joined transcript holds every word once, in order, at
its time, across seams in a pause and seams in unbroken sound."""

import json
import subprocess

import pytest

from tlv import guard
from tlv.index import speech

SECONDS = 300
# The longest job a search may wait behind: two minutes of audio.
LIMIT_S = 120.0
# The fake hears every other piece this much late, as Whisper's word times differ a little
# between two pieces hearing the same word.
JITTER_S = 0.15
# Like Whisper, the fake misses words this close to a piece's cut edges, even past a pause.
EDGE_S = 1.0
# A pause of SILENT_S starts every PAUSE_EVERY_S seconds in the "pauses" track.
PAUSE_EVERY_S = 45
SILENT_S = 1.0
TRACKS = {
    "pauses": f"aevalsrc=0.3*sin(2*PI*440*t)*gte(mod(t\\,{PAUSE_EVERY_S})\\,{SILENT_S}):d={SECONDS}:s=16000",
    "no pauses": f"anoisesrc=d={SECONDS}:a=0.2:r=16000:seed=1",
}


def spoken(track):
    """A word every 0.3 s, none inside a pause: (word, start, end)."""
    words = []
    for k in range(int(SECONDS / 0.3) - 1):
        start = round(0.3 * k + 0.02, 3)
        end = round(start + 0.25, 3)
        if track == "pauses" and (start % PAUSE_EVERY_S < SILENT_S + 0.1 or end % PAUSE_EVERY_S < SILENT_S + 0.1):
            continue
        words.append((f"word{k}", start, end))
    return words


class FakeWhisper:
    """Hears the words wholly inside a job's stretch of audio and clear of its cut edges: a word
    the stretch cuts, or one within EDGE_S of where it was cut from the track, is not heard."""

    def __init__(self, words):
        self.words = words
        self.jobs = []

    def __call__(self, module, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def call(self, job):
        start, end = job.get("start", 0.0), job.get("end", float(SECONDS))
        self.jobs.append((start, end))
        late = JITTER_S * (len(self.jobs) % 2)
        first = start + EDGE_S if start > 0 else start
        last = end - EDGE_S if end < SECONDS - EDGE_S else end
        heard = [{"w": w, "s": s + late, "e": e + late} for w, s, e in self.words if first <= s and e <= last]
        segments = [
            {
                "start": group[0]["s"],
                "end": group[-1]["e"],
                "text": " ".join(word["w"] for word in group),
                "words": group,
            }
            for group in (heard[i : i + 10] for i in range(0, len(heard), 10))
        ]
        return {"language": "en", "segments": segments}


@pytest.fixture(params=TRACKS, ids=list(TRACKS))
def transcribed(request, db, video, monkeypatch):
    """The fake model and the speech rows after transcribing a five-minute track."""
    lavfi = ["-f", "lavfi", "-i", TRACKS[request.param]]
    subprocess.run(["ffmpeg", "-loglevel", "error", *lavfi, "-c:a", "aac", video["local_path"]], check=True)
    whisper = FakeWhisper(spoken(request.param))
    monkeypatch.setattr(guard, "ModelProcess", whisper)
    speech.run(db, video, None)
    words = [
        (word["w"], word["s"])
        for row in db.execute("SELECT words FROM speech WHERE video_id = 'v_test' ORDER BY start")
        for word in json.loads(row["words"])
    ]
    return whisper, words


def test_a_long_track_goes_to_the_model_in_pieces_of_at_most_two_minutes(transcribed):
    whisper, _ = transcribed
    assert len(whisper.jobs) > 1
    assert max(end - start for start, end in whisper.jobs) <= LIMIT_S


def test_the_joined_transcript_holds_every_word_once_in_order_at_its_time(transcribed):
    whisper, words = transcribed
    assert [w for w, _ in words] == [w for w, _, _ in whisper.words]
    assert all(abs(s - said) <= JITTER_S + 1e-6 for (_, s), (_, said, _) in zip(words, whisper.words, strict=True))
