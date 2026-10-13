"""timeline shows only the evidence inside the asked range: speech cut down to the words spoken in
it, the on-screen text of each sampled frame in reading order (a frame repeating the one before
it left out), and the sounds and captions that overlap the range."""

import json

from tlv import core, store


def test_timeline_keeps_the_evidence_inside_the_range(video):
    words = [{"w": " and", "s": 1.0, "e": 1.4}, {"w": " he", "s": 2.1, "e": 2.3}, {"w": " scores", "s": 2.3, "e": 2.9}]
    ocr = [
        (2.0, "LAKERS 102", 0.9),
        (2.0, "3:45", 0.1),
        (2.5, "LAKERS 102", 0.9),
        (2.5, "3:45", 0.1),
        (3.0, "LAKERS 105", 0.9),
        (5.5, "LAKERS 105", 0.9),
    ]
    spans = [
        ("s1", video, 0.0, 2.5, "a three", 0),
        ("s2", video, 2.5, 5.0, "the same three again", 1),
        ("s3", video, 4.0, 8.0, None, None),
        ("s4", video, 6.0, 8.0, "a timeout", 0),
    ]
    with store.transaction(store.connect("tlv")) as db:
        db.execute("INSERT INTO speech VALUES (?, 1.0, 2.9, 'and he scores', ?)", (video, json.dumps(words)))
        db.execute("INSERT INTO speech VALUES (?, 6.0, 7.0, 'later', '[]')", (video,))
        db.executemany("INSERT INTO ocr VALUES (?, ?, ?, 0, ?, 0.1, 0.05, 1)", [(video, t, s, y) for t, s, y in ocr])
        db.execute("INSERT INTO sounds VALUES (?, 1.5, 2.5, 'cheering', 0.8)", (video,))
        db.execute("INSERT INTO sounds VALUES (?, 5.5, 7.0, 'whistle', 0.7)", (video,))
        db.executemany(
            "INSERT INTO spans (span_id, video_id, start, end, caption, replay) VALUES (?, ?, ?, ?, ?, ?)", spans
        )

    timeline = core.video_timeline("tlv", video, 2.0, 5.0)
    assert timeline["speech"] == [{"start": 2.1, "end": 2.9, "text": "he scores"}]
    assert timeline["screen_text"] == [{"t": 2.0, "text": "3:45 | LAKERS 102"}, {"t": 3.0, "text": "LAKERS 105"}]
    assert [sound["label"] for sound in timeline["sounds"]] == ["cheering"]
    assert [(c["span_id"], c["replay"]) for c in timeline["captions"]] == [("s1", False), ("s2", True)]
