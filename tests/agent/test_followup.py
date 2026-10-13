"""Follow-ups to the last find: the rules read the usual phrasings into clips to keep or a person to
find instead and leave the rest to the model, a swap renames the person wherever the search named
them, a narrow keeps the confirmed clips inside it, and the model's reading counts only where each
part names something that exists. The model test loads the vision-language model: run with
-m slow."""

import pytest

from tlv.agent import followup
from tlv.agent.clock import Window
from tlv.agent.followup import Followup, Narrow, Swap
from tlv.agent.model import Vlm

NAMES = ["LeBron James", "Stephen Curry"]
TITLES = {"v_lal": "WARRIORS at LAKERS | FULL GAME HIGHLIGHTS", "v_phi": "LeBron James 29 pts vs 76ers"}
STEPS = ["LeBron James dunks", "he celebrates"]
LEBRON = ("LeBron James", "LeBron", "James")
CURRY = Swap(LEBRON, "Stephen Curry")
FOURTH = Window(frozenset({"4"}))
SECOND_HALF = Window(frozenset({"3", "4"}))


@pytest.mark.parametrize(
    ("instruction", "change"),
    [
        ("now only fourth quarter", Followup(Narrow(clock=FOURTH))),
        ("without the replays", Followup(Narrow(no_replays=True))),
        ("live plays only", Followup(Narrow(no_replays=True))),
        ("no replays, 4th quarter only", Followup(Narrow(clock=FOURTH, no_replays=True))),
        ("only the warriors game", Followup(Narrow(video_ids=frozenset({"v_lal"})))),
        ("just v_phi", Followup(Narrow(video_ids=frozenset({"v_phi"})))),
        ("drop obs_1a2b and obs_3c4d", Followup(Narrow(drop=frozenset({"obs_1a2b", "obs_3c4d"})))),
        ("same but steph", Followup(Narrow(), CURRY)),
        ("what about Curry in the 4th", Followup(Narrow(clock=FOURTH), CURRY)),
        ("replace lebron with steph", Followup(Narrow(), CURRY)),
        ("steph instead of lebron", Followup(Narrow(), CURRY)),
        ("instead of James, Stephen Curry", Followup(Narrow(), CURRY)),
        ("make it purple", None),
        ("without the replays, and make it purple", None),
        ("only the celtics and warriors game", None),
        ("same but klay", None),
        ("drop the second one", None),
        ("now", None),
    ],
)
def test_the_rules_read_the_usual_follow_ups_and_leave_the_rest_to_the_model(instruction, change):
    assert followup.parse(instruction, STEPS, NAMES, TITLES) == change


@pytest.mark.parametrize(
    ("instruction", "steps"),
    [("same but steph", ["a dunk"]), ("replace klay with steph", STEPS)],
)
def test_a_swap_needs_the_search_to_name_the_person_it_replaces(instruction, steps):
    assert followup.parse(instruction, steps, NAMES, TITLES) is None


@pytest.mark.parametrize("instruction", ["same but steph", "replace james with steph"])
def test_a_name_sharing_a_word_with_the_searched_name_is_not_the_one_replaced(instruction):
    names = ["James Harden", *NAMES]
    assert followup.parse(instruction, STEPS, names, TITLES) == Followup(Narrow(), CURRY)


@pytest.mark.parametrize(
    ("steps", "swap", "renamed"),
    [
        (
            ["LeBron James dunks", "LeBron celebrates", "james runs back"],
            CURRY,
            ["Stephen Curry dunks", "Stephen Curry celebrates", "Stephen Curry runs back"],
        ),
        (["LeBron James dunks"], Swap(LEBRON, "James Harden"), ["James Harden dunks"]),
        (["Jameson scores"], CURRY, ["Jameson scores"]),
    ],
)
def test_a_swap_renames_the_person_wherever_and_however_the_search_named_them(steps, swap, renamed):
    assert followup.swapped(steps, swap) == renamed


def clip(obs_id, video_id, start, end):
    return {
        "id": obs_id,
        "video_id": video_id,
        "start": start,
        "end": end,
        "replays": [{"video_id": video_id, "start": end + 30, "end": end + 36}],
    }


FOUND = {
    "claim": "LeBron James dunks",
    "confirmed": [clip("obs_a", "v1", 10, 20), clip("obs_b", "v1", 100, 110), clip("obs_c", "v2", 5, 15)],
    "rejected": [{"video_id": "v1", "start": 50, "end": 60, "reason": "no dunk"}],
    "counts": {"candidates": 4, "checked": 4, "confirmed": 3, "rejected": 1},
}


@pytest.mark.parametrize(
    ("narrow", "ranges", "replayed", "kept"),
    [
        pytest.param(
            Narrow(clock=FOURTH),
            {"v1": [(90.0, 200.0)], "v2": [(15.0, 30.0)]},
            set(),
            ["obs_b"],
            id="inside the clock ranges, a clip ending where they begin left out",
        ),
        pytest.param(Narrow(video_ids=frozenset({"v2"})), {}, set(), ["obs_c"], id="in the videos"),
        pytest.param(Narrow(no_replays=True), {}, {"obs_a"}, ["obs_b", "obs_c"], id="not mostly replay"),
        pytest.param(Narrow(drop=frozenset({"obs_b"})), {}, set(), ["obs_a", "obs_c"], id="not dropped"),
        pytest.param(Narrow(), {"v1": []}, {"obs_a"}, ["obs_a", "obs_b", "obs_c"], id="only what the narrow asks"),
    ],
)
def test_a_narrow_keeps_the_confirmed_clips_inside_it_and_nothing_else_changes(narrow, ranges, replayed, kept):
    narrowed = followup.narrowed(FOUND, narrow, ranges, replayed)
    assert [c["id"] for c in narrowed["confirmed"]] == kept
    assert narrowed["counts"] == {**FOUND["counts"], "confirmed": len(kept)}
    assert narrowed["rejected"] == FOUND["rejected"]


@pytest.mark.parametrize(("no_replays", "replays"), [(True, [0, 0, 0]), (False, [1, 1, 1])])
def test_without_the_replays_the_kept_clips_lose_the_replays_merged_into_them(no_replays, replays):
    narrowed = followup.narrowed(FOUND, Narrow(no_replays=no_replays), {}, set())
    assert [len(c["replays"]) for c in narrowed["confirmed"]] == replays


def test_dropping_a_clip_the_find_did_not_confirm_is_an_error():
    with pytest.raises(ValueError, match="no clip obs_zz in the last find"):
        followup.narrowed(FOUND, Narrow(drop=frozenset({"obs_b", "obs_zz"})), {}, set())


@pytest.mark.parametrize(
    ("answer", "change"),
    [
        ({"replace": "LeBron", "with": "Steph", "clock": None}, Followup(Narrow(), CURRY)),
        ({"with": "Stephen Curry", "clock": "the 4th quarter"}, Followup(Narrow(FOURTH), CURRY)),
        ({"video": "Warriors at Lakers"}, Followup(Narrow(video_ids=frozenset({"v_lal"})))),
        ({"with": "Klay Thompson"}, Followup(Narrow(), Swap(LEBRON, "Klay Thompson"))),
        ({"clock": "purple"}, None),
        ({"video": "the Celtics game"}, None),
        ({"replace": "Kobe", "with": "Steph"}, None),
        ({"clock": None, "video": None, "replace": None, "with": None}, None),
        (["4th"], None),
        (None, None),
    ],
)
def test_the_models_reading_counts_only_where_each_part_names_something_that_exists(answer, change):
    assert followup.read_answer(answer, STEPS, NAMES, TITLES) == change


# Answers as the model wrote them (Qwen3-VL-4B, asked with the prompt that still had a no_replays
# part): every part filled in, "null" written as a word, the searched person named as the one to
# replace, and the replays left out whatever the change said.
@pytest.mark.parametrize(
    ("answer", "change"),
    [
        (  # "could you redo that search but with stephen curry in place of lebron"
            {"replace": "LeBron James", "with": "Stephen Curry", "clock": None, "video": None, "no_replays": True},
            Followup(Narrow(), CURRY),
        ),
        (  # "just the ones after halftime"
            {"replace": "lebron james", "with": "null", "clock": "2nd half", "video": "null", "no_replays": True},
            Followup(Narrow(SECOND_HALF)),
        ),
        (  # "make it purple"
            {"replace": "lebron james", "with": "null", "clock": "null", "video": "null", "no_replays": True},
            None,
        ),
    ],
)
def test_the_model_asks_for_no_more_than_the_parts_it_names(answer, change):
    assert followup.read_answer(answer, STEPS, NAMES, TITLES) == change


@pytest.mark.slow
def test_the_model_reads_a_swap_the_rules_cannot():
    instruction = "could you redo that search but with stephen curry in place of lebron"
    assert followup.parse(instruction, STEPS, NAMES, TITLES) is None
    with Vlm() as vlm:
        assert followup.with_model(vlm, instruction, STEPS, NAMES, TITLES) == Followup(Narrow(), CURRY)
