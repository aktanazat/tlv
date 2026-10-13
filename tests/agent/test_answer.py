"""An answer keeps only what its checked clips show."""

import pytest

from tlv.agent.answer import Kept, keep_supported


@pytest.mark.parametrize(
    ("draft", "held", "expected"),
    [
        pytest.param(
            "LeBron scores at 4:15 [3]. He flexes at 4:55 [1].",
            {1, 3},
            Kept("LeBron scores at 4:15 [1]. He flexes at 4:55 [2].", [3, 1], 0),
            id="sources renumbered by first citation",
        ),
        pytest.param(
            "LeBron scores at 4:15 [1]. He dunks at 6:00 [2]. The crowd cheers.",
            {1},
            Kept("LeBron scores at 4:15 [1]. The crowd cheers.", [1], 1),
            id="a sentence whose only source failed is left out",
        ),
        pytest.param(
            "LeBron scores twice [2][4].",
            {4},
            Kept("LeBron scores twice [1].", [4], 0),
            id="a sentence keeps the sources that held",
        ),
        pytest.param(
            "He scores [9]. Nothing else happens.",
            set(),
            Kept("Nothing else happens.", [], 1),
            id="a source never checked counts as failed",
        ),
        pytest.param(
            "He scores [1] and celebrates [2].",
            {2},
            Kept("He scores and celebrates [1].", [2], 0),
            id="a failed marker inside a kept sentence is taken out",
        ),
    ],
)
def test_an_answer_keeps_only_statements_its_clips_show(draft, held, expected):
    assert keep_supported(draft, held) == expected
