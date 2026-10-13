"""A search's query vectors from one model process are the ones each model gives on its own,
so they land in the spaces the index was built in. Loads both models: run with -m slow."""

import numpy as np
import pytest

from tlv import embed

TEXTS = ["LeBron James scores, then celebrates", "the crowd cheers after a dunk", "a three from the corner"]


@pytest.mark.slow
def test_queries_from_one_process_match_each_model_alone():
    looks, means = embed.queries(TEXTS)

    np.testing.assert_allclose(looks, embed.image_queries(TEXTS), atol=1e-3)
    np.testing.assert_allclose(means, embed.sentences(TEXTS, query=True), atol=1e-3)
