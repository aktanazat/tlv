"""The query side: plan a query into steps, retrieve candidates from every channel of the index,
connect steps in time order, inspect each candidate by looking at its frames, merge replays of
one play, and answer only with what was looked at, each claim tied to a video and a time range."""

from tlv.agent.answer import answer
from tlv.agent.ask import ask
from tlv.agent.find import find, search

__all__ = ["answer", "ask", "find", "search"]
