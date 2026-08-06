"""Structured scoring.

This is the part that makes the project a system rather than a wrapper.

Two design decisions worth defending in an interview about this project:

1. Scoring runs OFF the voice loop. It is fired as a background task the
   moment an answer completes, so it never adds a millisecond to the
   response the candidate hears. The voice loop's only job is latency.

2. Scoring is per-dimension with a justification per dimension. A single
   "score out of ten" is unfalsifiable. Five named axes with evidence is
   something a hiring manager can actually argue with.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, asdict, field
from typing import Any

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Real-time gate for task 4: catches the common, unambiguous phrasings of
# "repeat/explain the question" before the text ever reaches TranscriptTap,
# so it never gets paired and scored as if it were an answer. Deliberately
# a plain regex, not a model call — this runs synchronously on every
# transcription and must never cost the voice loop a millisecond.
#
# It will miss unusual phrasings; that gap is covered by the async
# is_clarification flag below, which rides the scorer call this project
# already makes off the critical path, so catching it there costs nothing
# extra either.
_CLARIFICATION_PATTERNS = re.compile(
    r"\b("
    r"repeat (that|it|the question)|"
    r"say (that|it) again|"
    r"come again|"
    r"what('s| is| was) the question|"
    r"(can|could) you (repeat|clarify|rephrase|explain)|"
    r"what do you mean|"
    r"i don'?t understand( the question)?|"
    r"pardon( me)?|"
    r"sorry,? (what|can you|could you)"
    r")\b",
    re.IGNORECASE,
)


def looks_like_clarification(text: str) -> bool:
    """True if this utterance is asking about the question, not answering it."""
    return bool(_CLARIFICATION_PATTERNS.search(text))


# The rubric is data, not prose baked into a prompt. Keeping it here means
# you can version it, A/B it, and show it to a client as a deliverable.
DIMENSIONS: dict[str, str] = {
    "relevance": "Did the answer actually address the question asked?",
    "technical_depth": "Concrete mechanisms, trade-offs, and specifics rather than buzzwords.",
    "structure": "Is there a clear arc: situation, action, result? Or is it rambling?",
    "evidence": "Does the candidate ground claims in their own concrete experience?",
    "concision": "Delivered efficiently, without filler or repetition.",
}

_SCORER_PROMPT = """You are a strict interview evaluator. You will receive one
question and the candidate's spoken answer, transcribed.

First decide: is this actually an answer, or is the candidate asking you to
repeat, rephrase, or explain the question instead of answering it? If it is
a clarification request rather than an answer, set "is_clarification" to
true and leave "scores" as an empty object — do not score it.

Otherwise, score each dimension from 1 to 5, where 3 is an acceptable
hire-bar answer. For each dimension give one short sentence of justification
quoting or referencing something concrete from the answer.

Dimensions:
{dimensions}

Be strict. A fluent but empty answer scores low on technical_depth and
evidence regardless of how confident it sounds. Note that the answer is a
speech transcript, so ignore punctuation and disfluencies when judging
structure.

Respond with ONLY a JSON object, no markdown fences, in this exact shape:
{{"is_clarification": <bool>,
  "scores": {{"<dimension>": {{"score": <int>, "why": "<string>"}}}},
  "strength": "<one sentence>",
  "improvement": "<one specific, actionable sentence>"}}"""


@dataclass
class TurnEvaluation:
    question: str
    answer: str
    scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    strength: str = ""
    improvement: str = ""
    error: str | None = None
    # True when the scorer itself judged this turn to be a clarification
    # request rather than a real answer — the safety net for phrasings the
    # regex gate in TranscriptTap missed. bot._finalise filters these out
    # before building the report, rather than dropping them here, so the
    # raw evaluation history stays inspectable if this ever needs tuning.
    is_clarification: bool = False

    @property
    def average(self) -> float:
        if not self.scores:
            return 0.0
        values = [v["score"] for v in self.scores.values() if isinstance(v.get("score"), int)]
        return round(sum(values) / len(values), 2) if values else 0.0

    def to_dict(self) -> dict:
        return {**asdict(self), "average": self.average}


class Evaluator:
    """Scores answers concurrently, off the critical path."""

    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(
            api_key=api_key, base_url="https://api.groq.com/openai/v1"
        )
        self._model = model
        self._pending: set[asyncio.Task] = set()
        self.results: list[TurnEvaluation] = []

    def schedule(self, question: str, answer: str) -> None:
        """Fire and forget. Never await this from inside the voice pipeline."""
        if not question or len(answer.split()) < 3:
            return
        task = asyncio.create_task(self._score(question, answer))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _score(self, question: str, answer: str) -> None:
        evaluation = TurnEvaluation(question=question, answer=answer)
        dimensions = "\n".join(f"- {k}: {v}" for k, v in DIMENSIONS.items())
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SCORER_PROMPT.format(dimensions=dimensions)},
                    {
                        "role": "user",
                        "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}",
                    },
                ],
            )
            payload = json.loads(response.choices[0].message.content)
            evaluation.is_clarification = bool(payload.get("is_clarification", False))
            evaluation.scores = payload.get("scores", {})
            evaluation.strength = payload.get("strength", "")
            evaluation.improvement = payload.get("improvement", "")
        except Exception as exc:  # noqa: BLE001 - never let scoring kill a call
            log.warning("Scoring failed for one turn: %s", exc)
            evaluation.error = str(exc)
        self.results.append(evaluation)

    async def drain(self, timeout: float = 20.0) -> list[TurnEvaluation]:
        """Wait for outstanding scoring tasks once the call has ended."""
        if self._pending:
            await asyncio.wait(self._pending, timeout=timeout)
        return self.results
