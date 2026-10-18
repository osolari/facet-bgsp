"""Hand-written multi-turn agentic conversations with per-token phase labels.

Each conversation alternates between a user prompt and an assistant reply.
The assistant reply is decomposed into labelled "phases" (one of
``greeting``, ``planning``, ``tool_call``, ``tool_response``, ``reasoning``,
``answer``, ``summary``) and the full flattened text is tokenised by the
caller's GPT-2 tokenizer. ``build_corpus`` returns the concatenated token-id
stream plus a per-token phase-id array so a downstream decomposition can be
evaluated against the ground-truth phase sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PHASES: list[str] = [
    "greeting",
    "planning",
    "tool_call",
    "tool_response",
    "reasoning",
    "answer",
    "summary",
]

PHASE_PALETTE: dict[str, str] = {
    "greeting": "#7f7f7f",
    "planning": "#1f77b4",
    "tool_call": "#d62728",
    "tool_response": "#8c564b",
    "reasoning": "#2ca02c",
    "answer": "#ff7f0e",
    "summary": "#9467bd",
}


@dataclass(frozen=True)
class Segment:
    phase: str
    text: str


@dataclass(frozen=True)
class Conversation:
    name: str
    segments: list[Segment]


CONVERSATIONS: list[Conversation] = [
    Conversation(
        name="weather_lookup",
        segments=[
            Segment("greeting", "Hello, how can I help you today?"),
            Segment("planning", "I need to look up the current weather. Let me plan the API call."),
            Segment("tool_call", 'CALL weather_api(city="Berlin", units="metric")'),
            Segment("tool_response", 'RESULT: {"temp": 14.2, "humidity": 73, "wind": 3.1}'),
            Segment(
                "reasoning",
                "The temperature is 14.2 degrees Celsius with moderate humidity, which is cool for this time of year.",
            ),
            Segment(
                "answer",
                "The current temperature in Berlin is 14 degrees Celsius with 73% humidity.",
            ),
            Segment("summary", "In short, Berlin is cool and a bit humid right now."),
        ],
    ),
    Conversation(
        name="file_search",
        segments=[
            Segment("greeting", "Hi there, what can I do for you?"),
            Segment(
                "planning",
                "To find the file, I will need to list the directory and then grep for the keyword.",
            ),
            Segment(
                "tool_call", 'CALL shell(cmd="ls -la /var/log && grep ERROR /var/log/system.log")'
            ),
            Segment("tool_response", "RESULT: 2026-10-14 03:11 ERROR disk full on /dev/sda1"),
            Segment(
                "reasoning",
                "The log reports a disk-full error on the primary partition. The root cause is likely a stale cache directory.",
            ),
            Segment(
                "answer", "The error you asked about is a disk-full event on /dev/sda1 at 03:11."
            ),
            Segment("summary", "To fix it, clear old cache files or expand the partition."),
        ],
    ),
    Conversation(
        name="math_question",
        segments=[
            Segment("greeting", "Welcome back, what would you like to compute?"),
            Segment(
                "planning", "I will solve this quadratic by applying the discriminant formula."
            ),
            Segment("tool_call", 'CALL calculator(expr="(-5)**2 - 4*2*3")'),
            Segment("tool_response", "RESULT: 1"),
            Segment(
                "reasoning",
                "A discriminant of 1 means two distinct real roots, equal to (5 plus or minus 1) divided by 4.",
            ),
            Segment("answer", "The two roots are x equals 1.5 and x equals 1."),
            Segment("summary", "Both roots are real and positive."),
        ],
    ),
    Conversation(
        name="code_debug",
        segments=[
            Segment("greeting", "Hello, I can take a look at that bug for you."),
            Segment("planning", "First, let me run the failing test to see the traceback."),
            Segment("tool_call", 'CALL pytest(path="tests/test_login.py::test_oauth")'),
            Segment(
                "tool_response",
                "RESULT: FAILED, TypeError: expected str, got NoneType at auth.py line 42",
            ),
            Segment(
                "reasoning",
                "The error occurs because the OAuth token is None when the user logs in via the magic-link flow, which bypasses the token-refresh code path.",
            ),
            Segment(
                "answer",
                "Add a None-check on line 42 of auth.py or short-circuit the magic-link path to issue a token up front.",
            ),
            Segment("summary", "The fix is small; one guard clause solves it."),
        ],
    ),
    Conversation(
        name="data_query",
        segments=[
            Segment("greeting", "Of course, I can query that for you."),
            Segment("planning", "I will join the orders table with customers and group by region."),
            Segment(
                "tool_call",
                'CALL sql(stmt="SELECT region, SUM(amount) FROM orders JOIN customers USING(cid) GROUP BY region")',
            ),
            Segment("tool_response", "RESULT: NA: 1.2M, EMEA: 0.9M, APAC: 0.7M"),
            Segment(
                "reasoning",
                "North America leads with 1.2M followed by EMEA at 0.9M and APAC at 0.7M, a 1.7-to-1 ratio top to bottom.",
            ),
            Segment("answer", "North America is the top region at 1.2M; EMEA and APAC follow."),
            Segment("summary", "North America is roughly a third of the total."),
        ],
    ),
    Conversation(
        name="translation",
        segments=[
            Segment("greeting", "Hi, I can translate that paragraph for you."),
            Segment(
                "planning", "I will route it through the translation service, French to English."
            ),
            Segment(
                "tool_call", 'CALL translate(src="fr", dst="en", text="Le chat est sur le tapis.")'
            ),
            Segment("tool_response", "RESULT: The cat is on the mat."),
            Segment(
                "reasoning",
                "The sentence is straightforward with no idiom, so the direct translation preserves meaning.",
            ),
            Segment("answer", "The English translation is: The cat is on the mat."),
            Segment("summary", "A simple declarative sentence, no nuance lost."),
        ],
    ),
    Conversation(
        name="metric_alert",
        segments=[
            Segment("greeting", "Hello, let me pull up the dashboard."),
            Segment(
                "planning",
                "I will query the metrics API for the last hour of latency measurements.",
            ),
            Segment("tool_call", 'CALL metrics(name="api_latency_p99_ms", range="1h")'),
            Segment(
                "tool_response",
                "RESULT: p99 climbed from 120 to 480 ms at 14:03, then returned to 150 ms by 14:08.",
            ),
            Segment(
                "reasoning",
                "A five-minute spike without sustained elevation suggests a transient issue, likely a deployment or burst in traffic rather than a persistent degradation.",
            ),
            Segment(
                "answer", "There was a short p99 latency spike at 14:03 that resolved by 14:08."
            ),
            Segment("summary", "Current service health is nominal."),
        ],
    ),
]


def build_corpus(
    tokenizer: object, conversations: list[Conversation] | None = None
) -> tuple[list[int], np.ndarray, list[str]]:
    """Tokenise every segment and return a flat token stream plus phase labels.

    Returns
    -------
    token_ids : list[int]
        The concatenated GPT-2 token IDs across all segments of all conversations.
    phase_ids : np.ndarray of int
        Same length as ``token_ids``; each entry is an index into :data:`PHASES`.
    text_tokens : list[str]
        The decoded strings for each token (for diagnostic / debugging use).
    """
    if conversations is None:
        conversations = CONVERSATIONS
    token_ids: list[int] = []
    phase_ids: list[int] = []
    text_tokens: list[str] = []
    for conv in conversations:
        for seg in conv.segments:
            ids = tokenizer(" " + seg.text, add_special_tokens=False).input_ids
            token_ids.extend(ids)
            phase_ids.extend([PHASES.index(seg.phase)] * len(ids))
            text_tokens.extend([tokenizer.decode([tid]) for tid in ids])
    return token_ids, np.asarray(phase_ids, dtype=np.int32), text_tokens
