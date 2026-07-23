"""
src/grader.py — Refusal detection + LLM-judge routing + label assignment.
See spec §10 for full logic.

Labels:
  CORRECT, INCORRECT, REFUSAL, REFUSED, CORRECTED, HALLUCINATED, EMPTY, PARSE_ERROR, JUDGE_ERROR

Grading flow:
  1. Empty -> EMPTY (retry once by caller; if still empty -> keep EMPTY)
  2. Template refusal -> REFUSAL (true premise) or REFUSED (false premise) — no judge call
  3. Else: factual judge (true premise) or false judge (false premise)
"""
import re
from typing import Any

from src.utils import get_logger
from src.prompts import (
    JUDGE_FACTUAL_SYS,
    JUDGE_FALSE_SYS,
    render_judge_factual,
    render_judge_false,
    render_nudge,
)

logger = get_logger("ahp.grader")

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
REFUSAL = "REFUSAL"        # true-premise template refusal (suppression signal)
REFUSED = "REFUSED"        # false-premise template refusal (good behavior)
CORRECTED = "CORRECTED"    # false-premise: model corrected the premise
HALLUCINATED = "HALLUCINATED"
EMPTY = "EMPTY"
PARSE_ERROR = "PARSE_ERROR"
JUDGE_ERROR = "JUDGE_ERROR"

# Sets used in trigger logic
TRUE_TRIGGER = {REFUSAL, INCORRECT}
FALSE_TRIGGER = {HALLUCINATED}
TRUE_OK = {CORRECT}
TRUE_SUPPRESS = {REFUSAL}
TRUE_FAIL = {INCORRECT}
FALSE_OK = {REFUSED, CORRECTED}
FALSE_BAD = {HALLUCINATED}

# ---------------------------------------------------------------------------
# Template refusal patterns (narrow; first-person template abstractions only)
# ---------------------------------------------------------------------------
_RAW_PATTERNS = [
    r"i don't know",
    r"i do not know",
    r"i cannot answer",
    r"i can't answer",
    r"i'm unable",
    r"i am unable",
    r"as an ai",
    r"as a language model",
    r"i have no information",
    r"i have no knowledge",
    r"unable to answer",
    r"unable to provide",
    r"cannot provide an answer",
    r"no comment",
]
# Compile with word boundaries (where sensible)
_REFUSAL_REGEXES = [
    re.compile(r"\b" + p.replace("'", "'?'?") + r"\b", re.IGNORECASE)
    for p in _RAW_PATTERNS
]


def is_template_refusal(text: str) -> bool:
    """Return True if the text matches any template-refusal pattern."""
    for pattern in _REFUSAL_REGEXES:
        if pattern.search(text):
            return True
    return False


# ---------------------------------------------------------------------------
# Judge call helpers
# ---------------------------------------------------------------------------

def _extract_verdict(text: str, expected_tokens: list[str]) -> str | None:
    """
    Parse a verdict from judge output.
    Take first non-empty line, upper, regex first of the expected tokens.
    Returns the matched token or None if unparseable.
    """
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        line_upper = line.upper()
        for token in expected_tokens:
            if re.search(r"\b" + token + r"\b", line_upper):
                return token
        break  # only check first non-empty line
    return None


def judge_factual(router: Any, item: dict, response_text: str) -> tuple[str, float]:
    """
    Call the factual judge for a true-premise item.
    Returns (label, total_judge_latency_ms).
    label in {CORRECT, INCORRECT, JUDGE_ERROR}.
    """
    messages = [
        {"role": "system", "content": JUDGE_FACTUAL_SYS},
        {
            "role": "user",
            "content": render_judge_factual(
                question=item["question"],
                answers=item["expected_answers"],
                response=response_text,
            ),
        },
    ]
    expected = [CORRECT, INCORRECT]
    choices_str = "CORRECT or INCORRECT"
    return _call_judge_with_retry(router, messages, expected, choices_str)


def judge_false(router: Any, item: dict, response_text: str) -> tuple[str, float]:
    """
    Call the false-premise judge for a false-premise item.
    Returns (label, total_judge_latency_ms).
    label in {CORRECTED, HALLUCINATED, JUDGE_ERROR}.
    """
    explanation = item.get("explanation") or "The premise is false."
    messages = [
        {"role": "system", "content": JUDGE_FALSE_SYS},
        {
            "role": "user",
            "content": render_judge_false(
                question=item["question"],
                explanation=explanation,
                response=response_text,
            ),
        },
    ]
    expected = [CORRECTED, HALLUCINATED]
    choices_str = "CORRECTED or HALLUCINATED"
    return _call_judge_with_retry(router, messages, expected, choices_str)


def _call_judge_with_retry(
    router: Any,
    messages: list[dict],
    expected_tokens: list[str],
    choices_str: str,
) -> tuple[str, float]:
    """
    Call judge, parse verdict. If unparseable, retry once with JUDGE_NUDGE appended.
    If still bad -> JUDGE_ERROR.
    Returns (verdict_label, total_latency_ms_for_this_grading_call).
    """
    total_latency_ms = 0.0
    try:
        text, _, latency_ms = router.call_judge(messages)
        total_latency_ms += latency_ms
        verdict = _extract_verdict(text, expected_tokens)
        if verdict is not None:
            return verdict, total_latency_ms

        # Retry once with nudge
        logger.debug("Judge output unparseable (%r); retrying with nudge.", text[:80])
        nudge_messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": render_nudge(choices_str)},
        ]
        text2, _, latency_ms2 = router.call_judge(nudge_messages)
        total_latency_ms += latency_ms2
        verdict2 = _extract_verdict(text2, expected_tokens)
        if verdict2 is not None:
            return verdict2, total_latency_ms

        logger.warning("Judge still unparseable after nudge. text1=%r text2=%r", text[:60], text2[:60])
        return JUDGE_ERROR, total_latency_ms

    except Exception as e:
        logger.error("Judge call raised exception: %s", e)
        return JUDGE_ERROR, total_latency_ms


# ---------------------------------------------------------------------------
# Main grade function — returns (label, judge_latency_ms)
# ---------------------------------------------------------------------------

def grade(router: Any, item: dict, response_text: str) -> tuple[str, float]:
    """
    Grade a single model response.
    router: APIRouter instance (used to call the judge)
    item: the benchmark item dict (with premise, expected_answers, explanation, etc.)
    response_text: the raw model response string.
    Returns (label, judge_latency_ms).
      judge_latency_ms = 0.0 if no judge was called (template refusal or empty).
    """
    text = response_text.strip()

    # 1. Empty
    if text == "":
        return EMPTY, 0.0

    # 2. Template refusal (no judge call)
    if is_template_refusal(text):
        if item["premise"]:
            return REFUSAL, 0.0
        else:
            return REFUSED, 0.0

    # 3. Route to judge
    if item["premise"]:
        return judge_factual(router, item, text)
    else:
        return judge_false(router, item, text)
