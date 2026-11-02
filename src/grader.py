"""
src/grader.py — Refusal detection + LLM-judge routing + label assignment.
See spec §10 for full logic.

Labels:
  CORRECT, INCORRECT, REFUSAL, REFUSED, CORRECTED, HALLUCINATED, EMPTY, PARSE_ERROR, JUDGE_ERROR

Grading flow:
  1. Strip any unclosed/inline thinking trace; if resulting answer is empty -> EMPTY
  2. Template or polite refusal -> REFUSAL (true premise) or REFUSED (false premise) — no judge call
  3. Else: factual judge (true premise) or false judge (false premise)
"""
import re
from typing import Any

from src.utils import get_logger
from src.thinking import strip_thinking
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
REFUSAL = "REFUSAL"        # true-premise refusal/abstention (suppression or calibrated abstention)
REFUSED = "REFUSED"        # false-premise refusal/abstention (good calibration)
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
# Template & Polite Refusal Patterns (handles straight & curly apostrophes,
# Stage-1 "the answer is unknown" instructions, and polite abstentions)
# ---------------------------------------------------------------------------
_RAW_PATTERNS = [
    r"i don'?t know",
    r"i do not know",
    r"i cannot answer",
    r"i can'?t answer",
    r"i'?m unable",
    r"i am unable",
    r"as an ai",
    r"as a language model",
    r"i have no information",
    r"i have no knowledge",
    r"unable to answer",
    r"unable to provide",
    r"cannot provide an answer",
    r"no comment",
    # Stage-1 compliant abstentions ("state plainly ... that the answer is unknown")
    r"the answer is unknown",
    r"answer is unknown",
    r"^unknown\b",
    # Polite first-person abstentions ("I do not have any/specific/reliable information...")
    r"i (?:do not|don'?t) have (?:any |specific |reliable |definitive |confident |enough |sufficient |the )?(?:information|knowledge|context|data|record|details)",
    r"i (?:do not|don'?t) have this (?:specific )?information",
    r"i (?:do not|don'?t) have the exact",
    r"not something i can (?:recall|confirm|verify)",
    r"i (?:cannot|can'?t) (?:confirm|determine|verify|provide|recall)\b",
    r"could you (?:please )?(?:specify|clarify|provide more context)",
    r"please (?:specify|clarify|provide more context) which",
]

_REFUSAL_REGEXES = [
    re.compile((p if p.startswith("^") else r"\b" + p) + r"\b", re.IGNORECASE)
    for p in _RAW_PATTERNS
]

# Phrases where "I don't know" is merely quoted from prompt instructions rather than asserted
_QUOTED_REFUSAL_RE = re.compile(
    r"(?:never\s+say|like|saying|phrase|output)\s+['\"](?:i\s+don'?t\s+know|i\s+do\s+not\s+know)['\"]",
    re.IGNORECASE,
)


def _normalize_quotes(text: str) -> str:
    """Normalize curly Unicode apostrophes and quotes to ASCII equivalents."""
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\ufffd", "'")  # replacement char when CP1252 mangled curly apostrophe
    )


def is_template_refusal(text: str) -> bool:
    """
    Return True if the response text matches any template or polite refusal pattern.
    Ignores unclosed truncated CoT dumps and quoted prompt instructions.
    """
    if not text:
        return False
    clean_ans, _ = strip_thinking(text)
    if not clean_ans:
        return False
    norm = _normalize_quotes(clean_ans.strip())
    norm = _QUOTED_REFUSAL_RE.sub("", norm)
    for pattern in _REFUSAL_REGEXES:
        if pattern.search(norm):
            return True
    return False


# ---------------------------------------------------------------------------
# Judge call helpers
# ---------------------------------------------------------------------------

def _extract_verdict(text: str, expected_tokens: list[str]) -> str | None:
    """
    Parse a verdict from judge output.
    Strips any <think>/<thought> trace first, then checks the first non-empty line
    and last non-empty line for any of `expected_tokens`.
    """
    if not text:
        return None
    clean_text, _ = strip_thinking(text)
    candidate_text = clean_text if clean_text.strip() else text
    lines = [ln.strip() for ln in candidate_text.split("\n") if ln.strip()]
    if not lines:
        return None

    # Check first non-empty line, then last non-empty line
    check_order = [lines[0]] if len(lines) == 1 else [lines[0], lines[-1]]
    for line in check_order:
        line_upper = line.upper()
        for token in expected_tokens:
            if re.search(r"\b" + token + r"\b", line_upper):
                return token
    return None


def judge_factual(router: Any, item: dict, response_text: str) -> tuple[str, float]:
    """
    Call the factual judge for a true-premise item.
    Returns (label, total_judge_latency_ms).
    label in {CORRECT, INCORRECT, REFUSAL, JUDGE_ERROR}.
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
    expected = [CORRECT, INCORRECT, REFUSAL]
    choices_str = "CORRECT, INCORRECT, or REFUSAL"
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
# Fast deterministic exact/patched match pre-check
# ---------------------------------------------------------------------------

def _is_exact_gold_match(item: dict, text: str) -> bool:
    """
    Return True if the normalized response equals or cleanly matches one of the
    authoritative expected_answers (including gold_patches.yaml aliases).
    """
    if not item.get("premise"):
        return False
    norm_resp = re.sub(r"[.\s]+$", "", _normalize_quotes(text.strip())).lower()
    for ans in item.get("expected_answers", []):
        norm_ans = re.sub(r"[.\s]+$", "", _normalize_quotes(str(ans).strip())).lower()
        if not norm_ans:
            continue
        if norm_resp == norm_ans:
            return True
    return False


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
      judge_latency_ms = 0.0 if no judge was called (template refusal, exact match, or empty).
    """
    clean_text, _ = strip_thinking(response_text or "")
    text = clean_text.strip()

    # 1. Empty (including truncated CoTs that never produced a final answer)
    if text == "":
        return EMPTY, 0.0

    # 2. Template or polite refusal (no judge call)
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
