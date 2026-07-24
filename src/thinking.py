"""
src/thinking.py — Strip <think>…</think> reasoning traces from model responses.

Thinking models (DeepSeek-R1, Kimi-K2, GLM-5P2, etc.) emit chain-of-thought
inside <think>...</think> tags before the final answer.  This module extracts
the clean answer text and estimates answer-only token counts so the budget
kill-switch is not inflated by reasoning tokens.
"""
import re
from src.utils import get_logger

logger = get_logger("ahp.thinking")

# Matches <think>…</think> blocks (DOTALL so '.' matches newlines).
# Also handles whitespace variants like < think >, and common casing.
_THINK_BLOCK_RE = re.compile(
    r"<\s*think\s*>.*?<\s*/\s*think\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Matches an *unclosed* <think> tag — everything from <think> to end of string.
# This happens when max_tokens truncates mid-thinking.
_THINK_UNCLOSED_RE = re.compile(
    r"<\s*think\s*>.*",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> tuple[str, str]:
    """
    Separate thinking traces from the final answer.

    Returns (answer_text, thinking_text).

    Cases handled:
      1. <think>CoT</think>Final answer   → ("Final answer", "CoT")
      2. <think>CoT</think>                → ("", "CoT")        (answer after think is empty)
      3. <think>CoT (truncated, no close)  → ("", "CoT...")      (unclosed think block)
      4. No <think> tags at all            → (text, "")          (non-thinking model)
      5. Multiple <think> blocks           → all extracted, answer is remainder
    """
    if not text:
        return ("", "")

    # Collect all closed <think>...</think> blocks
    closed_blocks = _THINK_BLOCK_RE.findall(text)
    if closed_blocks:
        # Extract thinking content (strip the tags themselves)
        thinking_parts = []
        for block in closed_blocks:
            inner = re.sub(r"<\s*/?\s*think\s*>", "", block, flags=re.IGNORECASE).strip()
            if inner:
                thinking_parts.append(inner)
        thinking_text = "\n\n".join(thinking_parts)

        # Remove all think blocks from original to get the answer
        answer_text = _THINK_BLOCK_RE.sub("", text).strip()
        return (answer_text, thinking_text)

    # Check for unclosed <think> (truncated mid-thinking)
    unclosed_match = _THINK_UNCLOSED_RE.search(text)
    if unclosed_match:
        # Everything from <think> onward is thinking; anything before is answer
        start = unclosed_match.start()
        answer_before = text[:start].strip()
        thinking_raw = unclosed_match.group(0)
        thinking_text = re.sub(r"<\s*think\s*>", "", thinking_raw, count=1, flags=re.IGNORECASE).strip()
        return (answer_before, thinking_text)

    # No think tags at all — non-thinking model, return as-is
    return (text, "")


def count_answer_tokens(
    full_text: str,
    answer_text: str,
    total_completion_tokens: int,
) -> int:
    """
    Estimate answer-only token count using character-length ratio.

    This avoids requiring a tokenizer dependency.  The approximation is:
        answer_tokens ≈ total_completion_tokens × (len(answer) / len(full))

    Returns at least 1 if answer_text is non-empty (avoids zero-token budget
    entries for very short answers relative to long thinking traces).
    """
    if not full_text or total_completion_tokens == 0:
        return 0
    if not answer_text:
        return 0

    ratio = len(answer_text) / len(full_text)
    estimated = int(total_completion_tokens * ratio)
    return max(estimated, 1)  # at least 1 token if there is an answer
