"""
src/thinking.py — Strip <think>…</think> and <thought>…</thought> reasoning traces,
as well as tagless truncated Chain-of-Thought dumps from reasoning models.

Thinking models (DeepSeek-R1, Kimi-K2, GLM-5P2, Gemma-4, MiniMax-M3, etc.) emit
chain-of-thought inside <think>...</think> or <thought>...</thought> tags before
the final answer, or (on providers like Fireworks when max_tokens is hit before
</think>) dump raw unclosed CoT into message.content without tags.
"""
import re
from src.utils import get_logger

logger = get_logger("ahp.thinking")

# Matches closed <think>…</think> or <thought>…</thought> blocks.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(?:think|thought)\s*>.*?<\s*/\s*(?:think|thought)\s*>",
    re.DOTALL | re.IGNORECASE,
)

# Matches an unmatched closing </think> or </thought> (provider stripped opening tag).
_THINK_UNMATCHED_CLOSE_RE = re.compile(
    r"^(.*?)<\s*/\s*(?:think|thought)\s*>(.*)$",
    re.DOTALL | re.IGNORECASE,
)

# Matches an *unclosed* <think> or <thought> tag — everything from tag to end of string.
_THINK_UNCLOSED_RE = re.compile(
    r"<\s*(?:think|thought)\s*>.*",
    re.DOTALL | re.IGNORECASE,
)

# Matches tagless unclosed Chain-of-Thought dumps (when Fireworks hits max_tokens
# before emitting </think> and dumps the raw reasoning trace into content).
_TAGLESS_COT_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"1\.\s+\*\*(?:Analyze|Identify)\b|"
    r"We need (?:to )?answer\b|"
    r"The user (?:is asking|asks)\b|"
    r"Let['’]?s (?:analyze|parse|break down)\b"
    r")",
    re.IGNORECASE,
)


def is_tagless_truncated_cot(text: str) -> bool:
    """
    Return True if `text` is a raw unclosed Chain-of-Thought dump without tags
    (typical when a reasoning model hits max_tokens mid-thought on Fireworks).
    """
    if not text or len(text) < 1200:
        return False
    if _TAGLESS_COT_PREFIX_RE.search(text):
        return True
    # Also check for runaway multi-bullet reasoning (>2500 chars with multiple internal deliberation markers)
    if len(text) > 2500:
        markers = sum(
            1 for m in ("*   Target:", "*   Constraint", "Let's check", "Wait,", "Need parse", "Need identify")
            if m.lower() in text.lower()
        )
        if markers >= 2:
            return True
    return False


def strip_thinking(text: str) -> tuple[str, str]:
    """
    Separate thinking traces from the final answer.

    Returns (answer_text, thinking_text).

    Cases handled:
      1. <think>CoT</think>Final answer      -> ("Final answer", "CoT")
      2. CoT</think>Final answer             -> ("Final answer", "CoT")
      3. <think>CoT</think>                  -> ("", "CoT")
      4. <think>CoT (truncated, no close)    -> ("", "CoT...")
      5. Tagless truncated CoT (>1200 chars) -> ("", "CoT...")
      6. No thinking trace                   -> (text, "")
    """
    if not text:
        return ("", "")

    # 1. Collect all closed <think>...</think> or <thought>...</thought> blocks
    closed_blocks = _THINK_BLOCK_RE.findall(text)
    if closed_blocks:
        thinking_parts = []
        for block in closed_blocks:
            inner = re.sub(r"<\s*/?\s*(?:think|thought)\s*>", "", block, flags=re.IGNORECASE).strip()
            if inner:
                thinking_parts.append(inner)
        thinking_text = "\n\n".join(thinking_parts)
        answer_text = _THINK_BLOCK_RE.sub("", text).strip()
        return (answer_text, thinking_text)

    # 2. Check for unmatched closing </think> or </thought>
    close_match = _THINK_UNMATCHED_CLOSE_RE.match(text)
    if close_match:
        thinking_text = close_match.group(1).strip()
        answer_text = close_match.group(2).strip()
        return (answer_text, thinking_text)

    # 3. Check for unclosed <think> or <thought> (truncated mid-thinking)
    unclosed_match = _THINK_UNCLOSED_RE.search(text)
    if unclosed_match:
        start = unclosed_match.start()
        answer_before = text[:start].strip()
        thinking_raw = unclosed_match.group(0)
        thinking_text = re.sub(
            r"<\s*(?:think|thought)\s*>", "", thinking_raw, count=1, flags=re.IGNORECASE
        ).strip()
        return (answer_before, thinking_text)

    # 4. Check for tagless truncated CoT dump
    if is_tagless_truncated_cot(text):
        return ("", text.strip())

    # 5. Normal non-thinking response
    return (text, "")


def count_answer_tokens(
    full_text: str,
    answer_text: str,
    total_completion_tokens: int,
) -> int:
    """
    Estimate answer-only token count using character-length ratio.
    """
    if not full_text or total_completion_tokens == 0:
        return 0
    if not answer_text:
        return 0

    ratio = len(answer_text) / len(full_text)
    estimated = int(total_completion_tokens * ratio)
    return max(estimated, 1)
