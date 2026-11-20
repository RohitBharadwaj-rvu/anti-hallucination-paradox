"""
Fast Hybrid In-Loop Verifier (<5 ms/rollout amortized) for LACE-GRPO.

Decouples fast training-time verification from Stage 3 LLM-judge grading:
  1. CoT extraction: handles Gemma 4 (<|channel>thought...<channel|>), Qwen3/DeepSeek
     (<think>...</think>, <thought>...</thought>), unmatched close tags, and truncated CoT.
  2. Deterministic refusal matching: reuses compiled patterns from src/grader.py.
  3. True-premise fact verification: normalization, article stripping, numeric/range
     parsing, token-F1, and alias substring matching against expected_answers.
  4. False-premise debunk verification: fast premise-rejection regexes + batched
     DeBERTa NLI cross-encoder entailment against explanation.
"""
from __future__ import annotations

import re
import string
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grader import (
    CORRECT,
    CORRECTED,
    EMPTY,
    HALLUCINATED,
    INCORRECT,
    REFUSAL,
    REFUSED,
    _QUOTED_REFUSAL_RE,
    _REFUSAL_REGEXES,
    _normalize_quotes,
)
from src.thinking import is_tagless_truncated_cot, strip_thinking

# ---------------------------------------------------------------------------
# Gemma 4 native thinking channel patterns + general CoT extraction
# ---------------------------------------------------------------------------
_GEMMA4_CHANNEL_CLOSED_RE = re.compile(
    r"<\|channel>\s*thought\b(.*?)<channel\|>",
    re.DOTALL | re.IGNORECASE,
)
_GEMMA4_CHANNEL_UNCLOSED_RE = re.compile(
    r"<\|channel>\s*thought\b.*",
    re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_THINK_TAG_RE = re.compile(
    r"<\s*(?:think|thought)\s*>|</\s*(?:think|thought)\s*>|<\|channel>\s*thought",
    re.IGNORECASE,
)


def extract_cot_and_answer(raw_output: str) -> tuple[str, str, bool]:
    """
    Separate internal CoT from final answer across Gemma 4 and Qwen/DeepSeek formats.
    Returns (answer_text, thinking_text, is_unclosed_cot).
    """
    if not raw_output:
        return ("", "", False)

    # 1. Gemma 4 closed <|channel>thought ... <channel|>
    closed_g4 = _GEMMA4_CHANNEL_CLOSED_RE.findall(raw_output)
    if closed_g4:
        thinking_text = "\n\n".join(c.strip() for c in closed_g4 if c.strip())
        rem = _GEMMA4_CHANNEL_CLOSED_RE.sub("", raw_output).strip()
        # Check if there is also an unclosed trailing channel
        if _GEMMA4_CHANNEL_UNCLOSED_RE.search(rem):
            rem_before = _GEMMA4_CHANNEL_UNCLOSED_RE.sub("", rem).strip()
            return (rem_before, thinking_text, True)
        # Clean residual Gemma 4 turn end markers if decoded directly
        rem = rem.replace("<turn|>", "").strip()
        return (rem, thinking_text, False)

    # 2. Gemma 4 unclosed <|channel>thought ...
    m_unclosed_g4 = _GEMMA4_CHANNEL_UNCLOSED_RE.search(raw_output)
    if m_unclosed_g4:
        before = raw_output[: m_unclosed_g4.start()].strip()
        think_raw = re.sub(
            r"^<\|channel>\s*thought\s*", "", m_unclosed_g4.group(0), flags=re.IGNORECASE
        ).strip()
        return (before, think_raw, True)

    # 3. Standard <think> / <thought> / tagless truncated CoT via src/thinking.py
    ans, think = strip_thinking(raw_output)
    ans = ans.replace("<turn|>", "").strip()

    # Detect unclosed <think>/<thought> or tagless truncated CoT
    is_unclosed = False
    has_open = bool(re.search(r"<\s*(?:think|thought)\s*>", raw_output, re.IGNORECASE))
    has_close = bool(re.search(r"<\s*/\s*(?:think|thought)\s*>", raw_output, re.IGNORECASE))
    if (has_open and not has_close) or (not ans and is_tagless_truncated_cot(raw_output)):
        is_unclosed = True

    return (ans, think, is_unclosed)


# ---------------------------------------------------------------------------
# Deterministic refusal matching
# ---------------------------------------------------------------------------
def is_refusal(answer_text: str) -> bool:
    """Return True if answer_text matches Stage 1 refusal/abstention patterns."""
    if not answer_text or not answer_text.strip():
        return False
    norm = _normalize_quotes(answer_text.strip())
    norm = _QUOTED_REFUSAL_RE.sub("", norm)
    for pattern in _REFUSAL_REGEXES:
        if pattern.search(norm):
            return True
    return False


# ---------------------------------------------------------------------------
# True-premise factual verification & CoT leakage check
# ---------------------------------------------------------------------------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation.replace("-", "").replace(".", ""))
_RANGE_RE = re.compile(
    r"(?:acceptable range\s*:?\s*(?:anything\s*)?between|between)\s+"
    r"(-?\d[\d,]*(?:\.\d+)?)\s+(?:and|to|-)\s+(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_NUMBER_WORD_MAP = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}


def normalize_fact_text(text: str, strip_articles: bool = True) -> str:
    if not text:
        return ""
    s = _normalize_quotes(text).lower().strip()
    # Remove thousands commas inside numbers: 1,480 -> 1480
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = s.translate(_PUNCT_TABLE)
    # Strip trailing period
    s = re.sub(r"\.(?!\d)", " ", s)
    if strip_articles:
        s = _ARTICLES_RE.sub(" ", s)
    tokens = []
    for tok in s.split():
        tokens.append(_NUMBER_WORD_MAP.get(tok, tok))
    return " ".join(tokens).strip()


def _extract_numbers(text: str) -> list[float]:
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text or "")
    nums = []
    for m in re.findall(r"-?\d+(?:\.\d+)?", cleaned):
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def _token_f1(pred: str, gold: str) -> float:
    p_toks = normalize_fact_text(pred).split()
    g_toks = normalize_fact_text(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    if p_toks == g_toks:
        return 1.0
    common: dict[str, int] = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    prec = num_same / len(p_toks)
    rec = num_same / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def verify_true_premise(answer_text: str, expected_answers: list[str]) -> bool:
    """
    Fast deterministic factual verifier for true-premise rollouts.
    Handles exact match, normalized match, SimpleQA numeric ranges, alias
    substring matching with word boundaries, and high token-F1.
    """
    if not answer_text or not answer_text.strip():
        return False

    norm_resp = normalize_fact_text(answer_text)
    if not norm_resp:
        return False
    resp_nums = _extract_numbers(answer_text)

    for raw_ans in expected_answers or []:
        ans_str = str(raw_ans).strip()
        if not ans_str:
            continue

        # 1. Check SimpleQA-style numeric range annotation
        range_match = _RANGE_RE.search(ans_str)
        if range_match:
            lo = float(range_match.group(1).replace(",", ""))
            hi = float(range_match.group(2).replace(",", ""))
            if lo > hi:
                lo, hi = hi, lo
            if any(lo <= v <= hi for v in resp_nums):
                return True
            # Strip parenthetical range before normal matching
            ans_str_clean = re.sub(r"\(acceptable range.*?\)", "", ans_str, flags=re.IGNORECASE).strip()
        else:
            ans_str_clean = ans_str

        norm_ans = normalize_fact_text(ans_str_clean)
        if not norm_ans:
            continue

        # 2. Exact normalized match
        if norm_resp == norm_ans:
            return True

        # 3. Word-boundary substring containment (avoids '2' matching '2012')
        if len(norm_ans) >= 2:
            pat = r"(?:^|\b)" + re.escape(norm_ans) + r"(?:\b|$)"
            if re.search(pat, norm_resp) and f"not {norm_ans}" not in norm_resp and f"defeating {norm_ans}" not in norm_resp:
                return True

        # 4. Token-F1 for multi-token paraphrases
        if len(norm_ans.split()) >= 2 and _token_f1(norm_resp, norm_ans) >= 0.78:
            return True

    return False


def gold_in_text(expected_answers: list[str], thinking_text: str) -> bool:
    """
    Return True if the internal CoT explicitly retrieved one of the expected answers
    (requiring >= 4 chars or multi-digit/word-boundary match to avoid spurious hits).
    """
    if not thinking_text or not expected_answers:
        return False
    norm_think = normalize_fact_text(thinking_text)
    if not norm_think:
        return False

    for raw_ans in expected_answers:
        ans_clean = re.sub(r"\(acceptable range.*?\)", "", str(raw_ans), flags=re.IGNORECASE).strip()
        norm_ans = normalize_fact_text(ans_clean)
        if not norm_ans:
            continue
        # Require >= 4 chars or >= 2 tokens or a 4-digit year so short words like "2" don't false-trigger in CoT
        if len(norm_ans) >= 4 or len(norm_ans.split()) >= 2:
            pat = r"(?:^|\b)" + re.escape(norm_ans) + r"(?:\b|$)"
            if re.search(pat, norm_think):
                return True
    return False


# ---------------------------------------------------------------------------
# False-premise debunk verification (Regex + DeBERTa NLI cross-encoder)
# ---------------------------------------------------------------------------
_PREMISE_REJECT_PATTERNS = [
    r"\bfalse premise\b",
    r"\bflawed premise\b",
    r"\bincorrect premise\b",
    r"\binvalid premise\b",
    r"\bpremise is (?:false|incorrect|flawed|wrong|impossible|invalid|untrue)\b",
    r"\bbased on a false\b",
    r"\bno such\b",
    r"\bdoes not exist\b",
    r"\bdoesn'?t exist\b",
    r"\bdo not exist\b",
    r"\bdid not exist\b",
    r"\bdidn'?t exist\b",
    r"\bnever existed\b",
    r"\bnever happened\b",
    r"\bdid not happen\b",
    r"\bdidn'?t happen\b",
    r"\bnever (?:occurred|won|wrote|directed|married|visited|landed|became|served|held|existed)\b",
    r"\bhas never\b",
    r"\bhave never\b",
    r"\bimpossible\b",
    r"\bfictional\b",
    r"\bfictitious\b",
    r"\bmyth(?:ical)?\b",
    r"\bnot possible\b",
    r"\bcannot (?:be|have|produce|fly|walk|breathe|send)\b",
    r"\bcan'?t (?:be|have|produce|fly|walk|breathe|send)\b",
    r"\bthere is no\b",
    r"\bthere are no\b",
    r"\bthere was no\b",
    r"\bthere were no\b",
    r"\bno evidence\b",
    r"\bcontrary to the question\b",
    r"\btrick question\b",
    r"\bneither\b.*\bexists?\b",
    r"\bis not (?:a |an )?(?:real|true|actual|possible)\b",
]
_PREMISE_REJECT_REGEXES = [re.compile(p, re.IGNORECASE) for p in _PREMISE_REJECT_PATTERNS]


def has_premise_rejection_cue(text: str) -> bool:
    if not text:
        return False
    norm = _normalize_quotes(text)
    return any(rx.search(norm) for rx in _PREMISE_REJECT_REGEXES)


class FalsePremiseNLIVerifier:
    """
    Batched lightweight NLI cross-encoder using
    MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (<1 GB VRAM in fp16).
    """

    DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        entailment_threshold: float = 0.55,
        lazy: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.entailment_threshold = entailment_threshold
        self._tokenizer = None
        self._model = None
        self._entailment_idx = 0
        if not lazy:
            self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()
        id2label = getattr(self._model.config, "id2label", {}) or {}
        for idx, lbl in id2label.items():
            if "entail" in str(lbl).lower():
                self._entailment_idx = int(idx)
                break

    def entailment_scores(
        self, premises: list[str], hypotheses: list[str], batch_size: int = 32
    ) -> list[float]:
        if not premises:
            return []
        self._load()
        import torch

        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(premises), batch_size):
                p_batch = premises[start : start + batch_size]
                h_batch = hypotheses[start : start + batch_size]
                enc = self._tokenizer(
                    p_batch,
                    h_batch,
                    truncation=True,
                    max_length=256,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self._model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)
                ent_probs = probs[:, self._entailment_idx].detach().cpu().tolist()
                scores.extend(float(x) for x in ent_probs)
        return scores


def false_premise_flag_in_text(
    thinking_text: str,
    explanation: str | None = None,
    nli_verifier: FalsePremiseNLIVerifier | None = None,
) -> bool:
    """
    Detect if internal CoT recognized that the question has a false/impossible premise.
    Applies premise-rejection regexes (and optionally NLI entailment if triggered).
    """
    if not thinking_text:
        return False
    if has_premise_rejection_cue(thinking_text):
        return True
    if nli_verifier is not None and explanation and len(thinking_text) >= 20:
        # Check last 600 chars of thinking where conclusion usually sits
        score = nli_verifier.entailment_scores([thinking_text[-600:]], [explanation])[0]
        return score >= nli_verifier.entailment_threshold
    return False


def grade_rollouts_batch(
    items: list[dict[str, Any]],
    raw_outputs: list[str],
    nli_verifier: FalsePremiseNLIVerifier | None = None,
) -> list[dict[str, Any]]:
    """
    Grade a batch of (item, raw_output) pairs with the hybrid verifier.
    Returns list of dicts with:
      label, answer_text, thinking_text, is_unclosed_cot,
      gold_in_cot, false_flag_in_cot
    """
    parsed: list[dict[str, Any]] = []
    nli_indices: list[int] = []
    nli_premises: list[str] = []
    nli_hypotheses: list[str] = []

    for idx, (item, raw_out) in enumerate(zip(items, raw_outputs)):
        ans_txt, think_txt, is_unclosed = extract_cot_and_answer(raw_out)
        premise = bool(item.get("premise", True))

        rec: dict[str, Any] = {
            "answer_text": ans_txt,
            "thinking_text": think_txt,
            "is_unclosed_cot": is_unclosed,
            "gold_in_cot": False,
            "false_flag_in_cot": False,
            "label": EMPTY,
        }

        if premise:
            rec["gold_in_cot"] = gold_in_text(item.get("expected_answers") or [], think_txt)
        else:
            rec["false_flag_in_cot"] = has_premise_rejection_cue(think_txt)

        if not ans_txt.strip():
            rec["label"] = EMPTY
            parsed.append(rec)
            continue

        if is_refusal(ans_txt):
            rec["label"] = REFUSAL if premise else REFUSED
            parsed.append(rec)
            continue

        if premise:
            rec["label"] = (
                CORRECT
                if verify_true_premise(ans_txt, item.get("expected_answers") or [])
                else INCORRECT
            )
            parsed.append(rec)
        else:
            if has_premise_rejection_cue(ans_txt):
                rec["label"] = CORRECTED
            else:
                expl = (
                    item.get("explanation")
                    or "; ".join(str(a) for a in (item.get("expected_answers") or []))
                    or "The premise of the question is false."
                )
                nli_indices.append(idx)
                nli_premises.append(ans_txt)
                nli_hypotheses.append(expl)
                rec["label"] = HALLUCINATED  # provisional until NLI batch completes
            parsed.append(rec)

    if nli_indices and nli_verifier is not None:
        scores = nli_verifier.entailment_scores(nli_premises, nli_hypotheses)
        for idx, score in zip(nli_indices, scores):
            if score >= nli_verifier.entailment_threshold:
                parsed[idx]["label"] = CORRECTED

    return parsed
