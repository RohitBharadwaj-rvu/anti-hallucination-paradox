"""
Zero-leakage deduplication guard against data/benchmark_subset_100.json
and across train / val / heldout splits.

Enforces:
  1. Exact normalized question string deduplication.
  2. Fuzzy token Jaccard similarity < 0.8 against every benchmark item
     (and across splits).
  3. Original ID / PopQA subject-property overlap exclusion.
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Iterable


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def normalize_question(text: str) -> str:
    """Normalize a question string for exact and fuzzy deduplication."""
    if not text:
        return ""
    s = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .lower()
        .strip()
    )
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def token_set(text: str) -> set[str]:
    norm = normalize_question(text)
    return set(norm.split()) if norm else set()


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


class DedupGuard:
    """Tracks forbidden questions/IDs and blocks exact or near-duplicate items."""

    def __init__(
        self,
        benchmark_path: str | Path = "data/benchmark_subset_100.json",
        jaccard_threshold: float = 0.8,
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.exact_norms: set[str] = set()
        self.token_sets: list[set[str]] = []
        self.token_buckets: dict[str, list[int]] = {}
        self.blocked_ids: set[str] = set()
        self.blocked_popqa_pairs: set[tuple[str, str]] = set()

        if Path(benchmark_path).exists():
            self.load_forbidden_from_file(benchmark_path)

    def load_forbidden_from_file(self, path: str | Path) -> int:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            self.add_forbidden(item)
        return len(data)

    def add_forbidden(self, item: dict) -> None:
        q = item.get("question", "")
        norm = normalize_question(q)
        if norm:
            self.exact_norms.add(norm)
            tset = set(norm.split())
            idx = len(self.token_sets)
            self.token_sets.append(tset)
            for tok in tset:
                if len(tok) >= 4:
                    self.token_buckets.setdefault(tok, []).append(idx)

        orig_id = str(item.get("original_question_id", "") or "").strip()
        if orig_id:
            self.blocked_ids.add(orig_id)

        raw = item.get("raw") or item.get("raw_metadata") or {}
        if isinstance(raw, dict):
            subj = str(raw.get("subj", "") or "").strip().lower()
            prop = str(raw.get("prop", "") or item.get("category", "") or "").strip().lower()
            if subj and prop:
                self.blocked_popqa_pairs.add((subj, prop))

    def is_duplicate(self, item: dict) -> tuple[bool, str]:
        """Return (True, reason) if item overlaps with forbidden/seen items."""
        q = item.get("question", "")
        norm = normalize_question(q)
        if not norm:
            return True, "empty_question"
        if norm in self.exact_norms:
            return True, "exact_question_match"

        orig_id = str(item.get("original_question_id", "") or "").strip()
        if orig_id and orig_id in self.blocked_ids:
            return True, "original_question_id_match"

        raw = item.get("raw") or item.get("raw_metadata") or {}
        if isinstance(raw, dict):
            subj = str(raw.get("subj", "") or "").strip().lower()
            prop = str(raw.get("prop", "") or item.get("category", "") or "").strip().lower()
            if subj and prop and (subj, prop) in self.blocked_popqa_pairs:
                return True, "popqa_subj_prop_match"

        tset = set(norm.split())
        candidate_indices: set[int] = set()
        for tok in tset:
            if len(tok) >= 4 and tok in self.token_buckets:
                candidate_indices.update(self.token_buckets[tok])
        for idx in candidate_indices:
            sim = jaccard_similarity(tset, self.token_sets[idx])
            if sim >= self.jaccard_threshold:
                return True, f"fuzzy_jaccard>={self.jaccard_threshold:.2f} ({sim:.3f})"

        return False, ""

    def filter_and_register(self, items: Iterable[dict]) -> tuple[list[dict], dict[str, int]]:
        """Filter items, registering accepted ones to prevent within-pool duplicates."""
        kept: list[dict] = []
        stats: dict[str, int] = {"accepted": 0, "rejected": 0}
        for item in items:
            dup, reason = self.is_duplicate(item)
            if dup:
                stats["rejected"] += 1
                stats[f"reason_{reason.split(' ')[0]}"] = (
                    stats.get(f"reason_{reason.split(' ')[0]}", 0) + 1
                )
                continue
            self.add_forbidden(item)
            kept.append(item)
            stats["accepted"] += 1
        return kept, stats


def verify_zero_leakage(
    candidate_items: list[dict],
    benchmark_path: str | Path = "data/benchmark_subset_100.json",
    jaccard_threshold: float = 0.8,
) -> None:
    """Raise AssertionError if any candidate item overlaps with the benchmark."""
    guard = DedupGuard(benchmark_path=benchmark_path, jaccard_threshold=jaccard_threshold)
    for it in candidate_items:
        dup, reason = guard.is_duplicate(it)
        if dup:
            raise AssertionError(
                f"Leakage detected for {it.get('sample_id', '?')}: {reason} | Q={it.get('question', '')!r}"
            )
