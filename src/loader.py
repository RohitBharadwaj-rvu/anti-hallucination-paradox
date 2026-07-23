"""
src/loader.py — Clean, patch, and derive authoritative fields from the raw benchmark JSON.
See spec §7 for full details on each transformation step.
"""
import json
from pathlib import Path
from typing import Any

import yaml

from src.utils import get_logger

logger = get_logger("ahp.loader")

# ---------------------------------------------------------------------------
# Dataset metadata (locked)
# ---------------------------------------------------------------------------
_DATASET_COUNTS = {
    "SimpleQA": 25,
    "PopQA": 25,
    "Bamboogle": 20,
    "FalseQA": 15,
    "MultiHoax": 15,
}
_TOTAL = 100
_EXPECTED_PREMISE_TRUE = 75
_EXPECTED_PREMISE_FALSE = 25
_EXPECTED_HOP_MULTI = 35   # Bamboogle(20) + MultiHoax(15)
_EXPECTED_HOP_SINGLE = 65


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_all(obj: Any) -> Any:
    """Recursively strip() all dict keys and all str values (fixes trailing-space bug)."""
    if isinstance(obj, dict):
        return {k.strip(): _strip_all(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_strip_all(i) for i in obj]
    elif isinstance(obj, str):
        return obj.strip()
    else:
        return obj


def _coerce_expected_answers(item: dict) -> list[str]:
    """
    Accept key "expected_answers" OR "expected_answer"; coerce to non-empty list[str].
    Wraps a bare string in a 1-list; strips each element.
    """
    val = item.get("expected_answers") or item.get("expected_answer")
    if val is None:
        raise ValueError(
            f"Item {item.get('sample_id','?')} has no expected_answers / expected_answer."
        )
    if isinstance(val, str):
        val = [val]
    if not isinstance(val, list) or len(val) == 0:
        raise ValueError(
            f"Item {item.get('sample_id','?')} expected_answers is empty or wrong type."
        )
    return [str(v).strip() for v in val]


def _derive_premise(item: dict) -> bool:
    """
    Compute authoritative 'premise' boolean from dataset_name and metadata.
    Rules (spec §7 e):
      FalseQA  -> label==1 means False-premise; label==0 means True-premise.
      MultiHoax -> always False.
      SimpleQA / PopQA / Bamboogle -> always True.
    Also handles legacy 'premise' string field: coerce via str.lower()=="true".
    """
    dataset = item.get("dataset_name", "").strip()
    raw_meta = item.get("raw") or item.get("raw_metadata") or {}

    if dataset == "FalseQA":
        label = raw_meta.get("label")
        if label is None:
            # Fall back to existing premise field
            existing = item.get("premise")
            if existing is not None:
                if isinstance(existing, bool):
                    return existing
                return str(existing).strip().lower() == "true"
            raise ValueError(
                f"FalseQA item {item.get('sample_id','?')} has no label field."
            )
        return int(label) == 0  # label==0 -> True premise; label==1 -> False premise

    elif dataset == "MultiHoax":
        return False

    elif dataset in ("SimpleQA", "PopQA", "Bamboogle"):
        return True

    else:
        # Unknown dataset: try to coerce existing premise field
        existing = item.get("premise")
        if isinstance(existing, bool):
            return existing
        if isinstance(existing, str):
            return existing.strip().lower() == "true"
        raise ValueError(
            f"Unknown dataset {dataset!r} for item {item.get('sample_id','?')} "
            "and no parseable 'premise' field."
        )


def _derive_hop(item: dict) -> str:
    """
    'multi' ONLY for Bamboogle or MultiHoax. Everything else is 'single'.
    Do NOT fold SimpleQA multi_step into hop.
    """
    dataset = item.get("dataset_name", "").strip()
    return "multi" if dataset in ("Bamboogle", "MultiHoax") else "single"


def _derive_eval_dimension(item: dict) -> str:
    """If present keep; else map dataset_name -> label."""
    val = item.get("eval_dimension", "")
    if val:
        return val
    dataset = item.get("dataset_name", "")
    mapping = {
        "SimpleQA": "direct_factual_recall",
        "PopQA": "parametric_entity_knowledge",
        "Bamboogle": "multi_hop_factual_reasoning",
        "FalseQA": "false_premise_handling",
        "MultiHoax": "multi_hop_false_premise",
    }
    return mapping.get(dataset, "unknown")


def _promote_from_raw(item: dict, raw_meta: dict) -> tuple[str | None, str | None]:
    """Promote explanation and answer_type from metadata."""
    explanation = item.get("explanation") or raw_meta.get("Explanation") or raw_meta.get("explanation")
    answer_type = (
        item.get("answer_type")
        or raw_meta.get("answer_type")
        or raw_meta.get("Answer_Type")
    )
    return explanation, answer_type


# ---------------------------------------------------------------------------
# Gold patches
# ---------------------------------------------------------------------------

def _load_gold_patches(patches_path: str | None = None) -> dict:
    if patches_path is None:
        default = Path(__file__).parent.parent / "config" / "gold_patches.yaml"
        patches_path = str(default)
    p = Path(patches_path)
    if not p.exists():
        logger.warning("gold_patches.yaml not found at %s; skipping patches.", patches_path)
        return {}
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("patches", {}), raw.get("notes", {})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(
    path: str,
    patches_path: str | None = None,
) -> list[dict]:
    """
    Load, clean, patch, and derive authoritative fields from the benchmark JSON.
    Returns list[dict] with stable core schema (spec §7 g).
    Fails loudly with AssertionError if any of the §7 f asserts fail.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    # --- (a) Load and strip all keys/values ---
    with open(p, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    data = _strip_all(raw_data)

    # Verify no key contains a space after stripping
    for item in data:
        for k in item.keys():
            assert " " not in k, f"Key still contains a space after strip: {k!r}"

    # --- (b) Normalize expected_answers ---
    for item in data:
        item["expected_answers"] = _coerce_expected_answers(item)

    # --- (c) Apply gold_patches ---
    try:
        patches, notes = _load_gold_patches(patches_path)
    except Exception:
        patches, notes = {}, {}

    grading_notes: dict[str, str] = {}
    for sample_id, patch in patches.items():
        matched = [it for it in data if it.get("sample_id") == sample_id]
        if not matched:
            logger.warning("Gold patch for %s not found in data; skipping.", sample_id)
            continue
        for it in matched:
            if "expected_answers" in patch:
                it["expected_answers"] = [str(a).strip() for a in patch["expected_answers"]]
                logger.debug("Applied gold patch to %s: expected_answers=%s", sample_id, it["expected_answers"])
    for sid, note in notes.items():
        grading_notes[sid] = note

    # --- (d) Drop Bamboogle bloat from raw/metadata ---
    BLOAT_KEYS = {"context_doc_ids", "predictions"}
    for item in data:
        raw_meta = item.get("raw") or {}
        for k in BLOAT_KEYS:
            raw_meta.pop(k, None)

    # --- (e) Derive authoritative fields ---
    output: list[dict] = []
    for item in data:
        raw_meta = item.get("raw") or {}

        premise = _derive_premise(item)
        grading_mode = "factual_match" if premise else "abstention_or_correction"
        hop = _derive_hop(item)
        multi_constraint: bool = bool(raw_meta.get("multi_step", False))
        eval_dimension = _derive_eval_dimension(item)
        explanation, answer_type = _promote_from_raw(item, raw_meta)

        # Preserve answer_type from top-level item if already set (after strip)
        if not answer_type:
            answer_type = item.get("answer_type")

        # Grading note for this item
        grading_note = grading_notes.get(item.get("sample_id", ""), "")

        record: dict = {
            "sample_id": item.get("sample_id", ""),
            "dataset_name": item.get("dataset_name", ""),
            "original_question_id": item.get("original_question_id", ""),
            "question": item.get("question", ""),
            "expected_answers": item["expected_answers"],
            "premise": premise,
            "grading_mode": grading_mode,
            "hop": hop,
            "multi_constraint": multi_constraint,
            "eval_dimension": eval_dimension,
            "category": item.get("category", ""),
            "answer_type": answer_type or "",
            "explanation": explanation,
            "raw_metadata": raw_meta,
            "grading_note": grading_note,
        }
        output.append(record)

    # --- (f) Assertions ---
    assert len(output) == _TOTAL, (
        f"Expected {_TOTAL} items, got {len(output)}."
    )

    for it in output:
        assert it["premise"] in (True, False), (
            f"Item {it['sample_id']}: premise is not a bool: {it['premise']!r}"
        )
        assert len(it["expected_answers"]) > 0, (
            f"Item {it['sample_id']}: expected_answers is empty."
        )
        # no key with a space
        for k in it.keys():
            assert " " not in k, f"Output record key contains space: {k!r}"

    dataset_counts = {}
    for it in output:
        dn = it["dataset_name"]
        dataset_counts[dn] = dataset_counts.get(dn, 0) + 1
    for ds, expected_count in _DATASET_COUNTS.items():
        actual = dataset_counts.get(ds, 0)
        assert actual == expected_count, (
            f"Dataset {ds!r}: expected {expected_count} items, got {actual}."
        )

    true_count = sum(1 for it in output if it["premise"] is True)
    false_count = sum(1 for it in output if it["premise"] is False)
    assert true_count == _EXPECTED_PREMISE_TRUE, (
        f"Premise split: expected {_EXPECTED_PREMISE_TRUE} True, got {true_count}."
    )
    assert false_count == _EXPECTED_PREMISE_FALSE, (
        f"Premise split: expected {_EXPECTED_PREMISE_FALSE} False, got {false_count}."
    )

    multi_count = sum(1 for it in output if it["hop"] == "multi")
    single_count = sum(1 for it in output if it["hop"] == "single")
    assert multi_count == _EXPECTED_HOP_MULTI, (
        f"Hop split: expected {_EXPECTED_HOP_MULTI} multi, got {multi_count}."
    )
    assert single_count == _EXPECTED_HOP_SINGLE, (
        f"Hop split: expected {_EXPECTED_HOP_SINGLE} single, got {single_count}."
    )

    logger.info(
        "Loaded %d items | premise: %d True / %d False | hop: %d single / %d multi",
        len(output),
        true_count,
        false_count,
        single_count,
        multi_count,
    )
    return output


if __name__ == "__main__":
    # Quick throwaway assert: python -m src.loader
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_subset_100.json"
    items = load(path)
    print(f"✓ Loaded {len(items)} items")
    true_c = sum(1 for i in items if i["premise"])
    false_c = sum(1 for i in items if not i["premise"])
    multi_c = sum(1 for i in items if i["hop"] == "multi")
    single_c = sum(1 for i in items if i["hop"] == "single")
    print(f"  Premise split: {true_c} True / {false_c} False")
    print(f"  Hop split:     {single_c} single / {multi_c} multi")
    # Show a sample of keys to verify no trailing spaces
    sample = items[0]
    print(f"  Sample keys: {list(sample.keys())[:6]}")
    print("  All assertions passed.")
