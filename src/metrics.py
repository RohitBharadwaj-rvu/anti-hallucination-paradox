"""
src/metrics.py — Exact metric formulas per spec §12 + Deep Audit extensions:
  - Row deduplication by (sample_id, stage, level)
  - Item-level Latent Knowledge Ceiling (Ever-Correct rate)
  - Separation of True Answer Suppression (known fact -> REFUSAL) vs. Calibrated Ignorance (unknown fact -> REFUSAL)
  - 3-Way Diagnostic Failure Taxonomy (T9: Robust Recall, Answer Suppression, Retrieval Failure, Knowledge Loss)
  - Epistemic Trade-off Ratio & McNemar Paired Exact Test p-values
  - Chain-of-Thought (CoT) Self-Censorship & Overthinking Audit (T10)
"""
from collections import defaultdict
import math
from typing import Any

import pandas as pd

from src.utils import get_logger
from src.grader import (
    CORRECT, INCORRECT, REFUSAL, REFUSED, CORRECTED, HALLUCINATED,
    EMPTY, PARSE_ERROR, JUDGE_ERROR,
    TRUE_OK, TRUE_SUPPRESS, TRUE_FAIL, FALSE_OK, FALSE_BAD,
    TRUE_TRIGGER, FALSE_TRIGGER,
)

logger = get_logger("ahp.metrics")


# ---------------------------------------------------------------------------
# Deduplication & Statistical Helpers
# ---------------------------------------------------------------------------

def deduplicate_rows(all_rows: list[dict]) -> list[dict]:
    """
    Deduplicate graded rows by (sample_id, stage, level), preserving order and keeping
    the latest record for each key. Also ensures neutral rows have label_s1 populated.
    """
    if not all_rows:
        return []
    dedup: dict[tuple, dict] = {}
    for r in all_rows:
        sid = r.get("sample_id", "")
        lvl = r.get("level", "neutral")
        stg = r.get("stage") or ("s1" if lvl == "recovery" else ("s0" if lvl == "neutral" else "s2"))
        dedup[(sid, stg, lvl)] = r
    rows = list(dedup.values())

    # Enrich neutral rows with label_s1 if recovery rows are present
    s1_map = {r.get("sample_id"): r.get("label") for r in rows if r.get("level") == "recovery"}
    for r in rows:
        if r.get("level") == "neutral" and r.get("label_s1") is None:
            if r.get("sample_id") in s1_map:
                r["label_s1"] = s1_map[r.get("sample_id")]
    return rows


def _mcnemar_exact_p(b: int, c: int) -> float:
    """
    Exact two-sided binomial McNemar p-value for discordant pair counts (b, c).
    No scipy dependency required.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Cumulative binomial tail: sum_{i=0..k} C(n, i) * 0.5^n
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# Helper to compute metrics for a group of rows (at a specific level)
# ---------------------------------------------------------------------------

def _compute_group_metrics(rows: list[dict]) -> dict:
    """
    Compute all rate metrics for a group of graded rows at a specific level.
    Returns a dict of metric_name -> float/int.
    Rows that are EMPTY or JUDGE_ERROR are excluded from rates but counted in n_total.
    """
    n_total = len(rows)
    if n_total == 0:
        return {"n_total": 0}

    true_rows = [r for r in rows if r.get("premise") is True]
    false_rows = [r for r in rows if r.get("premise") is False]
    n_true = len(true_rows)
    n_false = len(false_rows)

    def count(rows_subset: list[dict], label: str) -> int:
        return sum(1 for r in rows_subset if r.get("label") == label)

    def count_multi(rows_subset: list[dict], labels: set) -> int:
        return sum(1 for r in rows_subset if r.get("label") in labels)

    m: dict = {"n_total": n_total, "n_true": n_true, "n_false": n_false}

    # True-premise rates
    if n_true > 0:
        true_valid = [r for r in true_rows if r.get("label") not in (EMPTY, JUDGE_ERROR)]
        n_true_valid = len(true_valid)
        m["correct_rate_true"] = count(true_rows, CORRECT) / n_true_valid if n_true_valid else float("nan")
        m["refusal_rate_true"] = count(true_rows, REFUSAL) / n_true_valid if n_true_valid else float("nan")
        m["incorrect_rate_true"] = count(true_rows, INCORRECT) / n_true_valid if n_true_valid else float("nan")
        m["n_correct_true"] = count(true_rows, CORRECT)
        m["n_refusal_true"] = count(true_rows, REFUSAL)
        m["n_incorrect_true"] = count(true_rows, INCORRECT)
        m["n_true_valid"] = n_true_valid
    else:
        m.update({
            "correct_rate_true": float("nan"),
            "refusal_rate_true": float("nan"),
            "incorrect_rate_true": float("nan"),
            "n_correct_true": 0,
            "n_refusal_true": 0,
            "n_incorrect_true": 0,
            "n_true_valid": 0,
        })

    # False-premise rates
    if n_false > 0:
        false_valid = [r for r in false_rows if r.get("label") not in (EMPTY, JUDGE_ERROR)]
        n_false_valid = len(false_valid)
        m["refused_rate_false"] = count(false_rows, REFUSED) / n_false_valid if n_false_valid else float("nan")
        m["corrected_rate_false"] = count(false_rows, CORRECTED) / n_false_valid if n_false_valid else float("nan")
        m["hallucination_rate_false"] = count(false_rows, HALLUCINATED) / n_false_valid if n_false_valid else float("nan")
        m["calibration_success_false"] = (
            count_multi(false_rows, {REFUSED, CORRECTED}) / n_false_valid if n_false_valid else float("nan")
        )
        m["n_refused_false"] = count(false_rows, REFUSED)
        m["n_corrected_false"] = count(false_rows, CORRECTED)
        m["n_hallucinated_false"] = count(false_rows, HALLUCINATED)
        m["n_false_valid"] = n_false_valid
    else:
        m.update({
            "refused_rate_false": float("nan"),
            "corrected_rate_false": float("nan"),
            "hallucination_rate_false": float("nan"),
            "calibration_success_false": float("nan"),
            "n_refused_false": 0,
            "n_corrected_false": 0,
            "n_hallucinated_false": 0,
            "n_false_valid": 0,
        })

    return m


def _compute_recovery_metrics(rows: list[dict]) -> dict:
    """
    Compute Stage 1 recovery metrics from rows that have stage s0 and s1 info.
    """
    triggered_true = [
        r for r in rows
        if r.get("premise") is True and r.get("label") in TRUE_TRIGGER and r.get("label_s1") is not None
    ]
    triggered_false = [
        r for r in rows
        if r.get("premise") is False and r.get("label") in FALSE_TRIGGER and r.get("label_s1") is not None
    ]
    recovered_true = [r for r in triggered_true if r.get("label_s1") == CORRECT]
    recovered_false = [r for r in triggered_false if r.get("label_s1") in {REFUSED, CORRECTED}]

    m: dict = {
        "n_triggered_true": len(triggered_true),
        "n_triggered_false": len(triggered_false),
        "n_recovered_true": len(recovered_true),
        "n_recovered_false": len(recovered_false),
        "recovery_rate_true": len(recovered_true) / len(triggered_true) if triggered_true else float("nan"),
        "recovery_rate_false": len(recovered_false) / len(triggered_false) if triggered_false else float("nan"),
        "recovery_lift_true": float("nan"),
    }

    if triggered_true:
        s0_correct_on_triggered = sum(1 for r in triggered_true if r.get("label") == CORRECT)
        s0_rate = s0_correct_on_triggered / len(triggered_true)
        s1_rate = len(recovered_true) / len(triggered_true)
        m["recovery_lift_true"] = s1_rate - s0_rate
    return m


# ---------------------------------------------------------------------------
# Item-level 3-Way Failure Taxonomy & True vs. Calibrated Suppression
# ---------------------------------------------------------------------------

def _compute_item_level_taxonomy(all_rows: list[dict]) -> dict:
    """
    Pair every item across (permissive, neutral, recovery, conservative) to:
      1. Compute Latent Knowledge Ceiling (Ever-Correct rate across prompts)
      2. Separate True Answer Suppression (proven known -> REFUSAL) from
         Calibrated Ignorance Abstention (never known -> REFUSAL)
      3. Classify all true-premise items into the 3-Way Failure Taxonomy (T9)
      4. Compute McNemar exact p-values for permissive vs. conservative shifts
    """
    by_sid: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in all_rows:
        sid = r.get("sample_id", "")
        lvl = r.get("level", "neutral")
        if sid:
            by_sid[sid][lvl] = r

    true_sids = sorted(
        sid for sid, lvls in by_sid.items()
        if any(row.get("premise") is True for row in lvls.values())
    )
    false_sids = sorted(
        sid for sid, lvls in by_sid.items()
        if any(row.get("premise") is False for row in lvls.values())
    )

    n_true = len(true_sids)
    if n_true == 0:
        return {}

    n_ever_correct = 0
    n_robust_recall = 0
    n_answer_suppression = 0
    n_retrieval_failure = 0
    n_knowledge_loss_calibrated = 0
    n_knowledge_loss_uncalibrated = 0

    # Stage-0 (Neutral) failure breakdown
    s0_failures_total = 0
    s0_fail_suppressed_or_recovered = 0
    s0_fail_calibrated_ignorance = 0
    s0_fail_uncalibrated_ignorance = 0

    # Conservative refusal decomposition
    n_cons_refusal_total = 0
    n_cons_refusal_true_suppression = 0
    n_cons_refusal_calibrated_ignorance = 0

    # McNemar discordant pairs (permissive vs conservative on true premise)
    ref_perm_not_cons = 0
    ref_cons_not_perm = 0
    acc_perm_not_cons = 0
    acc_cons_not_perm = 0

    for sid in true_sids:
        lvls = by_sid[sid]
        l_neut = lvls.get("neutral", {}).get("label")
        l_rec = lvls.get("recovery", {}).get("label")
        l_perm = lvls.get("permissive", {}).get("label")
        l_cons = lvls.get("conservative", {}).get("label")

        ever_correct = any(l == CORRECT for l in (l_neut, l_rec, l_perm, l_cons))
        if ever_correct:
            n_ever_correct += 1

        # McNemar discordant counts (when both perm and cons were evaluated)
        if l_perm is not None and l_cons is not None:
            perm_is_ref = (l_perm == REFUSAL)
            cons_is_ref = (l_cons == REFUSAL)
            if perm_is_ref and not cons_is_ref:
                ref_perm_not_cons += 1
            elif cons_is_ref and not perm_is_ref:
                ref_cons_not_perm += 1

            perm_is_acc = (l_perm == CORRECT)
            cons_is_acc = (l_cons == CORRECT)
            if perm_is_acc and not cons_is_acc:
                acc_perm_not_cons += 1
            elif cons_is_acc and not perm_is_acc:
                acc_cons_not_perm += 1

        # Conservative refusal decomposition
        if l_cons == REFUSAL:
            n_cons_refusal_total += 1
            if ever_correct:
                n_cons_refusal_true_suppression += 1
            else:
                n_cons_refusal_calibrated_ignorance += 1

        # Stage-0 (Neutral) failure decomposition
        if l_neut != CORRECT:
            s0_failures_total += 1
            if ever_correct:
                s0_fail_suppressed_or_recovered += 1
            elif l_cons == REFUSAL or l_neut == REFUSAL:
                s0_fail_calibrated_ignorance += 1
            else:
                s0_fail_uncalibrated_ignorance += 1

        # Full 75-item 3-Way Diagnostic Classification
        if not ever_correct:
            # Knowledge Loss / Absence (never retrieved under any prompt)
            if l_cons == REFUSAL or l_neut == REFUSAL or l_rec == REFUSAL:
                n_knowledge_loss_calibrated += 1
            else:
                n_knowledge_loss_uncalibrated += 1
        else:
            # Knowledge is proven present in the model's parameters!
            if l_neut == CORRECT and l_cons == CORRECT:
                n_robust_recall += 1
            elif l_cons == REFUSAL or l_neut == REFUSAL:
                n_answer_suppression += 1
            else:
                n_retrieval_failure += 1

    # False-premise hallucination prevention count (perm vs cons)
    n_halluc_perm = sum(
        1 for sid in false_sids if by_sid[sid].get("permissive", {}).get("label") == HALLUCINATED
    )
    n_halluc_cons = sum(
        1 for sid in false_sids if by_sid[sid].get("conservative", {}).get("label") == HALLUCINATED
    )
    halluc_prevented = max(0, n_halluc_perm - n_halluc_cons)
    epistemic_efficiency = (
        halluc_prevented / n_cons_refusal_true_suppression
        if n_cons_refusal_true_suppression > 0 else float("inf")
    )

    latent_knowledge_rate = n_ever_correct / n_true
    true_suppression_rate = n_answer_suppression / n_true
    calibrated_abstention_rate = n_knowledge_loss_calibrated / n_true
    true_suppression_share_of_cons_ref = (
        n_cons_refusal_true_suppression / n_cons_refusal_total
        if n_cons_refusal_total > 0 else float("nan")
    )

    return {
        "n_true_items": n_true,
        "n_ever_correct": n_ever_correct,
        "latent_knowledge_rate": latent_knowledge_rate,
        "n_robust_recall": n_robust_recall,
        "robust_recall_rate": n_robust_recall / n_true,
        "n_answer_suppression": n_answer_suppression,
        "true_suppression_rate": true_suppression_rate,
        "n_retrieval_failure": n_retrieval_failure,
        "retrieval_failure_rate": n_retrieval_failure / n_true,
        "n_knowledge_loss_calibrated": n_knowledge_loss_calibrated,
        "calibrated_abstention_rate": calibrated_abstention_rate,
        "n_knowledge_loss_uncalibrated": n_knowledge_loss_uncalibrated,
        "uncalibrated_ignorance_rate": n_knowledge_loss_uncalibrated / n_true,
        "n_cons_refusal_total": n_cons_refusal_total,
        "n_cons_refusal_true_suppression": n_cons_refusal_true_suppression,
        "n_cons_refusal_calibrated_ignorance": n_cons_refusal_calibrated_ignorance,
        "true_suppression_share_of_cons_ref": true_suppression_share_of_cons_ref,
        "s0_failures_total": s0_failures_total,
        "s0_fail_suppressed_or_recovered": s0_fail_suppressed_or_recovered,
        "s0_fail_suppressed_or_recovered_rate": (
            s0_fail_suppressed_or_recovered / s0_failures_total if s0_failures_total else float("nan")
        ),
        "s0_fail_calibrated_ignorance": s0_fail_calibrated_ignorance,
        "s0_fail_uncalibrated_ignorance": s0_fail_uncalibrated_ignorance,
        "hallucinations_prevented_false": halluc_prevented,
        "epistemic_efficiency_ratio": epistemic_efficiency,
        "mcnemar_refusal_p": _mcnemar_exact_p(ref_perm_not_cons, ref_cons_not_perm),
        "mcnemar_accuracy_p": _mcnemar_exact_p(acc_perm_not_cons, acc_cons_not_perm),
    }


def _compute_cot_self_censorship(model_name: str, all_rows: list[dict]) -> list[dict]:
    """
    Audit reasoning traces (`thinking_text`) to find cases where:
      - The model's internal CoT mentions an expected gold answer (or key entity token >=4 chars),
        yet the final output was REFUSAL / INCORRECT / EMPTY, OR
      - The model deliberated about refusing due to the conservative prompt ("prefer silence",
        "only if highly certain") and self-censored.
    """
    audit_rows: list[dict] = []
    for r in all_rows:
        think = (r.get("thinking_text") or "").strip()
        if not think:
            continue
        lbl = r.get("label", "")
        if lbl in (CORRECT, CORRECTED):
            continue
        think_lower = think.lower()
        expected = r.get("expected_answers") or []
        matched_ans = []
        if r.get("premise") is True:
            for ans in expected:
                ans_s = str(ans).strip()
                if len(ans_s) >= 4 and ans_s.lower() in think_lower:
                    matched_ans.append(ans_s)

        caution_markers = [
            m for m in ("prefer silence", "highly certain", "any doubt", "better to refuse", "say i don't know")
            if m in think_lower
        ]
        if matched_ans or caution_markers:
            audit_rows.append({
                "model": model_name,
                "sample_id": r.get("sample_id", ""),
                "dataset": r.get("dataset_name", ""),
                "level": r.get("level", ""),
                "final_label": lbl,
                "gold_in_cot": ", ".join(matched_ans) if matched_ans else "—",
                "caution_triggers_in_cot": ", ".join(caution_markers) if caution_markers else "—",
                "final_response": (r.get("response_text") or "[TRUNCATED_MID_COT]")[:140],
                "cot_excerpt": think[-220:].replace("\n", " "),
            })
    return audit_rows


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(
    model_name: str,
    all_rows: list[dict],
) -> dict:
    """
    Compute all metrics for a model run.
    """
    all_rows = deduplicate_rows(all_rows)
    results: dict = {"model_name": model_name, "slices": {}}

    rows_by_level: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        lvl = r.get("level", "neutral")
        rows_by_level[lvl].append(r)

    for level in ["neutral", "permissive", "conservative"]:
        lvl_rows = rows_by_level.get(level, [])
        m = _compute_group_metrics(lvl_rows)
        results["slices"][f"level_{level}"] = m

    cons_refusal = results["slices"].get("level_conservative", {}).get("refusal_rate_true", float("nan"))
    perm_refusal = results["slices"].get("level_permissive", {}).get("refusal_rate_true", float("nan"))
    try:
        suppression_proxy = cons_refusal - perm_refusal
    except Exception:
        suppression_proxy = float("nan")

    perm_halluc = results["slices"].get("level_permissive", {}).get("hallucination_rate_false", float("nan"))
    cons_halluc = results["slices"].get("level_conservative", {}).get("hallucination_rate_false", float("nan"))
    try:
        induced_halluc_proxy = perm_halluc - cons_halluc
    except Exception:
        induced_halluc_proxy = float("nan")

    results["suppression_proxy"] = suppression_proxy
    results["induced_halluc_proxy"] = induced_halluc_proxy

    neutral_rows = rows_by_level.get("neutral", [])
    recovery_m = _compute_recovery_metrics(neutral_rows)
    results["recovery"] = recovery_m

    # Item-level 3-Way Failure Taxonomy & True vs Calibrated Suppression
    results["taxonomy"] = _compute_item_level_taxonomy(all_rows)
    results["cot_audit"] = _compute_cot_self_censorship(model_name, all_rows)

    # --- Per-dataset slice ---
    rows_by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in rows_by_level.get("neutral", []):
        ds = r.get("dataset_name", "unknown")
        rows_by_dataset[ds].append(r)
    results["slices"]["by_dataset"] = {
        ds: _compute_group_metrics(rs) for ds, rs in rows_by_dataset.items()
    }

    # --- Per-eval_dimension slice ---
    rows_by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in rows_by_level.get("neutral", []):
        dim = r.get("eval_dimension", "unknown")
        rows_by_dim[dim].append(r)
    results["slices"]["by_eval_dimension"] = {
        dim: _compute_group_metrics(rs) for dim, rs in rows_by_dim.items()
    }

    # --- By premise ---
    rows_true = [r for r in rows_by_level.get("neutral", []) if r.get("premise") is True]
    rows_false = [r for r in rows_by_level.get("neutral", []) if r.get("premise") is False]
    results["slices"]["by_premise"] = {
        "true": _compute_group_metrics(rows_true),
        "false": _compute_group_metrics(rows_false),
    }

    # --- By hop ---
    rows_by_hop: dict[str, list[dict]] = defaultdict(list)
    for r in rows_by_level.get("neutral", []):
        rows_by_hop[r.get("hop", "single")].append(r)
    results["slices"]["by_hop"] = {
        hop: _compute_group_metrics(rs) for hop, rs in rows_by_hop.items()
    }

    # --- By answer_type ---
    rows_by_at: dict[str, list[dict]] = defaultdict(list)
    for r in rows_by_level.get("neutral", []):
        at = r.get("answer_type", "unknown") or "unknown"
        rows_by_at[at].append(r)
    results["slices"]["by_answer_type"] = {
        at: _compute_group_metrics(rs) for at, rs in rows_by_at.items()
    }

    # --- Cross: premise x level ---
    results["slices"]["premise_x_level"] = {}
    for level in ["neutral", "permissive", "conservative"]:
        lvl_rows = rows_by_level.get(level, [])
        true_r = [r for r in lvl_rows if r.get("premise") is True]
        false_r = [r for r in lvl_rows if r.get("premise") is False]
        results["slices"]["premise_x_level"][level] = {
            "true": _compute_group_metrics(true_r),
            "false": _compute_group_metrics(false_r),
        }

    # --- Cross: dataset x premise (neutral) ---
    results["slices"]["dataset_x_premise"] = {}
    for ds, ds_rows in rows_by_dataset.items():
        true_r = [r for r in ds_rows if r.get("premise") is True]
        false_r = [r for r in ds_rows if r.get("premise") is False]
        results["slices"]["dataset_x_premise"][ds] = {
            "true": _compute_group_metrics(true_r),
            "false": _compute_group_metrics(false_r),
        }

    # --- Cross: model x level ---
    results["slices"]["model_x_level"] = {
        level: results["slices"].get(f"level_{level}", {})
        for level in ["neutral", "permissive", "conservative"]
    }

    # Judge health
    results["judge_health"] = _compute_judge_health(all_rows)

    return results


def _compute_judge_health(all_rows: list[dict]) -> dict:
    """
    Summarize judge call health metrics.
    Computes mean and median latency ONLY across rows where the judge was actually called
    (latency > 0 or judge_called is True), avoiding dilution from 0ms template refusals.
    """
    judge_calls = sum(
        1 for r in all_rows if r.get("judge_called") is True or (r.get("judge_api_calls") or 0) > 0
    )
    parse_failures = sum(1 for r in all_rows if r.get("label") == PARSE_ERROR)
    judge_errors = sum(1 for r in all_rows if r.get("label") == JUDGE_ERROR)
    latencies = [
        float(r.get("judge_latency_ms", 0.0))
        for r in all_rows
        if r.get("judge_latency_ms") is not None
        and (r.get("judge_called") is True or float(r.get("judge_latency_ms", 0.0)) > 0.0)
    ]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    median_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    return {
        "judge_calls": judge_calls,
        "parse_failures": parse_failures,
        "judge_errors": judge_errors,
        "mean_latency_ms": mean_latency,
        "median_latency_ms": median_latency,
    }


def flatten_to_dataframes(model_results: dict, all_rows: list[dict] | None = None) -> dict[str, pd.DataFrame]:
    """
    Convert nested metrics dict to flat DataFrames for CSV export (T1-T10).
    """
    dfs: dict[str, pd.DataFrame] = {}
    if all_rows is None:
        all_rows = []
    all_rows = deduplicate_rows(all_rows)

    model_name = model_results.get("model_name", "unknown")
    slices = model_results.get("slices", {})
    tax = model_results.get("taxonomy", {})

    # T1: per-model @ neutral (now enriched with latent_knowledge_rate & true_suppression_rate)
    t1_row = {"model": model_name}
    neutral_m = slices.get("level_neutral", {})
    t1_row.update({
        "latent_knowledge_rate": tax.get("latent_knowledge_rate"),
        "correct_rate_true": neutral_m.get("correct_rate_true"),
        "refusal_rate_true": neutral_m.get("refusal_rate_true"),
        "incorrect_rate_true": neutral_m.get("incorrect_rate_true"),
        "calibration_success_false": neutral_m.get("calibration_success_false"),
        "hallucination_rate_false": neutral_m.get("hallucination_rate_false"),
        "recovery_rate_true": model_results.get("recovery", {}).get("recovery_rate_true"),
        "recovery_rate_false": model_results.get("recovery", {}).get("recovery_rate_false"),
        "suppression_proxy": model_results.get("suppression_proxy"),
        "true_suppression_rate": tax.get("true_suppression_rate"),
        "calibrated_abstention_rate": tax.get("calibrated_abstention_rate"),
        "induced_halluc_proxy": model_results.get("induced_halluc_proxy"),
        "mcnemar_refusal_p": tax.get("mcnemar_refusal_p"),
        "n_total": neutral_m.get("n_total"),
    })
    dfs["T1_per_model_neutral"] = pd.DataFrame([t1_row])

    # T3: per-model x level pressure table
    t3_rows = []
    for level in ["permissive", "neutral", "conservative"]:
        lm = slices.get(f"level_{level}", {})
        t3_rows.append({
            "model": model_name,
            "level": level,
            "refusal_rate_true": lm.get("refusal_rate_true"),
            "correct_rate_true": lm.get("correct_rate_true"),
            "incorrect_rate_true": lm.get("incorrect_rate_true"),
            "hallucination_rate_false": lm.get("hallucination_rate_false"),
            "calibration_success_false": lm.get("calibration_success_false"),
        })
    dfs["T3_pressure_table"] = pd.DataFrame(t3_rows)

    # T7: Stage-1 detail
    rec = model_results.get("recovery", {})
    t7_row = {
        "model": model_name,
        "n_triggered_true": rec.get("n_triggered_true"),
        "n_triggered_false": rec.get("n_triggered_false"),
        "n_recovered_true": rec.get("n_recovered_true"),
        "n_recovered_false": rec.get("n_recovered_false"),
        "recovery_rate_true": rec.get("recovery_rate_true"),
        "recovery_rate_false": rec.get("recovery_rate_false"),
        "recovery_lift_true": rec.get("recovery_lift_true"),
    }
    dfs["T7_stage1_detail"] = pd.DataFrame([t7_row])

    # T8: judge health
    jh = model_results.get("judge_health", {})
    t8_row = {"model": model_name, **jh}
    dfs["T8_judge_health"] = pd.DataFrame([t8_row])

    # T9: 3-Way Diagnostic Failure Taxonomy & True vs Calibrated Suppression
    if tax:
        t9_row = {
            "model": model_name,
            "latent_knowledge_rate": tax.get("latent_knowledge_rate"),
            "robust_recall_rate": tax.get("robust_recall_rate"),
            "true_suppression_rate": tax.get("true_suppression_rate"),
            "retrieval_failure_rate": tax.get("retrieval_failure_rate"),
            "calibrated_abstention_rate": tax.get("calibrated_abstention_rate"),
            "uncalibrated_ignorance_rate": tax.get("uncalibrated_ignorance_rate"),
            "true_suppression_share_of_cons_ref": tax.get("true_suppression_share_of_cons_ref"),
            "s0_fail_recovered_or_suppressed_rate": tax.get("s0_fail_suppressed_or_recovered_rate"),
            "epistemic_efficiency_ratio": tax.get("epistemic_efficiency_ratio"),
            "mcnemar_refusal_p": tax.get("mcnemar_refusal_p"),
            "mcnemar_accuracy_p": tax.get("mcnemar_accuracy_p"),
        }
        dfs["T9_failure_taxonomy"] = pd.DataFrame([t9_row])

    # T10: Chain-of-Thought Self-Censorship Audit
    cot_audit = model_results.get("cot_audit", [])
    if cot_audit:
        dfs["T10_cot_self_censorship"] = pd.DataFrame(cot_audit)

    # T2: By-dataset
    ds_rows = []
    for ds, dm in slices.get("by_dataset", {}).items():
        ds_rows.append({
            "model": model_name,
            "dataset": ds,
            "correct_rate_true": dm.get("correct_rate_true"),
            "calibration_success_false": dm.get("calibration_success_false"),
            "n_true": dm.get("n_true"),
            "n_false": dm.get("n_false"),
        })
    dfs["T2_by_dataset"] = pd.DataFrame(ds_rows)

    # T4: By answer_type
    at_rows = []
    for at, atm in slices.get("by_answer_type", {}).items():
        at_rows.append({
            "model": model_name,
            "answer_type": at,
            "correct_rate_true": atm.get("correct_rate_true"),
            "n_true": atm.get("n_true"),
        })
    dfs["T4_by_answer_type"] = pd.DataFrame(at_rows)

    # T5: by premise x hop
    t5_rows = []
    neutral_rows_for_t5 = [r for r in all_rows if r.get("level") == "neutral"]
    premise_hop_groups: dict[tuple, list[dict]] = {}
    for r in neutral_rows_for_t5:
        key = (r.get("premise"), r.get("hop", "single"))
        premise_hop_groups.setdefault(key, []).append(r)

    if premise_hop_groups:
        for (premise_val, hop_val), ph_rows in sorted(premise_hop_groups.items()):
            m_ph = _compute_group_metrics(ph_rows) if ph_rows else {}
            if premise_val is True:
                accuracy_metric = m_ph.get("correct_rate_true")
                metric_name = "correct_rate_true"
            else:
                accuracy_metric = m_ph.get("calibration_success_false")
                metric_name = "calibration_success_false"
            t5_rows.append({
                "model": model_name,
                "premise": str(premise_val),
                "hop": hop_val,
                "accuracy_metric": metric_name,
                "accuracy_value": accuracy_metric,
                "n": m_ph.get("n_total", 0),
            })
    else:
        for hop_val, hm in slices.get("by_hop", {}).items():
            t5_rows.append({
                "model": model_name,
                "premise": "all",
                "hop": hop_val,
                "accuracy_metric": "correct_rate_true",
                "accuracy_value": hm.get("correct_rate_true"),
                "n": hm.get("n_total", 0),
            })
    dfs["T5_premise_x_hop"] = pd.DataFrame(t5_rows)

    return dfs
