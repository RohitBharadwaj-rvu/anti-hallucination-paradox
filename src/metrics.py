"""
src/metrics.py — Exact metric formulas per spec §12.
Computes per-model and per-slice metrics from graded rows.
"""
from collections import defaultdict
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

    # Split by premise
    true_rows = [r for r in rows if r.get("premise") is True]
    false_rows = [r for r in rows if r.get("premise") is False]
    n_true = len(true_rows)
    n_false = len(false_rows)

    def rate(rows_subset: list[dict], label: str) -> float:
        valid = [r for r in rows_subset if r.get("label") not in (EMPTY, JUDGE_ERROR)]
        if not valid:
            return float("nan")
        return sum(1 for r in valid if r.get("label") == label) / len(valid)

    def multi_rate(rows_subset: list[dict], labels: set) -> float:
        valid = [r for r in rows_subset if r.get("label") not in (EMPTY, JUDGE_ERROR)]
        if not valid:
            return float("nan")
        return sum(1 for r in valid if r.get("label") in labels) / len(valid)

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
    Rows must include: sample_id, premise, label (s0), label_s1 (or None).
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
        "recovery_lift_true": float("nan"),  # computed below
    }

    # Recovery lift = correct_rate_true in s1 on triggered - correct_rate_true in s0 on triggered
    if triggered_true:
        s0_correct_on_triggered = sum(1 for r in triggered_true if r.get("label") == CORRECT)
        s0_rate = s0_correct_on_triggered / len(triggered_true)  # usually 0 since triggered by INCORRECT
        s1_rate = len(recovered_true) / len(triggered_true)
        m["recovery_lift_true"] = s1_rate - s0_rate
    return m


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(
    model_name: str,
    all_rows: list[dict],
) -> dict:
    """
    Compute all metrics for a model run.
    all_rows: list of graded row dicts with fields:
      sample_id, dataset_name, premise, hop, answer_type, eval_dimension, category,
      label (s0 neutral label), label_s1 (or None), label_permissive, label_conservative,
      stage, level, model_name
    Returns nested dict of slices.
    """
    results: dict = {"model_name": model_name, "slices": {}}

    # Build per-level row groups (using s0 label for neutral, s2 labels for perm/cons)
    # We work with a "per-item summary" for cross-level analysis
    # and "per-level groups" for level-specific rates

    # Group rows by level
    rows_by_level: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        lvl = r.get("level", "neutral")
        rows_by_level[lvl].append(r)

    # Per-level group metrics
    for level in ["neutral", "permissive", "conservative"]:
        lvl_rows = rows_by_level.get(level, [])
        m = _compute_group_metrics(lvl_rows)
        results["slices"][f"level_{level}"] = m

    # Suppression proxy: refusal_rate_true[conservative] - refusal_rate_true[permissive]
    cons_refusal = results["slices"].get("level_conservative", {}).get("refusal_rate_true", float("nan"))
    perm_refusal = results["slices"].get("level_permissive", {}).get("refusal_rate_true", float("nan"))
    try:
        suppression_proxy = cons_refusal - perm_refusal
    except Exception:
        suppression_proxy = float("nan")

    # Induced hallucination proxy: hallucination_rate_false[permissive] - hallucination_rate_false[conservative]
    perm_halluc = results["slices"].get("level_permissive", {}).get("hallucination_rate_false", float("nan"))
    cons_halluc = results["slices"].get("level_conservative", {}).get("hallucination_rate_false", float("nan"))
    try:
        induced_halluc_proxy = perm_halluc - cons_halluc
    except Exception:
        induced_halluc_proxy = float("nan")

    results["suppression_proxy"] = suppression_proxy
    results["induced_halluc_proxy"] = induced_halluc_proxy

    # Recovery metrics (using s0 neutral rows enriched with s1 labels)
    neutral_rows = rows_by_level.get("neutral", [])
    recovery_m = _compute_recovery_metrics(neutral_rows)
    results["recovery"] = recovery_m

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

    # --- Cross: model x level (already computed per level above; store summary) ---
    results["slices"]["model_x_level"] = {
        level: results["slices"].get(f"level_{level}", {})
        for level in ["neutral", "permissive", "conservative"]
    }

    # Judge health
    results["judge_health"] = _compute_judge_health(all_rows)

    return results


def _compute_judge_health(all_rows: list[dict]) -> dict:
    """Summarize judge call health metrics."""
    judge_calls = sum(1 for r in all_rows if r.get("judge_called") is True)
    parse_failures = sum(1 for r in all_rows if r.get("label") == PARSE_ERROR)
    judge_errors = sum(1 for r in all_rows if r.get("label") == JUDGE_ERROR)
    latencies = [r.get("judge_latency_ms", 0) for r in all_rows if r.get("judge_latency_ms") is not None]
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
    Convert nested metrics dict to flat DataFrames for CSV export.
    all_rows: optional list of raw graded rows used for T5 premise x hop cross-product.
    Returns: {table_name: DataFrame}
    """
    dfs: dict[str, pd.DataFrame] = {}
    if all_rows is None:
        all_rows = []

    model_name = model_results.get("model_name", "unknown")
    slices = model_results.get("slices", {})

    # T1: per-model @ neutral
    t1_row = {"model": model_name}
    neutral_m = slices.get("level_neutral", {})
    t1_row.update({
        "correct_rate_true": neutral_m.get("correct_rate_true"),
        "refusal_rate_true": neutral_m.get("refusal_rate_true"),
        "incorrect_rate_true": neutral_m.get("incorrect_rate_true"),
        "calibration_success_false": neutral_m.get("calibration_success_false"),
        "hallucination_rate_false": neutral_m.get("hallucination_rate_false"),
        "recovery_rate_true": model_results.get("recovery", {}).get("recovery_rate_true"),
        "recovery_rate_false": model_results.get("recovery", {}).get("recovery_rate_false"),
        "suppression_proxy": model_results.get("suppression_proxy"),
        "induced_halluc_proxy": model_results.get("induced_halluc_proxy"),
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

    # By-dataset
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

    # By answer_type
    at_rows = []
    for at, atm in slices.get("by_answer_type", {}).items():
        at_rows.append({
            "model": model_name,
            "answer_type": at,
            "correct_rate_true": atm.get("correct_rate_true"),
            "n_true": atm.get("n_true"),
        })
    dfs["T4_by_answer_type"] = pd.DataFrame(at_rows)

    # T5: by premise x hop — the correct cross-product per spec §13
    t5_rows = []
    neutral_rows_for_t5 = [
        r for r in all_rows
        if r.get("level") == "neutral"
        # use all neutral rows (sourced from slices or raw rows if available)
    ]
    # Attempt to use raw rows from slices; fall back to by_hop only if no premise breakdown
    premise_hop_groups: dict[tuple, list[dict]] = {}
    for r in neutral_rows_for_t5:
        key = (r.get("premise"), r.get("hop", "single"))
        premise_hop_groups.setdefault(key, []).append(r)

    if premise_hop_groups:
        for (premise_val, hop_val), ph_rows in sorted(premise_hop_groups.items()):
            m_ph = _compute_group_metrics(ph_rows) if ph_rows else {}
            # Choose the appropriate accuracy metric by premise
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
        # Fallback: by hop only (no neutral rows available yet)
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

