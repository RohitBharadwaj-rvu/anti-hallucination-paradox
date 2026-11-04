"""
src/cost.py — Dry-run / post-micro cost estimator.
Judge is always $0 (local LM Studio).
Cloud token counts are APPROXIMATE (char/4 fallback; never exact).
"""
from typing import Any

from src.utils import get_logger

logger = get_logger("ahp.cost")

# Default assumptions for cost estimation
DEFAULT_AVG_PROMPT_IN = 250    # tokens per target prompt (system + question)
DEFAULT_AVG_COMPLETION_OUT = 30  # tokens per target response


def estimate(
    items: list[dict],
    settings: Any,
    assumed_true_fail: float = 0.6,
    assumed_false_halluc: float = 0.4,
    avg_prompt_in: int = DEFAULT_AVG_PROMPT_IN,
    avg_completion_out: int = DEFAULT_AVG_COMPLETION_OUT,
) -> dict:
    """
    Estimate expected target and judge call counts and approximate cloud token usage.

    Returns a dict with:
      n_items, n_true, n_false,
      s0_calls, s1_calls_expected, s2_calls,
      total_target_calls,
      judge_calls_expected,
      approx_cloud_tokens_in, approx_cloud_tokens_out, approx_cloud_tokens_total,
      judge_tokens_total (local, $0),
      budget_nim, budget_fireworks,
    """
    n = len(items)
    n_true = sum(1 for it in items if it["premise"])
    n_false = n - n_true

    # Stage 0: every item
    s0_calls = n
    # Stage 1: only triggered items
    s1_trigger_true = int(n_true * assumed_true_fail)
    s1_trigger_false = int(n_false * assumed_false_halluc)
    s1_calls = s1_trigger_true + s1_trigger_false
    # Stage 2: 2 extra levels (permissive + conservative) for every item
    s2_calls = 2 * n

    total_target_calls = s0_calls + s1_calls + s2_calls

    # Judge calls ≈ target calls minus assumed template-refusal fraction (~0.2 of true-premise)
    assumed_template_refusal = 0.20
    expected_no_judge_s0 = int(n_true * assumed_template_refusal)
    judge_calls_expected = max(0, total_target_calls - expected_no_judge_s0)

    # Approximate cloud tokens (target only; judge is local)
    approx_cloud_tokens_in = total_target_calls * avg_prompt_in
    approx_cloud_tokens_out = total_target_calls * avg_completion_out
    approx_cloud_tokens_total = approx_cloud_tokens_in + approx_cloud_tokens_out

    # Judge tokens (local, $0)
    judge_tokens_total = judge_calls_expected * (avg_prompt_in + 8)  # judge prompt + 1 token out

    # Budget fractions (approximate)
    budget_cfg = getattr(settings, "budget", {})
    nim_limit = budget_cfg.get("nim_max_tokens", 900_000)
    fw_limit = budget_cfg.get("fireworks_max_tokens", 2_000_000)
    or_limit = budget_cfg.get("openrouter_max_tokens", 5_000_000)

    result = {
        "n_items": n,
        "n_true": n_true,
        "n_false": n_false,
        "s0_calls": s0_calls,
        "s1_calls_expected": s1_calls,
        "s2_calls": s2_calls,
        "total_target_calls": total_target_calls,
        "judge_calls_expected": judge_calls_expected,
        "approx_cloud_tokens_in": approx_cloud_tokens_in,
        "approx_cloud_tokens_out": approx_cloud_tokens_out,
        "approx_cloud_tokens_total": approx_cloud_tokens_total,
        "judge_tokens_total": judge_tokens_total,
        "judge_cost_usd": 0.0,  # always local
        "budget_nim_limit": nim_limit,
        "budget_fireworks_limit": fw_limit,
        "budget_openrouter_limit": or_limit,
        "budget_nim_fraction": approx_cloud_tokens_total / nim_limit if nim_limit > 0 else 0,
        "budget_fw_fraction": approx_cloud_tokens_total / fw_limit if fw_limit > 0 else 0,
        "budget_or_fraction": approx_cloud_tokens_total / or_limit if or_limit > 0 else 0,
    }
    return result


def reestimate_from_observed(
    items: list[dict],
    settings: Any,
    observed_true_fail_rate: float,
    observed_false_halluc_rate: float,
    mean_judge_latency_ms: float,
    mean_target_latency_ms: float,
) -> dict:
    """
    Recompute cost estimate using observed rates from a micro-test.
    Also computes an ETA estimate for a full run.
    """
    base = estimate(
        items,
        settings,
        assumed_true_fail=observed_true_fail_rate,
        assumed_false_halluc=observed_false_halluc_rate,
    )

    n_target = base["total_target_calls"]
    n_judge = base["judge_calls_expected"]

    eta_target_s = (n_target * mean_target_latency_ms) / 1000
    eta_judge_s = (n_judge * mean_judge_latency_ms) / 1000
    eta_total_s = eta_target_s + eta_judge_s

    base["mean_judge_latency_ms"] = mean_judge_latency_ms
    base["mean_target_latency_ms"] = mean_target_latency_ms
    base["eta_target_s"] = eta_target_s
    base["eta_judge_s"] = eta_judge_s
    base["eta_total_s"] = eta_total_s
    base["eta_note"] = (
        f"Estimated full-run wall-clock: "
        f"{eta_total_s/60:.1f} min "
        f"(judge {eta_judge_s/60:.1f} min + target {eta_target_s/60:.1f} min). "
        "Local judge is the bottleneck."
    )
    return base


def print_cost_table(estimate_result: dict, run_name: str = "all") -> None:
    """Pretty-print the cost estimate table to console (ASCII-safe for Windows)."""
    r = estimate_result
    SEP = "=" * 66
    print()
    print(SEP)
    print(f"  COST ESTIMATE -- Run: {run_name}")
    print(SEP)
    print(f"  Items:                  {r['n_items']:>5} ({r['n_true']:>3} true / {r['n_false']:>2} false)")
    print(f"  Stage 0 calls:          {r['s0_calls']:>6}")
    print(f"  Stage 1 calls (est.):   {r['s1_calls_expected']:>6}")
    print(f"  Stage 2 calls:          {r['s2_calls']:>6}")
    print(f"  TOTAL target calls:     {r['total_target_calls']:>6}")
    print(f"  Judge calls (est.):     {r['judge_calls_expected']:>6}  [local, $0.00]")
    print(f"  Approx cloud tokens:    {r['approx_cloud_tokens_total']:>9,} (in+out)")
    print(f"  NIM budget fraction:    {r['budget_nim_fraction']:>8.1%}")
    print(f"  Fireworks budget frac:  {r['budget_fw_fraction']:>8.1%}")
    print(f"  OpenRouter budget frac: {r.get('budget_or_fraction', 0):>8.1%}")
    print(f"  JUDGE cost:             $0.00 (local Bionic / LM Studio)")
    if "eta_note" in r:
        print(f"  ---")
        print(f"  Mean judge latency:     {r.get('mean_judge_latency_ms', 0):>8.1f} ms")
        print(f"  Mean target latency:    {r.get('mean_target_latency_ms', 0):>8.1f} ms")
        print(f"  ETA (full run):         {r.get('eta_total_s', 0)/60:>8.1f} min")
        print(f"  Local judge is the wall-clock bottleneck.")
    print(SEP)
    print()
