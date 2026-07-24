"""
src/orchestrator.py — Stage ordering, modes, neutral-reuse, checkpoint/resume.
Implements the full pipeline per spec §11.

Stages:
  Stage 0 / Stage-2 NEUTRAL: level=neutral, prompt=TARGET_STANDARD
  Stage 1 (triggered):       level=recovery, prompt=TARGET_RECOVERY
  Stage 2 PERMISSIVE:        level=permissive, prompt=TARGET_PERMISSIVE
  Stage 2 CONSERVATIVE:      level=conservative, prompt=TARGET_CONSERVATIVE

Trigger conditions:
  True premise:  TRIGGER = {REFUSAL, INCORRECT}
  False premise: TRIGGER = {HALLUCINATED}
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from src.utils import get_logger, make_key, load_checkpoint, append_checkpoint, now_iso, ensure_dirs
from src.loader import load as load_data
from src.prompts import (
    TARGET_STANDARD, TARGET_RECOVERY, TARGET_PERMISSIVE, TARGET_CONSERVATIVE
)
from src.grader import (
    grade,
    REFUSAL, INCORRECT, HALLUCINATED,
    EMPTY, JUDGE_ERROR,
    TRUE_TRIGGER, FALSE_TRIGGER,
    CORRECT, REFUSED, CORRECTED,
)
from src.api_router import APIRouter, BudgetExceeded
from src import metrics as metrics_mod
from src import reporter as reporter_mod
from src.cost import estimate, reestimate_from_observed, print_cost_table
from src.env_loader import get_base_url

logger = get_logger("ahp.orchestrator")

# ---------------------------------------------------------------------------
# Settings object adapter (converts yaml dict to attribute access)
# ---------------------------------------------------------------------------

class _Settings:
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _build_settings(raw: dict) -> _Settings:
    return _Settings(raw)


# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------

def _checkpoint_path(run_name: str, stage: str, checkpoint_dir: str) -> str:
    return str(Path(checkpoint_dir) / f"{run_name}__{stage}.jsonl")


def _graded_path(run_name: str, graded_dir: str) -> str:
    return str(Path(graded_dir) / f"{run_name}__graded.jsonl")


def _audit_prompt_path(run_name: str, audit_dir: str) -> str:
    return str(Path(audit_dir) / f"{run_name}__prompts_audit.jsonl")


# ---------------------------------------------------------------------------
# DRY mode: print ALL 100 rendered prompts per stage + cost, no network
# ---------------------------------------------------------------------------

def run_dry(
    settings: _Settings,
    runs: list[dict],
    data_path: str,
) -> None:
    """Print ALL rendered prompts + cost estimate. ZERO network or LM Studio calls."""
    items = load_data(data_path)

    for run_cfg in runs:
        run_name = run_cfg["name"]
        print(f"\n{'='*70}")
        print(f"DRY RUN: {run_name}")
        print(f"  Target: {run_cfg['target']['provider']} / {run_cfg['target']['model_id']}")
        print(f"  Judge:  {run_cfg['judge']['provider']} / {run_cfg['judge']['model_id']} [LOCAL, $0]")
        print(f"{'='*70}\n")

        stages = [
            ("neutral", TARGET_STANDARD),
            ("recovery", TARGET_RECOVERY),
            ("permissive", TARGET_PERMISSIVE),
            ("conservative", TARGET_CONSERVATIVE),
        ]

        # Print ALL 100 rendered prompts per stage (spec §18: "prints 100 rendered prompts per stage")
        for stage_name, system_prompt in stages:
            print(f"\n{'--- Stage: ' + stage_name + ' ---':^70}")
            print(f"  system: {system_prompt[:120]}...")
            print()
            for i, item in enumerate(items):
                # Only the question goes to the target; no metadata/answers leakage
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["question"]},
                ]
                print(
                    f"  [{i+1:03d}] {item['sample_id']} ({item['dataset_name']}, "
                    f"premise={item['premise']}, hop={item['hop']})"
                )
                print(f"        Q: {item['question'][:100]}")
                # Verify no metadata leakage: user message must only contain the question
                user_msg = msgs[1]["content"]
                assert user_msg == item["question"], (
                    f"LEAKAGE DETECTED: user message for {item['sample_id']} contains more than question!"
                )

        print(f"\n--- Total items: {len(items)} rendered across {len(stages)} stages ---\n")

        # Cost estimate (judge = $0 local)
        cost = estimate(items, settings)
        print_cost_table(cost, run_name=run_name)

    print("\nDRY RUN COMPLETE -- No network calls made, LM Studio NOT contacted.\n")


# ---------------------------------------------------------------------------
# Core per-target-model pipeline
# ---------------------------------------------------------------------------

def run_one_target(
    run_cfg: dict,
    settings: _Settings,
    mode: str,
    data_path: str,
    checkpoint_dir: str,
    graded_dir: str,
    output_dir: str,
    no_resume: bool = False,
    limit: int | None = None,
) -> tuple[list[dict], dict | None, bool, APIRouter]:
    """
    Run the full pipeline for one target model config.
    Returns (all_rows, model_metrics_dict, is_partial, router).
    Raises LMStudioNotReachable / LMStudioNoModelLoaded with sys.exit(1) semantics
    (re-raised to caller, which maps to exit code 1).
    """
    run_name = run_cfg["name"]
    target_cfg = run_cfg["target"]
    judge_cfg = run_cfg["judge"]
    start_time = now_iso()

    logger.info("Starting run: %s (mode=%s)", run_name, mode)

    # Load items
    all_items = load_data(data_path)
    micro_ids = set(
        settings.micro_test.get("sample_ids", [])
        if hasattr(settings, "micro_test") else []
    )

    if mode == "micro":
        items = [it for it in all_items if it["sample_id"] in micro_ids]
        logger.info("Micro mode: %d items selected", len(items))
    else:
        items = all_items
        if limit is not None:
            items = items[:limit]
            logger.info("Limit applied: %d items", len(items))

    # Build router
    router = APIRouter(settings, run_cfg)

    # Pre-flight (lmstudio only; raises with clear message if not reachable)
    # These exceptions propagate to main() which exits with code 1.
    if judge_cfg.get("provider") == "lmstudio":
        router.preflight_lmstudio()

    audit_dir = str(Path(output_dir) / "audit")
    ensure_dirs(audit_dir)
    audit_prompt_path = _audit_prompt_path(run_name, audit_dir)

    # --no-resume: clear existing checkpoint/graded/audit files to avoid duplicates
    if no_resume:
        for cp in [
            _checkpoint_path(run_name, "s0", checkpoint_dir),
            _checkpoint_path(run_name, "s1", checkpoint_dir),
            _checkpoint_path(run_name, "s2", checkpoint_dir),
            _graded_path(run_name, graded_dir),
            audit_prompt_path,
        ]:
            p = Path(cp)
            if p.exists():
                p.unlink()
                logger.info("--no-resume: deleted %s", cp)

    # Load checkpoints
    cp_s0 = _checkpoint_path(run_name, "s0", checkpoint_dir)
    cp_s1 = _checkpoint_path(run_name, "s1", checkpoint_dir)
    cp_s2 = _checkpoint_path(run_name, "s2", checkpoint_dir)

    completed_keys: set = set()
    if not no_resume:
        for cp in [cp_s0, cp_s1, cp_s2]:
            completed_keys |= load_checkpoint(cp)
        if completed_keys:
            logger.info("Resume: %d keys already completed", len(completed_keys))
        else:
            logger.info("No existing checkpoints; starting fresh.")

    graded_path = _graded_path(run_name, graded_dir)

    # Row accumulator (all graded rows, including resumed ones from graded file)
    all_rows: list[dict] = []
    if not no_resume and Path(graded_path).exists():
        from src.utils import load_jsonl
        all_rows = load_jsonl(graded_path)
        logger.info("Loaded %d existing graded rows from %s", len(all_rows), graded_path)

    # Index existing rows by sample_id/level for neutral-reuse lookup
    existing_neutral: dict[str, dict] = {}
    for r in all_rows:
        if r.get("level") == "neutral":
            existing_neutral[r.get("sample_id", "")] = r

    is_partial = False
    total_skipped = 0

    # -----------------------------------------------------------------------
    # Main per-item loop
    # -----------------------------------------------------------------------
    for item in items:
        sid = item["sample_id"]

        # ---- Stage 0 / Stage-2 NEUTRAL ----
        key_s0 = make_key(run_name, sid, "s0", "neutral")
        if key_s0 in completed_keys:
            existing_row = existing_neutral.get(sid)
            label0 = existing_row.get("label") if existing_row else None
            resp0 = existing_row.get("response_text", "") if existing_row else ""
            total_skipped += 1
        else:
            messages = [
                {"role": "system", "content": TARGET_STANDARD},
                {"role": "user", "content": item["question"]},  # ONLY question, no metadata
            ]
            # Write raw prompt audit (verify no metadata leakage)
            _write_prompt_audit(audit_prompt_path, run_name, sid, "neutral", messages)
            try:
                resp0, usage0 = router.call_target(messages)
                # Empty retry once
                if resp0.strip() == "":
                    logger.debug("Empty response for %s (s0), retrying...", sid)
                    resp0, usage0 = router.call_target(messages)
            except BudgetExceeded as e:
                logger.error("Budget exceeded: %s", e)
                is_partial = True
                break
            except Exception as e:
                logger.error("Target call failed for %s s0: %s", sid, e)
                resp0 = ""
                usage0 = {}

            judge_calls_before0 = router.total_judge_calls()
            label0, judge_latency_ms0 = grade(router, item, resp0)
            judge_api_calls0 = router.total_judge_calls() - judge_calls_before0
            row0 = _build_row(
                run_name, item, "s0", "neutral", resp0, label0, usage0, judge_latency_ms0, mode,
                judge_api_calls0,
            )
            all_rows.append(row0)
            existing_neutral[sid] = row0
            append_checkpoint(cp_s0, {**row0, "key": key_s0})
            append_checkpoint(graded_path, row0)
            completed_keys.add(key_s0)

        # ---- Stage 1 (triggered) ----
        trigger = TRUE_TRIGGER if item["premise"] else FALSE_TRIGGER
        should_trigger = label0 in trigger if label0 else False

        label1 = None
        resp1 = ""
        if should_trigger:
            key_s1 = make_key(run_name, sid, "s1", "recovery")
            if key_s1 in completed_keys:
                for r in all_rows:
                    if r.get("sample_id") == sid and r.get("level") == "recovery":
                        label1 = r.get("label")
                        resp1 = r.get("response_text", "")
                        break
                total_skipped += 1
            else:
                messages1 = [
                    {"role": "system", "content": TARGET_RECOVERY},
                    {"role": "user", "content": item["question"]},
                ]
                _write_prompt_audit(audit_prompt_path, run_name, sid, "recovery", messages1)
                try:
                    resp1, usage1 = router.call_target(messages1)
                    if resp1.strip() == "":
                        resp1, usage1 = router.call_target(messages1)
                except BudgetExceeded as e:
                    logger.error("Budget exceeded: %s", e)
                    is_partial = True
                    break
                except Exception as e:
                    logger.error("Target call failed for %s s1: %s", sid, e)
                    resp1 = ""
                    usage1 = {}

                judge_calls_before1 = router.total_judge_calls()
                label1, judge_latency_ms1 = grade(router, item, resp1)
                judge_api_calls1 = router.total_judge_calls() - judge_calls_before1
                row1 = _build_row(
                    run_name, item, "s1", "recovery", resp1, label1, usage1, judge_latency_ms1, mode,
                    judge_api_calls1,
                )
                all_rows.append(row1)
                append_checkpoint(cp_s1, {**row1, "key": key_s1})
                append_checkpoint(graded_path, row1)
                completed_keys.add(key_s1)

        # ---- Stage 2: PERMISSIVE + CONSERVATIVE ----
        for level, prompt in [("permissive", TARGET_PERMISSIVE), ("conservative", TARGET_CONSERVATIVE)]:
            key_s2 = make_key(run_name, sid, "s2", level)
            if key_s2 in completed_keys:
                total_skipped += 1
                continue
            messages2 = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item["question"]},
            ]
            _write_prompt_audit(audit_prompt_path, run_name, sid, level, messages2)
            try:
                resp2, usage2 = router.call_target(messages2)
                if resp2.strip() == "":
                    resp2, usage2 = router.call_target(messages2)
            except BudgetExceeded as e:
                logger.error("Budget exceeded: %s", e)
                is_partial = True
                break
            except Exception as e:
                logger.error("Target call failed for %s s2 %s: %s", sid, level, e)
                resp2 = ""
                usage2 = {}

            judge_calls_before2 = router.total_judge_calls()
            label2, judge_latency_ms2 = grade(router, item, resp2)
            judge_api_calls2 = router.total_judge_calls() - judge_calls_before2
            row2 = _build_row(
                run_name, item, "s2", level, resp2, label2, usage2, judge_latency_ms2, mode,
                judge_api_calls2,
            )
            all_rows.append(row2)
            append_checkpoint(cp_s2, {**row2, "key": key_s2})
            append_checkpoint(graded_path, row2)
            completed_keys.add(key_s2)

        if is_partial:
            break

        # Stage 2 NEUTRAL: reuse label0/resp0 (NO extra call)
        key_s2_neutral = make_key(run_name, sid, "s2", "neutral")
        completed_keys.add(key_s2_neutral)

    if total_skipped > 0:
        logger.info("Skipped %d already-completed keys.", total_skipped)

    # -----------------------------------------------------------------------
    # Enrich neutral rows with label_s1 for recovery metrics
    # -----------------------------------------------------------------------
    s1_label_map: dict[str, str] = {}
    s1_resp_map: dict[str, str] = {}
    for r in all_rows:
        if r.get("level") == "recovery":
            s1_label_map[r.get("sample_id", "")] = r.get("label", "")
            s1_resp_map[r.get("sample_id", "")] = r.get("response_text", "")
    for r in all_rows:
        if r.get("level") == "neutral":
            r["label_s1"] = s1_label_map.get(r.get("sample_id", ""))
            r["resp_s1"] = s1_resp_map.get(r.get("sample_id", ""))

    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    model_results = None
    try:
        model_results = metrics_mod.compute(
            model_name=run_name,
            all_rows=all_rows,
        )
    except Exception as e:
        logger.error("Metrics computation failed for %s: %s", run_name, e)

    # -----------------------------------------------------------------------
    # Post-micro sanity report (micro mode only)
    # -----------------------------------------------------------------------
    if mode == "micro" and model_results is not None:
        _print_sanity_report(
            run_name=run_name,
            all_rows=all_rows,
            model_results=model_results,
            router=router,
            settings=settings,
            all_items=all_items,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            graded_dir=graded_dir,
            audit_prompt_path=audit_prompt_path,
        )
        _print_observed_cost(all_rows, all_items, settings, router, run_name)

    logger.info("Run %s complete: %d total rows | partial=%s", run_name, len(all_rows), is_partial)
    return all_rows, model_results, is_partial, router


# ---------------------------------------------------------------------------
# Row builder
# ---------------------------------------------------------------------------

def _build_row(
    run_name: str,
    item: dict,
    stage: str,
    level: str,
    response_text: str,
    label: str,
    usage: dict,
    judge_latency_ms: float,
    mode: str,
    judge_api_calls: int = 0,
) -> dict:
    """Build a standardized graded row dict with per-row judge latency."""
    judge_called = (
        label not in ("REFUSAL", "REFUSED", "EMPTY", "JUDGE_ERROR")
        and response_text.strip() != ""
    )

    return {
        "run_name": run_name,
        "sample_id": item["sample_id"],
        "dataset_name": item["dataset_name"],
        "question": item["question"],
        "expected_answers": item["expected_answers"],
        "premise": item["premise"],
        "hop": item["hop"],
        "answer_type": item.get("answer_type", ""),
        "eval_dimension": item.get("eval_dimension", ""),
        "category": item.get("category", ""),
        "stage": stage,
        "level": level,
        "response_text": response_text,
        "thinking_text": usage.get("thinking_text", ""),  # CoT trace (empty for non-thinking models)
        "label": label,
        "label_s1": None,          # filled in post-loop for neutral rows
        "judge_called": judge_called,
        "judge_api_calls": judge_api_calls,
        "judge_latency_ms": judge_latency_ms,   # real per-call latency (0.0 if no judge)
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "raw_completion_tokens": usage.get("raw_completion_tokens", usage.get("completion_tokens", 0)),
        "total_tokens": usage.get("total_tokens", 0),
        "mode": mode,
        "raw_metadata": item.get("raw_metadata", {}),
    }


# ---------------------------------------------------------------------------
# Prompt audit writer (raw target message list, verifies no metadata leakage)
# ---------------------------------------------------------------------------

def _write_prompt_audit(path: str, run_name: str, sample_id: str, level: str, messages: list[dict]) -> None:
    """
    Append one JSONL record per target call to the prompt audit file.
    Consumers can grep/assert that no expected_answers / explanation / options appear
    in any user-role message content.
    """
    record = {
        "run_name": run_name,
        "sample_id": sample_id,
        "level": level,
        "messages": messages,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Sanity report (micro mode) — ASCII-safe console output
# ---------------------------------------------------------------------------

def _print_sanity_report(
    run_name: str,
    all_rows: list[dict],
    model_results: dict,
    router: APIRouter,
    settings: _Settings,
    all_items: list[dict],
    output_dir: str,
    checkpoint_dir: str,
    graded_dir: str,
    audit_prompt_path: str,
) -> None:
    # Use ASCII-safe output throughout (Windows CP1252 console cannot encode Unicode symbols)
    print("\n" + "="*70)
    print("MICRO SANITY REPORT")
    print("="*70)

    # Rows per stage
    stages_seen: dict = {}
    for r in all_rows:
        k = (r.get("stage"), r.get("level"))
        stages_seen[k] = stages_seen.get(k, 0) + 1
    print("\nRows written per stage/level:")
    for k, cnt in sorted(stages_seen.items()):
        print(f"  {k[0]}/{k[1]}: {cnt}")

    # Labels seen by premise type
    true_labels = set(r.get("label") for r in all_rows if r.get("premise") is True)
    false_labels = set(r.get("label") for r in all_rows if r.get("premise") is False)
    print(f"\nLabels seen (true premise):  {sorted(true_labels)}")
    print(f"Labels seen (false premise): {sorted(false_labels)}")

    # Judge health from metrics
    jh = model_results.get("judge_health", {})
    print(f"\nJudge parse failures: {jh.get('parse_failures', 0)}")
    print(f"Judge errors:         {jh.get('judge_errors', 0)}")
    print(f"Mean judge latency:   {jh.get('mean_latency_ms', 0):.1f} ms")
    print(f"Median judge latency: {jh.get('median_latency_ms', 0):.1f} ms")
    # Also report from router directly (more accurate: includes ALL judge calls)
    print(f"Router judge calls:   {router.total_judge_calls()}")
    print(f"Router mean latency:  {router.mean_judge_latency_ms():.1f} ms")

    # LM Studio pre-flight result
    print(f"\nLM Studio pre-flight:")
    print(f"  Provider: {router.judge_provider}")
    if router.judge_provider == "lmstudio":
        print(f"  Base URL: {get_base_url('lmstudio')}")
        print(f"  Loaded models: {router.loaded_judge_models}")
        print(f"  Resolved judge model: {router.resolved_judge_model}")
        print(f"  Reachable: {'YES (pre-flight passed)' if router._preflight_done else 'NO'}")

    # Checkpoint file sizes
    print("\nCheckpoint file sizes:")
    for label_str, fpath in [
        ("s0",     _checkpoint_path(run_name, "s0", checkpoint_dir)),
        ("s1",     _checkpoint_path(run_name, "s1", checkpoint_dir)),
        ("s2",     _checkpoint_path(run_name, "s2", checkpoint_dir)),
        ("graded", _graded_path(run_name, graded_dir)),
        ("audit",  audit_prompt_path),
    ]:
        size = Path(fpath).stat().st_size if Path(fpath).exists() else 0
        print(f"  {label_str:8s}: {fpath} ({size} bytes)")

    # One fully-rendered example per stage for eyeballing
    print("\nExample message list (s0/neutral, first item):")
    for r in all_rows:
        if r.get("stage") == "s0":
            print(f"  Q:        {r.get('question', '')[:90]}")
            print(f"  Response: {r.get('response_text', '')[:90]}")
            print(f"  Label:    {r.get('label')}")
            print(f"  JudgeLatency: {r.get('judge_latency_ms', 0):.1f} ms")
            break

    # PASS/FAIL assessment
    all_stages_ok = all(cnt > 0 for cnt in stages_seen.values())
    lmstudio_ok = router.judge_provider != "lmstudio" or (
        router._preflight_done and len(router.loaded_judge_models) >= 1
    )
    no_exceptions = True  # if we reached here, no unhandled exception occurred
    sanity_pass = (
        all_stages_ok
        and jh.get("parse_failures", 0) == 0
        and lmstudio_ok
        and no_exceptions
    )
    # ASCII-safe PASS/FAIL (avoids Windows CP1252 crash with checkmark/cross)
    verdict = "SANITY PASS" if sanity_pass else "SANITY FAIL"
    print(f"\n*** {verdict} ***")

    # Write sanity JSON
    sanity_data = {
        "run_name": run_name,
        "stages_seen": {str(k): v for k, v in stages_seen.items()},
        "true_labels": sorted(true_labels),
        "false_labels": sorted(false_labels),
        "judge_health": jh,
        "router_total_judge_calls": router.total_judge_calls(),
        "router_mean_judge_latency_ms": router.mean_judge_latency_ms(),
        "lmstudio_reachable": lmstudio_ok,
        "loaded_models": router.loaded_judge_models,
        "resolved_judge_model": router.resolved_judge_model,
        "sanity_pass": sanity_pass,
    }
    sanity_path = str(Path(output_dir) / "micro_sanity.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity_data, f, indent=2)
    print(f"\nSanity report written to {sanity_path}")
    print("="*70 + "\n")


def _print_observed_cost(
    all_rows: list[dict],
    all_items: list[dict],
    settings: _Settings,
    router: APIRouter,
    run_name: str,
) -> None:
    """Print observed cost reestimate with ETA after micro-test."""
    neutral_rows = [r for r in all_rows if r.get("level") == "neutral"]
    true_triggered = sum(
        1 for r in neutral_rows
        if r.get("premise") is True and r.get("label") in ("REFUSAL", "INCORRECT")
    )
    true_total = sum(1 for r in neutral_rows if r.get("premise") is True)
    false_halluc = sum(
        1 for r in neutral_rows
        if r.get("premise") is False and r.get("label") == "HALLUCINATED"
    )
    false_total = sum(1 for r in neutral_rows if r.get("premise") is False)

    observed_fail_rate = true_triggered / true_total if true_total > 0 else 0.6
    observed_halluc_rate = false_halluc / false_total if false_total > 0 else 0.4

    cost = reestimate_from_observed(
        items=all_items,
        settings=settings,
        observed_true_fail_rate=observed_fail_rate,
        observed_false_halluc_rate=observed_halluc_rate,
        mean_judge_latency_ms=router.mean_judge_latency_ms(),
        mean_target_latency_ms=router.mean_target_latency_ms(),
    )
    print_cost_table(cost, run_name=f"{run_name} (observed rates)")


# ---------------------------------------------------------------------------
# Top-level run function
# ---------------------------------------------------------------------------

def run(
    settings_raw: dict,
    runs_raw: list[dict],
    mode: str,
    data_path: str,
    run_name_filter: str | None,
    output_dir: str,
    checkpoint_dir: str,
    graded_dir: str,
    no_resume: bool = False,
    limit: int | None = None,
) -> None:
    """Top-level run dispatcher. Raises on pre-flight failure (caller exits 1)."""
    settings = _build_settings(settings_raw)

    # Filter runs if --run-name given
    if run_name_filter:
        runs_filtered = [r for r in runs_raw if r["name"] == run_name_filter]
        if not runs_filtered:
            raise ValueError(
                f"Run name {run_name_filter!r} not found in runs.yaml. "
                f"Available: {[r['name'] for r in runs_raw]}"
            )
    else:
        runs_filtered = runs_raw

    # Dry mode: no network, no LM Studio
    if mode == "dry":
        run_dry(settings, runs_filtered, data_path)
        return

    # Make output dirs
    ensure_dirs(output_dir, checkpoint_dir, graded_dir,
                str(Path(output_dir) / "metrics"),
                str(Path(output_dir) / "figures"),
                str(Path(output_dir) / "audit"))

    all_runs_rows: dict[str, list[dict]] = {}
    all_model_results: dict[str, dict] = {}
    run_metadata_list: list[dict] = []
    any_partial = False

    # Sequential execution over runs
    for run_cfg in runs_filtered:
        rn = run_cfg["name"]
        logger.info("="*60)
        logger.info("STARTING RUN: %s", rn)
        start_time = now_iso()

        try:
            rows, model_results, is_partial, router = run_one_target(
                run_cfg=run_cfg,
                settings=settings,
                mode=mode,
                data_path=data_path,
                checkpoint_dir=checkpoint_dir,
                graded_dir=graded_dir,
                output_dir=output_dir,
                no_resume=no_resume,
                limit=limit,
            )
        except Exception as e:
            # LMStudioNotReachable and LMStudioNoModelLoaded propagate here;
            # re-raise so main() maps them to exit code 1.
            raise

        all_runs_rows[rn] = rows
        if model_results:
            all_model_results[rn] = model_results
        if is_partial:
            any_partial = True

        end_time = now_iso()

        # Derive report metadata from persisted rows too. On a resumed run the
        # router has made no new calls, but the saved rows still represent the
        # completed experiment.
        persisted_target_calls = len(rows)
        persisted_judge_calls = sum(
            r.get("judge_api_calls", 1 if r.get("judge_called") else 0)
            for r in rows
        )
        persisted_cloud_tokens = sum(r.get("total_tokens", 0) for r in rows)
        persisted_mean_judge_latency_ms = (
            sum(r.get("judge_latency_ms", 0.0) for r in rows) / len(rows)
            if rows else 0.0
        )
        approx_cloud_tokens = max(
            router.budget.used(run_cfg["target"]["provider"]),
            persisted_cloud_tokens,
        )
        budget_limit_map = {
            "nim": settings.budget.get("nim_max_tokens", 900_000),
            "fireworks": settings.budget.get("fireworks_max_tokens", 2_000_000),
        }
        budget_limit = budget_limit_map.get(run_cfg["target"]["provider"], 1)
        budget_fraction_str = (
            f"{approx_cloud_tokens / budget_limit:.1%}" if budget_limit > 0 else "N/A"
        )

        run_metadata_list.append({
            "run_name": rn,
            "target_model": run_cfg["target"]["model_id"],
            "target_provider": run_cfg["target"]["provider"],
            "judge_model": router.resolved_judge_model,
            "judge_provider": run_cfg["judge"]["provider"],
            "lmstudio_base_url": get_base_url("lmstudio") if run_cfg["judge"]["provider"] == "lmstudio" else "N/A",
            "mode": mode,
            "start_time": start_time,
            "end_time": end_time,
            "total_target_calls": max(router.total_target_calls(), persisted_target_calls),
            "total_judge_calls": max(router.total_judge_calls(), persisted_judge_calls),
            "approx_cloud_tokens": approx_cloud_tokens,
            "budget_fraction": budget_fraction_str,
            "mean_judge_latency_ms": persisted_mean_judge_latency_ms,
        })

    # Build report
    if all_runs_rows:
        try:
            report_filename = (
                f"report_{runs_filtered[0]['name']}.html"
                if len(runs_filtered) == 1 else "report.html"
            )
            report_path = reporter_mod.build(
                all_runs_rows=all_runs_rows,
                all_model_results=all_model_results,
                run_metadata_list=run_metadata_list,
                output_dir=output_dir,
                mode=mode,
                is_partial=any_partial,
                report_filename=report_filename,
                all_rows_by_run=all_runs_rows,  # passed for T5 premise×hop
            )
            logger.info("Report written to %s", report_path)
        except Exception as e:
            logger.error("Reporter failed: %s", e, exc_info=True)
    else:
        logger.warning("No runs produced output; report skipped.")
