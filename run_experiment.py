#!/usr/bin/env python3
"""
run_experiment.py — CLI entry point for the Anti-Hallucination Paradox pipeline.

Usage examples:
  # 1) Sanity check, no network, no LM Studio
  python run_experiment.py --mode dry

  # 2) Offline regrade & report regeneration (cleans existing checkpoints & graded JSONLs)
  python run_experiment.py --mode regrade

  # 3) 5-question micro test (needs LM Studio server ON + cloud keys)
  python run_experiment.py --mode micro

  # 4) Full run over all models in runs.yaml (re-running only previously truncated items if --retry-truncated)
  python run_experiment.py --mode full --retry-truncated
"""
import argparse
import sys
import os
from pathlib import Path

# ---- project root on sys.path ----
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.utils import setup_logging, get_logger
from src.env_loader import load_env
from src.orchestrator import run as orchestrator_run

logger = get_logger("ahp.main")


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = str(PROJECT_ROOT / "data" / "benchmark_subset_100.json")
DEFAULT_RUNS_YAML = str(PROJECT_ROOT / "config" / "runs.yaml")
DEFAULT_SETTINGS_YAML = str(PROJECT_ROOT / "config" / "settings.yaml")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs")
DEFAULT_CHECKPOINT_DIR = str(PROJECT_ROOT / "data" / "checkpoints")
DEFAULT_GRADED_DIR = str(PROJECT_ROOT / "outputs" / "graded")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Anti-Hallucination Paradox — behavioral evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--runs",
        default=DEFAULT_RUNS_YAML,
        metavar="PATH",
        help=f"Path to runs.yaml (default: {DEFAULT_RUNS_YAML})",
    )
    p.add_argument(
        "--mode",
        choices=["dry", "micro", "full", "regrade"],
        default="full",
        help="Execution mode: dry (no network), regrade (offline clean & rebuild), micro (5 items), full (all items). Default: full",
    )
    p.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Run a single entry from runs.yaml by name (overrides batch).",
    )
    p.add_argument(
        "--target-provider",
        choices=["nim", "fireworks", "openrouter"],
        default=None,
        metavar="PROVIDER",
        help="Single-run override: target provider (nim, fireworks, or openrouter).",
    )
    p.add_argument(
        "--target-model",
        default=None,
        metavar="MODEL_ID",
        help="Single-run override: target model ID.",
    )
    p.add_argument(
        "--judge-provider",
        choices=["nim", "fireworks", "openrouter", "lmstudio"],
        default=None,
        metavar="PROVIDER",
        help="Single-run override: judge provider (default lmstudio).",
    )
    p.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL_ID",
        help="Single-run override: judge model ID.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        metavar="N",
        help="Override concurrency_target (1 = sequential).",
    )
    p.add_argument(
        "--concurrency-judge",
        type=int,
        default=None,
        metavar="N",
        help="Override concurrency_judge (default 1).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N items (debug/testing).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoints and start fresh (careful: expensive).",
    )
    p.add_argument(
        "--retry-truncated",
        action="store_true",
        help="Purge previously truncated/empty CoT or JUDGE_ERROR checkpoint entries so they are re-queried.",
    )
    p.add_argument(
        "--run-tag",
        default=None,
        metavar="TAG",
        help="Optional custom tag/name for this run's archive folder and report (e.g. 'sanity_v1' or 'final_full').",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    p.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        metavar="PATH",
        help=f"Path to benchmark JSON (default: {DEFAULT_DATA_PATH})",
    )
    p.add_argument(
        "--settings",
        default=DEFAULT_SETTINGS_YAML,
        metavar="PATH",
        help=f"Path to settings.yaml (default: {DEFAULT_SETTINGS_YAML})",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="PATH",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    log = setup_logging(args.log_level)
    log.info("Anti-Hallucination Paradox pipeline starting (mode=%s)", args.mode)

    load_env()

    settings_path = args.settings
    if not Path(settings_path).exists():
        log.error("settings.yaml not found at %s", settings_path)
        return 1
    with open(settings_path, "r", encoding="utf-8") as f:
        settings_raw: dict = yaml.safe_load(f)

    if args.concurrency is not None:
        settings_raw["concurrency_target"] = args.concurrency
    if args.concurrency_judge is not None:
        settings_raw["concurrency_judge"] = args.concurrency_judge

    runs_raw: list[dict] = []

    if args.target_provider and args.target_model:
        judge_provider = args.judge_provider or "lmstudio"
        judge_model = args.judge_model or "google/gemma-4-e4b"
        ad_hoc_run_name = args.run_name or f"adhoc__{args.target_model.replace('/', '_')}"
        runs_raw = [{
            "name": ad_hoc_run_name,
            "target": {
                "provider": args.target_provider,
                "model_id": args.target_model,
            },
            "judge": {
                "provider": judge_provider,
                "model_id": judge_model,
            },
        }]
        log.info(
            "Ad-hoc single run: target=%s/%s judge=%s/%s",
            args.target_provider, args.target_model,
            judge_provider, judge_model,
        )
    else:
        runs_path = args.runs
        if not Path(runs_path).exists():
            log.error("runs.yaml not found at %s", runs_path)
            return 1
        with open(runs_path, "r", encoding="utf-8") as f:
            runs_cfg = yaml.safe_load(f)
        runs_raw = runs_cfg.get("runs", [])
        if not runs_raw:
            log.error("No runs found in %s", runs_path)
            return 1

    data_path = args.data
    if not Path(data_path).exists():
        log.error("Benchmark data file not found at %s", data_path)
        return 1

    output_dir = args.output_dir
    checkpoint_dir = str(PROJECT_ROOT / "data" / "checkpoints")
    graded_dir = str(Path(output_dir) / "graded")

    try:
        orchestrator_run(
            settings_raw=settings_raw,
            runs_raw=runs_raw,
            mode=args.mode,
            data_path=data_path,
            run_name_filter=args.run_name if not (args.target_provider and args.target_model) else None,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            graded_dir=graded_dir,
            no_resume=args.no_resume,
            limit=args.limit,
            retry_truncated=args.retry_truncated,
            run_tag=args.run_tag,
        )
    except KeyboardInterrupt:
        log.info("\nInterrupted by user. Checkpoints are safe; re-run to resume.")
        return 0
    except Exception as e:
        exc_type_name = type(e).__name__
        if exc_type_name in ("LMStudioNotReachable", "LMStudioNoModelLoaded"):
            log.error("LM Studio pre-flight failed: %s", e)
            log.error(
                "Fix: start LM Studio -> Developer -> Start Server, "
                "load a model, and verify LMSTUDIO_BASE_URL in .env."
            )
            return 1
        log.error("Pipeline failed: %s", e, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
