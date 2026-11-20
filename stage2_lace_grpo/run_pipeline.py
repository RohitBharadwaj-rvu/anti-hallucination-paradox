"""End-to-end resumable orchestrator for Stage 2 & Stage 3 experiments on Colab A100."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_TOKEN", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "stage2_lace_grpo/data/train_4000.json"
BENCH_100 = PROJECT_ROOT / "data/benchmark_subset_100.json"
TEST_500 = PROJECT_ROOT / "stage2_lace_grpo/data/test_heldout_500.json"


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n==================== {desc} ====================", flush=True)
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    print(f"=== FINISHED {desc} in {(time.monotonic() - t0) / 60:.1f}m ===", flush=True)


def run_gemma4_probe() -> Path:
    out_dir = PROJECT_ROOT / "outputs/stage2_gemma4_e4b"
    out_dir.mkdir(parents=True, exist_ok=True)
    kappa_path = out_dir / "gemma4_e4b_kappa.json"
    if kappa_path.exists():
        print(f"[Skip] {kappa_path} already exists.", flush=True)
        return kappa_path
    run_cmd(
        [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "stage2_lace_grpo/probing/probe_latent_kappa.py"),
            "--model-id",
            "google/gemma-4-E4B-it",
            "--train-path",
            str(TRAIN_PATH),
            "--out-path",
            str(kappa_path),
            "--m-samples",
            "6",
            "--max-new-tokens",
            "512",
            "--batch-size",
            "24",
            "--no-vllm",
        ],
        "Gemma 4 E4B Phase 1 Probing",
    )
    return kappa_path


def run_gemma4_train(kappa_path: Path) -> None:
    cfg = PROJECT_ROOT / "stage2_lace_grpo/configs/a100_gemma4_e4b.yaml"
    base_out = PROJECT_ROOT / "outputs/stage2_gemma4_e4b"

    # B1: R-Tuning
    b1_dir = base_out / "B1_RTuning"
    if not (b1_dir / "final_lora/adapter_config.json").exists():
        run_cmd(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "stage2_lace_grpo/train_r_tuning.py"),
                "--config",
                str(cfg),
                "--train-path",
                str(TRAIN_PATH),
                "--kappa-path",
                str(kappa_path),
                "--output-dir",
                str(b1_dir),
            ],
            "Gemma 4 E4B B1_RTuning",
        )
    else:
        print(f"[Skip] {b1_dir / 'final_lora'} exists.", flush=True)

    # 5 RL runs
    rl_runs = [
        ("B2_TruthRL", "truth_rl"),
        ("M1_LACE_GRPO", "lace_full"),
        ("A1_NoLatent", "ablation_no_latent"),
        ("A2_NoCoT", "ablation_no_cot"),
        ("A3_NoPressure", "ablation_no_pressure"),
    ]
    for run_name, variant in rl_runs:
        r_dir = base_out / run_name
        if (r_dir / "final_lora/adapter_config.json").exists():
            print(f"[Skip] {r_dir / 'final_lora'} exists.", flush=True)
            continue
        run_cmd(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "stage2_lace_grpo/train_lace_grpo.py"),
                "--config",
                str(cfg),
                "--variant",
                variant,
                "--train-path",
                str(TRAIN_PATH),
                "--kappa-path",
                str(kappa_path),
                "--output-dir",
                str(r_dir),
                "--gen-batch-size",
                "48",
                "--micro-batch-size",
                "2",
            ],
            f"Gemma 4 E4B {run_name} ({variant})",
        )


def run_gemma4_eval(judge_model: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    base_out = PROJECT_ROOT / "outputs/stage2_gemma4_e4b"
    runs_spec = [
        {"name": "B0_Base", "lora_path": None, "family": "Gemma4-E4B", "rl_stage": "Base"},
        {"name": "B1_RTuning", "lora_path": str(base_out / "B1_RTuning/final_lora"), "family": "Gemma4-E4B", "rl_stage": "R-Tuning (SFT)"},
        {"name": "B2_TruthRL", "lora_path": str(base_out / "B2_TruthRL/final_lora"), "family": "Gemma4-E4B", "rl_stage": "TruthRL (Static)"},
        {"name": "M1_LACE_GRPO", "lora_path": str(base_out / "M1_LACE_GRPO/final_lora"), "family": "Gemma4-E4B", "rl_stage": "LACE-GRPO"},
        {"name": "A1_NoLatent", "lora_path": str(base_out / "A1_NoLatent/final_lora"), "family": "Gemma4-E4B", "rl_stage": "w/o LatentAnchor"},
        {"name": "A2_NoCoT", "lora_path": str(base_out / "A2_NoCoT/final_lora"), "family": "Gemma4-E4B", "rl_stage": "w/o CoT Penalty"},
        {"name": "A3_NoPressure", "lora_path": str(base_out / "A3_NoPressure/final_lora"), "family": "Gemma4-E4B", "rl_stage": "w/o Pressure Sweep"},
    ]
    spec_path = base_out / "runs_spec_7.json"
    spec_path.write_text(json.dumps(runs_spec, indent=2), encoding="utf-8")

    for tag, data_path in [("bench100", BENCH_100), ("test500", TEST_500)]:
        eval_out = PROJECT_ROOT / f"outputs/stage3_gemma4_e4b_{tag}"
        if (eval_out / "report.html").exists():
            print(f"[Skip] {eval_out / 'report.html'} exists.", flush=True)
            continue
        run_cmd(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "stage2_lace_grpo/eval_bridge.py"),
                "--base-model",
                "google/gemma-4-E4B-it",
                "--runs-json",
                str(spec_path),
                "--data-path",
                str(data_path),
                "--judge-model",
                judge_model,
                "--output-dir",
                str(eval_out),
                "--batch-size",
                "48",
                "--max-new-tokens",
                "512",
            ],
            f"Gemma 4 E4B Stage 3 Eval ({tag}, judge={judge_model})",
        )


def run_qwen3_all(judge_model: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    cfg = PROJECT_ROOT / "stage2_lace_grpo/configs/a100_qwen3_8b.yaml"
    base_out = PROJECT_ROOT / "outputs/stage2_qwen3_8b"
    base_out.mkdir(parents=True, exist_ok=True)
    kappa_path = base_out / "qwen3_8b_kappa.json"

    if not kappa_path.exists():
        run_cmd(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "stage2_lace_grpo/probing/probe_latent_kappa.py"),
                "--model-id",
                "Qwen/Qwen3-8B",
                "--train-path",
                str(TRAIN_PATH),
                "--out-path",
                str(kappa_path),
                "--m-samples",
                "6",
                "--max-new-tokens",
                "512",
                "--batch-size",
                "24",
                "--no-vllm",
            ],
            "Qwen3-8B Phase 1 Probing",
        )

    for run_name, variant in [("B2_TruthRL", "truth_rl"), ("M1_LACE_GRPO", "lace_full")]:
        r_dir = base_out / run_name
        if (r_dir / "final_lora/adapter_config.json").exists():
            print(f"[Skip] {r_dir / 'final_lora'} exists.", flush=True)
            continue
        run_cmd(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "stage2_lace_grpo/train_lace_grpo.py"),
                "--config",
                str(cfg),
                "--variant",
                variant,
                "--train-path",
                str(TRAIN_PATH),
                "--kappa-path",
                str(kappa_path),
                "--output-dir",
                str(r_dir),
                "--gen-batch-size",
                "48",
                "--micro-batch-size",
                "2",
            ],
            f"Qwen3-8B {run_name} ({variant})",
        )

    runs_spec = [
        {"name": "B0_Base", "lora_path": None, "family": "Qwen3-8B", "rl_stage": "Base"},
        {"name": "B2_TruthRL", "lora_path": str(base_out / "B2_TruthRL/final_lora"), "family": "Qwen3-8B", "rl_stage": "TruthRL (Static)"},
        {"name": "M1_LACE_GRPO", "lora_path": str(base_out / "M1_LACE_GRPO/final_lora"), "family": "Qwen3-8B", "rl_stage": "LACE-GRPO"},
    ]
    spec_path = base_out / "runs_spec_3.json"
    spec_path.write_text(json.dumps(runs_spec, indent=2), encoding="utf-8")

    for tag, data_path in [("bench100", BENCH_100), ("test500", TEST_500)]:
        eval_out = PROJECT_ROOT / f"outputs/stage3_qwen3_8b_{tag}"
        if (eval_out / "report.html").exists():
            print(f"[Skip] {eval_out / 'report.html'} exists.", flush=True)
            continue
        run_cmd(
            [
                sys.executable,
                "-u",
                str(PROJECT_ROOT / "stage2_lace_grpo/eval_bridge.py"),
                "--base-model",
                "Qwen/Qwen3-8B",
                "--runs-json",
                str(spec_path),
                "--data-path",
                str(data_path),
                "--judge-model",
                judge_model,
                "--output-dir",
                str(eval_out),
                "--batch-size",
                "48",
                "--max-new-tokens",
                "512",
            ],
            f"Qwen3-8B Stage 3 Eval ({tag}, judge={judge_model})",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 & 3 Colab Pipeline Orchestrator")
    parser.add_argument(
        "--phase",
        choices=["gemma_probe", "gemma_train", "gemma_eval", "gemma_all", "qwen_all", "all"],
        required=True,
    )
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    if args.phase in ("gemma_probe", "gemma_all", "all"):
        k = run_gemma4_probe()
    else:
        k = PROJECT_ROOT / "outputs/stage2_gemma4_e4b/gemma4_e4b_kappa.json"

    if args.phase in ("gemma_train", "gemma_all", "all"):
        run_gemma4_train(k)

    if args.phase in ("gemma_eval", "gemma_all", "all"):
        run_gemma4_eval(args.judge_model)

    if args.phase in ("qwen_all", "all"):
        run_qwen3_all(args.judge_model)


if __name__ == "__main__":
    main()
