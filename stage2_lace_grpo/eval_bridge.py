"""
Stage 3 Evaluation Bridge: Evaluates Base, Baselines, LACE-GRPO, and Ablations
using the exact Stage 1 behavioral pipeline (src/prompts.py, src/grader.py,
src/metrics.py, src/reporter.py) on:
  1. data/benchmark_subset_100.json (including 100% OOD Bamboogle & MultiHoax)
  2. stage2_lace_grpo/data/test_heldout_500.json

Supports an independent non-Gemma judge (e.g. Qwen/Qwen2.5-7B-Instruct)
and/or Gemma judge via an in-process batched GPU engine
so an entire 7-model matrix evaluates in minutes on a single A100 GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import metrics as metrics_mod
from src import reporter as reporter_mod
from src.grader import FALSE_TRIGGER, TRUE_TRIGGER, grade
from src.loader import load as load_benchmark_100
from src.orchestrator import _build_row
from src.prompts import (
    TARGET_CONSERVATIVE,
    TARGET_PERMISSIVE,
    TARGET_RECOVERY,
    TARGET_STANDARD,
)
from src.utils import ensure_dirs, now_iso, write_jsonl
from stage2_lace_grpo.probing.probe_latent_kappa import format_chat_prompt
from stage2_lace_grpo.rewards.hybrid_verifier import extract_cot_and_answer
from stage2_lace_grpo.train_r_tuning import load_base_model_and_tokenizer


class LocalBatchedEngine:
    """Fast local inference engine (transformers bf16, optional LoRA swap)."""

    def __init__(self, model_id: str, enable_thinking: bool = True) -> None:
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        use_4bit = ("12B" in model_id or "26B" in model_id) and vram_gb < 60.0
        self.base_model, self.tokenizer = load_base_model_and_tokenizer(
            model_id, load_in_4bit=use_4bit
        )
        self.base_model.eval()
        self.active_model = self.base_model

    def load_lora(self, lora_path: str | None) -> None:
        from peft import PeftModel

        if isinstance(self.active_model, PeftModel):
            self.base_model = self.active_model.unload()
        if lora_path is None:
            self.active_model = self.base_model
            return
        self.active_model = PeftModel.from_pretrained(self.base_model, lora_path)
        self.active_model.eval()

    def generate_batch(
        self,
        system_prompt: str,
        questions: list[str],
        max_new_tokens: int = 512,
        batch_size: int = 48,
    ) -> list[tuple[str, str, int]]:
        """Returns list of (answer_text, thinking_text, completion_tokens)."""
        if not questions:
            return []
        prompts = [
            format_chat_prompt(
                self.tokenizer, system_prompt, q, enable_thinking=self.enable_thinking
            )
            for q in questions
        ]
        orig_pad = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        results: list[tuple[str, str, int]] = []
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start : start + batch_size]
                enc = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1536,
                    add_special_tokens=False,
                ).to(self.active_model.device)
                in_len = enc["input_ids"].shape[1]
                gen = self.active_model.generate(
                    **enc,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )
                for b in range(len(chunk)):
                    seq = gen[b, in_len:]
                    non_pad = (seq != self.tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                    if len(non_pad) > 0:
                        seq = seq[: int(non_pad[-1]) + 1]
                    else:
                        seq = seq[:0]
                    raw_txt = self.tokenizer.decode(seq, skip_special_tokens=False)
                    ans_txt, think_txt, _ = extract_cot_and_answer(raw_txt)
                    results.append((ans_txt, think_txt, int(len(seq))))
        self.tokenizer.padding_side = orig_pad
        return results

    def close(self) -> None:
        del self.active_model
        del self.base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LocalJudgeRouter:
    """Implements the APIRouter.call_judge(messages) interface expected by src/grader.py."""

    def __init__(self, judge_model_id: str) -> None:
        self.judge_model_id = judge_model_id
        self.judge_provider = "local_gpu"
        self._model, self._tokenizer = load_base_model_and_tokenizer(judge_model_id)
        self._model.eval()
        self._latencies: list[float] = []
        self._cache: dict[str, tuple[str, dict]] = {}

    def call_judge(self, messages: list[dict]) -> tuple[str, dict, float]:
        cache_key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if cache_key in self._cache:
            ans, usage = self._cache[cache_key]
            return ans, usage, 0.5

        t0 = time.monotonic()
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            prompt = self._tokenizer.apply_chat_template(messages, **kwargs)
        enc = self._tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(self._model.device)
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = self._model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=24,
                pad_token_id=self._tokenizer.pad_token_id,
                use_cache=True,
            )
        raw = self._tokenizer.decode(gen[0, in_len:], skip_special_tokens=True).strip()
        ans, _, _ = extract_cot_and_answer(raw)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        self._latencies.append(elapsed_ms)
        usage = {"completion_tokens": int(len(gen[0]) - in_len)}
        final_ans = ans or raw
        self._cache[cache_key] = (final_ans, usage)
        return final_ans, usage, elapsed_ms

    def close(self) -> None:
        del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_eval_items(path: str) -> list[dict]:
    p = Path(path)
    if p.name == "benchmark_subset_100.json":
        return load_benchmark_100(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 Evaluation Bridge")
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--runs-json",
        required=True,
        help="JSON list of {name, lora_path, family, rl_stage}",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    items = load_eval_items(args.data_path)
    if args.limit is not None:
        items = items[: args.limit]
    runs_spec = json.loads(Path(args.runs_json).read_text(encoding="utf-8"))

    out_dir = Path(args.output_dir)
    raw_cache_dir = out_dir / "raw_responses"
    graded_dir = out_dir / "graded"
    ensure_dirs(
        str(out_dir),
        str(raw_cache_dir),
        str(graded_dir),
        str(out_dir / "metrics"),
        str(out_dir / "figures"),
        str(out_dir / "audit"),
    )

    pending = [
        spec for spec in runs_spec if not (raw_cache_dir / f"{spec['name']}__raw.json").exists()
    ]
    raw_by_run: dict[str, dict[str, dict]] = {}
    questions = [it["question"] for it in items]

    if pending:
        engine = LocalBatchedEngine(args.base_model, enable_thinking=not args.no_thinking)
        for spec in runs_spec:
            rn = spec["name"]
            cache_file = raw_cache_dir / f"{rn}__raw.json"
            if cache_file.exists():
                print(f"[Eval Phase A] Using cached responses for {rn}", flush=True)
                raw_by_run[rn] = json.loads(cache_file.read_text(encoding="utf-8"))
                continue

            print(f"[Eval Phase A] Generating Stage 0 & Stage 2 responses for {rn}...", flush=True)
            t0 = time.monotonic()
            engine.load_lora(spec.get("lora_path"))
            s0_res = engine.generate_batch(
                TARGET_STANDARD, questions, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
            )
            s2_perm_res = engine.generate_batch(
                TARGET_PERMISSIVE, questions, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
            )
            s2_cons_res = engine.generate_batch(
                TARGET_CONSERVATIVE, questions, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
            )
            s1_res = engine.generate_batch(
                TARGET_RECOVERY, questions, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size
            )

            run_cache: dict[str, dict] = {}
            for idx, it in enumerate(items):
                sid = it["sample_id"]
                run_cache[sid] = {
                    "neutral": {"ans": s0_res[idx][0], "think": s0_res[idx][1], "toks": s0_res[idx][2]},
                    "permissive": {"ans": s2_perm_res[idx][0], "think": s2_perm_res[idx][1], "toks": s2_perm_res[idx][2]},
                    "conservative": {"ans": s2_cons_res[idx][0], "think": s2_cons_res[idx][1], "toks": s2_cons_res[idx][2]},
                    "recovery": {"ans": s1_res[idx][0], "think": s1_res[idx][1], "toks": s1_res[idx][2]},
                }
            cache_file.write_text(json.dumps(run_cache, ensure_ascii=False), encoding="utf-8")
            raw_by_run[rn] = run_cache
            print(f"[Eval Phase A] Finished {rn} in {time.monotonic()-t0:.1f}s", flush=True)
        engine.close()
    else:
        for spec in runs_spec:
            rn = spec["name"]
            raw_by_run[rn] = json.loads((raw_cache_dir / f"{rn}__raw.json").read_text(encoding="utf-8"))

    print(f"[Eval Phase B] Loading judge {args.judge_model}...", flush=True)
    judge_router = LocalJudgeRouter(args.judge_model)

    all_runs_rows: dict[str, list[dict]] = {}
    all_model_results: dict[str, dict] = {}
    run_metadata_list: list[dict] = []

    for spec in runs_spec:
        rn = spec["name"]
        graded_path = graded_dir / f"{rn}__graded.jsonl"
        run_cache = raw_by_run[rn]
        rows: list[dict] = []

        print(f"[Eval Phase B] Grading {rn} ({len(items)} items)...", flush=True)
        for it in items:
            sid = it["sample_id"]
            c = run_cache[sid]

            r0_txt = c["neutral"]["ans"]
            lbl0, lat0 = grade(judge_router, it, r0_txt)
            rows.append(
                _build_row(
                    rn,
                    it,
                    "s0",
                    "neutral",
                    r0_txt,
                    lbl0,
                    {"thinking_text": c["neutral"]["think"], "completion_tokens": c["neutral"]["toks"]},
                    lat0,
                    "full",
                    1 if lat0 > 0 else 0,
                )
            )

            trig = TRUE_TRIGGER if it["premise"] else FALSE_TRIGGER
            if lbl0 in trig:
                r1_txt = c["recovery"]["ans"]
                lbl1, lat1 = grade(judge_router, it, r1_txt)
                rows.append(
                    _build_row(
                        rn,
                        it,
                        "s1",
                        "recovery",
                        r1_txt,
                        lbl1,
                        {"thinking_text": c["recovery"]["think"], "completion_tokens": c["recovery"]["toks"]},
                        lat1,
                        "full",
                        1 if lat1 > 0 else 0,
                    )
                )

            for lvl in ("permissive", "conservative"):
                rl_txt = c[lvl]["ans"]
                lbll, latl = grade(judge_router, it, rl_txt)
                rows.append(
                    _build_row(
                        rn,
                        it,
                        "s2",
                        lvl,
                        rl_txt,
                        lbll,
                        {"thinking_text": c[lvl]["think"], "completion_tokens": c[lvl]["toks"]},
                        latl,
                        "full",
                        1 if latl > 0 else 0,
                    )
                )

        rows = metrics_mod.deduplicate_rows(rows)
        write_jsonl(str(graded_path), rows)
        all_runs_rows[rn] = rows
        all_model_results[rn] = metrics_mod.compute(model_name=rn, all_rows=rows)

        lats = [r["judge_latency_ms"] for r in rows if r.get("judge_latency_ms", 0.0) > 0]
        run_metadata_list.append(
            {
                "run_name": rn,
                "target_model": f"{args.base_model} ({rn})",
                "target_provider": "colab_a100",
                "family": spec.get("family", "Gemma4"),
                "rl_stage": spec.get("rl_stage", ""),
                "judge_model": args.judge_model,
                "judge_provider": "local_gpu",
                "lmstudio_base_url": "N/A",
                "mode": "full",
                "start_time": now_iso(),
                "end_time": now_iso(),
                "total_target_calls": len(rows),
                "total_judge_calls": sum(r.get("judge_api_calls", 0) for r in rows),
                "approx_cloud_tokens": 0,
                "budget_fraction": "0.0%",
                "mean_judge_latency_ms": sum(lats) / len(lats) if lats else 0.0,
            }
        )

    judge_router.close()

    report_path = reporter_mod.build(
        all_runs_rows=all_runs_rows,
        all_model_results=all_model_results,
        run_metadata_list=run_metadata_list,
        output_dir=str(out_dir),
        mode="full",
        is_partial=False,
        report_filename="report.html",
        all_rows_by_run=all_runs_rows,
        runs_cfg=runs_spec,
    )
    print(f"[Eval Complete] Full report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
