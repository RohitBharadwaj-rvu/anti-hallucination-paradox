"""
Phase 1: Permissive Multi-Sample Self-Knowledge Probing (k_hat(x)).

Samples M=6..8 rollouts at T=0.7 with TARGET_PERMISSIVE on every true-premise
training item to estimate empirical latent knowability k_hat(x) in [0, 1]
and cache one verified CORRECT rollout per known item for Zero-Variance Rescue
and R-Tuning confidence-split SFT.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import TARGET_PERMISSIVE
from stage2_lace_grpo.rewards.hybrid_verifier import (
    extract_cot_and_answer,
    verify_true_premise,
)


def format_chat_prompt(
    tokenizer: Any,
    system_prompt: str,
    user_question: str,
    enable_thinking: bool = True,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _score_item_cands(item: dict[str, Any], cands: list[str]) -> dict[str, Any]:
    n_correct = 0
    cached_raw = None
    cached_ans = None
    for raw in cands:
        ans, _, _ = extract_cot_and_answer(raw)
        if verify_true_premise(ans, item.get("expected_answers") or []):
            n_correct += 1
            if cached_raw is None:
                cached_raw = raw
                cached_ans = ans
    k_hat = n_correct / max(1, len(cands))
    return {
        "k_hat": k_hat,
        "n_correct": n_correct,
        "m_samples": len(cands),
        "cached_correct_raw": cached_raw,
        "cached_correct_answer": cached_ans,
    }


def run_probing(
    model_id: str,
    train_path: str | Path,
    out_path: str | Path,
    m_samples: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 768,
    enable_thinking: bool = True,
    lora_path: str | None = None,
    use_vllm: bool = True,
    max_items: int | None = None,
    gpu_memory_utilization: float = 0.85,
    batch_size: int = 16,
    load_in_4bit: bool = False,
) -> dict[str, Any]:
    with open(train_path, "r", encoding="utf-8") as f:
        items: list[dict] = json.load(f)

    true_items = [it for it in items if bool(it.get("premise"))]
    if max_items is not None:
        true_items = true_items[:max_items]

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_p.with_suffix(".part.jsonl")

    kappa_map: dict[str, dict[str, Any]] = {}
    if part_path.exists() and not out_p.exists():
        for line in part_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid:
                kappa_map[sid] = rec["entry"]
        if kappa_map:
            print(f"[probing] Resumed {len(kappa_map)} items from {part_path}")

    pending_items = [it for it in true_items if it["sample_id"] not in kappa_map]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos

    prompts = [
        format_chat_prompt(
            tokenizer, TARGET_PERMISSIVE, it["question"], enable_thinking=enable_thinking
        )
        for it in pending_items
    ]

    used_backend = "vllm" if use_vllm else "transformers"
    if pending_items and use_vllm:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest

            llm_kwargs: dict[str, Any] = {
                "model": model_id,
                "dtype": "bfloat16",
                "max_model_len": 2048,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": True,
                "enable_lora": bool(lora_path),
            }
            if lora_path:
                llm_kwargs["max_lora_rank"] = 64
            llm = LLM(**llm_kwargs)
            sampling_params = SamplingParams(
                n=m_samples,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens,
            )
            lora_req = LoRARequest("probe_adapter", 1, lora_path) if lora_path else None
            outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
            with open(part_path, "a", encoding="utf-8") as pf:
                for it, out in zip(pending_items, outputs):
                    entry = _score_item_cands(it, [cand.text for cand in out.outputs])
                    kappa_map[it["sample_id"]] = entry
                    pf.write(json.dumps({"sample_id": it["sample_id"], "entry": entry}, ensure_ascii=False) + "\n")

            del llm
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[probing] vLLM backend failed ({e}); falling back to transformers batched generation.")
            used_backend = "transformers"

    if len(kappa_map) < len(true_items):
        used_backend = "transformers"
        import torch
        from stage2_lace_grpo.train_r_tuning import load_base_model_and_tokenizer

        attn_impl = "eager" if "gemma-4" in model_id.lower() else "sdpa"
        model, tokenizer = load_base_model_and_tokenizer(
            model_id,
            load_in_4bit=load_in_4bit,
            attn_implementation=attn_impl,
        )
        if lora_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
        model.eval()
        orig_pad_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        rem_items = [it for it in true_items if it["sample_id"] not in kappa_map]
        rem_prompts = [
            format_chat_prompt(
                tokenizer, TARGET_PERMISSIVE, it["question"], enable_thinking=enable_thinking
            )
            for it in rem_items
        ]
        t_loop = time.monotonic()
        total_rem = len(rem_items)
        with open(part_path, "a", encoding="utf-8") as pf:
            for start in range(0, total_rem, batch_size):
                batch_items = rem_items[start : start + batch_size]
                batch_prompts = rem_prompts[start : start + batch_size]
                enc = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    add_special_tokens=False,
                ).to(model.device)
                in_len = enc["input_ids"].shape[1]
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=m_samples,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                for b_idx, it in enumerate(batch_items):
                    cands = []
                    for j in range(m_samples):
                        seq = gen[b_idx * m_samples + j][in_len:]
                        non_pad = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                        if len(non_pad) > 0:
                            seq = seq[: int(non_pad[-1]) + 1]
                        else:
                            seq = seq[:0]
                        cands.append(tokenizer.decode(seq, skip_special_tokens=False))
                    entry = _score_item_cands(it, cands)
                    kappa_map[it["sample_id"]] = entry
                    pf.write(json.dumps({"sample_id": it["sample_id"], "entry": entry}, ensure_ascii=False) + "\n")
                pf.flush()
                done_rem = start + len(batch_items)
                elapsed = time.monotonic() - t_loop
                rate = done_rem / max(elapsed, 1e-3)
                eta_m = (total_rem - done_rem) / max(rate, 1e-6) / 60.0
                print(
                    f"[probing] {len(kappa_map)}/{len(true_items)} items "
                    f"({elapsed:.0f}s elapsed, {rate:.2f} items/s, ETA {eta_m:.1f}m)",
                    flush=True,
                )

        tokenizer.padding_side = orig_pad_side
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    known_any = sum(1 for v in kappa_map.values() if v["k_hat"] > 0)
    known_reliable = sum(1 for v in kappa_map.values() if v["k_hat"] >= 0.5)
    sum_k = sum(v["k_hat"] for v in kappa_map.values())

    summary = {
        "model_id": model_id,
        "lora_path": lora_path,
        "backend": used_backend,
        "m_samples": m_samples,
        "n_true_probed": len(true_items),
        "known_any_count": known_any,
        "known_any_rate": known_any / max(1, len(true_items)),
        "known_reliable_count": known_reliable,
        "mean_k_hat": sum_k / max(1, len(true_items)),
        "kappa_by_id": kappa_map,
    }
    out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if part_path.exists():
        part_path.unlink()
    return {k: v for k, v in summary.items() if k != "kappa_by_id"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Latent Knowability Probing")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--m-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-items", type=int, default=None)
    args = parser.parse_args()

    stats = run_probing(
        model_id=args.model_id,
        train_path=args.train_path,
        out_path=args.out_path,
        m_samples=args.m_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=not args.no_thinking,
        lora_path=args.lora_path,
        use_vllm=not args.no_vllm,
        max_items=args.max_items,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
