"""
Stage 2 LACE-GRPO Trainer (Single NVIDIA A100).

Implements:
  - Counterfactual 3-prompt grouping: [2x NEUTRAL, 2x CONSERVATIVE, 2x PERMISSIVE]
  - Fast Hybrid In-Loop Verifier (regex + alias/numeric + DeBERTa NLI)
  - Composite reward: R_epistemic(k_hat) + R_CoT + R_press + R_fmt
  - Prompt-centered baseline + pooled group std advantage normalization
  - Zero-Variance Rescue (Anti-Collapse) using cached Phase 1 CORRECT rollouts
  - Clipped GRPO surrogate + Schulman k3 KL penalty via model.disable_adapter()
  - Mid-RL k_hat(x) refresh at S_total // 2
  - Full checkpoint auto-resume (adapter + optimizer + scheduler + RNG + pool state)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grader import CORRECT, REFUSAL
from src.prompts import TARGET_CONSERVATIVE, TARGET_NEUTRAL, TARGET_PERMISSIVE
from stage2_lace_grpo.probing.probe_latent_kappa import format_chat_prompt
from stage2_lace_grpo.rewards.hybrid_verifier import (
    FalsePremiseNLIVerifier,
    extract_cot_and_answer,
    verify_true_premise,
)
from stage2_lace_grpo.rewards.lace_rewards import (
    RewardConfig,
    compute_prompt_centered_advantages,
    score_grouped_rollouts,
)
from stage2_lace_grpo.train_r_tuning import (
    apply_text_lora,
    load_base_model_and_tokenizer,
)


LEVEL_TO_SYSTEM_PROMPT = {
    "neutral": TARGET_NEUTRAL,
    "conservative": TARGET_CONSERVATIVE,
    "permissive": TARGET_PERMISSIVE,
}


def _find_latest_checkpoint(out_dir: Path) -> tuple[int, Path | None]:
    if not out_dir.exists():
        return 0, None
    best_step = 0
    best_dir = None
    for p in out_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            s = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        if (p / "adapter_config.json").exists() and s > best_step:
            best_step = s
            best_dir = p
    return best_step, best_dir


def _compute_seq_logps(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: list[int],
    comp_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-token log-probabilities for completion tokens only.
    Uses chunked fp32 cross_entropy to avoid materializing [L, V] fp32 log_softmax buffers.
    Returns (comp_logps [B, max_comp], comp_mask [B, max_comp]).
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    max_comp = max(comp_lens) if comp_lens else 1
    bs = input_ids.shape[0]
    comp_logps = torch.zeros((bs, max_comp), dtype=torch.float32, device=input_ids.device)
    comp_mask = torch.zeros((bs, max_comp), dtype=torch.float32, device=input_ids.device)

    chunk_sz = 128
    for b in range(bs):
        p_len = prompt_lens[b]
        c_len = comp_lens[b]
        if c_len <= 0:
            continue
        start_idx = max(0, p_len - 1)
        end_idx = min(shift_logits.shape[1], start_idx + c_len)
        actual_len = end_idx - start_idx
        if actual_len <= 0:
            continue
        sl = shift_logits[b, start_idx:end_idx, :]
        lb = shift_labels[b, start_idx:end_idx]
        if actual_len <= chunk_sz:
            tok_logps = -F.cross_entropy(sl.float(), lb, reduction="none")
        else:
            parts = [
                -F.cross_entropy(sl[t0 : t0 + chunk_sz].float(), lb[t0 : t0 + chunk_sz], reduction="none")
                for t0 in range(0, actual_len, chunk_sz)
            ]
            tok_logps = torch.cat(parts, dim=0)
        comp_logps[b, :actual_len] = tok_logps
        comp_mask[b, :actual_len] = 1.0

    return comp_logps, comp_mask


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    gen_batch_size: int = 48,
) -> list[str]:
    was_training = model.training
    model.eval()
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    completions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(prompt_texts), gen_batch_size):
            chunk = prompt_texts[start : start + gen_batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            ).to(model.device)
            in_len = enc["input_ids"].shape[1]
            gen_ids = model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            for b in range(len(chunk)):
                seq = gen_ids[b, in_len:]
                non_pad = (seq != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
                if len(non_pad) > 0:
                    seq = seq[: int(non_pad[-1]) + 1]
                else:
                    seq = seq[:0]
                completions.append(tokenizer.decode(seq, skip_special_tokens=False))

    tokenizer.padding_side = orig_pad_side
    if was_training:
        model.train()
    return completions


def _refresh_kappa_in_memory(
    model: Any,
    tokenizer: Any,
    train_items: list[dict],
    kappa_by_id: dict[str, dict],
    enable_thinking: bool,
    m_samples: int = 4,
    max_new_tokens: int = 512,
    gen_batch_size: int = 48,
    target_sample_ids: set[str] | None = None,
) -> int:
    """Mid-RL refresh of k_hat(x) using the current policy in memory."""
    true_items = [
        it
        for it in train_items
        if bool(it.get("premise"))
        and (target_sample_ids is None or it["sample_id"] in target_sample_ids)
    ]
    prompts = []
    for it in true_items:
        p = format_chat_prompt(
            tokenizer, TARGET_PERMISSIVE, it["question"], enable_thinking=enable_thinking
        )
        for _ in range(m_samples):
            prompts.append(p)

    print(f"[Refresh] Re-probing {len(true_items)} true items (M={m_samples}, total={len(prompts)} seqs)...", flush=True)
    raw_outs = _generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        gen_batch_size=gen_batch_size,
    )
    updated = 0
    for idx, it in enumerate(true_items):
        sid = it["sample_id"]
        cands = raw_outs[idx * m_samples : (idx + 1) * m_samples]
        n_corr = 0
        cached_raw = kappa_by_id.get(sid, {}).get("cached_correct_raw")
        cached_ans = kappa_by_id.get(sid, {}).get("cached_correct_answer")
        for raw in cands:
            ans, _, _ = extract_cot_and_answer(raw)
            if verify_true_premise(ans, it.get("expected_answers") or []):
                n_corr += 1
                if cached_raw is None:
                    cached_raw = raw
                    cached_ans = ans
        kappa_by_id[sid] = {
            "k_hat": n_corr / m_samples,
            "n_correct": n_corr,
            "m_samples": m_samples,
            "cached_correct_raw": cached_raw,
            "cached_correct_answer": cached_ans,
        }
        updated += 1
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LACE-GRPO / TruthRL / Ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "lace_full",
            "truth_rl",
            "ablation_no_latent",
            "ablation_no_cot",
            "ablation_no_pressure",
        ],
    )
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--kappa-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--gen-batch-size", type=int, default=48)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    grpo_cfg = cfg["grpo"]
    rew_dict = cfg.get("rewards", {})
    reward_cfg = RewardConfig.from_variant(args.variant, **rew_dict)

    with open(args.train_path, "r", encoding="utf-8") as f:
        train_items: list[dict] = json.load(f)
    with open(args.kappa_path, "r", encoding="utf-8") as f:
        kappa_by_id: dict[str, dict] = json.load(f).get("kappa_by_id", {})

    for it in train_items:
        sid = it["sample_id"]
        it["k_hat"] = float(kappa_by_id.get(sid, {}).get("k_hat", 0.0) or 0.0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    resume_step, resume_dir = _find_latest_checkpoint(out_dir)

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    use_4bit = bool(model_cfg.get("load_in_4bit", False))
    print(f"[Setup] model={model_cfg['model_id']} variant={args.variant} VRAM={vram_gb:.1f}GB 4bit={use_4bit}", flush=True)

    model, tokenizer = load_base_model_and_tokenizer(
        model_cfg["model_id"],
        load_in_4bit=use_4bit,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    if resume_dir is not None:
        from peft import PeftModel, prepare_model_for_kbit_training

        if use_4bit:
            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, str(resume_dir), is_trainable=True)
        model.print_trainable_parameters()
        print(f"[Resume] Loaded LoRA checkpoint from {resume_dir} (step={resume_step})", flush=True)
    else:
        model = apply_text_lora(model, cfg["lora"], is_4bit=use_4bit)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    nli_verifier = None if args.no_nli else FalsePremiseNLIVerifier(lazy=False)

    enable_thinking = bool(model_cfg.get("enable_thinking", True))
    max_steps = args.max_steps if args.max_steps is not None else int(grpo_cfg.get("max_steps", 200))
    refresh_step = int(grpo_cfg.get("refresh_kappa_at_step", max_steps // 2))
    if args.max_steps is not None and args.max_steps < 30:
        refresh_step = -1  # skip mid-RL refresh during 10-step smoke test

    prompts_per_step = int(grpo_cfg.get("prompts_per_step", 8))
    group_size = int(grpo_cfg.get("group_size", 6))
    max_comp_len = int(grpo_cfg.get("max_completion_length", 512))
    temperature = float(grpo_cfg.get("temperature", 0.7))
    top_p = float(grpo_cfg.get("top_p", 0.95))
    beta_kl = float(grpo_cfg.get("beta_kl", 0.02))
    lr = float(grpo_cfg.get("learning_rate", 5e-6))
    save_steps = int(grpo_cfg.get("save_steps", 50))
    clip_eps = 0.2
    micro_bs = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else int(grpo_cfg.get("per_device_batch_size", 2))
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warmup_steps = max(1, int(0.05 * max_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    if args.variant in ("truth_rl", "ablation_no_pressure"):
        prompt_schedule = ["neutral"] * group_size
    else:
        prompt_schedule = [
            "neutral",
            "neutral",
            "conservative",
            "conservative",
            "permissive",
            "permissive",
        ]

    item_pool = list(train_items)
    rng.shuffle(item_pool)
    pool_cursor = 0
    elapsed_offset_s = 0.0

    if resume_dir is not None:
        state_path = resume_dir / "trainer_state.pt"
        if state_path.exists():
            st = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            pool_cursor = int(st.get("pool_cursor", 0))
            elapsed_offset_s = float(st.get("elapsed_s", 0.0))
            if "rng_state" in st:
                rng.setstate(st["rng_state"])
            if "torch_rng_state" in st:
                torch.set_rng_state(st["torch_rng_state"])
            if "item_pool_ids" in st:
                by_id = {it["sample_id"]: it for it in train_items}
                item_pool = [by_id[sid] for sid in st["item_pool_ids"] if sid in by_id]
            print(f"[Resume] Restored optimizer/scheduler/RNG state at step={resume_step}, pool_cursor={pool_cursor}", flush=True)
        else:
            for _ in range(resume_step):
                scheduler.step()
            pool_cursor = (resume_step * prompts_per_step) % len(item_pool)

    t_start = time.monotonic()

    eos_raw = tokenizer.eos_token_id
    eos_id = eos_raw[0] if isinstance(eos_raw, list) else eos_raw
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    for step in range(resume_step + 1, max_steps + 1):
        step_t0 = time.monotonic()

        if refresh_step > 0 and step == refresh_step and args.variant != "truth_rl":
            rem_steps = max_steps - step + 1
            needed = rem_steps * prompts_per_step
            upcoming_ids: set[str] = set()
            tmp_cur = pool_cursor
            for _ in range(needed):
                if tmp_cur >= len(item_pool):
                    tmp_cur = 0
                upcoming_ids.add(item_pool[tmp_cur]["sample_id"])
                tmp_cur += 1
            _refresh_kappa_in_memory(
                model,
                tokenizer,
                train_items,
                kappa_by_id,
                enable_thinking=enable_thinking,
                m_samples=4,
                max_new_tokens=min(512, max_comp_len),
                gen_batch_size=args.gen_batch_size,
                target_sample_ids=upcoming_ids,
            )
            for it in train_items:
                it["k_hat"] = float(kappa_by_id.get(it["sample_id"], {}).get("k_hat", 0.0) or 0.0)

        batch_items: list[dict] = []
        while len(batch_items) < prompts_per_step:
            if pool_cursor >= len(item_pool):
                rng.shuffle(item_pool)
                pool_cursor = 0
            batch_items.append(item_pool[pool_cursor])
            pool_cursor += 1

        flat_items: list[dict] = []
        flat_levels: list[str] = []
        flat_prompts: list[str] = []
        for it in batch_items:
            for lvl in prompt_schedule:
                flat_items.append(it)
                flat_levels.append(lvl)
                flat_prompts.append(
                    format_chat_prompt(
                        tokenizer,
                        LEVEL_TO_SYSTEM_PROMPT[lvl],
                        it["question"],
                        enable_thinking=enable_thinking,
                    )
                )

        raw_completions = _generate_batch(
            model,
            tokenizer,
            flat_prompts,
            max_new_tokens=max_comp_len,
            temperature=temperature,
            top_p=top_p,
            gen_batch_size=args.gen_batch_size,
        )

        rewards, details = score_grouped_rollouts(
            flat_items, flat_levels, raw_completions, reward_cfg, nli_verifier=nli_verifier
        )

        rescued_count = 0
        if args.variant not in ("truth_rl", "ablation_no_latent"):
            for g_idx, it in enumerate(batch_items):
                start = g_idx * group_size
                end = start + group_size
                grp_labels = [details[k]["label"] for k in range(start, end)]
                k_hat = float(it.get("k_hat", 0.0) or 0.0)
                cached_raw = kappa_by_id.get(it["sample_id"], {}).get("cached_correct_raw")
                if (
                    bool(it.get("premise"))
                    and k_hat > 0.0
                    and all(lbl == REFUSAL for lbl in grp_labels)
                    and cached_raw
                ):
                    raw_completions[start] = cached_raw
                    rescued_count += 1
            if rescued_count > 0:
                rewards, details = score_grouped_rollouts(
                    flat_items,
                    flat_levels,
                    raw_completions,
                    reward_cfg,
                    nli_verifier=nli_verifier,
                )

        sample_ids = [it["sample_id"] for it in flat_items]
        advantages = compute_prompt_centered_advantages(rewards, sample_ids, flat_levels)

        encoded_pairs: list[tuple[list[int], int, int, float]] = []
        for p_txt, c_txt, adv in zip(flat_prompts, raw_completions, advantages):
            p_ids = tokenizer(p_txt, add_special_tokens=False, truncation=True, max_length=1024)[
                "input_ids"
            ]
            c_ids = tokenizer(
                c_txt, add_special_tokens=False, truncation=True, max_length=max_comp_len
            )["input_ids"]
            if not c_ids or c_ids[-1] != eos_id:
                c_ids = c_ids + [eos_id]
            encoded_pairs.append((p_ids + c_ids, len(p_ids), len(c_ids), float(adv)))

        model.eval()
        old_logps_list: list[torch.Tensor] = []
        ref_logps_list: list[torch.Tensor] = []
        masks_list: list[torch.Tensor] = []
        padded_batches: list[dict[str, Any]] = []

        for mb_start in range(0, len(encoded_pairs), micro_bs):
            mb = encoded_pairs[mb_start : mb_start + micro_bs]
            max_len = max(len(x[0]) for x in mb)
            in_ids, attn_m, p_lens, c_lens, advs = [], [], [], [], []
            for seq, pl, cl, adv in mb:
                pad_n = max_len - len(seq)
                in_ids.append(seq + [pad_id] * pad_n)
                attn_m.append([1] * len(seq) + [0] * pad_n)
                p_lens.append(pl)
                c_lens.append(cl)
                advs.append(adv)
            t_in = torch.tensor(in_ids, dtype=torch.long, device=model.device)
            t_am = torch.tensor(attn_m, dtype=torch.long, device=model.device)
            t_adv = torch.tensor(advs, dtype=torch.float32, device=model.device)

            with torch.no_grad():
                old_lp, comp_m = _compute_seq_logps(model, t_in, t_am, p_lens, c_lens)
                with model.disable_adapter():
                    ref_lp, _ = _compute_seq_logps(model, t_in, t_am, p_lens, c_lens)

            old_logps_list.append(old_lp)
            ref_logps_list.append(ref_lp)
            masks_list.append(comp_m)
            padded_batches.append(
                {
                    "input_ids": t_in,
                    "attention_mask": t_am,
                    "prompt_lens": p_lens,
                    "comp_lens": c_lens,
                    "advantages": t_adv,
                }
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        n_mbs = len(padded_batches)
        total_loss_val = 0.0
        total_kl_val = 0.0

        for mb_idx, mb_data in enumerate(padded_batches):
            adv = mb_data["advantages"]
            if torch.all(adv.abs() < 1e-8):
                continue
            cur_lp, _ = _compute_seq_logps(
                model,
                mb_data["input_ids"],
                mb_data["attention_mask"],
                mb_data["prompt_lens"],
                mb_data["comp_lens"],
            )
            old_lp = old_logps_list[mb_idx]
            ref_lp = ref_logps_list[mb_idx]
            mask = masks_list[mb_idx]

            log_ratio = cur_lp - old_lp
            ratio = torch.exp(torch.clamp(log_ratio, -10.0, 10.0))
            adv_exp = adv.unsqueeze(-1)
            pg1 = ratio * adv_exp
            pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_exp
            pg_loss = -torch.min(pg1, pg2)

            diff_ref = torch.clamp(ref_lp - cur_lp, -10.0, 10.0)
            kl = torch.exp(diff_ref) - diff_ref - 1.0

            per_tok_loss = (pg_loss + beta_kl * kl) * mask
            denom = mask.sum().clamp(min=1.0)
            mb_loss = per_tok_loss.sum() / denom / n_mbs
            mb_loss.backward()

            total_loss_val += float(mb_loss.detach().cpu()) * n_mbs
            total_kl_val += float((kl * mask).sum().detach().cpu() / denom) / n_mbs

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step_s = time.monotonic() - step_t0
        elapsed_total = elapsed_offset_s + (time.monotonic() - t_start)
        mean_r = sum(rewards) / max(1, len(rewards))
        n_correct = sum(1 for d in details if d["label"] == CORRECT)
        n_refusal = sum(1 for d in details if d["label"] == REFUSAL)
        n_corrected = sum(1 for d in details if d["label"] == "CORRECTED")
        n_halluc = sum(1 for d in details if d["label"] == "HALLUCINATED")
        cot_penalties = sum(1 for d in details if d["r_cot"] < 0)
        press_penalties = sum(1 for d in details if d["r_press"] < 0)

        log_rec = {
            "step": step,
            "max_steps": max_steps,
            "variant": args.variant,
            "loss": round(total_loss_val, 5),
            "kl": round(total_kl_val, 5),
            "mean_reward": round(mean_r, 4),
            "n_correct": n_correct,
            "n_refusal": n_refusal,
            "n_corrected": n_corrected,
            "n_hallucinated": n_halluc,
            "cot_penalties": cot_penalties,
            "press_penalties": press_penalties,
            "rescued": rescued_count,
            "step_s": round(step_s, 2),
            "elapsed_s": round(elapsed_total, 1),
        }
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log_rec) + "\n")

        if step % int(grpo_cfg.get("logging_steps", 5)) == 0 or step == 1 or step == max_steps:
            print(
                f"[{args.variant}] step={step}/{max_steps} R={mean_r:+.3f} "
                f"loss={total_loss_val:.4f} kl={total_kl_val:.4f} "
                f"cot_pen={cot_penalties} press_pen={press_penalties} "
                f"rescued={rescued_count} ({step_s:.1f}s/step)",
                flush=True,
            )

        if save_steps > 0 and step % save_steps == 0 and step < max_steps:
            ckpt_dir = out_dir / f"checkpoint-{step}"
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            torch.save(
                {
                    "step": step,
                    "pool_cursor": pool_cursor,
                    "elapsed_s": elapsed_total,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "item_pool_ids": [it["sample_id"] for it in item_pool],
                },
                ckpt_dir / "trainer_state.pt",
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_dir = out_dir / "final_lora"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"[{args.variant}] Training complete! Saved final LoRA to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
