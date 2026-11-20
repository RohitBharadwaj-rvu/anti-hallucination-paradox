"""
Baseline B1: R-Tuning (Confidence-Split Supervised Fine-Tuning).

Uses Phase 1 probing results (k_hat(x)) to partition true-premise items into:
  - Known (k_hat >= 0.5): SFT target = cached verified answer (or gold answer)
  - Unknown (k_hat < 0.5) or False-premise: SFT target = calibrated refusal / correction
Trained under TARGET_NEUTRAL with LoRA on a single A100 GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import TARGET_NEUTRAL
from stage2_lace_grpo.probing.probe_latent_kappa import format_chat_prompt


def enable_gemma4_text_sdpa(model: Any, model_id: str) -> None:
    """Switch Gemma 4 text language_model attention from eager to sdpa while keeping audio/vision eager."""
    if "gemma-4" not in model_id.lower():
        return
    try:
        if hasattr(model.config, "text_config"):
            model.config.text_config._attn_implementation = "sdpa"
        lm = getattr(getattr(model, "model", model), "language_model", None)
        if lm is not None:
            if hasattr(lm, "config"):
                lm.config._attn_implementation = "sdpa"
            for layer in getattr(lm, "layers", []):
                sa = getattr(layer, "self_attn", None)
                if sa is not None and hasattr(sa, "config"):
                    sa.config._attn_implementation = "sdpa"
    except Exception:
        pass


def load_base_model_and_tokenizer(
    model_id: str,
    load_in_4bit: bool = False,
    attn_implementation: str | None = "sdpa",
) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        eos = tokenizer.eos_token_id
        tokenizer.pad_token_id = eos[0] if isinstance(eos, list) else eos
    tokenizer.padding_side = "right"

    if "gemma-4" in model_id.lower() and attn_implementation == "sdpa":
        attn_implementation = "eager"

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        load_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _try_load(kwargs: dict[str, Any]) -> Any:
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except ValueError as ve:
            if "scaled_dot_product_attention" in str(ve) or "attn_implementation" in str(ve):
                raise
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
        except Exception:
            return AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)

    try:
        model = _try_load(load_kwargs)
    except ValueError as ve:
        if "scaled_dot_product_attention" in str(ve) or "eager" in str(ve):
            load_kwargs["attn_implementation"] = "eager"
            model = _try_load(load_kwargs)
        else:
            raise

    enable_gemma4_text_sdpa(model, model_id)
    return model, tokenizer


def apply_text_lora(model: Any, lora_cfg: dict[str, Any], is_4bit: bool = False) -> Any:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if is_4bit:
        model = prepare_model_for_kbit_training(model)

    # Check if multimodal model (e.g. Gemma4ForConditionalGeneration with language_model)
    has_lang_model = any("language_model" in n for n, _ in model.named_modules())
    base_targets = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if has_lang_model:
        target_pattern = (
            r".*language_model.*\.(" + "|".join(base_targets) + r")$"
        )
    else:
        target_pattern = base_targets

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 64)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        target_modules=target_pattern,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


class RTuningDataset(Dataset):
    def __init__(
        self,
        items: list[dict],
        kappa_by_id: dict[str, dict],
        tokenizer: Any,
        enable_thinking: bool = True,
        max_seq_length: int = 1024,
        k_threshold: float = 0.5,
    ) -> None:
        self.samples: list[dict[str, torch.Tensor]] = []
        for it in items:
            sid = it["sample_id"]
            premise = bool(it.get("premise", True))
            if premise:
                k_info = kappa_by_id.get(sid, {})
                k_hat = float(k_info.get("k_hat", 0.0) or 0.0)
                if k_hat >= k_threshold:
                    target = (
                        k_info.get("cached_correct_answer")
                        or (it.get("expected_answers") or [""])[0]
                    )
                else:
                    target = "I do not know."
            else:
                expl = it.get("explanation") or ""
                if expl:
                    target = f"I cannot answer this because the premise is false: {expl}"
                else:
                    target = "I do not know; the question contains a false premise."

            prompt_str = format_chat_prompt(
                tokenizer, TARGET_NEUTRAL, it["question"], enable_thinking=enable_thinking
            )
            eos_tok = tokenizer.eos_token or ""
            full_str = prompt_str + target + eos_tok

            prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            full_enc = tokenizer(
                full_str,
                add_special_tokens=False,
                truncation=True,
                max_length=max_seq_length,
            )
            input_ids = full_enc["input_ids"]
            labels = [-100] * min(len(prompt_ids), len(input_ids)) + input_ids[
                min(len(prompt_ids), len(input_ids)) :
            ]
            if all(l == -100 for l in labels):
                continue
            self.samples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(full_enc["attention_mask"], dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def _collate_fn(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids, attn_mask, labels = [], [], []
    for b in batch:
        pad_n = max_len - b["input_ids"].shape[0]
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_n,), pad_id, dtype=torch.long)]))
        attn_mask.append(torch.cat([b["attention_mask"], torch.zeros(pad_n, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_n,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attn_mask),
        "labels": torch.stack(labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline B1: R-Tuning (Confidence-Split SFT)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--kappa-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    sft_cfg = cfg.get("sft_rtuning", {})

    with open(args.train_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    with open(args.kappa_path, "r", encoding="utf-8") as f:
        kappa_by_id = json.load(f).get("kappa_by_id", {})

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    use_4bit = bool(model_cfg.get("load_in_4bit", False)) or (
        "8B" in model_cfg["model_id"] and vram_gb < 60.0
    )
    model, tokenizer = load_base_model_and_tokenizer(
        model_cfg["model_id"],
        load_in_4bit=use_4bit,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model = apply_text_lora(model, cfg["lora"], is_4bit=use_4bit)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    ds = RTuningDataset(
        items=items,
        kappa_by_id=kappa_by_id,
        tokenizer=tokenizer,
        enable_thinking=bool(model_cfg.get("enable_thinking", True)),
        max_seq_length=int(sft_cfg.get("max_seq_length", 1024)),
    )
    bs = int(sft_cfg.get("per_device_batch_size", 8))
    grad_accum = int(sft_cfg.get("gradient_accumulation_steps", 4))
    epochs = int(sft_cfg.get("epochs", 2))
    lr = float(sft_cfg.get("learning_rate", 2e-5))

    dl = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        collate_fn=lambda b: _collate_fn(b, tokenizer.pad_token_id),
    )

    total_opt_steps = (len(dl) * epochs) // grad_accum
    if args.max_steps is not None:
        total_opt_steps = min(total_opt_steps, args.max_steps)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_opt_steps)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    model.train()
    opt_step = 0
    micro_step = 0
    running_loss = 0.0
    t0 = time.monotonic()

    for epoch in range(epochs):
        for batch in dl:
            if args.max_steps is not None and opt_step >= args.max_steps:
                break
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            running_loss += float(out.loss.detach().cpu())
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 5 == 0 or opt_step == total_opt_steps:
                    elapsed = time.monotonic() - t0
                    rec = {
                        "step": opt_step,
                        "epoch": epoch,
                        "loss": running_loss / grad_accum,
                        "elapsed_s": elapsed,
                    }
                    print(f"[R-Tuning] step={opt_step}/{total_opt_steps} loss={rec['loss']:.4f} elapsed={elapsed:.1f}s")
                    with open(log_path, "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(rec) + "\n")
                running_loss = 0.0
        if args.max_steps is not None and opt_step >= args.max_steps:
            break

    model.save_pretrained(str(out_dir / "final_lora"))
    tokenizer.save_pretrained(str(out_dir / "final_lora"))
    print(f"[R-Tuning] Saved LoRA adapter to {out_dir / 'final_lora'}")


if __name__ == "__main__":
    main()
