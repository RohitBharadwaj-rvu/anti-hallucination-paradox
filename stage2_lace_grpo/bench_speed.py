"""Benchmark probing batch throughput & 1-step GRPO speed/VRAM on Gemma 4 E4B."""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_TOKEN", "")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from stage2_lace_grpo.probing.probe_latent_kappa import format_chat_prompt
    from stage2_lace_grpo.train_r_tuning import apply_text_lora
    from src.prompts import TARGET_NEUTRAL, TARGET_CONSERVATIVE, TARGET_PERMISSIVE

    model_id = "google/gemma-4-E4B-it"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        eos = tok.eos_token_id
        tok.pad_token_id = eos[0] if isinstance(eos, list) else eos

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )
    # Switch language_model to sdpa
    model.config.text_config._attn_implementation = "sdpa"
    model.model.language_model.config._attn_implementation = "sdpa"
    for layer in model.model.language_model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "sdpa"

    items = json.loads((PROJECT_ROOT / "stage2_lace_grpo/data/train_4000.json").read_text(encoding="utf-8"))[:64]

    # 1. Test generation throughput at bs=48 (all 48 GRPO rollouts in 1 batch)
    prompts48 = [
        format_chat_prompt(tok, [TARGET_NEUTRAL, TARGET_CONSERVATIVE, TARGET_PERMISSIVE][i % 3], items[i % 8]["question"], True)
        for i in range(48)
    ]
    model.eval()
    tok.padding_side = "left"
    enc48 = tok(prompts48, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    with torch.no_grad():
        gen48 = model.generate(
            **enc48, do_sample=True, temperature=0.7, top_p=0.95,
            max_new_tokens=512, pad_token_id=tok.pad_token_id, use_cache=True,
        )
    dt48 = time.monotonic() - t0
    print(f"[GEN bs=48, max_new=512] time={dt48:.1f}s ({dt48/48:.2f}s/seq) peak_vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    # 2. Test probing batching: 16 prompts x 6 samples = 96 seqs in 1 call
    prompts16 = [
        format_chat_prompt(tok, TARGET_PERMISSIVE, it["question"], True)
        for it in items[:16]
    ]
    enc16 = tok(prompts16, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=False).to(model.device)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    with torch.no_grad():
        gen96 = model.generate(
            **enc16, do_sample=True, temperature=0.7, top_p=0.95,
            max_new_tokens=512, num_return_sequences=6, pad_token_id=tok.pad_token_id, use_cache=True,
        )
    dt96 = time.monotonic() - t0
    print(f"[PROBE 16x6=96 seqs, max_new=512] time={dt96:.1f}s ({dt96/16:.2f}s/item) peak_vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    # 3. Test LoRA forward+backward with F.cross_entropy & logits_to_keep check
    del gen48, gen96, enc48, enc16
    gc.collect()
    torch.cuda.empty_cache()

    cfg = yaml.safe_load((PROJECT_ROOT / "stage2_lace_grpo/configs/a100_gemma4_e4b.yaml").read_text(encoding="utf-8"))
    model = apply_text_lora(model, cfg["lora"], is_4bit=False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    dummy_in = torch.randint(10, 1000, (2, 600), device=model.device)
    dummy_am = torch.ones_like(dummy_in)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    model.train()
    out = model(input_ids=dummy_in, attention_mask=dummy_am)
    logits = out.logits[:, :-1, :]
    labels = dummy_in[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").mean()
    loss.backward()
    print(f"[TRAIN micro_bs=2 seq=600 cross_entropy] time={time.monotonic()-t0:.2f}s peak_vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    # Test 48 seqs total (24 microbatches of 2: old + ref + cur)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    model.eval()
    for _ in range(24):
        with torch.no_grad():
            _ = model(input_ids=dummy_in, attention_mask=dummy_am).logits
            with model.disable_adapter():
                _ = model(input_ids=dummy_in, attention_mask=dummy_am).logits
    t_noref = time.monotonic() - t0
    model.train()
    t1 = time.monotonic()
    for _ in range(24):
        out = model(input_ids=dummy_in, attention_mask=dummy_am)
        l = F.cross_entropy(out.logits[:, :-1, :].reshape(-1, out.logits.shape[-1]), dummy_in[:, 1:].reshape(-1))
        (l / 24).backward()
    t_bwd = time.monotonic() - t1
    print(f"[48 seqs micro_bs=2] old+ref={t_noref:.1f}s cur+bwd={t_bwd:.1f}s peak_vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")


if __name__ == "__main__":
    main()
