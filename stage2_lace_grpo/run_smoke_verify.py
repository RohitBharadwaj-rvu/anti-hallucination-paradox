"""One-shot smoke verification on Colab GPU: NLI + Gemma 4 E4B LoRA + 16-item probing + 10-step LACE-GRPO."""
from __future__ import annotations

import os
import subprocess
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
    import yaml

    from src.prompts import TARGET_NEUTRAL
    from stage2_lace_grpo.probing.probe_latent_kappa import format_chat_prompt, run_probing
    from stage2_lace_grpo.rewards.hybrid_verifier import (
        FalsePremiseNLIVerifier,
        extract_cot_and_answer,
    )
    from stage2_lace_grpo.train_r_tuning import (
        apply_text_lora,
        load_base_model_and_tokenizer,
    )

    print("=== [1/4] Verifying DeBERTa NLI Verifier ===", flush=True)
    t0 = time.monotonic()
    nli = FalsePremiseNLIVerifier(lazy=False)
    sc = nli.entailment_scores(
        ["Mars has no moons; that premise is wrong."],
        ["Mars actually has two moons, Phobos and Deimos."],
    )
    print(f"NLI loaded in {time.monotonic() - t0:.1f}s; entailment={sc}", flush=True)
    del nli
    torch.cuda.empty_cache()

    print("=== [2/4] Verifying Gemma 4 E4B LoRA + generate + backward + disable_adapter ===", flush=True)
    t0 = time.monotonic()
    cfg = yaml.safe_load(
        (PROJECT_ROOT / "stage2_lace_grpo/configs/a100_gemma4_e4b.yaml").read_text(encoding="utf-8")
    )
    model, tok = load_base_model_and_tokenizer(
        "google/gemma-4-E4B-it", load_in_4bit=False, attn_implementation="sdpa"
    )
    print(f"Loaded {type(model).__name__} in {time.monotonic() - t0:.1f}s", flush=True)
    model = apply_text_lora(model, cfg["lora"], is_4bit=False)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    prompt = format_chat_prompt(tok, TARGET_NEUTRAL, "What is the capital of France?", enable_thinking=True)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    model.eval()
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id)
    out_txt = tok.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=False)
    print(f"Raw Gemma4 output: {out_txt[:350]!r}", flush=True)
    print(f"Parsed CoT/Ans: {extract_cot_and_answer(out_txt)}", flush=True)

    model.train()
    out = model(**enc, labels=enc["input_ids"])
    out.loss.backward()
    with torch.no_grad():
        with model.disable_adapter():
            _ = model(**enc)
    print(
        f"Backward + disable_adapter OK! loss={float(out.loss):.4f} "
        f"VRAM_peak={torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
        flush=True,
    )
    del model
    torch.cuda.empty_cache()

    print("=== [3/4] Running 16-item mini Phase 1 Probing ===", flush=True)
    t0 = time.monotonic()
    smoke_dir = PROJECT_ROOT / "outputs/smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    kappa_smoke = smoke_dir / "kappa_smoke16.json"
    stats = run_probing(
        model_id="google/gemma-4-E4B-it",
        train_path=PROJECT_ROOT / "stage2_lace_grpo/data/train_4000.json",
        out_path=kappa_smoke,
        m_samples=4,
        max_new_tokens=384,
        enable_thinking=True,
        use_vllm=False,
        max_items=16,
        batch_size=8,
    )
    print(f"Mini-probing (16x4=64 rollouts) in {time.monotonic() - t0:.1f}s: {stats}", flush=True)

    print("=== [4/4] Running 10-step LACE-GRPO Smoke Test (first 32 train items) ===", flush=True)
    import json
    all_train = json.loads(
        (PROJECT_ROOT / "stage2_lace_grpo/data/train_4000.json").read_text(encoding="utf-8")
    )
    true_32 = [it for it in all_train if it.get("premise")][:22]
    false_10 = [it for it in all_train if not it.get("premise")][:10]
    smoke_train = smoke_dir / "train_smoke32.json"
    smoke_train.write_text(json.dumps(true_32 + false_10, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "stage2_lace_grpo/train_lace_grpo.py"),
        "--config",
        str(PROJECT_ROOT / "stage2_lace_grpo/configs/a100_gemma4_e4b.yaml"),
        "--variant",
        "lace_full",
        "--train-path",
        str(smoke_train),
        "--kappa-path",
        str(kappa_smoke),
        "--output-dir",
        str(smoke_dir / "grpo_10step"),
        "--max-steps",
        "10",
        "--gen-batch-size",
        "24",
        "--micro-batch-size",
        "4",
    ]
    t0 = time.monotonic()
    subprocess.run(cmd, check=True)
    print(f"=== SMOKE TEST COMPLETE in {time.monotonic() - t0:.1f}s ===", flush=True)


if __name__ == "__main__":
    main()
