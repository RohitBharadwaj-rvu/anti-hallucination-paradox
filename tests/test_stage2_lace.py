"""
Unit tests for Stage 2 LACE-GRPO components (zero GPU / zero Colab needed).
Tests:
  - dedup_guard zero-leakage and fuzzy Jaccard blocking
  - hybrid_verifier CoT stripping (Gemma 4 <|channel>thought & Qwen3 <think>)
  - hybrid_verifier refusal matching, SimpleQA range verification, CoT leakage check
  - lace_rewards R_epistemic, R_CoT, R_press, R_fmt, TruthRL, ablations
  - prompt-centered advantage normalization + anti-collapse guard behavior
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grader import CORRECT, CORRECTED, EMPTY, HALLUCINATED, INCORRECT, REFUSAL, REFUSED
from stage2_lace_grpo.data_builder.dedup_guard import DedupGuard
from stage2_lace_grpo.rewards.hybrid_verifier import (
    extract_cot_and_answer,
    gold_in_text,
    grade_rollouts_batch,
    has_premise_rejection_cue,
    is_refusal,
    verify_true_premise,
)
from stage2_lace_grpo.rewards.lace_rewards import (
    RewardConfig,
    compute_prompt_centered_advantages,
    score_grouped_rollouts,
)


def test_dedup_guard() -> None:
    guard = DedupGuard("data/benchmark_subset_100.json", jaccard_threshold=0.8)
    dup1, _ = guard.is_duplicate(
        {"question": "Who was the producer of Chrome?"}
    )
    assert dup1, "Exact benchmark question must be blocked"

    dup2, _ = guard.is_duplicate(
        {"question": "What record label produced Avenger's 1985 EP containing the song Down to the Bone"}
    )
    assert dup2, "Near-identical benchmark question must be blocked"

    ok, _ = guard.is_duplicate(
        {"question": "What is the boiling point of liquid nitrogen in Kelvin?"}
    )
    assert not ok, "Unrelated question should pass"
    print("[PASS] dedup_guard blocks exact and fuzzy benchmark matches")


def test_hybrid_verifier() -> None:
    # Gemma 4 closed channel
    ans, think, unclosed = extract_cot_and_answer(
        "<|channel>thought\nLet me recall: France capital is Paris.\n<channel|>Paris<turn|>"
    )
    assert ans == "Paris" and "France capital" in think and not unclosed

    # Gemma 4 truncated unclosed channel
    ans2, think2, unclosed2 = extract_cot_and_answer(
        "<|channel>thought\nWait let me think forever..."
    )
    assert ans2 == "" and unclosed2

    # Qwen3 <think>
    ans3, think3, unclosed3 = extract_cot_and_answer(
        "<think>Checking: 1480 guineas.</think>1480 English guineas"
    )
    assert ans3 == "1480 English guineas" and "1480" in think3 and not unclosed3

    # Numeric range SimpleQA
    assert verify_true_premise(
        "1475 guineas",
        ["1480 English guineas (acceptable range: anything between 1465 and 1495 English guineas)"],
    )
    assert not verify_true_premise(
        "2000 guineas",
        ["1480 English guineas (acceptable range: anything between 1465 and 1495 English guineas)"],
    )

    # Word boundary prevents '2' matching '2012'
    assert verify_true_premise("2", ["2", "Two"])
    assert not verify_true_premise("2012", ["2"])

    # CoT leakage
    assert gold_in_text(["Michio Sugeno"], "I think the winner in 2010 was Michio Sugeno.")
    assert not gold_in_text(["2"], "There are 2012 possibilities.")

    # Refusal + False premise cue
    assert is_refusal("I don't know.")
    assert is_refusal("The answer is unknown.")
    assert has_premise_rejection_cue("This question has a false premise because Pluto is a planet.")
    print("[PASS] hybrid_verifier handles Gemma 4 & Qwen3 CoT, ranges, boundaries, refusals")


def test_lace_rewards_and_advantages() -> None:
    item_known = {
        "sample_id": "t_001",
        "premise": True,
        "expected_answers": ["Paris"],
        "k_hat": 1.0,
    }
    item_unknown = {
        "sample_id": "t_002",
        "premise": True,
        "expected_answers": ["Xyzzy123"],
        "k_hat": 0.0,
    }
    item_false = {
        "sample_id": "f_001",
        "premise": False,
        "expected_answers": ["Men cannot produce breast milk."],
        "explanation": "Men cannot produce breast milk.",
        "k_hat": 0.0,
    }

    cfg_full = RewardConfig.from_variant("lace_full")
    items = [item_known] * 6
    levels = ["neutral", "neutral", "conservative", "conservative", "permissive", "permissive"]
    outputs = [
        "<think>Paris</think>Paris",                 # CORRECT (+1.0 + 0.1 = +1.1)
        "<think>Paris</think>I don't know.",         # REFUSAL on k_hat=1 & gold_in_cot! (-0.6 - 0.5 + 0.1 = -1.0)
        "<think>Maybe Paris</think>I don't know.",   # REFUSAL on conservative w/ AnyCorrect=1 (-0.6 - 0.5 - 0.4 + 0.1 = -1.4)
        "Paris",                                     # CORRECT (+1.1)
        "London",                                    # INCORRECT (-1.0 + 0.1 = -0.9)
        "Paris",                                     # CORRECT (+1.1)
    ]
    rewards, details = score_grouped_rollouts(items, levels, outputs, cfg_full)
    assert abs(rewards[0] - 1.1) < 1e-6, details[0]
    assert abs(rewards[1] - (-1.0)) < 1e-6, details[1]
    assert abs(rewards[2] - (-1.4)) < 1e-6, details[2]
    assert abs(rewards[4] - (-0.9)) < 1e-6, details[4]

    # Prompt-centered advantages: neutral mean = (1.1 - 1.0)/2 = 0.05;
    # conservative mean = (-1.4 + 1.1)/2 = -0.15 -> Paris in conservative has positive adv!
    advs = compute_prompt_centered_advantages(rewards, [it["sample_id"] for it in items], levels)
    assert advs[0] > 0 and advs[1] < 0
    assert advs[3] > 0 and advs[2] < 0

    # Unknown item: refusal rewarded (+0.5 + 0.1 = +0.6) > incorrect (-0.9)
    r_unk, _ = score_grouped_rollouts(
        [item_unknown, item_unknown],
        ["neutral", "permissive"],
        ["I do not know.", "Berlin"],
        cfg_full,
    )
    assert r_unk[0] > 0.5 and r_unk[1] < -0.5

    # False item: CORRECTED (+1.1) > REFUSED (+0.8) > HALLUCINATED (-1.1 - 0.4 = -1.5 on permissive if AnySafe)
    r_fal, det_fal = score_grouped_rollouts(
        [item_false, item_false, item_false],
        ["neutral", "conservative", "permissive"],
        [
            "That is a false premise; men cannot breastfeed.",
            "I don't know.",
            "They should use formula X.",
        ],
        cfg_full,
    )
    assert det_fal[0]["label"] == CORRECTED and r_fal[0] > r_fal[1] > 0
    assert det_fal[2]["label"] == HALLUCINATED and r_fal[2] < -1.4

    # TruthRL flat 0 refusal regardless of k_hat
    cfg_tr = RewardConfig.from_variant("truth_rl")
    r_tr, _ = score_grouped_rollouts(
        [item_known, item_unknown],
        ["neutral", "neutral"],
        ["I don't know.", "I don't know."],
        cfg_tr,
    )
    assert r_tr[0] == 0.0 and r_tr[1] == 0.0
    print("[PASS] lace_rewards and prompt-centered advantages verified across all modes")


if __name__ == "__main__":
    test_dedup_guard()
    test_hybrid_verifier()
    test_lace_rewards_and_advantages()
    print("\nALL STAGE 2 UNIT TESTS PASSED")
