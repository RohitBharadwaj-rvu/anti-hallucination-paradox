"""
Composite Epistemic Reward & Prompt-Centered Advantage Normalization for LACE-GRPO.

Supports all RL variants in the Stage 2 experimental matrix:
  - lace_full              (M1: Full LACE-GRPO)
  - truth_rl               (B2: Static +1 / 0 / -1 ternary GRPO baseline)
  - ablation_no_latent     (A1: LACE-GRPO w/o LatentAnchor -> flat 0 refusal reward)
  - ablation_no_cot        (A2: LACE-GRPO w/o CoT Penalty -> beta_censor = 0)
  - ablation_no_pressure   (A3: LACE-GRPO w/o Pressure Sweep -> neutral-only)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from src.grader import CORRECT, CORRECTED, EMPTY, HALLUCINATED, INCORRECT, REFUSAL, REFUSED
from stage2_lace_grpo.rewards.hybrid_verifier import (
    FalsePremiseNLIVerifier,
    grade_rollouts_batch,
)


@dataclass
class RewardConfig:
    mode: str = "lace_full"  # lace_full | truth_rl | ablation_no_latent | ablation_no_cot | ablation_no_pressure
    lambda_cal: float = 0.5
    lambda_sup: float = 0.6
    beta_censor: float = 0.5
    gamma_sup: float = 0.4
    gamma_hal: float = 0.4
    max_concise_words: int = 40
    fmt_bonus: float = 0.1
    fmt_penalty: float = -0.5

    @classmethod
    def from_variant(cls, variant: str, **overrides: Any) -> "RewardConfig":
        cfg = cls(mode=variant)
        if variant == "ablation_no_cot":
            cfg.beta_censor = 0.0
        elif variant == "ablation_no_pressure":
            cfg.gamma_sup = 0.0
            cfg.gamma_hal = 0.0
        elif variant == "truth_rl":
            cfg.beta_censor = 0.0
            cfg.gamma_sup = 0.0
            cfg.gamma_hal = 0.0
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def compute_epistemic_reward(
    premise: bool,
    label: str,
    k_hat: float,
    cfg: RewardConfig,
) -> float:
    if cfg.mode == "truth_rl":
        if premise:
            if label == CORRECT:
                return 1.0
            if label == REFUSAL:
                return 0.0
            return -1.0
        else:
            if label in (CORRECTED, REFUSED):
                return 1.0 if label == CORRECTED else 0.0
            return -1.0

    # LACE-GRPO & ablations
    if premise:
        if label == CORRECT:
            return 1.0
        if label == REFUSAL:
            if cfg.mode == "ablation_no_latent":
                return 0.0
            k = max(0.0, min(1.0, float(k_hat)))
            return cfg.lambda_cal * (1.0 - k) - cfg.lambda_sup * k
        return -1.0
    else:
        if label == CORRECTED:
            return 1.0
        if label == REFUSED:
            return 0.7
        if label == HALLUCINATED:
            return -1.2
        return -1.0


def compute_cot_reward(
    premise: bool,
    label: str,
    gold_in_cot: bool,
    false_flag_in_cot: bool,
    cfg: RewardConfig,
) -> float:
    if cfg.beta_censor == 0.0:
        return 0.0
    if premise and gold_in_cot and label != CORRECT:
        return -cfg.beta_censor
    if (not premise) and false_flag_in_cot and label == HALLUCINATED:
        return -cfg.beta_censor
    return 0.0


def compute_format_reward(
    label: str,
    answer_text: str,
    is_unclosed_cot: bool,
    cfg: RewardConfig,
) -> float:
    if cfg.mode == "truth_rl":
        return -0.5 if (label == EMPTY or is_unclosed_cot) else 0.0
    if label == EMPTY or is_unclosed_cot or not answer_text.strip():
        return cfg.fmt_penalty
    word_count = len(answer_text.strip().split())
    if word_count <= cfg.max_concise_words:
        return cfg.fmt_bonus
    return 0.0


def score_grouped_rollouts(
    items: list[dict[str, Any]],
    prompt_levels: list[str],
    raw_outputs: list[str],
    cfg: RewardConfig,
    nli_verifier: FalsePremiseNLIVerifier | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """
    Score a flat batch of rollouts grouped by `sample_id`.

    Each item must have: `sample_id`, `premise`, `expected_answers`,
    optional `explanation`, and optional `k_hat` (default 0.0).
    `prompt_levels[i]` is one of {"neutral", "conservative", "permissive"}.

    Returns (total_rewards, detailed_records).
    """
    assert len(items) == len(prompt_levels) == len(raw_outputs)
    graded = grade_rollouts_batch(items, raw_outputs, nli_verifier=nli_verifier)

    # Group indices by sample_id for counterfactual prompt-pressure invariance
    groups: dict[str, list[int]] = {}
    for idx, it in enumerate(items):
        sid = str(it.get("sample_id", idx))
        groups.setdefault(sid, []).append(idx)

    any_correct_neut_perm: dict[str, bool] = {}
    any_safe_neut_cons: dict[str, bool] = {}
    for sid, idxs in groups.items():
        any_correct_neut_perm[sid] = any(
            prompt_levels[i] in ("neutral", "permissive") and graded[i]["label"] == CORRECT
            for i in idxs
        )
        any_safe_neut_cons[sid] = any(
            prompt_levels[i] in ("neutral", "conservative")
            and graded[i]["label"] in (REFUSAL, REFUSED, CORRECTED)
            for i in idxs
        )

    rewards: list[float] = []
    for idx, (item, level, g) in enumerate(zip(items, prompt_levels, graded)):
        sid = str(item.get("sample_id", idx))
        premise = bool(item.get("premise", True))
        k_hat = float(item.get("k_hat", 0.0) or 0.0)
        lbl = g["label"]

        r_epi = compute_epistemic_reward(premise, lbl, k_hat, cfg)
        r_cot = compute_cot_reward(
            premise, lbl, g["gold_in_cot"], g["false_flag_in_cot"], cfg
        )
        r_fmt = compute_format_reward(lbl, g["answer_text"], g["is_unclosed_cot"], cfg)

        r_press = 0.0
        if cfg.mode not in ("truth_rl", "ablation_no_pressure"):
            if (
                level == "conservative"
                and premise
                and lbl == REFUSAL
                and any_correct_neut_perm[sid]
            ):
                r_press = -cfg.gamma_sup
            elif (
                level == "permissive"
                and ((not premise) or k_hat == 0.0)
                and lbl in (HALLUCINATED, INCORRECT)
                and any_safe_neut_cons[sid]
            ):
                r_press = -cfg.gamma_hal

        total = r_epi + r_cot + r_press + r_fmt
        rewards.append(total)
        g.update(
            {
                "sample_id": sid,
                "level": level,
                "k_hat": k_hat,
                "r_epistemic": r_epi,
                "r_cot": r_cot,
                "r_press": r_press,
                "r_fmt": r_fmt,
                "reward": total,
            }
        )

    return rewards, graded


def compute_prompt_centered_advantages(
    rewards: list[float],
    sample_ids: list[str],
    prompt_levels: list[str],
    eps: float = 1e-4,
) -> list[float]:
    """
    Prompt-centered baseline with pooled group standard deviation (LACE_GRPO_DESIGN §4.2.5):
      mu_p(x) = mean_{j in G_p} R(x, p, y_j)
      A_i = (R(x, p_i, y_i) - mu_{p_i}(x)) / (sigma_group(x) + eps)
    """
    n = len(rewards)
    assert n == len(sample_ids) == len(prompt_levels)
    groups: dict[str, list[int]] = {}
    for i, sid in enumerate(sample_ids):
        groups.setdefault(sid, []).append(i)

    advantages = [0.0] * n
    for sid, idxs in groups.items():
        grp_r = [rewards[i] for i in idxs]
        mean_all = sum(grp_r) / len(grp_r)
        var_all = sum((r - mean_all) ** 2 for r in grp_r) / max(1, len(grp_r))
        std_group = math.sqrt(var_all)
        if std_group < 1e-8:
            # Zero variance across entire group -> 0 advantage (unless rescued upstream)
            continue

        by_lvl: dict[str, list[float]] = {}
        for i in idxs:
            by_lvl.setdefault(prompt_levels[i], []).append(rewards[i])
        lvl_means = {lvl: sum(vs) / len(vs) for lvl, vs in by_lvl.items()}

        for i in idxs:
            advantages[i] = (rewards[i] - lvl_means[prompt_levels[i]]) / (std_group + eps)

    return advantages
