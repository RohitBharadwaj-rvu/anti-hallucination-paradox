"""
src/reporter.py — CSV + PNG + HTML report generation.
Implements T1-T10 tables and P1-P9 plots per spec §13 + Deep Audit upgrades.
"""
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
try:
    import seaborn as sns
except ImportError:
    sns = None
from jinja2 import Environment, BaseLoader

from src.utils import get_logger, now_iso
from src.metrics import deduplicate_rows

logger = get_logger("ahp.reporter")


def _draw_heatmap(
    df: pd.DataFrame,
    ax: plt.Axes,
    cmap: str = "Blues",
    fmt: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render an annotated heatmap using seaborn if installed, or pure matplotlib."""
    if sns is not None:
        sns.heatmap(
            df, annot=True, fmt=fmt, cmap=cmap,
            vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5,
        )
        return
    vals = df.values.astype(float)
    im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(list(df.columns))
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(list(df.index))
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if v == v:  # not NaN
                txt = f"{int(v)}" if fmt == "d" else format(v, fmt)
                ax.text(j, i, txt, ha="center", va="center", color="#111", fontweight="bold", fontsize=9)

# ---------------------------------------------------------------------------
# Color scheme (consistent across all plots)
# ---------------------------------------------------------------------------
COLORS = {
    "CORRECT": "#2ECC71",
    "REFUSED": "#27AE60",
    "CORRECTED": "#58D68D",
    "REFUSAL": "#F39C12",   # amber — suppression signal
    "INCORRECT": "#E74C3C",
    "HALLUCINATED": "#C0392B",
    "EMPTY": "#BDC3C7",
    "JUDGE_ERROR": "#7F8C8D",
}
LEVEL_ORDER = ["permissive", "neutral", "conservative"]
LEVEL_COLORS = {
    "permissive": "#3498DB",
    "neutral": "#95A5A6",
    "conservative": "#E67E22",
}


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved figure: %s", path)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_p1_outcome_distribution(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> str | None:
    """
    P1: Two-panel 100%-stacked bar of outcome distribution @ neutral.
    Left = true premise (CORRECT/REFUSAL/INCORRECT), right = false premise.
    """
    try:
        fig, (ax_true, ax_false) = plt.subplots(1, 2, figsize=(12, 5))

        for ax, premise_val, labels, title in [
            (ax_true, True, ["CORRECT", "REFUSAL", "INCORRECT"],
             "True-Premise Outcomes @ Neutral\n(REFUSAL = behavioral suppression / abstention)"),
            (ax_false, False, ["REFUSED", "CORRECTED", "HALLUCINATED"],
             "False-Premise Outcomes @ Neutral"),
        ]:
            model_names = list(all_runs_rows.keys())
            bottoms = [0.0] * len(model_names)
            for lbl in labels:
                vals = []
                for mn in model_names:
                    rows = [r for r in all_runs_rows[mn]
                            if r.get("level") == "neutral" and r.get("premise") is premise_val]
                    valid = [r for r in rows if r.get("label") not in ("EMPTY", "JUDGE_ERROR")]
                    n = len(valid)
                    v = sum(1 for r in valid if r.get("label") == lbl) / n if n > 0 else 0
                    vals.append(v * 100)
                color = COLORS.get(lbl, "#95A5A6")
                ax.bar(model_names, vals, bottom=bottoms, label=lbl, color=color)
                bottoms = [b + v for b, v in zip(bottoms, vals)]
            ax.set_ylim(0, 100)
            ax.set_ylabel("% of responses")
            ax.set_title(title, fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.tick_params(axis="x", rotation=15)

        fig.suptitle("P1: Outcome Distribution @ Neutral", fontsize=11)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P1_outcome_distribution.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P1 plot failed: %s", e)
        return None


def plot_p2_refusal_rate_by_level(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> str | None:
    """P2: Line plot x=level, y=refusal_rate_true, one line per model."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mn, rows in all_runs_rows.items():
            vals = []
            for level in LEVEL_ORDER:
                lvl_rows = [r for r in rows if r.get("level") == level and r.get("premise") is True]
                valid = [r for r in lvl_rows if r.get("label") not in ("EMPTY", "JUDGE_ERROR")]
                n = len(valid)
                rate = sum(1 for r in valid if r.get("label") == "REFUSAL") / n if n > 0 else float("nan")
                vals.append(rate * 100 if rate == rate else float("nan"))
            ax.plot(LEVEL_ORDER, vals, marker="o", linewidth=2, label=mn)

        ax.set_xlabel("Prompt Level")
        ax.set_ylabel("Refusal Rate (true premise) %")
        ax.set_title(
            "P2: Refusal Rate under Alignment Pressure\n"
            "(rising line = alignment-pressure-induced refusal on true premises)",
            fontsize=10,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P2_refusal_rate_by_level.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P2 plot failed: %s", e)
        return None


def plot_p3_hallucination_by_level(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> str | None:
    """P3: Line plot x=level, y=hallucination_rate_false."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mn, rows in all_runs_rows.items():
            vals = []
            for level in LEVEL_ORDER:
                lvl_rows = [r for r in rows if r.get("level") == level and r.get("premise") is False]
                valid = [r for r in lvl_rows if r.get("label") not in ("EMPTY", "JUDGE_ERROR")]
                n = len(valid)
                rate = sum(1 for r in valid if r.get("label") == "HALLUCINATED") / n if n > 0 else float("nan")
                vals.append(rate * 100 if rate == rate else float("nan"))
            ax.plot(LEVEL_ORDER, vals, marker="o", linewidth=2, label=mn)

        ax.set_xlabel("Prompt Level")
        ax.set_ylabel("Hallucination Rate (false premise) %")
        ax.set_title(
            "P3: Hallucination Rate under Alignment Pressure\n"
            "(falling line = good calibration under caution; high permissive = false-premise vulnerability)",
            fontsize=10,
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P3_hallucination_by_level.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P3 plot failed: %s", e)
        return None


def plot_p4_recovery_rates(
    all_model_results: dict[str, dict],
    output_dir: str,
) -> str | None:
    """P4: Grouped bar per model: [recovery_rate_true, recovery_rate_false]."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        model_names = list(all_model_results.keys())
        x = range(len(model_names))
        width = 0.35

        true_vals = [
            (all_model_results[mn].get("recovery", {}).get("recovery_rate_true") or 0) * 100
            for mn in model_names
        ]
        false_vals = [
            (all_model_results[mn].get("recovery", {}).get("recovery_rate_false") or 0) * 100
            for mn in model_names
        ]

        ax.bar([i - width/2 for i in x], true_vals, width, label="Recovery (true premise)", color=COLORS["CORRECT"])
        ax.bar([i + width/2 for i in x], false_vals, width, label="Recovery (false premise)", color=COLORS["CORRECTED"])
        ax.set_xticks(list(x))
        ax.set_xticklabels(model_names, rotation=15)
        ax.set_ylabel("Recovery Rate %")
        ax.set_title("P4: Stage-1 Recovery Rates\n(behavioral evidence of recoverability under direct-answer nudge)", fontsize=10)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P4_recovery_rates.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P4 plot failed: %s", e)
        return None


def plot_p5_calibration_heatmap(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> str | None:
    """P5: Heatmap model x level of calibration_success_false (annotated, RdYlGn)."""
    try:
        model_names = list(all_runs_rows.keys())
        matrix = []
        for mn in model_names:
            row_vals = []
            for level in LEVEL_ORDER:
                lvl_rows = [
                    r for r in all_runs_rows[mn]
                    if r.get("level") == level and r.get("premise") is False
                ]
                valid = [r for r in lvl_rows if r.get("label") not in ("EMPTY", "JUDGE_ERROR")]
                n = len(valid)
                rate = (
                    sum(1 for r in valid if r.get("label") in ("REFUSED", "CORRECTED")) / n
                    if n > 0 else float("nan")
                )
                row_vals.append(rate)
            matrix.append(row_vals)

        df = pd.DataFrame(matrix, index=model_names, columns=LEVEL_ORDER)
        fig, ax = plt.subplots(figsize=(7, max(3, len(model_names))))
        _draw_heatmap(df, ax=ax, cmap="RdYlGn", fmt=".2f", vmin=0, vmax=1)
        ax.set_title("P5: Calibration Success (false premise) by Level", fontsize=10)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P5_calibration_heatmap.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P5 plot failed: %s", e)
        return None


def plot_p6_popqa_longtail(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> str | None:
    """P6: PopQA only: bin o_pop into quartiles; line of correct_rate_true per quartile per model."""
    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        any_data = False
        for mn, rows in all_runs_rows.items():
            popqa = [
                r for r in rows
                if r.get("dataset_name") == "PopQA"
                and r.get("level") == "neutral"
                and r.get("premise") is True
                and r.get("label") not in ("EMPTY", "JUDGE_ERROR")
            ]
            pops = [r.get("o_pop") or r.get("raw_metadata", {}).get("o_pop") for r in popqa]
            pops_valid = [p for p in pops if p is not None]
            if not pops_valid:
                continue
            any_data = True
            pops_series = pd.Series([r.get("raw_metadata", {}).get("o_pop") for r in popqa], dtype=float)
            labels_series = pd.Series([r.get("label") for r in popqa])
            try:
                quartile_labels = pd.qcut(pops_series, q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
                df = pd.DataFrame({"quartile": quartile_labels, "label": labels_series})
                df = df.dropna(subset=["quartile"])
                rates = df.groupby("quartile", observed=True).apply(
                    lambda g: sum(g["label"] == "CORRECT") / len(g) if len(g) > 0 else float("nan")
                )
                ax.plot(rates.index.astype(str), rates.values * 100, marker="o", linewidth=2, label=mn)
            except Exception:
                pass

        if not any_data:
            plt.close(fig)
            return None
        ax.set_xlabel("Popularity Quartile (o_pop)")
        ax.set_ylabel("Correct Rate %")
        ax.set_title("P6: PopQA Long-Tail — Correct Rate by Popularity Quartile", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P6_popqa_longtail.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P6 plot failed: %s", e)
        return None


def plot_p7_accuracy_by_dataset(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> str | None:
    """
    P7: Grouped bar per dataset slice @ neutral.
    Splits FalseQA into FalseQA (True) and FalseQA (False) so mixed premises are never
    conflated into a single denominator, matching Table T2 exactly.
    """
    try:
        slices_spec = [
            ("SimpleQA\n(True)", "SimpleQA", True),
            ("PopQA\n(True)", "PopQA", True),
            ("Bamboogle\n(True)", "Bamboogle", True),
            ("FalseQA\n(True)", "FalseQA", True),
            ("FalseQA\n(False)", "FalseQA", False),
            ("MultiHoax\n(False)", "MultiHoax", False),
        ]
        model_names = list(all_runs_rows.keys())
        n_models = len(model_names)
        n_ds = len(slices_spec)
        width = 0.8 / max(n_models, 1)
        x = range(n_ds)

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, mn in enumerate(model_names):
            vals = []
            for label_str, ds_name, prem_val in slices_spec:
                rows = [
                    r for r in all_runs_rows[mn]
                    if r.get("dataset_name") == ds_name
                    and r.get("level") == "neutral"
                    and r.get("premise") is prem_val
                ]
                valid = [r for r in rows if r.get("label") not in ("EMPTY", "JUDGE_ERROR")]
                n = len(valid)
                if n == 0:
                    vals.append(float("nan"))
                    continue
                if prem_val is True:
                    rate = sum(1 for r in valid if r.get("label") == "CORRECT") / n
                else:
                    rate = sum(1 for r in valid if r.get("label") in ("REFUSED", "CORRECTED")) / n
                vals.append(rate * 100)
            offset = (i - n_models / 2 + 0.5) * width
            ax.bar([xi + offset for xi in x], vals, width, label=mn)

        ax.set_xticks(list(x))
        ax.set_xticklabels([s[0] for s in slices_spec], rotation=0)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Accuracy / Calibration %")
        ax.set_title("P7: Per-Dataset Accuracy @ Neutral (True-Premise Recall vs. False-Premise Calibration)", fontsize=10)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P7_accuracy_by_dataset.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P7 plot failed: %s", e)
        return None


def plot_p8_transition_heatmap(
    all_runs_rows: dict[str, list[dict]],
    output_dir: str,
) -> dict[str, str]:
    """P8: Transition heatmap for TRUE premise: rows=label0, cols=label1 (on triggered set)."""
    paths: dict[str, str] = {}
    labels_s0 = ["REFUSAL", "INCORRECT"]
    labels_s1 = ["CORRECT", "REFUSAL", "INCORRECT", "EMPTY", "JUDGE_ERROR"]
    for mn, rows in all_runs_rows.items():
        try:
            triggered = [
                r for r in rows
                if r.get("premise") is True
                and r.get("level") == "neutral"
                and r.get("label") in ("REFUSAL", "INCORRECT")
                and r.get("label_s1") is not None
            ]
            if not triggered:
                continue
            matrix = pd.DataFrame(0, index=labels_s0, columns=labels_s1)
            for r in triggered:
                l0 = r.get("label", "INCORRECT")
                l1 = r.get("label_s1", "INCORRECT")
                if l0 in matrix.index and l1 in matrix.columns:
                    matrix.loc[l0, l1] += 1

            fig, ax = plt.subplots(figsize=(8, 4))
            _draw_heatmap(matrix, ax=ax, cmap="Blues", fmt="d")
            ax.set_xlabel("Stage 1 (Recovery) Label")
            ax.set_ylabel("Stage 0 (Neutral) Label")
            ax.set_title(
                f"P8: Stage 0→1 Transition Heatmap (True Premise) — {mn}\n"
                "(shows recoverability of initially refused or missed factual questions)",
                fontsize=9,
            )
            plt.tight_layout()
            mn_safe = mn.replace("/", "_").replace(":", "_")
            out_path = str(Path(output_dir) / f"P8_transition_{mn_safe}.png")
            _save_fig(fig, out_path)
            paths[mn] = out_path
        except Exception as e:
            logger.warning("P8 plot for %s failed: %s", mn, e)
    return paths


def plot_p9_failure_taxonomy(
    all_model_results: dict[str, dict],
    output_dir: str,
) -> str | None:
    """
    P9: Two-panel visualization of the 3-Way Diagnostic Failure Taxonomy (T9):
      Left panel: 100%-stacked bar across the 75 true-premise items:
        - Robust Recall
        - Answer Suppression (Known Fact Refused)
        - Retrieval / Hedging Failure
        - Knowledge Loss (Calibrated Abstention)
        - Knowledge Loss (Uncalibrated Hallucination)
      Right panel: Breakdown of Conservative Refusals into:
        - True Answer Suppression (Model Knew the Answer)
        - Calibrated Ignorance Abstention (Model Never Knew the Answer)
    """
    try:
        model_names = [mn for mn, res in all_model_results.items() if res.get("taxonomy")]
        if not model_names:
            return None

        fig, (ax_tax, ax_ref) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Left panel: 100% stacked bar of 75 true-premise items
        categories = [
            ("robust_recall_rate", "Robust Recall (Immune)", "#2ECC71"),
            ("true_suppression_rate", "Answer Suppression (Known -> Refused)", "#F39C12"),
            ("retrieval_failure_rate", "Retrieval / Hedging Failure", "#3498DB"),
            ("calibrated_abstention_rate", "Knowledge Loss (Calibrated Abstention)", "#95A5A6"),
            ("uncalibrated_ignorance_rate", "Knowledge Loss (Uncalibrated Guess)", "#E74C3C"),
        ]
        bottoms = [0.0] * len(model_names)
        for key, label, color in categories:
            vals = [
                (all_model_results[mn]["taxonomy"].get(key) or 0.0) * 100
                for mn in model_names
            ]
            ax_tax.bar(model_names, vals, bottom=bottoms, label=label, color=color)
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax_tax.set_ylim(0, 100)
        ax_tax.set_ylabel("% of True-Premise Items (n=75)")
        ax_tax.set_title("3-Way Diagnostic Taxonomy Across True-Premise Items", fontsize=10)
        ax_tax.legend(loc="upper right", fontsize=8)
        ax_tax.tick_params(axis="x", rotation=15)

        # Right panel: Decomposition of Conservative Refusals (count of items)
        true_supp_counts = [
            all_model_results[mn]["taxonomy"].get("n_cons_refusal_true_suppression", 0)
            for mn in model_names
        ]
        calib_ign_counts = [
            all_model_results[mn]["taxonomy"].get("n_cons_refusal_calibrated_ignorance", 0)
            for mn in model_names
        ]
        ax_ref.bar(
            model_names, true_supp_counts,
            label="True Answer Suppression (Model Knew Answer)", color="#F39C12"
        )
        ax_ref.bar(
            model_names, calib_ign_counts, bottom=true_supp_counts,
            label="Calibrated Ignorance (Model Never Knew Answer)", color="#7F8C8D"
        )
        ax_ref.set_ylabel("Number of Conservative Refusals")
        ax_ref.set_title("Decomposing Conservative Refusals:\nTrue Suppression vs. Calibrated Ignorance", fontsize=10)
        ax_ref.legend(loc="upper right", fontsize=8)
        ax_ref.tick_params(axis="x", rotation=15)
        ax_ref.grid(True, axis="y", alpha=0.3)

        fig.suptitle("P9: Diagnostic Failure Taxonomy — Separating Suppression from Ignorance", fontsize=11)
        plt.tight_layout()
        out_path = str(Path(output_dir) / "P9_failure_taxonomy.png")
        _save_fig(fig, out_path)
        return out_path
    except Exception as e:
        logger.warning("P9 plot failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# HTML Report Template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Anti-Hallucination Paradox — Evaluation Report</title>
  <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f6fa; margin: 0; padding: 0; }
    .container { max-width: 1250px; margin: 0 auto; padding: 24px; }
    h1 { color: #2C3E50; font-size: 2em; border-bottom: 3px solid #3498DB; padding-bottom: 8px; }
    h2 { color: #2980B9; margin-top: 2em; }
    h3 { color: #16A085; }
    .meta-box { background: #fff; border-left: 5px solid #3498DB; padding: 16px; margin-bottom: 20px;
                border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
    .meta-box table { border-collapse: collapse; width: 100%; }
    .meta-box td { padding: 4px 12px; vertical-align: top; }
    .meta-box td:first-child { font-weight: bold; color: #555; width: 200px; }
    .judge-local { background: #EAFAF1; color: #1E8449; font-weight: bold; padding: 2px 6px; border-radius: 3px; }
    .partial-warning { background: #FEF9E7; border-left: 5px solid #F39C12; padding: 12px; border-radius: 4px; }
    table.data-table { border-collapse: collapse; width: 100%; background: #fff; box-shadow: 0 1px 4px rgba(0,0,0,0.07); border-radius: 4px; overflow: hidden; }
    table.data-table th { background: #2C3E50; color: #fff; padding: 8px 12px; text-align: left; font-size: 0.85em; }
    table.data-table td { padding: 7px 12px; border-bottom: 1px solid #ECF0F1; font-size: 0.85em; }
    table.data-table tr:nth-child(even) { background: #F8F9FA; }
    table.data-table tr:hover { background: #EBF5FB; }
    .figure-wrap { text-align: center; margin: 24px 0; }
    .figure-wrap img { max-width: 100%; border: 1px solid #DDD; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.10); }
    .caption { font-size: 0.82em; color: #7F8C8D; margin-top: 6px; font-style: italic; }
    .framing-note { background: #EBF5FB; border-left: 4px solid #3498DB; padding: 10px 16px; font-size: 0.88em; margin: 16px 0; border-radius: 4px; }
    footer { text-align: center; color: #BDC3C7; padding: 24px; font-size: 0.8em; }
    .section { margin-bottom: 40px; }
  </style>
</head>
<body>
<div class="container">
<h1>🧠 Anti-Hallucination Paradox — Evaluation Report</h1>

<div class="framing-note">
  <strong>Framing note:</strong> All metrics combine multi-stage prompt recoverability, alignment-pressure sweeps,
  and Chain-of-Thought auditing to separate <em>True Answer Suppression</em> (withholding proven latent knowledge)
  from <em>Calibrated Ignorance Abstention</em> and <em>Retrieval Failure</em>.
</div>

{% if is_partial %}
<div class="partial-warning">
  ⚠️ <strong>PARTIAL RUN:</strong> Budget was exceeded or run was interrupted. Metrics are computed on available data only.
</div>
{% endif %}

<!-- =========================================================
     CHAPTER 1: RUN METADATA
     ========================================================= -->
<div class="section">
<h2>Chapter 1: Run Metadata</h2>
{% for run_meta in run_metadata %}
<div class="meta-box">
  <h3>{{ run_meta.run_name }}</h3>
  <table>
    <tr><td>Target Model</td><td>{{ run_meta.target_model }}</td></tr>
    <tr><td>Target Provider</td><td>{{ run_meta.target_provider }}</td></tr>
    <tr><td>Judge Model</td><td><span class="judge-local">{{ run_meta.judge_model }}</span></td></tr>
    <tr><td>Judge Provider</td><td><span class="judge-local">{{ run_meta.judge_provider }}</span></td></tr>
    <tr><td>LM Studio Base URL</td><td>{{ run_meta.lmstudio_base_url }}</td></tr>
    <tr><td>Judge Cost</td><td><span class="judge-local">$0.00 (local)</span></td></tr>
    <tr><td>Mode</td><td>{{ run_meta.mode }}</td></tr>
    <tr><td>Start Time</td><td>{{ run_meta.start_time }}</td></tr>
    <tr><td>End Time</td><td>{{ run_meta.end_time }}</td></tr>
    <tr><td>Total Target Calls</td><td>{{ run_meta.total_target_calls }}</td></tr>
    <tr><td>Total Judge Calls</td><td>{{ run_meta.total_judge_calls }}</td></tr>
    <tr><td>Approx Cloud Tokens</td><td>{{ run_meta.approx_cloud_tokens | default('N/A') }}</td></tr>
    <tr><td>Budget Fraction Used</td><td>{{ run_meta.budget_fraction | default('N/A') }}</td></tr>
    <tr><td>Mean Judge Latency</td><td>{{ "%.1f"|format(run_meta.mean_judge_latency_ms) }} ms</td></tr>
  </table>
</div>
{% endfor %}
</div>

<!-- =========================================================
     CHAPTER 4: TABLES
     ========================================================= -->
<div class="section">
<h2>Chapter 4: Metric Tables</h2>

{% for table_html in tables_html %}
{{ table_html }}
{% endfor %}

</div>

<!-- =========================================================
     CHAPTER 5: PLOTS
     ========================================================= -->
<div class="section">
<h2>Chapter 5: Figures</h2>
{% for fig in figures %}
<div class="figure-wrap">
  {% if fig.path %}
  <img src="{{ fig.rel_path }}" alt="{{ fig.title }}">
  <div class="caption">{{ fig.caption }}</div>
  {% else %}
  <p style="color:#999;">{{ fig.title }} — not available (insufficient data)</p>
  {% endif %}
</div>
{% endfor %}
</div>

</div>
<footer>
  Generated by Anti-Hallucination Paradox pipeline &bull; {{ generated_at }}
</footer>
</body>
</html>"""

_TABLE_CAPTIONS = {
    "T1": "T1: Per-model summary @ neutral level (with Latent Knowledge Ceiling, True Suppression vs. Calibrated Abstention, and McNemar p-value)",
    "T2": "T2: Per-model × dataset_name accuracy",
    "T3": "T3: Pressure table — per-model × level (permissive / neutral / conservative)",
    "T4": "T4: Per-model × answer_type accuracy @ neutral",
    "T5": "T5: Per-model × premise × hop accuracy",
    "T6": "T6: Bamboogle audit — non-CORRECT items for manual review",
    "T7": "T7: Stage-1 recovery detail counts per model",
    "T8": "T8: Judge health — parse failures, errors, active judge latency",
    "T9": "T9: 3-Way Diagnostic Failure Taxonomy (Robust Recall, True Answer Suppression, Retrieval Failure, Calibrated vs. Uncalibrated Knowledge Loss)",
    "T10": "T10: Chain-of-Thought (CoT) Self-Censorship & Caution Audit (items where internal CoT surfaced gold facts or caution rules before refusing/failing)",
}

_FIG_CAPTIONS = {
    "P1": "P1: Outcome distribution @ neutral.",
    "P2": "P2: Refusal rate under alignment pressure. Monotonically rising line = alignment-pressure-induced refusal.",
    "P3": "P3: Hallucination rate under alignment pressure. Falling line = improved false-premise calibration under caution.",
    "P4": "P4: Stage-1 recovery rates (recoverability after failure under neutral prompt).",
    "P5": "P5: Calibration success heatmap (false premise × level). Green = good; red = poor calibration.",
    "P6": "P6: PopQA long-tail: correct rate per popularity quartile.",
    "P7": "P7: Per-dataset accuracy @ neutral (True-premise recall vs. False-premise calibration).",
    "P8": "P8: Stage 0→1 transition heatmap (true premise, triggered set).",
    "P9": "P9: 3-Way Failure Taxonomy & Decomposition of Conservative Refusals (True Answer Suppression vs. Calibrated Ignorance).",
}


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(
    all_runs_rows: dict[str, list[dict]],
    all_model_results: dict[str, dict],
    run_metadata_list: list[dict],
    output_dir: str,
    mode: str = "full",
    is_partial: bool = False,
    all_rows_by_run: dict[str, list[dict]] | None = None,
    report_filename: str = "report.html",
) -> str:
    """
    Build the full report: CSVs, PNGs, and HTML.
    Returns path to the HTML report.
    """
    # Deduplicate rows per run
    all_runs_rows = {mn: deduplicate_rows(rows) for mn, rows in all_runs_rows.items()}
    if all_rows_by_run is not None:
        all_rows_by_run = {mn: deduplicate_rows(rows) for mn, rows in all_rows_by_run.items()}

    output_path = Path(output_dir)
    report_filename = Path(report_filename).name
    if not report_filename.endswith(".html"):
        report_filename += ".html"
    metrics_dir = str(output_path / "metrics")
    figures_path = (
        output_path / "figures" / Path(report_filename).stem
        if report_filename != "report.html"
        else output_path / "figures"
    )
    figures_dir = str(figures_path)
    audit_dir = str(output_path / "audit")
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(audit_dir).mkdir(parents=True, exist_ok=True)

    # --- Write CSVs ---
    from src.metrics import flatten_to_dataframes
    raw_rows_map = all_rows_by_run or all_runs_rows
    for model_name, results in all_model_results.items():
        raw_rows = raw_rows_map.get(model_name, [])
        dfs = flatten_to_dataframes(results, all_rows=raw_rows)
        for table_key, df in dfs.items():
            mn_safe = model_name.replace("/", "_").replace(":", "_")
            csv_path = str(Path(metrics_dir) / f"{table_key}_{mn_safe}.csv")
            df.to_csv(csv_path, index=False)

    # --- T6: Bamboogle audit CSV ---
    bamboogle_audit_rows = []
    for model_name, rows in all_runs_rows.items():
        for r in rows:
            if (
                r.get("dataset_name") == "Bamboogle"
                and r.get("level") == "neutral"
                and r.get("label") != "CORRECT"
            ):
                bamboogle_audit_rows.append({
                    "model": model_name,
                    "sample_id": r.get("sample_id"),
                    "question": r.get("question"),
                    "expected_answers": str(r.get("expected_answers")),
                    "model_response": r.get("response_text"),
                    "judge_verdict": r.get("label"),
                })
    if bamboogle_audit_rows:
        audit_df = pd.DataFrame(bamboogle_audit_rows)
        audit_df.to_csv(str(Path(audit_dir) / "bamboogle_audit.csv"), index=False)

    # --- Generate plots ---
    fig_paths: dict[str, str | None] = {}
    fig_paths["P1"] = plot_p1_outcome_distribution(all_runs_rows, figures_dir)
    fig_paths["P2"] = plot_p2_refusal_rate_by_level(all_runs_rows, figures_dir)
    fig_paths["P3"] = plot_p3_hallucination_by_level(all_runs_rows, figures_dir)
    fig_paths["P4"] = plot_p4_recovery_rates(all_model_results, figures_dir)
    fig_paths["P5"] = plot_p5_calibration_heatmap(all_runs_rows, figures_dir)
    fig_paths["P6"] = plot_p6_popqa_longtail(all_runs_rows, figures_dir)
    fig_paths["P7"] = plot_p7_accuracy_by_dataset(all_runs_rows, figures_dir)
    p8_paths = plot_p8_transition_heatmap(all_runs_rows, figures_dir)
    fig_paths["P9"] = plot_p9_failure_taxonomy(all_model_results, figures_dir)

    # --- Build HTML ---
    tables_html = _build_tables_html(all_runs_rows, all_model_results, bamboogle_audit_rows, raw_rows_map=raw_rows_map)

    report_html_path = str(output_path / report_filename)
    figures_info = []
    for pk in ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]:
        ppath = fig_paths.get(pk)
        rel = os.path.relpath(ppath, output_path).replace("\\", "/") if ppath else None
        figures_info.append({
            "title": pk,
            "path": ppath,
            "rel_path": rel,
            "caption": _FIG_CAPTIONS.get(pk, ""),
        })

    # Include ALL models' P8 transition heatmaps (fixing Bug 6)
    if p8_paths:
        for mn, ppath in p8_paths.items():
            rel = os.path.relpath(ppath, output_path).replace("\\", "/") if ppath else None
            figures_info.append({
                "title": f"P8 ({mn})",
                "path": ppath,
                "rel_path": rel,
                "caption": f"{_FIG_CAPTIONS['P8']} — Model: {mn}",
            })
    else:
        figures_info.append({
            "title": "P8",
            "path": None,
            "rel_path": None,
            "caption": _FIG_CAPTIONS["P8"],
        })

    # Add P9 (3-Way Failure Taxonomy & True vs Calibrated Suppression)
    p9_path = fig_paths.get("P9")
    p9_rel = os.path.relpath(p9_path, output_path).replace("\\", "/") if p9_path else None
    figures_info.append({
        "title": "P9",
        "path": p9_path,
        "rel_path": p9_rel,
        "caption": _FIG_CAPTIONS["P9"],
    })

    env = Environment(loader=BaseLoader())
    template = env.from_string(_HTML_TEMPLATE)
    html = template.render(
        run_metadata=run_metadata_list,
        tables_html=tables_html,
        figures=figures_info,
        is_partial=is_partial,
        generated_at=now_iso(),
    )

    with open(report_html_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Report written to %s", report_html_path)
    return report_html_path


def _build_tables_html(
    all_runs_rows: dict[str, list[dict]],
    all_model_results: dict[str, dict],
    bamboogle_audit_rows: list[dict],
    raw_rows_map: dict[str, list[dict]] | None = None,
) -> list[str]:
    """Build HTML for each table T1-T10."""
    from src.metrics import flatten_to_dataframes
    html_blocks = []
    if raw_rows_map is None:
        raw_rows_map = all_runs_rows

    merged: dict[str, list[pd.DataFrame]] = {}
    for model_name, results in all_model_results.items():
        raw_rows = raw_rows_map.get(model_name, [])
        dfs = flatten_to_dataframes(results, all_rows=raw_rows)
        for k, df in dfs.items():
            merged.setdefault(k, []).append(df)

    def _to_html(df: pd.DataFrame, title: str, caption: str) -> str:
        float_cols = df.select_dtypes(include=["float"]).columns
        df_display = df.copy()
        for col in float_cols:
            if col.endswith("_p"):
                df_display[col] = df_display[col].apply(
                    lambda x: (f"{x:.2e}" if x < 0.001 else f"{x:.4f}") if pd.notna(x) and x == x else "—"
                )
            else:
                df_display[col] = df_display[col].apply(
                    lambda x: f"{x:.3f}" if pd.notna(x) and x == x and abs(x) != float("inf") else ("∞" if pd.notna(x) and abs(x) == float("inf") else "—")
                )
        table_str = df_display.to_html(
            index=False, classes="data-table", border=0, escape=True, na_rep="—"
        )
        return (
            f"<h3>{title}</h3>"
            f"<p style='font-size:0.85em;color:#555;'>{caption}</p>"
            f"{table_str}"
        )

    if "T1_per_model_neutral" in merged:
        df = pd.concat(merged["T1_per_model_neutral"], ignore_index=True)
        html_blocks.append(_to_html(df, "T1: Per-Model Summary @ Neutral", _TABLE_CAPTIONS["T1"]))

    if "T9_failure_taxonomy" in merged:
        df = pd.concat(merged["T9_failure_taxonomy"], ignore_index=True)
        html_blocks.append(_to_html(df, "T9: 3-Way Diagnostic Failure Taxonomy & Suppression Decomposition", _TABLE_CAPTIONS["T9"]))

    if "T2_by_dataset" in merged:
        df = pd.concat(merged["T2_by_dataset"], ignore_index=True)
        html_blocks.append(_to_html(df, "T2: Per-Model × Dataset", _TABLE_CAPTIONS["T2"]))

    if "T3_pressure_table" in merged:
        df = pd.concat(merged["T3_pressure_table"], ignore_index=True)
        html_blocks.append(_to_html(df, "T3: Pressure Table (Model × Level)", _TABLE_CAPTIONS["T3"]))

    if "T4_by_answer_type" in merged:
        df = pd.concat(merged["T4_by_answer_type"], ignore_index=True)
        html_blocks.append(_to_html(df, "T4: Per-Model × Answer Type @ Neutral", _TABLE_CAPTIONS["T4"]))

    if "T5_premise_x_hop" in merged:
        df = pd.concat(merged["T5_premise_x_hop"], ignore_index=True)
        html_blocks.append(_to_html(df, "T5: Premise × Hop Accuracy", _TABLE_CAPTIONS["T5"]))

    if bamboogle_audit_rows:
        audit_df = pd.DataFrame(bamboogle_audit_rows)
        html_blocks.append(_to_html(audit_df, "T6: Bamboogle Audit (non-CORRECT items)", _TABLE_CAPTIONS["T6"]))
    else:
        html_blocks.append(
            "<h3>T6: Bamboogle Audit</h3><p>No non-CORRECT Bamboogle items found.</p>"
        )

    if "T7_stage1_detail" in merged:
        df = pd.concat(merged["T7_stage1_detail"], ignore_index=True)
        html_blocks.append(_to_html(df, "T7: Stage-1 Recovery Detail", _TABLE_CAPTIONS["T7"]))

    if "T8_judge_health" in merged:
        df = pd.concat(merged["T8_judge_health"], ignore_index=True)
        html_blocks.append(_to_html(df, "T8: Judge Health", _TABLE_CAPTIONS["T8"]))

    if "T10_cot_self_censorship" in merged:
        df = pd.concat(merged["T10_cot_self_censorship"], ignore_index=True)
        html_blocks.append(_to_html(df, "T10: Chain-of-Thought Self-Censorship & Caution Audit", _TABLE_CAPTIONS["T10"]))

    return html_blocks
