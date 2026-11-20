"""
Build the 4,000-item disjoint training pool, 400-item validation pool, and
500-item held-out multi-dataset test suite for Stage 2 LACE-GRPO.

Quadrant distribution (per LACE_GRPO_DESIGN.md §6):
  Train (4,000) / Val (400) / Held-out Test (500):
    1. Single-Hop True       : 1,600 / 160 / 200  (PopQA stratified quartiles + SimpleQA)
    2. Multi-Hop True        :   800 /  80 / 100  (HotpotQA + 2WikiMultihopQA)
    3. Single-Hop False/Ctrl :   800 /  80 / 100  (FalseQA label=1/0 + FreshQA)
    4. Multi-Hop False       :   800 /  80 / 100  (Bridge-corrupted 2WikiMultihopQA + HotpotQA)

Enforces strict zero-leakage against data/benchmark_subset_100.json via DedupGuard.
Bamboogle and MultiHoax are 100% excluded so they remain pure OOD evaluation sets.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from stage2_lace_grpo.data_builder.dedup_guard import DedupGuard, verify_zero_leakage


FALSEQA_TRAIN_URL = "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/train.csv"


def _parse_json_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [s] if s else []
    return [str(val).strip()]


def _collect_popqa(guard: DedupGuard, rng: random.Random, needed: int) -> list[dict]:
    """Sample PopQA stratified across s_pop quartiles."""
    ds = load_dataset("akariasai/PopQA", split="test")
    rows = list(ds)
    scored: list[tuple[int, dict]] = []
    for r in rows:
        try:
            pop = int(r.get("s_pop") or 0)
        except Exception:
            pop = 0
        scored.append((pop, r))
    scored.sort(key=lambda x: x[0])

    n = len(scored)
    quartiles = [
        [x[1] for x in scored[0 : n // 4]],
        [x[1] for x in scored[n // 4 : n // 2]],
        [x[1] for x in scored[n // 2 : (3 * n) // 4]],
        [x[1] for x in scored[(3 * n) // 4 :]],
    ]
    for q_list in quartiles:
        rng.shuffle(q_list)

    per_q = needed // 4
    collected: list[dict] = []
    for q_idx, q_list in enumerate(quartiles):
        target_q = per_q + (1 if q_idx < (needed % 4) else 0)
        cnt = 0
        for r in q_list:
            if cnt >= target_q:
                break
            answers = _parse_json_list(r.get("possible_answers"))
            obj = str(r.get("obj") or "").strip()
            if obj and obj not in answers:
                answers.insert(0, obj)
            if not answers:
                continue
            item = {
                "dataset_name": "PopQA",
                "eval_dimension": "parametric_entity_knowledge",
                "original_question_id": str(r.get("id", "")),
                "question": str(r.get("question", "")).strip(),
                "expected_answers": answers,
                "premise": True,
                "hop": "single",
                "hop_count": 1,
                "category": str(r.get("prop", "")),
                "answer_type": str(r.get("prop", "")),
                "explanation": None,
                "raw": {
                    "subj": r.get("subj"),
                    "prop": r.get("prop"),
                    "s_pop": r.get("s_pop"),
                    "pop_quartile": q_idx + 1,
                },
            }
            kept, _ = guard.filter_and_register([item])
            if kept:
                collected.append(kept[0])
                cnt += 1
    rng.shuffle(collected)
    return collected


def _collect_simpleqa(guard: DedupGuard, rng: random.Random, needed: int) -> list[dict]:
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    collected: list[dict] = []
    for idx in idxs:
        if len(collected) >= needed:
            break
        r = ds[idx]
        q = str(r.get("problem", "")).strip()
        ans = str(r.get("answer", "")).strip()
        if not q or not ans:
            continue
        meta_raw = r.get("metadata") or "{}"
        meta: dict = {}
        if isinstance(meta_raw, dict):
            meta = meta_raw
        elif isinstance(meta_raw, str):
            try:
                import ast
                meta = ast.literal_eval(meta_raw)
            except Exception:
                meta = {}
        item = {
            "dataset_name": "SimpleQA",
            "eval_dimension": "direct_factual_recall",
            "original_question_id": f"simpleqa_row_{idx}",
            "question": q,
            "expected_answers": [ans],
            "premise": True,
            "hop": "single",
            "hop_count": 1,
            "category": str(meta.get("topic", "")),
            "answer_type": str(meta.get("answer_type", "")),
            "explanation": None,
            "raw": {"original_index": idx, **meta},
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            collected.append(kept[0])
    return collected


def _collect_hotpot_true(guard: DedupGuard, rng: random.Random, needed: int) -> list[dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    collected: list[dict] = []
    for idx in idxs:
        if len(collected) >= needed:
            break
        r = ds[idx]
        q = str(r.get("question", "")).strip()
        ans = str(r.get("answer", "")).strip()
        if not q or not ans:
            continue
        # Skip yes/no questions where possible to preserve open-ended multi-hop recall
        if ans.lower() in ("yes", "no") and rng.random() < 0.7:
            continue
        item = {
            "dataset_name": "HotpotQA",
            "eval_dimension": "multi_hop_factual_reasoning",
            "original_question_id": str(r.get("id", f"hotpot_{idx}")),
            "question": q,
            "expected_answers": [ans],
            "premise": True,
            "hop": "multi",
            "hop_count": 2,
            "category": str(r.get("type", "bridge")),
            "answer_type": "Other",
            "explanation": None,
            "raw": {"level": r.get("level"), "type": r.get("type")},
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            collected.append(kept[0])
    return collected


# Relation templates for 2Wiki bridge questions (used for BOTH corrupted false-premise
# and matched true-premise controls so syntax cannot leak label!)
_BRIDGE_REL1_LABEL = {
    "director": "director",
    "composer": "composer",
    "performer": "performer",
    "father": "father",
    "mother": "mother",
    "spouse": "spouse",
    "founded by": "founder",
    "producer": "producer",
}

_BRIDGE_REL2_TEMPLATES = {
    "date of birth": "When was {bridge}, the {rel1} of {e1}, born?",
    "place of birth": "Where was {bridge}, the {rel1} of {e1}, born?",
    "date of death": "When did {bridge}, the {rel1} of {e1}, die?",
    "place of death": "Where did {bridge}, the {rel1} of {e1}, die?",
    "country of citizenship": "What is the country of citizenship of {bridge}, the {rel1} of {e1}?",
    "educated at": "Where was {bridge}, the {rel1} of {e1}, educated?",
    "spouse": "Who was the spouse of {bridge}, the {rel1} of {e1}?",
}


def _collect_2wiki_true_and_corrupted(
    guard: DedupGuard,
    rng: random.Random,
    needed_true: int,
    needed_false: int,
) -> tuple[list[dict], list[dict]]:
    """
    From 2WikiMultihopQA train:
      - Collect `needed_true` multi-hop true items (including ~20% appositive-template
        true controls so appositive syntax does not become a false-premise shortcut).
      - Collect `needed_false` bridge-corrupted false-premise multi-hop items where
        bridge entity B_true is swapped with a disjoint B_false of the same rel1.
    """
    ds = load_dataset("framolfese/2WikiMultihopQA", split="train")
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    # First index 2-hop chains for clean bridge corruption + matched controls
    chains: list[dict] = []
    by_rel1: dict[str, list[str]] = defaultdict(list)
    true_pairs: set[tuple[str, str, str]] = set()

    for idx in idxs[:35000]:
        r = ds[idx]
        evs = r.get("evidences") or []
        for ev in evs:
            if isinstance(ev, (list, tuple)) and len(ev) == 3:
                true_pairs.add((str(ev[0]).strip(), str(ev[1]).strip(), str(ev[2]).strip()))

        if r.get("type") == "compositional" and len(evs) == 2:
            ev1, ev2 = evs[0], evs[1]
            if len(ev1) == 3 and len(ev2) == 3:
                e1, rel1, b1 = str(ev1[0]).strip(), str(ev1[1]).strip(), str(ev1[2]).strip()
                e2, rel2, ans = str(ev2[0]).strip(), str(ev2[1]).strip(), str(ev2[2]).strip()
                if (
                    e2 == b1
                    and rel1 in _BRIDGE_REL1_LABEL
                    and rel2 in _BRIDGE_REL2_TEMPLATES
                    and e1
                    and b1
                    and ans
                    and e1 != b1
                ):
                    by_rel1[rel1].append(b1)
                    chains.append(
                        {
                            "id": str(r.get("id", idx)),
                            "orig_q": str(r.get("question", "")).strip(),
                            "e1": e1,
                            "rel1": rel1,
                            "b_true": b1,
                            "rel2": rel2,
                            "ans": ans,
                        }
                    )

    for rel1 in by_rel1:
        by_rel1[rel1] = list(dict.fromkeys(by_rel1[rel1]))

    rng.shuffle(chains)

    # Build corrupted multi-hop false-premise items from the first chunk of chains
    false_items: list[dict] = []
    used_chain_ids: set[str] = set()
    for ch in chains:
        if len(false_items) >= needed_false:
            break
        rel1 = ch["rel1"]
        pool = by_rel1.get(rel1, [])
        if len(pool) < 5:
            continue
        b_false = None
        for _ in range(15):
            cand = rng.choice(pool)
            if cand != ch["b_true"] and (ch["e1"], rel1, cand) not in true_pairs:
                b_false = cand
                break
        if not b_false:
            continue

        rel1_lbl = _BRIDGE_REL1_LABEL[rel1]
        q_false = _BRIDGE_REL2_TEMPLATES[ch["rel2"]].format(
            bridge=b_false, rel1=rel1_lbl, e1=ch["e1"]
        )
        expl = (
            f"False premise: {b_false} is not the {rel1_lbl} of {ch['e1']}; "
            f"the {rel1_lbl} of {ch['e1']} is {ch['b_true']}."
        )
        item = {
            "dataset_name": "HotpotBridgeCorrupted",
            "eval_dimension": "multi_hop_false_premise",
            "original_question_id": f"2wiki_corrupt_{ch['id']}",
            "question": q_false,
            "expected_answers": ["I do not know", "I don't know"],
            "premise": False,
            "hop": "multi",
            "hop_count": 2,
            "category": rel1,
            "answer_type": ch["rel2"],
            "explanation": expl,
            "raw": {
                "e1": ch["e1"],
                "rel1": rel1,
                "b_true": ch["b_true"],
                "b_false": b_false,
                "rel2": ch["rel2"],
            },
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            false_items.append(kept[0])
            used_chain_ids.add(ch["id"])

    # Build Multi-Hop True items (include ~20% true appositive controls from unused chains!)
    true_items: list[dict] = []
    n_app_ctrl = max(1, needed_true // 5)
    for ch in chains:
        if len(true_items) >= n_app_ctrl:
            break
        if ch["id"] in used_chain_ids:
            continue
        rel1_lbl = _BRIDGE_REL1_LABEL[ch["rel1"]]
        q_ctrl = _BRIDGE_REL2_TEMPLATES[ch["rel2"]].format(
            bridge=ch["b_true"], rel1=rel1_lbl, e1=ch["e1"]
        )
        item = {
            "dataset_name": "2WikiMultihopQA",
            "eval_dimension": "multi_hop_factual_reasoning",
            "original_question_id": f"2wiki_ctrl_{ch['id']}",
            "question": q_ctrl,
            "expected_answers": [ch["ans"]],
            "premise": True,
            "hop": "multi",
            "hop_count": 2,
            "category": "appositive_true_control",
            "answer_type": ch["rel2"],
            "explanation": None,
            "raw": {"source_id": ch["id"], "type": "compositional"},
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            true_items.append(kept[0])
            used_chain_ids.add(ch["id"])

    for idx in idxs:
        if len(true_items) >= needed_true:
            break
        r = ds[idx]
        rid = str(r.get("id", idx))
        if rid in used_chain_ids:
            continue
        q = str(r.get("question", "")).strip()
        ans = str(r.get("answer", "")).strip()
        if not q or not ans:
            continue
        if ans.lower() in ("yes", "no") and rng.random() < 0.7:
            continue
        item = {
            "dataset_name": "2WikiMultihopQA",
            "eval_dimension": "multi_hop_factual_reasoning",
            "original_question_id": rid,
            "question": q,
            "expected_answers": [ans],
            "premise": True,
            "hop": "multi",
            "hop_count": 2,
            "category": str(r.get("type", "")),
            "answer_type": "Other",
            "explanation": None,
            "raw": {"type": r.get("type")},
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            true_items.append(kept[0])

    return true_items, false_items


def _collect_single_hop_false_and_controls(
    guard: DedupGuard,
    rng: random.Random,
    needed_false: int,
    needed_true_ctrl: int,
) -> list[dict]:
    """
    Fetch FalseQA (thunlp/FalseQA) label=1 (false) + label=0 (weird-but-true controls)
    and supplement with bojanbabic/freshqa_09302025.
    """
    txt = urllib.request.urlopen(FALSEQA_TRAIN_URL, timeout=30).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(txt)))
    false_rows = [(i, r) for i, r in enumerate(rows) if str(r.get("label", "")).strip() == "1"]
    true_rows = [(i, r) for i, r in enumerate(rows) if str(r.get("label", "")).strip() == "0"]
    rng.shuffle(false_rows)
    rng.shuffle(true_rows)

    collected_false: list[dict] = []
    collected_true: list[dict] = []

    for orig_idx, r in false_rows:
        if len(collected_false) >= needed_false:
            break
        q = str(r.get("question", "")).strip()
        ans = str(r.get("answer", "")).strip()
        if not q or not ans:
            continue
        item = {
            "dataset_name": "FalseQA",
            "eval_dimension": "false_premise_handling",
            "original_question_id": f"falseqa_train_{orig_idx}",
            "question": q,
            "expected_answers": [ans],
            "premise": False,
            "hop": "single",
            "hop_count": 1,
            "category": "False_Premise",
            "answer_type": "Correction",
            "explanation": ans,
            "raw": {"label": 1},
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            collected_false.append(kept[0])

    for orig_idx, r in true_rows:
        if len(collected_true) >= needed_true_ctrl:
            break
        q = str(r.get("question", "")).strip()
        ans = str(r.get("answer", "")).strip()
        if not q or not ans:
            continue
        item = {
            "dataset_name": "FalseQA",
            "eval_dimension": "false_premise_handling",
            "original_question_id": f"falseqa_train_{orig_idx}",
            "question": q,
            "expected_answers": [ans],
            "premise": True,
            "hop": "single",
            "hop_count": 1,
            "category": "True_Control",
            "answer_type": "Other",
            "explanation": None,
            "raw": {"label": 0},
        }
        kept, _ = guard.filter_and_register([item])
        if kept:
            collected_true.append(kept[0])

    # Supplement from FreshQA if more needed
    if len(collected_false) < needed_false or len(collected_true) < needed_true_ctrl:
        fqa = load_dataset("bojanbabic/freshqa_09302025")
        f_rows = list(fqa["test"]) + list(fqa["dev"])
        rng.shuffle(f_rows)
        for r in f_rows:
            is_fp = bool(r.get("false_premise"))
            if is_fp and len(collected_false) >= needed_false:
                continue
            if (not is_fp) and len(collected_true) >= needed_true_ctrl:
                continue
            answers = [
                str(r.get(f"answer_{k}")).strip()
                for k in range(10)
                if r.get(f"answer_{k}")
            ]
            if not answers:
                continue
            item = {
                "dataset_name": "FreshQA",
                "eval_dimension": "false_premise_handling" if is_fp else "direct_factual_recall",
                "original_question_id": f"freshqa_{r.get('id')}",
                "question": str(r.get("question", "")).strip(),
                "expected_answers": answers,
                "premise": not is_fp,
                "hop": "single",
                "hop_count": 1,
                "category": str(r.get("fact_type", "")),
                "answer_type": "Correction" if is_fp else "Other",
                "explanation": answers[0] if is_fp else None,
                "raw": {"false_premise": is_fp, "num_hops": r.get("num_hops")},
            }
            kept, _ = guard.filter_and_register([item])
            if kept:
                if is_fp:
                    collected_false.append(kept[0])
                else:
                    collected_true.append(kept[0])

    out = collected_false + collected_true
    rng.shuffle(out)
    return out


def _assign_ids(items: list[dict], prefix: str) -> list[dict]:
    out = []
    for idx, it in enumerate(items, start=1):
        rec = dict(it)
        rec["sample_id"] = f"{prefix}_{idx:04d}"
        rec["grading_mode"] = "factual_match" if rec["premise"] else "abstention_or_correction"
        out.append(rec)
    return out


def build_all_splits(
    benchmark_path: Path,
    out_dir: Path,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    guard = DedupGuard(benchmark_path=benchmark_path, jaccard_threshold=0.8)

    # Total needed across train (4000) + val (400) + test (500) = 4,900 items
    # Quadrant targets (4000 / 400 / 500 = 4900):
    #   Q1 Single-Hop True       : 1600 + 160 + 200 = 1960 (980 PopQA + 980 SimpleQA)
    #   Q2 Multi-Hop True        :  800 +  80 + 100 =  980 (490 HotpotQA + 490 2Wiki)
    #   Q3 Single-Hop False/Ctrl :  800 +  80 + 100 =  980 (735 False + 245 True Ctrl)
    #   Q4 Multi-Hop False       :  800 +  80 + 100 =  980 (980 Bridge-corrupted)
    print("[1/5] Collecting PopQA (980)...")
    popqa_all = _collect_popqa(guard, rng, 980)
    print(f"      Collected {len(popqa_all)} PopQA items")

    print("[2/5] Collecting SimpleQA (980)...")
    simpleqa_all = _collect_simpleqa(guard, rng, 980)
    print(f"      Collected {len(simpleqa_all)} SimpleQA items")

    print("[3/5] Collecting HotpotQA true (490)...")
    hotpot_all = _collect_hotpot_true(guard, rng, 490)
    print(f"      Collected {len(hotpot_all)} HotpotQA items")

    print("[4/5] Collecting 2Wiki true (490) & Bridge-Corrupted multi-hop false (980)...")
    wiki_true_all, multihop_false_all = _collect_2wiki_true_and_corrupted(
        guard, rng, needed_true=490, needed_false=980
    )
    print(
        f"      Collected {len(wiki_true_all)} 2Wiki true + {len(multihop_false_all)} Bridge-Corrupted false"
    )

    print("[5/5] Collecting Single-Hop FalseQA + FreshQA (735 false + 245 true controls)...")
    single_false_all = _collect_single_hop_false_and_controls(
        guard, rng, needed_false=735, needed_true_ctrl=245
    )
    print(f"      Collected {len(single_false_all)} Single-Hop False/Control items")

    # Partition each source strictly into train (800/400/...), val, test without overlap
    def split_source(items: list[dict], n_tr: int, n_va: int, n_te: int):
        rng.shuffle(items)
        assert len(items) >= n_tr + n_va + n_te, f"Shortfall: {len(items)} < {n_tr + n_va + n_te}"
        return items[:n_tr], items[n_tr : n_tr + n_va], items[n_tr + n_va : n_tr + n_va + n_te]

    p_tr, p_va, p_te = split_source(popqa_all, 800, 80, 100)
    s_tr, s_va, s_te = split_source(simpleqa_all, 800, 80, 100)
    h_tr, h_va, h_te = split_source(hotpot_all, 400, 40, 50)
    w_tr, w_va, w_te = split_source(wiki_true_all, 400, 40, 50)
    f_tr, f_va, f_te = split_source(single_false_all, 800, 80, 100)
    m_tr, m_va, m_te = split_source(multihop_false_all, 800, 80, 100)

    train_items = p_tr + s_tr + h_tr + w_tr + f_tr + m_tr
    val_items = p_va + s_va + h_va + w_va + f_va + m_va
    test_items = p_te + s_te + h_te + w_te + f_te + m_te

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    rng.shuffle(test_items)

    train_items = _assign_ids(train_items, "train")
    val_items = _assign_ids(val_items, "val")
    test_items = _assign_ids(test_items, "heldout")

    # Independent zero-leakage verification against benchmark_subset_100.json
    verify_zero_leakage(train_items, benchmark_path=benchmark_path)
    verify_zero_leakage(val_items, benchmark_path=benchmark_path)
    verify_zero_leakage(test_items, benchmark_path=benchmark_path)

    # Verify strict disjointness between train, val, and heldout test
    cross_guard = DedupGuard(benchmark_path=benchmark_path, jaccard_threshold=0.8)
    for it in train_items:
        cross_guard.add_forbidden(it)
    for it in val_items:
        dup, reason = cross_guard.is_duplicate(it)
        assert not dup, f"Train/Val overlap: {reason} | {it['question']}"
        cross_guard.add_forbidden(it)
    for it in test_items:
        dup, reason = cross_guard.is_duplicate(it)
        assert not dup, f"Train+Val/Test overlap: {reason} | {it['question']}"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_4000.json").write_text(
        json.dumps(train_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "val_400.json").write_text(
        json.dumps(val_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "test_heldout_500.json").write_text(
        json.dumps(test_items, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def summarize(lst: list[dict]) -> dict[str, Any]:
        by_ds: dict[str, int] = defaultdict(int)
        for x in lst:
            by_ds[x["dataset_name"]] += 1
        return {
            "total": len(lst),
            "premise_true": sum(1 for x in lst if x["premise"]),
            "premise_false": sum(1 for x in lst if not x["premise"]),
            "hop_single": sum(1 for x in lst if x["hop"] == "single"),
            "hop_multi": sum(1 for x in lst if x["hop"] == "multi"),
            "by_dataset": dict(by_ds),
        }

    manifest = {
        "seed": seed,
        "benchmark_leakage_verified": True,
        "train_val_test_disjoint_verified": True,
        "train": summarize(train_items),
        "val": summarize(val_items),
        "test_heldout": summarize(test_items),
    }
    (out_dir / "pool_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 2 LACE-GRPO disjoint datasets")
    parser.add_argument(
        "--benchmark",
        default=str(PROJECT_ROOT / "data" / "benchmark_subset_100.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "stage2_lace_grpo" / "data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_all_splits(
        benchmark_path=Path(args.benchmark),
        out_dir=Path(args.out_dir),
        seed=args.seed,
    )
    print("\n=== STAGE 2 DATASET POOL BUILT & VERIFIED ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
