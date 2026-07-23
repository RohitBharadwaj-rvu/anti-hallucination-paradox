# Anti-Hallucination Paradox (AHP)
## API-Only Behavioral Evaluation Pipeline

> **Framing note:** All metrics are *behavioral proxies* only. We measure API outputs; we cannot access model internals. Claims use the framing: "behavioral evidence consistent with suppression", never "proves suppression".

---

## 📁 Repository Layout

```
anti_hallucination_paradox/
├── .env                          # gitignored; cloud keys + optional LM Studio base url
├── .gitignore
├── requirements.txt
├── config/
│   ├── settings.yaml             # decoding, retry, budgets, concurrency
│   ├── runs.yaml                 # sequential list of target runs
│   └── gold_patches.yaml         # manual gold fixes (typo, ranges)
├── data/
│   ├── benchmark_subset_100.json # raw benchmark (100 items across 5 datasets)
│   └── checkpoints/              # auto-created JSONL checkpoints
├── src/
│   ├── __init__.py
│   ├── env_loader.py             # .env loading + provider config
│   ├── loader.py                 # clean + patch + derive authoritative fields
│   ├── prompts.py                # all prompt strings (target + judge), verbatim
│   ├── api_router.py             # OpenAI-compat; 3 providers; dual gates; retry; budget
│   ├── grader.py                 # regex refusal + LLM-judge routing + labels
│   ├── orchestrator.py           # stage order, modes, neutral-reuse, checkpoint
│   ├── metrics.py                # exact formulas + all slices
│   ├── reporter.py               # CSV + PNG + HTML report (T1-T8, P1-P8)
│   ├── cost.py                   # dry-run / post-micro cost estimator
│   └── utils.py                  # logging, jsonl checkpoint, key() builder
├── outputs/                      # auto-created
│   ├── raw/  graded/  metrics/  figures/  report.html  audit/
├── run_experiment.py             # CLI entry point
└── README.md
```

---

## 🎯 Scope & Design

- **NO** base-model comparison. **NO** hidden-state probing. Targets are **API-only**.
- **JUDGE** runs **LOCALLY** via LM Studio (free, no credits). Targets run on NIM / Fireworks.
- Pipeline = Stage 0 → Stage 1 (recoverability) → Stage 2 (PERMISSIVE + CONSERVATIVE sweep).

### Behavioral Proxies Measured
| Signal | Formula |
|--------|---------|
| `suppression_proxy` | `refusal_rate(true, CONSERVATIVE) − refusal_rate(true, PERMISSIVE)` |
| `induced_hallucination` | `halluc_rate(false, PERMISSIVE) − halluc_rate(false, CONSERVATIVE)` |
| `recovery` | Stage-1 success on items that failed Stage-0 |

### Benchmark Composition (100 items)
| Dataset | Count | Premise | Hop |
|---------|-------|---------|-----|
| SimpleQA | 25 | True | single |
| PopQA | 25 | True | single |
| Bamboogle | 20 | True | multi |
| FalseQA | 15 | 10 False + 5 True | single |
| MultiHoax | 15 | False | multi |

---

## ⚙️ Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Fill Cloud API Keys in `.env`

```env
NVIDIA_API_KEY=nvapi-...your-real-key...
FIREWORKS_API_KEY=fw_...your-real-key...
```

Leave `LMSTUDIO_*` at defaults unless you use a non-standard port.

### 3. Configure Target Models in `config/runs.yaml`

Edit `model_id` for each run entry. Replace the `judge.model_id` with the **exact** name shown in LM Studio's server tab.

### 4. LM Studio Pre-Requisites (for micro/full runs)

1. In LM Studio: download & LOAD a small instruct judge model (e.g., `qwen2.5-7b-instruct`, Q4/Q5 quant).
2. LM Studio → Developer tab → **Start Server** (default port 1234).
3. Confirm reachable:
   ```bash
   curl http://localhost:1234/v1/models
   ```

---

## 🚀 How to Run

```bash
# 1) Sanity check — no network, no LM Studio
python run_experiment.py --mode dry

# 2) 5-question live micro-test (needs LM Studio + cloud keys)
python run_experiment.py --mode micro

# 3) Full run — all runs in runs.yaml, sequentially
python run_experiment.py --mode full

# 4) Single named run
python run_experiment.py --mode full --run-name nematron_ultra

# 5) Ad-hoc single run (judge explicitly local)
python run_experiment.py --mode full \
    --target-provider nim --target-model nvidia/nemotron-ultra-253b-v1 \
    --judge-provider lmstudio --judge-model qwen2.5-7b-instruct

# 6) Open the HTML report
start outputs/report.html   # Windows
open outputs/report.html    # macOS
```

---

## 🔧 CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs` | `config/runs.yaml` | Path to runs config |
| `--mode` | `full` | `dry` / `micro` / `full` |
| `--run-name` | — | Run a single named entry |
| `--target-provider` | — | Override target provider |
| `--target-model` | — | Override target model ID |
| `--judge-provider` | — | Override judge provider |
| `--judge-model` | — | Override judge model ID |
| `--concurrency` | 1 | Override target concurrency |
| `--concurrency-judge` | 1 | Override judge concurrency |
| `--limit` | — | Run first N items only |
| `--no-resume` | False | Ignore checkpoints (expensive!) |
| `--log-level` | `INFO` | `DEBUG` / `INFO` |

---

## 📊 Outputs

After a full run, `outputs/` contains:

- `report.html` — Full HTML report (open in browser)
- `graded/` — Per-run JSONL graded rows
- `metrics/` — CSV tables T1–T8
- `figures/` — PNG plots P1–P8
- `audit/` — Bamboogle disagreement audit CSV
- `micro_sanity.json` — Sanity report after micro runs

### Tables (T1–T8)
- **T1**: Per-model summary @ neutral
- **T2**: Per-model × dataset accuracy
- **T3**: Pressure table (model × level)
- **T4**: Per-model × answer_type accuracy
- **T5**: Premise × hop accuracy
- **T6**: Bamboogle audit (non-CORRECT items)
- **T7**: Stage-1 recovery detail
- **T8**: Judge health (parse failures, latency)

### Plots (P1–P8)
- **P1**: Outcome distribution @ neutral (stacked bars)
- **P2**: Refusal rate under alignment pressure (line plot)
- **P3**: Hallucination rate under alignment pressure (line plot)
- **P4**: Stage-1 recovery rates (grouped bars)
- **P5**: Calibration heatmap (model × level)
- **P6**: PopQA long-tail by popularity quartile
- **P7**: Per-dataset accuracy @ neutral
- **P8**: Stage 0→1 transition heatmap (recoverability)

---

## 🔄 Resume & Robustness

- **Checkpoint/resume**: Every call is written to JSONL immediately. Re-running skips completed keys automatically.
- **Budget kill-switch**: Cloud token usage is tracked; hard-stops at 95% of the configured budget.
- **LM Studio preflight**: Server reachability + model loaded + warmup call performed at startup.
- **Auto-resolve model**: If the configured judge model ID is wrong, falls back to first listed model.
- **Retry**: Tenacity retry on 429/5xx and timeout, exponential backoff.

---

## ✅ Acceptance Criteria

```bash
# Loader assertions pass with correct counts
python -c "from src.loader import load; load('data/benchmark_subset_100.json')"

# Dry mode: no network, no LM Studio
python run_experiment.py --mode dry

# With LM Studio OFF: fast pre-flight failure
python run_experiment.py --mode micro   # -> "not reachable" message

# With LM Studio ON + model loaded: micro sanity pass
python run_experiment.py --mode micro   # -> SANITY PASS
```
