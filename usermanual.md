# Anti-Hallucination Paradox (AHP) — User Manual

## Overview
The Anti-Hallucination Paradox (AHP) pipeline is a behavioral evaluation tool designed to evaluate LLMs under varying prompt alignment pressures across true and false premises.

---

## 1. Setup & Installation

### Prerequisites
- Python 3.11+
- Git
- LM Studio (for local LLM judge execution)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/rohitbharadwaj-bit/anti-hallucination-paradox.git
   cd anti-hallucination-paradox
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration (`.env`)
Create/edit `.env` in the root directory:
```env
NVIDIA_API_KEY=
FIREWORKS_API_KEY=
LMSTUDIO_BASE_URL=http://localhost:1234/v1
```

---

## 2. Configuration Files

- `config/settings.yaml`: Controls API parameters, budget limits, rate limits (RPM), retries, and mode defaults.
- `config/runs.yaml`: List of target models and judge configurations for evaluation runs.
- `config/gold_patches.yaml`: Benchmark patch overrides.

---

## 3. Running Experiments

### Dry Run (No API/LM Studio calls)
Tests structure and prints all rendered prompts + cost estimates:
```bash
python run_experiment.py --mode dry
```

### Micro Mode (5 items sanity test)
Requires LM Studio running locally with a loaded judge model:
```bash
python run_experiment.py --mode micro
```

### Offline Regrade (No API / No LM Studio calls)
Re-applies all updated refusal regexes, gold patches, CoT extraction rules, and taxonomy metrics to existing checkpoints and regenerates `outputs/report.html`:
```bash
python run_experiment.py --mode regrade
```

### Fast Incremental Re-Run of Only Cut-Off Responses
Re-fetches only the previously truncated (`EMPTY` / `JUDGE_ERROR`) reasoning items using the 8,192-token limit while keeping all valid completed responses:
```bash
python run_experiment.py --mode full --retry-truncated
```

### Full Benchmark Run
Runs the evaluation across all 100 benchmark items:
```bash
python run_experiment.py --mode full
```

### Options & Overrides
```bash
# Target specific run from runs.yaml
python run_experiment.py --mode full --run-name nematron_ultra

# Start fresh without resuming checkpoints
python run_experiment.py --mode full --no-resume

# Single ad-hoc target override
python run_experiment.py --mode full \
  --target-provider nim --target-model nvidia/nemotron-ultra-253b-v1 \
  --judge-provider lmstudio --judge-model qwen2.5-7b-instruct
```

---

## 4. Output Directories & Artifacts

All outputs are saved to `outputs/`:
- `report.html`: Comprehensive HTML evaluation report with metrics & figures.
- `metrics/`: CSV metric tables (T1 through T10).
- `figures/`: High-resolution figures/plots (P1 through P9).
- `graded/`: Full JSONL records of all graded responses.
- `audit/`: Raw target prompt audit logs (`*_prompts_audit.jsonl`) verifying zero metadata leakage.
- `micro_sanity.json`: Micro run verification report.
