# agent-sft-align

**ELBO-Guided Sequential Trajectory Alignment for LLM Agent SFT (ICML 2027)**

This repository implements our offline trajectory alignment and causal step-masking pipeline for interactive LLM agents (**ALFWorld**, **WebShop**, **ScienceWorld**):

1. **Exact Offset-Mapped Token Span Scorer (`HFStepLogProbScorer`)**: Computes forward prior reachability $\frac{1}{|r_t|}\log \pi_0(r_t \mid \tilde{h}_t)$, top-10% tail surprise $\text{CVaR}_{0.10}(\text{NLL}_0)$, and pure semantic Action Information Gain $\Delta_{\text{act}}(r_t)$ in a **single teacher-forced forward pass** without BPE boundary artifacts or header probability bias.
2. **Causal-Preserving Error/Detour Masker (`annotate_detour_and_error_steps`)**: Keeps environment no-op (`Nothing happens.`) and consecutive loop steps inside context history $\tilde{h}_{>t}$ while masking their SFT supervision loss (`labels = -100`).
3. **Sequential Prefix-Consistent Rewriter (`SequentialELBORewriter`)**: Rewrites valid thoughts left-to-right conditioned on already-aligned prefixes $\tilde{h}_t$, with incremental JSONL auto-resume for Colab sessions.
4. **Unified SFT Trainer (`scripts/run_sft.py`)**: Trains `ours`, `vanilla_teacher`, `srft_mask_only`, and `action_only` under identical tokenization and optimization settings, with automatic SLURM checkpoint resume.

---

## Quickstart

### 1. Local Setup & Unit Tests (CPU)
```bash
pip install -e ".[dev]"
pytest -v
```

### 2. Google Colab (Week 1 Phase 0 Diagnostic + Pilot Rewriting)
1. Open `notebooks/Week1_Colab_Diagnostic_and_Pilot.ipynb` in Google Colab (A100 or L4 runtime).
2. Upload `/Users/ymx66/Workspace/agent-sft-align.zip` to `/content/` (or push this repo to a private GitHub repo and set `REPO_GIT_URL`).
3. Click **Run All** (~10–15 minutes) and download `phase0_results.zip` back into `outputs/` locally.

### 3. Full Dataset Rewriting (Colab, Auto-Resumable)
```bash
python3 scripts/run_rewrite_dataset.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --task alfworld \
  --output-jsonl outputs/rewritten_data/alfworld_qwen7b.jsonl
```

### 4. University SLURM Cluster SFT (Short 4h Backfill-Friendly Job)
```bash
MODEL="Qwen/Qwen2.5-7B-Instruct" TASK="alfworld" VARIANT="ours" \
  REWRITTEN_JSONL="outputs/rewritten_data/alfworld_qwen7b.jsonl" \
  sbatch scripts/slurm/sbatch_sft_short.sh
```
