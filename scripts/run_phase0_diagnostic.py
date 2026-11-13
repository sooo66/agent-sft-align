#!/usr/bin/env python3
"""
Phase 0 Diagnostic & Pilot ELBO Rewriting Runner (Week 1 Day 1).
Validates:
  1. Off-policy token surprise gap between Teacher Thought (r_t*) and Verified Action (a_t*).
  2. Fraction of environment no-op / duplicate detour steps caught by Causal-Preserving Step Masking.
  3. Pilot Sequential ELBO Rewriting (`--num-rewrite-trajectories`) shifting r_t into Base distribution
     while maintaining/improving Action Information Gain without hindsight leakage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from agent_sft_align.data import annotate_detour_and_error_steps, load_eto_trajectories
from agent_sft_align.engine import HFStepLogProbScorer, RewriterConfig, SequentialELBORewriter
from agent_sft_align.plotting import plot_phase0_diagnostics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Phase 0 Diagnostic & Pilot ELBO Rewriting")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HF model ID or local path")
    p.add_argument("--task", type=str, default="alfworld", choices=["alfworld", "webshop", "sciworld"])
    p.add_argument("--local-json", type=str, default=None, help="Optional local trajectory JSON path")
    p.add_argument("--num-diagnostic-trajectories", type=int, default=40)
    p.add_argument("--num-rewrite-trajectories", type=int, default=8)
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--lambda-prior", type=float, default=0.5)
    p.add_argument("--gamma-cvar", type=float, default=0.15)
    p.add_argument("--prompt-mode", type=str, default="auto", choices=["auto", "chat", "react_raw"])
    p.add_argument("--output-dir", type=str, default="outputs/phase0_alfworld")
    return p.parse_args()


def load_hf_model_and_tokenizer(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading {args.task} trajectories (max={args.num_diagnostic_trajectories})...")
    trajectories = load_eto_trajectories(
        task=args.task,
        local_json_path=args.local_json,
        max_trajectories=args.num_diagnostic_trajectories,
    )
    filter_stats = annotate_detour_and_error_steps(trajectories)
    print(f"      Loaded {len(trajectories)} trajectories | Step Filter Stats: {filter_stats}")

    print(f"[2/4] Loading target model `{args.model}`...")
    model, tokenizer = load_hf_model_and_tokenizer(args.model)
    scorer = HFStepLogProbScorer(
        model=model,
        tokenizer=tokenizer,
        prompt_mode=args.prompt_mode,
        model_name=args.model,
    )
    print(f"      Active prompt_mode = `{scorer.prompt_mode}`")

    print("[3/4] Scoring Teacher Thought vs. Action token distributions & Pilot Rewriting...")
    rewriter = SequentialELBORewriter(
        scorer=scorer,
        config=RewriterConfig(
            num_candidates=args.num_candidates,
            lambda_prior=args.lambda_prior,
            gamma_cvar=args.gamma_cvar,
        ),
    )

    step_records: List[Dict[str, Any]] = []
    qualitative_samples: List[Dict[str, Any]] = []

    for t_idx, traj in enumerate(tqdm(trajectories, desc="Trajectories")):
        do_rewrite = t_idx < args.num_rewrite_trajectories
        if do_rewrite:
            rankings = rewriter.rewrite_trajectory(traj)
            for s_idx, step in enumerate(traj.steps):
                if step.loss_mask == 0 or not rankings[s_idx]:
                    continue
                ranked = rankings[s_idx]
                winner = ranked[0]
                teacher_res = next((r for r in ranked if r.candidate_id == "teacher"), None)
                if teacher_res is None:
                    continue
                rec = {
                    "trajectory_id": traj.trajectory_id,
                    "step_idx": s_idx,
                    "teacher_thought_tokens": teacher_res.metrics.thought_num_tokens,
                    "teacher_action_tokens": teacher_res.metrics.action_num_tokens,
                    "teacher_thought_mean_lp": teacher_res.metrics.thought_mean_logprob,
                    "teacher_thought_ppl": teacher_res.metrics.thought_ppl,
                    "teacher_thought_cvar10_nll": teacher_res.metrics.thought_cvar10_nll,
                    "teacher_action_mean_lp": teacher_res.metrics.action_mean_logprob,
                    "teacher_action_ppl": teacher_res.metrics.action_ppl,
                    "teacher_info_gain": teacher_res.action_info_gain,
                    "teacher_elbo": teacher_res.elbo_score,
                    "rewritten_source": winner.candidate_id,
                    "rewritten_thought_mean_lp": winner.metrics.thought_mean_logprob,
                    "rewritten_thought_ppl": winner.metrics.thought_ppl,
                    "rewritten_thought_cvar10_nll": winner.metrics.thought_cvar10_nll,
                    "rewritten_info_gain": winner.action_info_gain,
                    "rewritten_elbo": winner.elbo_score,
                }
                step_records.append(rec)
                if len(qualitative_samples) < 12 and winner.candidate_id != "teacher":
                    qualitative_samples.append(
                        {
                            "trajectory_id": traj.trajectory_id,
                            "step_idx": s_idx,
                            "observation_snippet": step.observation[-320:],
                            "action": step.action,
                            "teacher_thought": step.teacher_thought,
                            "rewritten_thought": winner.thought_text,
                            "teacher_ppl": round(teacher_res.metrics.thought_ppl, 2),
                            "rewritten_ppl": round(winner.metrics.thought_ppl, 2),
                            "teacher_info_gain": round(teacher_res.action_info_gain, 3),
                            "rewritten_info_gain": round(winner.action_info_gain, 3),
                        }
                    )
        else:
            for s_idx, step in enumerate(traj.steps):
                if step.loss_mask == 0 or not step.teacher_thought.strip():
                    continue
                ranked = scorer.score_step_candidates(
                    trajectory=traj,
                    step_idx=s_idx,
                    candidates=[("teacher", step.teacher_thought)],
                    use_rewritten_prefix=False,
                    lambda_prior=args.lambda_prior,
                    gamma_cvar=args.gamma_cvar,
                )
                teacher_res = ranked[0]
                step_records.append(
                    {
                        "trajectory_id": traj.trajectory_id,
                        "step_idx": s_idx,
                        "teacher_thought_tokens": teacher_res.metrics.thought_num_tokens,
                        "teacher_action_tokens": teacher_res.metrics.action_num_tokens,
                        "teacher_thought_mean_lp": teacher_res.metrics.thought_mean_logprob,
                        "teacher_thought_ppl": teacher_res.metrics.thought_ppl,
                        "teacher_thought_cvar10_nll": teacher_res.metrics.thought_cvar10_nll,
                        "teacher_action_mean_lp": teacher_res.metrics.action_mean_logprob,
                        "teacher_action_ppl": teacher_res.metrics.action_ppl,
                        "teacher_info_gain": teacher_res.action_info_gain,
                        "teacher_elbo": teacher_res.elbo_score,
                    }
                )

    print("[4/4] Aggregating summary metrics & rendering diagnostic figures...")
    rew_subset = [r for r in step_records if "rewritten_thought_mean_lp" in r]
    summary = {
        "model": args.model,
        "task": args.task,
        "prompt_mode": scorer.prompt_mode,
        "num_trajectories_scored": len(trajectories),
        "num_valid_steps_scored": len(step_records),
        "step_filter_stats": filter_stats,
        "all_steps_diagnostic": {
            "median_teacher_thought_nll": float(np.median([-r["teacher_thought_mean_lp"] for r in step_records])),
            "median_teacher_action_nll": float(np.median([-r["teacher_action_mean_lp"] for r in step_records])),
            "median_teacher_thought_ppl": float(np.median([r["teacher_thought_ppl"] for r in step_records])),
            "median_teacher_action_ppl": float(np.median([r["teacher_action_ppl"] for r in step_records])),
            "mean_teacher_cvar10_nll": float(np.mean([r["teacher_thought_cvar10_nll"] for r in step_records])),
            "mean_teacher_info_gain": float(np.mean([r["teacher_info_gain"] for r in step_records])),
            "fraction_thought_tokens_in_turn": float(
                np.sum([r["teacher_thought_tokens"] for r in step_records])
                / max(1, np.sum([r["teacher_thought_tokens"] + r["teacher_action_tokens"] for r in step_records]))
            ),
        },
        "pilot_rewrite_comparison": (
            {
                "num_rewritten_steps": len(rew_subset),
                "rewrite_win_rate_over_teacher": float(
                    np.mean([1.0 if r["rewritten_source"] != "teacher" else 0.0 for r in rew_subset])
                ),
                "median_teacher_thought_ppl": float(np.median([r["teacher_thought_ppl"] for r in rew_subset])),
                "median_rewritten_thought_ppl": float(np.median([r["rewritten_thought_ppl"] for r in rew_subset])),
                "mean_teacher_cvar10_nll": float(np.mean([r["teacher_thought_cvar10_nll"] for r in rew_subset])),
                "mean_rewritten_cvar10_nll": float(np.mean([r["rewritten_thought_cvar10_nll"] for r in rew_subset])),
                "mean_teacher_info_gain": float(np.mean([r["teacher_info_gain"] for r in rew_subset])),
                "mean_rewritten_info_gain": float(np.mean([r["rewritten_info_gain"] for r in rew_subset])),
            }
            if rew_subset
            else {}
        ),
    }

    with open(out_dir / "phase0_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "step_records.json", "w", encoding="utf-8") as f:
        json.dump(step_records, f, indent=2, ensure_ascii=False)
    with open(out_dir / "qualitative_samples.json", "w", encoding="utf-8") as f:
        json.dump(qualitative_samples, f, indent=2, ensure_ascii=False)

    figs = plot_phase0_diagnostics(step_records, out_dir)
    print("\n=== PHASE 0 SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved artifacts & figures to: {out_dir.resolve()}")
    for fp in figs:
        print(f" - {fp.name}")


if __name__ == "__main__":
    main()
