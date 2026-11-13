#!/usr/bin/env python3
"""
Full-Dataset Sequential ELBO Rewriter with Incremental JSONL Checkpointing & Auto-Resume.
Designed for Google Colab / single-GPU batch execution: if interrupted, re-running with the
same `--output-jsonl` automatically resumes from the exact next trajectory!
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set

from tqdm import tqdm

from agent_sft_align.data import annotate_detour_and_error_steps, load_eto_trajectories
from agent_sft_align.engine import HFStepLogProbScorer, RewriterConfig, SequentialELBORewriter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sequential ELBO Trajectory Rewriting with Auto-Resume")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--task", type=str, default="alfworld", choices=["alfworld", "webshop", "sciworld"])
    p.add_argument("--local-json", type=str, default=None)
    p.add_argument("--max-trajectories", type=int, default=None)
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--lambda-prior", type=float, default=0.5)
    p.add_argument("--gamma-cvar", type=float, default=0.15)
    p.add_argument("--prompt-mode", type=str, default="auto", choices=["auto", "chat", "react_raw"])
    p.add_argument("--output-jsonl", type=str, default="outputs/rewritten_data/alfworld_rewritten.jsonl")
    return p.parse_args()


def load_completed_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(str(rec["trajectory_id"]))
            except Exception:
                continue
    return done


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = load_eto_trajectories(
        task=args.task,
        local_json_path=args.local_json,
        max_trajectories=args.max_trajectories,
    )
    filter_stats = annotate_detour_and_error_steps(trajectories)
    completed_ids = load_completed_ids(out_path)
    remaining = [t for t in trajectories if t.trajectory_id not in completed_ids]

    print(
        f"Loaded {len(trajectories)} trajectories ({len(completed_ids)} already completed, "
        f"{len(remaining)} remaining). Filter stats: {filter_stats}"
    )
    if not remaining:
        print("All trajectories already rewritten. Exiting cleanly.")
        return

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    model.eval()

    scorer = HFStepLogProbScorer(model=model, tokenizer=tokenizer, prompt_mode=args.prompt_mode, model_name=args.model)
    rewriter = SequentialELBORewriter(
        scorer=scorer,
        config=RewriterConfig(
            num_candidates=args.num_candidates,
            lambda_prior=args.lambda_prior,
            gamma_cvar=args.gamma_cvar,
        ),
    )

    with open(out_path, "a", encoding="utf-8") as out_f:
        for traj in tqdm(remaining, desc=f"Rewriting {args.task}"):
            rewriter.rewrite_trajectory(traj)
            out_f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Successfully wrote rewritten dataset to {out_path.resolve()}")


if __name__ == "__main__":
    main()
