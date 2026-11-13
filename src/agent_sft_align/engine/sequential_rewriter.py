from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from agent_sft_align.engine.hf_scorer import HFStepLogProbScorer
from agent_sft_align.schema import AgentTrajectory, CandidateEvalResult


@dataclass
class RewriterConfig:
    num_candidates: int = 6
    temperatures: List[float] = field(default_factory=lambda: [0.4, 0.7, 1.0])
    max_new_tokens: int = 160
    lambda_prior: float = 0.5
    gamma_cvar: float = 0.15
    allow_teacher_fallback: bool = True
    allow_minimal_thought_candidate: bool = False


_SYSTEM_REWRITE_PROMPT = (
    "You are rewriting an interactive agent's internal reasoning step so it sounds natural, concise, "
    "and strictly grounded in what the agent can currently observe.\n"
    "Rules:\n"
    "1. Output ONLY the reasoning sentences (1 to 3 short sentences). Do NOT include 'Thought:' or 'Action:' labels.\n"
    "2. Reason forward from the task goal, recent history, and current observation so the next action follows logically.\n"
    "3. NEVER mention 'the target action' or leak unvisited object details that are not yet visible in the observation."
)


def build_rewriter_prompt_messages(
    trajectory: AgentTrajectory,
    step_idx: int,
    max_recent_turns: int = 4,
) -> List[dict[str, str]]:
    """Construct the conditioned prompt for generating candidate base-aligned thoughts."""
    step = trajectory.steps[step_idx]
    start_idx = max(0, step_idx - max_recent_turns)
    history_lines: List[str] = []
    for i in range(start_idx, step_idx):
        s = trajectory.steps[i]
        history_lines.append(f"[Step {i}] Observation: {s.observation[:500]}")
        history_lines.append(f"[Step {i}] Thought: {s.active_thought}")
        history_lines.append(f"[Step {i}] Action: {s.action}")

    recovery_note = ""
    if step_idx > 0 and trajectory.steps[step_idx - 1].loss_mask == 0:
        recovery_note = (
            "\nNote: The immediately preceding action failed or had no effect. "
            "Briefly acknowledge why it did not work from the current observation and pivot naturally."
        )

    user_content = (
        f"Task Context:\n{trajectory.system_or_init_prompt[:700]}\n\n"
        f"Recent Trajectory:\n" + ("\n".join(history_lines) if history_lines else "(Start of episode)") + "\n\n"
        f"Current Observation:\n{step.observation[:900]}\n\n"
        f"Reference Expert Thought:\n{step.teacher_thought}\n\n"
        f"Next Action to Justify:\n{step.action}"
        f"{recovery_note}\n\n"
        "Rewrite the internal reasoning in your own natural, direct style (output ONLY the reasoning text):"
    )
    return [
        {"role": "system", "content": _SYSTEM_REWRITE_PROMPT},
        {"role": "user", "content": user_content},
    ]


def clean_generated_thought(raw_text: str, action_text: str) -> str:
    """Strip accidental prefixes (`Thought:`) or leaked trailing `Action:` lines."""
    txt = raw_text.strip()
    # Cut off if model accidentally started emitting `Action:`
    act_match = re.search(r"(?:^|\n)\s*Action\s*:", txt, flags=re.IGNORECASE)
    if act_match:
        txt = txt[: act_match.start()].strip()
    txt = re.sub(r"^\s*Thought\s*:\s*", "", txt, flags=re.IGNORECASE).strip()
    txt = txt.strip('"').strip("'").strip()
    # Collapse excessive blank lines into a single line/paragraph suitable for ReAct
    txt = re.sub(r"\s*\n+\s*", " ", txt).strip()
    return txt


class SequentialELBORewriter:
    """
    Sequential Left-to-Right Trajectory Rewriter:
    At each valid step `t`, conditions on the already-rewritten prefix `~h_t`, generates `K` candidate
    thoughts from the target model across temperatures, scores them in one teacher-forced pass under
    the ELBO + CVaR anti-leakage criterion, and commits the winning thought `~r_t` before advancing to `t+1`.
    """

    def __init__(
        self,
        scorer: HFStepLogProbScorer,
        config: Optional[RewriterConfig] = None,
        generator_model: Optional[Any] = None,
        generator_tokenizer: Optional[Any] = None,
    ) -> None:
        self.scorer = scorer
        self.config = config or RewriterConfig()
        self.gen_model = generator_model if generator_model is not None else scorer.model
        self.gen_tokenizer = generator_tokenizer if generator_tokenizer is not None else scorer.tokenizer

    def _sample_candidates_hf(self, messages: List[dict[str, str]], action_text: str) -> List[tuple[str, str]]:
        import torch

        if getattr(self.gen_tokenizer, "chat_template", None):
            prompt_text = self.gen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = self.gen_tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        else:
            prompt_text = f"{messages[0]['content']}\n\n{messages[1]['content']}\nReasoning: "
            enc = self.gen_tokenizer(prompt_text, return_tensors="pt")

        device = next(self.gen_model.parameters()).device
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        temps = self.config.temperatures or [0.7]
        per_temp = max(1, (self.config.num_candidates + len(temps) - 1) // len(temps))

        seen_texts: set[str] = set()
        candidates: List[tuple[str, str]] = []
        cand_counter = 0

        for temp in temps:
            if cand_counter >= self.config.num_candidates:
                break
            n_return = min(per_temp, self.config.num_candidates - cand_counter)
            with torch.no_grad():
                out = self.gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=(temp > 0.0),
                    temperature=max(temp, 1e-3),
                    top_p=0.92,
                    num_return_sequences=n_return,
                    pad_token_id=self.gen_tokenizer.eos_token_id,
                )
            for seq in out:
                raw_gen = self.gen_tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                cleaned = clean_generated_thought(raw_gen, action_text)
                cand_counter += 1
                if cleaned and cleaned not in seen_texts:
                    seen_texts.add(cleaned)
                    candidates.append((f"rewritten_t{temp}_k{len(candidates)}", cleaned))

        return candidates

    def rewrite_trajectory(self, trajectory: AgentTrajectory) -> List[List[CandidateEvalResult]]:
        """Rewrite a full trajectory sequentially in place and return full step-by-step candidate rankings."""
        all_rankings: List[List[CandidateEvalResult]] = []

        for step_idx, step in enumerate(trajectory.steps):
            # Skip sampling for masked detour/error steps (kept in history, loss_mask=0)
            if step.loss_mask == 0:
                step.selected_source = "masked_detour_kept_teacher"
                all_rankings.append([])
                continue

            messages = build_rewriter_prompt_messages(trajectory, step_idx)
            sampled = self._sample_candidates_hf(messages, step.action)

            pool: List[tuple[str, str]] = list(sampled)
            if self.config.allow_teacher_fallback and step.teacher_thought.strip():
                pool.append(("teacher", step.teacher_thought.strip()))
            if self.config.allow_minimal_thought_candidate:
                pool.append(("minimal", f"Observe current state and execute the next step."))

            if not pool:
                all_rankings.append([])
                continue

            ranked = self.scorer.score_step_candidates(
                trajectory=trajectory,
                step_idx=step_idx,
                candidates=pool,
                use_rewritten_prefix=True,
                lambda_prior=self.config.lambda_prior,
                gamma_cvar=self.config.gamma_cvar,
            )
            winner = ranked[0]
            step.rewritten_thought = winner.thought_text
            step.selected_source = winner.candidate_id

            # Record teacher vs winner comparison for instant diagnostic tracking
            teacher_entry = next((r for r in ranked if r.candidate_id == "teacher"), None)
            step.diagnostics = {
                "winner_source": winner.candidate_id,
                "winner_elbo": round(winner.elbo_score, 4),
                "winner_info_gain": round(winner.action_info_gain, 4),
                "winner_thought_ppl": round(winner.metrics.thought_ppl, 3),
                "winner_cvar10_nll": round(winner.metrics.thought_cvar10_nll, 3),
            }
            if teacher_entry is not None:
                step.diagnostics.update(
                    {
                        "teacher_elbo": round(teacher_entry.elbo_score, 4),
                        "teacher_info_gain": round(teacher_entry.action_info_gain, 4),
                        "teacher_thought_ppl": round(teacher_entry.metrics.thought_ppl, 3),
                        "teacher_cvar10_nll": round(teacher_entry.metrics.thought_cvar10_nll, 3),
                    }
                )
            all_rankings.append(ranked)

        return all_rankings
