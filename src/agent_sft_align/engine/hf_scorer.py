from __future__ import annotations

import inspect
from typing import Any, List, Optional

from agent_sft_align.metrics.elbo_math import compute_elbo_score, summarize_span_logprobs
from agent_sft_align.metrics.span_tokenizer import (
    build_step_layout,
    locate_span_token_indices,
    render_history_prefix,
    resolve_prompt_mode,
)
from agent_sft_align.schema import AgentTrajectory, CandidateEvalResult, SpanLogProbMetrics


class HFStepLogProbScorer:
    """
    Exact single-pass teacher-forced log-probability scorer on the target Base/Seed model.
    Uses character offset mappings for zero-BPE-boundary-bug span extraction and `logits_to_keep`
    (when supported by `transformers`) to avoid materializing huge prefix vocab logits.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt_mode: str = "auto",
        model_name: str = "",
        max_seq_length: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name or getattr(model, "name_or_path", "")
        self.prompt_mode = resolve_prompt_mode(tokenizer, prompt_mode, self.model_name)
        self.max_seq_length = max_seq_length
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(model.forward).parameters

    @property
    def device(self) -> Any:
        return next(self.model.parameters()).device

    def score_layout(self, prefix_str: str, action_text: str, thought_text: Optional[str]) -> SpanLogProbMetrics:
        import torch
        import torch.nn.functional as F

        layout = build_step_layout(prefix_str=prefix_str, action_text=action_text, thought_text=thought_text)
        enc = self.tokenizer(
            layout.full_text,
            add_special_tokens=(self.prompt_mode != "chat"),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_seq_length,
        )
        input_ids: List[int] = enc["input_ids"]
        offsets = enc["offset_mapping"]

        action_indices = locate_span_token_indices(offsets, layout.action_body_span)
        thought_indices = (
            locate_span_token_indices(offsets, layout.thought_body_span)
            if layout.thought_body_span is not None
            else []
        )

        all_target_indices = thought_indices + action_indices
        if not all_target_indices:
            return summarize_span_logprobs([], [])

        earliest_tok_idx = min(all_target_indices)
        # To score `input_ids[tok_idx]` for `tok_idx >= earliest_tok_idx`, we only need logits from `earliest_tok_idx - 1` onward!
        keep_len = len(input_ids) - (earliest_tok_idx - 1)

        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            if self._supports_logits_to_keep:
                outputs = self.model(input_ids=ids_tensor, logits_to_keep=keep_len, use_cache=False)
                sliced_logits = outputs.logits[0]  # shape: [keep_len, vocab]
                logit_offset = len(input_ids) - keep_len
            else:
                outputs = self.model(input_ids=ids_tensor, use_cache=False)
                logit_offset = earliest_tok_idx - 1
                sliced_logits = outputs.logits[0, logit_offset:, :]

            log_probs = F.log_softmax(sliced_logits.float(), dim=-1)

            def _gather_span(indices: List[int]) -> List[float]:
                vals: List[float] = []
                for idx in indices:
                    local_pos = (idx - 1) - logit_offset
                    tok_id = input_ids[idx]
                    vals.append(float(log_probs[local_pos, tok_id].item()))
                return vals

            t_lps = _gather_span(thought_indices)
            a_lps = _gather_span(action_indices)

        return summarize_span_logprobs(t_lps, a_lps)

    def score_step_candidates(
        self,
        trajectory: AgentTrajectory,
        step_idx: int,
        candidates: List[tuple[str, str]],
        use_rewritten_prefix: bool = True,
        lambda_prior: float = 0.5,
        gamma_cvar: float = 0.15,
    ) -> List[CandidateEvalResult]:
        """
        Evaluate multiple `(candidate_id, candidate_thought)` pairs on step `step_idx`
        against the thought-free baseline `p_0(a_t | h_t, empty)`.
        """
        step = trajectory.steps[step_idx]
        prefix_str = render_history_prefix(
            trajectory=trajectory,
            step_idx=step_idx,
            tokenizer=self.tokenizer,
            prompt_mode=self.prompt_mode,
            use_rewritten_prefix=use_rewritten_prefix,
        )

        no_thought_metrics = self.score_layout(
            prefix_str=prefix_str,
            action_text=step.action,
            thought_text=None,
        )
        baseline_action_lp = no_thought_metrics.action_sum_logprob

        results: List[CandidateEvalResult] = []
        for cand_id, cand_thought in candidates:
            m = self.score_layout(
                prefix_str=prefix_str,
                action_text=step.action,
                thought_text=cand_thought,
            )
            info_gain, elbo = compute_elbo_score(
                metrics=m,
                baseline_no_thought_action_logprob=baseline_action_lp,
                lambda_prior=lambda_prior,
                gamma_cvar=gamma_cvar,
            )
            results.append(
                CandidateEvalResult(
                    candidate_id=cand_id,
                    thought_text=cand_thought,
                    metrics=m,
                    baseline_no_thought_action_logprob=baseline_action_lp,
                    action_info_gain=info_gain,
                    elbo_score=elbo,
                )
            )

        results.sort(key=lambda r: r.elbo_score, reverse=True)
        return results
