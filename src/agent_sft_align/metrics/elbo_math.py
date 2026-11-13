from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from agent_sft_align.schema import SpanLogProbMetrics


def compute_cvar_nll(token_nlls: Sequence[float], tail_fraction: float = 0.10) -> float:
    """
    Compute Conditional Value-at-Risk (average of worst `tail_fraction` token NLLs).
    Captures localized perplexity spikes caused by hindsight causal leakage or alien jargon.
    """
    if len(token_nlls) == 0:
        return 0.0
    arr = np.asarray(token_nlls, dtype=np.float64)
    k = max(1, int(math.ceil(len(arr) * tail_fraction)))
    top_k = np.partition(arr, len(arr) - k)[len(arr) - k :]
    return float(np.mean(top_k))


def _safe_ppl(mean_nll: float, cap: float = 20.0) -> float:
    return float(math.exp(min(mean_nll, cap)))


def summarize_span_logprobs(
    thought_token_logprobs: Sequence[float],
    action_token_logprobs: Sequence[float],
) -> SpanLogProbMetrics:
    """Aggregate token-level log-probabilities for Thought and Action spans."""
    t_arr = np.asarray(thought_token_logprobs, dtype=np.float64)
    a_arr = np.asarray(action_token_logprobs, dtype=np.float64)

    if len(t_arr) > 0:
        t_sum = float(np.sum(t_arr))
        t_mean = float(np.mean(t_arr))
        t_nlls = -t_arr
        t_ppl = _safe_ppl(float(np.mean(t_nlls)))
        t_cvar10 = compute_cvar_nll(t_nlls.tolist(), tail_fraction=0.10)
        t_max_nll = float(np.max(t_nlls))
    else:
        t_sum = 0.0
        t_mean = 0.0
        t_ppl = 1.0
        t_cvar10 = 0.0
        t_max_nll = 0.0

    if len(a_arr) > 0:
        a_sum = float(np.sum(a_arr))
        a_mean = float(np.mean(a_arr))
        a_ppl = _safe_ppl(float(-a_mean))
    else:
        a_sum = 0.0
        a_mean = 0.0
        a_ppl = 1.0

    return SpanLogProbMetrics(
        thought_num_tokens=int(len(t_arr)),
        action_num_tokens=int(len(a_arr)),
        thought_sum_logprob=t_sum,
        thought_mean_logprob=t_mean,
        thought_ppl=t_ppl,
        thought_cvar10_nll=t_cvar10,
        thought_max_token_nll=t_max_nll,
        action_sum_logprob=a_sum,
        action_mean_logprob=a_mean,
        action_ppl=a_ppl,
    )


def compute_elbo_score(
    metrics: SpanLogProbMetrics,
    baseline_no_thought_action_logprob: float,
    lambda_prior: float = 0.5,
    gamma_cvar: float = 0.15,
) -> tuple[float, float]:
    """
    Compute `(action_info_gain, elbo_score)`:
      J(r_t) = [log p_0(a_t | h_t, r_t) - log p_0(a_t | h_t, empty)]
               + lambda_prior * mean_log_p_0(r_t | h_t)
               - gamma_cvar * CVaR_0.10(NLL_0(r_t | h_t))
    """
    action_info_gain = metrics.action_sum_logprob - baseline_no_thought_action_logprob
    score = (
        action_info_gain
        + lambda_prior * metrics.thought_mean_logprob
        - gamma_cvar * metrics.thought_cvar10_nll
    )
    return float(action_info_gain), float(score)
