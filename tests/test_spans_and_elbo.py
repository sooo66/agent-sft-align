from __future__ import annotations

import re
from typing import Dict, List, Tuple

from agent_sft_align.data.eto_loader import parse_sharegpt_record
from agent_sft_align.data.step_filter import annotate_detour_and_error_steps
from agent_sft_align.metrics.elbo_math import compute_cvar_nll, compute_elbo_score, summarize_span_logprobs
from agent_sft_align.metrics.span_tokenizer import build_step_layout, locate_span_token_indices
from agent_sft_align.plotting.diagnostic_plots import plot_phase0_diagnostics
from agent_sft_align.training.sft_dataset import IGNORE_INDEX, build_sft_features_for_trajectory


class DummyFastTokenizer:
    """Lightweight regex word/punctuation tokenizer mimicking HuggingFace fast offset mapping."""

    chat_template = None

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [m.group(0) for m in re.finditer(r"\s*\w+|\S", text)]
        ids = [abs(hash(t)) % 10000 + 10 for t in tokens]
        return ([1] + ids) if add_special_tokens else ids

    def __call__(self, text: str, return_offsets_mapping: bool = True, **kwargs) -> Dict[str, List]:
        spans: List[Tuple[int, int]] = [(0, 0)]  # BOS token at index 0
        ids: List[int] = [1]
        for m in re.finditer(r"\s*\w+|\S", text):
            spans.append((m.start(), m.end()))
            ids.append(abs(hash(m.group(0))) % 10000 + 10)
        return {"input_ids": ids, "offset_mapping": spans}


def test_exact_span_locator_excludes_headers() -> None:
    tok = DummyFastTokenizer()
    prefix = "Observation: You see a closed cabinet 1.\n"
    thought = "Cabinet 1 is closed so I should open it."
    action = "open cabinet 1"

    layout = build_step_layout(prefix_str=prefix, action_text=action, thought_text=thought)
    enc = tok(layout.full_text, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    assert layout.thought_body_span is not None
    t_idx = locate_span_token_indices(offsets, layout.thought_body_span)
    a_idx = locate_span_token_indices(offsets, layout.action_body_span)

    recovered_thought = "".join(layout.full_text[offsets[i][0] : offsets[i][1]] for i in t_idx).strip()
    recovered_action = "".join(layout.full_text[offsets[i][0] : offsets[i][1]] for i in a_idx).strip()

    assert recovered_thought == thought
    assert recovered_action == action
    assert set(t_idx).isdisjoint(set(a_idx))


def test_elbo_and_cvar_penalizes_hindsight_leakage() -> None:
    baseline_no_thought_lp = -6.0

    # Candidate A: Natural step-by-step base-aligned reasoning (smooth token logprobs, strong action gain)
    smooth_thought_lps = [-0.6] * 10
    strong_action_lps = [-0.4, -0.3, -0.3]  # sum = -1.0 -> +5.0 nats info gain!
    m_good = summarize_span_logprobs(smooth_thought_lps, strong_action_lps)
    gain_good, score_good = compute_elbo_score(m_good, baseline_no_thought_lp, lambda_prior=0.5, gamma_cvar=0.2)

    # Candidate B: Leaked hindsight token (9 smooth tokens + 1 leaked un-derived ID with NLL = 16.0!)
    leaked_thought_lps = [-0.6] * 9 + [-16.0]
    m_leaked = summarize_span_logprobs(leaked_thought_lps, strong_action_lps)
    gain_leaked, score_leaked = compute_elbo_score(m_leaked, baseline_no_thought_lp, lambda_prior=0.5, gamma_cvar=0.2)

    # Candidate C: Generic empty-calorie boilerplate (ultra-low PPL, zero action info gain!)
    generic_thought_lps = [-0.3] * 10
    unhelpful_action_lps = [-2.0, -2.0, -2.0]  # sum = -6.0 -> 0.0 info gain
    m_generic = summarize_span_logprobs(generic_thought_lps, unhelpful_action_lps)
    _, score_generic = compute_elbo_score(m_generic, baseline_no_thought_lp, lambda_prior=0.5, gamma_cvar=0.2)

    assert abs(gain_good - 5.0) < 1e-6
    assert abs(gain_leaked - 5.0) < 1e-6
    assert compute_cvar_nll([0.6] * 9 + [16.0], tail_fraction=0.10) == 16.0
    # Good candidate strictly beats BOTH hindsight-leaked candidate AND generic boilerplate!
    assert score_good > score_leaked
    assert score_good > score_generic


def test_sft_dataset_masks_obs_and_detour_steps(tmp_path) -> None:
    tok = DummyFastTokenizer()
    rec = {
        "id": "demo",
        "conversations": [
            {"from": "human", "value": "Task: find apple."},
            {"from": "gpt", "value": "Thought: Try fridge 9.\nAction: open fridge 9"},
            {"from": "human", "value": "Nothing happens."},
            {"from": "gpt", "value": "Thought: Fridge 1 is visible, open it.\nAction: open fridge 1"},
        ],
    }
    traj = parse_sharegpt_record(rec)
    assert traj is not None
    annotate_detour_and_error_steps([traj])
    traj.steps[1].rewritten_thought = "Let's open fridge 1 instead."

    feats_ours = build_sft_features_for_trajectory(traj, tok, variant="ours", prompt_mode="react_raw")
    feats_vanilla = build_sft_features_for_trajectory(traj, tok, variant="vanilla_teacher", prompt_mode="react_raw")

    supervised_ours = sum(1 for x in feats_ours["labels"] if x != IGNORE_INDEX)
    supervised_vanilla = sum(1 for x in feats_vanilla["labels"] if x != IGNORE_INDEX)
    # `ours` masks out the failed Step 0 while keeping Step 1 supervised
    assert 0 < supervised_ours < supervised_vanilla

    # Also smoke-test figure generation
    figs = plot_phase0_diagnostics(
        [
            {
                "teacher_thought_tokens": 12,
                "teacher_action_tokens": 4,
                "teacher_thought_mean_lp": -2.1,
                "teacher_action_mean_lp": -0.5,
                "teacher_info_gain": 1.8,
                "rewritten_thought_mean_lp": -0.9,
                "rewritten_info_gain": 2.1,
            },
            {
                "teacher_thought_tokens": 15,
                "teacher_action_tokens": 5,
                "teacher_thought_mean_lp": -2.6,
                "teacher_action_mean_lp": -0.7,
                "teacher_info_gain": 1.2,
                "rewritten_thought_mean_lp": -1.1,
                "rewritten_info_gain": 1.6,
            },
        ],
        tmp_path,
    )
    assert len(figs) == 4
    for fp in figs:
        assert fp.exists() and fp.stat().st_size > 0
