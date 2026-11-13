from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_sft_align.schema import AgentTrajectory


@dataclass
class StepTextLayout:
    """Character-exact layout of a single step sequence `[prefix][thought][action]`."""

    full_text: str
    prefix_end: int
    thought_body_span: Optional[Tuple[int, int]]  # None when include_thought=False
    action_body_span: Tuple[int, int]


def resolve_prompt_mode(tokenizer: Any, prompt_mode: str = "auto", model_name: str = "") -> str:
    if prompt_mode in ("chat", "react_raw"):
        return prompt_mode
    has_chat_tmpl = getattr(tokenizer, "chat_template", None) is not None
    lower_name = model_name.lower()
    is_instruct = any(tag in lower_name for tag in ("instruct", "chat", "it", "sft"))
    if has_chat_tmpl and is_instruct:
        return "chat"
    return "react_raw" if not has_chat_tmpl else "chat"


def build_messages_up_to_step(
    trajectory: AgentTrajectory,
    step_idx: int,
    use_rewritten_prefix: bool = True,
) -> List[Dict[str, str]]:
    """Construct multi-turn conversation messages up to `observation_t` (inclusive)."""
    messages: List[Dict[str, str]] = []
    for turn in trajectory.metadata.get("handshake_turns", []):
        messages.append({"role": turn["role"], "content": turn["content"]})

    for idx in range(step_idx):
        prev = trajectory.steps[idx]
        thought_to_use = prev.active_thought if use_rewritten_prefix else prev.teacher_thought
        messages.append({"role": "user", "content": prev.observation})
        messages.append(
            {
                "role": "assistant",
                "content": prev.format_assistant_turn(
                    thought=thought_to_use,
                    include_thought=bool(thought_to_use.strip()),
                ),
            }
        )

    messages.append({"role": "user", "content": trajectory.steps[step_idx].observation})
    return messages


def render_history_prefix(
    trajectory: AgentTrajectory,
    step_idx: int,
    tokenizer: Any = None,
    prompt_mode: str = "chat",
    use_rewritten_prefix: bool = True,
) -> str:
    """Render conditioning history `h_t` up to the start of the assistant's step-t turn."""
    messages = build_messages_up_to_step(
        trajectory=trajectory,
        step_idx=step_idx,
        use_rewritten_prefix=use_rewritten_prefix,
    )
    if prompt_mode == "chat" and tokenizer is not None and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Plain ReAct transcript fallback for pure pretrained Base models
    chunks: List[str] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            chunks.append(f"{m['content'].strip()}\n")
        else:
            chunks.append(f"{m['content'].strip()}\n")
    return "".join(chunks)


def build_step_layout(
    prefix_str: str,
    action_text: str,
    thought_text: Optional[str] = None,
) -> StepTextLayout:
    """
    Build character-indexed full sequence so BPE tokenization hits clean `:` -> ` <word>` boundaries.
    Notice that scoring `action_body_span` strictly after `Action:` isolates semantic action prediction
    from structural header probability (`Thought:` vs `Action:`).
    """
    prefix_end = len(prefix_str)
    clean_action = action_text.strip()

    if thought_text is not None and thought_text.strip() != "":
        clean_thought = thought_text.strip()
        thought_header = "Thought:"
        thought_body = " " + clean_thought
        action_header = "\nAction:"
        action_body = " " + clean_action

        t_start = prefix_end + len(thought_header)
        t_end = t_start + len(thought_body)
        a_start = t_end + len(action_header)
        a_end = a_start + len(action_body)
        full_text = prefix_str + thought_header + thought_body + action_header + action_body
        return StepTextLayout(
            full_text=full_text,
            prefix_end=prefix_end,
            thought_body_span=(t_start, t_end),
            action_body_span=(a_start, a_end),
        )
    else:
        action_header = "Action:"
        action_body = " " + clean_action
        a_start = prefix_end + len(action_header)
        a_end = a_start + len(action_body)
        full_text = prefix_str + action_header + action_body
        return StepTextLayout(
            full_text=full_text,
            prefix_end=prefix_end,
            thought_body_span=None,
            action_body_span=(a_start, a_end),
        )


def locate_span_token_indices(
    offset_mapping: Sequence[Tuple[int, int]],
    char_span: Tuple[int, int],
) -> List[int]:
    """
    Given fast-tokenizer `offset_mapping`, return token indices `i >= 1` whose character midpoint
    lies inside `[char_start, char_end)`. (`i >= 1` ensures `logits[i - 1]` predicts `input_ids[i]`).
    """
    c_start, c_end = char_span
    indices: List[int] = []
    for tok_idx, (s, e) in enumerate(offset_mapping):
        if tok_idx == 0 or e <= s:
            continue
        mid = 0.5 * (s + e)
        if c_start <= mid < c_end:
            indices.append(tok_idx)
    return indices
