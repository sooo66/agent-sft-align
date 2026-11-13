from __future__ import annotations

from typing import Any, Dict, List, Literal

from agent_sft_align.schema import AgentTrajectory

SFTVariant = Literal["ours", "vanilla_teacher", "srft_mask_only", "action_only"]

IGNORE_INDEX = -100


def build_sft_features_for_trajectory(
    trajectory: AgentTrajectory,
    tokenizer: Any,
    variant: SFTVariant = "ours",
    prompt_mode: str = "chat",
    max_seq_length: int = 4096,
) -> Dict[str, List[int]]:
    """
    Tokenize a multi-turn trajectory into `(input_ids, attention_mask, labels)` where:
      - User observations and handshake turns ALWAYS have `labels = -100`.
      - If `variant in ("ours", "srft_mask_only")` and `step.loss_mask == 0`, that step's assistant
        turn remains in `input_ids` (preserving context causality) while its `labels` are `-100`.
    """
    segments: List[tuple[str, bool]] = []  # (rendered_chunk, supervise_flag)

    def _render_single_delta(messages_before: List[Dict[str, str]], new_msg: Dict[str, str]) -> str:
        if prompt_mode == "chat" and getattr(tokenizer, "chat_template", None):
            before_txt = (
                tokenizer.apply_chat_template(messages_before, tokenize=False, add_generation_prompt=False)
                if messages_before
                else ""
            )
            after_txt = tokenizer.apply_chat_template(
                messages_before + [new_msg], tokenize=False, add_generation_prompt=False
            )
            if after_txt.startswith(before_txt):
                return after_txt[len(before_txt) :]
        return f"{new_msg['content'].strip()}\n"

    running_msgs: List[Dict[str, str]] = []

    # 1. Handshake turns (never supervised)
    for h_turn in trajectory.metadata.get("handshake_turns", []):
        chunk = _render_single_delta(running_msgs, h_turn)
        segments.append((chunk, False))
        running_msgs.append(h_turn)

    # 2. Interactive episode steps
    for step in trajectory.steps:
        u_msg = {"role": "user", "content": step.observation}
        u_chunk = _render_single_delta(running_msgs, u_msg)
        segments.append((u_chunk, False))
        running_msgs.append(u_msg)

        if variant == "ours":
            assistant_text = step.format_assistant_turn(thought=step.active_thought, include_thought=True)
            supervise = step.loss_mask == 1
        elif variant == "vanilla_teacher":
            assistant_text = step.format_assistant_turn(thought=step.teacher_thought, include_thought=True)
            supervise = True
        elif variant == "srft_mask_only":
            assistant_text = step.format_assistant_turn(thought=step.teacher_thought, include_thought=True)
            supervise = step.loss_mask == 1
        elif variant == "action_only":
            assistant_text = step.format_assistant_turn(thought="", include_thought=False)
            supervise = step.loss_mask == 1
        else:
            raise ValueError(f"Unknown SFTVariant: {variant}")

        a_msg = {"role": "assistant", "content": assistant_text}
        a_chunk = _render_single_delta(running_msgs, a_msg)
        segments.append((a_chunk, supervise))
        running_msgs.append(a_msg)

    input_ids: List[int] = []
    labels: List[int] = []

    for idx, (chunk_text, supervise) in enumerate(segments):
        add_bos = idx == 0 and prompt_mode != "chat"
        tok_ids: List[int] = tokenizer.encode(chunk_text, add_special_tokens=add_bos)
        input_ids.extend(tok_ids)
        if supervise:
            labels.extend(tok_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(tok_ids))

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        labels = labels[:max_seq_length]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
