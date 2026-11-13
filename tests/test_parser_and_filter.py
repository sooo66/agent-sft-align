from __future__ import annotations

from agent_sft_align.data import annotate_detour_and_error_steps, parse_thought_and_action
from agent_sft_align.data.eto_loader import parse_sharegpt_record


def test_parse_thought_and_action_variants() -> None:
    # Standard ReAct turn
    res = parse_thought_and_action(
        "Thought: I see cabinet 1 is closed. I should open it first.\nAction: open cabinet 1"
    )
    assert res == ("I see cabinet 1 is closed. I should open it first.", "open cabinet 1")

    # Multiline thought with extra blank line
    res2 = parse_thought_and_action(
        "Thought: First check desk 1.\nIt might hold the mug.\n\nAction: go to desk 1\nExtra trailing line"
    )
    assert res2 == ("First check desk 1.\nIt might hold the mug.", "go to desk 1")

    # Action-only turn
    res3 = parse_thought_and_action("Action: look")
    assert res3 == ("", "look")

    # Handshake turn without action
    assert parse_thought_and_action("OK") is None


def test_sharegpt_handshake_and_detour_masking() -> None:
    sample_record = {
        "id": "alfworld_demo_01",
        "conversations": [
            {"from": "human", "value": "Interact with a household to solve a task."},
            {"from": "gpt", "value": "OK"},
            {"from": "human", "value": "You are in the middle of a room. Task: put a clean mug in cabinet 1."},
            {"from": "gpt", "value": "Thought: Let me go to countertop 1.\nAction: go to countertop 1"},
            {"from": "human", "value": "Observation: On the countertop 1, you see a mug 1."},
            # Invalid action syntax triggers "Nothing happens." on next turn!
            {"from": "gpt", "value": "Thought: Pick up the mug.\nAction: take mug 99 from countertop 1"},
            {"from": "human", "value": "Nothing happens."},
            # Recovery step (valid!)
            {"from": "gpt", "value": "Thought: Oh, it is mug 1, not mug 99. Take mug 1.\nAction: take mug 1 from countertop 1"},
            {"from": "human", "value": "You pick up the mug 1 from the countertop 1."},
            # Duplicate consecutive action loop!
            {"from": "gpt", "value": "Thought: Take mug 1 again.\nAction: take mug 1 from countertop 1"},
        ],
    }

    traj = parse_sharegpt_record(sample_record, default_task="alfworld")
    assert traj is not None
    assert len(traj.metadata["handshake_turns"]) == 2
    assert len(traj.steps) == 4

    stats = annotate_detour_and_error_steps([traj])
    assert stats["total_steps"] == 4
    assert stats["kept_steps"] == 2
    assert stats["masked_env_error"] == 1
    assert stats["masked_duplicate_action"] == 1

    # Step 0: valid
    assert traj.steps[0].loss_mask == 1
    # Step 1: caused "Nothing happens." -> masked!
    assert traj.steps[1].loss_mask == 0
    assert traj.steps[1].mask_reason == "env_noop_or_invalid_action"
    # Step 2: valid recovery step -> supervised!
    assert traj.steps[2].loss_mask == 1
    # Step 3: duplicate consecutive action -> masked!
    assert traj.steps[3].loss_mask == 0
    assert traj.steps[3].mask_reason == "duplicate_consecutive_action"
