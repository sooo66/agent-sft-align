from __future__ import annotations

from typing import Dict, List

from agent_sft_align.schema import AgentTrajectory

# Canonical environment feedback strings indicating the preceding action was invalid or a no-op
_ERROR_OBSERVATION_PATTERNS = (
    "nothing happens.",
    "nothing happens",
    "no known action matches that input",
    "invalid action",
    "you can't do that",
    "is not currently open",
    "ambiguous request",
    "syntaxerror",
    "command not found",
)


def _is_error_observation(next_obs: str) -> bool:
    norm = next_obs.strip().lower()
    # Normalize "Observation: Nothing happens." -> "nothing happens."
    if norm.startswith("observation:"):
        norm = norm[len("observation:") :].strip()
    for pat in _ERROR_OBSERVATION_PATTERNS:
        if norm == pat or norm.startswith(pat):
            return True
    return False


def annotate_detour_and_error_steps(trajectories: List[AgentTrajectory]) -> Dict[str, int]:
    """
    Causal-Preserving Step Masking:
    Keep erroneous / duplicate steps in the conversation history (preserving causal state transitions
    for o_{t+1}), while setting `loss_mask = 0` so SFT does not imitate invalid or looping actions.
    """
    stats = {
        "total_steps": 0,
        "kept_steps": 0,
        "masked_env_error": 0,
        "masked_duplicate_action": 0,
    }

    for traj in trajectories:
        n = len(traj.steps)
        for idx, step in enumerate(traj.steps):
            stats["total_steps"] += 1
            step.loss_mask = 1
            step.mask_reason = None

            # 1. Consecutive identical action loop
            if idx > 0 and step.action.strip().lower() == traj.steps[idx - 1].action.strip().lower():
                step.loss_mask = 0
                step.mask_reason = "duplicate_consecutive_action"
                stats["masked_duplicate_action"] += 1
                continue

            # 2. Next observation reports an invalid/no-op action
            if idx + 1 < n:
                next_obs = traj.steps[idx + 1].observation
                if _is_error_observation(next_obs):
                    step.loss_mask = 0
                    step.mask_reason = "env_noop_or_invalid_action"
                    stats["masked_env_error"] += 1
                    continue

            stats["kept_steps"] += 1

    return stats
