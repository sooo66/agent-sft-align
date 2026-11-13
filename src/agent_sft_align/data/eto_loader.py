from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_sft_align.schema import AgentStep, AgentTrajectory

_THOUGHT_ACTION_RE = re.compile(
    r"^\s*(?:Thought\s*:\s*(?P<thought>.*?)\s*)?(?:^|\n)\s*Action\s*:\s*(?P<action>.+?)\s*$",
    re.DOTALL | re.IGNORECASE,
)


def parse_thought_and_action(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse an assistant ReAct turn into (thought, action).
    Returns None if the turn does not contain an Action (e.g., an 'OK' handshake turn).
    """
    cleaned = text.strip()
    match = _THOUGHT_ACTION_RE.match(cleaned)
    if match:
        thought = (match.group("thought") or "").strip()
        action = match.group("action").strip()
        # Keep only the first line of action if trailing commentary slipped in
        action = action.splitlines()[0].strip()
        return thought, action

    # Fallback: look for the last 'Action:' occurrence in case Thought lacked prefix
    lower = cleaned.lower()
    idx = lower.rfind("action:")
    if idx != -1:
        raw_thought = cleaned[:idx].strip()
        raw_thought = re.sub(r"^\s*thought\s*:\s*", "", raw_thought, flags=re.IGNORECASE).strip()
        raw_action = cleaned[idx + len("action:") :].strip()
        if raw_action:
            return raw_thought, raw_action.splitlines()[0].strip()
    return None


def parse_sharegpt_record(record: Dict[str, Any], default_task: str = "alfworld") -> Optional[AgentTrajectory]:
    """Convert a ShareGPT-style ETO record into an AgentTrajectory."""
    conv = record.get("conversations", [])
    if len(conv) < 2:
        return None

    traj_id = str(record.get("id", "unknown"))
    handshake_turns: List[Dict[str, str]] = []
    steps: List[AgentStep] = []

    i = 0
    step_idx = 0
    while i + 1 < len(conv):
        user_turn = conv[i]
        gpt_turn = conv[i + 1]
        u_role = user_turn.get("from", "").lower()
        g_role = gpt_turn.get("from", "").lower()
        if u_role not in ("human", "user") or g_role not in ("gpt", "assistant"):
            i += 1
            continue

        obs_text = user_turn.get("value", "").strip()
        ans_text = gpt_turn.get("value", "").strip()
        parsed = parse_thought_and_action(ans_text)

        if parsed is None and len(steps) == 0:
            # Handshake turn (e.g., Instruction -> "OK") prior to episode start
            handshake_turns.append({"role": "user", "content": obs_text})
            handshake_turns.append({"role": "assistant", "content": ans_text})
        elif parsed is not None:
            thought, action = parsed
            steps.append(
                AgentStep(
                    step_idx=step_idx,
                    observation=obs_text,
                    teacher_thought=thought,
                    action=action,
                )
            )
            step_idx += 1
        i += 2

    if not steps:
        return None

    init_prompt = handshake_turns[0]["content"] if handshake_turns else steps[0].observation
    return AgentTrajectory(
        trajectory_id=traj_id,
        task_type=default_task,
        system_or_init_prompt=init_prompt,
        steps=steps,
        metadata={"handshake_turns": handshake_turns},
    )


def _resolve_eto_json_from_hf(repo_id: str, task: str, cache_dir: Optional[str] = None) -> Path:
    """Locate and download the task JSON file inside `agent-eto/eto-sft-trajectory`."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    task_key = task.lower()
    aliases = {
        "alfworld": ["alfworld"],
        "webshop": ["webshop"],
        "sciworld": ["sciworld", "scienceworld"],
    }.get(task_key, [task_key])

    candidates = [
        f for f in files if f.endswith(".json") and any(alias in f.lower() for alias in aliases)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find JSON file for task '{task}' in HF dataset '{repo_id}'. Available files: {files}"
        )
    # Prefer `*_sft.json` if multiple exist
    candidates.sort(key=lambda x: (0 if "sft" in x.lower() else 1, len(x)))
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=candidates[0],
        repo_type="dataset",
        cache_dir=cache_dir,
    )
    return Path(local_path)


def load_eto_trajectories(
    task: str = "alfworld",
    repo_id: str = "agent-eto/eto-sft-trajectory",
    local_json_path: Optional[str] = None,
    max_trajectories: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[AgentTrajectory]:
    """
    Load expert trajectories from local JSON or automatically pull from HuggingFace (`agent-eto/eto-sft-trajectory`).
    """
    path = Path(local_json_path) if local_json_path else _resolve_eto_json_from_hf(repo_id, task, cache_dir)
    with open(path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    trajectories: List[AgentTrajectory] = []
    for rec in raw_records:
        traj = parse_sharegpt_record(rec, default_task=task)
        if traj is not None:
            trajectories.append(traj)
            if max_trajectories is not None and len(trajectories) >= max_trajectories:
                break
    return trajectories
