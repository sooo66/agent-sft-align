from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentStep:
    """Represents a single (Observation -> Thought -> Action) turn in an agent trajectory."""

    step_idx: int
    observation: str
    teacher_thought: str
    action: str
    rewritten_thought: Optional[str] = None
    selected_source: str = "teacher"  # e.g., "rewritten_k2", "minimal", "teacher"
    loss_mask: int = 1  # 1 = supervise in SFT, 0 = keep in context only (detour/error step)
    mask_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def active_thought(self) -> str:
        if self.rewritten_thought is not None:
            return self.rewritten_thought
        return self.teacher_thought

    def format_assistant_turn(self, thought: Optional[str] = None, include_thought: bool = True) -> str:
        t = self.active_thought if thought is None else thought
        t = t.strip()
        a = self.action.strip()
        if not include_thought or t == "":
            return f"Action: {a}"
        return f"Thought: {t}\nAction: {a}"


@dataclass
class AgentTrajectory:
    """Multi-step interactive agent trajectory."""

    trajectory_id: str
    task_type: str
    system_or_init_prompt: str
    steps: List[AgentStep]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTrajectory":
        steps = [AgentStep(**s) for s in data["steps"]]
        return cls(
            trajectory_id=data["trajectory_id"],
            task_type=data["task_type"],
            system_or_init_prompt=data["system_or_init_prompt"],
            steps=steps,
            metadata=data.get("metadata", {}),
        )


@dataclass
class SpanLogProbMetrics:
    """Exact token-level log-probability statistics for a single step's Thought and Action spans."""

    thought_num_tokens: int
    action_num_tokens: int
    thought_sum_logprob: float
    thought_mean_logprob: float
    thought_ppl: float
    thought_cvar10_nll: float
    thought_max_token_nll: float
    action_sum_logprob: float
    action_mean_logprob: float
    action_ppl: float


@dataclass
class CandidateEvalResult:
    """Evaluation summary of a candidate rewritten thought under the ELBO objective."""

    candidate_id: str
    thought_text: str
    metrics: SpanLogProbMetrics
    baseline_no_thought_action_logprob: float
    action_info_gain: float
    elbo_score: float
