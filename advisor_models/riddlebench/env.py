"""RiddleBench domain environments for SkyRL training.

Three progressive env classes for ablation study:
- RiddleBenchStandardEnv: standard GRPO, single response, R_outcome only
- RiddleBenchDiagEnv:     + diagnosis format + k=2 responses, R_outcome only
- RiddleBenchAdvisorEnv:  full method: + R_diag + R_adh + counterfactual subtraction
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig

from ..env_base import BaseAdvisorEnv
from .config import (
    ADVISOR_INSTRUCTIONS,
    ADVISOR_SYSTEM_PROMPT,
    STUDENT_SYSTEM_PROMPT,
    compute_diagnosis_reward,
    compute_riddle_score,
    compute_specificity_reward,
    extract_advice_section,
)


class RiddleBenchStandardEnv(BaseAdvisorEnv):
    """Ablation 1: standard GRPO advisor, single initial response, R_outcome only."""

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        msgs = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.original_question},
        ]
        return msgs, self.original_question

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        if "</think>" in advisor_feedback:
            advisor_feedback = advisor_feedback.split("</think>", 1)[1]
        user_content = (
            f"{advisor_feedback.strip()}\n\n"
            "Your previous answer was wrong. Do NOT adjust it — start completely "
            "from scratch, reason step by step, then give your final answer."
        )
        msgs = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.original_question},
            {"role": "assistant", "content": self.original_response},
            {"role": "user", "content": user_content},
        ]
        return msgs, "[ADVISOR FEEDBACK]"

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        r_outcome, msg = compute_riddle_score(self.final_response, self.ground_truth)
        return r_outcome, True, {"r_outcome": r_outcome, "msg": msg}

    def _get_metadata(self) -> Dict[str, Any]:
        meta = super()._get_metadata()
        meta["other_info"] = str(self.reward_info)
        return meta


class RiddleBenchDiagEnv(BaseAdvisorEnv):
    """Ablation 2: diagnosis format + k=2 initial responses, R_outcome only."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)
        k = extras.get("k_responses", [self.original_response])
        self.k_responses: List[str] = k if isinstance(k, list) else [self.original_response]

    def _build_baseline_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        msgs = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.original_question},
        ]
        return msgs, self.original_question

    def _build_advisor_prompt(
        self, prompt: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], str]:
        attempts = "\n\n".join(
            f"Attempt {i + 1}:\n{r}" for i, r in enumerate(self.k_responses)
        )
        user_content = (
            f"{self.original_question}\n\n"
            f"The student made the following attempt(s):\n\n{attempts}\n\n"
            f"{ADVISOR_INSTRUCTIONS}"
        )
        msgs = [
            {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return msgs, user_content

    def _build_student_prompt(
        self, advisor_feedback: str
    ) -> Tuple[List[Dict[str, str]], str]:
        if "</think>" in advisor_feedback:
            advisor_feedback = advisor_feedback.split("</think>", 1)[1]
        # Pass full output (diagnosis + advice) so student gets maximum context
        user_content = (
            f"{advisor_feedback.strip()}\n\n"
            "Your previous answer was wrong. Do NOT adjust it — start completely "
            "from scratch, reason step by step, then give your final answer."
        )
        msgs = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.original_question},
            {"role": "assistant", "content": self.original_response},
            {"role": "user", "content": user_content},
        ]
        return msgs, "[FULL ADVISOR OUTPUT]"

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        r_outcome, msg = compute_riddle_score(self.final_response, self.ground_truth)
        return r_outcome, True, {"r_outcome": r_outcome, "msg": msg}

    def _get_metadata(self) -> Dict[str, Any]:
        meta = super()._get_metadata()
        meta["other_info"] = str(self.reward_info)
        return meta


class RiddleBenchAdvisorEnv(RiddleBenchDiagEnv):
    """Full method: diagnosis format + k=2 + multi-component reward + counterfactual."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)
        self.error_type: str = extras["reward_spec"].get("error_type", "")
        self.null_reward: float = float(extras["reward_spec"].get("null_reward", 0.0))
        self.is_null_advice: bool = bool(extras["reward_spec"].get("is_null_advice", False))

    def _compute_step(self) -> Tuple[float, bool, Dict[str, Any]]:
        alpha, beta, gamma = 1.0, 0.3, 0.2

        r_outcome, msg = compute_riddle_score(self.final_response, self.ground_truth)
        r_diag = compute_diagnosis_reward(self.action, self.error_type)
        r_spec = compute_specificity_reward(self.action, self.original_question)

        r_total = alpha * r_outcome + beta * r_diag + gamma * r_spec

        info = {
            "r_outcome": r_outcome,
            "r_diag": r_diag,
            "r_spec": r_spec,
            "r_total": r_total,
            "msg": msg,
        }
        return r_total, True, info
