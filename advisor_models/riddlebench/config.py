"""Configuration for RiddleBench domain.

Contains system prompts, error categories, and scoring for logic/spatial/constraint puzzles.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

ADVISOR_SYSTEM_PROMPT = (
    "You are an expert puzzle-solving coach. Your student got a logic puzzle wrong. "
    "Your job is to give targeted, specific corrective guidance that references the "
    "actual numbers, names, or constraints from THIS problem — not generic advice that "
    "could apply to any puzzle."
)

STUDENT_SYSTEM_PROMPT = (
    "You are a puzzle solver. Follow the advisor's guidance carefully and provide "
    "a concise, correct answer to the puzzle."
)

ADVISOR_INSTRUCTIONS = """\
The student attempted the puzzle above but got it wrong. Respond using EXACTLY this format:

<diagnosis>
State precisely what the student got wrong, referencing specific values or names from \
the problem (e.g. "The student computed 9-5=4 but the jump from 5 to 6 is +1, not +4").
</diagnosis>
<advice>
Give one concrete, step-by-step corrective action that references the actual numbers, \
names, or constraints in THIS problem. Show the student exactly which values to compute \
or which constraint to re-examine. Do NOT reveal the final answer. Do NOT give advice \
so generic it could apply to any puzzle (e.g. "identify the pattern" is NOT acceptable).
</advice>"""

ERROR_CATEGORIES: Dict[str, str] = {
    "sequence tasks": "LOGICAL_ERROR",
    "coding and decoding sum": "LOGICAL_ERROR",
    "blood relations": "FALSE_ASSUMPTION",
    "seating task": "MISSED_CONSTRAINT",
}

_CATEGORY_KEYWORDS: Dict[str, list[str]] = {
    "LOGICAL_ERROR": ["logical", "logic", "reasoning", "pattern", "calculation", "arithmetic"],
    "FALSE_ASSUMPTION": ["assumption", "relation", "assume", "misidentif", "incorrectly identified"],
    "MISSED_CONSTRAINT": ["constraint", "condition", "missed", "overlook", "ignored", "rule"],
    "SPATIAL_ERROR": ["spatial", "order", "position", "arrangement", "seating"],
    "MISREAD": ["misread", "misunderstood", "question", "overlooked"],
}


def compute_riddle_score(response: str, ground_truth: str) -> Tuple[float, str]:
    """Binary score: 1.0 if ground_truth appears correctly in response, else 0.0."""
    response_clean = response.strip().lower()
    gt_clean = ground_truth.strip().lower()

    if not gt_clean:
        return 0.0, "empty ground truth"

    # Single-letter MCQ — look for letter as a standalone answer marker
    # Patterns: "answer ... D", "option D", "D.", "**D**", "(D)", letter on own line
    if len(gt_clean) == 1 and gt_clean.isalpha():
        letter = gt_clean.upper()
        patterns = [
            rf"(?:answer|option|correct)[^.!?\n]{{0,30}}\b{letter}\b",
            rf"\*\*{letter}[.\s*]",
            rf"\({letter}\)",
            rf"\b{letter}\.[^\w]",
            rf"^\s*{letter}\s*$",
        ]
        for pat in patterns:
            if re.search(pat, response, re.IGNORECASE | re.MULTILINE):
                return 1.0, "correct (MCQ)"
        return 0.0, "incorrect (MCQ)"

    # Numeric or short string answer — containment check
    if gt_clean in response_clean:
        return 1.0, "correct"
    return 0.0, "incorrect"


def extract_tagged_section(text: str, tag: str) -> str:
    """Extract content between <tag>...</tag>. Returns empty string if not found."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_advice_section(advisor_output: str) -> str:
    """Extract <advice> content; strip </think> blocks; fallback to full output."""
    # Strip Qwen3 thinking blocks
    if "</think>" in advisor_output:
        advisor_output = advisor_output.split("</think>", 1)[1]

    advice = extract_tagged_section(advisor_output, "advice")
    if advice:
        return advice

    # Fallback: strip diagnosis block and return the rest
    no_diag = re.sub(r"<diagnosis>.*?</diagnosis>", "", advisor_output, flags=re.DOTALL | re.IGNORECASE)
    return no_diag.strip() or advisor_output.strip()


def compute_diagnosis_reward(advisor_output: str, error_type: str) -> float:
    """
    0.5  if <diagnosis> section is present and non-empty (format compliance)
    +0.5 if the diagnosis text mentions any keyword associated with error_type
    """
    diagnosis = extract_tagged_section(advisor_output, "diagnosis")
    if not diagnosis:
        return 0.0

    score = 0.5  # format compliance
    if error_type and error_type in _CATEGORY_KEYWORDS:
        diag_lower = diagnosis.lower()
        if any(kw in diag_lower for kw in _CATEGORY_KEYWORDS[error_type]):
            score += 0.5

    return score


def compute_specificity_reward(advisor_output: str, question: str) -> float:
    """Reward advice that references actual numbers/names from the problem.

    Penalises generic boilerplate that doesn't mention any problem-specific content.
    Returns a value in [0, 1].
    """
    advice = extract_advice_section(advisor_output)
    if not advice:
        return 0.0

    # Extract numbers and capitalised names from the question
    numbers = set(re.findall(r'\b\d+\.?\d*\b', question))
    names = set(re.findall(r'\b[A-Z][a-z]+\b', question))
    specific_terms = numbers | names
    if not specific_terms:
        return 0.5  # no landmarks to check; neutral score

    hits = sum(1 for t in specific_terms if t in advice)
    # Reward increases with hits; saturates at ~3 hits
    return min(1.0, hits / max(1, min(3, len(specific_terms))))
