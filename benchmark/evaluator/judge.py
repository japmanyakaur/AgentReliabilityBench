"""
LLM-as-judge for evaluating agent responses against constraint checklists.
Uses a cheap model (GPT-4o-mini by default) to grade each constraint.
"""

import json
import logging
from typing import Optional

from .providers import get_provider, LLMResponse

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for an LLM agent benchmark.
You will be given:
1. A TASK PROMPT that was sent to the agent.
2. The AGENT'S RESPONSE.
3. The EXPECTED OUTCOME (what a correct response should contain/do).
4. A CONSTRAINT CHECKLIST (list of constraints the response must satisfy).

For each constraint, output PASS or FAIL with a one-line justification.
Then give an overall PASS (all constraints met) or FAIL.

Also evaluate:
- Did the agent request clarification? (yes/no)
- Did the agent use a tool? (yes/no, and which tool)
- Did the agent follow the priority rule if one was given? (yes/no/na)

Output valid JSON with this schema:
{
  "constraints": {
    "<constraint_name>": {"verdict": "PASS"|"FAIL", "reason": "..."},
    ...
  },
  "overall": "PASS"|"FAIL",
  "requested_clarification": true|false,
  "used_tool": true|false,
  "tool_names": ["..."],
  "followed_priority": true|false|null,
  "notes": "any additional observations"
}
"""


class Judge:
    """Evaluates agent responses against task constraints."""

    def __init__(self, model: str = "gemini-1.5-flash", provider_name: str = "google"):
        self.provider = get_provider(provider_name, model)

    def evaluate(
        self,
        task_prompt: str,
        agent_response: str,
        expected_outcome: dict,
        constraint_checklist: list[str],
        priority_rule: Optional[str] = None,
        variant_type: str = "baseline",
    ) -> dict:
        """
        Run LLM judge to check each constraint.

        Returns dict with per-constraint verdicts and overall pass/fail.
        """
        user_prompt = self._build_prompt(
            task_prompt, agent_response, expected_outcome,
            constraint_checklist, priority_rule, variant_type,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = self.provider.complete(
                messages, temperature=0.0, max_tokens=1500
            )
            result = self._parse_judge_output(resp.content)
            result["judge_tokens"] = {
                "input": resp.input_tokens,
                "output": resp.output_tokens,
            }
            return result
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {
                "constraints": {c: {"verdict": "ERROR", "reason": str(e)} for c in constraint_checklist},
                "overall": "ERROR",
                "error": str(e),
            }

    def _build_prompt(
        self, task_prompt, agent_response, expected_outcome,
        constraint_checklist, priority_rule, variant_type,
    ) -> str:
        parts = [
            f"VARIANT TYPE: {variant_type}",
            f"\nTASK PROMPT:\n{task_prompt}",
            f"\nAGENT RESPONSE:\n{agent_response}",
            f"\nEXPECTED OUTCOME:\n{json.dumps(expected_outcome, indent=2)}",
            f"\nCONSTRAINT CHECKLIST:\n" + "\n".join(f"- {c}" for c in constraint_checklist),
        ]
        if priority_rule:
            parts.append(f"\nPRIORITY RULE:\n{priority_rule}")
        return "\n".join(parts)

    def _parse_judge_output(self, text: str) -> dict:
        """Extract JSON from judge response, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            # Remove markdown fences
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            return {"overall": "PARSE_ERROR", "raw": text}
