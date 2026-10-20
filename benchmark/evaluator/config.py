"""
Configuration for AgentReliabilityBench evaluation runs.
"""

from dataclasses import dataclass, field
from typing import Optional

# ── Model registry ──────────────────────────────────────────────────
MODELS = {
    "gpt-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "cost_per_1m_input": 2.50,
        "cost_per_1m_output": 10.00,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "cost_per_1m_input": 3.00,
        "cost_per_1m_output": 15.00,
    },
    "gemini-3.5-flash-lite": {
        "provider": "google",
        "model_id": "gemini-3.5-flash-lite",
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
    },
    "llama-3.1-8b": {
        "provider": "together",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "cost_per_1m_input": 0.18,
        "cost_per_1m_output": 0.18,
    },
    "llama-3.1-70b": {
        "provider": "together",
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "cost_per_1m_input": 0.88,
        "cost_per_1m_output": 0.88,
    },
    "qwen-2.5-72b": {
        "provider": "together",
        "model_id": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "cost_per_1m_input": 1.20,
        "cost_per_1m_output": 1.20,
    },
}

# ── Agent configurations ────────────────────────────────────────────
CONFIGURATIONS = {
    "direct": {
        "system_suffix": "",
        "description": "Direct prompting — task instruction only",
    },
    "cot": {
        "system_suffix": (
            "\n\nBefore acting, think step by step. Lay out your reasoning, "
            "then provide your final answer or action."
        ),
        "description": "Chain-of-thought planning prepended",
    },
    "tool": {
        "system_suffix": (
            "\n\nYou have access to the following tools. Use them to complete the task. "
            "Always call the appropriate tool rather than answering from memory."
        ),
        "description": "Tool-enabled agent with function calling",
    },
}

# ── Defaults ────────────────────────────────────────────────────────
@dataclass
class RunConfig:
    """Parameters for a single evaluation run."""
    models: list[str] = field(default_factory=lambda: list(MODELS.keys()))
    configurations: list[str] = field(default_factory=lambda: list(CONFIGURATIONS.keys()))
    repeats: int = 5
    temperature: float = 0.7
    max_tokens: int = 2048
    task_families_dir: str = "../"
    output_dir: str = "results"
    seed: Optional[int] = None
    parallel_workers: int = 4
    timeout_seconds: int = 60
    retry_attempts: int = 2
    judge_model: str = "gemini-3.5-flash-lite"  # for constraint checking
    judge_provider: str = "google"  # provider for the judge model
    max_families: Optional[int] = None  # limit number of task families (None = all)
    skip_families: int = 0  # skip first N families (for resuming)
