"""
Main evaluation runner for AgentReliabilityBench.

Orchestrates:
  1. Loading task families from JSON
  2. Sending prompts to models via providers
  3. Judging responses against constraint checklists
  4. Computing metrics
  5. Saving results
"""

import json
import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

from .config import MODELS, CONFIGURATIONS, RunConfig
from .providers import get_provider, LLMResponse
from .judge import Judge
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ── Task loading ────────────────────────────────────────────────────

def load_task_families(directory: str) -> list[dict]:
    """Load all task family JSON files from a directory."""
    families = []
    for fname in sorted(Path(directory).glob("task_families_*.json")):
        with open(fname) as f:
            families.extend(json.load(f))
    logger.info(f"Loaded {len(families)} task families from {directory}")
    return families


def build_system_prompt(family: dict, variant: dict, config_key: str) -> str:
    """
    Build the system prompt for a task, including environment
    context and configuration suffix (e.g., CoT instruction).
    """
    env = family.get("environment", {})
    config = CONFIGURATIONS[config_key]

    parts = [
        f"You are an AI agent operating in the {family['domain']} domain.",
        f"\n## Environment\n```json\n{json.dumps(env, indent=2)}\n```",
    ]

    # Add tool descriptions if tool config
    tools = env.get("tools_available", [])
    if tools and config_key == "tool":
        parts.append("\n## Available Tools\n" + "\n".join(f"- {t}" for t in tools))

    # Priority rule for conflict variants
    if "priority_rule" in variant:
        parts.append(f"\n## Priority Rule\n{variant['priority_rule']}")

    parts.append(config["system_suffix"])
    return "\n".join(parts)


# ── Single task execution ───────────────────────────────────────────

def run_single_task(
    provider,
    family: dict,
    variant_name: str,
    variant: dict,
    config_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict:
    """
    Send one task variant to one model and return the raw response.
    """
    system_prompt = build_system_prompt(family, variant, config_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": variant["prompt"]},
    ]

    try:
        resp = provider.complete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "family_id": family["family_id"],
            "variant": variant_name,
            "config": config_key,
            "response": resp.content,
            "tool_calls": resp.tool_calls,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
            "finish_reason": resp.finish_reason,
            "status": "success",
        }
    except Exception as e:
        logger.error(f"Error on {family['family_id']}/{variant_name}: {e}")
        return {
            "family_id": family["family_id"],
            "variant": variant_name,
            "config": config_key,
            "response": "",
            "status": "error",
            "error": str(e),
        }


# ── Full evaluation run ────────────────────────────────────────────

def run_evaluation(config: RunConfig) -> dict:
    """
    Run the full evaluation pipeline.

    Returns a nested results dict saved to disk.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    families = load_task_families(config.task_families_dir)
    if config.max_families:
        families = families[:config.max_families]
        logger.info(f"  Limited to {config.max_families} task families")
    judge = Judge(model=config.judge_model, provider_name=config.judge_provider)

    all_results = {}

    for model_name in config.models:
        model_info = MODELS[model_name]
        logger.info(f"\n{'='*60}\nEvaluating model: {model_name}\n{'='*60}")

        provider = get_provider(model_info["provider"], model_info["model_id"])
        model_results = {}

        for config_key in config.configurations:
            logger.info(f"  Configuration: {config_key}")
            config_results = {}

            for family in families:
                fid = family["family_id"]
                family_results = {}
                trial_outcomes = []

                for repeat in range(config.repeats):
                    logger.info(f"    {fid} repeat {repeat+1}/{config.repeats}")
                    repeat_results = {}

                    for vname, variant in family["variants"].items():
                        # Run task
                        result = run_single_task(
                            provider, family, vname, variant,
                            config_key, config.temperature,
                            config.max_tokens, config.timeout_seconds,
                        )

                        # Judge response
                        if result["status"] == "success":
                            judge_result = judge.evaluate(
                                task_prompt=variant["prompt"],
                                agent_response=result["response"],
                                expected_outcome=variant.get("expected_outcome", {}),
                                constraint_checklist=variant.get("constraint_checklist", []),
                                priority_rule=variant.get("priority_rule"),
                                variant_type=variant["variant_type"],
                            )
                            result["judge"] = judge_result
                        else:
                            result["judge"] = {"overall": "ERROR"}

                        repeat_results[vname] = result

                    # Track trial pass/fail for RS@k (based on neutral variant)
                    neutral_judge = repeat_results.get("neutral", {}).get("judge", {})
                    trial_outcomes.append(neutral_judge.get("overall") == "PASS")

                    # Store repeat
                    family_results[f"repeat_{repeat}"] = repeat_results

                family_results["trial_outcomes"] = trial_outcomes
                config_results[fid] = family_results

            model_results[config_key] = config_results

        all_results[model_name] = model_results

        # Save per-model results incrementally
        model_file = output_dir / f"raw_{model_name}.json"
        with open(model_file, "w") as f:
            json.dump(model_results, f, indent=2, default=str)
        logger.info(f"  Saved raw results to {model_file}")

    # Save combined raw results
    with open(output_dir / "all_raw_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Compute aggregate metrics ───────────────────────────────
    logger.info("\nComputing aggregate metrics...")
    aggregate = compute_aggregate_metrics(all_results)

    with open(output_dir / "aggregate_metrics.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {output_dir}")
    return aggregate


def compute_aggregate_metrics(all_results: dict) -> dict:
    """
    Compute per-model, per-config aggregate metrics from raw results.
    """
    aggregate = {}

    for model_name, model_results in all_results.items():
        model_agg = {}

        for config_key, config_results in model_results.items():
            # For metric computation, use the last repeat's judge results
            # (or average across repeats for more robust estimates)
            results_by_family = {}
            trial_outcomes_map = {}

            for fid, family_data in config_results.items():
                # Average across repeats: use last repeat for simplicity
                last_repeat_key = f"repeat_{max(int(k.split('_')[1]) for k in family_data if k.startswith('repeat_'))}"
                last_repeat = family_data.get(last_repeat_key, {})

                family_judge_results = {}
                for vname, vresult in last_repeat.items():
                    if isinstance(vresult, dict) and "judge" in vresult:
                        family_judge_results[vname] = vresult["judge"]

                if family_judge_results:
                    results_by_family[fid] = family_judge_results

                if "trial_outcomes" in family_data:
                    trial_outcomes_map[fid] = family_data["trial_outcomes"]

            metrics = compute_all_metrics(results_by_family, trial_outcomes_map)
            model_agg[config_key] = metrics

        aggregate[model_name] = model_agg

    return aggregate
