#!/usr/bin/env python3
"""
AgentReliabilityBench — Main entry point.

Usage:
  # Full run (all models, all configs, 5 repeats)
  python run_benchmark.py

  # Pilot run (one model, one config, 2 repeats)
  python run_benchmark.py --pilot

  # Custom
  python run_benchmark.py --models gpt-4o-mini gemini-3.5-flash-lite \
                          --configs direct cot \
                          --repeats 3

  # Single family test
  python run_benchmark.py --pilot --family SCHED-001
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env file automatically

from evaluator.config import RunConfig, MODELS, CONFIGURATIONS
from evaluator.runner import run_evaluation, load_task_families

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("benchmark_run.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AgentReliabilityBench Evaluation")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Models to evaluate. Available: {list(MODELS.keys())}")
    parser.add_argument("--configs", nargs="+", default=None,
                        help=f"Configurations. Available: {list(CONFIGURATIONS.keys())}")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of repeated trials per task (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--task-dir", type=str, default=".",
                        help="Directory containing task_families_*.json files")
    parser.add_argument("--pilot", action="store_true",
                        help="Run a quick pilot: 1 cheap model, 1 config, 2 repeats")
    parser.add_argument("--family", type=str, default=None,
                        help="Run only a specific task family ID")
    parser.add_argument("--judge-model", type=str, default="gemini-3.5-flash-lite",
                        help="Model to use as judge (default: gemini-3.5-flash-lite)")
    parser.add_argument("--judge-provider", type=str, default="google",
                        help="Provider for judge model (default: google)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--max-families", type=int, default=None,
                        help="Limit number of task families (default: all)")

    args = parser.parse_args()

    # Build config
    if args.pilot:
        max_fam = args.max_families or 5  # pilot defaults to 5 families
        config = RunConfig(
            models=["gemini-3.5-flash-lite"],
            configurations=["direct"],
            repeats=2,
            temperature=args.temperature,
            task_families_dir=args.task_dir,
            output_dir=args.output_dir,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            parallel_workers=1,
            max_families=max_fam,
        )
        logger.info(f"🧪 PILOT MODE: 1 model, 1 config, 2 repeats, {max_fam} families")
    else:
        config = RunConfig(
            models=args.models or list(MODELS.keys()),
            configurations=args.configs or list(CONFIGURATIONS.keys()),
            repeats=args.repeats,
            temperature=args.temperature,
            task_families_dir=args.task_dir,
            output_dir=args.output_dir,
            judge_model=args.judge_model,
            judge_provider=args.judge_provider,
            parallel_workers=args.workers,
            max_families=args.max_families,
        )

    # Validate
    for m in config.models:
        if m not in MODELS:
            logger.error(f"Unknown model: {m}. Available: {list(MODELS.keys())}")
            sys.exit(1)
    for c in config.configurations:
        if c not in CONFIGURATIONS:
            logger.error(f"Unknown config: {c}. Available: {list(CONFIGURATIONS.keys())}")
            sys.exit(1)

    # Print plan
    families = load_task_families(config.task_families_dir)
    total_calls = (
        len(families) * len(list(families[0]["variants"].keys())) *
        config.repeats * len(config.models) * len(config.configurations)
    )
    logger.info(f"\n{'='*60}")
    logger.info(f"AgentReliabilityBench Evaluation")
    logger.info(f"{'='*60}")
    logger.info(f"  Task families:   {len(families)}")
    logger.info(f"  Variants/family: {len(list(families[0]['variants'].keys()))}")
    logger.info(f"  Models:          {config.models}")
    logger.info(f"  Configurations:  {config.configurations}")
    logger.info(f"  Repeats:         {config.repeats}")
    logger.info(f"  Temperature:     {config.temperature}")
    logger.info(f"  Judge model:     {config.judge_model}")
    logger.info(f"  Total API calls: ~{total_calls} (agent) + ~{total_calls} (judge)")
    logger.info(f"  Output:          {config.output_dir}")
    logger.info(f"{'='*60}\n")

    # Run
    results = run_evaluation(config)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for model_name, model_data in results.items():
        for config_key, metrics in model_data.items():
            print(f"\n{model_name} / {config_key}:")
            print(f"  DCR: {metrics.get('dcr', 'N/A'):.3f}" if isinstance(metrics.get('dcr'), float) else f"  DCR: N/A")
            print(f"  IAR (neutral): {metrics['iar'].get('neutral', 'N/A'):.3f}" if isinstance(metrics['iar'].get('neutral'), float) else f"  IAR: N/A")
            print(f"  CRA: {metrics.get('cra', 'N/A'):.3f}" if isinstance(metrics.get('cra'), float) else f"  CRA: N/A")
            print(f"  CPS: {metrics.get('cps', 'N/A'):.3f}" if isinstance(metrics.get('cps'), float) else f"  CPS: N/A")
            print(f"  AHR: {metrics.get('ahr', 'N/A'):.3f}" if isinstance(metrics.get('ahr'), float) else f"  AHR: N/A")
            rs = metrics.get("rs_at_k", {})
            if rs:
                for k, v in rs.items():
                    print(f"  RS@{k}: {v:.3f}" if isinstance(v, float) else f"  RS@{k}: N/A")


if __name__ == "__main__":
    main()
