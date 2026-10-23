# AgentReliabilityBench

**"AgentReliabilityBench: A Controlled Benchmark for Reliability, Consistency, and Conflict Resolution in LLM Agents"**

A controlled benchmark for measuring LLM agent reliability — consistency, conflict resolution, and repeatability — under instruction variation.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/AgentReliabilityBench.git
cd AgentReliabilityBench
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and add your API keys
```

Then load them before running:

```bash
# Linux/Mac
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match "^([^#][^=]+)=(.+)$") {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
    }
}
```

### 3. Run a pilot test (cheapest option)

```bash
python run_benchmark.py --pilot
```

This runs 1 model (GPT-4o-mini), 1 configuration (direct), 2 repeats — costs ~$0.50.

### 4. Run the full benchmark

```bash
# All models, all configs, 5 repeats (~27,000 API calls)
python run_benchmark.py

# Or customize:
python run_benchmark.py --models gpt-4o-mini gemini-1.5-flash --configs direct cot --repeats 3

# Test a single task family:
python run_benchmark.py --pilot --family SCHED-001
```

## Project Structure

```
AgentReliabilityBench/
├── run_benchmark.py                          # CLI entry point
├── requirements.txt                          # Python dependencies
├── .env.example                              # API key template
├── task_families_scheduling.json             # 20 scheduling task families
├── task_families_customer_service.json        # 20 customer service task families
├── task_families_information_retrieval.json   # 20 information retrieval task families
└── evaluator/
    ├── __init__.py
    ├── config.py       # Model registry (7 models), 3 agent configurations
    ├── providers.py    # OpenAI / Anthropic / Google / Together API abstraction
    ├── judge.py        # LLM-as-judge (GPT-4o-mini) constraint evaluation
    ├── metrics.py      # All 7 metrics: DCR, IAR, CRA, CPS, RS@k, AHR, RD
    ├── stats.py        # Statistical analysis: bootstrap CI, permutation tests, ICC
    └── runner.py       # Full pipeline orchestration
```

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **DCR** (Decision Consistency Rate) | Same answer to same question asked differently? |
| **IAR** (Instruction Adherence Rate) | Fraction of constraints satisfied |
| **CRA** (Conflict Resolution Accuracy) | Follows the right instruction when they conflict? |
| **CPS** (Context Perturbation Stability) | Maintains performance under added complexity? |
| **RS@k** (Repeatable Success) | All k repeated trials succeed? (Credits τ-bench pass^k) |
| **AHR** (Ambiguity Handling Rate) | Asks for clarification when info is missing? |
| **RD** (Reliability Degradation) | Rate of performance decline under stress |

## Models Supported

| Model | Provider | API Key Needed |
|-------|----------|---------------|
| GPT-4o | OpenAI | `OPENAI_API_KEY` |
| GPT-4o-mini | OpenAI | `OPENAI_API_KEY` |
| Claude 3.5 Sonnet | Anthropic | `ANTHROPIC_API_KEY` |
| Gemini 1.5 Flash | Google | `GOOGLE_API_KEY` |
| Llama 3.1 8B | Together AI | `TOGETHER_API_KEY` |
| Llama 3.1 70B | Together AI | `TOGETHER_API_KEY` |
| Qwen 2.5 72B | Together AI | `TOGETHER_API_KEY` |

## Output

Results are saved to `results/YYYYMMDD_HHMMSS/`:

- `all_raw_results.json` — Every API response and judge verdict
- `aggregate_metrics.json` — Computed metrics per model per configuration
- `raw_<model>.json` — Per-model raw results (saved incrementally)

## CLI Options

```
--models       Models to evaluate (space-separated)
--configs      Configurations: direct, cot, tool
--repeats      Number of repeated trials (default: 5)
--temperature  Sampling temperature (default: 0.7)
--pilot        Quick test: 1 model, 1 config, 2 repeats
--family       Run only one task family (e.g., SCHED-001)
--judge-model  Judge model (default: gpt-4o-mini)
--workers      Parallel workers (default: 4)
--output-dir   Where to save results (default: results/)
--task-dir     Where task family JSONs live (default: current dir)
```

## License

MIT

## Citation

```bibtex
@article{kapoor2026agentreliabilitybench,
  title={AgentReliabilityBench: A Controlled Benchmark for Reliability, Consistency, and Conflict Resolution in LLM Agents},
  author={Kapoor, Manmeet Singh and Kaur, Gurleen},
  journal={IEEE Access},
  year={2026}
}
```
