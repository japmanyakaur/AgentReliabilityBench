# AgentReliabilityBench

A benchmark for measuring how reliably LLM-based agents follow instructions when those instructions are rephrased, complicated, or made deliberately contradictory. Instead of testing what a model *can* do, this tests whether it does the *same thing consistently* across surface-level variations of the same task.

The core idea: take a task, produce five variants of it (neutral, paraphrased, high-constraint, conflicting, ambiguous), run each through an agent, and measure whether the outputs stay consistent and correct.

## What this measures

Seven metrics, each targeting a different failure mode:

| Metric | Question it answers |
|--------|-------------------|
| DCR (Decision Consistency Rate) | Does the agent give the same answer when the question is rephrased? |
| IAR (Instruction Adherence Rate) | What fraction of explicit constraints does it actually satisfy? |
| CRA (Conflict Resolution Accuracy) | When instructions contradict, does it follow the stated priority? |
| CPS (Context Perturbation Stability) | Does performance hold up when you pile on extra constraints? |
| RS@k (Repeatable Success) | If you run the same task k times, do all k succeed? |
| AHR (Ambiguity Handling Rate) | Does it ask for clarification when information is genuinely missing? |
| RD (Reliability Degradation) | How fast does performance drop as task complexity increases? |

RS@k is adapted from the pass^k metric in [tau-bench](https://github.com/sierra-research/tau-bench) (Sierra Research, 2024).

## Task structure

60 task families across three domains:

- **Scheduling** (20 families) -- meeting coordination, calendar conflicts, timezone handling
- **Customer service** (20 families) -- policy lookup, complaint resolution, escalation decisions
- **Information retrieval** (20 families) -- multi-source queries, filtering, ranked results

Each family has five variants sharing the same underlying task but differing in surface form or constraint structure. The `neutral` and `paraphrase` variants should produce identical outcomes (these form the equivalence subset for DCR). The `high_constraint`, `conflict`, and `ambiguous` variants test robustness.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/AgentReliabilityBench.git
cd AgentReliabilityBench
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

The benchmark uses a judge model (Gemini 3.5 Flash Lite by default) to evaluate agent responses against constraint checklists. You need at least one agent API key and a judge API key.

**Supported providers:**

| Provider | Env variable | Models |
|----------|-------------|--------|
| Google | `GOOGLE_API_KEY` | Gemini 3.5 Flash Lite (also used as default judge) |
| Groq | `GROQ_API_KEY` | GPT-OSS 120B, GPT-OSS 20B, Qwen 3.8 27B |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini |


## Running the benchmark

**Pilot run** (sanity check -- small, fast, cheap):

```bash
python run_benchmark.py --pilot
```

**Full run** (all registered models, all configs, 5 repeats):

```bash
python run_benchmark.py
```

**Custom runs:**

```bash
# Specific models and configs
python run_benchmark.py --models gemini-3.5-flash-lite gpt-oss-120b --configs direct cot --repeats 3

# Single task family (debugging)
python run_benchmark.py --pilot --family SCHED-001

# Resume after hitting a daily quota limit
python run_benchmark.py --models gemini-3.5-flash-lite --skip-families 15
```

The runner saves results incrementally per model, so if a run is interrupted (rate limits, quota exhaustion), you don't lose completed work.

## Agent configurations

Each model is tested under three prompting strategies:

- **direct** -- the task instruction alone, no additional scaffolding
- **cot** -- chain-of-thought prompt appended ("think step by step...")
- **tool** -- function-calling mode with tool definitions provided

This lets you compare how much prompting strategy affects reliability independently of model capability.

## Evaluation pipeline

The judge evaluates each constraint independently and returns a structured verdict. This avoids the known problems with holistic LLM-as-judge scoring (position bias, verbosity preference) by reducing each judgment to a binary pass/fail on a specific, concrete constraint.

## Output

Results land in `results/YYYYMMDD_HHMMSS/`:
