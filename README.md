# AgentReliabilityBench
A benchmark for measuring how reliably LLM-based agents follow instructions when those instructions are rephrased, complicated, or made deliberately contradictory. Instead of testing what a model can do, this tests whether it does the same thing consistently across surface-level variations of the same task.

The core idea: take a task, produce five variants of it (neutral, paraphrased, high-constraint, conflicting, ambiguous), run each through an agent, and measure whether the outputs stay consistent and correct.
