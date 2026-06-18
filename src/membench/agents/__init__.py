"""The agent loop and language-model clients.

Holds the LLM client abstraction (Anthropic / OpenAI / deterministic-stub
backends with per-call token and dollar accounting) and the task runner that
wires a memory arm to a model: ``write`` the corpus, ``retrieve`` under the fixed
budget, prompt the model, score, and emit one structured row per attempt.
"""
