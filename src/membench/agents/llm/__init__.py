"""Language-model clients with per-call token and dollar accounting.

Every client returns an :class:`LLMResponse` carrying the model's reported input
and output token counts and the dollar cost, because per-call tokens are a
reported control and cost-to-correct in dollars is a primary payoff metric -- both
must travel with each response, not be reconstructed later.

The deterministic stub (:class:`StubLLMClient`) lets the whole pipeline run offline
and reproducibly: it "honours" exactly the decisions whose identifiers appear in
the retrieved context, so an arm that recovers a governing decision scores compliant
and an arm that does not scores a miss -- the accuracy and token-cost story fall out
of real retrieval differences with no API calls. Real Anthropic and OpenAI backends
implement the same interface for live runs.
"""
