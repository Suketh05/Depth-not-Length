"""Scoring: turning a model response into measured quantities.

Decision compliance, merge-ready correctness, the full suite of retrieval-quality
metrics (precision/recall@k, MRR, nDCG, MAP, full-chain recovery), cost-to-correct
in tokens and dollars, the useful-signal quantity ``U = tokens x precision``, and
calibration. Scoring never decides control flow; it only measures.
"""
