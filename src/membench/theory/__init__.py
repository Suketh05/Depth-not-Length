"""Analytical models behind the benchmark.

This subpackage is the mathematical core: the geometric *similarity-decay* model
(``s_i = s_0 rho^i``), the full-chain *recovery bounds* for similarity vs.
structured memory (``P_sim`` and ``P_struct``), the *crossover-depth* solver that
locates ``d*`` once the structured store is charged its honest overhead, and the
*token-economics* model for the joint accuracy/cost win. Every quantity here is
both derived in closed form and fit to measured data, so predicted ``d*`` can be
compared against the empirically observed crossover.
"""
