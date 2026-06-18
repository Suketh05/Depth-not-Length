"""Dataset loaders, all emitting the uniform :class:`membench.types.Task`.

Covers the three locked datasets (dcbench, SWE-bench, LongMemEval), the
spec-stripping / depth-dialing harness shared by the two coding datasets, a
controllable *synthetic* depth-``d`` chain generator for clean crossover
measurement, and the depth-labelling protocol (automated + dual-annotation with
inter-annotator agreement).
"""
