"""Typed run configuration and the fairness lock.

Pydantic schemas for the run grid plus validation of the fairness invariant: the
same model, code-search tool, and per-dataset retrieval budget across every arm,
so memory architecture is the sole independent variable. Configuration is data;
this subpackage only loads and validates it.
"""
