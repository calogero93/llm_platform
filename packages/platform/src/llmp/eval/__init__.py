"""Evaluation harness: suites declared by modules, run by the platform, compared across runs."""

from llmp.eval.suite import EvalSuite
from llmp.eval.types import CaseResult, DatasetRef, FieldCounts, RunReport

__all__ = ["CaseResult", "DatasetRef", "EvalSuite", "FieldCounts", "RunReport"]
