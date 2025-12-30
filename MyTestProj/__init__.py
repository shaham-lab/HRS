"""
PyHealthTest - MIMIC-IV Mortality Prediction Benchmark

This package contains benchmark implementations for healthcare predictive modeling
using the PyHealth library with MIMIC-IV data.
"""

__version__ = "1.0.0"
__author__ = "HRS Team"

# Make the main benchmark function easily importable
from .mimic4_mortality_benchmark import main as run_benchmark

__all__ = ["run_benchmark"]
