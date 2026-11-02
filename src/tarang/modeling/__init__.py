"""TARANG predictive modeling package.

Provides reference-aligned regression benchmarks, baseline estimators, evaluation
metrics, and experiment execution workflows for duration and energy targets.
"""

from tarang.modeling.models import (
    EmptyModel,
    MeanValue,
    MedianValue,
    LastValue,
    get_benchmark_models,
)
from tarang.modeling.metrics import (
    absolute_error,
    absolute_percentage_error,
    squared_error,
    symmetric_mean_absolute_percentage_error,
    calculate_metrics,
)
from tarang.modeling.experiment import (
    validate_modeling_input,
    run_experiment,
)

__all__ = [
    "EmptyModel",
    "MeanValue",
    "MedianValue",
    "LastValue",
    "get_benchmark_models",
    "absolute_error",
    "absolute_percentage_error",
    "squared_error",
    "symmetric_mean_absolute_percentage_error",
    "calculate_metrics",
    "validate_modeling_input",
    "run_experiment",
]

