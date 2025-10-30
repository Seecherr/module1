"""
Analysis and metrics calculation modules.

This package provides:
- Performance metrics calculation (RMSE, MAE, R², etc.)
- Fuel consumption optimization
- Bootstrap analysis for confidence intervals
- Statistical validation methods
"""

from .metrics import MetricsCalculator
from .optimizer import FuelOptimizer
from .bootstrap import BootstrapAnalyzer

__all__ = ['MetricsCalculator', 'FuelOptimizer', 'BootstrapAnalyzer']
