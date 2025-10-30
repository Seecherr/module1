"""
Regression models for fuel consumption analysis.

This package provides:
- Base model interface for all regression models
- Linear regression implementation
- Quadratic (polynomial) regression implementation
- RANSAC robust regression for outlier handling
"""

from .base_model import BaseModel
from .linear_model import LinearRegressionModel
from .quadratic_model import QuadraticRegressionModel
from .ransac_model import RANSACRegressionModel

__all__ = [
    'BaseModel',
    'LinearRegressionModel', 
    'QuadraticRegressionModel',
    'RANSACRegressionModel'
]
