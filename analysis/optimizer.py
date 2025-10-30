import numpy as np
from models.quadratic_model import QuadraticRegressionModel

class FuelOptimizer:
    """Optimizes fuel consumption"""
    
    @staticmethod
    def calculate_optimal_consumption(model, v_opt):
        """Calculate optimal fuel consumption"""
        if not isinstance(model, QuadraticRegressionModel):
            raise ValueError("Model must be a QuadraticRegressionModel")
        
        coefficients, intercept = model.get_coefficients()
        a, b, c = intercept, coefficients[0], coefficients[1]
        y_opt = a + b * v_opt + c * v_opt**2
        return y_opt
