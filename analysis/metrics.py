import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class MetricsCalculator:
    """Calculates and compares model metrics"""
    
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, model_name):
        """Calculate all metrics for a model"""
        return {
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R^2': r2_score(y_true, y_pred),
            'Max Error': np.max(np.abs(y_true - y_pred))
        }
    
    @staticmethod
    def create_metrics_dataframe(metrics_list):
        """Create a DataFrame from metrics list"""
        return pd.DataFrame(metrics_list)
