import pandas as pd
from config.settings import Config

class FileHandler:
    """Handles file operations for saving results"""
    
    @staticmethod
    def save_summary(quad_model, v_opt, y_opt, ci_lower, ci_upper, v_opt_ransac):
        """Save summary results to CSV"""
        coefficients, intercept = quad_model.get_coefficients()
        
        summary_data = {
            'parameter': [
                'Quadratic Coef (a)', 
                'Quadratic Coef (b)', 
                'Quadratic Coef (c)',
                'v_opt (km/h)',
                'y_min (L/100km)',
                'v_opt_95_CI_lower',
                'v_opt_95_CI_upper',
                'v_opt_RANSAC (km/h)'
            ],
            'value': [
                intercept,
                coefficients[0],
                coefficients[1],
                v_opt,
                y_opt,
                ci_lower,
                ci_upper,
                v_opt_ransac
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(Config.OUTPUT_SUMMARY_FILE, index=False)
        return summary_df
    
    @staticmethod
    def save_metrics(metrics_df):
        """Save metrics to CSV"""
        metrics_df.to_csv(Config.OUTPUT_METRICS_FILE, index=False)
        return metrics_df
