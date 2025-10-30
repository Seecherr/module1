import numpy as np
from sklearn.utils import resample
from config.settings import Config

class BootstrapAnalyzer:
    """Performs bootstrap analysis for confidence intervals"""
    
    def __init__(self, n_iterations=Config.N_BOOTSTRAP_ITERATIONS):
        self.n_iterations = n_iterations
        self.v_opt_samples = []
    
    def analyze(self, X_poly, y):
        """Perform bootstrap analysis"""
        print(f"----------- ЕТАП 5: АНАЛІЗ НЕВИЗНАЧЕНОСТІ (BOOTSTRAP) -----------")
        print(f"Запуск {self.n_iterations} ітерацій бутстрепу для v_opt...")
        
        for i in range(self.n_iterations):
            X_sample, y_sample = resample(X_poly, y)
            
            # Fit model on resampled data
            from models.quadratic_model import QuadraticRegressionModel
            boot_model = QuadraticRegressionModel()
            boot_model.fit(X_sample, y_sample)
            
            # Calculate optimal speed
            try:
                v_opt_boot = boot_model.get_optimal_speed()
                
                # Keep only realistic values
                if 0 < v_opt_boot < 150 and boot_model.coefficients[1] > 0:
                    self.v_opt_samples.append(v_opt_boot)
            except (ValueError, ZeroDivisionError):
                continue
        
        return self._calculate_confidence_interval()
    
    def _calculate_confidence_interval(self):
        """Calculate confidence interval from bootstrap samples"""
        v_opt_samples = np.array(self.v_opt_samples)
        alpha = (100 - Config.CONFIDENCE_LEVEL) / 2
        ci_lower = np.percentile(v_opt_samples, alpha)
        ci_upper = np.percentile(v_opt_samples, 100 - alpha)
        
        return ci_lower, ci_upper, v_opt_samples
