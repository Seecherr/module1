from sklearn.linear_model import RANSACRegressor, LinearRegression
import numpy as np
from .base_model import BaseModel
from config.settings import Config

class RANSACRegressionModel(BaseModel):
    """RANSAC robust regression model"""
    
    def __init__(self):
        super().__init__("RANSAC")
        self.model = RANSACRegressor(
            LinearRegression(),
            min_samples=Config.RANSAC_MIN_SAMPLES,
            residual_threshold=Config.RANSAC_RESIDUAL_THRESHOLD,
            max_trials=Config.RANSAC_MAX_TRIALS,
            random_state=Config.RANDOM_SEED
        )
        self.inlier_mask = None
    
    def fit(self, X_poly, y):
        self.model.fit(X_poly, y)
        self.inlier_mask = self.model.inlier_mask_
        self.coefficients = self.model.estimator_.coef_
        self.intercept = self.model.estimator_.intercept_
    
    def predict(self, X_poly):
        return self.model.predict(X_poly)
    
    def get_optimal_speed(self):
        """Calculate optimal speed from RANSAC coefficients"""
        if self.coefficients is None or len(self.coefficients) < 2:
            raise ValueError("Model not fitted or insufficient coefficients")
        
        b, c = self.coefficients[0], self.coefficients[1]
        return -b / (2 * c)
    
    def get_outlier_info(self, y):
        """Get information about outliers"""
        outlier_mask = ~self.inlier_mask
        return {
            'n_outliers': np.sum(outlier_mask),
            'n_inliers': np.sum(self.inlier_mask),
            'outlier_mask': outlier_mask
        }
