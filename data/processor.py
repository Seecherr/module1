import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from config.settings import Config

class DataProcessor:
    """Processes data for model training"""
    
    @staticmethod
    def create_polynomial_features(X, degree=2):
        """Create polynomial features"""
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        return X_poly, poly_features
    
    @staticmethod
    def create_plot_data():
        """Create data for smooth plotting"""
        X_plot = np.linspace(*Config.PLOT_RANGE, Config.PLOT_POINTS).reshape(-1, 1)
        return X_plot
