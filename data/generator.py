import numpy as np
from config.settings import Config

class DataGenerator:
    """Generates synthetic fuel consumption data"""
    
    @staticmethod
    def true_fuel_consumption(speed):
        """True fuel consumption function"""
        return Config.TRUE_A + Config.TRUE_B * speed + Config.TRUE_C * speed**2
    
    @classmethod
    def generate_data(cls):
        """Generate synthetic data with noise and outliers"""
        np.random.seed(Config.RANDOM_SEED)
        
        v_data = np.random.uniform(*Config.SPEED_RANGE, Config.N_SAMPLES)
        noise = np.random.normal(0, Config.NOISE_STD, Config.N_SAMPLES)
        y_data = cls.true_fuel_consumption(v_data) + noise
        
        v_data = np.append(v_data, Config.OUTLIERS_SPEED)
        y_data = np.append(y_data, Config.OUTLIERS_CONSUMPTION)
        
        X = v_data.reshape(-1, 1)
        y = y_data
        
        return X, y, v_data, y_data
