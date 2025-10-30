from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.coefficients = None
        self.intercept = None
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    def get_coefficients(self):
        return self.coefficients, self.intercept
