from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class LinearRegressionModel(BaseModel):
    """Linear regression model"""
    
    def __init__(self):
        super().__init__("Linear")
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
    
    def predict(self, X):
        return self.model.predict(X)
