from sklearn.linear_model import LinearRegression
from .base_model import BaseModel

class QuadraticRegressionModel(BaseModel):
    """Quadratic regression model"""
    
    def __init__(self):
        super().__init__("Quadratic")
        self.model = LinearRegression()
    
    def fit(self, X_poly, y):
        self.model.fit(X_poly, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
    
    def predict(self, X_poly):
        return self.model.predict(X_poly)
    
    def get_optimal_speed(self):
        """Calculate optimal speed from quadratic coefficients"""
        if self.coefficients is None or len(self.coefficients) < 2:
            raise ValueError("Model not fitted or insufficient coefficients")
        
        b, c = self.coefficients[0], self.coefficients[1]
        return -b / (2 * c)
