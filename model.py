# model.py
import numpy as np
import pandas as pd
from stochtree import BART

class BartModel:
    def __init__(self):
        self.model = BART()

    def fit(self, X, y):
        """
        Fit the BART model with provided data
        X: numpy array of features
        y: numpy array of target values
        """
        try:
            self.model.sample(X, y, 200, 100)
        except Exception as e:
            print(f"Error during model fitting: {e}")
            raise

    def predict(self, X):
        """
        Make predictions using the BART model
        X: numpy array of features
        """
        y_hat_samples = self.model.predict(X)
        y_hat_avg = y_hat_samples.mean(axis=1, keepdims=True)
        return y_hat_avg

def process_input_data(input_data):
    """
    Process the input data to be suitable for model prediction
    """
    # Convert input data to numpy array
    X = np.array(input_data)
    return X

def generate_sample_data():
    """
    Generate sample data for testing/demo purposes
    """
    rng = np.random.default_rng(101)
    n = 1000
    p = 10
    X = rng.uniform(0, 1, (n, p))
    y = X[:, 0] * 100 + X[:, 1] * 2 + rng.normal(0, 1, n)
    return X, y
