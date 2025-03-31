# /models/train_model.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
import joblib

def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Train logistic regression on 8x8 digit dataset
    model = LogisticRegression(max_iter=5000, solver='lbfgs', multi_class='auto')
    model.fit(X, y)
    
    # Save trained model
    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    train_model()
