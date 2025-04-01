# /models/train_model.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import joblib

def train_model():
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # Train SVM with probability enabled
    model = SVC(probability=True)
    model.fit(X, y)
    
    # Save trained model
    joblib.dump(model, "model.pkl")

if __name__ == "__main__":
    train_model()
