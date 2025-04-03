# /models/train_model.py


import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import numpy as np
import joblib
from config import config
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def load_dataset(dataset_name):
    if dataset_name == "digits":
        data = load_digits()
        X, y = data.data, data.target
        # We know scikit-learn digits is shape 8×8
        config.image_height = 8
        config.image_width = 8
        config.pixel_max = 16  # or 255 if you prefer
        return X, y
    elif dataset_name == "mnist":
        mnist = fetch_openml("mnist_784", version=1)
        X, y = mnist.data, mnist.target
        # MNIST is 28×28 if we reshape
        config.image_height = 28
        config.image_width = 28
        config.pixel_max = 255
        return X, y
    elif dataset_name == "fashion_mnist":
        fmnist = fetch_openml("Fashion-MNIST", version=1)
        X, y = fmnist.data, fmnist.target
        config.image_height = 28
        config.image_width = 28
        config.pixel_max = 255
        return X, y
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_model(model_name):
    if model_name == "SVM":
        return SVC(probability=True)
    elif model_name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    elif model_name == "MLP":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def train_model():
    # 1) Load dataset and set config info
    X, y = load_dataset(config.dataset_name)

    print(f"Dataset: {config.dataset_name}, shape: {X.shape}, label set size: {len(np.unique(y))}")
    print(f"Image dims: {config.image_height}x{config.image_width}, pixel range: 0..{config.pixel_max}")

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3) Create model
    model = create_model(config.model_name)
    print(f"Training {config.model_name} on {config.dataset_name}...")

    # 4) Fit model
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Training complete. Test accuracy: {accuracy:.4f}")

    # 5) Save model
    filename = f"model_{config.model_name}_{config.dataset_name}.pkl"
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

    # 6) Update config with model file path
    config.model_file = filename

if __name__ == "__main__":
    train_model()
