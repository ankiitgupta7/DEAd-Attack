# /models/load_model.py
from config import config
import joblib
import os

def load_trained_model():
    if config.model_file:
        model_path = config.model_file
    else:
        # fallback: guess model filename from config
        model_path = f"model_{config.model_name}_{config.dataset_name}.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    return model
