import pickle
import os

def load_model():
    model_path = os.path.join("../models", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_encoder():
    encoder_path = os.path.join("../models", "ohe.pkl")
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    return encoder
