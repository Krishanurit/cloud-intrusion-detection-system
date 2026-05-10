import pickle
import os

model_path = "models/saved/ids_model.pkl"

if not os.path.exists(model_path):
    raise FileNotFoundError("❌ Train model first!")

with open(model_path, "rb") as f:
    model = pickle.load(f)

def predict(data):
    result = model.predict([data])
    return "Attack" if result[0] == 1 else "Normal"