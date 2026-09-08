from flask import Flask, render_template
import pickle
import pandas as pd
from src.preprocessing.preprocess import load_data, preprocess

app = Flask(__name__)

# Load trained model
with open("models/saved/ids_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load and preprocess dataset once
train, test = load_data("data/raw/train.txt", "data/raw/test.txt")
X_train, X_test, y_train, y_test = preprocess(train, test)


@app.route("/")
def home():
    return render_template("index.html")

import random
@app.route("/predict")
def predict():

    # Pick a random sample from the test dataset
    index = random.randint(0, len(X_test) - 1)

    sample = X_test[index]

    prediction = model.predict([sample])[0]

    if prediction == 1:
        result = "⚠️ Attack Detected"
    else:
        result = "✅ Normal Traffic"

    return render_template(
        "index.html",
        prediction=result
    )


if __name__ == "__main__":
    app.run(debug=True)