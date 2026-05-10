from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("models/saved/ids_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.form["features"]
    data = list(map(float, data.split(",")))

    result = model.predict([data])[0]

    output = "⚠️ Attack Detected!" if result == 1 else "✅ Normal"

    return render_template("index.html", prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)