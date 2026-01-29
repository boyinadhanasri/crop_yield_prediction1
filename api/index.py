from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "..", "model.pkl")
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

@app.route("/")
def home():
    return "Crop Yield Prediction API is running 🌾🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
