# app.py
import os
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model (joblib file)
MODEL_PATH = "gbc_model.pkl"
model = joblib.load(MODEL_PATH)

# Feature order MUST match the order used during training
FEATURES = [
    "Sex",
    "Birth Year",
    "Chest pain",
    "Chills or sweats",
    "Confused or disoriented",
    "Cough",
    "Diarrhea",
    "Difficulty breathing or Dyspnea",
    "Fatigue or general weakness",
    "Fever",
    "Fluid in lung cavity in auscultation",
    "Fluid in cavity through X-Ray",
    "Headache",
    "Joint pain or arthritis",
    "Thorax (sore throat)",
    "Muscle pain",
    "Nausea",
    "Other clinical symptoms",
    "Rapid breathing",
    "Runny nose",
    "Maculopapular rash",
    "Sore throat or pharyngitis",
    "Bleeding or bruising",
    "Vomiting",
    "Abnormal lung X-Ray findings",
    "Conjunctivitis",
    "Acute respiratory distress syndrome",
    "Pneumonia (clinical or radiologic)",
    "Loss of Taste",
    "Loss of Smell",
    "Cough with sputum",
    "Cough with heamoptysis",
    "Enlarged lymph nodes",
    "Wheezing",
    "Skin ulcers",
    "Inability to walk",
    "Indrawing of chest wall",
    "Other complications"
]

def build_feature_vector_from_json(json_data):
    """Return numpy array shaped (1, n_features) using JSON keys matching FEATURES.
       Missing keys default to 0 (or 0.0 for float features)."""
    vals = []
    for feat in FEATURES:
        # if key present in JSON, use it; else default to 0
        if feat in json_data:
            vals.append(json_data[feat])
        else:
            vals.append(0)
    # ensure numeric types
    return np.array([vals], dtype=float)

def build_feature_vector_from_form(form):
    vals = []
    for feat in FEATURES:
        raw = form.get(feat, None)
        if raw is None or raw == "":
            vals.append(0)
        else:
            # try to coerce to float
            try:
                vals.append(float(raw))
            except ValueError:
                # if user submitted "Male"/"Female", map to 1/0
                s = str(raw).strip().lower()
                if s in ("male", "m", "1"):
                    vals.append(1.0)
                elif s in ("female", "f", "0"):
                    vals.append(0.0)
                elif s in ("yes", "y", "true", "1"):
                    vals.append(1.0)
                else:
                    vals.append(0.0)
    return np.array([vals], dtype=float)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Try JSON first
    json_data = request.get_json(silent=True)
    try:
        if json_data:
            X = build_feature_vector_from_json(json_data)
            pred = model.predict(X)[0]
            proba = float(model.predict_proba(X)[0][1])

            # Map prediction to human-friendly message
            if pred == 0:
                message = "You have not contracted COVID-19"
            else:
                message = "You most likely have contracted COVID-19"

            # If request was JSON, return JSON
            if request.is_json:
                return jsonify({"prediction": message, "probability": proba})
            else:
                return render_template("index.html", prediction_text=f"{message} (Probability: {proba:.2f})")

        # Otherwise, handle form POST (from HTML)
        form = request.form
        X = build_feature_vector_from_form(form)
        pred = model.predict(X)[0]
        proba = float(model.predict_proba(X)[0][1])

        if pred == 0:
            message = "You have not contracted COVID-19"
        else:
            message = "You most likely have contracted COVID-19"

        return render_template("index.html", prediction_text=f"{message} (Probability: {proba:.2f})")

    except Exception as e:
        # return error as JSON for API clients, and in-page message for browser
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    # For local testing only; production runs via gunicorn
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
