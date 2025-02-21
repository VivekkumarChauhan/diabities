from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and scaler
with open("diabetes_detection_model.pkl", "rb") as model_file:model, scaler = pickle.load(model_file)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        required_keys = [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age"
        ]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            return jsonify({"error": f"Missing fields: {', '.join(missing_keys)}"}), 400

        input_features = [
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["blood_pressure"]),
            float(data["skin_thickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["diabetes_pedigree_function"]),
            float(data["age"]),
        ]

        scaled_features = scaler.transform([input_features])
        prediction = model.predict(scaled_features)[0]
        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"

        return jsonify({
            "prediction": result,
            "input_features": input_features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
