from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("ensemble_density_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return "✅ Ensemble Density Prediction API is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Detect JSON or form-data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        print("🔹 Received data:", data)  # Debug log

        # Required keys
        required_keys = [
            "Temperature",
            "Volume_Concentration",
            "Density_NanoParticle_1",
            "Density_NanoParticle_2",
            "Density_BaseFluid",
            "Volume_Mixture_Particle_1",
            "Volume_Mixture_Particle_2",
        ]

        # Check missing keys
        missing = [k for k in required_keys if k not in data]
        if missing:
            return jsonify({"error": f"Missing keys: {missing}"}), 400

        # Convert to float
        Temperature = float(data["Temperature"])
        Volume_Concentration = float(data["Volume_Concentration"])
        Density_NP1 = float(data["Density_NanoParticle_1"])
        Density_NP2 = float(data["Density_NanoParticle_2"])
        Density_Base = float(data["Density_BaseFluid"])
        Volume_Mix1 = float(data["Volume_Mixture_Particle_1"])
        Volume_Mix2 = float(data["Volume_Mixture_Particle_2"])

        # Prepare input for prediction
        input_query = np.array([[Temperature, Volume_Concentration,
                                 Density_NP1, Density_NP2,
                                 Density_Base, Volume_Mix1, Volume_Mix2]])

        # Predict
        result = model.predict(input_query)[0]

        return jsonify({"Predicted_Density": float(result)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
