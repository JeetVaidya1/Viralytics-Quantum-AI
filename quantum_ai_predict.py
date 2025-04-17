# quantum_ai_predict.py
import sys
import pandas as pd
import tensorflow as tf
import numpy as np
import json
import os
import urllib.parse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

RISK_THRESHOLDS = {
    "Low": (0.0, 0.35),
    "Medium": (0.35, 0.7),
    "High": (0.7, 1.0)
}

def classify_threat(score):
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= score < high:
            return level
    return "Unknown"

def suggest_action(level):
    return {
        "Low": "Allow",
        "Medium": "Quarantine",
        "High": "Block"
    }.get(level, "Review")

def generate_explanation(score):
    if score < 0.35:
        return "No significant anomalies detected."
    elif score < 0.7:
        return "Moderate anomaly pattern observed. May indicate suspicious behavior."
    else:
        return "High-risk pattern detected. File may be malicious."

def compute_entropy(scores):
    probs = scores / np.sum(scores)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return round(float(entropy), 4)

def quantum_feature_map(x):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev)
    def circuit(x):
        qml.Hadamard(wires=0)
        qml.RX(x[0], wires=1)
        qml.RY(x[1], wires=2)
        qml.RZ(x[0] * x[1], wires=3)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 4])
        qml.CNOT(wires=[4, 5])
        qml.RX(x[1] + x[0], wires=4)
        qml.RY(x[0] - x[1], wires=5)
        return qml.probs(wires=[0, 1, 2, 3, 4, 5])

    return circuit(x)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python quantum_ai_predict.py <path_to_csv> [feature1,feature2]"}))
        return

    csv_path = sys.argv[1]
    selected_features = None

    if len(sys.argv) == 3:
        raw_query = sys.argv[2]
        decoded = urllib.parse.unquote(raw_query)
        selected_features = decoded.split(",")

    try:
        print("📥 Loading model...")
        model = tf.keras.models.load_model("model.keras")

        print("📊 Reading CSV data...")
        df = pd.read_csv(csv_path)

        if selected_features:
            if not all(f in df.columns for f in selected_features):
                raise ValueError(f"Selected features {selected_features} not found in file.")
            X = df[selected_features].to_numpy().astype(np.float32)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                raise ValueError("CSV must contain at least two numeric columns.")
            selected_features = numeric_cols[:2]
            X = df[selected_features].to_numpy().astype(np.float32)

        X = X[:100]  # ✅ Limit to first 100 rows for performance
        print(f"⚛️ Applying quantum feature map to {len(X)} rows...")

        quantum_transformed = np.array([quantum_feature_map(row) for row in X])

        print("🤖 Running prediction...")
        predictions = model.predict(quantum_transformed, verbose=0).flatten()

        avg_conf = float(np.mean(predictions))
        conf_min = float(np.min(predictions))
        conf_max = float(np.max(predictions))
        entropy = compute_entropy(predictions)

        threat_level = classify_threat(avg_conf)
        risk_score = int(avg_conf * 100)
        explanation = generate_explanation(avg_conf)
        action = suggest_action(threat_level)
        suspicious_rows = [int(i) for i in np.argsort(predictions)[-3:]]
        flagged_features = selected_features

        result = {
            "threat_level": threat_level,
            "confidence": round(avg_conf, 4),
            "confidence_min": round(conf_min, 4),
            "confidence_max": round(conf_max, 4),
            "certainty": entropy,
            "risk_score": risk_score,
            "explanation": explanation,
            "suggested_action": action,
            "suspicious_rows": suspicious_rows,
            "flagged_features": flagged_features
        }

        print("✅ Prediction complete.")
        print(json.dumps(result))

    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
