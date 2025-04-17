
# 🔐 Viralytics Quantum AI Cybersecurity API

A quantum-enhanced AI API for real-time threat detection on cybersecurity data.

> Upload a `.csv` file. Get back instant threat classification, risk scores, and intelligent, explainable results.

---

## 🚀 Features

- 🧠 Quantum-processed feature mapping (via PennyLane)
- ⚠️ Risk classification: Low, Medium, High
- 📊 Confidence, Certainty, and Risk Score metrics
- 🔍 Flagged rows & feature explainability
- 🔐 Secured with API key access

---

## 📦 API Endpoint

```
POST /predict-upload/
```

### 🔒 Headers

| Key | Value |
|-----|-------|
| Authorization | `Bearer your-api-key-here` |

### 📤 Form Data

| Key | Type | Description |
|-----|------|-------------|
| `file` | `.csv` | Your data file |

### 🔎 Optional Query Param

```bash
?features=feature1,feature2
```

Let users select which features to scan (defaults to first 2 numeric columns).

---

## ✅ Example `curl` Request

```bash
curl -X POST "https://your-api-url/predict-upload/?features=Flow%20Duration,Total%20Fwd%20Packets" \
  -H "Authorization: Bearer your-api-key" \
  -F "file=@yourfile.csv"
```

---

## 🔍 Sample Response

```json
{
  "threat_level": "Low",
  "confidence": 0.2402,
  "confidence_min": 0.0,
  "confidence_max": 0.9997,
  "certainty": 3.5523,
  "risk_score": 24,
  "explanation": "No significant anomalies detected.",
  "suggested_action": "Allow",
  "suspicious_rows": [76, 83, 46],
  "flagged_features": ["Flow Duration", "Total Fwd Packets"]
}
```

---

## 📁 File Requirements

- Format: `.csv`
- Must include at least **2 numeric columns**
- Only the first **100 rows** are analyzed per request

---

## 🧠 How It Works: Quantum-Enhanced AI

This API integrates **quantum circuit encoding** to model subtle, nonlinear patterns in cybersecurity data that classical models may overlook.

### ⚛️ Step-by-step:

1. **Feature Extraction**  
   Two selected numerical features (e.g. `Flow Duration`, `Packet Rate`) are pulled from your CSV file.

2. **Quantum Feature Mapping**  
   These values are embedded into a **quantum circuit** using PennyLane and simulated qubit gates (Hadamard, RX, RY, RZ, CNOT).

   This produces a **64-dimensional quantum probability vector** that captures entangled relationships between the inputs.

3. **Neural Inference**  
   The quantum output is passed into a **deep neural network** trained to detect cyber threats.

4. **Prediction & Certainty**  
   The model returns:
   - Risk Level: Low, Medium, or High
   - Confidence Range: min/max predictions
   - Entropy: to measure uncertainty
   - Flagged rows: most suspicious based on scores

### 🧪 Why Quantum?

Quantum circuits offer a **richer, more expressive representation of relationships**, ideal for:
- Noisy or adversarial data
- Nonlinear attack signatures
- Next-gen threat detection

This is not a traditional ML model — it’s a **quantum-classical hybrid** optimized for real-world security.

---

## 🛠 Tech Stack

- FastAPI + Uvicorn
- TensorFlow / Keras
- PennyLane (quantum circuit processing)
- Python 3.10+

---

## 📈 Monetization-Ready

This API is ready for:
- 🚀 Paid tier access via Whop or Stripe
- 🔑 Per-customer API keys
- 📊 Request-based billing

Want to try it or buy access? Reach out:  
📧 **vaidyajeet4@gmail.com**
