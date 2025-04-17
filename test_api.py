import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 🚫 Disable GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Avoid OneDNN errors (macOS specific)

import tensorflow as tf
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ✅ Load your model safely now
model = tf.keras.models.load_model("model.keras")

app = FastAPI()

class InputData(BaseModel):
    features: list[float]

@app.get("/")
def root():
    return {"message": "Quantum AI Cybersecurity API is running!"}

@app.post("/predict")
def predict(data: InputData):
    try:
        input_array = np.array(data.features).reshape(1, -1)
        prediction = model.predict(input_array)[0][0]
        result = "⚠️ THREAT DETECTED" if prediction > 0.5 else "✅ SAFE"
        return {"prediction": result, "confidence": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
