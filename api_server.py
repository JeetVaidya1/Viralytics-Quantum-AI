import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tensorflow as tf
import numpy as np
from io import StringIO
import subprocess
import tempfile
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ Loading model...")
model = tf.keras.models.load_model("model.keras")
print("✅ Model loaded successfully.")

MASTER_API_KEY = os.getenv("MASTER_API_KEY")

@app.get("/")
def read_root():
    return {"message": "Quantum AI Cybersecurity API is running!"}

@app.post("/predict-upload/")
async def predict_upload(
    file: UploadFile = File(...),
    authorization: str = Header(default=None),
    features: str = Query(default="")
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    token = authorization.replace("Bearer ", "").strip()
    if token != MASTER_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized API key.")

    try:
        contents = await file.read()
        print(f"INFO:     📁 Received file: {file.filename}, size: {len(contents)} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        print(f"INFO:     🗂️ Temp file path: {tmp_path}")

        cmd = ["python3", "quantum_ai_predict.py", tmp_path]
        if features:
            cmd.append(features)

        print(f"INFO:     🧠 Running prediction script with command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        print(f"DEBUG RETURN CODE: {result.returncode}")
        print(f"DEBUG STDOUT:\n{result.stdout}")
        print(f"DEBUG STDERR:\n{result.stderr}")

        os.remove(tmp_path)

        if result.returncode != 0:
            return JSONResponse(status_code=500, content={"error": f"Inference failed", "stderr": result.stderr})

        if not result.stdout.strip():
            return JSONResponse(status_code=500, content={"error": "No output received from model script."})

        return JSONResponse(content=json.loads(result.stdout))

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
