import os
import logging

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

# ✅ Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

# ✅ Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load trained model
try:
    model = tf.keras.models.load_model("model.keras")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# ✅ Load API Key from environment
MASTER_API_KEY = os.getenv("MASTER_API_KEY")

@app.get("/")
def read_root():
    return {"message": "Quantum AI Cybersecurity API is running!"}

@app.get("/debug")
def debug():
    return {"status": "OK", "model_loaded": model is not None, "api_key_present": MASTER_API_KEY is not None}

@app.post("/predict-upload/")
async def predict_upload(
    file: UploadFile = File(...),
    authorization: str = Header(default=None),
    features: str = Query(default="")
):
    # ✅ Check API Key from header against env key
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    token = authorization.replace("Bearer ", "").strip()
    if token != MASTER_API_KEY:
        logger.warning("Unauthorized API key attempt")
        raise HTTPException(status_code=403, detail="Unauthorized API key.")

    try:
        contents = await file.read()
        logger.info(f"📁 Received file: {file.filename}, size: {len(contents)} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        cmd = ["python3", "quantum_ai_predict.py", tmp_path]
        if features:
            cmd.append(features)

        logger.info(f"🧠 Running prediction script with command: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        os.remove(tmp_path)

        logger.info(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            raise RuntimeError(f"Inference failed: {result.stderr}")

        if not result.stdout.strip():
            raise RuntimeError("No output from predictor script.")

        return JSONResponse(content=json.loads(result.stdout))

    except Exception as e:
        logger.exception("🔥 Exception during /predict-upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
