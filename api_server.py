import os
import logging

# Disable GPU for safety on Render free tier
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import tensorflow as tf
import numpy as np
import subprocess
import tempfile
import json

# ✅ Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load model
try:
    model = tf.keras.models.load_model("model.keras")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None

# ✅ API key from environment
MASTER_API_KEY = os.getenv("MASTER_API_KEY")
logger.info(f"🔐 MASTER_API_KEY Loaded: {'Yes' if MASTER_API_KEY else 'No'}")

@app.get("/")
def read_root():
    return {"message": "Quantum AI Cybersecurity API is running!"}

@app.get("/debug")
def debug():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "api_key_present": MASTER_API_KEY is not None
    }

@app.post("/predict-upload/")
async def predict_upload(
    file: UploadFile = File(...),
    authorization: str = Header(default=None),
    features: str = Query(default="")
):
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("❌ Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

    token = authorization.replace("Bearer ", "").strip()
    if token != MASTER_API_KEY:
        logger.warning("❌ Unauthorized API key attempt")
        raise HTTPException(status_code=403, detail="Unauthorized API key.")

    try:
        contents = await file.read()
        logger.info(f"📁 Received file: {file.filename}, size: {len(contents)} bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        logger.info(f"🗂️ Temp file created: {tmp_path}")

        cmd = ["python3", "quantum_ai_predict.py", tmp_path]
        if features:
            cmd.append(features)

        logger.info(f"🧠 Executing: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # ⏱️ timeout safety
        )

        os.remove(tmp_path)

        logger.info(f"🔁 Return code: {result.returncode}")
        logger.info(f"📤 STDOUT:\n{result.stdout.strip()}")
        if result.stderr.strip():
            logger.error(f"⚠️ STDERR:\n{result.stderr.strip()}")

        if result.returncode != 0:
            raise RuntimeError(f"Inference failed: {result.stderr}")

        if not result.stdout.strip():
            raise RuntimeError("No output returned from quantum_ai_predict.py")

        try:
            return JSONResponse(content=json.loads(result.stdout))
        except json.JSONDecodeError as json_err:
            logger.exception("❌ Failed to decode JSON from quantum_ai_predict output")
            return JSONResponse(status_code=500, content={
                "error": "Failed to parse script output",
                "stdout_raw": result.stdout,
                "stderr_raw": result.stderr
            })

    except subprocess.TimeoutExpired:
        logger.error("⏰ Prediction script timed out!")
        return JSONResponse(status_code=504, content={"error": "Prediction timed out."})
    except Exception as e:
        logger.exception("🔥 Exception during /predict-upload")
        return JSONResponse(status_code=500, content={"error": str(e)})
#h