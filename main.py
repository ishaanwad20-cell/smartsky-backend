from fastapi import FastAPI, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import datetime
import joblib
import tensorflow as tf
from skyfield.api import load
import os
import uvicorn

# -------------------------------
# Load AI model and scaler
# -------------------------------

lstm_model = tf.keras.models.load_model("mars_lstm.h5")
scaler = joblib.load("mars_scaler.pkl")

# -------------------------------
# Load ephemeris
# -------------------------------

ts = load.timescale()
eph = load("de421.bsp")
earth = eph["earth"]
mars = eph["mars"]

# -------------------------------
# FastAPI setup
# -------------------------------

app = FastAPI(title="SmartSky Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter()

# -------------------------------
# Utility: get Mars position
# -------------------------------

def get_mars_position(t):
    pos = earth.at(t).observe(mars).position.au
    return pos[0], pos[1], pos[2]

# -------------------------------
# Health check
# -------------------------------

@api.get("/")
def root():
    return {"status": "SmartSky backend running 🚀"}

# -------------------------------
# Real-time Mars position
# -------------------------------

@api.get("/api/planet/mars")
def mars_now():
    t = ts.now()
    x, y, z = get_mars_position(t)
    return {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "x_au": x,
        "y_au": y,
        "z_au": z
    }

# -------------------------------
# AI Prediction endpoint
# -------------------------------

@api.get("/api/predict/mars")
def predict_mars(days: int = Query(5, ge=1, le=30)):

    history = []

    # Last 7 days of Mars positions
    for i in range(7):
        t = ts.utc(
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=7 - i)
        )
        x, y, z = get_mars_position(t)
        history.append([x, y])

    history = np.array(history)

    # Scale and reshape for LSTM
    history_scaled = scaler.transform(history).reshape(1, 7, 2)

    predictions = []
    current = history_scaled.copy()

    for i in range(days):
        pred_scaled = lstm_model.predict(current)[0]
        pred = scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]

        predictions.append({
            "day": i + 1,
            "x_au": float(pred[0]),
            "y_au": float(pred[1])
        })

        # Roll window forward
        new_scaled = scaler.transform(pred.reshape(1, -1))
        current = np.roll(current, -1, axis=1)
        current[0, -1] = new_scaled

    return {
        "planet": "mars",
        "days": days,
        "predictions": predictions
    }

# -------------------------------
# Register routes
# -------------------------------

app.include_router(api)

# -------------------------------
# Render / Local startup
# -------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
