import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
import joblib
import datetime
from skyfield.api import load

# ----------------------------------------
# FastAPI App
# ----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Load AI Model & Scaler
# ----------------------------------------
model = tf.keras.models.load_model("mars_lstm.h5", compile=False)
scaler = joblib.load("mars_scaler.pkl")

# ----------------------------------------
# Load NASA Ephemeris
# ----------------------------------------
ts = load.timescale()
eph = load("de421.bsp")
earth = eph["earth"]
mars = eph["mars"]

# ----------------------------------------
# Helpers
# ----------------------------------------
def get_mars_position(t):
    astrometric = earth.at(t).observe(mars)
    x, y, z = astrometric.position.au
    return float(x), float(y), float(z)

# ----------------------------------------
# Health Check (Render needs this)
# ----------------------------------------
@app.get("/")
def home():
    return {"status": "SmartSky backend running"}

# ----------------------------------------
# Planetary Positions API (Lovable)
# ----------------------------------------
@app.get("/api/planets")
def get_planets(
    planets: str,
    startDate: str,
    endDate: str
):
    planet_list = [p.strip().lower() for p in planets.split(",")]

    start = datetime.datetime.fromisoformat(startDate).replace(tzinfo=datetime.timezone.utc)
    end = datetime.datetime.fromisoformat(endDate).replace(tzinfo=datetime.timezone.utc)

    data = {}

    current = start
    while current <= end:
        t = ts.utc(current)

        if "mars" in planet_list:
            x, y, z = get_mars_position(t)
            data.setdefault("mars", []).append({
                "date": current.isoformat(),
                "x": x,
                "y": y,
                "z": z
            })

        current += datetime.timedelta(days=1)

    return data

# ----------------------------------------
# Mars AI Prediction (Lovable)
# ----------------------------------------
@app.get("/api/predict/mars")
def predict_mars(days: int = 5):
    history = []
    now = datetime.datetime.now(datetime.timezone.utc)

    # last 7 days of Mars data
    for i in range(7):
        t = ts.utc(now - datetime.timedelta(days=7 - i))
        x, y, z = get_mars_position(t)
        history.append([x, y])

    history = np.array(history)
    history_scaled = scaler.transform(history).reshape(1, 7, 2)

    predictions = []
    current = history_scaled

    for _ in range(days):
        pred = model.predict(current, verbose=0)[0]
        predictions.append(pred)

        current = np.roll(current, -1, axis=1)
        current[0, -1] = pred

    predictions = scaler.inverse_transform(predictions)

    output = []
    for i in range(days):
        output.append({
            "day": i + 1,
            "x": float(predictions[i][0]),
            "y": float(predictions[i][1])
        })

    return {
        "planet": "Mars",
        "predictions": output
    }

# ----------------------------------------
# Lovable Prediction API Wrapper
# ----------------------------------------
@app.get("/api/predictions")
def lovable_predictions(
    planets: str = Query(...),
    days: int = 30
):
    if planets.lower() != "mars":
        return {"error": "Only Mars AI available in this version"}

    return predict_mars(days)
