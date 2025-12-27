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
# App
# ----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Load AI model & scaler
# ----------------------------------------
model = tf.keras.models.load_model("mars_lstm.h5", compile=False)
scaler = joblib.load("mars_scaler.pkl")

# ----------------------------------------
# Load ephemeris
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
# Routes
# ----------------------------------------

@app.get("/")
def home():
    return {"status": "SmartSky backend running 🚀"}

@app.get("/api/planet/mars")
def live_mars():
    t = ts.now()
    x, y, z = get_mars_position(t)
    return {
        "time": t.utc_iso(),
        "x_au": x,
        "y_au": y,
        "z_au": z
    }

@app.get("/api/predict/mars")
def predict_mars(days: int = 5):
    # Build 7-day history
    history = []

    for i in range(7):
        t = ts.utc(
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=7 - i)
        )
        x, y, z = get_mars_position(t)
        history.append([x, y])

    history = np.array(history)
    history_scaled = scaler.transform(history)
    history_scaled = history_scaled.reshape(1, 7, 2)

    preds = []

    for i in range(days):
        pred = model.predict(history_scaled, verbose=0)[0]
        preds.append({"day": i+1, "x": float(pred[0]), "y": float(pred[1])})

        next_step = np.array([[pred[0], pred[1]]])
        next_step = scaler.transform(next_step)
        history_scaled = np.append(history_scaled[:,1:,:], [[next_step]], axis=1)

    return {
        "planet": "Mars",
        "predictions": preds
    }

# ----------------------------------------
# Lovable-compatible endpoints
# ----------------------------------------

@app.get("/api/predictions")
def lovable_predictions(planets: str = "Mars", days: int = 30):
    data = predict_mars(days)
    return data

@app.get("/api/planets")
def lovable_planets(
    planets: str,
    startDate: str,
    endDate: str
):
    start = datetime.datetime.fromisoformat(startDate).replace(tzinfo=datetime.timezone.utc)
    end = datetime.datetime.fromisoformat(endDate).replace(tzinfo=datetime.timezone.utc)

    results = []

    current = start
    while current <= end:
        t = ts.utc(current)
        x, y, z = get_mars_position(t)
        results.append({
            "date": current.isoformat(),
            "x": x,
            "y": y,
            "z": z
        })
        current += datetime.timedelta(days=1)

    return {
        "planet": planets,
        "data": results
    }
