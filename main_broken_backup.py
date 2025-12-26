import joblib
from skyfield.api import utc
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from skyfield.api import load
import datetime

app = FastAPI(title="SmartSky API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a stable ephemeris that supports named planets
planets = load("de421.bsp")
ts = load.timescale()

@app.get("/api/planet/{planet_name}")
def get_planet_position(
    planet_name: str,
    days: int = Query(3, ge=1, le=30)
):
    planet_name = planet_name.lower()

    try:
        target = planets[planet_name]
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Ephemeris data for '{planet_name}' not available."
        )

    earth = planets["earth"]
    now = datetime.datetime.utcnow()
    results = []

    for i in range(days):
        now = datetime.datetime.utcnow().replace(tzinfo=utc)
        t = ts.from_datetime(now - datetime.timedelta(days=7 - i))
        pos = earth.at(t).observe(target).apparent()
        ra, dec, distance = pos.radec()

        results.append({
            "date": (now + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
            "ra_hours": round(ra.hours, 6),
            "dec_degrees": round(dec.degrees, 6),
            "distance_au": round(distance.au, 6),
        })

    return results
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load trained model & scaler (only once)
lstm_model = load_model("mars_lstm.h5", compile=False)
lstm_scaler = joblib.load("mars_scaler.pkl")

from skyfield.api import utc

@app.get("/api/predict/mars")
def predict_mars(days: int = 5):
    earth = planets["earth"]
    target = planets["mars"]

    history = []

    for i in range(7):
        now = datetime.datetime.utcnow().replace(tzinfo=utc)
        t = ts.from_datetime(now - datetime.timedelta(days=7 - i))
        pos = earth.at(t).observe(target).apparent()
        ra, dec, _ = pos.radec()
        history.append([ra.hours, dec.degrees])

    history_np = np.array(history).reshape(1, 7, 2)
    history_scaled = scaler.transform(history_np.reshape(-1, 2)).reshape(1, 7, 2)

    preds = []
    current_input = history_scaled

    for i in range(days):
        pred_scaled = lstm_model.predict(current_input, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0]
        preds.append({
            "day": i + 1,
            "ra_hours": round(float(pred[0]), 6),
            "dec_degrees": round(float(pred[1]), 6)
        })

        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1, :] = pred_scaled[0]

    return preds
