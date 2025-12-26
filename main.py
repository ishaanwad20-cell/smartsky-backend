from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import datetime
import numpy as np
import joblib

from skyfield.api import load, utc
from tensorflow.keras.models import load_model

app = FastAPI(title="SmartSky API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ephemeris and timescale
planets = load("de421.bsp")
ts = load.timescale()

# Load ML model and scaler ONCE
lstm_model = load_model("mars_lstm.h5", compile=False)
scaler = joblib.load("mars_scaler.pkl")


@app.get("/api/planet/{planet_name}")
def planet_positions(
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
    results = []

    for i in range(days):
        now = datetime.datetime.utcnow().replace(tzinfo=utc)
        t = ts.from_datetime(now + datetime.timedelta(days=i))
        pos = earth.at(t).observe(target).apparent()
        ra, dec, distance = pos.radec()

        results.append({
            "date": (now + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
            "ra_hours": round(ra.hours, 6),
            "dec_degrees": round(dec.degrees, 6),
            "distance_au": round(distance.au, 6),
        })

    return results


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
    history_scaled = scaler.transform(
        history_np.reshape(-1, 2)
    ).reshape(1, 7, 2)

    preds = []
    current = history_scaled

    for i in range(days):
        pred_scaled = lstm_model.predict(current, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0]

        preds.append({
            "day": i + 1,
            "ra_hours": round(float(pred[0]), 6),
            "dec_degrees": round(float(pred[1]), 6),
        })

        current = np.roll(current, -1, axis=1)
        current[0, -1, :] = pred_scaled[0]

    return preds
