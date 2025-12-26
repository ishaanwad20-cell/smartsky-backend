from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from skyfield.api import load
import numpy as np
import datetime
import joblib
import tensorflow as tf

# ================================
# APP CONFIG
# ================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# LOAD EPHEMERIS
# ================================

planets = load("de421.bsp")
ts = load.timescale()

# ================================
# LOAD ML MODEL
# ================================

lstm_model = tf.keras.models.load_model("mars_lstm.h5", compile=False)
scaler = joblib.load("mars_scaler.pkl")

# ================================
# HELPER
# ================================

def get_mars_position(t):
    mars = planets["mars"]
    earth = planets["earth"]
    astrometric = earth.at(t).observe(mars)
    x, y, z = astrometric.position.au
    return x, y, z

# ================================
# API ROUTES
# ================================

@app.get("/")
def home():
    return {"status": "SmartSky backend running 🚀"}

# -----------------------
# Current Mars Position
# -----------------------

@app.get("/api/planet/mars")
def mars_position():
    t = ts.now()
    x, y, z = get_mars_position(t)
    return {
        "time": t.utc_iso(),
        "x_au": float(x),
        "y_au": float(y),
        "z_au": float(z)
    }

# -----------------------
# LSTM Prediction
# -----------------------

@app.get("/api/predict/mars")
def predict_mars(days: int = Query(5, ge=1, le=30)):

    # Get last 7 days of Mars data
    history = []
    for i in range(7):
        t = ts.utc(datetime.datetime.utcnow() - datetime.timedelta(days=7 - i))
        x, y, z = get_mars_position(t)
        history.append([x, y])

    history = np.array(history)

    # Scale
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

        # Roll window
        new_scaled = scaler.transform(pred.reshape(1, -1))
        current = np.roll(current, -1, axis=1)
        current[0, -1] = new_scaled

    return {
        "planet": "mars",
        "predictions": predictions
    }
