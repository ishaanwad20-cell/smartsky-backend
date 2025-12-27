import datetime
import numpy as np
import joblib
import tensorflow as tf

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from skyfield.api import load, utc

# ---------------------------
# App
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Load astronomy data
# ---------------------------
ts = load.timescale()
planets = load("de421.bsp")
earth = planets["earth"]
mars = planets["mars"]

# ---------------------------
# Load AI
# ---------------------------
model = tf.keras.models.load_model("mars_lstm.h5")
scaler = joblib.load("mars_scaler.pkl")

# ---------------------------
# Utilities
# ---------------------------
def get_mars_position(t):
    pos = earth.at(t).observe(mars).position.au
    return pos[0], pos[1], pos[2]

# ---------------------------
# Root
# ---------------------------
@app.get("/")
def root():
    return {"status": "SmartSky backend running 🚀"}

# ---------------------------
# AI Prediction
# ---------------------------
@app.get("/api/predict/mars")
def predict_mars(days: int = 5):

    # Build last 7 days of Mars history
    history = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for i in range(7):
        t = ts.utc(now - datetime.timedelta(days=7 - i))
        x, y, z = get_mars_position(t)
        history.append([x, y])

    history = np.array(history)
    scaled = scaler.transform(history)
    X = scaled.reshape(1, 7, 2)

    predictions = []

    last = X.copy()

    for i in range(days):
        pred = model.predict(last, verbose=0)[0]
        unscaled = scaler.inverse_transform([pred])[0]

        date = (now + datetime.timedelta(days=i + 1)).date().isoformat()

        predictions.append({
            "date": date,
            "x_au": float(unscaled[0]),
            "y_au": float(unscaled[1])
        })

        next_scaled = scaler.transform([unscaled])
        last = np.append(last[:, 1:, :], next_scaled.reshape(1, 1, 2), axis=1)

    return {
        "planet": "Mars",
        "days": days,
        "predictions": predictions
    }

# ---------------------------
# Live Mars Position
# ---------------------------
@app.get("/api/planet/mars")
def mars_now():
    t = ts.now()
    x, y, z = get_mars_position(t)

    return {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "x_au": float(x),
        "y_au": float(y),
        "z_au": float(z)
    }

# ---------------------------
# Lovable compatibility
# ---------------------------
@app.get("/api/planets")
def lovable_planets(planets: str, startDate: str, endDate: str):
    return {
        "status": 200,
        "planets": planets.split(","),
        "data": [],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.get("/api/predictions")
def lovable_predictions(planets: str, days: int = 30):
    if "mars" not in planets.lower():
        return {"status": 200, "data": []}

    ai = predict_mars(days)

    return {
        "status": 200,
        "planet": "Mars",
        "days": days,
        "data": ai["predictions"],
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
