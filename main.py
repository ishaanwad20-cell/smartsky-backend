from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import datetime
import pytz
import joblib
import tensorflow as tf

from skyfield.api import load

# ------------------ App Setup ------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load Ephemeris ------------------

eph = load("de421.bsp")
ts = load.timescale()

# ------------------ Load AI Model ------------------

model = tf.keras.models.load_model("mars_lstm.h5", compile=False)
scaler = joblib.load("mars_scaler.pkl")

# ------------------ Physics Engine ------------------

def get_planet_position(planet_name, t):
    sun = eph["sun"]
    planet = eph[planet_name]
    pos = sun.at(t).observe(planet).position.km
    return float(pos[0]), float(pos[1]), float(pos[2])

# ------------------ Planets ------------------

PLANETS = ["mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]

# ------------------ REAL POSITIONS ------------------

@app.get("/api/planet/{planet}")
def get_planet(planet: str, days: int = 5):
    planet = planet.lower()

    if planet not in PLANETS:
        return {"error": "Invalid planet"}

    results = []
    now = datetime.datetime.now(pytz.UTC)

    for i in range(days):
        t = ts.utc(now + datetime.timedelta(days=i))
        x,y,z = get_planet_position(planet, t)
        results.append({"day": i, "x": x, "y": y, "z": z})

    return {"planet": planet, "type": "physics", "data": results}

# ------------------ AI PREDICTIONS ------------------

AI_PLANETS = ["mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]

@app.get("/api/predict/{planet}")
def predict_planet(planet: str, days: int = 30):
    planet = planet.lower()

    if planet not in AI_PLANETS:
        return {"error": "Planet not supported"}

    # Get last 7 days of physics data
    history = []
    now = datetime.datetime.now(pytz.UTC)

    for i in range(7):
        t = ts.utc(now - datetime.timedelta(days=7 - i))
        x, y, z = get_planet_position(planet, t)
        history.append([x, y])

    history = np.array(history)
    history_scaled = scaler.transform(history).reshape(1,7,2)

    preds = []

    for i in range(days):
        p = model.predict(history_scaled, verbose=0)[0]
        preds.append({"day": i, "x": float(p[0]), "y": float(p[1])})
        history_scaled = np.roll(history_scaled, -1, axis=1)
        history_scaled[0,-1] = p

    return {"planet": planet, "type": "AI_prediction", "data": preds}

# ------------------ Health Check ------------------

@app.get("/")
def home():
    return {
        "status": "SmartSky Backend Running",
        "planets": PLANETS,
        "ai_enabled": AI_PLANETS
    }
