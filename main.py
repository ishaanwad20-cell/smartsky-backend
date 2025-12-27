from fastapi import FastAPI
import numpy as np
import joblib
import tensorflow as tf
import datetime
from skyfield.api import load

# ---------- Setup ----------
app = FastAPI()

ts = load.timescale()
eph = load("de421.bsp")

mars = eph["mars"]
earth = eph["earth"]

model = tf.keras.models.load_model("mars_lstm.h5")
scaler = joblib.load("mars_scaler.pkl")

# ---------- Helpers ----------
def get_mars_position(t):
    pos = mars.at(t).observe(earth).position.au
    return pos[0], pos[1], pos[2]

# ---------- Root ----------
@app.get("/")
def root():
    return {"status": "SmartSky backend running 🚀"}

# ---------- Planet endpoint ----------
@app.get("/api/planet/{planet}")
def planet(planet: str):
    if planet.lower() != "mars":
        return {"error": "Only Mars supported right now"}

    t = ts.now()
    x, y, z = get_mars_position(t)

    return {
        "time": t.utc_iso(),
        "x_au": x,
        "y_au": y,
        "z_au": z
    }

# ---------- Lovable Compatibility Layer ----------
@app.get("/api/positions")
def positions(planets: str = "mars", days: int = 30):
    names = planets.split(",")
    now = datetime.datetime.now(datetime.timezone.utc)

    data = []

    for i in range(days):
        t = ts.utc(now + datetime.timedelta(days=i))
        x, y, z = get_mars_position(t)

        data.append({
            "date": (now + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
            "name": "mars",
            "x": x,
            "y": y,
            "z": z
        })

    return {
        "status": 200,
        "count": len(data),
        "planets": ["mars"],
        "data": data,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# ---------- AI Prediction ----------
@app.get("/api/predict/mars")
def predict(days: int = 5):
    history = []

    for i in range(7):
        t = ts.utc(
            datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=7 - i)
        )
        x, y, _ = get_mars_position(t)
        history.append([x, y])

    history = np.array(history)
    scaled = scaler.transform(history).reshape(1, 7, 2)

    preds = model.predict(scaled)[0]

    result = []
    for i in range(days):
        result.append({
            "day": i + 1,
            "x": float(preds[i][0]),
            "y": float(preds[i][1])
        })

    return {
        "planet": "mars",
        "days": days,
        "predictions": result
    }
